#include "dmt/algorithms/fdmt_fft.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <omp.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "dmt/fdmt_fft_tree.hpp"
#include "dmt/fft.hpp"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"
#include "dmt/simd_math.hpp"

#include "fdmt_fft_cpu_kernels.hpp"
#include "packed_source_cpu.hpp"

// FDMT-FFT on the CPU.
//
// In the Fourier domain every tree merge is `out[k] = tail[k] + head[k] *
// W^(delay * k)` with W = exp(-2 pi i / N): each frequency bin k runs the
// same tree independently of the others. execute() exploits this by cutting
// the spectrum into tiles of kTile bins and running the whole tree for one
// tile at a time out of per-thread, cache-resident buffers:
//
//   1. forward R2C of every channel row, scattered tile-major in split
//      layout (spectra [tile][nchans][re, im]: a tile's input is one stream)
//   2. per tile: levels 1..F depth-first per group of 2^F channels (level 0
//      formed on the fly), levels F+1..M tile-wide, root rows streamed to the
//      inverse input                              (out_spec [ndms][bins])
//   3. inverse C2R of every DM row, trimmed into the output
//
// The phasors of a tile, W^(s * (k0 + j)) = W^(s * k0) * W^(s * j), come
// from the exact twiddle table W^m (m = s * k0 mod N) times a small shared
// table W^(s * j), j < kTile: no (shift x bins) table is stored. 1/N is
// folded into the level-0 windows, so the inverse needs no scaling.
//
// Long blocks are transformed in overlap-save segments of a length chosen
// for FFT speed (Geometry, choose_segment()), so the transform length and
// the working set do not grow with nsamps. The segmented result equals the
// single-transform one: both are the linear convolution on the kept samples.
//
// The stepper (reset/advance/view/finalize) needs every level of the whole
// block, so it keeps level-by-level full-width state at the plan's FFT
// length, allocated on first use.

namespace dmt::algorithms {

namespace {

using fft_cpu::cmadd;
using fft_cpu::cmadd_level1;
using fft_cpu::cmul;
using fft_cpu::copy_samples;
using fft_cpu::copy_tile;
using fft_cpu::cscale;
using fft_cpu::Source;
using utils::FFTVector;

// Bins per tile: one AVX-512 register (two AVX2, four NEON) per component
// and one 128-byte cache line (Apple) per tree node. 8, 16 and 32 measure
// the same for integer delays on the M1 (the tree is not the bottleneck);
// wider tiles amortise the per-merge phasor set-up of fractional delays.
constexpr int kTile        = 16;
constexpr SizeType kTileFl = 2 * kTile;
// Largest per-group level slice (tree nodes) kept in L1-sized scratch by the
// depth-first lower levels: 128 * 16 bins * 8 B = 16 KB per buffer.
constexpr SizeType kMaxSlice = 2048 / kTile;
// Channels per forward-FFT task (see forward()): their rows stay in L2.
constexpr SizeType kFwdBatch = 8;

// Slides the overlap window of one channel (row `row` of `src`) forward by
// the block.
void advance_overlap_window(const Source& src,
                            SizeType row,
                            float* __restrict__ hist,
                            SizeType capacity) noexcept {
    const auto nsamps = src.nsamps;
    if (capacity == 0) {
        return;
    }
    if (nsamps >= capacity) {
        copy_samples(src, row, nsamps - capacity, capacity, hist);
    } else {
        std::copy(hist + nsamps, hist + capacity, hist);
        copy_samples(src, row, 0, nsamps, hist + capacity - nsamps);
    }
}

struct Level0Node {
    std::uint32_t chan;
    std::uint32_t shift;
};
struct MergeOp {
    std::uint32_t cur;
    std::uint32_t tail;
    std::uint32_t head;
    std::uint32_t delay;
};
struct CopyOp {
    std::uint32_t cur;
    std::uint32_t tail;
};
struct LevelOps {
    std::vector<MergeOp> sum;
    std::vector<CopyOp> copy;
};

std::uint32_t to_u32(SizeType v) {
    if (v > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("FDMTFFT: tree index exceeds 32 bits");
    }
    return static_cast<std::uint32_t>(v);
}

// One transform length: sizes, exact twiddles and FFTW row plans.
struct Geometry {
    SizeType n_fft{};
    SizeType n_bins{};
    SizeType n_tiles{};
    SizeType bins_stride{};
    FFTVector<ComplexType> twiddle; // W^m, m in [0, N), W = exp(-2 pi i / N)
    std::unique_ptr<utils::FFTWRowPlan> r2c;
    std::unique_ptr<utils::FFTWRowPlan> c2r;

    explicit Geometry(SizeType n)
        : n_fft(n),
          n_bins((n / 2) + 1),
          n_tiles((n_bins + kTile - 1) / kTile),
          bins_stride(n_tiles * kTile) {
        // Twiddles in double, so every phasor is correctly rounded.
        twiddle.resize(n_fft);
        const double step = -2.0 * std::numbers::pi / static_cast<double>(n);
        for (SizeType m = 0; m < n_fft; ++m) {
            const double a = step * static_cast<double>(m);
            twiddle[m]     = ComplexType(static_cast<float>(std::cos(a)),
                                         static_cast<float>(std::sin(a)));
        }
        r2c = std::make_unique<utils::FFTWRowPlan>(utils::FFTKind::kR2C, n);
        c2r = std::make_unique<utils::FFTWRowPlan>(utils::FFTKind::kC2R, n);
    }
};

class FDMTFFTCpuEngine final : public detail::FDMTFFTEngine {
public:
    FDMTFFTCpuEngine(const plans::FDMTPlan& plan,
                     const detail::FDMTFFTEngineConfig& cfg)
        : m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_nbeams(cfg.nbeams),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_use_box_smearing(cfg.use_box_smearing),
          m_mode(cfg.mode),
          m_frac(cfg.fractional_delays),
          m_kill(cfg.kill_mask),
          m_plan(&plan) {
        initialize();
    }

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        check_buffers(waterfall.size(), dmt, "execute");
        run(float_source(waterfall), dmt);
    }

    void execute(std::span<const uint8_t> packed,
                 SizeType nbits,
                 bool time_major,
                 std::span<float> dmt) override {
        if (!time_major) {
            const auto src = packed_source(packed, nbits, "execute");
            check_buffers(m_nbeams * m_nchans * m_nsamps, dmt, "execute");
            run(src, dmt);
            return;
        }
        check_nbits(nbits, "execute_time_major");
        const auto samp_bytes =
            bit_pack_utils::packed_row_bytes(m_nchans, nbits);
        if (packed.size() != m_nbeams * m_nsamps * samp_bytes) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute_time_major: expected {} bytes, got {}",
                m_nbeams * m_nsamps * samp_bytes, packed.size()));
        }
        check_buffers(m_nbeams * m_nchans * m_nsamps, dmt,
                      "execute_time_major");
        m_unpacked.resize(m_nbeams * m_nchans * m_nsamps);
        const auto n = static_cast<std::int64_t>(m_nbeams * m_nsamps);
#pragma omp parallel num_threads(m_nthreads)
        {
            std::vector<float> spectrum(m_nchans);
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < n; ++i) {
                const auto b = static_cast<SizeType>(i) / m_nsamps;
                const auto t = static_cast<SizeType>(i) % m_nsamps;
                bit_pack_utils::unpack_row(
                    packed.data() + (((b * m_nsamps) + t) * samp_bytes), nbits,
                    m_nchans, spectrum.data());
                float* dst = m_unpacked.data() + (b * m_nchans * m_nsamps) + t;
                for (SizeType c = 0; c < m_nchans; ++c) {
                    dst[c * m_nsamps] = spectrum[c];
                }
            }
        }
        run(float_source(m_unpacked), dmt);
    }

    void reset(std::span<const float> waterfall,
               std::span<float> dmt) override {
        check_buffers(waterfall.size(), dmt, "reset");
        start_stepper(float_source(waterfall), dmt);
    }

    void reset(std::span<const uint8_t> packed,
               SizeType nbits,
               std::span<float> dmt) override {
        const auto src = packed_source(packed, nbits, "reset");
        check_buffers(m_nbeams * m_nchans * m_nsamps, dmt, "reset");
        start_stepper(src, dmt);
    }

    void advance(SizeType levels, Stream /*stream*/) override {
        require_stepper();
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            for (SizeType b = 0; b < m_nbeams; ++b) {
                step_merge(m_state_in + (b * m_state_stride),
                           m_state_out + (b * m_state_stride), next_level);
            }
            std::swap(m_state_in, m_state_out);
            m_current_level = next_level;
            m_view_valid    = false;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        require_stepper();
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        require_stepper();
        materialize_view();
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        return {m_view_time.data(), shape.ncoords * view_nsamps()};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        require_stepper();
        materialize_view();
        const auto& plan_c = m_plan->get_container();
        const auto& shape  = plan_c.state_shape[m_current_level];
        if (subband_idx >= shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTFFT: Subband index {} out of range ({} subbands)",
                subband_idx, shape.nchans));
        }
        const auto& grid    = plan_c.grids[m_current_level][subband_idx];
        const auto nsamps_v = view_nsamps();
        const auto offset   = grid.coord_offset * nsamps_v;
        const auto count    = grid.ndt * nsamps_v;
        return FDMTSubbandView{
            .data = std::span<const float>(m_view_time.data() + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps_v,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    [[nodiscard]] SizeType current_level() const noexcept override {
        return m_current_level;
    }
    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan->get_niters() + 1;
    }
    [[nodiscard]] SizeType num_subbands() const override {
        require_stepper();
        return m_plan->get_container().state_shape[m_current_level].nchans;
    }
    [[nodiscard]] bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream) override {
        require_stepper();
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        for (SizeType b = 0; b < m_nbeams; ++b) {
            step_inverse(m_state_in + (b * m_state_stride), m_ndms,
                         m_single_skip, m_nsamps_out,
                         m_dmt_target_ptr + (b * m_ndms * m_nsamps_out));
            if (m_mode == FDMTMode::kValid) {
                update_overlap(m_step_src, b);
            }
        }
        m_is_initialized = false;
        m_view_valid     = false;
    }

    void reset_history() noexcept override {
        std::ranges::fill(m_overlap, 0.0F);
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        return m_overlap.size();
    }
    void save_history(std::span<float> out) const override {
        check_history(out.size(), "save_history");
        std::ranges::copy(m_overlap, out.begin());
    }
    void load_history(std::span<const float> in) override {
        check_history(in.size(), "load_history");
        std::ranges::copy(in, m_overlap.begin());
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    // Per-thread scratch. The tile buffers are split-layout rows of kTileFl
    // floats.
    struct Workspace {
        FFTVector<float> time;         // one real row (longest transform)
        FFTVector<ComplexType> cplx;   // one complex row (widest spectrum)
        FFTVector<ComplexType> batch;  // kFwdBatch complex rows (execute)
        FFTVector<float> q;            // phasors, (max_s + 1) tiles
        FFTVector<float> w0;           // level-0 windows, (dt0_max + 1) tiles
        FFTVector<float> buf_a;        // upper tree state, m_upper_max tiles
        FFTVector<float> buf_b;        // upper tree state, m_upper_max tiles
        FFTVector<float> s0;           // group-local state, m_slice_max tiles
        FFTVector<float> s1;           // group-local state, m_slice_max tiles
        FFTVector<ComplexType> anchor; // fractional: W^(s * k0) per merge op
        FFTVector<float> p;            // fractional: one phasor tile
    };

    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_nbeams;
    int m_nthreads;
    bool m_use_box_smearing;
    FDMTMode m_mode;
    bool m_frac;                 // exact merge delays
    std::vector<uint8_t> m_kill; // per channel, 1 = keep (empty: all)

    const plans::FDMTPlan* m_plan; // owned by the FDMTFFT facade
    fdmt_fft::Geometry m_geom;     // single-transform geometry
    SizeType m_q_max{};            // largest integer shift in the Q table
    // Fractional delays: MergeOp::delay is an op id into these.
    std::vector<double> m_frac_s; // exact shift per merge op
    FFTVector<float> m_frac_r;    // W^(s * j), j < kTile, per op (tiles)
    SizeType m_ndms{};
    SizeType m_max_s{};
    SizeType m_dt0_max{};
    SizeType m_max_coords{};
    SizeType m_overlap_len{};
    SizeType m_nsamps_out{};
    // Single-transform geometry (the plan's FFT length): the window is the
    // mode's window, output sample o is transform sample m_single_skip + o.
    SizeType m_single_skip{};

    // Segments of execute(). The virtual input of a beam and channel is
    // v = [zeros(m_seg_zeros) | overlap (valid) | block | zeros...]; segment
    // j transforms v[j * m_seg_hop, j * m_seg_hop + N) and yields output
    // samples [j * m_seg_hop, (j + 1) * m_seg_hop) from transform sample
    // m_seg_skip on (m_seg_skip - m_seg_zeros == m_single_skip). One segment
    // is the single-transform case.
    SizeType m_nseg{1};
    SizeType m_seg_hop{};
    SizeType m_seg_skip{};
    SizeType m_seg_zeros{};
    std::unique_ptr<Geometry> m_geo; // execute()'s transform length

    std::vector<Level0Node> m_level0;
    std::vector<LevelOps> m_levels; // index 1..niters (global node indices)
    // Depth-first lower tree: levels 1..m_fuse run one level-m_fuse subband
    // (a group of 2^m_fuse channels) at a time. m_groups[g][l - 1] holds the
    // ops of level l for group g with group-local node indices (levels
    // < m_fuse) or global ones (level m_fuse, and level-0 nodes at l = 1).
    SizeType m_fuse{1};
    SizeType m_slice_max{1};
    SizeType m_upper_max{1};
    std::vector<std::vector<LevelOps>> m_groups;
    FFTVector<float> m_r_table; // W^(s * j), (max_s + 1) tiles
    // Tile-major split layout [n_tiles][nchans][kTileFl]: one tile's input
    // is a single stream.
    FFTVector<float> m_spectra;
    // Interleaved DM rows [ndms][bins_stride], the inverse FFT's input.
    FFTVector<ComplexType> m_out_spec;
    std::vector<float> m_overlap;
    std::vector<float> m_unpacked; // time-major input, channel-major
    std::vector<Workspace> m_ws;

    // Stepper state (lazy), at the plan's FFT length.
    std::unique_ptr<Geometry> m_step_geo;
    FFTVector<ComplexType> m_step_spectra; // [nchans][bins_stride]
    SizeType m_state_stride{};
    std::vector<ComplexType> m_state_a;
    std::vector<ComplexType> m_state_b;
    mutable std::vector<float> m_view_time;
    Source m_step_src{}; // the stepper's block (read again by finalize())
    float* m_dmt_target_ptr{nullptr};
    ComplexType* m_state_in{nullptr};
    ComplexType* m_state_out{nullptr};
    SizeType m_current_level{0};
    bool m_is_initialized{false};
    mutable bool m_view_valid{false};

    void initialize() {
        const auto& plan_c = m_plan->get_container();
        m_ndms             = m_plan->get_dmt_ndms();
        m_max_s            = m_plan->get_max_shift();
        m_dt0_max          = plan_c.state_shape[0].dt_max;
        m_geom             = fdmt_fft::make_geometry(*m_plan, m_mode, m_frac,
                                                     m_use_box_smearing);
        m_overlap_len      = m_geom.overlap;
        m_nsamps_out       = m_plan->get_dmt_nsamps();
        m_single_skip      = m_geom.skip;
        if (m_mode == FDMTMode::kValid && m_nsamps_out != m_nsamps) {
            throw std::logic_error("FDMTFFT: valid mode expects nsamps output");
        }
        if (m_single_skip + m_nsamps_out > m_geom.n_fft) {
            throw std::logic_error("FDMTFFT: output exceeds the FFT length");
        }

        build_tree_ops();
        // Merges read the Q table only with integer delays; the level-0
        // windows always do.
        m_q_max = m_frac ? m_dt0_max : m_max_s;
        choose_segment();
        logging::debug("FDMTFFT cpu: transform length {} x {} segment(s) "
                       "(single {}), tree group level {}, fractional {}",
                       m_seg_len, m_nseg, m_geom.n_fft, m_fuse, m_frac);
        m_geo = std::make_unique<Geometry>(m_seg_len);

        const auto& g = *m_geo;
        m_r_table.assign((m_q_max + 1) * kTileFl, 0.0F);
        for (SizeType s = 0; s <= m_q_max; ++s) {
            float* r = &m_r_table[s * kTileFl];
            for (SizeType j = 0; j < static_cast<SizeType>(kTile); ++j) {
                const auto w = g.twiddle[(s * j) % g.n_fft];
                r[j]         = w.real();
                r[kTile + j] = w.imag();
            }
        }
        if (m_frac) {
            // Per merge: W^(s * j) for j < kTile, exact in double.
            m_frac_r.assign(m_frac_s.size() * kTileFl, 0.0F);
            for (SizeType op = 0; op < m_frac_s.size(); ++op) {
                for (SizeType j = 0; j < static_cast<SizeType>(kTile); ++j) {
                    const double a =
                        -2.0 * std::numbers::pi *
                        simd::wrap_turns(m_frac_s[op] * static_cast<double>(j) /
                                         static_cast<double>(g.n_fft));
                    m_frac_r[(op * kTileFl) + j] =
                        static_cast<float>(std::cos(a));
                    m_frac_r[(op * kTileFl) + kTile + j] =
                        static_cast<float>(std::sin(a));
                }
            }
        }
        m_spectra.assign(g.n_tiles * m_nchans * kTileFl, 0.0F);
        m_out_spec.assign(m_ndms * g.bins_stride, ComplexType{});
        if (m_mode == FDMTMode::kValid) {
            m_overlap.assign(m_nbeams * m_nchans * m_overlap_len, 0.0F);
        }

        // Pad bins [n_bins, bins_stride) of the per-thread rows are never
        // written by FFTW and stay zero; tiles read them harmlessly.
        m_ws.resize(static_cast<SizeType>(m_nthreads));
        for (auto& ws : m_ws) {
            ws.time.assign(utils::fft_row_stride<float>(g.n_fft), 0.0F);
            ws.cplx.assign(g.bins_stride, ComplexType{});
            ws.batch.assign(kFwdBatch * g.bins_stride, ComplexType{});
            ws.q.assign((m_q_max + 1) * kTileFl, 0.0F);
            ws.anchor.assign(m_frac ? m_frac_s.size() : 0, ComplexType{});
            ws.p.assign(kTileFl, 0.0F);
            ws.w0.assign((m_dt0_max + 1) * kTileFl, 0.0F);
            ws.buf_a.assign(m_upper_max * kTileFl, 0.0F);
            ws.buf_b.assign(m_upper_max * kTileFl, 0.0F);
            ws.s0.assign(m_slice_max * kTileFl, 0.0F);
            ws.s1.assign(m_slice_max * kTileFl, 0.0F);
        }
    }

    SizeType m_seg_len{};

    // Picks execute()'s transform length (fdmt_fft::choose_segments(), the
    // rule every engine shares).
    void choose_segment() {
        SizeType nodes = m_level0.size();
        for (const auto& l : m_levels) {
            nodes += l.sum.size() + l.copy.size();
        }
        const auto seg = fdmt_fft::choose_segments(m_geom, m_mode, m_nsamps_out,
                                                   nodes, m_nchans, m_ndms,
                                                   detail::fft_segment_cap());
        m_seg_len      = seg.n_fft;
        m_nseg         = seg.nseg;
        m_seg_hop      = seg.hop;
        m_seg_skip     = seg.skip;
        m_seg_zeros    = seg.zeros;
    }

    void build_tree_ops() {
        const auto& plan_c = m_plan->get_container();
        const auto niters  = m_plan->get_niters();
        m_level0.assign(plan_c.state_shape[0].ncoords, Level0Node{});
        for (SizeType c = 0; c < plan_c.grids[0].size(); ++c) {
            const auto& g = plan_c.grids[0][c];
            for (SizeType i = 0; i < g.ndt; ++i) {
                const auto s = static_cast<SizeType>(std::abs(g.dt_grid[i]));
                m_level0[g.coord_offset + i] = {to_u32(c), to_u32(s)};
            }
        }
        m_levels.assign(niters + 1, LevelOps{});
        m_frac_s.clear();
        const auto frac =
            m_frac ? fdmt_fft::fractional_delays(*m_plan, m_use_box_smearing)
                   : fdmt_fft::FractionalDelays{};
        for (SizeType l = 1; l <= niters; ++l) {
            auto& ops        = m_levels[l];
            const auto& sums = plan_c.coordinates_sum[l];
            for (SizeType i = 0; i < sums.size(); ++i) {
                const auto& c = sums[i];
                // With fractional delays `delay` is the op id into m_frac_s.
                const auto delay = m_frac ? m_frac_s.size() : c.delay;
                if (m_frac) {
                    m_frac_s.push_back(frac.shift[l][i]);
                }
                ops.sum.push_back({
                    .cur   = to_u32(index_of(c.buf_offset, c.nsamps)),
                    .tail  = to_u32(index_of(c.tail_buf_offset, c.tail_nsamps)),
                    .head  = to_u32(index_of(c.head_buf_offset, c.head_nsamps)),
                    .delay = to_u32(delay),
                });
            }
            for (const auto& c : plan_c.coordinates_copy[l]) {
                ops.copy.push_back({
                    .cur  = to_u32(index_of(c.buf_offset, c.nsamps)),
                    .tail = to_u32(index_of(c.tail_buf_offset, c.tail_nsamps)),
                });
            }
        }
        if (plan_c.state_shape[niters].ncoords != m_ndms) {
            throw std::logic_error("FDMTFFT: root coordinates != DM trials");
        }
        m_max_coords = 0;
        for (SizeType l = 0; l <= niters; ++l) {
            m_max_coords =
                std::max(m_max_coords, plan_c.state_shape[l].ncoords);
        }
        build_groups();
    }

    // First node index of level-l subband `sub` (or the level's node count
    // past the last subband).
    [[nodiscard]] SizeType node_base(SizeType l, SizeType sub) const {
        const auto& plan_c = m_plan->get_container();
        const auto& grids  = plan_c.grids[l];
        return sub < grids.size() ? grids[sub].coord_offset
                                  : plan_c.state_shape[l].ncoords;
    }

    // Largest group slice of levels 1..f-1 when grouping at level f.
    [[nodiscard]] SizeType slice_for(SizeType f) const {
        const auto& shapes = m_plan->get_container().state_shape;
        SizeType worst     = 0;
        for (SizeType g = 0; g < shapes[f].nchans; ++g) {
            for (SizeType l = 1; l < f; ++l) {
                const auto lo = node_base(l, g << (f - l));
                const auto hi = node_base(l, (g + 1) << (f - l));
                worst         = std::max(worst, hi - lo);
            }
        }
        return worst;
    }

    void build_groups() {
        const auto& plan_c = m_plan->get_container();
        const auto niters  = m_plan->get_niters();
        // Group level: the one that minimises the tile-wide (upper) buffers
        // while the group slices stay L1-sized.
        SizeType best_f     = 1;
        SizeType best_upper = std::numeric_limits<SizeType>::max();
        for (SizeType f = 1; f <= niters; ++f) {
            if (f > 1 && slice_for(f) > kMaxSlice) {
                break;
            }
            SizeType upper = 0;
            for (SizeType l = f; l <= niters; ++l) {
                upper = std::max(upper, plan_c.state_shape[l].ncoords);
            }
            if (upper < best_upper) {
                best_upper = upper;
                best_f     = f;
            }
        }
        m_fuse      = best_f;
        m_upper_max = std::max<SizeType>(best_upper, 1);
        m_slice_max = std::max<SizeType>(slice_for(m_fuse), 1);

        const auto ngroups = plan_c.state_shape[m_fuse].nchans;
        m_groups.assign(ngroups, std::vector<LevelOps>(m_fuse));
        for (SizeType l = 1; l <= m_fuse; ++l) {
            const auto shift  = m_fuse - l;
            const auto& grids = plan_c.grids[l];
            // Owning group of each level-l node.
            std::vector<std::uint32_t> group_of(plan_c.state_shape[l].ncoords);
            for (SizeType sub = 0; sub < grids.size(); ++sub) {
                for (SizeType i = 0; i < grids[sub].ndt; ++i) {
                    group_of[grids[sub].coord_offset + i] =
                        to_u32(sub >> shift);
                }
            }
            const auto local = [&](SizeType level, SizeType node,
                                   SizeType g) -> std::uint32_t {
                if (level == 0 || level == m_fuse) {
                    return to_u32(node); // level-0 node / global level-F node
                }
                const auto lo = node_base(level, g << (m_fuse - level));
                if (node < lo || node - lo >= m_slice_max) {
                    throw std::logic_error("FDMTFFT: node outside its group");
                }
                return to_u32(node - lo);
            };
            for (const auto& op : m_levels[l].sum) {
                const auto g = group_of[op.cur];
                m_groups[g][l - 1].sum.push_back({
                    .cur   = local(l, op.cur, g),
                    .tail  = local(l - 1, op.tail, g),
                    .head  = local(l - 1, op.head, g),
                    .delay = op.delay,
                });
            }
            for (const auto& op : m_levels[l].copy) {
                const auto g = group_of[op.cur];
                m_groups[g][l - 1].copy.push_back({
                    .cur  = local(l, op.cur, g),
                    .tail = local(l - 1, op.tail, g),
                });
            }
        }
    }

    static SizeType index_of(SizeType buf_offset, SizeType nsamps) {
        assert(nsamps != 0 && "FDMTFFT: nsamps must be non-zero");
        return buf_offset / nsamps;
    }

    void check_buffers(SizeType in_size,
                       std::span<float> dmt,
                       std::string_view what) const {
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (in_size != total_in) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: expected waterfall size {}, got {}",
                            what, total_in, in_size));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: dmt buffer size {} must be >= {}",
                            what, dmt.size(), total_out));
        }
    }

    static void check_nbits(SizeType nbits, std::string_view what) {
        if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 &&
            nbits != 16) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::{}: nbits must be 1, 2, 4, 8 or 16, got {}", what,
                nbits));
        }
    }

    void check_history(SizeType size, std::string_view what) const {
        if (size != m_overlap.size()) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: expected {} floats, got {}", what,
                            m_overlap.size(), size));
        }
    }

    [[nodiscard]] Source float_source(std::span<const float> wf) const {
        return {.f = wf.data(), .nsamps = m_nsamps};
    }

    [[nodiscard]] Source packed_source(std::span<const uint8_t> packed,
                                       SizeType nbits,
                                       std::string_view what) const {
        check_nbits(nbits, what);
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_nsamps, nbits);
        if (packed.size() != m_nbeams * m_nchans * row_bytes) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::{}: expected {} packed bytes, got {}", what,
                m_nbeams * m_nchans * row_bytes, packed.size()));
        }
        return {.p         = packed.data(),
                .row_bytes = row_bytes,
                .nbits     = nbits,
                .nsamps    = m_nsamps};
    }

    [[nodiscard]] bool killed(SizeType c) const noexcept {
        return !m_kill.empty() && m_kill[c] == 0;
    }

    void run(const Source& src, std::span<float> dmt) {
        for (SizeType b = 0; b < m_nbeams; ++b) {
            float* out = dmt.data() + (b * m_ndms * m_nsamps_out);
            for (SizeType seg = 0; seg < m_nseg; ++seg) {
                forward(src, b, seg);
                run_tiles();
                inverse(out, seg);
            }
            if (m_mode == FDMTMode::kValid) {
                update_overlap(src, b);
            }
        }
    }

    void start_stepper(const Source& src, std::span<float> dmt) {
        ensure_stepper_state();
        m_step_src       = src;
        m_dmt_target_ptr = dmt.data();
        m_current_level  = 0;
        m_state_in       = m_state_a.data();
        m_state_out      = m_state_b.data();
        m_view_valid     = false;

        // Every beam is transformed (same as the CPU FDMT). Views still expose
        // beam 0 only.
        for (SizeType b = 0; b < m_nbeams; ++b) {
            step_forward(src, b);
            step_level0(m_state_in + (b * m_state_stride));
        }
        m_is_initialized = true;
    }

    [[nodiscard]] Workspace& ws() {
        return m_ws[static_cast<SizeType>(omp_get_thread_num())];
    }
    [[nodiscard]] const Workspace& ws() const {
        return m_ws[static_cast<SizeType>(omp_get_thread_num())];
    }

    [[nodiscard]] const float* overlap_row(SizeType beam, SizeType c) const {
        return m_overlap.data() + (((beam * m_nchans) + c) * m_overlap_len);
    }

    // Transform input of segment `seg` of one channel: the samples
    // [seg * hop, seg * hop + N) of v = [zeros(z) | overlap | block | 0...].
    void fill_segment(float* row,
                      const Source& src,
                      SizeType src_row,
                      const float* ov,
                      SizeType seg) const {
        const auto n    = m_geo->n_fft;
        const auto lov  = (m_mode == FDMTMode::kValid) ? m_overlap_len : 0;
        const auto z    = m_seg_zeros;
        const auto p0   = seg * m_seg_hop;
        const auto part = [&](SizeType begin, SizeType len, const float* src) {
            // Copy v[begin, begin + len) = src[0, len) where it meets the
            // segment [p0, p0 + n).
            const auto lo = std::max(begin, p0);
            const auto hi = std::min(begin + len, p0 + n);
            if (lo < hi) {
                std::copy_n(src + (lo - begin), hi - lo, row + (lo - p0));
            }
        };
        std::fill_n(row, n, 0.0F);
        if (killed(src_row % m_nchans)) {
            return;
        }
        if (lov > 0) {
            part(z, lov, ov);
        }
        // Block samples v[z + lov, z + lov + nsamps).
        const auto begin = z + lov;
        const auto lo    = std::max(begin, p0);
        const auto hi    = std::min(begin + m_nsamps, p0 + n);
        if (lo < hi) {
            copy_samples(src, src_row, lo - begin, hi - lo, row + (lo - p0));
        }
    }

    // Channel rows of one beam and segment -> m_spectra. kFwdBatch
    // consecutive channels per task, so each tile receives one contiguous
    // run of kFwdBatch * kTile bins.
    void forward(const Source& src, SizeType beam, SizeType seg) {
        const auto& g = *m_geo;
        const auto nblocks =
            static_cast<std::int64_t>((m_nchans + kFwdBatch - 1) / kFwdBatch);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t bi = 0; bi < nblocks; ++bi) {
            auto& w       = ws();
            const auto c0 = static_cast<SizeType>(bi) * kFwdBatch;
            const auto nc = std::min(kFwdBatch, m_nchans - c0);
            for (SizeType i = 0; i < nc; ++i) {
                const auto c = c0 + i;
                fill_segment(w.time.data(), src, (beam * m_nchans) + c,
                             m_overlap_len > 0 ? overlap_row(beam, c) : nullptr,
                             seg);
                g.r2c->r2c(w.time.data(), w.batch.data() + (i * g.bins_stride));
            }
            float* dst = m_spectra.data() + (c0 * kTileFl);
            for (SizeType t = 0; t < g.n_tiles; ++t) {
                float* d = dst + (t * m_nchans * kTileFl);
                for (SizeType i = 0; i < nc; ++i) {
                    fft_cpu::stream_split<kTile>(
                        d + (i * kTileFl),
                        w.batch.data() + (i * g.bins_stride) + (t * kTile));
                }
            }
        }
        fft_cpu::stream_fence();
    }

    void update_overlap(const Source& src, SizeType beam) {
        if (m_overlap_len == 0) {
            return;
        }
        const auto nchans = static_cast<std::int64_t>(m_nchans);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t ci = 0; ci < nchans; ++ci) {
            const auto c = static_cast<SizeType>(ci);
            float* ov    = &m_overlap[((beam * m_nchans) + c) * m_overlap_len];
            advance_overlap_window(src, (beam * m_nchans) + c, ov,
                                   m_overlap_len);
        }
    }

    // Phasor tiles W^(s * (k0 + j)) for s <= max_s and the level-0 windows
    // (box sums of them when smearing), scaled by 1/N.
    void build_tile_tables(Workspace& w, SizeType k0) const {
        const auto& g   = *m_geo;
        float* q        = w.q.data();
        SizeType m      = 0; // s * k0 mod N
        const auto step = k0 % g.n_fft;
        for (SizeType s = 0; s <= m_q_max; ++s) {
            cscale<kTile>(q + (s * kTileFl), g.twiddle[m],
                          &m_r_table[s * kTileFl]);
            m += step;
            if (m >= g.n_fft) {
                m -= g.n_fft;
            }
        }
        const float norm = 1.0F / static_cast<float>(g.n_fft);
        float acc[kTileFl]{};
        for (SizeType s = 0; s <= m_dt0_max; ++s) {
            const float* qs = q + (s * kTileFl);
            float* out      = w.w0.data() + (s * kTileFl);
            for (SizeType j = 0; j < kTileFl; ++j) {
                acc[j] += qs[j];
                out[j] = (m_use_box_smearing ? acc[j] : qs[j]) * norm;
            }
        }
    }

    void run_tiles() {
        const auto n_tiles = static_cast<std::int64_t>(m_geo->n_tiles);
#pragma omp parallel for schedule(dynamic, 4) num_threads(m_nthreads)
        for (std::int64_t t = 0; t < n_tiles; ++t) {
            run_tile(ws(), static_cast<SizeType>(t) * kTile);
        }
        fft_cpu::stream_fence();
    }

    void run_tile(Workspace& w, SizeType k0) {
        if (m_frac) {
            run_tile_impl<true>(w, k0);
        } else {
            run_tile_impl<false>(w, k0);
        }
    }

    // Fractional delays: W^(s * k0) of every merge op for this tile, the
    // turn count reduced exactly in double, one vectorised sincos.
    void build_anchors(Workspace& w, SizeType k0) const {
        const double kn =
            static_cast<double>(k0) / static_cast<double>(m_geo->n_fft);
        const auto nops  = m_frac_s.size();
        ComplexType* out = w.anchor.data();
        auto* f          = reinterpret_cast<float*>(out);
#pragma omp simd
        for (SizeType op = 0; op < nops; ++op) {
            float sn = 0.0F;
            float cs = 0.0F;
            simd::sincos_turns(
                static_cast<float>(simd::wrap_turns(-m_frac_s[op] * kn)), sn,
                cs);
            f[2 * op]       = cs;
            f[(2 * op) + 1] = sn;
        }
    }

    // Phasor tile of merge `op`: exact per-op table (fractional) or the
    // tile's integer-shift table.
    template <bool FRAC>
    [[nodiscard]] const float*
    phasor(Workspace& w, const float* q, std::uint32_t delay) const noexcept {
        if constexpr (FRAC) {
            cscale<kTile>(w.p.data(), w.anchor[delay],
                          &m_frac_r[delay * kTileFl]);
            return w.p.data();
        } else {
            return q + (delay * kTileFl);
        }
    }

    template <bool FRAC> void run_tile_impl(Workspace& w, SizeType k0) {
        const auto& g = *m_geo;
        build_tile_tables(w, k0);
        if constexpr (FRAC) {
            build_anchors(w, k0);
        }
        const float* q  = w.q.data();
        const float* w0 = w.w0.data();
        const float* spec =
            m_spectra.data() + ((k0 / kTile) * m_nchans * kTileFl);
        float* upper_in  = w.buf_a.data();
        float* upper_out = w.buf_b.data();

        // Lower levels, depth-first: one group of 2^m_fuse channels at a
        // time through L1-sized scratch; level m_fuse lands in upper_in.
        for (const auto& group : m_groups) {
            float* prev = w.s0.data();
            float* cur  = w.s1.data();
            for (SizeType l = 1; l <= m_fuse; ++l) {
                const auto& ops = group[l - 1];
                float* dst      = (l == m_fuse) ? upper_in : cur;
                if (l == 1) {
                    run_level1<FRAC>(w, ops, spec, w0, q, dst);
                } else {
                    run_merges<FRAC>(w, ops, prev, dst, q);
                }
                std::swap(prev, cur);
            }
        }
        for (SizeType l = m_fuse + 1; l < m_levels.size(); ++l) {
            run_merges<FRAC>(w, m_levels[l], upper_in, upper_out, q);
            std::swap(upper_in, upper_out);
        }
        ComplexType* dst = m_out_spec.data() + k0;
        for (SizeType dm = 0; dm < m_ndms; ++dm) {
            fft_cpu::stream_interleaved<kTile>(dst + (dm * g.bins_stride),
                                               upper_in + (dm * kTileFl));
        }
    }

    // Level 1 straight from the spectra: level-0 nodes are formed on the fly
    // (spectrum tile times its window).
    template <bool FRAC>
    void run_level1(Workspace& w,
                    const LevelOps& ops,
                    const float* spec,
                    const float* w0,
                    const float* q,
                    float* out) const {
        for (const auto& op : ops.sum) {
            const auto& t = m_level0[op.tail];
            const auto& h = m_level0[op.head];
            cmadd_level1<kTile>(
                out + (op.cur * kTileFl), spec + (t.chan * kTileFl),
                w0 + (t.shift * kTileFl), spec + (h.chan * kTileFl),
                w0 + (h.shift * kTileFl), phasor<FRAC>(w, q, op.delay));
        }
        for (const auto& op : ops.copy) {
            const auto& t = m_level0[op.tail];
            cmul<kTile>(out + (op.cur * kTileFl), spec + (t.chan * kTileFl),
                        w0 + (t.shift * kTileFl));
        }
    }

    template <bool FRAC>
    void run_merges(Workspace& w,
                    const LevelOps& ops,
                    const float* in,
                    float* out,
                    const float* q) const noexcept {
        for (const auto& op : ops.sum) {
            cmadd<kTile>(out + (op.cur * kTileFl), in + (op.tail * kTileFl),
                         in + (op.head * kTileFl),
                         phasor<FRAC>(w, q, op.delay));
        }
        for (const auto& op : ops.copy) {
            copy_tile<kTile>(out + (op.cur * kTileFl),
                             in + (op.tail * kTileFl));
        }
    }

    // DM rows of m_out_spec -> time domain; segment `seg` fills output
    // samples [seg * hop, min((seg + 1) * hop, nsamps_out)).
    void inverse(float* dmt_ptr, SizeType seg) {
        const auto& g   = *m_geo;
        const auto o0   = seg * m_seg_hop;
        const auto cnt  = std::min(m_seg_hop, m_nsamps_out - o0);
        const auto ndms = static_cast<std::int64_t>(m_ndms);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t di = 0; di < ndms; ++di) {
            const auto dm = static_cast<SizeType>(di);
            float* row    = ws().time.data();
            g.c2r->c2r(&m_out_spec[dm * g.bins_stride], row);
            std::copy_n(row + m_seg_skip, cnt,
                        dmt_ptr + (dm * m_nsamps_out) + o0);
        }
    }

    // ---- stepper (full width, level by level, plan FFT length) ----

    void ensure_stepper_state() {
        if (m_step_geo) {
            return;
        }
        m_step_geo    = std::make_unique<Geometry>(m_geom.n_fft);
        const auto& g = *m_step_geo;
        m_step_spectra.assign(m_nchans * g.bins_stride, ComplexType{});
        m_state_stride = m_max_coords * g.n_bins;
        m_state_a.assign(m_nbeams * m_state_stride, ComplexType{});
        m_state_b.assign(m_nbeams * m_state_stride, ComplexType{});
        for (auto& ws : m_ws) {
            const auto nt = utils::fft_row_stride<float>(g.n_fft);
            if (ws.time.size() < nt) {
                ws.time.assign(nt, 0.0F);
            }
            if (ws.cplx.size() < g.bins_stride) {
                ws.cplx.assign(g.bins_stride, ComplexType{});
            }
        }
    }

    // Channel rows of one beam -> m_step_spectra (single transform window).
    void step_forward(const Source& src, SizeType beam) {
        const auto& g     = *m_step_geo;
        const auto nchans = static_cast<std::int64_t>(m_nchans);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t ci = 0; ci < nchans; ++ci) {
            const auto c = static_cast<SizeType>(ci);
            float* row   = ws().time.data();
            SizeType pos = 0;
            std::fill_n(row, g.n_fft, 0.0F);
            if (!killed(c)) {
                if (m_mode == FDMTMode::kValid && m_overlap_len > 0) {
                    std::copy_n(overlap_row(beam, c), m_overlap_len, row);
                    pos = m_overlap_len;
                }
                copy_samples(src, (beam * m_nchans) + c, 0, m_nsamps,
                             row + pos);
            }
            g.r2c->r2c(row, &m_step_spectra[c * g.bins_stride]);
        }
    }

    // Phasor row of merge op `delay` (integer shift, or op id with
    // fractional delays), k < n_bins.
    void merge_phasor_row(std::uint32_t delay, ComplexType* row) const {
        if (!m_frac) {
            phasor_row(delay, row);
            return;
        }
        // exp(-2 pi i s k / N) by a double-precision recurrence, re-anchored
        // exactly every 256 bins.
        const auto& g   = *m_step_geo;
        const double s  = m_frac_s[delay];
        const double nd = static_cast<double>(g.n_fft);
        const auto at   = [&](SizeType k) {
            const double a = -2.0 * std::numbers::pi *
                             simd::wrap_turns(s * static_cast<double>(k) / nd);
            return std::complex<double>(std::cos(a), std::sin(a));
        };
        const auto step = at(1);
        std::complex<double> p;
        for (SizeType k = 0; k < g.n_bins; ++k) {
            p      = (k % 256 == 0) ? at(k) : p * step;
            row[k] = ComplexType(static_cast<float>(p.real()),
                                 static_cast<float>(p.imag()));
        }
    }

    // Phasor row W^(s * k), k < n_bins (exact twiddle lookups).
    void phasor_row(SizeType s, ComplexType* row) const noexcept {
        const auto& g = *m_step_geo;
        SizeType m    = 0;
        for (SizeType k = 0; k < g.n_bins; ++k) {
            row[k] = g.twiddle[m];
            m += s;
            if (m >= g.n_fft) {
                m -= g.n_fft;
            }
        }
    }

    void step_level0(ComplexType* state) {
        const auto& g    = *m_step_geo;
        const auto n     = static_cast<std::int64_t>(m_level0.size());
        const float norm = 1.0F / static_cast<float>(g.n_fft);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t i = 0; i < n; ++i) {
            const auto& node = m_level0[static_cast<SizeType>(i)];
            ComplexType* win = ws().cplx.data();
            // Window: W^(s k) (no smearing) or sum_{u <= s} W^(u k).
            const auto s = static_cast<SizeType>(node.shift);
            if (m_use_box_smearing) {
                std::fill_n(win, g.n_bins, ComplexType{});
                for (SizeType u = 0; u <= s; ++u) {
                    SizeType m = 0;
                    for (SizeType k = 0; k < g.n_bins; ++k) {
                        win[k] += g.twiddle[m];
                        m += u;
                        if (m >= g.n_fft) {
                            m -= g.n_fft;
                        }
                    }
                }
            } else {
                phasor_row(s, win);
            }
            const ComplexType* spec =
                &m_step_spectra[node.chan * g.bins_stride];
            ComplexType* out = state + (static_cast<SizeType>(i) * g.n_bins);
            for (SizeType k = 0; k < g.n_bins; ++k) {
                out[k] = spec[k] * win[k] * norm;
            }
        }
    }

    void step_merge(const ComplexType* in, ComplexType* out, SizeType level) {
        const auto nb     = m_step_geo->n_bins;
        const auto& ops   = m_levels[level];
        const auto n_sum  = static_cast<std::int64_t>(ops.sum.size());
        const auto n_copy = static_cast<std::int64_t>(ops.copy.size());
#pragma omp parallel num_threads(m_nthreads)
        {
            ComplexType* p = ws().cplx.data();
#pragma omp for schedule(static) nowait
            for (std::int64_t i = 0; i < n_sum; ++i) {
                const auto& op = ops.sum[static_cast<SizeType>(i)];
                merge_phasor_row(op.delay, p);
                const ComplexType* tail = in + (op.tail * nb);
                const ComplexType* head = in + (op.head * nb);
                ComplexType* dst        = out + (op.cur * nb);
                for (SizeType k = 0; k < nb; ++k) {
                    dst[k] = tail[k] + (head[k] * p[k]);
                }
            }
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < n_copy; ++i) {
                const auto& op = ops.copy[static_cast<SizeType>(i)];
                std::copy_n(in + (op.tail * nb), nb, out + (op.cur * nb));
            }
        }
    }

    // Inverse of `nrows` state rows (left intact), keeping [skip, skip +
    // count) of each into dst rows of count samples (zero past the
    // transform).
    void step_inverse(const ComplexType* rows,
                      SizeType nrows,
                      SizeType skip,
                      SizeType count,
                      float* dst) const {
        const auto& g = *m_step_geo;
        const auto n  = static_cast<std::int64_t>(nrows);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t ri = 0; ri < n; ++ri) {
            const auto r = static_cast<SizeType>(ri);
            auto& w      = const_cast<Workspace&>(ws());
            std::copy_n(rows + (r * g.n_bins), g.n_bins, w.cplx.data());
            g.c2r->c2r(w.cplx.data(), w.time.data());
            const auto ncopy =
                (g.n_fft > skip) ? std::min(count, g.n_fft - skip) : 0;
            std::copy_n(w.time.data() + skip, ncopy, dst + (r * count));
            std::fill(dst + (r * count) + ncopy, dst + ((r + 1) * count), 0.0F);
        }
    }

    [[nodiscard]] SizeType view_nsamps() const {
        if (m_mode == FDMTMode::kFull) {
            return m_plan->get_container().state_shape[m_current_level].nsamps;
        }
        return m_nsamps;
    }

    void materialize_view() const {
        if (m_view_valid) {
            return;
        }
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        const auto nsamps_v = view_nsamps();
        m_view_time.assign(shape.ncoords * nsamps_v, 0.0F);
        step_inverse(m_state_in, shape.ncoords, m_single_skip, nsamps_v,
                     m_view_time.data());
        m_view_valid = true;
    }

    void require_stepper() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTFFT: Stepper is not initialized. Call reset() first.");
        }
    }
};

} // namespace

std::unique_ptr<detail::FDMTFFTEngine>
detail::make_fdmt_fft_cpu(const plans::FDMTPlan& plan,
                          const detail::FDMTFFTEngineConfig& cfg) {
    return std::make_unique<FDMTFFTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
