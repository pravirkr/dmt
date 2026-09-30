#include "dmt/algorithms/ddmt_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <omp.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_fft_common.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft.hpp"
#include "dmt/logging.hpp"
#include "dmt/nufft.hpp"
#include "dmt/simd_math.hpp"

#include "fdmt_fft_cpu_kernels.hpp"
#include "packed_source_cpu.hpp"

// DDMT-FFT on the CPU.
//
// Every call dedisperses the retained history followed by the new block in
// one transform, or overlap-save segments when that would exceed the memory
// cap (ddmt_fft::plan_segments):
//
//   1. forward R2C of every active channel's segment, scattered tile-major
//      in split layout (spectra [tile][active chans][re, im])
//   2. channel sum per bin: brute force (exact phase rotation of each
//      channel for each trial) or NUFFT (uniform DM grids), into interleaved
//      DM rows scaled by 1/N                          (out_spec [ndm][bins])
//   3. inverse C2R of every DM row, keeping the segment's valid samples
//
// Brute force, per work item (tile of kBins bins x block of kDmBlock
// trials): the accumulators of the block stay in L2 while the channels are
// taken kChanGroup at a time from L1; each (channel, trial) phasor
// exp(2 pi i tau (k0 + j) / N) is set up exactly in double on kLanes bins
// (one vectorised sincos per block of trials) and rotated along the tile by
// its exact kLanes-bin step, two complex multiplies per complex term.

namespace dmt::algorithms {

namespace {

using fft_cpu::copy_samples;
using fft_cpu::Source;
using utils::FFTVector;

constexpr int kBins           = 512; // bins per tile
constexpr SizeType kBinsFl    = 2 * kBins;
constexpr int kLanes          = 16; // bins per rotation vector
constexpr SizeType kDmBlock   = 64; // trials per work item
constexpr SizeType kChanGroup = 4;  // channels per accumulator pass
constexpr int kNuTile         = 16; // bins per NUFFT group (and tile)
constexpr SizeType kNuBins    = kNuTile;
constexpr SizeType kFwdBatch  = 4; // channel rows per forward task
// Cap on the channel spectra of one segment (bytes).
constexpr SizeType kMaxSpectraBytes = SizeType{1} << 29;

static_assert(kBins % kLanes == 0, "kBins must be a multiple of kLanes");
static_assert(kBins % kNuBins == 0, "bins_stride must hold whole groups");

struct Geometry {
    SizeType n_fft{};
    SizeType n_bins{};
    SizeType n_tiles{};
    SizeType bins_stride{};
    std::unique_ptr<utils::FFTWRowPlan> r2c;
    std::unique_ptr<utils::FFTWRowPlan> c2r;

    explicit Geometry(SizeType n)
        : n_fft(n),
          n_bins((n / 2) + 1),
          n_tiles((n_bins + kBins - 1) / kBins),
          bins_stride(n_tiles * kBins),
          r2c(std::make_unique<utils::FFTWRowPlan>(utils::FFTKind::kR2C, n)),
          c2r(std::make_unique<utils::FFTWRowPlan>(utils::FFTKind::kC2R, n)) {}
};


class DDMTFFTCpuEngine final : public detail::DDMTFFTEngine {
public:
    DDMTFFTCpuEngine(const plans::DDMTPlan& plan,
                     const detail::DDMTFFTEngineConfig& cfg)
        : m_plan(plan),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_nbeams(cfg.nbeams),
          m_opts(cfg.options),
          m_model(plan, cfg.options.guard),
          m_nchans(plan.get_nchans()),
          m_ndm(m_model.dm.size()),
          m_nact(m_model.active.size()),
          m_ctx(m_model.context()) {
        m_seg_n = ddmt_fft::max_segment_length(m_ctx, m_nact, kMaxSpectraBytes);
        if (const auto cap = detail::fft_segment_cap(); cap > 0) {
            m_seg_n = std::min(m_seg_n, cap); // testing hook
        }
        m_hist.assign(m_nbeams * m_nchans * m_ctx, 0.0F);
        m_hist_next.assign(m_hist.size(), 0.0F);
        reset_history();
        m_ws.resize(static_cast<SizeType>(m_nthreads));
        logging::debug("DDMTFFT cpu: method={} ndm={} active chans={} "
                       "max_tau={:.2f} guard={} segment={}",
                       to_string(m_opts.method), m_ndm, m_nact, m_model.max_tau,
                       m_model.guard, m_seg_n);
        // NUFFT runs (one plan per distinct run length); every other trial
        // is summed by brute force.
        const bool nufft = m_opts.method == DDMTFFTMethod::kNUFFT;
        std::vector<bool> by_nufft(m_ndm, false);
        for (const auto& r : m_model.runs) {
            if (!nufft || !m_model.nufft_run(r, /*explicit_nufft=*/true)) {
                continue;
            }
            auto& plan_r = m_plans[r.count];
            if (!plan_r) {
                plan_r = std::make_unique<nufft::Type1Plan>(
                    r.count, m_opts.tolerance, m_nthreads);
            }
            m_runs.push_back({r, plan_r.get()});
            std::fill_n(by_nufft.begin() + static_cast<std::ptrdiff_t>(r.begin),
                        r.count, true);
        }
        for (SizeType d = 0; d < m_ndm; ++d) {
            if (!by_nufft[d]) {
                m_brute_dms.push_back(static_cast<std::uint32_t>(d));
            }
        }
        logging::debug("DDMTFFT cpu: {} NUFFT run(s), {} brute-force trials",
                       m_runs.size(), m_brute_dms.size());
    }

    // ---- streaming model ----

    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept override {
        const auto total = m_hist_len + input_nsamps;
        return total > m_ctx ? total - m_ctx : 0;
    }

    // The stream is taken as zero before its first sample: a cold history
    // is `guard` zero samples of look-behind.
    void reset_history() noexcept override {
        std::ranges::fill(m_hist, 0.0F);
        m_hist_len = m_model.guard;
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        return m_nbeams * m_nchans * m_ctx;
    }
    [[nodiscard]] SizeType max_delay() const noexcept override {
        return m_model.max_delay;
    }
    void set_gulp_size(SizeType gulp_size) override { m_gulp = gulp_size; }
    [[nodiscard]] SizeType get_gulp_size() const noexcept override {
        return m_gulp == 0 ? 65536 : m_gulp;
    }
    [[nodiscard]] std::string_view method_used() const noexcept override {
        return detail::ddmt_fft_method_used(m_runs.size(), m_brute_dms.size());
    }

    void save_history(std::span<float> out) const override {
        if (m_hist_len != m_ctx) {
            throw std::logic_error(std::format(
                "DDMTFFT::save_history: stream is not fully warmed up yet "
                "({} of {} history samples/channel)",
                m_hist_len, m_ctx));
        }
        check_size(out.size(), history_state_size(), "save_history");
        std::ranges::copy(m_hist, out.begin());
    }
    void load_history(std::span<const float> in) override {
        check_size(in.size(), history_state_size(), "load_history");
        std::ranges::copy(in, m_hist.begin());
        m_hist_len = m_ctx;
    }

    // ---- execute ----

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        check_float("execute(float)");
        const auto rows = m_nbeams * m_nchans;
        if (waterfall.size() % rows != 0) {
            throw std::invalid_argument(std::format(
                "DDMTFFT::execute: waterfall size {} is not a multiple of "
                "nbeams * nchans = {}",
                waterfall.size(), rows));
        }
        const auto nsamps = waterfall.size() / rows;
        check_size(dmt.size(), m_nbeams * m_ndm * get_output_nsamps(nsamps),
                   "execute (output)");
        core(Source{.f = waterfall.data(), .nsamps = nsamps}, dmt.data());
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<float> dmt) override {
        check_packed("execute(packed)");
        const auto nbits     = m_plan.get_nbits();
        const auto rows      = m_nbeams * m_nchans;
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        check_size(waterfall_packed.size(), rows * row_bytes,
                   "execute (packed input)");
        check_size(dmt.size(), m_nbeams * m_ndm * get_output_nsamps(nsamps),
                   "execute (output)");
        core(Source{.p         = waterfall_packed.data(),
                    .row_bytes = row_bytes,
                    .nbits     = nbits,
                    .nsamps    = nsamps},
             dmt.data());
    }

    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<float> dmt) override {
        check_packed("execute_time_major");
        const auto nbits = m_plan.get_nbits();
        const auto samp_bytes =
            bit_pack_utils::packed_row_bytes(m_nchans, nbits);
        check_size(filterbank_packed.size(), m_nbeams * nsamps * samp_bytes,
                   "execute_time_major (input)");
        check_size(dmt.size(), m_nbeams * m_ndm * get_output_nsamps(nsamps),
                   "execute_time_major (output)");
        const auto rows = m_nbeams * m_nchans;
        m_unpacked.resize(rows * nsamps);
        const auto nb = static_cast<std::int64_t>(m_nbeams * nsamps);
#pragma omp parallel num_threads(m_nthreads)
        {
            std::vector<float> spectrum(m_nchans);
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < nb; ++i) {
                const auto b = static_cast<SizeType>(i) / nsamps;
                const auto s = static_cast<SizeType>(i) % nsamps;
                bit_pack_utils::unpack_row(
                    filterbank_packed.data() +
                        (((b * nsamps) + s) * samp_bytes),
                    nbits, m_nchans, spectrum.data());
                float* dst = m_unpacked.data() + (b * m_nchans * nsamps) + s;
                for (SizeType c = 0; c < m_nchans; ++c) {
                    dst[c * nsamps] = spectrum[c];
                }
            }
        }
        core(Source{.f = m_unpacked.data(), .nsamps = nsamps}, dmt.data());
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    struct Workspace {
        FFTVector<float> time;        // one real row
        FFTVector<ComplexType> batch; // kFwdBatch complex rows
        FFTVector<float> acc;         // kDmBlock tiles of kBinsFl
        FFTVector<float> turns;       // phase set-up (turns)
        FFTVector<float> ph_re;       // phase set-up: sin/cos out
        FFTVector<float> ph_im;
        FFTVector<ComplexType> nufft; // NUFFT scratch (see nufft.hpp)
    };

    const plans::DDMTPlan& m_plan; // owned by the DDMTFFT facade
    int m_nthreads;
    SizeType m_nbeams;
    DDMTFFTOptions m_opts;
    ddmt_fft::DelayModel m_model;
    SizeType m_nchans;
    SizeType m_ndm;
    SizeType m_nact;
    SizeType m_ctx;     // guard + max_delay
    SizeType m_seg_n{}; // longest transform (memory cap)
    SizeType m_gulp{0};

    std::vector<float> m_hist; // [beam][chan][ctx], first m_hist_len valid
    std::vector<float> m_hist_next;
    SizeType m_hist_len{0};
    std::vector<float> m_unpacked;

    std::map<SizeType, std::unique_ptr<Geometry>> m_geos;
    FFTVector<float> m_spectra;        // [tile][nact][kBinsFl]
    FFTVector<ComplexType> m_out_spec; // [ndm][bins_stride]
    std::vector<Workspace> m_ws;
    struct NufftRun {
        ddmt_fft::UniformRun run;
        const nufft::Type1Plan* plan;
    };
    std::map<SizeType, std::unique_ptr<nufft::Type1Plan>> m_plans;
    std::vector<NufftRun> m_runs;           // trials summed by the NUFFT
    std::vector<std::uint32_t> m_brute_dms; // all other trials

    void check_float(std::string_view what) const {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMTFFT::{}: plan nbits={} != 32; use the packed overload",
                what, m_plan.get_nbits()));
        }
    }
    void check_packed(std::string_view what) const {
        if (!bit_pack_utils::detail::is_packed_nbits(
                static_cast<unsigned>(m_plan.get_nbits()))) {
            throw std::invalid_argument(
                std::format("DDMTFFT::{}: plan nbits={} is not a packed width "
                            "(1,2,4,8,16); use the float overload",
                            what, m_plan.get_nbits()));
        }
    }
    static void
    check_size(SizeType got, SizeType expected, std::string_view what) {
        if (got != expected) {
            throw std::invalid_argument(
                std::format("DDMTFFT: {} buffer size mismatch: expected {}, "
                            "got {}",
                            what, expected, got));
        }
    }

    [[nodiscard]] Workspace& ws() {
        return m_ws[static_cast<SizeType>(omp_get_thread_num())];
    }

    Geometry& geometry(SizeType n) {
        // One entry per transform length seen (warm-up, steady state, odd
        // final blocks); a stream of ever-changing block sizes must not grow
        // the plan cache without bound.
        constexpr SizeType kMaxGeometries = 4;
        if (!m_geos.contains(n) && m_geos.size() >= kMaxGeometries) {
            m_geos.clear();
        }
        auto& g = m_geos[n];
        if (!g) {
            g = std::make_unique<Geometry>(n);
        }
        const auto spec_floats = g->n_tiles * m_nact * kBinsFl;
        if (m_spectra.size() < spec_floats) {
            m_spectra.assign(spec_floats, 0.0F);
        }
        if (m_out_spec.size() < m_ndm * g->bins_stride) {
            m_out_spec.assign(m_ndm * g->bins_stride, ComplexType{});
        }
        for (auto& w : m_ws) {
            const auto nt = utils::fft_row_stride<float>(n);
            if (w.time.size() < nt) {
                w.time.assign(nt, 0.0F);
            }
            if (w.batch.size() < kFwdBatch * g->bins_stride) {
                w.batch.assign(kFwdBatch * g->bins_stride, ComplexType{});
            }
        }
        return *g;
    }

    // One streaming step (all beams): combined = history ++ block.
    void core(const Source& data, float* out) {
        const auto n_new = data.nsamps;
        const auto total = m_hist_len + n_new;
        const auto n_out = total > m_ctx ? total - m_ctx : 0;
        // While the stream warms up, also prepare the steady-state
        // transform (n_new outputs per call) so the first warm call does not
        // pay for new FFT plans and first-touch of its buffers.
        if (m_hist_len < m_ctx && n_new > 0) {
            geometry(ddmt_fft::plan_segments(n_new, m_ctx, m_seg_n).n_fft);
        }
        if (n_out > 0 && m_ndm > 0) {
            const auto seg  = ddmt_fft::plan_segments(n_out, m_ctx, m_seg_n);
            auto& g         = geometry(seg.n_fft);
            const auto hop  = seg.hop;
            const auto nseg = seg.nseg;
            for (SizeType b = 0; b < m_nbeams; ++b) {
                for (SizeType s = 0; s < nseg; ++s) {
                    const auto p0 = s * hop; // combined position of row[0]
                    // Spectra layout: 16-bin tiles whenever a NUFFT runs
                    // (the brute-force rest then uses the same tiles).
                    if (m_runs.empty()) {
                        forward<kBins>(g, data, b, p0);
                        channel_sum_brute<kBins>(g);
                    } else {
                        forward<kNuTile>(g, data, b, p0);
                        channel_sum_nufft(g);
                        if (!m_brute_dms.empty()) {
                            channel_sum_brute<kNuTile>(g);
                        }
                    }
                    const auto cnt = std::min(hop, n_out - p0);
                    inverse(g, out + (b * m_ndm * n_out), n_out, p0, cnt);
                }
            }
        }
        update_history(data);
    }

    // Row of combined positions [p0, p0 + N) of beam b, channel c.
    void fill_row(float* row,
                  SizeType n,
                  const Source& data,
                  SizeType b,
                  SizeType c,
                  SizeType p0) const {
        std::fill_n(row, n, 0.0F);
        const float* hist = m_hist.data() + (((b * m_nchans) + c) * m_ctx);
        // Combined position q = history (m_hist_len) then the block.
        const auto lo_h = p0;
        const auto hi_h = std::min(m_hist_len, p0 + n);
        if (lo_h < hi_h) {
            std::copy_n(hist + lo_h, hi_h - lo_h, row + (lo_h - p0));
        }
        const auto lo_b = std::max(m_hist_len, p0);
        const auto hi_b = std::min(m_hist_len + data.nsamps, p0 + n);
        if (lo_b < hi_b) {
            copy_samples(data, (b * m_nchans) + c, lo_b - m_hist_len,
                         hi_b - lo_b, row + (lo_b - p0));
        }
    }

    // Spectra tile-major in split layout with TB bins per tile: [tile][nact]
    // [re x TB, im x TB]. Brute force uses kBins tiles, the NUFFT kNuBins
    // (one bin group is then one contiguous block).
    template <int TB>
    void forward(Geometry& g, const Source& data, SizeType b, SizeType p0) {
        const auto nblocks =
            static_cast<std::int64_t>((m_nact + kFwdBatch - 1) / kFwdBatch);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t bi = 0; bi < nblocks; ++bi) {
            auto& w       = ws();
            const auto a0 = static_cast<SizeType>(bi) * kFwdBatch;
            const auto na = std::min(kFwdBatch, m_nact - a0);
            for (SizeType i = 0; i < na; ++i) {
                fill_row(w.time.data(), g.n_fft, data, b,
                         m_model.active[a0 + i], p0);
                g.r2c->r2c(w.time.data(), w.batch.data() + (i * g.bins_stride));
            }
            constexpr auto kTB = static_cast<SizeType>(TB);
            const auto ntb     = (g.n_bins + kTB - 1) / kTB;
            float* dst         = m_spectra.data() + (a0 * 2 * kTB);
            for (SizeType t = 0; t < ntb; ++t) {
                float* d = dst + (t * m_nact * 2 * kTB);
                for (SizeType i = 0; i < na; ++i) {
                    fft_cpu::stream_split<TB>(
                        d + (i * 2 * kTB),
                        w.batch.data() + (i * g.bins_stride) + (t * kTB));
                }
            }
        }
        fft_cpu::stream_fence();
    }

    // ---- brute force ----

    // Trials m_brute_dms, TB-bin tiles of the spectra (kBins alone; the
    // NUFFT's kNuTile when both methods share the spectra).
    template <int TB> void channel_sum_brute(Geometry& g) {
        const auto nd_all  = m_brute_dms.size();
        const auto ntb     = (g.n_bins + TB - 1) / TB;
        const auto nblk    = (nd_all + kDmBlock - 1) / kDmBlock;
        const auto items   = static_cast<std::int64_t>(ntb * nblk);
        const double inv_n = 1.0 / static_cast<double>(g.n_fft);
        // Items tile-major: threads running concurrently share a tile's
        // spectra in the last-level cache.
#pragma omp parallel for schedule(dynamic, 1) num_threads(m_nthreads)
        for (std::int64_t it = 0; it < items; ++it) {
            const auto t = static_cast<SizeType>(it) / nblk;
            const auto k = static_cast<SizeType>(it) % nblk;
            brute_item<TB>(ws(), g, t, k * kDmBlock, inv_n);
        }
        fft_cpu::stream_fence();
    }

    template <int TB>
    void brute_item(Workspace& w,
                    const Geometry& g,
                    SizeType tile,
                    SizeType i0,
                    double inv_n) {
        constexpr auto kTB       = static_cast<SizeType>(TB);
        constexpr auto kTBFl     = 2 * kTB;
        const auto nd            = std::min(kDmBlock, m_brute_dms.size() - i0);
        const std::uint32_t* dms = m_brute_dms.data() + i0;
        const auto k0            = tile * kTB;
        // phase set-up: per (group channel, trial) kLanes anchors + 1 step
        constexpr SizeType kPer = kLanes + 1;
        const auto nph          = kChanGroup * kDmBlock * kPer;
        if (w.acc.size() < kDmBlock * kTBFl) {
            w.acc.assign(kDmBlock * kTBFl, 0.0F);
        }
        if (w.turns.size() < nph) {
            w.turns.assign(nph, 0.0F);
            w.ph_re.assign(nph, 0.0F);
            w.ph_im.assign(nph, 0.0F);
        }
        float* acc = w.acc.data();
        std::fill_n(acc, nd * kTBFl, 0.0F);
        const float* spec = m_spectra.data() + (tile * m_nact * kTBFl);

        for (SizeType a0 = 0; a0 < m_nact; a0 += kChanGroup) {
            const auto na = std::min(kChanGroup, m_nact - a0);
            setup_phases(w, a0, na, dms, nd, k0, inv_n);
            const float* x[kChanGroup];
            for (SizeType i = 0; i < kChanGroup; ++i) {
                x[i] = spec + ((a0 + std::min(i, na - 1)) * kTBFl);
            }
            for (SizeType d = 0; d < nd; ++d) {
                accumulate<TB>(acc + (d * kTBFl), x, na, w, d);
            }
        }
        // 1/N and store the block's DM rows (interleaved, write-once).
        const auto scale = static_cast<float>(inv_n);
        for (SizeType d = 0; d < nd; ++d) {
            float* a = acc + (d * kTBFl);
#pragma omp simd
            for (SizeType j = 0; j < kTBFl; ++j) {
                a[j] *= scale;
            }
            fft_cpu::stream_interleaved<TB>(
                m_out_spec.data() + (dms[d] * g.bins_stride) + k0, a);
        }
    }

    // turns[(i * kDmBlock + d) * kPer + v]: tau (k0 + v) / N for v <
    // kLanes, and tau kLanes / N at v = kLanes, each wrapped to [-0.5, 0.5)
    // in double; then sin/cos of all of them in one vector pass.
    void setup_phases(Workspace& w,
                      SizeType a0,
                      SizeType na,
                      const std::uint32_t* dms,
                      SizeType nd,
                      SizeType k0,
                      double inv_n) const {
        constexpr SizeType kPer = kLanes + 1;
        float* turns            = w.turns.data();
        for (SizeType i = 0; i < na; ++i) {
            const double rate = m_model.rate[m_model.active[a0 + i]];
            for (SizeType d = 0; d < nd; ++d) {
                const double a = m_model.dm[dms[d]] * rate * inv_n;
                float* u       = turns + (((i * kDmBlock) + d) * kPer);
#pragma omp simd
                for (SizeType v = 0; v < static_cast<SizeType>(kLanes); ++v) {
                    u[v] = static_cast<float>(
                        simd::wrap_turns(a * static_cast<double>(k0 + v)));
                }
                u[kLanes] = static_cast<float>(
                    simd::wrap_turns(a * static_cast<double>(kLanes)));
            }
        }
        const auto n = na * kDmBlock * kPer;
        float* s     = w.ph_im.data();
        float* c     = w.ph_re.data();
#pragma omp simd
        for (SizeType j = 0; j < n; ++j) {
            simd::sincos_turns(turns[j], s[j], c[j]);
        }
    }

    // acc += sum_i x_i * P_i over the TB-bin tile, P_i rotated by R_i every
    // kLanes bins (up to kChanGroup channels).
    template <int TB>
    static void accumulate(float* __restrict__ acc,
                           const float* const* x,
                           SizeType na,
                           const Workspace& w,
                           SizeType d) {
        constexpr SizeType kPer = kLanes + 1;
        float pr[kChanGroup][kLanes];
        float pi[kChanGroup][kLanes];
        float rr[kChanGroup];
        float ri[kChanGroup];
        for (SizeType i = 0; i < kChanGroup; ++i) {
            // Missing channels of a short group get a zero phasor.
            const bool on = i < na;
            const auto o  = ((i * kDmBlock) + d) * kPer;
            for (int v = 0; v < kLanes; ++v) {
                pr[i][v] = on ? w.ph_re[o + v] : 0.0F;
                pi[i][v] = on ? w.ph_im[o + v] : 0.0F;
            }
            rr[i] = on ? w.ph_re[o + kLanes] : 1.0F;
            ri[i] = on ? w.ph_im[o + kLanes] : 0.0F;
        }
        float* __restrict__ yr = acc;
        float* __restrict__ yi = acc + TB;
#if defined(__AVX512F__)
        if constexpr (kLanes == 16 && kChanGroup == 4) {
            accumulate_avx512<TB>(yr, yi, x, pr, pi, rr, ri);
            return;
        }
#endif
        for (int s = 0; s < TB; s += kLanes) {
#pragma omp simd
            for (int v = 0; v < kLanes; ++v) {
                const int j = s + v;
                float ar    = yr[j];
                float ai    = yi[j];
#pragma GCC unroll 4
                for (SizeType i = 0; i < kChanGroup; ++i) {
                    const float xr = x[i][j];
                    const float xi = x[i][TB + j];
                    const float p  = pr[i][v];
                    const float q  = pi[i][v];
                    ar += (xr * p) - (xi * q);
                    ai += (xr * q) + (xi * p);
                    pr[i][v] = (p * rr[i]) - (q * ri[i]);
                    pi[i][v] = (p * ri[i]) + (q * rr[i]);
                }
                yr[j] = ar;
                yi[j] = ai;
            }
        }
    }

#if defined(__AVX512F__)
    // accumulate() for 16-lane zmm registers and 4 channels: 32 FMA-class
    // operations per 64 complex terms, the two partial accumulators keeping
    // the dependent FMA chains short enough to issue at full rate.
    template <int TB>
    static void accumulate_avx512(float* __restrict__ yr,
                                  float* __restrict__ yi,
                                  const float* const* x,
                                  const float (&pr0)[kChanGroup][kLanes],
                                  const float (&pi0)[kChanGroup][kLanes],
                                  const float (&rr0)[kChanGroup],
                                  const float (&ri0)[kChanGroup]) {
        __m512 pr[kChanGroup];
        __m512 pi[kChanGroup];
        __m512 rr[kChanGroup];
        __m512 ri[kChanGroup];
        for (SizeType i = 0; i < kChanGroup; ++i) {
            pr[i] = _mm512_loadu_ps(pr0[i]);
            pi[i] = _mm512_loadu_ps(pi0[i]);
            rr[i] = _mm512_set1_ps(rr0[i]);
            ri[i] = _mm512_set1_ps(ri0[i]);
        }
        const float* x0 = x[0];
        const float* x1 = x[1];
        const float* x2 = x[2];
        const float* x3 = x[3];
        for (int s = 0; s < TB; s += 16) {
            __m512 ar0      = _mm512_loadu_ps(yr + s);
            __m512 ai0      = _mm512_loadu_ps(yi + s);
            __m512 ar1      = _mm512_setzero_ps();
            __m512 ai1      = _mm512_setzero_ps();
            const auto term = [&](const float* xc, int i, __m512& ar,
                                  __m512& ai) {
                const __m512 xr = _mm512_loadu_ps(xc + s);
                const __m512 xi = _mm512_loadu_ps(xc + TB + s);
                ar              = _mm512_fmadd_ps(xr, pr[i], ar);
                ai              = _mm512_fmadd_ps(xr, pi[i], ai);
                ar              = _mm512_fnmadd_ps(xi, pi[i], ar);
                ai              = _mm512_fmadd_ps(xi, pr[i], ai);
                const __m512 np =
                    _mm512_fmsub_ps(pr[i], rr[i], _mm512_mul_ps(pi[i], ri[i]));
                pi[i] =
                    _mm512_fmadd_ps(pr[i], ri[i], _mm512_mul_ps(pi[i], rr[i]));
                pr[i] = np;
            };
            term(x0, 0, ar0, ai0);
            term(x1, 1, ar1, ai1);
            term(x2, 2, ar0, ai0);
            term(x3, 3, ar1, ai1);
            _mm512_storeu_ps(yr + s, _mm512_add_ps(ar0, ar1));
            _mm512_storeu_ps(yi + s, _mm512_add_ps(ai0, ai1));
        }
    }
#endif

    // ---- NUFFT (uniform DM grids) ----

    // Per bin k: Y(d) = sum_c a_c exp(2 pi i d x_c) with a_c = X_c(k)
    // exp(2 pi i k dm0 r_c / N) and x_c = k ddm r_c / N (turns); a_c is
    // handed over already centred (times exp(2 pi i half x_c)). Per bin the
    // points are prepared in one vector pass over the channels (grid cell,
    // kernel offset, rotated strength) and spread by Type1Plan's located-
    // point path. Bins are taken kNuBins at a time so each DM row receives
    // one contiguous run.

    void channel_sum_nufft(Geometry& g) {
        const double inv_n = 1.0 / static_cast<double>(g.n_fft);
        const auto groups =
            static_cast<std::int64_t>((g.n_bins + kNuBins - 1) / kNuBins);
        const auto scale = static_cast<float>(inv_n);
        const auto nact  = m_nact;
        const auto nruns = m_runs.size();
        // Per run and channel: x_c per bin (ddm r_c / N) and the phase rate
        // of a_c, folded with the NUFFT's centring shift half * x_c.
        std::vector<double> xr(nruns * nact);
        std::vector<double> sr(nruns * nact);
        SizeType max_count = 0;
        for (SizeType q = 0; q < nruns; ++q) {
            const auto& r = m_runs[q].run;
            const double shift =
                r.dm0 + (static_cast<double>(m_runs[q].plan->half()) * r.ddm);
            for (SizeType a = 0; a < nact; ++a) {
                const double rt    = m_model.rate[m_model.active[a]] * inv_n;
                xr[(q * nact) + a] = r.ddm * rt;
                sr[(q * nact) + a] = shift * rt;
            }
            max_count = std::max(max_count, r.count);
        }
#pragma omp parallel num_threads(m_nthreads)
        {
            auto& w = ws();
            std::vector<std::int32_t> l0(nact);
            std::vector<float> u(nact);
            std::vector<float> ar(nact);
            std::vector<float> ai(nact);
            std::vector<ComplexType> y(max_count);
            std::vector<ComplexType> ybuf(m_ndm * kNuBins);
#pragma omp for schedule(dynamic, 1)
            for (std::int64_t gi = 0; gi < groups; ++gi) {
                const auto k0 = static_cast<SizeType>(gi) * kNuBins;
                const auto nk = std::min(kNuBins, g.n_bins - k0);
                const float* grp =
                    m_spectra.data() +
                    (static_cast<SizeType>(gi) * nact * 2 * kNuBins);
                for (SizeType b = 0; b < nk; ++b) {
                    const float* re = grp + b;
                    const double kk = static_cast<double>(k0 + b);
                    for (SizeType q = 0; q < nruns; ++q) {
                        const auto& run = m_runs[q];
                        prepare_points(re, kk, &xr[q * nact], &sr[q * nact],
                                       run.plan->locator(), l0.data(), u.data(),
                                       ar.data(), ai.data());
                        const std::span<ComplexType> yr(y.data(),
                                                        run.run.count);
                        run.plan->execute_points(l0, u, ar, ai, yr, w.nufft);
                        ComplexType* dst =
                            ybuf.data() + (run.run.begin * kNuBins) + b;
                        for (SizeType d = 0; d < run.run.count; ++d) {
                            dst[d * kNuBins] = yr[d] * scale;
                        }
                    }
                }
                // One contiguous run of nk bins per DM row.
                for (const auto& run : m_runs) {
                    for (SizeType d = run.run.begin;
                         d < run.run.begin + run.run.count; ++d) {
                        std::copy_n(&ybuf[d * kNuBins], nk,
                                    &m_out_spec[(d * g.bins_stride) + k0]);
                    }
                }
            }
        }
    }

    // Points of one bin: channel a's spectrum value is re[a * 2 kNuBins] +
    // i re[a * 2 kNuBins + kNuBins].
    void prepare_points(const float* __restrict__ re,
                        double kk,
                        const double* __restrict__ xr,
                        const double* __restrict__ sr,
                        nufft::Type1Plan::Locator loc,
                        std::int32_t* __restrict__ l0,
                        float* __restrict__ u,
                        float* __restrict__ ar,
                        float* __restrict__ ai) const noexcept {
        const auto nact = static_cast<std::int64_t>(m_nact);
#pragma omp simd
        for (std::int64_t a = 0; a < nact; ++a) {
            nufft::locate(kk * xr[a], loc, l0[a], u[a]);
            float s = 0.0F;
            float c = 0.0F;
            simd::sincos_turns(static_cast<float>(simd::wrap_turns(kk * sr[a])),
                               s, c);
            constexpr auto kStride = static_cast<std::int64_t>(2 * kNuBins);
            const float xre        = re[a * kStride];
            const float xim        = re[(a * kStride) + kNuBins];
            ar[a]                  = (xre * c) - (xim * s);
            ai[a]                  = (xre * s) + (xim * c);
        }
    }

    // DM rows -> time; keep transform samples [guard, guard + cnt) as output
    // samples [p0, p0 + cnt) of this call.
    void inverse(
        Geometry& g, float* out, SizeType n_out, SizeType p0, SizeType cnt) {
        const auto ndm = static_cast<std::int64_t>(m_ndm);
        const auto g0  = m_model.guard;
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t di = 0; di < ndm; ++di) {
            const auto d = static_cast<SizeType>(di);
            float* row   = ws().time.data();
            g.c2r->c2r(&m_out_spec[d * g.bins_stride], row);
            std::copy_n(row + g0, cnt, out + (d * n_out) + p0);
        }
    }

    // Retain the last min(total, ctx) combined samples of every row.
    void update_history(const Source& data) {
        const auto n_new   = data.nsamps;
        const auto total   = m_hist_len + n_new;
        const auto new_len = std::min(total, m_ctx);
        const auto start   = total - new_len; // combined position
        const auto rows    = static_cast<std::int64_t>(m_nbeams * m_nchans);
#pragma omp parallel for schedule(static) num_threads(m_nthreads)
        for (std::int64_t r = 0; r < rows; ++r) {
            const auto ri  = static_cast<SizeType>(r);
            const float* h = m_hist.data() + (ri * m_ctx);
            float* dst     = m_hist_next.data() + (ri * m_ctx);
            // [start, m_hist_len) from the old history, the rest from the
            // block.
            const auto from_h =
                start < m_hist_len ? m_hist_len - start : SizeType{0};
            const auto nh = std::min(from_h, new_len);
            std::copy_n(h + start, nh, dst);
            if (nh < new_len) {
                copy_samples(data, ri, (start + nh) - m_hist_len, new_len - nh,
                             dst + nh);
            }
        }
        std::swap(m_hist, m_hist_next);
        m_hist_len = new_len;
    }
};

} // namespace

std::unique_ptr<detail::DDMTFFTEngine>
detail::make_ddmt_fft_cpu(const plans::DDMTPlan& plan,
                          const detail::DDMTFFTEngineConfig& cfg) {
    return std::make_unique<DDMTFFTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
