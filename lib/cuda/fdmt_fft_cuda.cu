#include "dmt/algorithms/fdmt_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "dmt/fdmt_fft_tree.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/fourier_gpu.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"

// FDMT-FFT on the GPU.
//
// Every frequency bin runs the same tree, node = tail + head * W^(s * k), so
// the tree is run as programs over tiles of 32 bins (one bin per lane): each
// warp applies one merge to 32 bins at a time, so the merge list is read
// once per 32 bins. The tree is cut into stages (fdmt_fft_tree.hpp); a
// stage's programs keep their intermediate levels in shared memory and only
// the stage boundaries go through device memory. At the reference point
// (4096 channels, 2049 trials) that is two stages, meeting near level 7:
// about 90 KB of device traffic per bin instead of ~770 KB for a
// level-by-level tree. Per call and overlap-save segment (the segment rule
// every engine shares):
//
//   1. fill the channel windows (float or packed input, history, kill mask)
//   2. R2C of every channel row                 spectra [beam][chan][bins]
//   3. tree stages                              out_spec [beam][dm][bins]
//   4. C2R of every DM row, trimmed into the output
//
// Phases W^(s k) = exp(-2 pi i s k / N) come from an exact 0.64 fixed-point
// turn count per shift (s / N mod 1), multiplied by k modulo 2^64, then
// sincospi: no tables, no FP64, integer and fractional shifts alike. 1/N is
// folded into the level-0 windows.
//
// The stepper (reset/advance/view/finalize) runs one single-level stage per
// advance on full-width state at the single-transform length, allocated on
// first use.

namespace dmt::algorithms {

namespace {

using fourier_gpu::blocks_for;
using fourier_gpu::cfma;
using fourier_gpu::cmul;
using fourier_gpu::DevBuf;
using fourier_gpu::grid_y;
using fourier_gpu::kBlock;
using fourier_gpu::kMaxGridY;
using fourier_gpu::phase_fixed;
using fourier_gpu::phasor;

constexpr int kTileBins     = 32;  // bins per program tile (one per lane)
constexpr int kStageThreads = 768; // 24 warps: two blocks per SM
// Intermediate nodes per program (two shared-memory buffers of 32 bins).
constexpr SizeType kNodeBudget = 96;
// Output nodes per program (bounds the work of one block).
constexpr SizeType kMaxOut = 512;
// Rows per C2R batch of the stepper views.
constexpr SizeType kViewBatch = 256;

// Sample s of row `row` of the input: float32 (NB = 32) or packed at NB bits.
template <int NB>
__device__ __forceinline__ float
read_sample(const void* __restrict__ src, int64_t row, int64_t row_len, int64_t s) {
    if constexpr (NB == 32) {
        return static_cast<const float*>(src)[(row * row_len) + s];
    } else {
        const auto* r = static_cast<const uint8_t*>(src) + (row * row_len);
        return static_cast<float>(
            bit_pack_utils::read_packed_sample<NB>(r, static_cast<SizeType>(s)));
    }
}

// Transform input of one segment: window row (beam, chan) sample t is v[p0 +
// t] of v = [zeros(z) | history (lov) | block (nsamps) | zeros...]. Killed
// channels read as zero. row_len: floats (NB = 32) or bytes per input row.
template <int NB>
__global__ void kernel_fill(const void* __restrict__ src,
                            int64_t row_len,
                            const float* __restrict__ hist,
                            const uint8_t* __restrict__ keep,
                            float* __restrict__ win,
                            int64_t nrows,
                            int nchans,
                            int64_t nsamps,
                            int64_t n_fft,
                            int64_t p0,
                            int64_t z,
                            int64_t lov) {
    const auto t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (t >= n_fft) {
        return;
    }
    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        const auto chan = static_cast<int>(row % nchans);
        const auto q    = p0 + t;
        float v         = 0.0F;
        if (keep == nullptr || keep[chan] != 0) {
            if (q >= z && q < z + lov) {
                v = hist[(row * lov) + (q - z)];
            } else if (q >= z + lov && q < z + lov + nsamps) {
                v = read_sample<NB>(src, row, row_len, q - z - lov);
            }
        }
        win[(row * n_fft) + t] = v;
    }
}

// New history: the last L samples of [history (L) | block (nsamps)].
template <int NB>
__global__ void kernel_update_history(const void* __restrict__ src,
                                      int64_t row_len,
                                      const float* __restrict__ hist,
                                      float* __restrict__ hist_next,
                                      int64_t nrows,
                                      int64_t nsamps,
                                      int64_t len) {
    const auto i = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= len) {
        return;
    }
    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        const auto q = nsamps + i;
        hist_next[(row * len) + i] =
            q < len ? hist[(row * len) + q]
                    : read_sample<NB>(src, row, row_len, q - len);
    }
}

// Level-0 windows W0[s][k] = (box ? sum_{u <= s} W^(u k) : W^(s k)) / N.
__global__ void kernel_level0_windows(const unsigned long long* __restrict__ sfx,
                                      float2* __restrict__ w0,
                                      int dt0_max,
                                      int nb,
                                      int box,
                                      float norm) {
    const auto k = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (k >= nb) {
        return;
    }
    float2 acc{0.0F, 0.0F};
    for (int s = 0; s <= dt0_max; ++s) {
        const float2 w = phasor(sfx[s], static_cast<unsigned>(k));
        acc.x += w.x;
        acc.y += w.y;
        const float2 v = box != 0 ? acc : w;
        w0[(static_cast<int64_t>(s) * nb) + k] = {v.x * norm, v.y * norm};
    }
}

// One tree stage (fdmt_fft_tree.hpp): program blockIdx.x on bins
// [32 * (blockIdx.y + tile0), + 32) of beam blockIdx.z. Warps take the ops
// of a level in turn; intermediate levels live in shared memory (two
// buffers of max_nodes x 32 bins), the last level goes to `out`. kL0: the
// input level is level 0, formed from the channel spectra and windows.
template <bool kL0>
__global__ void __launch_bounds__(kStageThreads)
    kernel_tree_stage(const uint32_t* __restrict__ out_begin,
                      const uint32_t* __restrict__ lvl_off,
                      const uint4* __restrict__ ops,
                      const unsigned long long* __restrict__ sfx,
                      const float2* __restrict__ in,
                      int64_t in_beam,
                      const uint32_t* __restrict__ l0_chan,
                      const uint32_t* __restrict__ l0_shift,
                      const float2* __restrict__ w0,
                      float2* __restrict__ out,
                      int64_t out_beam,
                      int64_t nb,
                      int nlev,
                      int max_nodes,
                      int tile0,
                      int64_t half_m,
                      unsigned long long s1) {
    extern __shared__ float2 smem[];
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int nwarp = static_cast<int>(blockDim.x) >> 5;
    const auto p    = static_cast<int64_t>(blockIdx.x);
    const auto k    = ((static_cast<int64_t>(blockIdx.y) + tile0) * kTileBins) +
                   lane;
    const auto beam = static_cast<int64_t>(blockIdx.z);
    const float2* src = in + (beam * in_beam);
    float2* dst       = out + (beam * out_beam);
    float2* bufs[2]   = {smem, smem + (static_cast<int64_t>(max_nodes) * 32)};
    const uint32_t* off = lvl_off + (p * nlev);
    const uint32_t ob   = out_begin[p];
    // kL0 with half_m > 0: `in` holds the length-M complex FFTs of the
    // channel rows read as x[2n] + i x[2n + 1] (M = N / 2); bin k of the
    // real FFT is E + W^k O from Z[k] and conj(Z[M - k]).
    const float2 wk = (kL0 && half_m > 0)
                          ? phasor(s1, static_cast<unsigned>(k))
                          : float2{1.0F, 0.0F};
    int cur = 0;
    for (int lev = 0; lev < nlev; ++lev) {
        const uint32_t o0   = off[lev];
        const uint32_t o1   = off[lev + 1];
        const bool last     = lev == nlev - 1;
        const float2* prev  = bufs[cur ^ 1];
        float2* next        = bufs[cur];
        const auto fetch    = [&](uint32_t idx) -> float2 {
            if (lev == 0) {
                if constexpr (kL0) {
                    const float2 w =
                        w0[(static_cast<int64_t>(l0_shift[idx]) * nb) + k];
                    if (half_m == 0) {
                        return cmul(
                            src[(static_cast<int64_t>(l0_chan[idx]) * nb) + k],
                            w);
                    }
                    if (k > half_m) {
                        return float2{0.0F, 0.0F};
                    }
                    const float2* z =
                        src + (static_cast<int64_t>(l0_chan[idx]) * half_m);
                    const float2 a  = z[k == half_m ? 0 : k];
                    const float2 bz = z[k == 0 ? 0 : half_m - k];
                    // E = (a + conj(bz)) / 2, D = (a - conj(bz)) / 2,
                    // O = -i D, X = E + W^k O.
                    const float2 e{0.5F * (a.x + bz.x), 0.5F * (a.y - bz.y)};
                    const float2 d{0.5F * (a.x - bz.x), 0.5F * (a.y + bz.y)};
                    const float2 x = cfma(e, float2{d.y, -d.x}, wk);
                    return cmul(x, w);
                } else {
                    return src[(static_cast<int64_t>(idx) * nb) + k];
                }
            }
            return prev[(static_cast<int64_t>(idx) * 32) + lane];
        };
        for (uint32_t j = o0 + warp; j < o1; j += nwarp) {
            const uint4 op = ops[j];
            float2 r       = fetch(op.x);
            if (op.y != fdmt_fft::kTreeCopy) {
                r = cfma(r, fetch(op.y),
                         phasor(sfx[op.z], static_cast<unsigned>(k)));
            }
            if (last) {
                dst[((static_cast<int64_t>(ob) + (j - o0)) * nb) + k] = r;
            } else {
                next[(static_cast<int64_t>(j - o0) * 32) + lane] = r;
            }
        }
        __syncthreads();
        cur ^= 1;
    }
}

// Level-0 state (stepper): state[beam][node][k] = spectrum * window.
__global__ void kernel_level0_state(const float2* __restrict__ spec,
                                    const uint32_t* __restrict__ l0_chan,
                                    const uint32_t* __restrict__ l0_shift,
                                    const float2* __restrict__ w0,
                                    float2* __restrict__ state,
                                    int64_t nodes,
                                    int64_t nb,
                                    int64_t spec_beam,
                                    int64_t state_beam) {
    const auto k    = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const auto beam = static_cast<int64_t>(blockIdx.z);
    if (k >= nb) {
        return;
    }
    for (int64_t n = blockIdx.y; n < nodes; n += gridDim.y) {
        state[(beam * state_beam) + (n * nb) + k] =
            cmul(spec[(beam * spec_beam) + (static_cast<int64_t>(l0_chan[n]) * nb) + k],
                 w0[(static_cast<int64_t>(l0_shift[n]) * nb) + k]);
    }
}

// dst row r, samples [o0, o0 + cnt) = time row r from sample skip; with
// zero_tail, samples past the transform are zero (stepper views).
__global__ void kernel_trim(const float* __restrict__ time,
                            float* __restrict__ dst,
                            int64_t nrows,
                            int64_t n_fft,
                            int64_t dst_len,
                            int64_t o0,
                            int64_t cnt,
                            int64_t skip) {
    const auto t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (t >= cnt) {
        return;
    }
    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        const auto s = skip + t;
        dst[(row * dst_len) + o0 + t] =
            s < n_fft ? time[(row * n_fft) + s] : 0.0F;
    }
}

// Copies `rows` spectrum rows (dist nb) and fixes their Hermitian edges.
__global__ void kernel_copy_rows(const float2* __restrict__ src,
                                 float2* __restrict__ dst,
                                 int64_t nrows,
                                 int64_t nb) {
    const auto k = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (k >= nb) {
        return;
    }
    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        dst[(row * nb) + k] = src[(row * nb) + k];
    }
}

// ---- host helpers ----

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

bool valid_nbits(SizeType nbits) {
    return nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8 || nbits == 16;
}

// Device image of a list of tree stages (one upload).
struct StagesD {
    struct Stage {
        SizeType level_in;
        SizeType level_out;
        SizeType nprog;
        SizeType nlev;
        SizeType max_nodes;
        SizeType out_off; // offsets into the flat arrays
        SizeType lvl_off;
        SizeType ops_off;
    };
    std::vector<Stage> stages;
    DevBuf<uint32_t> out_begin;
    DevBuf<uint32_t> lvl_off;
    DevBuf<uint4> ops;

    void build(const std::vector<fdmt_fft::TreeStage>& ts) {
        std::vector<uint32_t> ob;
        std::vector<uint32_t> lo;
        std::vector<uint4> op;
        stages.clear();
        for (const auto& s : ts) {
            stages.push_back({
                .level_in  = s.level_in,
                .level_out = s.level_out,
                .nprog     = s.nprograms(),
                .nlev      = s.nlevels(),
                .max_nodes = s.max_nodes,
                .out_off   = ob.size(),
                .lvl_off   = lo.size(),
                .ops_off   = op.size(),
            });
            ob.insert(ob.end(), s.out_begin.begin(), s.out_begin.end());
            lo.insert(lo.end(), s.lvl_off.begin(), s.lvl_off.end());
            for (const auto& o : s.ops) {
                op.push_back({o.tail, o.head, o.sid, 0U});
            }
        }
        out_begin.upload(ob);
        lvl_off.upload(lo);
        ops.upload(op);
    }
};

// One transform length: FFT plans, phase and window tables and the buffers
// of execute().
struct Geo {
    SizeType n_fft{};
    SizeType n_bins{};
    SizeType nb{}; // spectrum row distance: n_bins rounded up to the tile
    std::unique_ptr<utils::CUFFTManager> r2c; // nbeams * nchans rows (lazy
                                              // when half-length is used)
    std::unique_ptr<utils::CUFFTManager> c2r; // nbeams * ndms rows
    // Even N: forward transforms as in-place length-N/2 C2C of the window
    // rows (the real-FFT post-processing is done by the first tree stage).
    std::unique_ptr<utils::CUFFTManager> half;
    unsigned long long s1{}; // phase_fixed(1, N)
    DevBuf<unsigned long long> sfx;           // per merge sid
    DevBuf<float2> w0;                        // [dt0_max + 1][nb]
};

class FDMTFFTCudaEngine final : public detail::FDMTFFTEngine {
public:
    FDMTFFTCudaEngine(const plans::FDMTPlan& plan,
                      const detail::FDMTFFTEngineConfig& cfg)
        : m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_nbeams(cfg.nbeams),
          m_device_id(cfg.exec.device),
          m_use_box_smearing(cfg.use_box_smearing),
          m_mode(cfg.mode),
          m_frac(cfg.fractional_delays),
          m_plan(&plan) {
        if (m_nbeams == 0) {
            throw std::invalid_argument("FDMTFFT: nbeams must be positive");
        }
        gpu_utils::set_device(m_device_id);
        initialize(cfg.kill_mask);
    }

    ~FDMTFFTCudaEngine() override {
        try {
            gpu_utils::set_device(m_device_id);
            m_device_work.wait();
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
    }

    FDMTFFTCudaEngine(const FDMTFFTCudaEngine&)            = delete;
    FDMTFFTCudaEngine& operator=(const FDMTFFTCudaEngine&) = delete;
    FDMTFFTCudaEngine(FDMTFFTCudaEngine&&)                 = delete;
    FDMTFFTCudaEngine& operator=(FDMTFFTCudaEngine&&)      = delete;

    // ---- execute ----

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        check_sizes(waterfall.size(), dmt.size(), "execute");
        const auto s = host_begin();
        m_in_d.reserve(waterfall.size());
        m_host.to_device(m_in_d.data(), waterfall.data(),
                         waterfall.size_bytes());
        run(Input{.data = m_in_d.data(), .row_len = m_nsamps, .nbits = 32},
            out_stage(), s);
        host_finish(dmt);
    }

    void execute(std::span<const uint8_t> packed,
                 SizeType nbits,
                 bool time_major,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        check_nbits(nbits, time_major ? "execute_time_major" : "execute");
        const auto bytes = time_major ? tm_bytes(nbits) : cm_bytes(nbits);
        if (packed.size() != bytes) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::{}: expected {} packed bytes, got {}",
                time_major ? "execute_time_major" : "execute", bytes,
                packed.size()));
        }
        check_out(dmt.size(), "execute");
        const auto s = host_begin();
        m_packed_d.reserve(packed.size());
        m_host.to_device(m_packed_d.data(), packed.data(), packed.size());
        if (time_major) {
            run(unpack_time_major(m_packed_d.data(), nbits, s), out_stage(),
                s);
        } else {
            run(packed_input(m_packed_d.data(), nbits), out_stage(), s);
        }
        host_finish(dmt);
    }

    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        gpu_utils::set_device(m_device_id);
        check_sizes(d_waterfall.size(), d_dmt.size(), "execute");
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        run(Input{.data = d_waterfall.data(), .row_len = m_nsamps, .nbits = 32},
            d_dmt.data(), s);
        m_device_work.mark(s);
    }

    void execute(DeviceSpan<const uint8_t> d_packed,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        gpu_utils::set_device(m_device_id);
        check_nbits(nbits, "execute");
        if (d_packed.size() != cm_bytes(nbits)) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: expected {} packed bytes, got {}",
                cm_bytes(nbits), d_packed.size()));
        }
        check_out(d_dmt.size(), "execute");
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        run(packed_input(d_packed.data(), nbits), d_dmt.data(), s);
        m_device_work.mark(s);
    }

    // ---- stepper ----

    void reset(std::span<const float> waterfall,
               std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        check_sizes(waterfall.size(), dmt.size(), "reset");
        const auto s = host_begin();
        m_step_in_d.reserve(waterfall.size());
        m_host.to_device(m_step_in_d.data(), waterfall.data(),
                         waterfall.size_bytes());
        start_stepper(
            Input{.data = m_step_in_d.data(), .row_len = m_nsamps, .nbits = 32},
            out_stage(), s);
        m_host_dmt = dmt;
    }

    void reset(std::span<const uint8_t> packed,
               SizeType nbits,
               std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        check_nbits(nbits, "reset");
        if (packed.size() != cm_bytes(nbits)) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: expected {} packed bytes, got {}",
                cm_bytes(nbits), packed.size()));
        }
        check_out(dmt.size(), "reset");
        const auto s = host_begin();
        m_step_in_d.reserve((packed.size() + 3) / 4);
        m_host.to_device(m_step_in_d.data(), packed.data(), packed.size());
        start_stepper(packed_input(m_step_in_d.data(), nbits), out_stage(), s);
        m_host_dmt = dmt;
    }

    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream) override {
        gpu_utils::set_device(m_device_id);
        check_sizes(d_waterfall.size(), d_dmt.size(), "reset");
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        start_stepper(
            Input{.data = d_waterfall.data(), .row_len = m_nsamps, .nbits = 32},
            d_dmt.data(), s);
        m_host_dmt = {};
        m_device_work.mark(s);
    }

    void reset(DeviceSpan<const uint8_t> d_packed,
               SizeType nbits,
               DeviceSpan<float> d_dmt,
               Stream stream) override {
        gpu_utils::set_device(m_device_id);
        check_nbits(nbits, "reset");
        if (d_packed.size() != cm_bytes(nbits)) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: expected {} packed bytes, got {}",
                cm_bytes(nbits), d_packed.size()));
        }
        check_out(d_dmt.size(), "reset");
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        start_stepper(packed_input(d_packed.data(), nbits), d_dmt.data(), s);
        m_host_dmt = {};
        m_device_work.mark(s);
    }

    void advance(SizeType levels, Stream stream) override {
        require_stepper();
        gpu_utils::set_device(m_device_id);
        const auto s = active(stream);
        while (levels > 0 && m_level < total_levels() - 1) {
            const auto& st = m_step_stages.stages[m_level];
            launch_stage(m_step_stages, st, *m_step_geo, m_step_in, m_step_out,
                         m_max_coords * m_step_geo->nb,
                         m_max_coords * m_step_geo->nb, false, s);
            std::swap(m_step_in, m_step_out);
            ++m_level;
            m_view_valid = false;
            --levels;
        }
        gpu_utils::check_last_gpu_error("FDMTFFT (cuda): advance");
        m_device_work.mark(s);
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        require_stepper();
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target = total_lvl - 1 - remaining_levels;
        if (target > m_level) {
            advance(target - m_level, stream);
        }
    }

    void finalize(Stream stream) override {
        require_stepper();
        gpu_utils::set_device(m_device_id);
        const auto s = active(stream);
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        auto& g = *m_step_geo;
        // Root rows of every beam -> out_spec layout, then C2R.
        {
            const dim3 grid(blocks_for(static_cast<int64_t>(g.nb)),
                            grid_y(static_cast<int64_t>(m_ndms)));
            for (SizeType b = 0; b < m_nbeams; ++b) {
                kernel_copy_rows<<<grid, kBlock, 0, s>>>(
                    m_step_in + (b * m_max_coords * g.nb),
                    m_step_out + (b * m_ndms * g.nb),
                    static_cast<int64_t>(m_ndms), static_cast<int64_t>(g.nb));
            }
        }
        inverse(g, m_step_out, m_step_time_d.data(), m_step_dmt,
                m_geom.skip, 0, m_nsamps_out, s);
        if (m_mode == FDMTMode::kValid) {
            update_history(m_step_src, s);
        }
        gpu_utils::check_last_gpu_error("FDMTFFT (cuda): finalize");
        if (!m_host_dmt.empty()) {
            host_finish(m_host_dmt);
            m_host_dmt = {};
        } else {
            m_device_work.mark(s);
        }
        m_initialized = false;
        m_view_valid  = false;
    }

    [[nodiscard]] SizeType current_level() const noexcept override {
        return m_level;
    }
    [[nodiscard]] SizeType num_subbands() const override {
        require_stepper();
        return m_plan->get_container().state_shape[m_level].nchans;
    }
    [[nodiscard]] bool is_finished() const noexcept override {
        return m_initialized && (m_level >= total_levels() - 1);
    }

    [[nodiscard]] DeviceSpan<const float>
    view_level_data_device() const override {
        require_stepper();
        materialize_view();
        const auto& shape = m_plan->get_container().state_shape[m_level];
        return {m_view_d.data(), shape.ncoords * view_nsamps(), device()};
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        const float* v    = host_view();
        const auto& shape = m_plan->get_container().state_shape[m_level];
        return {v, shape.ncoords * view_nsamps()};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        const auto dev    = view_subband_device(subband_idx);
        const float* view = host_view();
        const auto offset =
            static_cast<SizeType>(dev.data.data() - m_view_d.data());
        return FDMTSubbandView{
            .data = std::span<const float>(view + offset, dev.data.size()),
            .subband_idx = dev.subband_idx,
            .ndt         = dev.ndt,
            .nsamps      = dev.nsamps,
            .f_start     = dev.f_start,
            .f_end       = dev.f_end,
            .dt_grid     = dev.dt_grid,
        };
    }

    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const override {
        require_stepper();
        materialize_view();
        const auto& plan_c = m_plan->get_container();
        const auto& shape  = plan_c.state_shape[m_level];
        if (subband_idx >= shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTFFT: Subband index {} out of range ({} subbands)",
                subband_idx, shape.nchans));
        }
        const auto& grid    = plan_c.grids[m_level][subband_idx];
        const auto nsamps_v = view_nsamps();
        return FDMTSubbandDeviceView{
            .data        = DeviceSpan<const float>(
                m_view_d.data() + (grid.coord_offset * nsamps_v),
                grid.ndt * nsamps_v, device()),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps_v,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    // ---- history ----

    void reset_history() noexcept override {
        try {
            gpu_utils::set_device(m_device_id);
            m_device_work.wait();
            if (m_hist_len > 0) {
                cudaMemset(m_hist[m_hist_cur].data(), 0,
                           history_state_size() * sizeof(float));
            }
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        return m_nbeams * m_nchans * m_hist_len;
    }

    void save_history(std::span<float> out) const override {
        check_history(out.size(), "save_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        if (!out.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpy(out.data(), m_hist[m_hist_cur].data(),
                           out.size_bytes(), cudaMemcpyDeviceToHost),
                "FDMTFFT::save_history");
        }
    }

    void load_history(std::span<const float> in) override {
        check_history(in.size(), "load_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        if (!in.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpy(m_hist[m_hist_cur].data(), in.data(),
                           in.size_bytes(), cudaMemcpyHostToDevice),
                "FDMTFFT::load_history");
        }
    }

    void save_history(DeviceSpan<float> out, Stream stream) const override {
        check_history(out.size(), "save_history");
        gpu_utils::set_device(m_device_id);
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        if (out.size() > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(out.data(), m_hist[m_hist_cur].data(),
                                out.size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, s),
                "FDMTFFT::save_history");
        }
        m_device_work.mark(s);
    }

    void load_history(DeviceSpan<const float> in, Stream stream) override {
        check_history(in.size(), "load_history");
        gpu_utils::set_device(m_device_id);
        const auto s = to_cuda(stream);
        m_device_work.order(s);
        if (in.size() > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_hist[m_hist_cur].data(), in.data(),
                                in.size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, s),
                "FDMTFFT::load_history");
        }
        m_device_work.mark(s);
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    // One call's input rows on the device: float32 (nbits 32, row_len =
    // floats per row) or packed (row_len = bytes per row).
    struct Input {
        const void* data{nullptr};
        SizeType row_len{0};
        SizeType nbits{32};
    };

    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_nbeams;
    int m_device_id;
    bool m_use_box_smearing;
    FDMTMode m_mode;
    bool m_frac;
    const plans::FDMTPlan* m_plan; // owned by the FDMTFFT facade

    fdmt_fft::Geometry m_geom;
    fdmt_fft::TreeDag m_dag;
    fdmt_fft::Segments m_seg;
    SizeType m_ndms{};
    SizeType m_nsamps_out{};
    SizeType m_max_coords{};
    SizeType m_mid_rows{}; // rows of the largest inner stage boundary
    int m_smem_optin{};

    DevBuf<uint32_t> m_l0_chan;
    DevBuf<uint32_t> m_l0_shift;
    DevBuf<uint8_t> m_keep; // empty: keep every channel
    bool m_has_kill{false};

    // execute()
    std::unique_ptr<Geo> m_geo;
    StagesD m_stages;
    DevBuf<float> m_win;     // windows, then C2R output
    DevBuf<float2> m_spec;   // [beam][chan][nb]
    DevBuf<float2> m_mid[2]; // inner stage boundaries [beam][rows][nb]
    DevBuf<float2> m_out_spec; // [beam][ndm][nb]

    // History [beam][chan][m_hist_len] (valid mode), double-buffered.
    SizeType m_hist_len{0};
    DevBuf<float> m_hist[2];
    int m_hist_cur{0};

    // Host-memory calls.
    gpu_host::ChunkedStager m_host;
    DevBuf<float> m_in_d;
    DevBuf<uint8_t> m_packed_d;
    DevBuf<float> m_unpacked_d; // time-major input, channel-major float
    DevBuf<float> m_out_d;
    std::span<float> m_host_dmt; // stepper result destination (host reset)
    mutable gpu_utils::DeviceWorkFence m_device_work;

    // Stepper (lazy).
    std::unique_ptr<Geo> m_step_geo_own;
    Geo* m_step_geo{nullptr};
    StagesD m_step_stages;
    DevBuf<float> m_step_win;
    DevBuf<float2> m_step_spec;
    DevBuf<float2> m_state[2];
    DevBuf<float> m_step_time_d;
    DevBuf<float> m_step_in_d; // host reset(): the staged block
    float2* m_step_in{nullptr};
    float2* m_step_out{nullptr};
    float* m_step_dmt{nullptr};
    Input m_step_src{};
    cudaStream_t m_step_stream{nullptr};
    SizeType m_level{0};
    bool m_initialized{false};
    mutable bool m_view_valid{false};
    mutable DevBuf<float2> m_view_spec;
    mutable DevBuf<float> m_view_time;
    mutable DevBuf<float> m_view_d;
    mutable std::unique_ptr<utils::CUFFTManager> m_view_c2r;
    mutable std::vector<float> m_view_h;
    mutable bool m_view_h_valid{false};

    [[nodiscard]] Device device() const noexcept {
        return {.backend = detail::kGPUBackend, .id = m_device_id};
    }
    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan->get_niters() + 1;
    }
    [[nodiscard]] cudaStream_t active(Stream stream) const noexcept {
        const auto s = to_cuda(stream);
        return s != nullptr ? s : m_step_stream;
    }

    // ---- set-up ----

    void initialize(const std::vector<uint8_t>& kill_mask) {
        m_geom       = fdmt_fft::make_geometry(*m_plan, m_mode, m_frac,
                                               m_use_box_smearing);
        m_ndms       = m_plan->get_dmt_ndms();
        m_nsamps_out = m_plan->get_dmt_nsamps();
        if (m_mode == FDMTMode::kValid && m_nsamps_out != m_nsamps) {
            throw std::logic_error("FDMTFFT: valid mode expects nsamps output");
        }
        if (m_geom.skip + m_nsamps_out > m_geom.n_fft) {
            throw std::logic_error("FDMTFFT: output exceeds the FFT length");
        }
        m_dag = fdmt_fft::make_tree_dag(*m_plan, m_frac, m_use_box_smearing);
        if (m_dag.ncoords.back() != m_ndms) {
            throw std::logic_error("FDMTFFT: root coordinates != DM trials");
        }
        m_max_coords = *std::ranges::max_element(m_dag.ncoords);
        m_l0_chan.upload(m_dag.l0_chan);
        m_l0_shift.upload(m_dag.l0_shift);
        if (!kill_mask.empty()) {
            m_keep.upload(kill_mask);
            m_has_kill = true;
        }
        m_hist_len = m_mode == FDMTMode::kValid ? m_geom.overlap : 0;
        if (m_hist_len > 0) {
            for (auto& h : m_hist) {
                h.reserve(history_state_size());
                gpu_utils::check_gpu_call(
                    cudaMemset(h.data(), 0, history_state_size() * sizeof(float)),
                    "FDMTFFT (cuda): history");
            }
        }

        const auto& props = gpu_utils::device_properties(m_device_id);
        m_smem_optin      = static_cast<int>(props.sharedMemPerBlockOptin);
        const auto budget = std::min<SizeType>(
            kNodeBudget, static_cast<SizeType>(m_smem_optin) /
                             (2 * kTileBins * sizeof(float2)));
        const auto stages = fdmt_fft::plan_tree_stages(m_dag, budget, kMaxOut);
        m_mid_rows        = 0;
        for (SizeType i = 0; i + 1 < stages.size(); ++i) {
            m_mid_rows =
                std::max(m_mid_rows, m_dag.ncoords[stages[i].level_out]);
        }
        m_stages.build(stages);
        for (const auto kernel :
             {kernel_tree_stage<true>, kernel_tree_stage<false>}) {
            gpu_utils::check_gpu_call(
                cudaFuncSetAttribute(kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     m_smem_optin),
                "FDMTFFT (cuda): shared-memory opt-in");
        }

        m_seg = fdmt_fft::choose_segments(
            m_geom, m_mode, m_nsamps_out, m_dag.ncoords[0] + m_dag.nops(),
            m_nchans, m_ndms, segment_cap());
        m_geo = make_geo(m_seg.n_fft);
        const auto rows_win = m_nbeams * std::max(m_nchans, m_ndms);
        m_win.reserve(rows_win * m_seg.n_fft);
        if (!m_geo->half) {
            m_spec.reserve(m_nbeams * m_nchans * m_geo->nb);
        }
        m_out_spec.reserve(m_nbeams * m_ndms * m_geo->nb);
        if (stages.size() > 1) {
            m_mid[0].reserve(m_nbeams * m_mid_rows * m_geo->nb);
        }
        if (stages.size() > 2) {
            m_mid[1].reserve(m_nbeams * m_mid_rows * m_geo->nb);
        }
        std::string split;
        for (const auto& st : stages) {
            split += std::format(" {}->{}({} prog, {} nodes)", st.level_in,
                                 st.level_out, st.nprograms(), st.max_nodes);
        }
        logging::debug("FDMTFFT cuda: transform length {} x {} segment(s) "
                       "(single {}), fractional {}, stages{}",
                       m_seg.n_fft, m_seg.nseg, m_geom.n_fft, m_frac, split);
    }

    // Segment length cap: the testing hook, and device memory (execute()
    // buffers per transform sample within half the free memory).
    [[nodiscard]] SizeType segment_cap() const {
        SizeType cap = detail::fft_segment_cap();
        size_t free  = 0;
        size_t total = 0;
        gpu_utils::check_gpu_call(cudaMemGetInfo(&free, &total),
                                  "FDMTFFT (cuda): cudaMemGetInfo");
        const auto nstage = m_stages.stages.size();
        const double per_sample =
            static_cast<double>(m_nbeams) *
            ((4.0 * static_cast<double>(std::max(m_nchans, m_ndms))) +
             (4.3 * static_cast<double>(m_nchans + m_ndms)) +
             (4.3 * static_cast<double>(m_mid_rows) *
              static_cast<double>(std::min<SizeType>(nstage - 1, 2))) +
             (4.0 * static_cast<double>(m_nchans + m_ndms))); // cuFFT work
        const auto fit =
            static_cast<SizeType>(0.5 * static_cast<double>(free) / per_sample);
        if (fit < m_geom.n_fft) {
            if (m_mode == FDMTMode::kRoll) {
                throw std::runtime_error(std::format(
                    "FDMTFFT (cuda): a roll-mode transform of {} samples needs "
                    "more device memory than is free",
                    m_geom.n_fft));
            }
            logging::debug("FDMTFFT (cuda): device memory caps the transform "
                          "at {} samples (single transform {})",
                          fit, m_geom.n_fft);
            cap = cap == 0 ? fit : std::min(cap, fit);
        }
        return cap;
    }

    [[nodiscard]] std::unique_ptr<utils::CUFFTManager>
    make_r2c(SizeType n, SizeType nb) const {
        return std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kR2C, n, m_nbeams * m_nchans, m_device_id, n, nb);
    }

    [[nodiscard]] std::unique_ptr<Geo> make_geo(SizeType n) const {
        auto g    = std::make_unique<Geo>();
        g->n_fft  = n;
        g->n_bins = (n / 2) + 1;
        g->nb     = ((g->n_bins + kTileBins - 1) / kTileBins) * kTileBins;
        if (n % 2 == 0) {
            g->half = std::make_unique<utils::CUFFTManager>(
                utils::FFTKind::kC2CForward, n / 2, m_nbeams * m_nchans,
                m_device_id, 0, n / 2);
            g->s1 = phase_fixed(1.0, n);
        } else {
            g->r2c = make_r2c(n, g->nb);
        }
        g->c2r = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kC2R, n, m_nbeams * m_ndms, m_device_id, n, g->nb);
        std::vector<unsigned long long> sfx(m_dag.shift.size());
        for (SizeType i = 0; i < sfx.size(); ++i) {
            sfx[i] = phase_fixed(m_dag.shift[i], n);
        }
        g->sfx.upload(sfx);
        std::vector<unsigned long long> s0(m_dag.dt0_max + 1);
        for (SizeType s = 0; s <= m_dag.dt0_max; ++s) {
            s0[s] = phase_fixed(static_cast<double>(s), n);
        }
        DevBuf<unsigned long long> s0_d;
        s0_d.upload(s0);
        g->w0.reserve((m_dag.dt0_max + 1) * g->nb);
        kernel_level0_windows<<<blocks_for(static_cast<int64_t>(g->nb)),
                                kBlock>>>(
            s0_d.data(), g->w0.data(), static_cast<int>(m_dag.dt0_max),
            static_cast<int>(g->nb), m_use_box_smearing ? 1 : 0,
            1.0F / static_cast<float>(n));
        gpu_utils::check_gpu_call(cudaDeviceSynchronize(),
                                  "FDMTFFT (cuda): level-0 windows");
        return g;
    }

    // ---- checks ----

    void check_sizes(SizeType in, SizeType out, std::string_view what) const {
        const auto total_in = m_nbeams * m_nchans * m_nsamps;
        if (in != total_in) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: expected waterfall size {}, got {}",
                            what, total_in, in));
        }
        check_out(out, what);
    }
    void check_out(SizeType out, std::string_view what) const {
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (out < total_out) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: dmt buffer size {} must be >= {}",
                            what, out, total_out));
        }
    }
    static void check_nbits(SizeType nbits, std::string_view what) {
        if (!valid_nbits(nbits)) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::{}: nbits must be 1, 2, 4, 8 or 16, got {}", what,
                nbits));
        }
    }
    void check_history(SizeType n, std::string_view what) const {
        if (n != history_state_size()) {
            throw std::invalid_argument(
                std::format("FDMTFFT::{}: expected {} floats, got {}", what,
                            history_state_size(), n));
        }
    }
    [[nodiscard]] SizeType cm_bytes(SizeType nbits) const {
        return m_nbeams * m_nchans *
               bit_pack_utils::packed_row_bytes(m_nsamps, nbits);
    }
    [[nodiscard]] SizeType tm_bytes(SizeType nbits) const {
        return m_nbeams * m_nsamps *
               bit_pack_utils::packed_row_bytes(m_nchans, nbits);
    }
    void require_stepper() const {
        if (!m_initialized) {
            throw std::logic_error(
                "FDMTFFT: Stepper is not initialized. Call reset() first.");
        }
    }

    // ---- host-memory staging ----

    // Host calls run on the stager's stream, after earlier device calls.
    cudaStream_t host_begin() {
        const auto s = m_host.stream();
        m_device_work.order(s);
        return s;
    }
    float* out_stage() {
        m_out_d.reserve(m_nbeams * m_ndms * m_nsamps_out);
        return m_out_d.data();
    }
    void host_finish(std::span<float> dmt) {
        gpu_utils::check_last_gpu_error("FDMTFFT (cuda): execute");
        const gpu_host::ChunkedStager::Segment seg{
            dmt.data(), m_out_d.data(),
            m_nbeams * m_ndms * m_nsamps_out * sizeof(float)};
        m_host.to_host(std::span(&seg, 1));
    }

    [[nodiscard]] Input packed_input(const void* data, SizeType nbits) const {
        return {.data    = data,
                .row_len = bit_pack_utils::packed_row_bytes(m_nsamps, nbits),
                .nbits   = nbits};
    }

    Input unpack_time_major(const uint8_t* packed,
                            SizeType nbits,
                            cudaStream_t s) {
        m_unpacked_d.reserve(m_nbeams * m_nchans * m_nsamps);
        fourier_gpu::unpack_time_major(packed, m_unpacked_d.data(), nbits,
                                       m_nbeams, m_nchans, m_nsamps, s);
        return {.data = m_unpacked_d.data(), .row_len = m_nsamps, .nbits = 32};
    }

    // ---- pipeline ----

    template <typename F> void dispatch_nbits(SizeType nbits, const F& f) const {
        switch (nbits) {
        case 1:
            f(std::integral_constant<int, 1>{});
            break;
        case 2:
            f(std::integral_constant<int, 2>{});
            break;
        case 4:
            f(std::integral_constant<int, 4>{});
            break;
        case 8:
            f(std::integral_constant<int, 8>{});
            break;
        case 16:
            f(std::integral_constant<int, 16>{});
            break;
        default:
            f(std::integral_constant<int, 32>{});
            break;
        }
    }

    // Window rows of one segment (p0: its first virtual-input sample).
    void fill(const Input& in,
              const Geo& g,
              float* win,
              SizeType p0,
              SizeType zeros,
              cudaStream_t s) const {
        const auto nrows = static_cast<int64_t>(m_nbeams * m_nchans);
        const dim3 grid(blocks_for(static_cast<int64_t>(g.n_fft)),
                        grid_y(nrows));
        const float* hist = m_hist_len > 0 ? m_hist[m_hist_cur].data() : nullptr;
        dispatch_nbits(in.nbits, [&](auto nb) {
            kernel_fill<decltype(nb)::value><<<grid, kBlock, 0, s>>>(
                in.data, static_cast<int64_t>(in.row_len), hist,
                m_has_kill ? m_keep.data() : nullptr, win, nrows,
                static_cast<int>(m_nchans), static_cast<int64_t>(m_nsamps),
                static_cast<int64_t>(g.n_fft), static_cast<int64_t>(p0),
                static_cast<int64_t>(zeros),
                static_cast<int64_t>(m_hist_len));
        });
    }

    void update_history(const Input& in, cudaStream_t s) {
        if (m_hist_len == 0) {
            return;
        }
        const auto nrows = static_cast<int64_t>(m_nbeams * m_nchans);
        const dim3 grid(blocks_for(static_cast<int64_t>(m_hist_len)),
                        grid_y(nrows));
        dispatch_nbits(in.nbits, [&](auto nb) {
            kernel_update_history<decltype(nb)::value><<<grid, kBlock, 0, s>>>(
                in.data, static_cast<int64_t>(in.row_len),
                m_hist[m_hist_cur].data(), m_hist[m_hist_cur ^ 1].data(),
                nrows, static_cast<int64_t>(m_nsamps),
                static_cast<int64_t>(m_hist_len));
        });
        m_hist_cur ^= 1;
    }

    void launch_stage(const StagesD& sd,
                      const StagesD::Stage& st,
                      const Geo& g,
                      const float2* in,
                      float2* out,
                      SizeType in_beam,
                      SizeType out_beam,
                      bool from_level0,
                      cudaStream_t s,
                      bool half = false) const {
        const auto smem = 2 * st.max_nodes * kTileBins * sizeof(float2);
        const auto ntiles = static_cast<int64_t>(g.nb / kTileBins);
        for (int64_t t0 = 0; t0 < ntiles; t0 += kMaxGridY) {
            const dim3 grid(static_cast<unsigned>(st.nprog),
                            static_cast<unsigned>(
                                std::min<int64_t>(ntiles - t0, kMaxGridY)),
                            static_cast<unsigned>(m_nbeams));
            const auto kernel = from_level0 ? kernel_tree_stage<true>
                                            : kernel_tree_stage<false>;
            kernel<<<grid, kStageThreads, smem, s>>>(
                sd.out_begin.data() + st.out_off, sd.lvl_off.data() + st.lvl_off,
                sd.ops.data() + st.ops_off, g.sfx.data(), in,
                static_cast<int64_t>(in_beam),
                m_l0_chan.data(), m_l0_shift.data(), g.w0.data(), out,
                static_cast<int64_t>(out_beam), static_cast<int64_t>(g.nb),
                static_cast<int>(st.nlev), static_cast<int>(st.max_nodes),
                static_cast<int>(t0),
                half ? static_cast<int64_t>(g.n_fft / 2) : int64_t{0}, g.s1);
        }
    }

    // out_spec rows (nbeams * ndms, dist nb; overwritten) -> dst[beam][dm]
    // samples [o0, o0 + cnt) from transform sample skip.
    void inverse(const Geo& g,
                 float2* out_spec,
                 float* time,
                 float* dst,
                 SizeType skip,
                 SizeType o0,
                 SizeType cnt,
                 cudaStream_t s) const {
        const auto rows = static_cast<int64_t>(m_nbeams * m_ndms);
        fourier_gpu::hermitian_edges(out_spec, static_cast<SizeType>(rows),
                                     g.nb, g.n_bins, g.n_fft, s);
        g.c2r->execute(
            cuda::std::span<float>(time, static_cast<SizeType>(rows) * g.n_fft),
            cuda::std::span<ComplexTypeGPU>(
                reinterpret_cast<ComplexTypeGPU*>(out_spec),
                static_cast<SizeType>(rows) * g.nb),
            s);
        const dim3 grid(blocks_for(static_cast<int64_t>(cnt)), grid_y(rows));
        kernel_trim<<<grid, kBlock, 0, s>>>(
            time, dst, rows, static_cast<int64_t>(g.n_fft),
            static_cast<int64_t>(m_nsamps_out), static_cast<int64_t>(o0),
            static_cast<int64_t>(cnt), static_cast<int64_t>(skip));
    }

    // Channel spectra of one segment: R2C into `spec`, or (half) the
    // in-place half-length C2C of the window rows in `win`.
    void forward(const Input& in,
                 const Geo& g,
                 float* win,
                 float2* spec,
                 SizeType p0,
                 SizeType zeros,
                 cudaStream_t s,
                 bool half = false) const {
        fill(in, g, win, p0, zeros, s);
        const auto rows = m_nbeams * m_nchans;
        if (half) {
            g.half->execute(cuda::std::span<ComplexTypeGPU>(
                                reinterpret_cast<ComplexTypeGPU*>(win),
                                rows * (g.n_fft / 2)),
                            s);
            return;
        }
        g.r2c->execute(
            cuda::std::span<float>(win, rows * g.n_fft),
            cuda::std::span<ComplexTypeGPU>(
                reinterpret_cast<ComplexTypeGPU*>(spec), rows * g.nb),
            s);
    }

    void run(const Input& in, float* dmt, cudaStream_t s) {
        const auto& g   = *m_geo;
        const auto& sts = m_stages.stages;
        const bool half = g.half != nullptr;
        for (SizeType seg = 0; seg < m_seg.nseg; ++seg) {
            const auto o0 = seg * m_seg.hop;
            forward(in, g, m_win.data(), m_spec.data(), o0, m_seg.zeros, s,
                    half);
            for (SizeType i = 0; i < sts.size(); ++i) {
                const bool first = i == 0;
                const bool last  = i + 1 == sts.size();
                const float2* src =
                    !first ? m_mid[(i - 1) % 2].data()
                    : half ? reinterpret_cast<const float2*>(m_win.data())
                           : m_spec.data();
                float2* dst = last ? m_out_spec.data() : m_mid[i % 2].data();
                const auto in_beam =
                    first ? m_nchans * (half ? g.n_fft / 2 : g.nb)
                          : m_mid_rows * g.nb;
                const auto out_beam = (last ? m_ndms : m_mid_rows) * g.nb;
                launch_stage(m_stages, sts[i], g, src, dst, in_beam, out_beam,
                             first, s, first && half);
            }
            inverse(g, m_out_spec.data(), m_win.data(), dmt, m_seg.skip, o0,
                    std::min(m_seg.hop, m_nsamps_out - o0), s);
        }
        update_history(in, s);
        gpu_utils::check_last_gpu_error("FDMTFFT (cuda): execute");
    }

    // ---- stepper ----

    void ensure_stepper() {
        if (m_step_geo != nullptr) {
            return;
        }
        if (m_geo->n_fft == m_geom.n_fft) {
            m_step_geo = m_geo.get();
        } else {
            m_step_geo_own = make_geo(m_geom.n_fft);
            m_step_geo     = m_step_geo_own.get();
        }
        if (!m_step_geo->r2c) {
            m_step_geo->r2c = make_r2c(m_step_geo->n_fft, m_step_geo->nb);
        }
        const auto& g = *m_step_geo;
        m_step_stages.build(fdmt_fft::level_tree_stages(m_dag, kMaxOut));
        m_step_win.reserve(m_nbeams * m_nchans * g.n_fft);
        m_step_spec.reserve(m_nbeams * m_nchans * g.nb);
        for (auto& st : m_state) {
            st.reserve(m_nbeams * m_max_coords * g.nb);
        }
        m_step_time_d.reserve(m_nbeams * m_ndms * g.n_fft);
    }

    void start_stepper(const Input& in, float* dmt, cudaStream_t s) {
        ensure_stepper();
        const auto& g = *m_step_geo;
        forward(in, g, m_step_win.data(), m_step_spec.data(), 0, 0, s);
        const dim3 grid(blocks_for(static_cast<int64_t>(g.nb)),
                        grid_y(static_cast<int64_t>(m_dag.ncoords[0])),
                        static_cast<unsigned>(m_nbeams));
        kernel_level0_state<<<grid, kBlock, 0, s>>>(
            m_step_spec.data(), m_l0_chan.data(), m_l0_shift.data(),
            g.w0.data(), m_state[0].data(),
            static_cast<int64_t>(m_dag.ncoords[0]), static_cast<int64_t>(g.nb),
            static_cast<int64_t>(m_nchans * g.nb),
            static_cast<int64_t>(m_max_coords * g.nb));
        gpu_utils::check_last_gpu_error("FDMTFFT (cuda): reset");
        m_step_in     = m_state[0].data();
        m_step_out    = m_state[1].data();
        m_step_dmt    = dmt;
        m_step_src    = in;
        m_step_stream = s;
        m_level       = 0;
        m_initialized = true;
        m_view_valid  = false;
    }

    [[nodiscard]] SizeType view_nsamps() const {
        if (m_mode == FDMTMode::kFull) {
            return m_plan->get_container().state_shape[m_level].nsamps;
        }
        return m_nsamps;
    }

    // Beam 0 of the current level in the time domain, in batches of
    // kViewBatch rows through a small C2R plan.
    void materialize_view() const {
        if (m_view_valid) {
            return;
        }
        gpu_utils::set_device(m_device_id);
        const auto& g   = *m_step_geo;
        const auto s    = m_step_stream;
        const auto rows = m_plan->get_container().state_shape[m_level].ncoords;
        const auto nsv  = view_nsamps();
        if (!m_view_c2r) {
            m_view_c2r = std::make_unique<utils::CUFFTManager>(
                utils::FFTKind::kC2R, g.n_fft, kViewBatch, m_device_id,
                g.n_fft, g.nb);
            m_view_spec.reserve(kViewBatch * g.nb);
            m_view_time.reserve(kViewBatch * g.n_fft);
        }
        m_view_d.reserve(m_max_coords * std::max(nsv, m_nsamps_out));
        const auto skip = m_geom.skip;
        const auto cnt  = nsv;
        for (SizeType r0 = 0; r0 < rows; r0 += kViewBatch) {
            const auto n = std::min(kViewBatch, rows - r0);
            const dim3 grid(blocks_for(static_cast<int64_t>(g.nb)),
                            grid_y(static_cast<int64_t>(n)));
            kernel_copy_rows<<<grid, kBlock, 0, s>>>(
                m_step_in + (r0 * g.nb), m_view_spec.data(),
                static_cast<int64_t>(n), static_cast<int64_t>(g.nb));
            fourier_gpu::hermitian_edges(m_view_spec.data(), n, g.nb,
                                         g.n_bins, g.n_fft, s);
            m_view_c2r->execute(
                cuda::std::span<float>(m_view_time.data(), kViewBatch * g.n_fft),
                cuda::std::span<ComplexTypeGPU>(
                    reinterpret_cast<ComplexTypeGPU*>(m_view_spec.data()),
                    kViewBatch * g.nb),
                s);
            const dim3 tgrid(blocks_for(static_cast<int64_t>(cnt)),
                             grid_y(static_cast<int64_t>(n)));
            kernel_trim<<<tgrid, kBlock, 0, s>>>(
                m_view_time.data(), m_view_d.data() + (r0 * nsv),
                static_cast<int64_t>(n), static_cast<int64_t>(g.n_fft),
                static_cast<int64_t>(nsv), 0, static_cast<int64_t>(cnt),
                static_cast<int64_t>(skip));
        }
        gpu_utils::check_gpu_call(cudaStreamSynchronize(s),
                                  "FDMTFFT (cuda): view");
        m_view_valid   = true;
        m_view_h_valid = false;
    }

    [[nodiscard]] const float* host_view() const {
        require_stepper();
        materialize_view();
        if (!m_view_h_valid) {
            const auto n =
                m_plan->get_container().state_shape[m_level].ncoords *
                view_nsamps();
            m_view_h.resize(n);
            gpu_utils::check_gpu_call(
                cudaMemcpy(m_view_h.data(), m_view_d.data(), n * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "FDMTFFT (cuda): host view");
            m_view_h_valid = true;
        }
        return m_view_h.data();
    }
};

} // namespace

std::unique_ptr<detail::FDMTFFTEngine>
detail::make_fdmt_fft_gpu(const plans::FDMTPlan& plan,
                          const detail::FDMTFFTEngineConfig& cfg) {
    return std::make_unique<FDMTFFTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
