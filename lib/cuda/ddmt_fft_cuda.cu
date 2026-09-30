#include "dmt/algorithms/ddmt_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_fft_common.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/fourier_gpu.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/logging.hpp"
#include "dmt/nufft.hpp"

// DDMT-FFT on CUDA/HIP. Same streaming model, delay model, segmentation and
// NUFFT parameters as the CPU engine (ddmt_fft_common.hpp, nufft.hpp),
// including the piecewise-uniform split of the DM grid: one NUFFT per
// uniform run, brute force for the other trials. Per segment and beam:
//
//   1. kernel_fill_rows: combined (history ++ block) samples of every active
//      channel for the segment; cuFFT R2C          -> spec [nact][n_bins]
//   2a. brute force (trials outside the NUFFT runs), register blocked: a
//       thread holds kBJ bins (stride 32) x kBD trials, the spectra of a
//       channel chunk are staged in shared memory, and each (trial, channel)
//       phasor is set up once per thread (exact 0.64 fixed-point phase,
//       fourier_gpu.cuh) and rotated along its bins by a shared step:
//       ~8 FMAs per term, one sincos per kBJ terms
//   2b. NUFFT, per run: one block per bin spreads the channels onto its fine
//       grid in shared memory (fixed-point positions, the CPU's kernel
//       polynomials, channels interleaved across the band so a warp's
//       atomics rarely collide), a batched cuFFT C2C, then a tiled
//       (coalesced) deconvolution into the run's DM rows
//   3. cuFFT C2R of the DM rows (Hermitian edges zeroed), trimmed into the
//      output
//
// A uniform run whose fine grid would not fit shared memory is transformed
// as a few equal sub-runs (still NUFFT; method_used() reports the grid's
// runs, as on the CPU). The transform length follows the CPU engine's rule,
// capped further only by the device's free memory. Host-memory calls stage
// through device buffers and block; device-memory calls run asynchronously
// on the given stream, ordered after the engine's previous work.

namespace dmt::algorithms {

namespace {

using fourier_gpu::blocks_for;
using fourier_gpu::cfma;
using fourier_gpu::cmul;
using fourier_gpu::DevBuf;
using fourier_gpu::fixed_turn;
using fourier_gpu::grid_y;
using fourier_gpu::kBlock;
using fourier_gpu::phase_fixed;
using fourier_gpu::phasor;
using fourier_gpu::phasor_fast;

// Brute force: bins per thread (stride 32), trials per thread, warps per
// block, channels per shared-memory chunk.
constexpr int kBJ          = 8;
constexpr int kBD          = 4;
constexpr int kBWarps      = 16;
constexpr int kBChan       = 16;
constexpr int kBBins       = 32 * kBJ;      // bins per block
constexpr int kBTrials     = kBWarps * kBD; // trials per block
constexpr int kBThreads    = 32 * kBWarps;
constexpr int kSpreadThr   = 512;
constexpr int kSpreadChunk = 1024; // channels staged per spread pass
// Largest NUFFT sub-run: its fine grid (~2.2 x trials complex floats) must
// fit the shared memory of one block.
constexpr SizeType kMaxSubRun = 4096;
constexpr SizeType kMaxWidth  = 16; // NUFFT kernel width bound (nufft_cpu)
constexpr SizeType kMaxDegree = kMaxWidth + 2;
constexpr int kTile           = 32; // deconvolution transpose tile
constexpr int kTileRows       = 8;
// The CPU engine's channel-spectra cap (ddmt_fft_cpu.cpp): same segments.
constexpr SizeType kMaxSpectraBytes = SizeType{1} << 29;

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

// row (beam b, active channel a) of the segment: combined positions
// [p0, p0 + n_fft) where combined = hist[0, hist_len) ++ block[0, n_new).
__global__ void kernel_fill_rows(const float* __restrict__ hist,
                                 const float* __restrict__ block,
                                 const unsigned* __restrict__ active,
                                 float* __restrict__ rows,
                                 int64_t beam,
                                 int64_t nchans,
                                 int64_t nact,
                                 int64_t ctx,
                                 int64_t hist_len,
                                 int64_t n_new,
                                 int64_t p0,
                                 int64_t n_fft) {
    const auto t =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (t >= n_fft) {
        return;
    }
    for (int64_t a = blockIdx.y; a < nact; a += gridDim.y) {
        const auto c = (beam * nchans) + active[a];
        const auto p = p0 + t;
        float v      = 0.0F;
        if (p < hist_len) {
            v = hist[(c * ctx) + p];
        } else if (p - hist_len < n_new) {
            v = block[(c * n_new) + (p - hist_len)];
        }
        rows[(a * n_fft) + t] = v;
    }
}

// Brute force over the listed trials: Y[rows[i]][k] = (1/N) sum_a X[a][k]
// exp(2 pi i tau(i, a) k / N), with s[i][a] = tau / N mod 1 in 0.64 fixed
// point. Block: kBTrials trials x kBBins bins (tile blockIdx.y + tile0);
// warp w owns trials [w kBD, (w + 1) kBD) of the block, lane l bins l + 32 j.
__global__ void __launch_bounds__(kBThreads)
    kernel_brute(const float2* __restrict__ spec,
                 const unsigned long long* __restrict__ sfx,
                 const unsigned* __restrict__ rows,
                 float2* __restrict__ out,
                 int nact,
                 int ntrials,
                 int64_t n_bins,
                 float inv_n,
                 int tile0) {
    __shared__ float2 xs[kBChan][kBBins];
    __shared__ unsigned long long ss[kBTrials][kBChan];
    __shared__ float2 st[kBTrials][kBChan];
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const auto k0  = (static_cast<int64_t>(blockIdx.y) + tile0) * kBBins;
    const int tr0  = static_cast<int>(blockIdx.x) * kBTrials;
    const auto kl  = static_cast<unsigned long long>(k0 + lane);
    float2 acc[kBD][kBJ];
#pragma unroll
    for (int d = 0; d < kBD; ++d) {
#pragma unroll
        for (int j = 0; j < kBJ; ++j) {
            acc[d][j] = {0.0F, 0.0F};
        }
    }
    for (int a0 = 0; a0 < nact; a0 += kBChan) {
        const int na = min(kBChan, nact - a0);
        __syncthreads();
        for (int i = tid; i < kBChan * kBBins; i += kBThreads) {
            const int c  = i / kBBins;
            const int b  = i % kBBins;
            const auto k = k0 + b;
            xs[c][b] = (c < na && k < n_bins)
                           ? spec[((static_cast<int64_t>(a0) + c) * n_bins) + k]
                           : float2{0.0F, 0.0F};
        }
        for (int i = tid; i < kBTrials * kBChan; i += kBThreads) {
            const int t  = i / kBChan;
            const int c  = i % kBChan;
            const int tr = tr0 + t;
            const auto sv =
                (tr < ntrials && c < na)
                    ? sfx[(static_cast<int64_t>(tr) * nact) + a0 + c]
                    : 0ULL;
            ss[t][c] = sv;
            st[t][c] = phasor_fast(sv, 32ULL); // step to bin + 32
        }
        __syncthreads();
        for (int c = 0; c < na; ++c) {
            float2 x[kBJ];
#pragma unroll
            for (int j = 0; j < kBJ; ++j) {
                x[j] = xs[c][lane + (32 * j)];
            }
#pragma unroll
            for (int d = 0; d < kBD; ++d) {
                const int t    = (warp * kBD) + d;
                float2 p       = phasor_fast(ss[t][c], kl);
                const float2 w = st[t][c];
#pragma unroll
                for (int j = 0; j < kBJ; ++j) {
                    acc[d][j] = cfma(acc[d][j], x[j], p);
                    if (j + 1 < kBJ) {
                        p = cmul(p, w);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int d = 0; d < kBD; ++d) {
        const int tr = tr0 + (warp * kBD) + d;
        if (tr >= ntrials) {
            continue;
        }
        float2* o = out + (static_cast<int64_t>(rows[tr]) * n_bins);
#pragma unroll
        for (int j = 0; j < kBJ; ++j) {
            const auto k = k0 + lane + (32 * j);
            if (k < n_bins) {
                o[k] = {acc[d][j].x * inv_n, acc[d][j].y * inv_n};
            }
        }
    }
}

// NUFFT spreading of one run: block k spreads the active channels of bin k
// onto its fine grid (shared memory, nf cells) and writes it to grids[k].
// Point a sits at x = k xr[a] (turns; 0.64 fixed point) with strength
// X[a][k] exp(2 pi i k sr[a]); the kernel on its w cells comes from the
// plan's polynomials in the offset u (nufft::locate()).
//
// The grid accumulates in 32-bit fixed point: shared-memory float atomics
// are compare-and-swap loops before sm_90, integer adds are native. The
// scale 2^30 / sum_a |X[a][k]|_1 keeps every partial sum in range (kernel
// values are <= 1) and the rounding at ~1e-8 of sum |a|, well below the
// NUFFT's tolerance (relative to the same sum).
template <int W>
__global__ void __launch_bounds__(kSpreadThr)
    kernel_nufft_spread(const float2* __restrict__ spec,
                        const unsigned long long* __restrict__ xr,
                        const unsigned long long* __restrict__ sr,
                        const float* __restrict__ coef,
                        float2* __restrict__ grids,
                        int nact,
                        int64_t n_bins,
                        int nf) {
    constexpr int kNp = W + 3; // polynomial coefficients per cell
    extern __shared__ int2 s_grid[];
    __shared__ float s_coef[W * kNp];
    __shared__ float s_part[kSpreadThr];
    // Channel chunk staged with coalesced loads: position, phase, value.
    __shared__ unsigned long long s_x[kSpreadChunk];
    __shared__ unsigned long long s_s[kSpreadChunk];
    __shared__ float2 s_v[kSpreadChunk];
    const auto k   = static_cast<int64_t>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int nthr = static_cast<int>(blockDim.x);
    for (int j = tid; j < nf; j += nthr) {
        s_grid[j] = {0, 0};
    }
    for (int j = tid; j < W * kNp; j += nthr) {
        s_coef[j] = coef[j];
    }
    // Fixed-point scale from sum_a (|Re X| + |Im X|).
    float part = 0.0F;
    for (int a = tid; a < nact; a += nthr) {
        const float2 xv = spec[(static_cast<int64_t>(a) * n_bins) + k];
        part += fabsf(xv.x) + fabsf(xv.y);
    }
    // Block tree reduction in shared memory (portable: no warp shuffles).
    s_part[tid] = part;
    __syncthreads();
    for (int o = kSpreadThr / 2; o > 0; o >>= 1) {
        if (tid < o) {
            s_part[tid] += s_part[tid + o];
        }
        __syncthreads();
    }
    const float total = s_part[0];
    const float scale = total > 0.0F ? 1073741824.0F / total : 0.0F;
    const auto ku     = static_cast<unsigned long long>(k);
    for (int c0 = 0; c0 < nact; c0 += kSpreadChunk) {
        const int nc = min(kSpreadChunk, nact - c0);
        __syncthreads();
        for (int i = tid; i < nc; i += nthr) {
            s_x[i] = xr[c0 + i];
            s_s[i] = sr[c0 + i];
            s_v[i] = spec[(static_cast<int64_t>(c0 + i) * n_bins) + k];
        }
        __syncthreads();
        // Point j -> staged channel: lanes spread across the chunk (stride
        // per), so the lanes of a warp land on different cells.
        const int per = (nc + 31) / 32;
        for (int j = tid; j < 32 * per; j += nthr) {
            const int a = ((j % 32) * per) + (j / 32);
            if (a >= nc) {
                continue;
            }
            // g = frac(x) * nf = cell + fr.
            const unsigned long long x = s_x[a] * ku;
            const auto cell            = static_cast<int>(
                __umul64hi(x, static_cast<unsigned long long>(nf)));
            const unsigned long long lo =
                x * static_cast<unsigned long long>(nf);
            const float fr =
                static_cast<float>(lo >> 40U) * 5.9604644775390625e-08F;
            int lc    = 0;
            float off = 0.0F; // lc - g + w/2, in [0, 1]
            if constexpr (W % 2 == 0) {
                const int inc = fr > 0.0F ? 1 : 0;
                lc            = cell - (W / 2) + inc;
                off           = static_cast<float>(inc) - fr;
            } else {
                const int inc = fr > 0.5F ? 1 : 0;
                lc            = cell - (W / 2) + inc;
                off           = static_cast<float>(inc) - fr + 0.5F;
            }
            const float u = (2.0F * off) - 1.0F;
            float sn      = 0.0F;
            float cs      = 0.0F;
            sincospif(2.0F * fixed_turn(s_s[a], ku), &sn, &cs);
            const float2 amp = cmul(s_v[a], float2{cs * scale, sn * scale});
#pragma unroll
            for (int i = 0; i < W; ++i) {
                const float* c = s_coef + (i * kNp);
                float v        = c[kNp - 1];
#pragma unroll
                for (int q = kNp - 2; q >= 0; --q) {
                    v = fmaf(v, u, c[q]);
                }
                int cc = lc + i;
                cc     = cc < 0 ? cc + nf : (cc >= nf ? cc - nf : cc);
                atomicAdd(&s_grid[cc].x, __float2int_rn(amp.x * v));
                atomicAdd(&s_grid[cc].y, __float2int_rn(amp.y * v));
            }
        }
    }
    __syncthreads();
    const float inv = scale > 0.0F ? 1.0F / scale : 0.0F;
    float2* dst     = grids + (k * nf);
    for (int j = tid; j < nf; j += nthr) {
        const int2 v = s_grid[j];
        dst[j] = {static_cast<float>(v.x) * inv, static_cast<float>(v.y) * inv};
    }
}

// Y[row0 + d][k] = G_k((d - half) mod nf) / Psi(d - half) / N, through a
// shared-memory tile so both the grid reads (along d) and the DM-row writes
// (along k) are coalesced.
__global__ void kernel_nufft_deconv(const float2* __restrict__ grids,
                                    const float* __restrict__ inv_psi,
                                    float2* __restrict__ out,
                                    int count,
                                    int64_t row0,
                                    int64_t n_bins,
                                    int nf,
                                    int half,
                                    float inv_n) {
    __shared__ float2 tile[kTile][kTile + 1];
    const auto k0 = static_cast<int64_t>(blockIdx.x) * kTile;
    const int nd  = (count + kTile - 1) / kTile;
    for (int db = static_cast<int>(blockIdx.y); db < nd;
         db += static_cast<int>(gridDim.y)) {
        const int d0 = db * kTile;
        __syncthreads();
        for (auto ky = static_cast<int>(threadIdx.y); ky < kTile;
             ky += kTileRows) {
            const auto k = k0 + ky;
            const int d  = d0 + static_cast<int>(threadIdx.x);
            if (k < n_bins && d < count) {
                int m                 = d - half;
                m                     = m < 0 ? m + nf : m;
                tile[ky][threadIdx.x] = grids[(k * nf) + m];
            }
        }
        __syncthreads();
        for (auto dy = static_cast<int>(threadIdx.y); dy < kTile;
             dy += kTileRows) {
            const int d  = d0 + dy;
            const auto k = k0 + static_cast<int64_t>(threadIdx.x);
            if (k < n_bins && d < count) {
                const float s                  = inv_psi[d] * inv_n;
                const float2 v                 = tile[threadIdx.x][dy];
                out[((row0 + d) * n_bins) + k] = {v.x * s, v.y * s};
            }
        }
    }
}

__global__ void kernel_trim(const float* __restrict__ time,
                            float* __restrict__ out,
                            int64_t ndm,
                            int64_t n_fft,
                            int64_t guard,
                            int64_t n_out,
                            int64_t p0,
                            int64_t cnt) {
    const auto i =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= cnt) {
        return;
    }
    for (int64_t d = blockIdx.y; d < ndm; d += gridDim.y) {
        out[(d * n_out) + p0 + i] = time[(d * n_fft) + guard + i];
    }
}

// New history: combined positions [start, start + new_len) of every row.
__global__ void kernel_update_history(const float* __restrict__ hist,
                                      const float* __restrict__ block,
                                      float* __restrict__ hist_next,
                                      int64_t rows,
                                      int64_t ctx,
                                      int64_t hist_len,
                                      int64_t n_new,
                                      int64_t start,
                                      int64_t new_len) {
    const auto i =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= new_len) {
        return;
    }
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const auto p             = start + i;
        hist_next[(r * ctx) + i] = p < hist_len
                                       ? hist[(r * ctx) + p]
                                       : block[(r * n_new) + (p - hist_len)];
    }
}

// Channel-major packed rows -> float rows.
template <int NBITS>
__global__ void kernel_unpack(const uint8_t* __restrict__ packed,
                              float* __restrict__ out,
                              int64_t row_bytes,
                              int64_t nsamps,
                              int64_t rows) {
    const auto s =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (s >= nsamps) {
        return;
    }
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        out[(r * nsamps) + s] =
            static_cast<float>(bit_pack_utils::read_packed_sample<NBITS>(
                packed + (r * row_bytes), static_cast<SizeType>(s)));
    }
}

void unpack_channel_major(const uint8_t* packed,
                          float* out,
                          SizeType nbits,
                          SizeType row_bytes,
                          SizeType nsamps,
                          SizeType rows,
                          cudaStream_t st) {
    const dim3 grid(blocks_for(static_cast<int64_t>(nsamps)),
                    grid_y(static_cast<int64_t>(rows)));
    const auto launch = [&](auto kernel) {
        kernel<<<grid, kBlock, 0, st>>>(
            packed, out, static_cast<int64_t>(row_bytes),
            static_cast<int64_t>(nsamps), static_cast<int64_t>(rows));
    };
    switch (nbits) {
    case 1:
        launch(kernel_unpack<1>);
        break;
    case 2:
        launch(kernel_unpack<2>);
        break;
    case 4:
        launch(kernel_unpack<4>);
        break;
    case 8:
        launch(kernel_unpack<8>);
        break;
    default:
        launch(kernel_unpack<16>);
        break;
    }
}

// One NUFFT (sub-)run on the device.
struct GpuRun {
    ddmt_fft::UniformRun run;
    const nufft::Type1Plan* plan{nullptr};
    DevBuf<float> inv_psi;
    DevBuf<float> coef; // [w][w + 3] kernel monomials
};

// Per transform length: FFT plans and the length-dependent phase tables.
struct GpuGeometry {
    SizeType n_fft{};
    SizeType n_bins{};
    std::unique_ptr<utils::CUFFTManager> r2c; // nact rows
    std::unique_ptr<utils::CUFFTManager> c2r; // ndm rows
    // One batched C2C over the n_bins fine grids per distinct run grid.
    std::map<SizeType, std::unique_ptr<utils::CUFFTManager>> nu;
    std::vector<DevBuf<unsigned long long>> xr; // per run: [nact]
    std::vector<DevBuf<unsigned long long>> sr; // per run: [nact]
    DevBuf<unsigned long long> brute;           // [brute trial][nact]
    SizeType work_bytes{};
};

class DDMTFFTCudaEngine final : public detail::DDMTFFTEngine {
public:
    DDMTFFTCudaEngine(const plans::DDMTPlan& plan,
                      const detail::DDMTFFTEngineConfig& cfg)
        : m_plan(plan),
          m_device_id(cfg.exec.device),
          m_nbeams(cfg.nbeams),
          m_opts(cfg.options),
          m_model(plan, cfg.options.guard),
          m_nchans(plan.get_nchans()),
          m_ndm(m_model.dm.size()),
          m_nact(m_model.active.size()),
          m_ctx(m_model.context()) {
        gpu_utils::set_device(m_device_id);
        setup_methods();
        m_seg_n = segment_cap();
        for (auto& h : m_hist) {
            h.reserve(history_state_size());
            gpu_utils::check_gpu_call(
                cudaMemset(h.data(), 0, history_state_size() * sizeof(float)),
                "DDMTFFT (gpu): history");
        }
        m_hist_len = m_model.guard;
        std::vector<unsigned> act(m_model.active.begin(), m_model.active.end());
        m_active_d.upload(act);
        logging::debug("DDMTFFT gpu: {} ({} NUFFT run(s) on the device, {} "
                       "brute trials) ndm={} active chans={} max_tau={:.2f} "
                       "guard={} segment={}",
                       method_used(), m_runs.size(), m_brute_rows.size(), m_ndm,
                       m_nact, m_model.max_tau, m_model.guard, m_seg_n);
    }

    ~DDMTFFTCudaEngine() override {
        try {
            gpu_utils::set_device(m_device_id);
            m_device_work.wait();
        } catch (...) { // NOLINT(bugprone-empty-catch)
        }
    }
    DDMTFFTCudaEngine(const DDMTFFTCudaEngine&)            = delete;
    DDMTFFTCudaEngine& operator=(const DDMTFFTCudaEngine&) = delete;
    DDMTFFTCudaEngine(DDMTFFTCudaEngine&&)                 = delete;
    DDMTFFTCudaEngine& operator=(DDMTFFTCudaEngine&&)      = delete;

    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept override {
        const auto total = m_hist_len + input_nsamps;
        return total > m_ctx ? total - m_ctx : 0;
    }
    // The zeroing of the look-behind happens on the next call's stream.
    void reset_history() noexcept override {
        m_hist_len  = m_model.guard;
        m_hist_zero = true;
    }
    [[nodiscard]] SizeType history_state_size() const noexcept override {
        return m_nbeams * m_nchans * m_ctx;
    }
    [[nodiscard]] SizeType max_delay() const noexcept override {
        return m_model.max_delay;
    }
    // Stored only, as on the CPU: a gulp of a host call would change the
    // transform boundaries, and with them the (guard-level) result bits.
    void set_gulp_size(SizeType gulp_size) override { m_gulp = gulp_size; }
    [[nodiscard]] SizeType get_gulp_size() const noexcept override {
        return m_gulp == 0 ? 65536 : m_gulp;
    }
    [[nodiscard]] std::string_view method_used() const noexcept override {
        return detail::ddmt_fft_method_used(m_nruns_model, m_brute_rows.size());
    }

    // ---- history ----

    void save_history(std::span<float> out) const override {
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        gpu_utils::check_gpu_call(cudaMemcpy(out.data(), hist(),
                                             out.size() * sizeof(float),
                                             cudaMemcpyDeviceToHost),
                                  "DDMTFFT::save_history");
    }
    void load_history(std::span<const float> in) override {
        check_size(in.size(), history_state_size(), "load_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        gpu_utils::check_gpu_call(cudaMemcpy(hist(), in.data(),
                                             in.size() * sizeof(float),
                                             cudaMemcpyHostToDevice),
                                  "DDMTFFT::load_history");
        m_hist_len  = m_ctx;
        m_hist_zero = false;
    }
    // Device analogues run on the given stream after the engine's previous
    // work; later calls are ordered after them.
    void save_history(DeviceSpan<float> out, Stream stream) const override {
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        gpu_utils::set_device(m_device_id);
        const auto st = to_cuda(stream);
        m_device_work.order(st);
        gpu_utils::check_gpu_call(cudaMemcpyAsync(out.data(), hist(),
                                                  out.size() * sizeof(float),
                                                  cudaMemcpyDeviceToDevice, st),
                                  "DDMTFFT::save_history (device)");
        m_device_work.mark(st);
    }
    void load_history(DeviceSpan<const float> in, Stream stream) override {
        check_size(in.size(), history_state_size(), "load_history");
        gpu_utils::set_device(m_device_id);
        const auto st = to_cuda(stream);
        m_device_work.order(st);
        gpu_utils::check_gpu_call(cudaMemcpyAsync(hist(), in.data(),
                                                  in.size() * sizeof(float),
                                                  cudaMemcpyDeviceToDevice, st),
                                  "DDMTFFT::load_history (device)");
        m_device_work.mark(st);
        m_hist_len  = m_ctx;
        m_hist_zero = false;
    }

    // ---- execute: host memory ----

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        check_float("execute(float)");
        const auto nsamps = rows_nsamps(waterfall.size());
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        gpu_utils::set_device(m_device_id);
        const auto st = host_begin();
        m_in_d.reserve(waterfall.size());
        m_host.to_device(m_in_d.data(), waterfall.data(),
                         waterfall.size_bytes());
        finish_host(m_in_d.data(), nsamps, dmt, st);
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
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        host_packed(waterfall_packed, row_bytes, nsamps, false, dmt);
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
        check_size(dmt.size(), out_size(nsamps), "execute_time_major (output)");
        host_packed(filterbank_packed, samp_bytes, nsamps, true, dmt);
    }

    // ---- execute: device memory ----

    void execute(DeviceSpan<const float> waterfall,
                 DeviceSpan<float> dmt,
                 Stream stream) override {
        check_float("execute(device float)");
        const auto nsamps = rows_nsamps(waterfall.size());
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        gpu_utils::set_device(m_device_id);
        const auto st = to_cuda(stream);
        m_device_work.order(st);
        core(waterfall.data(), nsamps, dmt.data(), st);
        m_device_work.mark(st);
    }

    void execute(DeviceSpan<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 DeviceSpan<float> dmt,
                 Stream stream) override {
        check_packed("execute(device packed)");
        const auto nbits     = m_plan.get_nbits();
        const auto rows      = m_nbeams * m_nchans;
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        check_size(waterfall_packed.size(), rows * row_bytes,
                   "execute (packed input)");
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        gpu_utils::set_device(m_device_id);
        const auto st = to_cuda(stream);
        m_device_work.order(st);
        m_unpacked_d.reserve(rows * nsamps);
        unpack_channel_major(waterfall_packed.data(), m_unpacked_d.data(),
                             nbits, row_bytes, nsamps, rows, st);
        core(m_unpacked_d.data(), nsamps, dmt.data(), st);
        m_device_work.mark(st);
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    const plans::DDMTPlan& m_plan; // owned by the DDMTFFT facade
    int m_device_id;
    SizeType m_nbeams;
    DDMTFFTOptions m_opts;
    ddmt_fft::DelayModel m_model;
    SizeType m_nchans;
    SizeType m_ndm;
    SizeType m_nact;
    SizeType m_ctx;
    SizeType m_seg_n{};
    SizeType m_gulp{0};
    SizeType m_hist_len{0};
    bool m_hist_zero{false}; // history must be zeroed before next use
    mutable gpu_utils::DeviceWorkFence m_device_work;
    gpu_host::ChunkedStager m_host;

    // Methods: NUFFT sub-runs (plans per distinct length) and the trials
    // left to brute force (their output rows and exact delays).
    std::map<SizeType, std::unique_ptr<nufft::Type1Plan>> m_plans;
    std::vector<std::unique_ptr<GpuRun>> m_runs;
    SizeType m_nruns_model{0}; // the grid's runs that use the NUFFT
    std::vector<unsigned> m_brute_rows;
    SizeType m_nf_max{0};
    int m_width{0};
    DevBuf<unsigned> m_brute_rows_d;

    DevBuf<float> m_hist[2];
    int m_hist_cur{0};
    DevBuf<unsigned> m_active_d;
    DevBuf<float> m_rows_d; // fill rows, then C2R output
    DevBuf<float2> m_spec_d;
    DevBuf<float2> m_out_spec_d;
    DevBuf<float2> m_grids_d;
    DevBuf<float> m_unpacked_d;
    DevBuf<uint8_t> m_packed_d;
    DevBuf<float> m_in_d;
    DevBuf<float> m_out_d;
    std::map<SizeType, GpuGeometry> m_geos;

    [[nodiscard]] float* hist() const noexcept {
        return m_hist[m_hist_cur].data();
    }

    void setup_methods() {
        std::vector<bool> by_nufft(m_ndm, false);
        if (m_opts.method == DDMTFFTMethod::kNUFFT && m_nact > 0) {
            for (const auto& r : m_model.runs) {
                if (!m_model.nufft_run(r, /*explicit_nufft=*/true)) {
                    continue;
                }
                ++m_nruns_model;
                // Equal sub-runs whose fine grid fits shared memory.
                const auto nsub = (r.count + kMaxSubRun - 1) / kMaxSubRun;
                for (SizeType i = 0; i < nsub; ++i) {
                    const auto b0 = (r.count * i) / nsub;
                    const auto b1 = (r.count * (i + 1)) / nsub;
                    add_run({.begin = r.begin + b0,
                             .count = b1 - b0,
                             .dm0   = r.dm0 + (static_cast<double>(b0) * r.ddm),
                             .ddm   = r.ddm});
                }
                std::fill_n(by_nufft.begin() +
                                static_cast<std::ptrdiff_t>(r.begin),
                            r.count, true);
            }
            if (!m_runs.empty()) {
                const auto smem = m_nf_max * sizeof(float2);
                const auto set  = [&](auto kernel) {
                    gpu_utils::check_gpu_call(
                        cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            static_cast<int>(smem)),
                        "DDMTFFT (gpu): NUFFT shared memory");
                };
                dispatch_width(m_width, [&](auto w) {
                    set(kernel_nufft_spread<decltype(w)::value>);
                });
            }
        }
        for (SizeType d = 0; d < m_ndm; ++d) {
            if (!by_nufft[d]) {
                m_brute_rows.push_back(static_cast<unsigned>(d));
            }
        }
        if (!m_brute_rows.empty()) {
            m_brute_rows_d.upload(m_brute_rows);
        }
    }

    void add_run(const ddmt_fft::UniformRun& r) {
        auto& p = m_plans[r.count];
        if (!p) {
            p = std::make_unique<nufft::Type1Plan>(r.count, m_opts.tolerance);
        }
        int optin = 0;
        gpu_utils::check_gpu_call(
            cudaDeviceGetAttribute(
                &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, m_device_id),
            "DDMTFFT (gpu): shared memory limit");
        if (p->fine_grid() * sizeof(float2) > static_cast<SizeType>(optin)) {
            throw std::runtime_error(std::format(
                "DDMTFFT (gpu): a NUFFT grid of {} cells exceeds the device's "
                "{} B of shared memory per block",
                p->fine_grid(), optin));
        }
        if (static_cast<SizeType>(p->width()) > kMaxWidth ||
            p->kernel_degree() != p->width() + 2) {
            throw std::logic_error("DDMTFFT (gpu): unexpected NUFFT kernel");
        }
        auto run  = std::make_unique<GpuRun>();
        run->run  = r;
        run->plan = p.get();
        run->inv_psi.upload(p->deconvolution());
        run->coef.upload(p->kernel_monomials());
        m_runs.push_back(std::move(run));
        m_nf_max = std::max(m_nf_max, p->fine_grid());
        m_width  = p->width();
    }

    template <typename F> static void dispatch_width(int w, const F& f) {
        switch (w) {
        case 2:
            f(std::integral_constant<int, 2>{});
            break;
        case 3:
            f(std::integral_constant<int, 3>{});
            break;
        case 4:
            f(std::integral_constant<int, 4>{});
            break;
        case 5:
            f(std::integral_constant<int, 5>{});
            break;
        case 6:
            f(std::integral_constant<int, 6>{});
            break;
        case 7:
            f(std::integral_constant<int, 7>{});
            break;
        case 8:
            f(std::integral_constant<int, 8>{});
            break;
        default:
            throw std::logic_error(
                std::format("DDMTFFT (gpu): NUFFT width {} unsupported", w));
        }
    }

    // Longest transform: the CPU engine's (same segments, same result
    // bits), capped by the testing hook and by half the free device memory
    // for the per-segment buffers (bytes per transform sample below).
    [[nodiscard]] SizeType segment_cap() const {
        auto cap =
            ddmt_fft::max_segment_length(m_ctx, m_nact, kMaxSpectraBytes);
        if (const auto hook = detail::fft_segment_cap(); hook > 0) {
            cap = std::min(cap, hook);
        }
        std::size_t free_b  = 0;
        std::size_t total_b = 0;
        gpu_utils::check_gpu_call(cudaMemGetInfo(&free_b, &total_b),
                                  "DDMTFFT (gpu): memory info");
        // rows/time (max), spectra, DM spectra, NUFFT grids, cuFFT work.
        const SizeType per_sample =
            (4 * std::max(m_nact, m_ndm)) + (4 * m_nact) + (4 * m_ndm) +
            (4 * m_nf_max) + (4 * (m_nact + m_ndm + m_nf_max)) + 1;
        const auto fit = static_cast<SizeType>(free_b) / 2 / per_sample;
        if (fit < cap) {
            logging::debug("DDMTFFT (gpu): device memory caps the transform at "
                           "{} samples (engine rule {})",
                           fit, cap);
            cap = std::max<SizeType>(fit, utils::next_fft_size(2 * m_ctx + 2));
        }
        return cap;
    }

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
    void check_warm() const {
        if (m_hist_len != m_ctx) {
            throw std::logic_error(std::format(
                "DDMTFFT::save_history: stream is not fully warmed up yet "
                "({} of {} history samples/channel)",
                m_hist_len, m_ctx));
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
    [[nodiscard]] SizeType rows_nsamps(SizeType size) const {
        const auto rows = m_nbeams * m_nchans;
        if (size % rows != 0) {
            throw std::invalid_argument(std::format(
                "DDMTFFT::execute: waterfall size {} is not a multiple of "
                "nbeams * nchans = {}",
                size, rows));
        }
        return size / rows;
    }
    [[nodiscard]] SizeType out_size(SizeType nsamps) const noexcept {
        return m_nbeams * m_ndm * get_output_nsamps(nsamps);
    }

    GpuGeometry& geometry(SizeType n) {
        // A few transform lengths (warm-up, steady state, odd blocks); a
        // stream of ever-changing block sizes must not grow the plan cache.
        constexpr SizeType kMaxGeometries = 4;
        if (!m_geos.contains(n) && m_geos.size() >= kMaxGeometries) {
            m_geos.clear();
        }
        auto& g = m_geos[n];
        if (!g.r2c) {
            g.n_fft  = n;
            g.n_bins = (n / 2) + 1;
            g.r2c    = std::make_unique<utils::CUFFTManager>(
                utils::FFTKind::kR2C, n, m_nact, m_device_id);
            g.c2r = std::make_unique<utils::CUFFTManager>(
                utils::FFTKind::kC2R, n, m_ndm, m_device_id);
            std::vector<unsigned long long> xr(m_nact);
            std::vector<unsigned long long> sr(m_nact);
            for (const auto& run : m_runs) {
                const auto nf = run->plan->fine_grid();
                if (!g.nu.contains(nf)) {
                    g.nu[nf] = std::make_unique<utils::CUFFTManager>(
                        utils::FFTKind::kC2CBackward, nf, g.n_bins,
                        m_device_id);
                }
                // x = k ddm r / N and the centring + dm0 phase k shift r / N
                // (turns per bin), as the CPU's prepare_points().
                const double shift =
                    run->run.dm0 +
                    (static_cast<double>(run->plan->half()) * run->run.ddm);
                for (SizeType a = 0; a < m_nact; ++a) {
                    const double r = m_model.rate[m_model.active[a]];
                    xr[a]          = phase_fixed(-(run->run.ddm * r), n);
                    sr[a]          = phase_fixed(-(shift * r), n);
                }
                g.xr.emplace_back().upload(xr);
                g.sr.emplace_back().upload(sr);
            }
            if (!m_brute_rows.empty()) {
                // +tau / N per (brute trial, active channel), exact.
                const auto nb = m_brute_rows.size();
                std::vector<unsigned long long> s(nb * m_nact);
                for (SizeType i = 0; i < nb; ++i) {
                    for (SizeType a = 0; a < m_nact; ++a) {
                        s[(i * m_nact) + a] = phase_fixed(
                            -m_model.tau(m_brute_rows[i], m_model.active[a]),
                            n);
                    }
                }
                g.brute.upload(s);
            }
        }
        m_rows_d.reserve(std::max(m_nact, m_ndm) * n);
        m_spec_d.reserve(m_nact * g.n_bins);
        m_out_spec_d.reserve(m_ndm * g.n_bins);
        if (m_nf_max > 0) {
            m_grids_d.reserve(g.n_bins * m_nf_max);
        }
        return g;
    }

    // Host calls run on the stager's stream, after earlier device calls.
    cudaStream_t host_begin() {
        const auto st = m_host.stream();
        m_device_work.order(st);
        return st;
    }

    // Host packed input: upload the packed bytes, unpack on the device.
    void host_packed(std::span<const uint8_t> packed,
                     SizeType row_bytes,
                     SizeType nsamps,
                     bool time_major,
                     std::span<float> dmt) {
        gpu_utils::set_device(m_device_id);
        const auto st   = host_begin();
        const auto rows = m_nbeams * m_nchans;
        m_packed_d.reserve(packed.size());
        m_host.to_device(m_packed_d.data(), packed.data(), packed.size());
        m_unpacked_d.reserve(rows * nsamps);
        if (time_major) {
            fourier_gpu::unpack_time_major(
                m_packed_d.data(), m_unpacked_d.data(), m_plan.get_nbits(),
                m_nbeams, m_nchans, nsamps, st);
        } else {
            unpack_channel_major(m_packed_d.data(), m_unpacked_d.data(),
                                 m_plan.get_nbits(), row_bytes, nsamps, rows,
                                 st);
        }
        finish_host(m_unpacked_d.data(), nsamps, dmt, st);
    }

    // Host call on device input rows: dedisperse, copy back, wait.
    void finish_host(const float* rows_d,
                     SizeType n_new,
                     std::span<float> out,
                     cudaStream_t st) {
        const auto out_n = out_size(n_new);
        m_out_d.reserve(out_n);
        core(rows_d, n_new, m_out_d.data(), st);
        const gpu_host::ChunkedStager::Segment seg{out.data(), m_out_d.data(),
                                                   out_n * sizeof(float)};
        m_host.to_host(std::span(&seg, out_n > 0 ? 1 : 0));
    }

    void core(const float* data, SizeType n_new, float* out, cudaStream_t st) {
        if (m_hist_zero) {
            gpu_utils::check_gpu_call(
                cudaMemsetAsync(hist(), 0, history_state_size() * sizeof(float),
                                st),
                "DDMTFFT: reset history");
            m_hist_zero = false;
        }
        const auto total = m_hist_len + n_new;
        const auto n_out = total > m_ctx ? total - m_ctx : 0;
        const bool work  = m_ndm > 0 && m_nact > 0;
        // While the stream warms up, also prepare the steady-state
        // transform (n_new outputs per call) so the first warm call does not
        // pay for new FFT plans and first-touch of its buffers.
        if (work && m_hist_len < m_ctx && n_new > 0) {
            geometry(ddmt_fft::plan_segments(n_new, m_ctx, m_seg_n).n_fft);
        }
        if (n_out > 0 && work) {
            const auto seg = ddmt_fft::plan_segments(n_out, m_ctx, m_seg_n);
            auto& g        = geometry(seg.n_fft);
            for (SizeType b = 0; b < m_nbeams; ++b) {
                for (SizeType s = 0; s < seg.nseg; ++s) {
                    const auto p0  = s * seg.hop;
                    const auto cnt = std::min(seg.hop, n_out - p0);
                    run_segment(g, data, n_new, b, p0, cnt, n_out,
                                out + (b * m_ndm * n_out), st);
                }
            }
        } else if (n_out > 0) {
            gpu_utils::check_gpu_call(
                cudaMemsetAsync(out, 0,
                                m_nbeams * m_ndm * n_out * sizeof(float), st),
                "DDMTFFT: zero output");
        }
        update_history(data, n_new, st);
        gpu_utils::check_last_gpu_error("DDMTFFT (gpu) kernels");
    }

    void run_segment(GpuGeometry& g,
                     const float* data,
                     SizeType n_new,
                     SizeType beam,
                     SizeType p0,
                     SizeType cnt,
                     SizeType n_out,
                     float* out,
                     cudaStream_t st) {
        const auto nb = static_cast<int64_t>(g.n_bins);
        const dim3 fill_grid(blocks_for(static_cast<int64_t>(g.n_fft)),
                             grid_y(static_cast<int64_t>(m_nact)));
        kernel_fill_rows<<<fill_grid, kBlock, 0, st>>>(
            hist(), data, m_active_d.data(), m_rows_d.data(),
            static_cast<int64_t>(beam), static_cast<int64_t>(m_nchans),
            static_cast<int64_t>(m_nact), static_cast<int64_t>(m_ctx),
            static_cast<int64_t>(m_hist_len), static_cast<int64_t>(n_new),
            static_cast<int64_t>(p0), static_cast<int64_t>(g.n_fft));
        g.r2c->execute(
            cuda::std::span<float>(m_rows_d.data(), m_nact * g.n_fft),
            cuda::std::span<ComplexTypeGPU>(
                reinterpret_cast<ComplexTypeGPU*>(m_spec_d.data()),
                m_nact * g.n_bins),
            st);
        const float inv_n = 1.0F / static_cast<float>(g.n_fft);
        for (SizeType q = 0; q < m_runs.size(); ++q) {
            const auto& run = *m_runs[q];
            const auto& p   = *run.plan;
            const auto nf   = p.fine_grid();
            dispatch_width(p.width(), [&](auto w) {
                kernel_nufft_spread<decltype(w)::value>
                    <<<static_cast<unsigned>(g.n_bins), kSpreadThr,
                       nf * sizeof(float2), st>>>(
                        m_spec_d.data(), g.xr[q].data(), g.sr[q].data(),
                        run.coef.data(), m_grids_d.data(),
                        static_cast<int>(m_nact), nb, static_cast<int>(nf));
            });
            g.nu.at(nf)->execute(
                cuda::std::span<ComplexTypeGPU>(
                    reinterpret_cast<ComplexTypeGPU*>(m_grids_d.data()),
                    g.n_bins * nf),
                st);
            const dim3 dgrid(
                blocks_for(nb, kTile),
                grid_y((static_cast<int64_t>(run.run.count) + kTile - 1) /
                       kTile));
            const dim3 dblock(kTile, kTileRows);
            kernel_nufft_deconv<<<dgrid, dblock, 0, st>>>(
                m_grids_d.data(), run.inv_psi.data(), m_out_spec_d.data(),
                static_cast<int>(run.run.count),
                static_cast<int64_t>(run.run.begin), nb, static_cast<int>(nf),
                static_cast<int>(p.half()), inv_n);
        }
        if (!m_brute_rows.empty()) {
            const auto ntr    = static_cast<int64_t>(m_brute_rows.size());
            const auto ntiles = (nb + kBBins - 1) / kBBins;
            for (int64_t t0 = 0; t0 < ntiles; t0 += fourier_gpu::kMaxGridY) {
                const dim3 bgrid(
                    static_cast<unsigned>((ntr + kBTrials - 1) / kBTrials),
                    static_cast<unsigned>(std::min<int64_t>(
                        ntiles - t0, fourier_gpu::kMaxGridY)));
                gpu_utils::check_kernel_launch_params(bgrid, dim3(kBThreads));
                kernel_brute<<<bgrid, kBThreads, 0, st>>>(
                    m_spec_d.data(), g.brute.data(), m_brute_rows_d.data(),
                    m_out_spec_d.data(), static_cast<int>(m_nact),
                    static_cast<int>(ntr), nb, inv_n, static_cast<int>(t0));
            }
        }
        fourier_gpu::hermitian_edges(m_out_spec_d.data(), m_ndm, g.n_bins,
                                     g.n_bins, g.n_fft, st);
        g.c2r->execute(
            cuda::std::span<float>(m_rows_d.data(), m_ndm * g.n_fft),
            cuda::std::span<ComplexTypeGPU>(
                reinterpret_cast<ComplexTypeGPU*>(m_out_spec_d.data()),
                m_ndm * g.n_bins),
            st);
        const dim3 tgrid(blocks_for(static_cast<int64_t>(cnt)),
                         grid_y(static_cast<int64_t>(m_ndm)));
        kernel_trim<<<tgrid, kBlock, 0, st>>>(
            m_rows_d.data(), out, static_cast<int64_t>(m_ndm),
            static_cast<int64_t>(g.n_fft), static_cast<int64_t>(m_model.guard),
            static_cast<int64_t>(n_out), static_cast<int64_t>(p0),
            static_cast<int64_t>(cnt));
    }

    void update_history(const float* data, SizeType n_new, cudaStream_t st) {
        const auto total   = m_hist_len + n_new;
        const auto new_len = std::min(total, m_ctx);
        const auto rows    = m_nbeams * m_nchans;
        if (new_len > 0) {
            const dim3 grid(blocks_for(static_cast<int64_t>(new_len)),
                            grid_y(static_cast<int64_t>(rows)));
            kernel_update_history<<<grid, kBlock, 0, st>>>(
                hist(), data, m_hist[m_hist_cur ^ 1].data(),
                static_cast<int64_t>(rows), static_cast<int64_t>(m_ctx),
                static_cast<int64_t>(m_hist_len), static_cast<int64_t>(n_new),
                static_cast<int64_t>(total - new_len),
                static_cast<int64_t>(new_len));
        }
        m_hist_cur ^= 1;
        m_hist_len = new_len;
    }
};

} // namespace

std::unique_ptr<detail::DDMTFFTEngine>
detail::make_ddmt_fft_gpu(const plans::DDMTPlan& plan,
                          const detail::DDMTFFTEngineConfig& cfg) {
    return std::make_unique<DDMTFFTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
