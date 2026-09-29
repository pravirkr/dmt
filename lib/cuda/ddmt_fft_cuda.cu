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

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_fft_common.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/logging.hpp"
#include "dmt/nufft.hpp"

// DDMT-FFT on CUDA/HIP. Same streaming model, delay model, segmentation and
// NUFFT parameters as the CPU engine (ddmt_fft_common.hpp, nufft.hpp),
// including the piecewise-uniform split of the DM grid: one NUFFT per
// uniform run, brute force for the other trials. Per segment and beam:
//
//   1. kernel_fill_rows: combined (history ++ block) samples of every active
//      channel for the segment; cuFFT R2C          -> spec [nact][n_bins]
//   2a. brute force (trials outside the NUFFT runs): one thread per bin,
//       kDmPerThread trials in registers, the per-(trial, channel) delays
//       streamed through shared memory; every phase reduced exactly: tau =
//       n + f (n integer), turns = ((n k) mod N + f k) / N with an integer
//       Barrett reduction, then SFU sincos of the reduced angle
//   2b. NUFFT, per run: one block per bin spreads the channels onto its fine
//       grid in shared memory (atomics), a batched cuFFT C2C, then a tiled
//       (coalesced) deconvolution into the run's DM rows
//   3. cuFFT C2R of the DM rows, trimmed into the output
//
// A run whose fine grid does not fit the device's shared memory falls back
// to brute force (method_used() reports what runs). Every per-segment
// buffer is sized from the transform length, which is capped by the
// device's free memory. Host-memory calls stage through device buffers on
// the engine's stream and block (packed input is unpacked on the device);
// device-memory calls run asynchronously on the given stream.

namespace dmt::algorithms {

namespace {

constexpr int kThreads              = 256;
constexpr int kDmPerThread          = 8;
constexpr int kChanTile             = 64; // channels staged in shared memory
constexpr SizeType kMaxSpectraBytes = SizeType{1} << 30;
constexpr SizeType kMaxWorkBytes    = SizeType{2} << 30;
constexpr SizeType kMaxGridY        = 65535;
constexpr int kTile                 = 32; // deconvolution transpose tile
constexpr int kTileRows             = 8;

template <typename T> T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
template <typename T> const T* raw(const thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}
unsigned blocks_for(SizeType n, unsigned threads) {
    return static_cast<unsigned>(
        std::max<SizeType>((n + threads - 1) / threads, 1));
}
// Grid y extent for n rows: kernels loop over rows with stride gridDim.y.
unsigned grid_y(SizeType n) {
    return static_cast<unsigned>(std::clamp<SizeType>(n, 1, kMaxGridY));
}

// (n * k) mod N for n, k < N < 2^31, exact: Barrett reduction with
// magic = floor((2^64 - 1) / N), whose quotient is at most one short.
__device__ __forceinline__ unsigned
mod_mul(unsigned n, unsigned k, unsigned big_n, unsigned long long magic) {
    const unsigned long long nk =
        static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k);
    const unsigned long long q = __umul64hi(nk, magic);
    unsigned long long r       = nk - (q * big_n);
    if (r >= big_n) {
        r -= big_n;
    }
    return static_cast<unsigned>(r);
}

// row (beam b, active channel a) of the segment: combined positions
// [p0, p0 + n_fft) where combined = hist[0, hist_len) ++ block[0, n_new).
__global__ void kernel_fill_rows(const float* __restrict__ hist,
                                 const float* __restrict__ block,
                                 const unsigned* __restrict__ active,
                                 float* __restrict__ rows,
                                 int beam,
                                 int nchans,
                                 int nact,
                                 int ctx,
                                 int hist_len,
                                 int64_t n_new,
                                 int64_t p0,
                                 int n_fft) {
    const auto t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= n_fft) {
        return;
    }
    for (int a = static_cast<int>(blockIdx.y); a < nact;
         a += static_cast<int>(gridDim.y)) {
        const auto c = static_cast<int64_t>(beam) * nchans + active[a];
        const auto p = p0 + t;
        float v      = 0.0F;
        if (p < hist_len) {
            v = hist[(c * ctx) + p];
        } else if (p - hist_len < n_new) {
            v = block[(c * n_new) + (p - hist_len)];
        }
        rows[(static_cast<int64_t>(a) * n_fft) + t] = v;
    }
}

// Brute force over the listed trials: Y[row[i]][k] = (1/N) sum_a X[a][k]
// exp(2 pi i tau(i, a) k / N), tau = tau_int + tau_frac.
__global__ void kernel_brute(const ComplexTypeGPU* __restrict__ spec,
                             const int* __restrict__ tau_int,
                             const float* __restrict__ tau_frac,
                             const unsigned* __restrict__ rows,
                             ComplexTypeGPU* __restrict__ out,
                             int nact,
                             int ntrials,
                             int n_bins,
                             unsigned n_fft,
                             unsigned long long magic,
                             float inv_n) {
    __shared__ int s_int[kDmPerThread][kChanTile];
    __shared__ float s_frac[kDmPerThread][kChanTile];
    const int k       = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const float kf    = static_cast<float>(k);
    const auto ku     = static_cast<unsigned>(k);
    const int nblocks = (ntrials + kDmPerThread - 1) / kDmPerThread;
    for (int yb = static_cast<int>(blockIdx.y); yb < nblocks;
         yb += static_cast<int>(gridDim.y)) {
        const int i0 = yb * kDmPerThread;
        float ar[kDmPerThread];
        float ai[kDmPerThread];
#pragma unroll
        for (int i = 0; i < kDmPerThread; ++i) {
            ar[i] = 0.0F;
            ai[i] = 0.0F;
        }
        for (int a0 = 0; a0 < nact; a0 += kChanTile) {
            const int na = min(kChanTile, nact - a0);
            __syncthreads();
            for (int j = threadIdx.x; j < kDmPerThread * kChanTile;
                 j += blockDim.x) {
                const int i    = j / kChanTile;
                const int a    = j % kChanTile;
                const int tr   = i0 + i;
                const bool ok  = (tr < ntrials) && (a < na);
                const auto idx = (static_cast<int64_t>(ok ? tr : 0) * nact) +
                                 a0 + (ok ? a : 0);
                s_int[i][a]    = ok ? tau_int[idx] : 0;
                s_frac[i][a]   = ok ? tau_frac[idx] : 0.0F;
            }
            __syncthreads();
            if (k < n_bins) {
                for (int a = 0; a < na; ++a) {
                    const ComplexTypeGPU x =
                        spec[(static_cast<int64_t>(a0 + a) * n_bins) + k];
#pragma unroll
                    for (int i = 0; i < kDmPerThread; ++i) {
                        const unsigned m =
                            mod_mul(static_cast<unsigned>(s_int[i][a]), ku,
                                    n_fft, magic);
                        float u =
                            fmaf(s_frac[i][a], kf, static_cast<float>(m)) *
                            inv_n;
                        u -= rintf(u);
                        float sn = 0.0F;
                        float cs = 0.0F;
                        // SFU sincos: ~4e-7 absolute on the reduced
                        // |angle| <= pi, and far faster than sincospif.
                        __sincosf(6.28318530717958647692F * u, &sn, &cs);
                        ar[i] = fmaf(x.real(), cs, fmaf(-x.imag(), sn, ar[i]));
                        ai[i] = fmaf(x.real(), sn, fmaf(x.imag(), cs, ai[i]));
                    }
                }
            }
        }
        if (k < n_bins) {
#pragma unroll
            for (int i = 0; i < kDmPerThread; ++i) {
                const int tr = i0 + i;
                if (tr < ntrials) {
                    out[(static_cast<int64_t>(rows[tr]) * n_bins) + k] =
                        ComplexTypeGPU(ar[i] * inv_n, ai[i] * inv_n);
                }
            }
        }
    }
}

// NUFFT spreading of one run: block k spreads the channels of bin k onto its
// fine grid (shared memory, nf cells) and writes it to grids[k].
__global__ void kernel_nufft_spread(const ComplexTypeGPU* __restrict__ spec,
                                    const double* __restrict__ rate,
                                    ComplexTypeGPU* __restrict__ grids,
                                    int nact,
                                    int n_bins,
                                    int nf,
                                    int w,
                                    float beta,
                                    double inv_n,
                                    double ddm,
                                    double shift) {
    extern __shared__ float s_grid[]; // nf complex (re, im interleaved)
    const int k = static_cast<int>(blockIdx.x);
    for (int j = threadIdx.x; j < 2 * nf; j += blockDim.x) {
        s_grid[j] = 0.0F;
    }
    __syncthreads();
    const double kn    = static_cast<double>(k) * inv_n;
    const float half_w = 0.5F * static_cast<float>(w);
    for (int a = threadIdx.x; a < nact; a += blockDim.x) {
        const double r = rate[a];
        const double x = kn * ddm * r;   // turns
        double ph      = kn * shift * r; // centring + dm0 phase
        ph -= rint(ph);
        float sn = 0.0F;
        float cs = 0.0F;
        sincospif(2.0F * static_cast<float>(ph), &sn, &cs);
        const ComplexTypeGPU xv = spec[(static_cast<int64_t>(a) * n_bins) + k];
        const float amp_r       = (xv.real() * cs) - (xv.imag() * sn);
        const float amp_i       = (xv.real() * sn) + (xv.imag() * cs);
        const double g          = (x - floor(x)) * static_cast<double>(nf);
        const double lc         = ceil(g - (0.5 * w));
        const auto l0           = static_cast<int>(lc);
        const auto s0           = static_cast<float>(lc - g);
        for (int i = 0; i < w; ++i) {
            const float z  = (s0 + static_cast<float>(i)) / half_w;
            const float rr = 1.0F - (z * z);
            const float ker =
                rr > 0.0F ? __expf(beta * (sqrtf(rr) - 1.0F)) : 0.0F;
            int cell = l0 + i;
            cell     = cell < 0 ? cell + nf : (cell >= nf ? cell - nf : cell);
            atomicAdd(&s_grid[2 * cell], amp_r * ker);
            atomicAdd(&s_grid[(2 * cell) + 1], amp_i * ker);
        }
    }
    __syncthreads();
    auto* dst =
        reinterpret_cast<float*>(grids + (static_cast<int64_t>(k) * nf));
    for (int j = threadIdx.x; j < 2 * nf; j += blockDim.x) {
        dst[j] = s_grid[j];
    }
}

// Y[row0 + d][k] = G_k((d - half) mod nf) / Psi(d - half) / N, through a
// shared-memory tile so both the grid reads (along d) and the DM-row writes
// (along k) are coalesced.
__global__ void kernel_nufft_deconv(const ComplexTypeGPU* __restrict__ grids,
                                    const float* __restrict__ inv_psi,
                                    ComplexTypeGPU* __restrict__ out,
                                    int count,
                                    int row0,
                                    int n_bins,
                                    int nf,
                                    int half,
                                    float inv_n) {
    __shared__ float2 tile[kTile][kTile + 1];
    const int k0 = static_cast<int>(blockIdx.x) * kTile;
    const int nd = (count + kTile - 1) / kTile;
    for (int db = static_cast<int>(blockIdx.y); db < nd;
         db += static_cast<int>(gridDim.y)) {
        const int d0 = db * kTile;
        __syncthreads();
        for (int ky = static_cast<int>(threadIdx.y); ky < kTile;
             ky += kTileRows) {
            const int k = k0 + ky;
            const int d = d0 + static_cast<int>(threadIdx.x);
            if (k < n_bins && d < count) {
                int m = d - half;
                m     = m < 0 ? m + nf : m;
                const ComplexTypeGPU v =
                    grids[(static_cast<int64_t>(k) * nf) + m];
                tile[ky][threadIdx.x] = make_float2(v.real(), v.imag());
            }
        }
        __syncthreads();
        for (int dy = static_cast<int>(threadIdx.y); dy < kTile;
             dy += kTileRows) {
            const int d = d0 + dy;
            const int k = k0 + static_cast<int>(threadIdx.x);
            if (k < n_bins && d < count) {
                const float s  = inv_psi[d] * inv_n;
                const float2 v = tile[threadIdx.x][dy];
                out[(static_cast<int64_t>(row0 + d) * n_bins) + k] =
                    ComplexTypeGPU(v.x * s, v.y * s);
            }
        }
    }
}

__global__ void kernel_trim(const float* __restrict__ time,
                            float* __restrict__ out,
                            int ndm,
                            int n_fft,
                            int guard,
                            int64_t n_out,
                            int64_t p0,
                            int cnt) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= cnt) {
        return;
    }
    for (int d = static_cast<int>(blockIdx.y); d < ndm;
         d += static_cast<int>(gridDim.y)) {
        out[(static_cast<int64_t>(d) * n_out) + p0 + i] =
            time[(static_cast<int64_t>(d) * n_fft) + guard + i];
    }
}

// New history: combined positions [start, start + new_len) of every row.
__global__ void kernel_update_history(const float* __restrict__ hist,
                                      const float* __restrict__ block,
                                      float* __restrict__ hist_next,
                                      int rows,
                                      int ctx,
                                      int hist_len,
                                      int64_t n_new,
                                      int64_t start,
                                      int new_len) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= new_len) {
        return;
    }
    for (int r = static_cast<int>(blockIdx.y); r < rows;
         r += static_cast<int>(gridDim.y)) {
        const auto p = start + i;
        hist_next[(static_cast<int64_t>(r) * ctx) + i] =
            p < hist_len
                ? hist[(static_cast<int64_t>(r) * ctx) + p]
                : block[(static_cast<int64_t>(r) * n_new) + (p - hist_len)];
    }
}

// Channel-major packed rows -> float rows.
template <unsigned NBITS>
__global__ void kernel_unpack(const uint8_t* __restrict__ packed,
                              float* __restrict__ out,
                              int64_t row_bytes,
                              int64_t nsamps,
                              int rows) {
    const auto s = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= nsamps) {
        return;
    }
    for (int r = static_cast<int>(blockIdx.y); r < rows;
         r += static_cast<int>(gridDim.y)) {
        out[(static_cast<int64_t>(r) * nsamps) + s] =
            static_cast<float>(bit_pack_utils::read_packed_sample<NBITS>(
                packed + (static_cast<int64_t>(r) * row_bytes),
                static_cast<SizeType>(s)));
    }
}

// Time-major packed spectra (nbeams, nsamps, nchans) -> float rows
// (nbeams, nchans, nsamps).
template <unsigned NBITS>
__global__ void kernel_unpack_time_major(const uint8_t* __restrict__ packed,
                                         float* __restrict__ out,
                                         int64_t samp_bytes,
                                         int64_t nsamps,
                                         int nchans,
                                         int64_t nspectra) {
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= nchans) {
        return;
    }
    for (int64_t j = blockIdx.y; j < nspectra; j += gridDim.y) {
        const auto b = j / nsamps;
        const auto s = j % nsamps;
        out[(((b * nchans) + c) * nsamps) + s] =
            static_cast<float>(bit_pack_utils::read_packed_sample<NBITS>(
                packed + (j * samp_bytes), static_cast<SizeType>(c)));
    }
}

template <unsigned NB>
void launch_unpack(const uint8_t* packed,
                   float* out,
                   SizeType row_bytes,
                   SizeType nsamps,
                   SizeType rows,
                   bool time_major,
                   SizeType nchans,
                   cudaStream_t st) {
    if (time_major) {
        const auto nspectra = rows / nchans * nsamps; // nbeams * nsamps
        const dim3 grid(blocks_for(nchans, kThreads), grid_y(nspectra));
        kernel_unpack_time_major<NB><<<grid, kThreads, 0, st>>>(
            packed, out, static_cast<int64_t>(row_bytes),
            static_cast<int64_t>(nsamps), static_cast<int>(nchans),
            static_cast<int64_t>(nspectra));
    } else {
        const dim3 grid(blocks_for(nsamps, kThreads), grid_y(rows));
        kernel_unpack<NB><<<grid, kThreads, 0, st>>>(
            packed, out, static_cast<int64_t>(row_bytes),
            static_cast<int64_t>(nsamps), static_cast<int>(rows));
    }
}

void unpack_on_device(const uint8_t* packed,
                      float* out,
                      SizeType nbits,
                      SizeType row_bytes,
                      SizeType nsamps,
                      SizeType rows,
                      bool time_major,
                      SizeType nchans,
                      cudaStream_t st) {
    switch (nbits) {
    case 1:
        launch_unpack<1>(packed, out, row_bytes, nsamps, rows, time_major,
                         nchans, st);
        break;
    case 2:
        launch_unpack<2>(packed, out, row_bytes, nsamps, rows, time_major,
                         nchans, st);
        break;
    case 4:
        launch_unpack<4>(packed, out, row_bytes, nsamps, rows, time_major,
                         nchans, st);
        break;
    case 8:
        launch_unpack<8>(packed, out, row_bytes, nsamps, rows, time_major,
                         nchans, st);
        break;
    default:
        launch_unpack<16>(packed, out, row_bytes, nsamps, rows, time_major,
                          nchans, st);
        break;
    }
}

struct GpuGeometry {
    SizeType n_fft{};
    SizeType n_bins{};
    std::unique_ptr<utils::CUFFTManager> r2c; // nact rows
    std::unique_ptr<utils::CUFFTManager> c2r; // ndm rows
    // One batched C2C over the n_bins fine grids per distinct run grid.
    std::map<SizeType, std::unique_ptr<utils::CUFFTManager>> nu;
};

// One NUFFT run on the device.
struct GpuRun {
    ddmt_fft::UniformRun run;
    const nufft::Type1Plan* plan{nullptr};
    thrust::device_vector<float> inv_psi;
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
        m_hist_d.assign(m_nbeams * m_nchans * m_ctx, 0.0F);
        m_hist_next_d.assign(m_hist_d.size(), 0.0F);
        m_hist_len = m_model.guard;
        std::vector<unsigned> act(m_model.active.begin(), m_model.active.end());
        m_active_d = act;
        // Last: nothing after it throws, so the destructor always frees it.
        gpu_utils::check_gpu_call(cudaStreamCreate(&m_stream),
                                  "DDMTFFT (gpu): stream");
        logging::debug("DDMTFFT gpu: {} ({} NUFFT run(s), {} brute trials) "
                       "ndm={} active chans={} max_tau={:.2f} guard={} "
                       "segment={}",
                       method_used(), m_runs.size(), m_brute_rows.size(), m_ndm,
                       m_nact, m_model.max_tau, m_model.guard, m_seg_n);
    }

    ~DDMTFFTCudaEngine() override {
        if (m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
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
    void set_gulp_size(SizeType gulp_size) override { m_gulp = gulp_size; }
    [[nodiscard]] SizeType get_gulp_size() const noexcept override {
        return m_gulp == 0 ? 65536 : m_gulp;
    }
    [[nodiscard]] std::string_view method_used() const noexcept override {
        return detail::ddmt_fft_method_used(m_runs.size(), m_brute_rows.size());
    }

    // ---- history ----

    void save_history(std::span<float> out) const override {
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        m_device_work.wait();
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaMemcpy(out.data(), raw(m_hist_d),
                                             out.size() * sizeof(float),
                                             cudaMemcpyDeviceToHost),
                                  "DDMTFFT::save_history");
    }
    void load_history(std::span<const float> in) override {
        check_size(in.size(), history_state_size(), "load_history");
        m_device_work.wait();
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaMemcpy(raw(m_hist_d), in.data(),
                                             in.size() * sizeof(float),
                                             cudaMemcpyHostToDevice),
                                  "DDMTFFT::load_history");
        m_hist_len  = m_ctx;
        m_hist_zero = false;
    }
    // Device analogues are ordered on the given stream after the engine's
    // own queued work (the last call's stream).
    void save_history(DeviceSpan<float> out, Stream stream) const override {
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        gpu_utils::check_gpu_call(cudaMemcpyAsync(out.data(), raw(m_hist_d),
                                                  out.size() * sizeof(float),
                                                  cudaMemcpyDeviceToDevice,
                                                  to_cuda(stream)),
                                  "DDMTFFT::save_history (device)");
    }
    void load_history(DeviceSpan<const float> in, Stream stream) override {
        check_size(in.size(), history_state_size(), "load_history");
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(raw(m_hist_d), in.data(), in.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice, to_cuda(stream)),
            "DDMTFFT::load_history (device)");
        m_device_work.mark(to_cuda(stream));
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
        m_device_work.wait();
        const auto in_n = m_nbeams * m_nchans * nsamps;
        m_in_d.resize(std::max<SizeType>(in_n, 1));
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(raw(m_in_d), waterfall.data(), in_n * sizeof(float),
                            cudaMemcpyHostToDevice, m_stream),
            "DDMTFFT::execute H2D");
        finish_host(raw(m_in_d), nsamps, dmt);
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
        // Scratch and history are shared by every call: order after the
        // previous call's stream.
        m_device_work.wait();
        core(waterfall.data(), nsamps, dmt.data(), to_cuda(stream));
        m_device_work.mark(to_cuda(stream));
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
        m_device_work.wait();
        const auto st = to_cuda(stream);
        m_unpacked_d.resize(std::max<SizeType>(rows * nsamps, 1));
        unpack_on_device(waterfall_packed.data(), raw(m_unpacked_d), nbits,
                         row_bytes, nsamps, rows, false, m_nchans, st);
        core(raw(m_unpacked_d), nsamps, dmt.data(), st);
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
    cudaStream_t m_stream{nullptr};
    mutable gpu_utils::DeviceWorkFence m_device_work;

    // Methods: NUFFT runs (plans per distinct run length) and the trials
    // left to brute force (their output rows and exact delays).
    std::map<SizeType, std::unique_ptr<nufft::Type1Plan>> m_plans;
    std::vector<std::unique_ptr<GpuRun>> m_runs;
    std::vector<unsigned> m_brute_rows;
    SizeType m_nf_max{0};
    thrust::device_vector<unsigned> m_brute_rows_d;
    thrust::device_vector<int> m_tau_int_d;
    thrust::device_vector<float> m_tau_frac_d;
    thrust::device_vector<double> m_rate_d;

    thrust::device_vector<float> m_hist_d;
    thrust::device_vector<float> m_hist_next_d;
    thrust::device_vector<unsigned> m_active_d;
    thrust::device_vector<float> m_rows_d;
    thrust::device_vector<ComplexTypeGPU> m_spec_d;
    thrust::device_vector<ComplexTypeGPU> m_out_spec_d;
    thrust::device_vector<ComplexTypeGPU> m_grids_d;
    thrust::device_vector<float> m_time_d;
    thrust::device_vector<float> m_unpacked_d;
    thrust::device_vector<uint8_t> m_packed_d;
    thrust::device_vector<float> m_in_d;
    thrust::device_vector<float> m_out_d;
    std::map<SizeType, GpuGeometry> m_geos;

    void setup_methods() {
        std::vector<bool> by_nufft(m_ndm, false);
        if (m_opts.method == DDMTFFTMethod::kNUFFT) {
            int optin = 0;
            gpu_utils::check_gpu_call(
                cudaDeviceGetAttribute(&optin,
                                       cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                       m_device_id),
                "DDMTFFT (gpu): shared memory limit");
            SizeType smem_max = 0;
            for (const auto& r : m_model.runs) {
                if (!m_model.nufft_run(r, /*explicit_nufft=*/true)) {
                    continue;
                }
                auto& p = m_plans[r.count];
                if (!p) {
                    p = std::make_unique<nufft::Type1Plan>(r.count,
                                                           m_opts.tolerance);
                }
                const auto smem = 2 * p->fine_grid() * sizeof(float);
                if (smem > static_cast<SizeType>(optin)) {
                    logging::debug("DDMTFFT (gpu): run of {} trials needs {} B "
                                   "of shared memory (limit {}); brute force",
                                   r.count, smem, optin);
                    continue;
                }
                auto run      = std::make_unique<GpuRun>();
                run->run      = r;
                run->plan     = p.get();
                const auto dc = p->deconvolution();
                run->inv_psi.assign(dc.begin(), dc.end());
                m_runs.push_back(std::move(run));
                smem_max = std::max(smem_max, smem);
                m_nf_max = std::max(m_nf_max, p->fine_grid());
                std::fill_n(by_nufft.begin() +
                                static_cast<std::ptrdiff_t>(r.begin),
                            r.count, true);
            }
            if (smem_max > 0) {
                gpu_utils::check_gpu_call(
                    cudaFuncSetAttribute(
                        kernel_nufft_spread,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(smem_max)),
                    "DDMTFFT (gpu): NUFFT shared memory");
            }
            std::vector<double> rates(m_nact);
            for (SizeType a = 0; a < m_nact; ++a) {
                rates[a] = m_model.rate[m_model.active[a]];
            }
            m_rate_d = rates;
        }
        for (SizeType d = 0; d < m_ndm; ++d) {
            if (!by_nufft[d]) {
                m_brute_rows.push_back(static_cast<unsigned>(d));
            }
        }
        if (!m_brute_rows.empty()) {
            // tau = n + f per (brute trial, active channel), exact in double.
            const auto nb = m_brute_rows.size();
            std::vector<int> ti(nb * m_nact);
            std::vector<float> tf(nb * m_nact);
            for (SizeType i = 0; i < nb; ++i) {
                for (SizeType a = 0; a < m_nact; ++a) {
                    const double t =
                        m_model.tau(m_brute_rows[i], m_model.active[a]);
                    const double fl      = std::floor(t);
                    ti[(i * m_nact) + a] = static_cast<int>(fl);
                    tf[(i * m_nact) + a] = static_cast<float>(t - fl);
                }
            }
            m_tau_int_d    = ti;
            m_tau_frac_d   = tf;
            m_brute_rows_d = m_brute_rows;
        }
    }

    // Longest transform: the channel spectra cap of the CPU engine, and
    // every per-segment buffer (rows, spectra, DM spectra, DM rows, NUFFT
    // grids; bytes per transform sample below) within half the free device
    // memory.
    [[nodiscard]] SizeType segment_cap() const {
        std::size_t free_b  = 0;
        std::size_t total_b = 0;
        gpu_utils::check_gpu_call(cudaMemGetInfo(&free_b, &total_b),
                                  "DDMTFFT (gpu): memory info");
        const SizeType budget = std::min<SizeType>(
            kMaxWorkBytes, static_cast<SizeType>(free_b) / 2);
        const SizeType per_sample =
            (8 * m_nact) + (8 * m_ndm) + (4 * m_nf_max) + 1;
        const SizeType cap = std::max<SizeType>(
            budget / per_sample, utils::next_fft_size(2 * m_ctx + 2));
        return std::min(
            ddmt_fft::max_segment_length(m_ctx, m_nact, kMaxSpectraBytes), cap);
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
            for (const auto& run : m_runs) {
                const auto nf = run->plan->fine_grid();
                if (!g.nu.contains(nf)) {
                    g.nu[nf] = std::make_unique<utils::CUFFTManager>(
                        utils::FFTKind::kC2CBackward, nf, g.n_bins,
                        m_device_id);
                }
            }
        }
        m_rows_d.resize(m_nact * n);
        m_spec_d.resize(m_nact * g.n_bins);
        m_out_spec_d.resize(m_ndm * g.n_bins);
        m_time_d.resize(m_ndm * n);
        if (m_nf_max > 0) {
            m_grids_d.resize(g.n_bins * m_nf_max);
        }
        return g;
    }

    // Host packed input: upload the packed bytes, unpack on the device.
    void host_packed(std::span<const uint8_t> packed,
                     SizeType row_bytes,
                     SizeType nsamps,
                     bool time_major,
                     std::span<float> dmt) {
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        const auto rows = m_nbeams * m_nchans;
        m_packed_d.resize(std::max<SizeType>(packed.size(), 1));
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(raw(m_packed_d), packed.data(), packed.size(),
                            cudaMemcpyHostToDevice, m_stream),
            "DDMTFFT::execute H2D (packed)");
        m_unpacked_d.resize(std::max<SizeType>(rows * nsamps, 1));
        unpack_on_device(raw(m_packed_d), raw(m_unpacked_d), m_plan.get_nbits(),
                         row_bytes, nsamps, rows, time_major, m_nchans,
                         m_stream);
        finish_host(raw(m_unpacked_d), nsamps, dmt);
    }

    // Host call on device input rows: dedisperse, copy back, wait.
    void
    finish_host(const float* rows_d, SizeType n_new, std::span<float> out) {
        const auto out_n = out_size(n_new);
        m_out_d.resize(std::max<SizeType>(out_n, 1));
        core(rows_d, n_new, raw(m_out_d), m_stream);
        if (out_n > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(out.data(), raw(m_out_d), out_n * sizeof(float),
                                cudaMemcpyDeviceToHost, m_stream),
                "DDMTFFT::execute D2H");
        }
        gpu_utils::check_gpu_call(cudaStreamSynchronize(m_stream),
                                  "DDMTFFT::execute sync");
    }

    void core(const float* data, SizeType n_new, float* out, cudaStream_t st) {
        if (m_hist_zero) {
            gpu_utils::check_gpu_call(
                cudaMemsetAsync(raw(m_hist_d), 0,
                                m_hist_d.size() * sizeof(float), st),
                "DDMTFFT: reset history");
            m_hist_zero = false;
        }
        const auto total = m_hist_len + n_new;
        const auto n_out = total > m_ctx ? total - m_ctx : 0;
        // While the stream warms up, also prepare the steady-state
        // transform (n_new outputs per call) so the first warm call does not
        // pay for new FFT plans and first-touch of its buffers.
        if (m_hist_len < m_ctx && n_new > 0) {
            geometry(ddmt_fft::plan_segments(n_new, m_ctx, m_seg_n).n_fft);
        }
        if (n_out > 0 && m_ndm > 0 && m_nact > 0) {
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
        const dim3 fill_grid(blocks_for(g.n_fft, kThreads), grid_y(m_nact));
        gpu_utils::check_kernel_launch_params(fill_grid, dim3(kThreads));
        kernel_fill_rows<<<fill_grid, kThreads, 0, st>>>(
            raw(m_hist_d), data, raw(m_active_d), raw(m_rows_d),
            static_cast<int>(beam), static_cast<int>(m_nchans),
            static_cast<int>(m_nact), static_cast<int>(m_ctx),
            static_cast<int>(m_hist_len), static_cast<int64_t>(n_new),
            static_cast<int64_t>(p0), static_cast<int>(g.n_fft));
        g.r2c->execute(
            cuda::std::span<float>(raw(m_rows_d), m_nact * g.n_fft),
            cuda::std::span<ComplexTypeGPU>(raw(m_spec_d), m_nact * g.n_bins),
            st);
        const float inv_n = 1.0F / static_cast<float>(g.n_fft);
        for (const auto& run : m_runs) {
            const auto& p = *run->plan;
            const auto nf = p.fine_grid();
            const double sh =
                run->run.dm0 + (static_cast<double>(p.half()) * run->run.ddm);
            const dim3 sgrid(static_cast<unsigned>(g.n_bins));
            gpu_utils::check_kernel_launch_params(sgrid, dim3(kThreads));
            kernel_nufft_spread<<<sgrid, kThreads, 2 * nf * sizeof(float),
                                  st>>>(
                raw(m_spec_d), raw(m_rate_d), raw(m_grids_d),
                static_cast<int>(m_nact), static_cast<int>(g.n_bins),
                static_cast<int>(nf), p.width(), static_cast<float>(p.beta()),
                1.0 / static_cast<double>(g.n_fft), run->run.ddm, sh);
            g.nu.at(nf)->execute(
                cuda::std::span<ComplexTypeGPU>(raw(m_grids_d), g.n_bins * nf),
                st);
            const dim3 dgrid(blocks_for(g.n_bins, kTile),
                             grid_y((run->run.count + kTile - 1) / kTile));
            const dim3 dblock(kTile, kTileRows);
            gpu_utils::check_kernel_launch_params(dgrid, dblock);
            kernel_nufft_deconv<<<dgrid, dblock, 0, st>>>(
                raw(m_grids_d), raw(run->inv_psi), raw(m_out_spec_d),
                static_cast<int>(run->run.count),
                static_cast<int>(run->run.begin), static_cast<int>(g.n_bins),
                static_cast<int>(nf), static_cast<int>(p.half()), inv_n);
        }
        if (!m_brute_rows.empty()) {
            const auto nb = m_brute_rows.size();
            const dim3 bgrid(blocks_for(g.n_bins, kThreads),
                             grid_y(blocks_for(nb, kDmPerThread)));
            gpu_utils::check_kernel_launch_params(bgrid, dim3(kThreads));
            const auto big_n = static_cast<unsigned>(g.n_fft);
            kernel_brute<<<bgrid, kThreads, 0, st>>>(
                raw(m_spec_d), raw(m_tau_int_d), raw(m_tau_frac_d),
                raw(m_brute_rows_d), raw(m_out_spec_d),
                static_cast<int>(m_nact), static_cast<int>(nb),
                static_cast<int>(g.n_bins), big_n, ~0ULL / big_n, inv_n);
        }
        g.c2r->execute(cuda::std::span<float>(raw(m_time_d), m_ndm * g.n_fft),
                       cuda::std::span<ComplexTypeGPU>(raw(m_out_spec_d),
                                                       m_ndm * g.n_bins),
                       st);
        const dim3 tgrid(blocks_for(cnt, kThreads), grid_y(m_ndm));
        gpu_utils::check_kernel_launch_params(tgrid, dim3(kThreads));
        kernel_trim<<<tgrid, kThreads, 0, st>>>(
            raw(m_time_d), out, static_cast<int>(m_ndm),
            static_cast<int>(g.n_fft), static_cast<int>(m_model.guard),
            static_cast<int64_t>(n_out), static_cast<int64_t>(p0),
            static_cast<int>(cnt));
    }

    void update_history(const float* data, SizeType n_new, cudaStream_t st) {
        const auto total   = m_hist_len + n_new;
        const auto new_len = std::min(total, m_ctx);
        const auto rows    = m_nbeams * m_nchans;
        if (new_len > 0) {
            const dim3 grid(blocks_for(new_len, kThreads), grid_y(rows));
            gpu_utils::check_kernel_launch_params(grid, dim3(kThreads));
            kernel_update_history<<<grid, kThreads, 0, st>>>(
                raw(m_hist_d), data, raw(m_hist_next_d), static_cast<int>(rows),
                static_cast<int>(m_ctx), static_cast<int>(m_hist_len),
                static_cast<int64_t>(n_new),
                static_cast<int64_t>(total - new_len),
                static_cast<int>(new_len));
        }
        m_hist_d.swap(m_hist_next_d);
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
