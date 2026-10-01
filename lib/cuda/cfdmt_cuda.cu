#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/baseband_layout.hpp"
#include "dmt/cfdmt_common.hpp"
#include "dmt/cfdmt_fused_gpu.cuh"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/fourier_gpu.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/unpacker_cuda.cuh"

// GPU engine: the CPU engine's pipeline (lib/cpu/cfdmt_cpu.cpp) as kernels.
// Per block:
//   front end, per chunk of subbands: (upload) -> unpack -> batched forward
//   cuFFT -> channel powers; then the channel noise statistics;
//   per coarse trial: coherent stage -> fine FDMT -> crop of the valid
//   output window.
// The coherent stage (gather x chirp with exact fixed-point phases identical
// to the CPU, inverse channel FFT, detect, normalise, trim, delay-aligned
// write) is one fused kernel (cfdmt_fused_gpu.cuh) for power-of-two channel
// FFTs up to 4096, else gather -> batched inverse cuFFT -> detect through a
// work buffer.
//
// Host-memory calls upload chunk by chunk on a copy stream while the front
// end of the chunks already on the device runs, and return the result
// through gpu_host::ChunkedStager.

namespace dmt::algorithms {

namespace {

using fourier_gpu::DevBuf;
using fourier_gpu::kBlock;

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

unsigned grid_for(uint64_t n) {
    return static_cast<unsigned>(
        std::clamp<uint64_t>((n + kBlock - 1) / kBlock, 1, uint64_t{1} << 20));
}

// Front-end chunk size as a share of the device L2: unpack, the forward
// FFT's passes and the channel powers of a chunk then run out of L2, and the
// host path's upload overlaps with the front end chunk by chunk.
constexpr double kChunkL2Share = 0.25;
// Share of the free device memory the cuFFT path's work buffer may take.
constexpr double kWorkShare = 0.25;

// Per (subband, FFT block, channel) cell in [cell0, cell0 + gridDim.x):
// filtered power of each polarisation, sum_b |c_b|^2 |X_p(b)|^2 (see the
// CPU engine's front_end()).
__global__ void channel_power_kernel(const float2* __restrict__ spec,
                                     const float* __restrict__ taper,
                                     uint64_t cell0,
                                     uint64_t n_p,
                                     uint64_t nbin,
                                     uint64_t mbin,
                                     float* __restrict__ power) {
    __shared__ float red[2][kBlock];
    const uint64_t cell  = cell0 + blockIdx.x; // (s * nfft + j) * n_p + ichan
    const uint64_t ichan = cell % n_p;
    const uint64_t task  = cell / n_p;
    const float2* row0   = spec + (task * 2 * nbin);
    const float2* row1   = row0 + nbin;
    const uint64_t off   = ((ichan * mbin) + (nbin / 2)) % nbin;
    float acc0           = 0.0F;
    float acc1           = 0.0F;
    for (uint64_t b = threadIdx.x; b < mbin; b += blockDim.x) {
        uint64_t idx   = off + b;
        idx            = idx >= nbin ? idx - nbin : idx;
        const float w2 = taper[b] * taper[b];
        const float2 x = row0[idx];
        const float2 y = row1[idx];
        acc0 += w2 * ((x.x * x.x) + (x.y * x.y));
        acc1 += w2 * ((y.x * y.x) + (y.y * y.y));
    }
    red[0][threadIdx.x] = acc0;
    red[1][threadIdx.x] = acc1;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            red[0][threadIdx.x] += red[0][threadIdx.x + s];
            red[1][threadIdx.x] += red[1][threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        power[(cell * 2) + 0] = red[0][0];
        power[(cell * 2) + 1] = red[1][0];
    }
}

// Per channel: mean m0 + m1 and 1 / sqrt(m0^2 + m1^2) of detected noise.
__global__ void channel_stats_kernel(const float* __restrict__ power,
                                     uint64_t nchans,
                                     uint64_t n_p,
                                     uint64_t nfft,
                                     float* __restrict__ mean,
                                     float* __restrict__ inv_sigma) {
    const uint64_t c =
        (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (c >= nchans) {
        return;
    }
    const uint64_t isub  = c / n_p;
    const uint64_t ichan = c % n_p;
    double m0            = 0.0;
    double m1            = 0.0;
    for (uint64_t j = 0; j < nfft; ++j) {
        const float* pw = power + (((((isub * nfft) + j) * n_p) + ichan) * 2);
        m0 += static_cast<double>(pw[0]);
        m1 += static_cast<double>(pw[1]);
    }
    m0 /= static_cast<double>(nfft);
    m1 /= static_cast<double>(nfft);
    const double sigma = sqrt((m0 * m0) + (m1 * m1));
    mean[c]            = static_cast<float>(m0 + m1);
    inv_sigma[c]       = sigma > 0.0 ? static_cast<float>(1.0 / sigma) : 0.0F;
}

// Channel c's window of channel samples [m_lo, m_hi) and first FFT block.
struct Window {
    int64_t m_lo;
    int64_t m_hi;
    int64_t j_first;
};

__device__ __forceinline__ Window channel_window(
    int64_t a, int64_t nf, int64_t msamp, int64_t shift, int64_t lc) {
    Window w{};
    const int64_t lo = a - shift;
    const int64_t hi = a + nf - shift;
    w.m_lo           = lo > 0 ? lo : 0;
    w.m_hi           = hi < msamp ? hi : msamp;
    w.j_first        = w.m_lo / lc;
    return w;
}

// cuFFT path. Thread per (channel of the chunk, bin): the trial's chirp
// (chirp_kernel) applied to the channel's bins of every FFT block of its
// window (both pols).
__global__ void gather_chirp_kernel(const float2* __restrict__ spec,
                                    const float2* __restrict__ chirp,
                                    const int64_t* __restrict__ shifts,
                                    uint64_t c0,
                                    uint64_t nchans,
                                    uint64_t n_p,
                                    uint64_t nfft,
                                    uint64_t nbin,
                                    uint64_t mbin,
                                    uint64_t nwin,
                                    int64_t a,
                                    int64_t nf,
                                    int64_t msamp,
                                    int64_t lc,
                                    float2* __restrict__ work) {
    const uint64_t b =
        (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const uint64_t cc = blockIdx.y; // channel within the chunk
    const uint64_t c  = c0 + cc;
    if (b >= mbin) {
        return;
    }
    float2* dst = work + (cc * nwin * 2 * mbin) + b;
    if (c >= nchans) {
        for (uint64_t r = 0; r < nwin * 2; ++r) {
            dst[r * mbin] = {0.0F, 0.0F};
        }
        return;
    }
    const auto w         = channel_window(a, nf, msamp, shifts[c], lc);
    const float2 h       = chirp[(c * mbin) + b];
    const uint64_t isub  = c / n_p;
    const uint64_t ichan = c % n_p;
    uint64_t idx         = (ichan * mbin) + (nbin / 2) + b;
    idx                  = idx % nbin;
    for (uint64_t q = 0; q < nwin; ++q) {
        const int64_t j = w.j_first + static_cast<int64_t>(q);
        float2 y0       = {0.0F, 0.0F};
        float2 y1       = {0.0F, 0.0F};
        if (w.m_lo < w.m_hi && j < static_cast<int64_t>(nfft)) {
            const float2* row =
                spec + (((isub * nfft) + static_cast<uint64_t>(j)) * 2 * nbin);
            y0 = fourier_gpu::cmul(row[idx], h);
            y1 = fourier_gpu::cmul(row[nbin + idx], h);
        }
        dst[(2 * q) * mbin]       = y0;
        dst[((2 * q) + 1) * mbin] = y1;
    }
}

// cuFFT path. Thread per (window sample u, channel of the chunk): detected
// (and normalised) intensity at aligned time a + u.
__global__ void detect_kernel(const float2* __restrict__ work,
                              const int64_t* __restrict__ shifts,
                              const float* __restrict__ mean,
                              const float* __restrict__ inv_sigma,
                              bool norm,
                              uint64_t c0,
                              uint64_t nchans,
                              uint64_t mbin,
                              uint64_t nwin,
                              uint64_t novc,
                              int64_t a,
                              int64_t nf,
                              int64_t msamp,
                              int64_t lc,
                              float* __restrict__ waterfall) {
    const uint64_t u =
        (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const uint64_t cc = blockIdx.y;
    const uint64_t c  = c0 + cc;
    if (u >= static_cast<uint64_t>(nf) || c >= nchans) {
        return;
    }
    const auto w    = channel_window(a, nf, msamp, shifts[c], lc);
    const int64_t m = a + static_cast<int64_t>(u) - shifts[c];
    float v         = 0.0F;
    if (m >= w.m_lo && m < w.m_hi) {
        const int64_t j = m / lc;
        const auto q    = static_cast<uint64_t>(j - w.j_first);
        const auto i    = static_cast<uint64_t>(m - (j * lc)) + novc;
        const float2* y = work + (((cc * nwin) + q) * 2 * mbin) + i;
        const float2 y0 = y[0];
        const float2 y1 = y[mbin];
        v = (y0.x * y0.x) + (y0.y * y0.y) + (y1.x * y1.x) + (y1.y * y1.y);
        if (norm) {
            v = (v - mean[c]) * inv_sigma[c];
        }
    }
    waterfall[(c * static_cast<uint64_t>(nf)) + u] = v;
}

// out row r of trial k = FDMT output row r from its reference offset.
__global__ void crop_kernel(const float* __restrict__ fdmt_out,
                            const uint64_t* __restrict__ offsets,
                            uint64_t nfine,
                            uint64_t nf,
                            uint64_t nout,
                            float* __restrict__ out) {
    const uint64_t total = nfine * nout;
    for (uint64_t idx =
             (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < total; idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t r = idx / nout;
        const uint64_t t = idx % nout;
        out[idx]         = fdmt_out[(r * nf) + offsets[r] + t];
    }
}

using FusedKernel = void (*)(cfdmt_gpu::FusedArgs);

template <int L> FusedKernel fused_kernel_for(int log2n) {
    if constexpr (L > cfdmt_gpu::kFusedMaxLog2) {
        static_cast<void>(log2n);
        return nullptr;
    } else {
        return log2n == L ? &cfdmt_gpu::fused_coherent_kernel<L>
                          : fused_kernel_for<L + 1>(log2n);
    }
}

// log2(n) if n is a power of two, else -1.
int exact_log2(SizeType n) {
    if (n == 0 || (n & (n - 1)) != 0) {
        return -1;
    }
    int l = 0;
    while ((SizeType{1} << l) < n) {
        ++l;
    }
    return l;
}

class CohFDMTCudaEngine final : public detail::CohFDMTEngine {
public:
    CohFDMTCudaEngine(const plans::CohFDMTPlan& plan,
                      const detail::CohFDMTEngineConfig& cfg)
        : m_plan(plan),
          m_device_id(cfg.exec.device),
          m_unpacker((gpu_utils::set_device(m_device_id), plan.get_format()),
                     plan.get_subband_groups(),
                     plan.get_nbin(),
                     plan.get_nfft(),
                     plan.get_noverlap(),
                     m_device_id),
          m_fdmt(plan.get_f_min(),
                 plan.get_f_max(),
                 plan.get_nchans(),
                 plan.get_fdmt_nsamps(),
                 plan.get_tsamp(),
                 static_cast<IndexType>(plan.get_fine_dt_max()),
                 -static_cast<IndexType>(plan.get_fine_dt_max()),
                 plan.get_config().dt_step,
                 true,
                 "valid",
                 Exec{.backend  = detail::kGPUBackend,
                      .nthreads = 1,
                      .device   = m_device_id}) {
        initialise(cfg);
    }

    ~CohFDMTCudaEngine() override {
        gpu_utils::set_device(m_device_id);
        if (m_copy_stream != nullptr) {
            cudaStreamSynchronize(m_copy_stream);
        }
        for (auto* e : m_chunk_ready) {
            cudaEventDestroy(e);
        }
        if (m_copy_stream != nullptr) {
            cudaStreamDestroy(m_copy_stream);
        }
    }
    CohFDMTCudaEngine(const CohFDMTCudaEngine&)            = delete;
    CohFDMTCudaEngine& operator=(const CohFDMTCudaEngine&) = delete;
    CohFDMTCudaEngine(CohFDMTCudaEngine&&)                 = delete;
    CohFDMTCudaEngine& operator=(CohFDMTCudaEngine&&)      = delete;

    void execute(std::span<const std::span<const uint8_t>> groups,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        check_output(dmt.size());
        const auto& sizes = m_plan.get_subband_groups();
        if (groups.size() != sizes.size()) {
            throw std::invalid_argument(
                std::format("CohFDMT: expected {} input group(s), got {}",
                            sizes.size(), groups.size()));
        }
        std::vector<cuda::std::span<const uint8_t>> d_groups;
        d_groups.reserve(groups.size());
        for (SizeType g = 0; g < groups.size(); ++g) {
            if (groups[g].size() != m_plan.get_input_size(g)) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: input group {} has {} bytes, expected {}", g,
                    groups[g].size(), m_plan.get_input_size(g)));
            }
            d_groups.emplace_back(m_raw[g].data(), groups[g].size());
        }
        cudaStream_t stream = m_host.stream();
        m_fence.order(stream);
        run(d_groups, groups, m_out.data(), stream);
        const gpu_host::ChunkedStager::Segment seg{
            dmt.data(), m_out.data(), m_plan.get_dmt_size() * sizeof(float)};
        m_host.to_host(std::span(&seg, 1)); // synchronises the stream
    }

    void execute(std::span<const DeviceSpan<const uint8_t>> groups,
                 DeviceSpan<float> dmt,
                 Stream stream) override {
        gpu_utils::set_device(m_device_id);
        check_output(dmt.size());
        std::vector<cuda::std::span<const uint8_t>> d_groups;
        d_groups.reserve(groups.size());
        for (const auto& g : groups) {
            d_groups.push_back(to_cuda(g));
        }
        const cudaStream_t s = to_cuda(stream);
        // Calls on different streams share the engine's buffers.
        m_fence.order(s);
        run(d_groups, {}, to_cuda(dmt).data(), s);
        m_fence.mark(s);
    }

    [[nodiscard]] plans::CohFDMTMemoryUsage
    memory_usage() const noexcept override {
        const auto fdmt_mem = m_fdmt.get_memory_usage();
        SizeType workspace  = (m_work.capacity() * sizeof(ComplexTypeGPU)) +
                              (m_chirp.capacity() * sizeof(ComplexTypeGPU)) +
                              (m_twiddle.capacity() * sizeof(ComplexTypeGPU)) +
                              ((m_base.capacity() + m_inc.capacity()) *
                               sizeof(unsigned long long)) +
                              (m_power.capacity() * sizeof(float)) +
                              (m_shifts.capacity() * sizeof(int64_t));
        for (const auto& f : m_fwd) {
            workspace += f->workspace_bytes();
        }
        if (m_inv) {
            workspace += m_inv->workspace_bytes();
        }
        SizeType input = 0;
        for (const auto& r : m_raw) {
            input += r.capacity();
        }

        return {.spectrum  = m_spec.capacity() * sizeof(ComplexTypeGPU),
                .waterfall = m_waterfall.capacity() * sizeof(float),
                .fdmt =
                    fdmt_mem.total() + (m_fdmt_out.capacity() * sizeof(float)),
                .workspace = workspace + input,
                .output    = m_out.capacity() * sizeof(float)};
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    struct Chunk {
        SizeType sub_begin;
        SizeType sub_end;
        const utils::CUFFTManager* fwd;
    };

    const plans::CohFDMTPlan& m_plan; // owned by the CohFDMT facade
    int m_device_id;
    utils::BasebandUnpackerCUDA m_unpacker;
    algorithms::FDMT m_fdmt;

    std::vector<Chunk> m_chunks;
    std::vector<std::unique_ptr<utils::CUFFTManager>> m_fwd; // per chunk size
    std::vector<bool> m_f_outer; // per group: subbands are contiguous bytes

    // Coherent stage.
    cfdmt_gpu::FusedArgs m_fused_args{};
    FusedKernel m_fused{nullptr};
    uint64_t m_fused_ctas{0};
    uint64_t m_fused_smem{0};
    std::unique_ptr<utils::CUFFTManager> m_inv; // cuFFT path
    SizeType m_nwin{};
    SizeType m_chunk{};

    DevBuf<ComplexTypeGPU> m_spec;    // (sub, ifft, pol, nbin)
    DevBuf<ComplexTypeGPU> m_work;    // cuFFT path: (chan, block, pol, mbin)
    DevBuf<ComplexTypeGPU> m_chirp;   // (nchans, mbin) of the current trial
    DevBuf<ComplexTypeGPU> m_twiddle; // fused path: (mbin / 2)
    DevBuf<float> m_power;
    DevBuf<float> m_mean;
    DevBuf<float> m_inv_sigma;
    DevBuf<float> m_taper;
    DevBuf<unsigned long long> m_base;
    DevBuf<unsigned long long> m_inc;
    DevBuf<int64_t> m_shifts; // (ndm_coh, nchans)
    DevBuf<uint64_t> m_offsets;
    DevBuf<float> m_waterfall;
    DevBuf<float> m_fdmt_out;
    DevBuf<float> m_out;                // host entry point only
    std::vector<DevBuf<uint8_t>> m_raw; // host entry point only

    gpu_host::ChunkedStager m_host; // compute stream + result download
    cudaStream_t m_copy_stream{nullptr};
    std::vector<cudaEvent_t> m_chunk_ready;
    gpu_utils::DeviceWorkFence m_fence;

    void check_output(SizeType size) const {
        if (size < m_plan.get_dmt_size()) {
            throw std::invalid_argument(std::format(
                "CohFDMT (gpu): dmt buffer too small. Expected at least {} "
                "(get_dmt_size()), got {}",
                m_plan.get_dmt_size(), size));
        }
    }

    [[nodiscard]] SizeType lc() const noexcept {
        return m_plan.get_mbin() -
               (2 * (m_plan.get_noverlap() / m_plan.get_n_p()));
    }

    void initialise(const detail::CohFDMTEngineConfig& cfg) {
        gpu_utils::set_device(m_device_id);
        const auto nchans = m_plan.get_nchans();
        const auto nsub   = m_plan.get_nsub();
        const auto nfft   = m_plan.get_nfft();
        const auto nbin   = m_plan.get_nbin();
        const auto mbin   = m_plan.get_mbin();
        const auto n_p    = m_plan.get_n_p();
        m_nwin            = ((m_plan.get_fdmt_nsamps() + lc() - 1) / lc()) + 1;

        // Front-end chunks of equal size (plus a remainder).
        int l2 = 0;
        gpu_utils::check_gpu_call(
            cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, m_device_id),
            "CohFDMT: device attribute query failed");
        const SizeType sub_bytes =
            SizeType{2} * nfft * nbin * sizeof(ComplexTypeGPU);
        const SizeType per = std::clamp<SizeType>(
            static_cast<SizeType>(kChunkL2Share * static_cast<double>(l2)) /
                sub_bytes,
            1, nsub);
        for (SizeType s0 = 0; s0 < nsub; s0 += per) {
            const SizeType n = std::min(per, nsub - s0);
            if (m_fwd.empty() ||
                m_chunks.back().sub_end - m_chunks.back().sub_begin != n) {
                m_fwd.push_back(std::make_unique<utils::CUFFTManager>(
                    utils::FFTKind::kC2CForward, nbin, n * nfft * 2,
                    m_device_id));
            }
            m_chunks.push_back({s0, s0 + n, m_fwd.back().get()});
        }
        for (const auto g : m_plan.get_subband_groups()) {
            static_cast<void>(g);
            m_f_outer.push_back(
                utils::parse_baseband_order(m_plan.get_format().order)[0] ==
                utils::BasebandAxis::kFreq);
        }

        m_spec.reserve(m_unpacker.output_size());
        m_power.reserve(nsub * nfft * n_p * 2);
        m_waterfall.reserve(nchans * m_plan.get_fdmt_nsamps());
        m_fdmt_out.reserve(m_fdmt.get_plan().get_buffer_size());
        m_chirp.reserve(nchans * mbin);

        const auto ph = cfdmt::make_chirp_phases(m_plan);
        std::vector<unsigned long long> base(ph.base.begin(), ph.base.end());
        std::vector<unsigned long long> inc(ph.inc.begin(), ph.inc.end());
        m_base.upload(base);
        m_inc.upload(inc);
        const auto& taper = m_plan.get_channel_taper();
        std::vector<float> scaled(mbin);
        for (SizeType b = 0; b < mbin; ++b) {
            scaled[b] = taper[b] / static_cast<float>(nbin);
        }
        m_taper.upload(scaled);
        std::vector<int64_t> shifts;
        shifts.reserve(m_plan.get_ndm_coh() * nchans);
        for (SizeType k = 0; k < m_plan.get_ndm_coh(); ++k) {
            const auto s = m_plan.get_channel_shifts(k);
            shifts.insert(shifts.end(), s.begin(), s.end());
        }
        m_shifts.upload(shifts);
        const auto& offs = m_plan.get_row_offsets();
        m_offsets.upload(std::vector<uint64_t>(offs.begin(), offs.end()));
        // Without normalisation the stats are identity.
        m_mean.upload(std::vector<float>(nchans, 0.0F));
        m_inv_sigma.upload(std::vector<float>(nchans, 1.0F));

        const bool fused = choose_fused(cfg.coherent_path);
        if (!fused) {
            init_cufft_path(cfg.work_bytes);
        }

        // Host entry point: device copies of the input and the result.
        m_raw.resize(m_plan.get_subband_groups().size());
        for (SizeType g = 0; g < m_raw.size(); ++g) {
            m_raw[g].reserve(m_plan.get_input_size(g));
        }
        m_out.reserve(m_plan.get_dmt_size());
        gpu_utils::check_gpu_call(
            cudaStreamCreateWithFlags(&m_copy_stream, cudaStreamNonBlocking),
            "CohFDMT: copy stream creation failed");
        m_chunk_ready.resize(m_chunks.size());
        for (auto& e : m_chunk_ready) {
            gpu_utils::check_gpu_call(
                cudaEventCreateWithFlags(&e, cudaEventDisableTiming),
                "CohFDMT: event creation failed");
        }
    }

    // Selects and configures the fused coherent kernel; false: cuFFT path.
    bool choose_fused(detail::CohFDMTCoherentPath path) {
        using detail::CohFDMTCoherentPath;
        const auto mbin  = m_plan.get_mbin();
        const int log2n  = exact_log2(mbin);
        const bool shape = log2n >= cfdmt_gpu::kFusedMinLog2 &&
                           log2n <= cfdmt_gpu::kFusedMaxLog2;
        int optin        = 0;
        gpu_utils::check_gpu_call(
            cudaDeviceGetAttribute(
                &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, m_device_id),
            "CohFDMT: device attribute query failed");
        const bool fits = shape && cfdmt_gpu::fused_smem_bytes(log2n) <=
                                       static_cast<uint64_t>(optin);
        if (path == CohFDMTCoherentPath::kCuFFT ||
            (path == CohFDMTCoherentPath::kAuto && !fits)) {
            return false;
        }
        if (!fits) {
            throw std::invalid_argument(std::format(
                "CohFDMT (gpu): the fused coherent kernel needs a "
                "power-of-two channel FFT of 16-4096 bins fitting in shared "
                "memory; mbin={}",
                mbin));
        }
        m_fused      = fused_kernel_for<cfdmt_gpu::kFusedMinLog2>(log2n);
        m_fused_smem = cfdmt_gpu::fused_smem_bytes(log2n);
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(reinterpret_cast<const void*>(m_fused),
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(m_fused_smem)),
            "CohFDMT: fused kernel shared-memory opt-in failed");

        std::vector<ComplexTypeGPU> tw(mbin / 2);
        for (SizeType k = 0; k < mbin / 2; ++k) {
            const double ang = 2.0 * std::numbers::pi * static_cast<double>(k) /
                               static_cast<double>(mbin);
            tw[k]            = {static_cast<float>(std::cos(ang)),
                                static_cast<float>(std::sin(ang))};
        }
        m_twiddle.upload(tw);

        // FFT blocks per thread block: aim for several waves of thread
        // blocks over the device, in whole rounds.
        int nsm      = 0;
        int resident = 0;
        gpu_utils::check_gpu_call(
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount,
                                   m_device_id),
            "CohFDMT: device attribute query failed");
        gpu_utils::check_gpu_call(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                      &resident, m_fused,
                                      cfdmt_gpu::kFusedThreads,
                                      static_cast<SizeType>(m_fused_smem)),
                                  "CohFDMT: occupancy query failed");
        const auto round =
            static_cast<uint64_t>(cfdmt_gpu::fused_round_blocks(log2n));
        const uint64_t target =
            4 * static_cast<uint64_t>(std::max(1, nsm * resident));
        const uint64_t tasks = m_plan.get_nchans() * m_nwin;
        uint64_t per         = (tasks + target - 1) / target;
        per = std::clamp<uint64_t>(((per + round - 1) / round) * round, round,
                                   ((m_nwin + round - 1) / round) * round);
        m_fused_args.blocks_per_cta = per;
        m_fused_args.ngroups        = (m_nwin + per - 1) / per;
        m_fused_ctas = m_plan.get_nchans() * m_fused_args.ngroups;
        return true;
    }

    void init_cufft_path(SizeType work_bytes) {
        const auto nchans       = m_plan.get_nchans();
        const auto mbin         = m_plan.get_mbin();
        const SizeType per_chan = m_nwin * 2 * mbin * sizeof(ComplexTypeGPU);
        if (work_bytes == 0) {
            std::size_t free_bytes  = 0;
            std::size_t total_bytes = 0;
            gpu_utils::check_gpu_call(cudaMemGetInfo(&free_bytes, &total_bytes),
                                      "CohFDMT: memory query failed");
            work_bytes = static_cast<SizeType>(kWorkShare *
                                               static_cast<double>(free_bytes));
        }
        // Channels per chunk: the grid's y extent (<= kMaxGridY) and the
        // work buffer budget.
        m_chunk = std::clamp<SizeType>(
            work_bytes / per_chan, 1,
            std::min<SizeType>(nchans, fourier_gpu::kMaxGridY));
        m_inv = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kC2CBackward, mbin, m_chunk * m_nwin * 2,
            m_device_id);
        m_work.reserve(m_chunk * m_nwin * 2 * mbin);
    }

    // Bytes of subbands [s0, s1) of group g when its subbands are
    // contiguous (frequency is the outermost axis).
    [[nodiscard]] std::pair<SizeType, SizeType> group_bytes(SizeType s0,
                                                            SizeType s1) const {
        const auto per_sub = utils::baseband_block_bytes(
            m_plan.get_format(), m_plan.get_block_nsamps(), 1);
        return {s0 * per_sub, (s1 - s0) * per_sub};
    }

    // Host bytes each front-end chunk needs that earlier chunks did not
    // upload (groups[g] -> m_raw[g]); seg_chunk[i] is segment i's chunk.
    void upload_plan(std::span<const std::span<const uint8_t>> h_groups,
                     std::vector<gpu_host::ChunkedStager::Segment>& segs,
                     std::vector<SizeType>& seg_chunk) const {
        const auto& sizes = m_plan.get_subband_groups();
        std::vector<bool> uploaded(sizes.size(), false);
        for (SizeType ic = 0; ic < m_chunks.size(); ++ic) {
            const auto& ch = m_chunks[ic];
            SizeType g0    = 0;
            for (SizeType g = 0; g < sizes.size(); g0 += sizes[g], ++g) {
                const SizeType lo = std::max(ch.sub_begin, g0);
                const SizeType hi = std::min(ch.sub_end, g0 + sizes[g]);
                if (lo >= hi || uploaded[g]) {
                    continue;
                }
                SizeType off   = 0;
                SizeType bytes = h_groups[g].size();
                if (m_f_outer[g]) {
                    std::tie(off, bytes) = group_bytes(lo - g0, hi - g0);
                    uploaded[g]          = hi == g0 + sizes[g];
                } else {
                    uploaded[g] = true; // whole group with its first chunk
                }
                segs.push_back(
                    {m_raw[g].data() + off, h_groups[g].data() + off, bytes});
                seg_chunk.push_back(ic);
            }
        }
    }

    // Unpack -> forward FFT -> channel powers of chunk ic.
    void front_chunk(SizeType ic, cudaStream_t stream) {
        const auto n_p  = m_plan.get_n_p();
        const auto nfft = m_plan.get_nfft();
        const auto nbin = m_plan.get_nbin();
        const auto& ch  = m_chunks[ic];
        const cuda::std::span<ComplexTypeGPU> spec(m_spec.data(),
                                                   m_unpacker.output_size());
        m_unpacker.unpack(spec, ch.sub_begin, ch.sub_end, stream);
        const auto rows = SizeType{2} * nfft * nbin;
        ch.fwd->execute(spec.subspan(ch.sub_begin * rows,
                                     (ch.sub_end - ch.sub_begin) * rows),
                        stream);
        if (m_plan.get_config().normalize) {
            const auto ncells = (ch.sub_end - ch.sub_begin) * nfft * n_p;
            channel_power_kernel<<<static_cast<unsigned>(ncells), kBlock, 0,
                                   stream>>>(
                reinterpret_cast<const float2*>(m_spec.data()), m_taper.data(),
                ch.sub_begin * nfft * n_p, n_p, nbin, m_plan.get_mbin(),
                m_power.data());
        }
    }

    // Front end chunk by chunk, then the channel statistics. On a host
    // call, the upload runs on the copy stream and each chunk's front end
    // is enqueued as soon as its bytes are, so it overlaps the upload of
    // the chunks after it.
    void front_end(std::span<const cuda::std::span<const uint8_t>> d_groups,
                   std::span<const std::span<const uint8_t>> h_groups,
                   cudaStream_t stream) {
        m_unpacker.prepare(d_groups, stream);
        if (h_groups.empty()) {
            for (SizeType ic = 0; ic < m_chunks.size(); ++ic) {
                front_chunk(ic, stream);
            }
        } else {
            std::vector<gpu_host::ChunkedStager::Segment> segs;
            std::vector<SizeType> seg_chunk;
            upload_plan(h_groups, segs, seg_chunk);
            // The copies reuse m_raw: order them after earlier device work.
            m_fence.order(m_copy_stream);
            SizeType next           = 0; // next chunk to enqueue
            const auto enqueue_upto = [&](SizeType last) {
                for (; next <= last; ++next) {
                    gpu_utils::check_gpu_call(
                        cudaEventRecord(m_chunk_ready[next], m_copy_stream),
                        "CohFDMT: event record failed");
                    gpu_utils::check_gpu_call(
                        cudaStreamWaitEvent(stream, m_chunk_ready[next], 0),
                        "CohFDMT: stream wait failed");
                    front_chunk(next, stream);
                }
            };
            // Pageable copies: the driver stages them as fast as host-side
            // staging threads into pinned buffers here (see
            // gpu_host::ChunkedStager), and each call returns once its
            // bytes are staged, so the front end of the chunks before it
            // runs meanwhile.
            for (SizeType i = 0; i < segs.size(); ++i) {
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(segs[i].dst, segs[i].src, segs[i].bytes,
                                    cudaMemcpyHostToDevice, m_copy_stream),
                    "CohFDMT: input upload failed");
                if (i + 1 == segs.size() || seg_chunk[i + 1] != seg_chunk[i]) {
                    enqueue_upto(seg_chunk[i]);
                }
            }
            if (m_chunks.size() > next) {
                enqueue_upto(m_chunks.size() - 1);
            }
        }
        if (m_plan.get_config().normalize) {
            channel_stats_kernel<<<grid_for(m_plan.get_nchans()), kBlock, 0,
                                   stream>>>(
                m_power.data(), m_plan.get_nchans(), m_plan.get_n_p(),
                m_plan.get_nfft(), m_mean.data(), m_inv_sigma.data());
        }
        gpu_utils::check_last_gpu_error("CohFDMT: front end");
    }

    void coherent_fused(SizeType k, cudaStream_t stream) {
        auto args      = m_fused_args;
        args.spec      = reinterpret_cast<const float2*>(m_spec.data());
        args.chirp     = reinterpret_cast<const float2*>(m_chirp.data());
        args.twiddle   = reinterpret_cast<const float2*>(m_twiddle.data());
        args.shifts    = m_shifts.data() + (k * m_plan.get_nchans());
        args.mean      = m_mean.data();
        args.inv_sigma = m_inv_sigma.data();
        args.waterfall = m_waterfall.data();
        args.nchans    = m_plan.get_nchans();
        args.n_p       = m_plan.get_n_p();
        args.nfft      = m_plan.get_nfft();
        args.nbin      = m_plan.get_nbin();
        args.novc      = m_plan.get_noverlap() / m_plan.get_n_p();
        args.lc        = static_cast<int64_t>(lc());
        args.a         = m_plan.get_fdmt_window_start();
        args.nf        = static_cast<int64_t>(m_plan.get_fdmt_nsamps());
        args.msamp     = static_cast<int64_t>(m_plan.get_msamp());
        args.norm      = m_plan.get_config().normalize;
        m_fused<<<static_cast<unsigned>(m_fused_ctas), cfdmt_gpu::kFusedThreads,
                  m_fused_smem, stream>>>(args);
    }

    void coherent_cufft(SizeType k, cudaStream_t stream) {
        const auto nchans = m_plan.get_nchans();
        const auto nfft   = m_plan.get_nfft();
        const auto nbin   = m_plan.get_nbin();
        const auto mbin   = m_plan.get_mbin();
        const auto n_p    = m_plan.get_n_p();
        const auto novc   = m_plan.get_noverlap() / n_p;
        const auto l      = static_cast<int64_t>(lc());
        const auto nf     = static_cast<int64_t>(m_plan.get_fdmt_nsamps());
        const auto msamp  = static_cast<int64_t>(m_plan.get_msamp());
        const auto a    = static_cast<int64_t>(m_plan.get_fdmt_window_start());
        const bool norm = m_plan.get_config().normalize;
        const int64_t* shifts = m_shifts.data() + (k * nchans);
        auto* work            = reinterpret_cast<float2*>(m_work.data());
        const cuda::std::span<ComplexTypeGPU> work_span(
            m_work.data(), m_chunk * m_nwin * 2 * mbin);
        for (SizeType c0 = 0; c0 < nchans; c0 += m_chunk) {
            const dim3 g_gather(
                static_cast<unsigned>((mbin + kBlock - 1) / kBlock),
                static_cast<unsigned>(m_chunk));
            gather_chirp_kernel<<<g_gather, kBlock, 0, stream>>>(
                reinterpret_cast<const float2*>(m_spec.data()),
                reinterpret_cast<const float2*>(m_chirp.data()), shifts, c0,
                nchans, n_p, nfft, nbin, mbin, m_nwin, a, nf, msamp, l, work);
            m_inv->execute(work_span, stream);
            const dim3 g_detect(
                static_cast<unsigned>((static_cast<uint64_t>(nf) + kBlock - 1) /
                                      kBlock),
                static_cast<unsigned>(std::min(m_chunk, nchans - c0)));
            detect_kernel<<<g_detect, kBlock, 0, stream>>>(
                work, shifts, m_mean.data(), m_inv_sigma.data(), norm, c0,
                nchans, mbin, m_nwin, novc, a, nf, msamp, l,
                m_waterfall.data());
        }
    }

    void run(std::span<const cuda::std::span<const uint8_t>> d_groups,
             std::span<const std::span<const uint8_t>> h_groups,
             float* out,
             cudaStream_t stream) {
        const auto nchans = m_plan.get_nchans();
        const auto mbin   = m_plan.get_mbin();
        const auto nf     = m_plan.get_fdmt_nsamps();
        const auto nfine  = m_plan.get_ndm_fine();
        const auto nout   = m_plan.get_output_nsamps();

        front_end(d_groups, h_groups, stream);
        for (SizeType k = 0; k < m_plan.get_ndm_coh(); ++k) {
            cfdmt_gpu::
                chirp_kernel<<<grid_for(nchans * mbin), kBlock, 0, stream>>>(
                    m_base.data(), m_inc.data(), m_taper.data(),
                    static_cast<unsigned long long>(k), nchans, mbin,
                    reinterpret_cast<float2*>(m_chirp.data()));
            if (m_fused != nullptr) {
                coherent_fused(k, stream);
            } else {
                coherent_cufft(k, stream);
            }
            gpu_utils::check_last_gpu_error("CohFDMT: coherent stage");
            // No reset_history(): the FDMT window's lead-in keeps its history
            // out of every cropped output (see CohFDMTPlan).
            m_fdmt.execute(
                DeviceSpan<const float>(m_waterfall.data(), nchans * nf),
                DeviceSpan<float>(m_fdmt_out.data(),
                                  m_fdmt.get_plan().get_buffer_size()),
                Stream{stream});
            crop_kernel<<<grid_for(nfine * nout), kBlock, 0, stream>>>(
                m_fdmt_out.data(), m_offsets.data(), nfine, nf, nout,
                out + (k * nfine * nout));
            gpu_utils::check_last_gpu_error("CohFDMT: crop");
        }
        if (!h_groups.empty()) {
            // The copy stream's uploads are done once the front end is.
            m_fence.mark(stream);
        }
    }
}; // End CohFDMTCudaEngine definition

} // namespace

std::unique_ptr<detail::CohFDMTEngine>
detail::make_cfdmt_gpu(const plans::CohFDMTPlan& plan,
                       const detail::CohFDMTEngineConfig& cfg) {
    return std::make_unique<CohFDMTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
