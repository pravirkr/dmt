#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/cfdmt_common.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/fourier_gpu.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/unpacker_cuda.cuh"

// GPU engine: the CPU engine's pipeline (lib/cpu/cfdmt_cpu.cpp) as kernels
// on one stream. Per block:
//   unpack -> batched forward cuFFT -> channel noise statistics ->
//   per coarse trial, in channel chunks: gather x chirp (exact fixed-point
//   phases, identical to the CPU) -> batched inverse cuFFT -> detect,
//   normalise, trim and write the delay-aligned waterfall -> fine FDMT ->
//   crop of the valid output window.

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
    return static_cast<unsigned>(std::clamp<uint64_t>(
        (n + kBlock - 1) / kBlock, 1, uint64_t{1} << 20));
}

// Per (subband, FFT block, channel): filtered power of each polarisation,
// sum_b |c_b|^2 |X_p(b)|^2 (see the CPU engine's front_end()).
__global__ void channel_power_kernel(const float2* __restrict__ spec,
                                     const float* __restrict__ taper,
                                     uint64_t n_p,
                                     uint64_t nbin,
                                     uint64_t mbin,
                                     float* __restrict__ power) {
    __shared__ float red[2][kBlock];
    const uint64_t cell  = blockIdx.x; // (s * nfft + j) * n_p + ichan
    const uint64_t ichan = cell % n_p;
    const uint64_t task  = cell / n_p;
    const float2* row0   = spec + (task * 2 * nbin);
    const float2* row1   = row0 + nbin;
    const uint64_t off   = ((ichan * mbin) + (nbin / 2)) % nbin;
    float acc0           = 0.0F;
    float acc1           = 0.0F;
    for (uint64_t b = threadIdx.x; b < mbin; b += blockDim.x) {
        uint64_t idx = off + b;
        idx          = idx >= nbin ? idx - nbin : idx;
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
    inv_sigma[c] = sigma > 0.0 ? static_cast<float>(1.0 / sigma) : 0.0F;
}

// Channel c's window of channel samples [m_lo, m_hi) and first FFT block.
struct Window {
    int64_t m_lo;
    int64_t m_hi;
    int64_t j_first;
};

__device__ __forceinline__ Window channel_window(int64_t a,
                                                 int64_t nf,
                                                 int64_t msamp,
                                                 int64_t shift,
                                                 int64_t lc) {
    Window w{};
    const int64_t lo = a - shift;
    const int64_t hi = a + nf - shift;
    w.m_lo           = lo > 0 ? lo : 0;
    w.m_hi           = hi < msamp ? hi : msamp;
    w.j_first = w.m_lo / lc;
    return w;
}

// Thread per (channel of the chunk, bin): the trial's chirp once, applied to
// the channel's bins of every FFT block of its window (both pols).
__global__ void gather_chirp_kernel(const float2* __restrict__ spec,
                                    const unsigned long long* __restrict__ base,
                                    const unsigned long long* __restrict__ inc,
                                    const float* __restrict__ taper,
                                    const int64_t* __restrict__ shifts,
                                    unsigned long long k,
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
    const uint64_t b  = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                        threadIdx.x;
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
    const auto w = channel_window(a, nf, msamp, shifts[c], lc);
    const unsigned long long ph = base[(c * mbin) + b] + (k * inc[(c * mbin) + b]);
    float sn = 0.0F;
    float cs = 0.0F;
    sincospif(2.0F * static_cast<float>(static_cast<long long>(ph)) *
                  5.42101086242752217e-20F,
              &sn, &cs);
    const float2 h = {taper[b] * cs, taper[b] * sn};
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

// Thread per (window sample u, channel of the chunk): detected (and
// normalised) intensity at aligned time a + u.
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
    const uint64_t u  = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                        threadIdx.x;
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
    for (uint64_t idx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                        threadIdx.x;
         idx < total;
         idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t r = idx / nout;
        const uint64_t t = idx % nout;
        out[idx]         = fdmt_out[(r * nf) + offsets[r] + t];
    }
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
        initialise();
    }

    void execute(std::span<const std::span<const uint8_t>> groups,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        if (dmt.size() < m_plan.get_dmt_size()) {
            throw std::invalid_argument(std::format(
                "CohFDMT (gpu): dmt buffer too small. Expected at least {} "
                "(get_dmt_size()), got {}",
                m_plan.get_dmt_size(), dmt.size()));
        }
        const auto& sizes = m_plan.get_subband_groups();
        if (groups.size() != sizes.size()) {
            throw std::invalid_argument(
                std::format("CohFDMT: expected {} input group(s), got {}",
                            sizes.size(), groups.size()));
        }
        cudaStream_t stream = nullptr;
        m_raw.resize(groups.size());
        std::vector<cuda::std::span<const uint8_t>> d_groups;
        for (SizeType g = 0; g < groups.size(); ++g) {
            if (groups[g].size() != m_plan.get_input_size(g)) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: input group {} has {} bytes, expected {}", g,
                    groups[g].size(), m_plan.get_input_size(g)));
            }
            m_raw[g].reserve(groups[g].size());
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_raw[g].data(), groups[g].data(),
                                groups[g].size(), cudaMemcpyHostToDevice,
                                stream),
                "CohFDMT: input upload failed");
            d_groups.emplace_back(m_raw[g].data(), groups[g].size());
        }
        m_out.reserve(m_plan.get_dmt_size());
        run(d_groups, m_out.data(), stream);
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(dmt.data(), m_out.data(),
                            m_plan.get_dmt_size() * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "CohFDMT: output download failed");
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream),
                                  "CohFDMT: execute failed");
    }

    void execute(std::span<const DeviceSpan<const uint8_t>> groups,
                 DeviceSpan<float> dmt,
                 Stream stream) override {
        gpu_utils::set_device(m_device_id);
        if (dmt.size() < m_plan.get_dmt_size()) {
            throw std::invalid_argument(std::format(
                "CohFDMT (gpu): dmt buffer too small. Expected at least {} "
                "(get_dmt_size()), got {}",
                m_plan.get_dmt_size(), dmt.size()));
        }
        std::vector<cuda::std::span<const uint8_t>> d_groups;
        d_groups.reserve(groups.size());
        for (const auto& g : groups) {
            d_groups.push_back(to_cuda(g));
        }
        run(d_groups, to_cuda(dmt).data(), to_cuda(stream));
    }

    [[nodiscard]] plans::CohFDMTMemoryUsage
    memory_usage() const noexcept override {
        const auto fdmt_mem = m_fdmt.get_memory_usage();
        return {.spectrum  = m_spec.capacity() * sizeof(ComplexTypeGPU),
                .waterfall = m_waterfall.capacity() * sizeof(float),
                .fdmt      = fdmt_mem.total() +
                        (m_fdmt_out.capacity() * sizeof(float)),
                .workspace = (m_work.capacity() * sizeof(ComplexTypeGPU)) +
                             m_fwd->workspace_bytes() +
                             m_inv->workspace_bytes() +
                             ((m_base.capacity() + m_inc.capacity()) *
                              sizeof(unsigned long long)) +
                             (m_power.capacity() * sizeof(float)),
                .output    = m_plan.get_dmt_size() * sizeof(float)};
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    // Inverse-FFT work buffer budget per channel chunk.
    static constexpr SizeType kWorkBytes = SizeType{256} << 20U;

    const plans::CohFDMTPlan& m_plan; // owned by the CohFDMT facade
    int m_device_id;
    utils::BasebandUnpackerCUDA m_unpacker;
    algorithms::FDMT m_fdmt;
    std::unique_ptr<utils::CUFFTManager> m_fwd;
    std::unique_ptr<utils::CUFFTManager> m_inv;
    SizeType m_nwin{};
    SizeType m_chunk{};

    DevBuf<ComplexTypeGPU> m_spec; // (sub, ifft, pol, nbin)
    DevBuf<ComplexTypeGPU> m_work; // (chunk chan, window block, pol, mbin)
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
    DevBuf<float> m_out; // host entry point only
    std::vector<DevBuf<uint8_t>> m_raw;

    void initialise() {
        gpu_utils::set_device(m_device_id);
        const auto nchans = m_plan.get_nchans();
        const auto nsub   = m_plan.get_nsub();
        const auto nfft   = m_plan.get_nfft();
        const auto nbin   = m_plan.get_nbin();
        const auto mbin   = m_plan.get_mbin();
        const auto n_p    = m_plan.get_n_p();
        const auto novc   = m_plan.get_noverlap() / n_p;
        const auto lc     = mbin - (2 * novc);
        m_nwin  = ((m_plan.get_fdmt_nsamps() + lc - 1) / lc) + 1;
        // Channels per chunk: the grid's y extent (<= kMaxGridY) and the
        // work buffer budget.
        m_chunk = std::clamp<SizeType>(
            kWorkBytes / (m_nwin * 2 * mbin * sizeof(ComplexTypeGPU)), 1,
            std::min<SizeType>(nchans, fourier_gpu::kMaxGridY));

        m_fwd = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kC2CForward, nbin, nsub * nfft * 2, m_device_id);
        m_inv = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kC2CBackward, mbin, m_chunk * m_nwin * 2,
            m_device_id);
        m_spec.reserve(m_unpacker.output_size());
        m_work.reserve(m_chunk * m_nwin * 2 * mbin);
        m_power.reserve(nsub * nfft * n_p * 2);
        m_waterfall.reserve(nchans * m_plan.get_fdmt_nsamps());
        m_fdmt_out.reserve(m_fdmt.get_plan().get_buffer_size());

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
    }

    void run(std::span<const cuda::std::span<const uint8_t>> groups,
             float* out,
             cudaStream_t stream) {
        const auto nchans = m_plan.get_nchans();
        const auto nsub   = m_plan.get_nsub();
        const auto nfft   = m_plan.get_nfft();
        const auto nbin   = m_plan.get_nbin();
        const auto mbin   = m_plan.get_mbin();
        const auto n_p    = m_plan.get_n_p();
        const auto novc   = m_plan.get_noverlap() / n_p;
        const auto lc     = static_cast<int64_t>(mbin - (2 * novc));
        const auto nf     = static_cast<int64_t>(m_plan.get_fdmt_nsamps());
        const auto msamp  = static_cast<int64_t>(m_plan.get_msamp());
        const auto a      = static_cast<int64_t>(m_plan.get_fdmt_window_start());
        const bool norm   = m_plan.get_config().normalize;
        const auto nfine  = m_plan.get_ndm_fine();
        const auto nout   = m_plan.get_output_nsamps();
        auto* spec        = reinterpret_cast<float2*>(m_spec.data());
        auto* work        = reinterpret_cast<float2*>(m_work.data());

        m_unpacker.execute(groups,
                           cuda::std::span<ComplexTypeGPU>(
                               m_spec.data(), m_unpacker.output_size()),
                           stream);
        m_fwd->execute(cuda::std::span<ComplexTypeGPU>(
                           m_spec.data(), m_unpacker.output_size()),
                       stream);
        if (norm) {
            const auto ncells = nsub * nfft * n_p;
            channel_power_kernel<<<static_cast<unsigned>(ncells), kBlock, 0,
                                   stream>>>(spec, m_taper.data(), n_p, nbin,
                                             mbin, m_power.data());
            channel_stats_kernel<<<grid_for(nchans), kBlock, 0, stream>>>(
                m_power.data(), nchans, n_p, nfft, m_mean.data(),
                m_inv_sigma.data());
            gpu_utils::check_last_gpu_error("CohFDMT: channel statistics");
        }
        const cuda::std::span<ComplexTypeGPU> work_span(
            m_work.data(), m_chunk * m_nwin * 2 * mbin);
        for (SizeType k = 0; k < m_plan.get_ndm_coh(); ++k) {
            const int64_t* shifts = m_shifts.data() + (k * nchans);
            for (SizeType c0 = 0; c0 < nchans; c0 += m_chunk) {
                const dim3 g_gather(
                    static_cast<unsigned>((mbin + kBlock - 1) / kBlock),
                    static_cast<unsigned>(m_chunk));
                gather_chirp_kernel<<<g_gather, kBlock, 0, stream>>>(
                    spec, m_base.data(), m_inc.data(), m_taper.data(), shifts,
                    static_cast<unsigned long long>(k), c0, nchans, n_p, nfft,
                    nbin, mbin, m_nwin, a, nf, msamp, lc, work);
                m_inv->execute(work_span, stream);
                const dim3 g_detect(
                    static_cast<unsigned>((static_cast<uint64_t>(nf) + kBlock -
                                           1) /
                                          kBlock),
                    static_cast<unsigned>(std::min(m_chunk, nchans - c0)));
                detect_kernel<<<g_detect, kBlock, 0, stream>>>(
                    work, shifts, m_mean.data(), m_inv_sigma.data(), norm, c0,
                    nchans, mbin, m_nwin, novc, a, nf, msamp, lc,
                    m_waterfall.data());
                gpu_utils::check_last_gpu_error("CohFDMT: coherent stage");
            }
            // No reset_history(): the FDMT window's lead-in keeps its history
            // out of every cropped output (see CohFDMTPlan).
            m_fdmt.execute(
                DeviceSpan<const float>(m_waterfall.data(),
                                        nchans * static_cast<SizeType>(nf)),
                DeviceSpan<float>(m_fdmt_out.data(),
                                  m_fdmt.get_plan().get_buffer_size()),
                Stream{stream});
            crop_kernel<<<grid_for(nfine * nout), kBlock, 0, stream>>>(
                m_fdmt_out.data(), m_offsets.data(), nfine,
                static_cast<uint64_t>(nf), nout, out + (k * nfine * nout));
            gpu_utils::check_last_gpu_error("CohFDMT: crop");
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
