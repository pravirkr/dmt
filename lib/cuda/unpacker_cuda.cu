#include "dmt/unpacker_cuda.cuh"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

#include "dmt/baseband_layout.hpp"
#include "dmt/fourier_gpu.cuh"

namespace dmt::utils {

namespace {

// Where one subband's elements live: element e of (pol, ri, t) is
// data[e] (8-bit) or a field of data[e / per_byte].
struct SubbandDesc {
    const uint8_t* data;
    uint64_t base;   // subband offset (elements)
    uint64_t pol;    // element strides
    uint64_t ri;
    uint64_t time;
};

__device__ __forceinline__ float decode(const uint8_t* data,
                                        uint64_t e,
                                        unsigned nbits,
                                        bool is_signed,
                                        const float* lut) {
    if (nbits == 8) {
        const uint8_t b = data[e];
        return is_signed ? static_cast<float>(static_cast<int8_t>(b))
                         : static_cast<float>(b) - 128.0F;
    }
    const unsigned per = 8U / nbits;
    return lut[(static_cast<unsigned>(data[e / per]) * 4U) +
               static_cast<unsigned>(e % per)];
}

// One thread per (subband, FFT block, sample) of subbands [sub_begin,
// sub_begin + nsub): both polarisations.
__global__ void unpack_kernel(const SubbandDesc* __restrict__ desc,
                              const float* __restrict__ lut,
                              unsigned nbits,
                              bool is_signed,
                              uint64_t sub_begin,
                              uint64_t nsub,
                              uint64_t nfft,
                              uint64_t nbin,
                              uint64_t step,
                              float2* __restrict__ out) {
    const uint64_t total = nsub * nfft * nbin;
    for (uint64_t idx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) +
                        threadIdx.x;
         idx < total;
         idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const uint64_t i   = idx % nbin;
        const uint64_t row = (idx / nbin) + (sub_begin * nfft); // s * nfft + j
        const uint64_t j   = row % nfft;
        const uint64_t s   = row / nfft;
        const SubbandDesc d = desc[s];
        const uint64_t e0   = d.base + (((j * step) + i) * d.time);
        const uint64_t e1   = e0 + d.pol;
        float2* dst         = out + (row * 2 * nbin) + i;
        dst[0]    = {decode(d.data, e0, nbits, is_signed, lut),
                     decode(d.data, e0 + d.ri, nbits, is_signed, lut)};
        dst[nbin] = {decode(d.data, e1, nbits, is_signed, lut),
                     decode(d.data, e1 + d.ri, nbits, is_signed, lut)};
    }
}

} // namespace

class BasebandUnpackerCUDA::Impl {
public:
    Impl(const BasebandFormat& format,
         std::span<const SizeType> subband_groups,
         SizeType nbin,
         SizeType nfft,
         SizeType noverlap,
         int device_id)
        : m_format(format),
          m_groups(subband_groups.begin(), subband_groups.end()),
          m_nbin(nbin),
          m_nfft(nfft),
          m_noverlap(noverlap),
          m_device_id(device_id) {
        if (m_groups.empty() || nbin <= 2 * noverlap || nfft == 0) {
            throw std::invalid_argument(std::format(
                "BasebandUnpackerCUDA: invalid geometry (groups={}, nbin={}, "
                "noverlap={}, nfft={})",
                m_groups.size(), nbin, noverlap, nfft));
        }
        gpu_utils::set_device(m_device_id);
        m_step   = m_nbin - (2 * m_noverlap);
        m_nsamps = (m_nfft * m_step) + (2 * m_noverlap);
        m_nsub   = 0;
        for (const auto g : m_groups) {
            m_nsub += g;
        }
        const auto table = make_decode_table(m_format);
        std::vector<float> lut(256 * 4);
        for (SizeType b = 0; b < 256; ++b) {
            for (SizeType k = 0; k < 4; ++k) {
                lut[(b * 4) + k] = table.value[b][k];
            }
        }
        m_lut.upload(lut);
        m_desc_h.resize(m_nsub);
        m_desc.reserve(m_nsub);
    }

    SizeType block_nsamps() const noexcept { return m_nsamps; }
    SizeType input_size(SizeType igroup) const {
        return baseband_block_bytes(m_format, m_nsamps, m_groups.at(igroup));
    }
    SizeType output_size() const noexcept {
        return m_nsub * m_nfft * 2 * m_nbin;
    }

    void execute(std::span<const cuda::std::span<const uint8_t>> groups,
                 cuda::std::span<ComplexTypeGPU> out,
                 cudaStream_t stream) {
        prepare(groups, stream);
        unpack(out, 0, m_nsub, stream);
    }

    void prepare(std::span<const cuda::std::span<const uint8_t>> groups,
                 cudaStream_t stream) {
        if (groups.size() != m_groups.size()) {
            throw std::invalid_argument(
                std::format("CohFDMT: expected {} input group(s), got {}",
                            m_groups.size(), groups.size()));
        }
        SizeType isub = 0;
        for (SizeType g = 0; g < groups.size(); ++g) {
            if (groups[g].size() != input_size(g)) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: input group {} has {} bytes, expected {}", g,
                    groups[g].size(), input_size(g)));
            }
            const auto st = baseband_strides(m_format, m_nsamps, m_groups[g]);
            for (SizeType s = 0; s < m_groups[g]; ++s) {
                m_desc_h[isub++] = {.data = groups[g].data(),
                                    .base = s * st.freq,
                                    .pol  = st.pol,
                                    .ri   = st.ri,
                                    .time = st.time};
            }
        }
        gpu_utils::set_device(m_device_id);
        // The descriptors hold this call's pointers; the (pageable) copy is
        // staged by the runtime before cudaMemcpyAsync returns.
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(m_desc.data(), m_desc_h.data(),
                            m_nsub * sizeof(SubbandDesc),
                            cudaMemcpyHostToDevice, stream),
            "BasebandUnpackerCUDA: descriptor upload failed");
    }

    void unpack(cuda::std::span<ComplexTypeGPU> out,
                SizeType sub_begin,
                SizeType sub_end,
                cudaStream_t stream) {
        if (out.size() != output_size()) {
            throw std::invalid_argument(std::format(
                "BasebandUnpackerCUDA: output has {} elements, expected {}",
                out.size(), output_size()));
        }
        if (sub_begin > sub_end || sub_end > m_nsub) {
            throw std::invalid_argument(std::format(
                "BasebandUnpackerCUDA: subband range [{}, {}) out of [0, {})",
                sub_begin, sub_end, m_nsub));
        }
        if (sub_begin == sub_end) {
            return;
        }
        gpu_utils::set_device(m_device_id);
        const auto nsub  = sub_end - sub_begin;
        const auto total = static_cast<int64_t>(nsub * m_nfft * m_nbin);
        const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(
            (total + fourier_gpu::kBlock - 1) / fourier_gpu::kBlock,
            int64_t{1} << 20));
        unpack_kernel<<<blocks, fourier_gpu::kBlock, 0, stream>>>(
            m_desc.data(), m_lut.data(),
            static_cast<unsigned>(m_format.nbits), m_format.is_signed,
            sub_begin, nsub, m_nfft, m_nbin, m_step,
            reinterpret_cast<float2*>(out.data()));
        gpu_utils::check_last_gpu_error("BasebandUnpackerCUDA: unpack kernel");
    }

private:
    BasebandFormat m_format;
    std::vector<SizeType> m_groups;
    SizeType m_nbin;
    SizeType m_nfft;
    SizeType m_noverlap;
    int m_device_id;
    SizeType m_step{};
    SizeType m_nsamps{};
    SizeType m_nsub{};
    fourier_gpu::DevBuf<float> m_lut;
    std::vector<SubbandDesc> m_desc_h;
    fourier_gpu::DevBuf<SubbandDesc> m_desc;
};

BasebandUnpackerCUDA::BasebandUnpackerCUDA(
    const BasebandFormat& format,
    std::span<const SizeType> subband_groups,
    SizeType nbin,
    SizeType nfft,
    SizeType noverlap,
    int device_id)
    : m_impl(std::make_unique<Impl>(
          format, subband_groups, nbin, nfft, noverlap, device_id)) {}
BasebandUnpackerCUDA::~BasebandUnpackerCUDA() = default;
BasebandUnpackerCUDA::BasebandUnpackerCUDA(BasebandUnpackerCUDA&&) noexcept =
    default;
BasebandUnpackerCUDA&
BasebandUnpackerCUDA::operator=(BasebandUnpackerCUDA&&) noexcept = default;
SizeType BasebandUnpackerCUDA::block_nsamps() const noexcept {
    return m_impl->block_nsamps();
}
SizeType BasebandUnpackerCUDA::input_size(SizeType igroup) const {
    return m_impl->input_size(igroup);
}
SizeType BasebandUnpackerCUDA::output_size() const noexcept {
    return m_impl->output_size();
}
void BasebandUnpackerCUDA::execute(
    std::span<const cuda::std::span<const uint8_t>> groups,
    cuda::std::span<ComplexTypeGPU> out,
    cudaStream_t stream) {
    m_impl->execute(groups, out, stream);
}
void BasebandUnpackerCUDA::prepare(
    std::span<const cuda::std::span<const uint8_t>> groups,
    cudaStream_t stream) {
    m_impl->prepare(groups, stream);
}
void BasebandUnpackerCUDA::unpack(cuda::std::span<ComplexTypeGPU> out,
                                  SizeType sub_begin,
                                  SizeType sub_end,
                                  cudaStream_t stream) {
    m_impl->unpack(out, sub_begin, sub_end, stream);
}

} // namespace dmt::utils
