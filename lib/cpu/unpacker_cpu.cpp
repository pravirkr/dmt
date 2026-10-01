#include "dmt/unpacker.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <vector>

#include "dmt/baseband_layout.hpp"

namespace dmt::utils {

namespace {

// 8-bit decoding by arithmetic (vectorises; no table gather).
template <bool Signed> inline float decode8(uint8_t b) noexcept {
    if constexpr (Signed) {
        return static_cast<float>(static_cast<int8_t>(b));
    } else {
        return static_cast<float>(b) - 128.0F;
    }
}

} // namespace

class BasebandUnpackerCPU::Impl {
public:
    Impl(const BasebandFormat& format,
         std::span<const SizeType> subband_groups,
         SizeType nbin,
         SizeType nfft,
         SizeType noverlap,
         int nthreads)
        : m_format(format),
          m_groups(subband_groups.begin(), subband_groups.end()),
          m_nbin(nbin),
          m_nfft(nfft),
          m_noverlap(noverlap),
          m_nthreads(std::max(1, nthreads)),
          m_table(make_decode_table(format)) {
        if (m_groups.empty() || nbin <= 2 * noverlap || nfft == 0) {
            throw std::invalid_argument(std::format(
                "BasebandUnpackerCPU: invalid geometry (groups={}, nbin={}, "
                "noverlap={}, nfft={})",
                m_groups.size(), nbin, noverlap, nfft));
        }
        m_step          = m_nbin - (2 * m_noverlap);
        m_nsamps        = (m_nfft * m_step) + (2 * m_noverlap);
        SizeType offset = 0;
        for (const auto g : m_groups) {
            for (SizeType s = 0; s < g; ++s) {
                m_sub_group.push_back(m_group_offset.size());
            }
            m_group_offset.push_back(offset);
            m_strides.push_back(baseband_strides(m_format, m_nsamps, g));
            offset += g;
        }
        m_nsub = offset;
    }

    SizeType block_nsamps() const noexcept { return m_nsamps; }
    SizeType input_size(SizeType igroup) const {
        return baseband_block_bytes(m_format, m_nsamps, m_groups.at(igroup));
    }
    SizeType output_size() const noexcept {
        return m_nsub * m_nfft * 2 * m_nbin;
    }

    void validate(std::span<const std::span<const uint8_t>> groups) const {
        if (groups.size() != m_groups.size()) {
            throw std::invalid_argument(
                std::format("CohFDMT: expected {} input group(s), got {}",
                            m_groups.size(), groups.size()));
        }
        for (SizeType g = 0; g < groups.size(); ++g) {
            if (groups[g].size() != input_size(g)) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: input group {} has {} bytes, expected {} "
                    "(block_nsamps={} x {} subbands x 4 x {} bits / 8)",
                    g, groups[g].size(), input_size(g), m_nsamps, m_groups[g],
                    m_format.nbits));
            }
        }
    }

    void unpack_block(std::span<const std::span<const uint8_t>> groups,
                      SizeType isub,
                      SizeType ifft,
                      ComplexType* pol0,
                      ComplexType* pol1) const noexcept {
        const SizeType g    = m_sub_group[isub];
        const auto& st      = m_strides[g];
        const uint8_t* data = groups[g].data();
        const SizeType base =
            ((isub - m_group_offset[g]) * st.freq) + (ifft * m_step * st.time);
        const bool interleaved = st.ri == 1 && st.pol == 2 && st.time == 4;
        if (m_format.nbits == 8) {
            if (interleaved) {
                // T-inner, (P, RI) interleaved (GUPPI FTPRI): 4 bytes per
                // sample, contiguous in time.
                if (m_format.is_signed) {
                    interleaved8<true>(data + base, pol0, pol1);
                } else {
                    interleaved8<false>(data + base, pol0, pol1);
                }
            } else if (m_format.is_signed) {
                strided8<true>(data, base, st, pol0, pol1);
            } else {
                strided8<false>(data, base, st, pol0, pol1);
            }
            return;
        }
        if (m_table.per_byte == 2) {
            strided_lut<2>(data, base, st, pol0, pol1);
        } else {
            strided_lut<4>(data, base, st, pol0, pol1);
        }
    }

    void execute(std::span<const std::span<const uint8_t>> groups,
                 std::span<ComplexType> out) const {
        validate(groups);
        if (out.size() != output_size()) {
            throw std::invalid_argument(std::format(
                "BasebandUnpackerCPU: output has {} elements, expected {}",
                out.size(), output_size()));
        }
        const auto nrows     = m_nsub * m_nfft;
        ComplexType* out_ptr = out.data();
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType row = 0; row < nrows; ++row) {
            ComplexType* dst = out_ptr + (row * 2 * m_nbin);
            unpack_block(groups, row / m_nfft, row % m_nfft, dst, dst + m_nbin);
        }
    }

private:
    BasebandFormat m_format;
    std::vector<SizeType> m_groups;
    std::vector<SizeType> m_group_offset;
    std::vector<SizeType> m_sub_group;
    std::vector<BasebandStrides> m_strides;
    SizeType m_nbin;
    SizeType m_nfft;
    SizeType m_noverlap;
    SizeType m_step{};
    SizeType m_nsamps{};
    SizeType m_nsub{};
    int m_nthreads;
    BasebandDecodeTable m_table;

    template <bool Signed>
    void interleaved8(const uint8_t* __restrict__ src,
                      ComplexType* __restrict__ pol0,
                      ComplexType* __restrict__ pol1) const noexcept {
        auto* d0 = reinterpret_cast<float*>(pol0);
        auto* d1 = reinterpret_cast<float*>(pol1);
#pragma omp simd
        for (SizeType i = 0; i < m_nbin; ++i) {
            d0[2 * i]       = decode8<Signed>(src[4 * i]);
            d0[(2 * i) + 1] = decode8<Signed>(src[(4 * i) + 1]);
            d1[2 * i]       = decode8<Signed>(src[(4 * i) + 2]);
            d1[(2 * i) + 1] = decode8<Signed>(src[(4 * i) + 3]);
        }
    }

    template <bool Signed>
    void strided8(const uint8_t* __restrict__ data,
                  SizeType base,
                  const BasebandStrides& st,
                  ComplexType* __restrict__ pol0,
                  ComplexType* __restrict__ pol1) const noexcept {
        auto* d0          = reinterpret_cast<float*>(pol0);
        auto* d1          = reinterpret_cast<float*>(pol1);
        const uint8_t* p0 = data + base;
        const uint8_t* p1 = data + base + st.pol;
        const SizeType ts = st.time;
        const SizeType ri = st.ri;
        for (SizeType i = 0; i < m_nbin; ++i) {
            d0[2 * i]       = decode8<Signed>(p0[i * ts]);
            d0[(2 * i) + 1] = decode8<Signed>(p0[(i * ts) + ri]);
            d1[2 * i]       = decode8<Signed>(p1[i * ts]);
            d1[(2 * i) + 1] = decode8<Signed>(p1[(i * ts) + ri]);
        }
    }

    template <SizeType PerByte>
    void strided_lut(const uint8_t* __restrict__ data,
                     SizeType base,
                     const BasebandStrides& st,
                     ComplexType* __restrict__ pol0,
                     ComplexType* __restrict__ pol1) const noexcept {
        const auto& v = m_table.value;
        const auto at = [&](SizeType e) {
            return v[data[e / PerByte]][e % PerByte];
        };
        for (SizeType i = 0; i < m_nbin; ++i) {
            const SizeType e0 = base + (i * st.time);
            const SizeType e1 = e0 + st.pol;
            pol0[i]           = {at(e0), at(e0 + st.ri)};
            pol1[i]           = {at(e1), at(e1 + st.ri)};
        }
    }
};

BasebandUnpackerCPU::BasebandUnpackerCPU(
    const BasebandFormat& format,
    std::span<const SizeType> subband_groups,
    SizeType nbin,
    SizeType nfft,
    SizeType noverlap,
    int nthreads)
    : m_impl(std::make_unique<Impl>(
          format, subband_groups, nbin, nfft, noverlap, nthreads)) {}
BasebandUnpackerCPU::~BasebandUnpackerCPU() = default;
BasebandUnpackerCPU::BasebandUnpackerCPU(BasebandUnpackerCPU&&) noexcept =
    default;
BasebandUnpackerCPU&
BasebandUnpackerCPU::operator=(BasebandUnpackerCPU&&) noexcept = default;
SizeType BasebandUnpackerCPU::block_nsamps() const noexcept {
    return m_impl->block_nsamps();
}
SizeType BasebandUnpackerCPU::input_size(SizeType igroup) const {
    return m_impl->input_size(igroup);
}
SizeType BasebandUnpackerCPU::output_size() const noexcept {
    return m_impl->output_size();
}
void BasebandUnpackerCPU::validate(
    std::span<const std::span<const uint8_t>> groups) const {
    m_impl->validate(groups);
}
void BasebandUnpackerCPU::unpack_block(
    std::span<const std::span<const uint8_t>> groups,
    SizeType isub,
    SizeType ifft,
    ComplexType* pol0,
    ComplexType* pol1) const noexcept {
    m_impl->unpack_block(groups, isub, ifft, pol0, pol1);
}
void BasebandUnpackerCPU::execute(
    std::span<const std::span<const uint8_t>> groups,
    std::span<ComplexType> out) const {
    m_impl->execute(groups, out);
}

} // namespace dmt::utils
