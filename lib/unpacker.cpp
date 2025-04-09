#include "dmt/common/unpacker.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "dmt/common/types.hpp"

template <BasebandDataOrder Order>
class DataUnpackerImpl {
public:
    static constexpr SizeType kChunkSize = 64;
    static constexpr SizeType kNpol      = 2;

    DataUnpackerImpl(SizeType nsub,
                     SizeType nbin,
                     SizeType noverlap,
                     SizeType nfft)
        : m_nsub(nsub),
          m_nbin(nbin),
          m_noverlap(noverlap),
          m_nfft(nfft) {
        if (m_nbin < 2 * m_noverlap) {
            throw std::runtime_error("Invalid nbin and noverlap values");
        }
        m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);
    }

    template <IntegralDataType DataType>
    void unpack_and_padd(const DataType* __restrict__ data_in,
                         SizeType in_size,
                         ComplexType* __restrict__ data_p1,
                         ComplexType* __restrict__ data_p2,
                         SizeType out_size) const {
        validate_sizes(in_size, out_size);
#ifdef USE_OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (SizeType ifft = 0; ifft < m_nfft; ++ifft) {
            for (SizeType isub = 0; isub < m_nsub; ++isub) {
                process_chunk<DataType>(data_in, data_p1, data_p2, ifft, isub);
            }
        }
    }

private:
    SizeType m_nsub, m_nbin, m_noverlap, m_nfft, m_nsamp;

    constexpr void validate_sizes(SizeType in_size, SizeType out_size) const {
        if (in_size != 2 * kNpol * m_nsamp * m_nsub) {
            throw std::runtime_error("Invalid input size");
        }
        if (out_size != m_nfft * m_nsub * m_nbin) {
            throw std::runtime_error("Invalid output size");
        }
    }

    constexpr SizeType
    calculate_indices(SizeType isamp, SizeType ipol, SizeType isub) const {
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            return (ipol * 2 * m_nsamp * m_nsub) + (isamp * m_nsub) + isub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            return (isub * m_nsamp * kNpol * 2) + (isamp * kNpol * 2) + ipol;
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            return (isamp * m_nsub * kNpol) + (isub * kNpol) + ipol;
        }
    }

    constexpr SizeType calculate_ri_stride() const {
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            return m_nsamp * m_nsub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            return 1;
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            return m_nsamp * m_nsub * kNpol;
        }
    }

    template <IntegralDataType DataType>
    void process_chunk(const DataType* __restrict__ data_in,
                       ComplexType* __restrict__ data_p1,
                       ComplexType* __restrict__ data_p2,
                       SizeType ifft,
                       SizeType isub) const {
        const auto out_offset = (ifft * m_nsub * m_nbin) + (isub * m_nbin);
        const auto start_sample =
            static_cast<IndexType>((m_nbin - 2 * m_noverlap) * ifft) -
            static_cast<IndexType>(m_noverlap);
        const auto bin_start =
            static_cast<SizeType>(start_sample < 0 ? -start_sample : 0);
        const auto bin_end =
            static_cast<SizeType>(std::min(m_nbin, m_nsamp - start_sample));
        const auto ri_stride = calculate_ri_stride();

        for (SizeType ibin = bin_start; ibin < bin_end; ibin += kChunkSize) {
            const SizeType chunk_end = std::min(ibin + kChunkSize, bin_end);
            for (SizeType i = ibin; i < chunk_end; ++i) {
                SizeType isamp            = start_sample + i;
                const auto idx_in_base_p1 = calculate_indices(isamp, 0, isub);
                const auto idx_in_base_p2 = calculate_indices(isamp, 1, isub);
                data_p1[out_offset + i]   = ComplexType(
                    static_cast<float>(data_in[idx_in_base_p1]),
                    static_cast<float>(data_in[idx_in_base_p1 + ri_stride]));
                data_p2[out_offset + i] = ComplexType(
                    static_cast<float>(data_in[idx_in_base_p2]),
                    static_cast<float>(data_in[idx_in_base_p2 + ri_stride]));
            }
        }

        // Pad the remaining bins with zeros
        std::fill_n(data_p1 + out_offset, bin_start, ComplexType(0.0F, 0.0F));
        std::fill_n(data_p2 + out_offset, bin_start, ComplexType(0.0F, 0.0F));
        std::fill_n(data_p1 + out_offset + bin_end, m_nbin - bin_end,
                    ComplexType(0.0F, 0.0F));
        std::fill_n(data_p2 + out_offset + bin_end, m_nbin - bin_end,
                    ComplexType(0.0F, 0.0F));
    }
};

template <BasebandDataOrder Order>
class DataUnpackerImplWrapper : public DataUnpackerImplBase {
    DataUnpackerImpl<Order> m_impl;

public:
    DataUnpackerImplWrapper(SizeType nsub,
                            SizeType nbin,
                            SizeType noverlap,
                            SizeType nfft)
        : m_impl(nsub, nbin, noverlap, nfft) {}

    void unpack_and_padd_uint8(const uint8_t* __restrict__ data_in,
                               SizeType in_size,
                               ComplexType* __restrict__ data_p1,
                               ComplexType* __restrict__ data_p2,
                               SizeType out_size) const override {
        m_impl.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    }

    void unpack_and_padd_int8(const int8_t* __restrict__ data_in,
                              SizeType in_size,
                              ComplexType* __restrict__ data_p1,
                              ComplexType* __restrict__ data_p2,
                              SizeType out_size) const override {
        m_impl.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    }
};

// DataUnpacker implementation
DataUnpacker::DataUnpacker(SizeType nsub,
                           SizeType nbin,
                           SizeType noverlap,
                           SizeType nfft,
                           std::string_view in_order) {
    const auto order = kBasebandDataOrderMap.at(in_order);
    switch (order) {
    case BasebandDataOrder::kPRITF:
        m_pimpl = std::make_unique<
            DataUnpackerImplWrapper<BasebandDataOrder::kPRITF>>(nsub, nbin,
                                                                noverlap, nfft);
        break;
    case BasebandDataOrder::kFTPRI:
        m_pimpl = std::make_unique<
            DataUnpackerImplWrapper<BasebandDataOrder::kFTPRI>>(nsub, nbin,
                                                                noverlap, nfft);
        break;
    case BasebandDataOrder::kRITFP:
        m_pimpl = std::make_unique<
            DataUnpackerImplWrapper<BasebandDataOrder::kRITFP>>(nsub, nbin,
                                                                noverlap, nfft);
        break;
    }
}

template <typename DataType>
void DataUnpacker::unpack_and_padd(const DataType* __restrict__ data_in,
                                   SizeType in_size,
                                   ComplexType* __restrict__ data_p1,
                                   ComplexType* __restrict__ data_p2,
                                   SizeType out_size) const {
    if constexpr (std::is_same_v<DataType, uint8_t>) {
        m_pimpl->unpack_and_padd_uint8(data_in, in_size, data_p1, data_p2,
                                       out_size);
    } else if constexpr (std::is_same_v<DataType, int8_t>) {
        m_pimpl->unpack_and_padd_int8(data_in, in_size, data_p1, data_p2,
                                      out_size);
    } else {
        throw std::runtime_error("Invalid data type");
    }
}
// Explicit template instantiations
template void
DataUnpacker::unpack_and_padd<uint8_t>(const uint8_t* __restrict__,
                                       SizeType,
                                       ComplexType* __restrict__,
                                       ComplexType* __restrict__,
                                       SizeType) const;
template void DataUnpacker::unpack_and_padd<int8_t>(const int8_t* __restrict__,
                                                    SizeType,
                                                    ComplexType* __restrict__,
                                                    ComplexType* __restrict__,
                                                    SizeType) const;