#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "dmt/common/types.hpp"

/**
 * @enum DataOrder
 * @brief Describes the data order for different unpacking methods.
 *
 */
enum class DataOrder : uint8_t {
    kPRITF, /**< Polarisation-Real/Imag-time-frequency */
    kFTPRI, /**< Frequency-Time-Polarisation-Real/Imag */
    kRITFP  /**< Real/Imag-time-frequency-Polarisation */
};

/**
 * @brief Convert a string to the corresponding DataOrder enum value.
 *
 * @param sv String representing the data order (e.g. "FTPRI").
 * @return DataOrder The corresponding DataOrder enum value.
 * @throw std::runtime_error If the input string is invalid.
 */
constexpr DataOrder string_to_data_order(std::string_view sv) {
    if (sv == "FTPRI") {
        return DataOrder::kFTPRI;
    }
    if (sv == "PRITF") {
        return DataOrder::kPRITF;
    }
    if (sv == "RITFP") {
        return DataOrder::kRITFP;
    }
    throw std::runtime_error("Invalid data order");
}

// Input ordered as polarisation-Real/Imag-time-frequency (PTF) - LOFAR
// data_in shape: (npol=2, R/I=2, nsamp, nsub)
// noverlap = n_d * n_c // 2
// nsamp = nfft * (nbin - 2 * noverlap)
// data_out shape: (2, nfft, nsub, nbin)

/**
 * @brief Unpacks and pads the input data to generate complex timeseries.
 *
 * @tparam Order The data order for unpacking.
 * @tparam DataType The data type of the input data (e.g. uint8_t, int8_t).
 * @tparam Npol The number of polarisations.
 */
class DataUnpackerBase {
public:
    DataUnpackerBase(SizeType nsub,
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

    virtual ~DataUnpackerBase() = default;

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
                process_chunk(data_in, data_p1, data_p2, ifft, isub);
            }
        }
    }

protected:
    static constexpr SizeType kChunkSize = 64;
    static constexpr SizeType kNpol      = 2;

    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_noverlap;
    SizeType m_nfft;
    SizeType m_nsamp;

    constexpr void validate_sizes(SizeType in_size, SizeType out_size) const {
        if (in_size != 2 * kNpol * m_nsamp * m_nsub) {
            throw std::runtime_error("Invalid input size");
        }
        if (out_size != m_nfft * m_nsub * m_nbin) {
            throw std::runtime_error("Invalid output size");
        }
    }

    virtual SizeType
    calculate_indices(SizeType isamp, SizeType ipol, SizeType isub) const = 0;
    virtual SizeType calculate_ri_stride() const                          = 0;

    template <IntegralDataType DataType>
    void process_chunk(const DataType* __restrict__ data_in,
                       ComplexType* __restrict__ data_p1,
                       ComplexType* __restrict__ data_p2,
                       SizeType ifft,
                       SizeType isub) {
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
        for (SizeType i = 0; i < bin_start; ++i) {
            data_p1[out_offset + i] = ComplexType(0.0F, 0.0F);
            data_p2[out_offset + i] = ComplexType(0.0F, 0.0F);
        }
        for (SizeType i = bin_end; i < m_nbin; ++i) {
            data_p1[out_offset + i] = ComplexType(0.0F, 0.0F);
            data_p2[out_offset + i] = ComplexType(0.0F, 0.0F);
        }
    }
};

template <DataOrder Order> class DataUnpackerOld : public DataUnpackerBase {
public:
    using DataUnpackerBase::DataUnpackerBase;

protected:
    SizeType calculate_indices(SizeType isamp,
                               SizeType ipol,
                               SizeType isub) const override {
        if constexpr (Order == DataOrder::kPRITF) {
            return (ipol * 2 * m_nsamp * m_nsub) + (isamp * m_nsub) + isub;
        } else if constexpr (Order == DataOrder::kFTPRI) {
            return (isub * m_nsamp * kNpol * 2) + (isamp * kNpol * 2) + ipol;
        } else if constexpr (Order == DataOrder::kRITFP) {
            return (isamp * m_nsub * kNpol) + (isub * kNpol) + ipol;
        }
    }

    SizeType calculate_ri_stride() const override {
        if constexpr (Order == DataOrder::kPRITF) {
            return m_nsamp * m_nsub;
        } else if constexpr (Order == DataOrder::kFTPRI) {
            return 1;
        } else if constexpr (Order == DataOrder::kRITFP) {
            return m_nsamp * m_nsub * kNpol;
        }
    }
};

// Factory function
inline std::unique_ptr<DataUnpackerBase>
create_data_unpacker(SizeType nsub,
                     SizeType nbin,
                     SizeType noverlap,
                     SizeType nfft,
                     const std::string& in_order) {
    DataOrder order = string_to_data_order(in_order);
    switch (order) {
    case DataOrder::kPRITF:
        return std::make_unique<DataUnpackerOld<DataOrder::kPRITF>>(
            nsub, nbin, noverlap, nfft);
    case DataOrder::kFTPRI:
        return std::make_unique<DataUnpackerOld<DataOrder::kFTPRI>>(
            nsub, nbin, noverlap, nfft);
    case DataOrder::kRITFP:
        return std::make_unique<DataUnpackerOld<DataOrder::kRITFP>>(
            nsub, nbin, noverlap, nfft);
    default:
        throw std::runtime_error("Invalid data order");
    }
}

void pointwise_complex_multiply(const ComplexType* __restrict__ a,
                                const ComplexType* __restrict__ b,
                                ComplexType* __restrict__ c,
                                SizeType nx,
                                SizeType ny,
                                SizeType idm,
                                float scale);