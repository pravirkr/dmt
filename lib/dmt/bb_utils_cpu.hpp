#pragma once

#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
#include <string_view>

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
 * @tparam Npol The number of polarisations.
 */
template <DataOrder Order, SizeType Npol = 2> class DataUnpacker {
public:
    DataUnpacker(SizeType nsub, SizeType nbin, SizeType noverlap, SizeType nfft)
        : m_nsub(nsub),
          m_nbin(nbin),
          m_noverlap(noverlap),
          m_nfft(nfft) {
        if (m_nbin < 2 * m_noverlap) {
            throw std::runtime_error("Invalid nbin and noverlap values");
        }
        m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);
    }

    void unpack_and_padd(const uint8_t* __restrict__ data_in,
                         SizeType in_size,
                         ComplexType* __restrict__ data_p1,
                         ComplexType* __restrict__ data_p2,
                         SizeType out_size) {
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

private:
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_noverlap;
    SizeType m_nfft;
    SizeType m_nsamp;
    static constexpr SizeType kChunkSize = 64;

    void validate_sizes(SizeType in_size, SizeType out_size) {
        if (in_size != 2 * Npol * m_nsamp * m_nsub) {
            throw std::runtime_error("Invalid input size");
        }
        if (out_size != m_nfft * m_nsub * m_nbin) {
            throw std::runtime_error("Invalid output size");
        }
    }

    SizeType calculate_indices(SizeType isamp, SizeType ipol, SizeType isub) {
        if constexpr (Order == DataOrder::kPRITF) {
            return (ipol * 2 * m_nsamp * m_nsub) + (isamp * m_nsub) + isub;
        } else if constexpr (Order == DataOrder::kFTPRI) {
            return (isub * m_nsamp * Npol * 2) + (isamp * Npol * 2) + ipol;
        } else if constexpr (Order == DataOrder::kRITFP) {
            return (isamp * m_nsub * Npol) + (isub * Npol) + ipol;
        }
    }

    SizeType calculate_ri_stride() {
        if constexpr (Order == DataOrder::kPRITF) {
            return m_nsamp * m_nsub;
        } else if constexpr (Order == DataOrder::kFTPRI) {
            return 1;
        } else if constexpr (Order == DataOrder::kRITFP) {
            return m_nsamp * m_nsub * Npol;
        }
    }

    void process_chunk(const uint8_t* __restrict__ data_in,
                       ComplexType* __restrict__ data_p1,
                       ComplexType* __restrict__ data_p2,
                       SizeType ifft,
                       SizeType isub) {
        const auto out_offset = (ifft * m_nsub * m_nbin) + (isub * m_nbin);
        const auto start_sample =
            static_cast<IndexType>((m_nbin - 2 * m_noverlap) * ifft) -
            static_cast<IndexType>(m_noverlap);
        const auto bin_start = start_sample < 0 ? -start_sample : 0;
        const auto bin_end   = std::min(m_nbin, m_nsamp - start_sample);
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
    }
};

void pointwise_complex_multiply(const ComplexType* __restrict__ a,
                                const ComplexType* __restrict__ b,
                                ComplexType* __restrict__ c,
                                SizeType nx,
                                SizeType ny,
                                SizeType idm,
                                float scale);