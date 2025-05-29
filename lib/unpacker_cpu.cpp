#include "dmt/utils/unpacker.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "spdlog/spdlog.h"

namespace dmt::utils {

template <>
class DataUnpacker<backend::CPU>::Impl {
public:
    static constexpr SizeType kChunkSize = 64;
    static constexpr SizeType kNpol      = 2;

    Impl(SizeType nsub,
         SizeType nbin,
         SizeType noverlap,
         SizeType nfft,
         std::string_view in_order,
         int nthreads)
        : m_nsub(nsub),
          m_nbin(nbin),
          m_noverlap(noverlap),
          m_nfft(nfft),
          m_nthreads(set_dmt_openmp_threads(nthreads)) {
        if (m_nbin <= 2 * m_noverlap) {
            throw std::invalid_argument(
                std::format("Invalid nbin and noverlap values: {} and {}",
                            m_nbin, m_noverlap));
        }
        m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);
        if (kBasebandDataOrderMap.contains(in_order)) {
            m_order = kBasebandDataOrderMap.at(in_order);
        } else {
            throw std::invalid_argument(
                std::format("Invalid input order: {}", in_order));
        }
        spdlog::debug("DataUnpacker<CPU>::Impl: Initialised with {} threads.",
                      m_nthreads);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in,
                 std::span<ComplexType> data_p1,
                 std::span<ComplexType> data_p2) const {
        validate_sizes(data_in.size(), data_p1.size(), data_p2.size());
        switch (m_order) {
        case BasebandDataOrder::kPRITF:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kPRITF>(
                data_in, data_p1, data_p2);
            break;
        case BasebandDataOrder::kFTPRI:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kFTPRI>(
                data_in, data_p1, data_p2);
            break;
        case BasebandDataOrder::kRITFP:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kRITFP>(
                data_in, data_p1, data_p2);
            break;
        default:
            throw std::logic_error("DataUnpacker<CPU>: Unsupported data order "
                                   "encountered in execute.");
        }
        spdlog::debug("DataUnpacker<CPU>::Impl: Execution complete.");
    }

private:
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_noverlap;
    SizeType m_nfft;
    SizeType m_nsamp;
    int m_nthreads;
    BasebandDataOrder m_order;

    void validate_sizes(SizeType in_size,
                        SizeType out1_size,
                        SizeType out2_size) const {
        // Expected input size: 2 polarizations * Real/Imag * total input
        // samples * subbands
        const SizeType expected_in_size = kNpol * 2 * m_nsamp * m_nsub;
        if (in_size != expected_in_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid input size. Expected {}, got {}.",
                expected_in_size, in_size));
        }
        // Expected output size: FFT blocks * subbands * bins per FFT
        const SizeType expected_out_size = m_nfft * m_nsub * m_nbin;
        if (out1_size != expected_out_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid output size. Expected {}, got {}.",
                expected_out_size, out1_size));
        }
        if (out2_size != expected_out_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid output size. Expected {}, got {}.",
                expected_out_size, out2_size));
        }
    }

    template <IntegralDataType DataType, BasebandDataOrder Order>
    void unpack_and_pad_dispatch(std::span<const DataType> data_in,
                                 std::span<ComplexType> data_p1,
                                 std::span<ComplexType> data_p2) const {
        const DataType* data_in_ptr = data_in.data();
        ComplexType* data_p1_ptr    = data_p1.data();
        ComplexType* data_p2_ptr    = data_p2.data();
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for collapse(2) default(none)                             \
    shared(data_in_ptr, data_p1_ptr, data_p2_ptr) num_threads(m_nthreads)
#endif
        for (SizeType ifft = 0; ifft < m_nfft; ++ifft) {
            for (SizeType isub = 0; isub < m_nsub; ++isub) {
                process_chunk<DataType, Order>(data_in_ptr, data_p1_ptr,
                                               data_p2_ptr, ifft, isub);
            }
        }
    }

    template <BasebandDataOrder Order>
    constexpr SizeType
    calculate_base_index(SizeType isamp, SizeType ipol, SizeType isub) const {
        static_assert(Order == BasebandDataOrder::kPRITF ||
                          Order == BasebandDataOrder::kFTPRI ||
                          Order == BasebandDataOrder::kRITFP,
                      "Unsupported Order");
        // Note: Index calculation assumes Real component. Imag is base +
        // ri_stride.
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            // Pol(2)-Real/Imag(2)-Time(nsamp)-Freq(nsub)
            // Index = pol_offset + real_offset + time_offset + freq_offset
            //       = ipol * (2 * nsamp * nsub) + 0 + isamp * nsub + isub
            return (ipol * 2 * m_nsamp * m_nsub) + (isamp * m_nsub) + isub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            // Freq(nsub)-Time(nsamp)-Pol(2)-Real/Imag(2)
            // Index = freq_offset + time_offset + pol_offset + real_offset
            //       = isub * (nsamp * kNpol * 2) + isamp * (kNpol * 2) + ipol *
            //       2 + 0
            return (isub * m_nsamp * kNpol * 2) + (isamp * kNpol * 2) + ipol;
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            // Real/Imag(2)-Time(nsamp)-Freq(nsub)-Pol(2)
            // Index = real_offset + time_offset + freq_offset + pol_offset
            //       = 0 + isamp * (nsub * kNpol) + isub * kNpol + ipol
            return (isamp * m_nsub * kNpol) + (isub * kNpol) + ipol;
        }
    }

    template <BasebandDataOrder Order>
    constexpr SizeType calculate_ri_stride() const {
        static_assert(Order == BasebandDataOrder::kPRITF ||
                          Order == BasebandDataOrder::kFTPRI ||
                          Order == BasebandDataOrder::kRITFP,
                      "Unsupported Order");
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            // Pol(2)-Real/Imag(2)-Time(nsamp)-Freq(nsub)
            // Stride between R/I is Time*Freq = nsamp * nsub
            return m_nsamp * m_nsub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            // Freq(nsub)-Time(nsamp)-Pol(2)-Real/Imag(2)
            // Stride between R/I is 1
            return 1;
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            // Real/Imag(2)-Time(nsamp)-Freq(nsub)-Pol(2)
            // Stride between R/I is Time*Freq*Pol = nsamp * nsub * kNpol
            return m_nsamp * m_nsub * kNpol;
        }
    }

    // Processes a single chunk (FFT block 'ifft' for subband 'isub')
    // Templated on DataType and Order
    template <IntegralDataType DataType, BasebandDataOrder Order>
    void process_chunk(const DataType* __restrict__ data_in,
                       ComplexType* __restrict__ data_p1,
                       ComplexType* __restrict__ data_p2,
                       SizeType ifft,
                       SizeType isub) const {
        const auto out_offset = (ifft * m_nsub + isub) * m_nbin;
        const auto start_sample =
            static_cast<IndexType>((m_nbin - 2 * m_noverlap) * ifft) -
            static_cast<IndexType>(m_noverlap);
        const auto bin_start =
            static_cast<SizeType>(start_sample < 0 ? -start_sample : 0);
        const auto bin_end =
            static_cast<SizeType>(std::min(m_nbin, m_nsamp - start_sample));
        const auto ri_stride = calculate_ri_stride<Order>();

        // --- Pad beginning ---
        std::fill_n(data_p1 + out_offset, bin_start, ComplexType(0.0F, 0.0F));
        std::fill_n(data_p2 + out_offset, bin_start, ComplexType(0.0F, 0.0F));

        // --- Process valid data range ---
        for (SizeType ibin = bin_start; ibin < bin_end; ibin += kChunkSize) {
            const SizeType chunk_end = std::min(ibin + kChunkSize, bin_end);
            for (SizeType i = ibin; i < chunk_end; ++i) {
                const auto isamp = static_cast<SizeType>(
                    start_sample + static_cast<std::ptrdiff_t>(i));
                const auto idx_in_base_p1 =
                    calculate_base_index<Order>(isamp, 0, isub);
                const auto idx_in_base_p2 =
                    calculate_base_index<Order>(isamp, 1, isub);
                data_p1[out_offset + i] = ComplexType(
                    static_cast<float>(data_in[idx_in_base_p1]),
                    static_cast<float>(data_in[idx_in_base_p1 + ri_stride]));
                data_p2[out_offset + i] = ComplexType(
                    static_cast<float>(data_in[idx_in_base_p2]),
                    static_cast<float>(data_in[idx_in_base_p2 + ri_stride]));
            }
        }

        // --- Pad end ---
        if (bin_end < m_nbin) {
            std::fill_n(data_p1 + out_offset + bin_end, m_nbin - bin_end,
                        ComplexType(0.0F, 0.0F));
            std::fill_n(data_p2 + out_offset + bin_end, m_nbin - bin_end,
                        ComplexType(0.0F, 0.0F));
        }
    }

}; // End DataUnpacker<backend::CPU>::Impl definition

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CPU> P>
DataUnpacker<backend::CPU>::DataUnpacker(SizeType nsub,
                                         SizeType nbin,
                                         SizeType noverlap,
                                         SizeType nfft,
                                         std::string_view in_order,
                                         int nthreads)
    : m_impl(std::make_unique<Impl>(
          nsub, nbin, noverlap, nfft, in_order, nthreads)) {
    spdlog::debug("DataUnpacker<CPU> object created.");
}
template <>
DataUnpacker<backend::CPU>::~DataUnpacker() {
    spdlog::debug("DataUnpacker<CPU> object destroyed.");
}
template <>
DataUnpacker<backend::CPU>::DataUnpacker(DataUnpacker&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("DataUnpacker<CPU> object moved.");
}
template <>
DataUnpacker<backend::CPU>&
DataUnpacker<backend::CPU>::operator=(DataUnpacker&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
template <IntegralDataType DataType, std::same_as<backend::CPU> P>
void DataUnpacker<backend::CPU>::execute(std::span<const DataType> data_in,
                                         std::span<ComplexType> data_p1,
                                         std::span<ComplexType> data_p2) const {
    m_impl->execute<DataType>(data_in, data_p1, data_p2);
}
// Explicit instantiation (for linking)
template DataUnpacker<backend::CPU>::DataUnpacker(
    SizeType, SizeType, SizeType, SizeType, std::string_view, int);

// Instantiate the public execute method for each supported DataType
template void
    DataUnpacker<backend::CPU>::execute<int8_t>(std::span<const int8_t>,
                                                std::span<ComplexType>,
                                                std::span<ComplexType>) const;
template void
    DataUnpacker<backend::CPU>::execute<uint8_t>(std::span<const uint8_t>,
                                                 std::span<ComplexType>,
                                                 std::span<ComplexType>) const;

} // namespace dmt::utils
