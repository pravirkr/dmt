#include "dmt/common/types.hpp"
#include "dmt/utils/unpacker.hpp"

#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <spdlog/spdlog.h>

#include "dmt/cuda_utils.cuh"

namespace dmt::utils {

namespace {
// Templated functor to perform the unpacking logic for a single output element
template <IntegralDataType InDataType, BasebandDataOrder Order>
struct UnpackAndPadFunctor {
    const InDataType* __restrict__ d_in_ptr;
    ComplexTypeCUDA* __restrict__ d_p1_ptr;
    ComplexTypeCUDA* __restrict__ d_p2_ptr;
    int nsub;
    int nbin;
    int noverlap;
    int nfft;
    int nsamp;
    int ri_stride;

    // Constants
    static constexpr int kNpol = 2;

    __host__ __device__ inline int
    calculate_base_index(int isamp, int ipol, int isub) const {
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            return (ipol * 2 * nsamp * nsub) + (isamp * nsub) + isub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            return (isub * nsamp * kNpol * 2) + (isamp * kNpol * 2) +
                   (ipol * 2);
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            return (isamp * nsub * kNpol) + (isub * kNpol) + ipol;
        }
        return 0; // Should not be reached
    }

    __host__ __device__ void operator()(int idx_out) const {
        // Deconstruct linear output index idx_out into (ifft, isub, ibin)
        // Output layout is assumed: [nfft][nsub][nbin]
        const int isub_x_nbin = nsub * nbin;
        const int ifft        = idx_out / isub_x_nbin;
        const int remainder   = idx_out % isub_x_nbin;
        const int isub        = remainder / nbin;
        const int ibin        = remainder % nbin;

        const auto start_sample = ((nbin - 2 * noverlap) * ifft) - noverlap;
        const auto current_input_samp_signed = start_sample + ibin;

        // Check if padding is needed (input sample index out of bounds)
        if (current_input_samp_signed < 0 ||
            current_input_samp_signed >= nsamp) {
            d_p1_ptr[idx_out] = ComplexTypeCUDA(0.0F, 0.0F);
            d_p2_ptr[idx_out] = ComplexTypeCUDA(0.0F, 0.0F);
        } else {
            const auto isamp          = current_input_samp_signed;
            const auto idx_in_base_p1 = calculate_base_index(isamp, 0, isub);
            const auto idx_in_base_p2 = calculate_base_index(isamp, 1, isub);
            d_p1_ptr[idx_out]         = ComplexTypeCUDA(
                static_cast<float>(d_in_ptr[idx_in_base_p1]),
                static_cast<float>(d_in_ptr[idx_in_base_p1 + ri_stride]));
            d_p2_ptr[idx_out] = ComplexTypeCUDA(
                static_cast<float>(d_in_ptr[idx_in_base_p2]),
                static_cast<float>(d_in_ptr[idx_in_base_p2 + ri_stride]));
        }
    }
};
} // namespace

template <>
class DataUnpacker<backend::CUDA>::Impl {
public:
    static constexpr SizeType kNpol = 2;

    Impl(SizeType nsub,
         SizeType nbin,
         SizeType noverlap,
         SizeType nfft,
         std::string_view in_order,
         int device_id)
        : m_nsub(nsub),
          m_nbin(nbin),
          m_noverlap(noverlap),
          m_nfft(nfft),
          m_device_id(device_id) {
        cuda_utils::set_device(m_device_id);
        spdlog::debug("DataUnpacker<CUDA>::Impl: Set device to {}.",
                      m_device_id);
        if (m_nbin <= 2 * m_noverlap) {
            throw std::invalid_argument(
                std::format("DataUnpacker<CUDA>: Invalid nbin and noverlap "
                            "values: {} and {}",
                            m_nbin, m_noverlap));
        }
        m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);
        if (kBasebandDataOrderMap.contains(in_order)) {
            m_order = kBasebandDataOrderMap.at(in_order);
        } else {
            throw std::invalid_argument(
                std::format("Invalid input order: {}", in_order));
        }
        // Expected input size: 2 polarizations * Real/Imag * total input
        // samples * subbands
        m_expected_in_size = kNpol * 2 * m_nsamp * m_nsub;
        // Expected output size: FFT blocks * subbands * bins per FFT
        m_expected_out_size = m_nfft * m_nsub * m_nbin;

        // Make this robust if other DataType sizes are supported later
        const SizeType sizeof_datatype = 1; // Assuming int8/uint8
        const auto m_expected_in_bytes = m_expected_in_size * sizeof_datatype;
        m_d_in_buffer.resize(m_expected_in_bytes);

        spdlog::debug("DataUnpacker<CUDA>::Impl: Initialised using device {}.",
                      m_device_id);
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in,
                 cuda::std::span<ComplexTypeCUDA> data_p1,
                 cuda::std::span<ComplexTypeCUDA> data_p2,
                 cudaStream_t stream) {
        validate_sizes(data_in.size(), data_p1.size(), data_p2.size());

        // Copy input data from host to device
        // Cast the byte pointer to the specific DataType pointer needed now
        std::byte* d_in_byte_ptr =
            thrust::raw_pointer_cast(m_d_in_buffer.data());
        auto* d_in_ptr = reinterpret_cast<DataType*>(d_in_byte_ptr);
        DMT_CHECK_CUDA_CALL(
            cudaMemcpyAsync(d_in_ptr, data_in.data(), data_in.size_bytes(),
                            cudaMemcpyHostToDevice, stream),
            "DataUnpacker<backend::CUDA>::Impl: cudaMemcpyAsync");

        auto data_in_d_span = cuda::std::span(d_in_ptr, m_expected_in_size);

        switch (m_order) {
        case BasebandDataOrder::kPRITF:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kPRITF>(
                data_in_d_span, data_p1, data_p2, stream);
            break;
        case BasebandDataOrder::kFTPRI:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kFTPRI>(
                data_in_d_span, data_p1, data_p2, stream);
            break;
        case BasebandDataOrder::kRITFP:
            unpack_and_pad_dispatch<DataType, BasebandDataOrder::kRITFP>(
                data_in_d_span, data_p1, data_p2, stream);
            break;
        default:
            throw std::logic_error("DataUnpacker<CUDA>: Unsupported data order "
                                   "encountered in execute.");
        }
        spdlog::debug(
            "DataUnpacker<CUDA>::Impl: Execution submitted to stream");
    }

private:
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_noverlap;
    SizeType m_nfft;
    SizeType m_nsamp;
    int m_device_id;
    BasebandDataOrder m_order;
    SizeType m_expected_in_size;
    SizeType m_expected_out_size;
    DeviceVector<std::byte> m_d_in_buffer;

    void validate_sizes(SizeType in_size,
                        SizeType out1_size,
                        SizeType out2_size) const {
        if (in_size != m_expected_in_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid input size. Expected {}, got {}.",
                m_expected_in_size, in_size));
        }
        if (out1_size != m_expected_out_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid output size. Expected {}, got {}.",
                m_expected_out_size, out1_size));
        }
        if (out2_size != m_expected_out_size) {
            throw std::runtime_error(std::format(
                "DataUnpacker<CPU>: Invalid output size. Expected {}, got {}.",
                m_expected_out_size, out2_size));
        }
    }

    template <BasebandDataOrder Order>
    constexpr SizeType calculate_ri_stride() const {
        static_assert(Order == BasebandDataOrder::kPRITF ||
                          Order == BasebandDataOrder::kFTPRI ||
                          Order == BasebandDataOrder::kRITFP,
                      "Unsupported Order");
        if constexpr (Order == BasebandDataOrder::kPRITF) {
            return m_nsamp * m_nsub;
        } else if constexpr (Order == BasebandDataOrder::kFTPRI) {
            return 1;
        } else if constexpr (Order == BasebandDataOrder::kRITFP) {
            return m_nsamp * m_nsub * kNpol;
        }
    }

    template <IntegralDataType DataType, BasebandDataOrder Order>
    void unpack_and_pad_dispatch(cuda::std::span<const DataType> data_in_d,
                                 cuda::std::span<ComplexTypeCUDA> data_p1_d,
                                 cuda::std::span<ComplexTypeCUDA> data_p2_d,
                                 cudaStream_t stream) const {
        const auto ri_stride = calculate_ri_stride<Order>();
        const auto n_output  = static_cast<int>(data_p1_d.size());

        auto first = thrust::counting_iterator<int>(0);
        auto last  = thrust::counting_iterator<int>(n_output);
        UnpackAndPadFunctor<DataType, Order> functor{
            .d_in_ptr  = data_in_d.data(),
            .d_p1_ptr  = data_p1_d.data(),
            .d_p2_ptr  = data_p2_d.data(),
            .nsub      = static_cast<int>(m_nsub),
            .nbin      = static_cast<int>(m_nbin),
            .noverlap  = static_cast<int>(m_noverlap),
            .nfft      = static_cast<int>(m_nfft),
            .nsamp     = static_cast<int>(m_nsamp),
            .ri_stride = static_cast<int>(ri_stride)};
        thrust::for_each(thrust::cuda::par.on(stream), first, last, functor);
        cuda_utils::check_last_cuda_error(
            "thrust::for_each unpack/pad failed");
    }

}; // End DataUnpacker<backend::CUDA>::Impl definition

// CUDA-specific constructor implementation
template <>
template <std::same_as<backend::CUDA> P>
DataUnpacker<backend::CUDA>::DataUnpacker(SizeType nsub,
                                          SizeType nbin,
                                          SizeType noverlap,
                                          SizeType nfft,
                                          std::string_view in_order,
                                          int device_id)
    : m_impl(std::make_unique<Impl>(
          nsub, nbin, noverlap, nfft, in_order, device_id)) {
    spdlog::debug("DataUnpacker<CUDA> object created.");
}
template <>
DataUnpacker<backend::CUDA>::~DataUnpacker() {
    spdlog::debug("DataUnpacker<CUDA> object destroyed.");
}
template <>
DataUnpacker<backend::CUDA>::DataUnpacker(DataUnpacker&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("DataUnpacker<CUDA> object moved.");
}
template <>
DataUnpacker<backend::CUDA>&
DataUnpacker<backend::CUDA>::operator=(DataUnpacker&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
template <IntegralDataType DataType, std::same_as<backend::CUDA> P>
void DataUnpacker<backend::CUDA>::execute(
    std::span<const DataType> data_in,
    cuda::std::span<ComplexTypeCUDA> data_p1,
    cuda::std::span<ComplexTypeCUDA> data_p2,
    cudaStream_t stream) const {
    m_impl->execute<DataType>(data_in, data_p1, data_p2, stream);
}

} // namespace dmt::utils