#pragma once

#include <memory>
#include <span>
#include <string_view>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/types.hpp"

namespace dmt {

// Input ordered as polarisation-Real/Imag-time-frequency (PTF) - LOFAR
// data_in shape: (npol=2, R/I=2, nsamp, nsub)
// noverlap = n_d * n_c // 2
// nsamp = nfft * (nbin - 2 * noverlap)
// data_out shape: (2, nfft, nsub, nbin)
/**
 * @brief Unpacks and pads input data based on specified order and type.
 *
 * Handles different input data types (uint8_t, int8_t) and memory layouts
 * (BasebandDataOrder), converting to complex float output suitable for FFT.
 *
 * @tparam Backend The execution backend (dmt::backend::CPU or
 * dmt::backend::CUDA).
 */
template <backend::ExecutionBackend Backend = backend::CPU>
class DataUnpacker {
public:
    /**
     * @brief Construct for CPU backend.
     * @param nsub Number of subbands.
     * @param nbin Number of frequency bins per subband in output.
     * @param noverlap Overlap size for FFT processing.
     * @param nfft Number of FFTs / time blocks.
     * @param in_order String identifier for input data order (e.g., "PRITF",
     * "FTPRI").
     * @param nthreads Number of threads for OpenMP execution.
     */
    template <std::same_as<backend::CPU> P = Backend>
    DataUnpacker(SizeType nsub,
                 SizeType nbin,
                 SizeType noverlap,
                 SizeType nfft,
                 std::string_view in_order,
                 int nthreads = 1);

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Construct for CUDA backend.
     * @param nsub Number of subbands.
     * @param nbin Number of frequency bins per subband in output.
     * @param noverlap Overlap size for FFT processing.
     * @param nfft Number of FFTs / time blocks.
     * @param in_order String identifier for input data order (e.g., "PRITF",
     * "FTPRI").
     * @param device_id CUDA device ID.
     */
    template <std::same_as<backend::CUDA> P = Backend>
    DataUnpacker(SizeType nsub,
                 SizeType nbin,
                 SizeType noverlap,
                 SizeType nfft,
                 std::string_view in_order,
                 int device_id = 0);
#endif // DMT_ENABLE_CUDA

    ~DataUnpacker();
    DataUnpacker(DataUnpacker&&) noexcept;
    DataUnpacker& operator=(DataUnpacker&&) noexcept;
    DataUnpacker(const DataUnpacker&)            = delete;
    DataUnpacker& operator=(const DataUnpacker&) = delete;

    /**
     * @brief Unpacks input data and pads for FFT (CPU version).
     * @tparam DataType The integral input data type (e.g., uint8_t, int8_t).
     * @param data_in Span viewing the input host data.
     * @param data_p1 Span viewing the output host buffer for polarization 1
     * (ComplexType).
     * @param data_p2 Span viewing the output host buffer for polarization 2
     * (ComplexType).
     */
    template <IntegralDataType DataType, std::same_as<backend::CPU> P = Backend>
    void execute(std::span<const DataType> data_in,
                 std::span<ComplexType> data_p1,
                 std::span<ComplexType> data_p2) const;

#ifdef DMT_ENABLE_CUDA
    /**
     * @brief Unpacks input data and pads for FFT (CUDA version).
     * @tparam DataType The integral input data type (e.g., uint8_t, int8_t).
     * @param data_in Span viewing the input host data (copy to device is
     * internal).
     * @param data_p1 Span viewing the output device buffer for polarization 1
     * (ComplexTypeCUDA).
     * @param data_p2 Span viewing the output device buffer for polarization 2
     * (ComplexTypeCUDA).
     * @param stream CUDA stream for execution.
     */
    template <IntegralDataType DataType,
              std::same_as<backend::CUDA> P = Backend>
    void execute(std::span<const DataType> data_in,
                 cuda::std::span<ComplexTypeCUDA> data_p1,
                 cuda::std::span<ComplexTypeCUDA> data_p2,
                 cudaStream_t stream = nullptr) const;
#endif // DMT_ENABLE_CUDA

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using DataUnpackerCPU = DataUnpacker<backend::CPU>;
#ifdef DMT_ENABLE_CUDA
using DataUnpackerCUDA = DataUnpacker<backend::CUDA>;
#endif // DMT_ENABLE_CUDA

} // namespace dmt