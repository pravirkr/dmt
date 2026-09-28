#pragma once

/**
 * @file unpacker_cuda.cuh
 * @brief Baseband unpacker on CUDA (private).
 */

#include <memory>
#include <span>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/common/types.hpp"
#include "dmt/gpu_utils.cuh"
#include "dmt/unpacker.hpp"

namespace dmt::utils {

/**
 * @brief Unpacks and pads input data based on specified order and type.
 *
 * Handles different input data types (uint8_t, int8_t) and memory layouts
 * (BasebandDataOrder), converting to complex float output suitable for FFT.
 *
 * @tparam Backend The execution backend (dmt::backend::CPU or
 * dmt::backend::CUDA).
 */
class DataUnpackerCUDA {
public:
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
    DataUnpackerCUDA(SizeType nsub,
                     SizeType nbin,
                     SizeType noverlap,
                     SizeType nfft,
                     std::string_view in_order,
                     int device_id = 0);
    ~DataUnpackerCUDA();
    DataUnpackerCUDA(DataUnpackerCUDA&&) noexcept;
    DataUnpackerCUDA& operator=(DataUnpackerCUDA&&) noexcept;
    DataUnpackerCUDA(const DataUnpackerCUDA&)            = delete;
    DataUnpackerCUDA& operator=(const DataUnpackerCUDA&) = delete;

    /**
     * @brief Unpacks input data and pads for FFT (CUDA version).
     * @tparam DataType The integral input data type (e.g., uint8_t, int8_t).
     * @param data_in Span viewing the input host data (copy to device is
     * internal).
     * @param data_p1 Span viewing the output device buffer for polarization 1
     * (ComplexTypeGPU).
     * @param data_p2 Span viewing the output device buffer for polarization 2
     * (ComplexTypeGPU).
     * @param stream CUDA stream for execution.
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in,
                 cuda::std::span<ComplexTypeGPU> data_p1,
                 cuda::std::span<ComplexTypeGPU> data_p2,
                 cudaStream_t stream = nullptr) const;

    /**
     * @brief Unpacks input device data and pads for FFT (device-resident
     * version).
     * @tparam DataType The integral input data type (e.g., uint8_t, int8_t).
     * @param data_in Span viewing the input device data.
     * @param data_p1 Span viewing the output device buffer for polarization 1
     * (ComplexTypeGPU).
     * @param data_p2 Span viewing the output device buffer for polarization 2
     * (ComplexTypeGPU).
     * @param stream CUDA stream for execution.
     */
    template <IntegralDataType DataType>
    void execute(cuda::std::span<const DataType> data_in,
                 cuda::std::span<ComplexTypeGPU> data_p1,
                 cuda::std::span<ComplexTypeGPU> data_p2,
                 cudaStream_t stream = nullptr) const;

    template <IntegralDataType DataType,
              typename Alloc = std::allocator<DataType>>
    void execute(const std::vector<DataType, Alloc>& data_in,
                 cuda::std::span<ComplexTypeGPU> data_p1,
                 cuda::std::span<ComplexTypeGPU> data_p2,
                 cudaStream_t stream = nullptr) const {
        execute<DataType>(std::span<const DataType>(data_in), data_p1, data_p2,
                          stream);
    }

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
