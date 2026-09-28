#pragma once

/**
 * @file unpacker.hpp
 * @brief Baseband voltage stream unpackers and format converters for
 * LOFAR/telescope DAQ layouts.
 */

#include <memory>
#include <span>
#include <string_view>

#include "dmt/common/types.hpp"

namespace dmt::utils {

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
 */
class DataUnpackerCPU {
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
    DataUnpackerCPU(SizeType nsub,
                    SizeType nbin,
                    SizeType noverlap,
                    SizeType nfft,
                    std::string_view in_order,
                    int nthreads = 1);

    ~DataUnpackerCPU();
    DataUnpackerCPU(DataUnpackerCPU&&) noexcept;
    DataUnpackerCPU& operator=(DataUnpackerCPU&&) noexcept;
    DataUnpackerCPU(const DataUnpackerCPU&)            = delete;
    DataUnpackerCPU& operator=(const DataUnpackerCPU&) = delete;

    /**
     * @brief Unpacks input data and pads for FFT (CPU version).
     * @tparam DataType The integral input data type (e.g., uint8_t, int8_t).
     * @param data_in Span viewing the input host data.
     * @param data_p1 Span viewing the output host buffer for polarization 1
     * (ComplexType).
     * @param data_p2 Span viewing the output host buffer for polarization 2
     * (ComplexType).
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in,
                 std::span<ComplexType> data_p1,
                 std::span<ComplexType> data_p2) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
