#pragma once

/**
 * @file cfdmt.hpp
 * @brief Coherent Fast Dispersion Measure Transform (CFDMT) hybrid dedispersion algorithm.
 */

#include <memory>
#include <span>
#include <string_view>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Computes the Coherent Fast Dispersion Measure Transform (CFDMT) on the CPU.
 *
 * Implements the hybrid coherent/incoherent search algorithm (Zackay et al.).
 * 1. Unpacks raw dual-polarization baseband voltage streams into complex float channels.
 * 2. Applies double-precision coherent dedispersion chirps across sparse coarse DM trials.
 * 3. Channelizes subbands and detects to total intensity (Stokes @f$ I = |P_1|^2 + |P_2|^2 @f$).
 * 4. Applies bulk inter-channel delays and executes an internal fine FDMT tree symmetrically
 *    covering @f$ [-\Delta\text{DM}/2, +\Delta\text{DM}/2] @f$ around each coarse trial.
 */
class CohFDMTCPU {
public:
    /**
     * @brief Constructs a CohFDMTCPU search engine.
     *
     * @param f_center Central RF frequency of the observation in MHz.
     * @param bw_sub Subband bandwidth in MHz.
     * @param nsub Number of synthesized subbands.
     * @param tbin Voltage sampling interval in seconds (1 / total_bandwidth).
     * @param nbin FFT block size for coherent dedispersion.
     * @param nfft Contiguous FFT blocks per coherent processing segment.
     * @param t_p Output detected time resolution in seconds.
     * @param dm_max Maximum coherent DM trial in pc/cm^3.
     * @param dm_min Minimum coherent DM trial in pc/cm^3 (default: 0).
     * @param noverlap Overlap sample count for overlap-save baseband convolution (default: 8192).
     * @param data_order Voltage memory order: "PRITF", "FTPRI", or "RITFP" (default: "PRITF").
     * @param verbose Enable diagnostic log output (default: false).
     * @param nthreads Number of OpenMP worker threads (default: 1).
     */
    CohFDMTCPU(float f_center,
               float bw_sub,
               SizeType nsub,
               float tbin,
               SizeType nbin,
               SizeType nfft,
               float t_p,
               float dm_max,
               float dm_min                = 0.0F,
               SizeType noverlap           = 8192,
               std::string_view data_order = "PRITF",
               bool verbose                = false,
               int nthreads                = 1);

    ~CohFDMTCPU();
    CohFDMTCPU(CohFDMTCPU&&) noexcept;
    CohFDMTCPU& operator=(CohFDMTCPU&&) noexcept;
    CohFDMTCPU(const CohFDMTCPU&)            = delete;
    CohFDMTCPU& operator=(const CohFDMTCPU&) = delete;

    /// @brief Read-only reference to underlying CohFDMT execution plan
    [[nodiscard]] const plans::CohFDMTPlan& get_plan() const noexcept;

    /// @brief Total float elements in the output 2D DMT buffer (ndm_total * nsamps)
    [[nodiscard]] SizeType get_dmt_size() const noexcept;

    /**
     * @brief Executes the end-to-end CFDMT pipeline on packed baseband voltage data.
     *
     * @tparam DataType Integral baseband sample type (e.g. uint8_t, int8_t).
     * @param data_in View of contiguous input packed baseband voltages.
     * @param dmt View of destination float output buffer of size get_dmt_size().
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

    /**
     * @brief Resets every coarse-DM trial's fine-search streaming history to
     * a cold state (e.g. for a new observation / non-contiguous data).
     */
    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Computes the Coherent Fast Dispersion Measure Transform (CFDMT) on CUDA GPUs.
 *
 * Implements the Zackay et al. hybrid search pipeline leveraging GPU cuFFT plans,
 * batched chirp multiplication kernels, and device-memory fine FDMT execution.
 */
class CohFDMTCUDA {
public:
    /**
     * @brief Constructs a CohFDMTCUDA GPU search engine.
     *
     * @param f_center Central RF frequency of the observation in MHz.
     * @param bw_sub Subband bandwidth in MHz.
     * @param nsub Number of synthesized subbands.
     * @param tbin Voltage sampling interval in seconds.
     * @param nbin FFT block size for coherent dedispersion.
     * @param nfft Contiguous FFT blocks per processing segment.
     * @param t_p Output detected time resolution in seconds.
     * @param dm_max Maximum coherent DM trial in pc/cm^3.
     * @param dm_min Minimum coherent DM trial in pc/cm^3 (default: 0).
     * @param noverlap Overlap sample count for convolution (default: 8192).
     * @param data_order Voltage memory order: "PRITF", "FTPRI", or "RITFP" (default: "PRITF").
     * @param verbose Enable diagnostic log output (default: false).
     * @param device_id Target CUDA device ID (default: 0).
     */
    CohFDMTCUDA(float f_center,
                float bw_sub,
                SizeType nsub,
                float tbin,
                SizeType nbin,
                SizeType nfft,
                float t_p,
                float dm_max,
                float dm_min                = 0.0F,
                SizeType noverlap           = 8192,
                std::string_view data_order = "PRITF",
                bool verbose                = false,
                int device_id               = 0);

    ~CohFDMTCUDA();
    CohFDMTCUDA(CohFDMTCUDA&&) noexcept;
    CohFDMTCUDA& operator=(CohFDMTCUDA&&) noexcept;
    CohFDMTCUDA(const CohFDMTCUDA&)            = delete;
    CohFDMTCUDA& operator=(const CohFDMTCUDA&) = delete;

    /// @brief Read-only reference to underlying CohFDMT execution plan
    [[nodiscard]] const plans::CohFDMTPlan& get_plan() const noexcept;

    /// @brief Total float elements in the output 2D DMT buffer (ndm_total * nsamps)
    [[nodiscard]] SizeType get_dmt_size() const noexcept;

    /**
     * @brief Executes CFDMT from host memory (internally transfers to device and back).
     * @tparam DataType Integral baseband sample type (uint8_t, int8_t).
     * @param data_in Contiguous host input packed baseband voltages.
     * @param dmt Destination host float buffer of size get_dmt_size().
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

    /**
     * @brief Executes CFDMT directly on GPU device memory.
     * @tparam DataType Integral baseband sample type.
     * @param d_data_in Device pointer/span to input packed baseband voltages.
     * @param d_dmt Device pointer/span to output DMT buffer.
     * @param stream CUDA stream for non-blocking asynchronous execution.
     */
    template <IntegralDataType DataType>
    void execute(cuda::std::span<const DataType> d_data_in,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr) const;

    /**
     * @brief Resets every coarse-DM trial's fine-search streaming history to a cold state.
     */
    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
