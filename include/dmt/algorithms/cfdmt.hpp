#pragma once

/**
 * @file cfdmt.hpp
 * @brief Coherent Fast Dispersion Measure Transform (CFDMT) hybrid dedispersion
 * algorithm.
 */

#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Computes the Coherent Fast Dispersion Measure Transform (CFDMT).
 *
 * Implements the hybrid coherent/incoherent search algorithm (Zackay et al.).
 * 1. Unpacks raw dual-polarization baseband voltage streams into complex float
 * channels.
 * 2. Applies double-precision coherent dedispersion chirps across sparse coarse
 * DM trials.
 * 3. Channelizes subbands and detects to total intensity (Stokes @f$ I =
 * |P_1|^2 + |P_2|^2 @f$).
 * 4. Applies bulk inter-channel delays and executes an internal fine FDMT tree
 * symmetrically covering @f$ [-\Delta\text{DM}/2, +\Delta\text{DM}/2] @f$
 * around each coarse trial.
 *
 * Runs on the backend chosen by the `Exec` constructor argument (GPU: cuFFT
 * plans, batched chirp kernels and a device-memory fine FDMT). Host memory
 * works on every backend; device memory (`DeviceSpan`) on GPU backends only,
 * asynchronous on the given Stream.
 */
class CohFDMT {
public:
    /**
     * @brief Constructs a CohFDMT search engine.
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
     * @param noverlap Overlap sample count for overlap-save baseband
     * convolution (default: 8192).
     * @param data_order Voltage memory order: "PRITF", "FTPRI", or "RITFP"
     * (default: "PRITF").
     * @param exec Backend and its resources (default: CPU, 1 thread).
     */
    CohFDMT(float f_center,
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
            Exec exec                   = {});

    ~CohFDMT();
    CohFDMT(CohFDMT&&) noexcept;
    CohFDMT& operator=(CohFDMT&&) noexcept;
    CohFDMT(const CohFDMT&)            = delete;
    CohFDMT& operator=(const CohFDMT&) = delete;

    /// @brief Read-only reference to underlying CohFDMT execution plan
    [[nodiscard]] const plans::CohFDMTPlan& get_plan() const noexcept;
    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;
    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;
    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;

    /// @brief Float elements of the result: (ndm_total, nsamps), the leading
    /// part of the execute() output buffer.
    [[nodiscard]] SizeType get_dmt_size() const noexcept;

    /// @brief Output buffer length execute() needs (>= get_dmt_size(); see
    /// plans::CohFDMTPlan::get_buffer_size()). The tail past get_dmt_size()
    /// is scratch.
    [[nodiscard]] SizeType get_buffer_size() const noexcept;

    /**
     * @brief Executes the end-to-end CFDMT pipeline on packed baseband voltage
     * data.
     *
     * @tparam DataType Integral baseband sample type (e.g. uint8_t, int8_t).
     * @param data_in View of contiguous input packed baseband voltages.
     * @param dmt Output buffer; the leading get_dmt_size() floats hold the
     * (ndm_total, nsamps) result, coarse-DM trial i's fine trials at offset
     * i * fine get_dmt_size(). The CPU backend steps in place and needs at
     * least get_buffer_size() floats; a GPU backend steps in a device arena
     * and needs only get_dmt_size().
     * @throws std::invalid_argument if dmt is too small.
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

    /**
     * @brief Device-memory execute() (GPU backends): @p d_dmt holds at least
     * get_buffer_size() floats. Asynchronous on @p stream.
     */
    template <IntegralDataType DataType>
    void execute(DeviceSpan<const DataType> d_data_in,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {}) const;

    template <IntegralDataType DataType,
              typename Alloc1 = std::allocator<DataType>,
              typename Alloc2 = std::allocator<float>>
    void execute(const std::vector<DataType, Alloc1>& data_in,
                 std::vector<float, Alloc2>& dmt) const {
        execute<DataType>(std::span<const DataType>(data_in),
                          std::span<float>(dmt));
    }

    /**
     * @brief Resets every coarse-DM trial's fine-search streaming history to
     * a cold state (e.g. for a new observation / non-contiguous data).
     */
    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
