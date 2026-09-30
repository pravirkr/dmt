#pragma once

/**
 * @file cfdmt.hpp
 * @brief Coherent Fast Dispersion Measure Transform (CohFDMT): hybrid
 * coherent + FDMT dedispersion of baseband voltages.
 */

#include <memory>
#include <span>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/baseband.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Hybrid coherent/incoherent dedispersion search of recorded
 * dual-polarisation, subband-channelised baseband voltages.
 *
 * For each coarse DM trial the voltages are coherently dedispersed within
 * fine channels, detected to Stokes I (|X|^2 + |Y|^2) at tsamp ~ t_p,
 * aligned across channels and searched for the residual DM with a fine
 * FDMT (see plans::CohFDMTPlan for the geometry).
 *
 * **Usage (offline, block by block).** The engine is stateless: every
 * execute() processes one self-contained block of get_block_nsamps() raw
 * samples per subband. Advance the file read position by
 * get_stride_nsamps() samples between blocks (blocks overlap by the
 * dispersion sweep); the get_output_nsamps() valid samples per row of
 * consecutive blocks then tile the time axis:
 * @code
 * dmt::algorithms::CohFDMT search(cfg, dmt::Exec::cpu(8));
 * std::vector<float> dmt(search.get_dmt_size());
 * for (SizeType s0 = 0; s0 + search.get_block_nsamps() <= nsamps_file;
 *      s0 += search.get_stride_nsamps()) {
 *     read_block(s0, search.get_block_nsamps(), block); // caller's reader
 *     search.execute<int8_t>(block, dmt);
 *     // row i, sample j: DM plan.get_dm_grid_final()[i], arrival time at
 *     // plan.get_f_ref() of s0 * tbin + plan.get_output_time_offset()
 *     //     + j * plan.get_tsamp()
 * }
 * @endcode
 *
 * With `normalize` (the default) every channel is scaled to zero mean and
 * unit variance using the block's own noise statistics, so row i's noise
 * variance is plan.get_effective_variance_grid()[i].
 *
 * Runs on the backend chosen by the `Exec` constructor argument. Host memory
 * works on every backend; device memory (`DeviceSpan`) on GPU backends only,
 * asynchronous on the given Stream.
 */
class CohFDMT {
public:
    /**
     * @brief Builds the plan and allocates all working memory.
     * @throws std::invalid_argument for an invalid config or unavailable
     * backend.
     */
    explicit CohFDMT(const CohFDMTConfig& config, Exec exec = {});

    ~CohFDMT();
    CohFDMT(CohFDMT&&) noexcept;
    CohFDMT& operator=(CohFDMT&&) noexcept;
    CohFDMT(const CohFDMT&)            = delete;
    CohFDMT& operator=(const CohFDMT&) = delete;

    /// @brief The search geometry
    [[nodiscard]] const plans::CohFDMTPlan& get_plan() const noexcept;
    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;
    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;
    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;

    /// @brief Raw samples per subband per block (plan.get_block_nsamps())
    [[nodiscard]] SizeType get_block_nsamps() const noexcept;
    /// @brief Raw samples per subband between blocks
    [[nodiscard]] SizeType get_stride_nsamps() const noexcept;
    /// @brief Valid output samples per row per block
    [[nodiscard]] SizeType get_output_nsamps() const noexcept;
    /// @brief Input bytes of group @p igroup per block
    [[nodiscard]] SizeType get_input_size(SizeType igroup = 0) const;
    /// @brief Floats of the (ndm, output_nsamps) result
    [[nodiscard]] SizeType get_dmt_size() const noexcept;
    /// @brief Floats execute() needs in its output (== get_dmt_size())
    [[nodiscard]] SizeType get_buffer_size() const noexcept;
    /// @brief Memory allocated by the engine
    [[nodiscard]] plans::CohFDMTMemoryUsage get_memory_usage() const noexcept;

    /**
     * @brief Searches one block given as a single span (one subband group).
     *
     * @tparam DataType int8_t or uint8_t; the bytes are decoded per the
     * config's BasebandFormat (the type does not select the signedness).
     * @param data_in get_input_size() bytes in the configured order.
     * @param dmt At least get_dmt_size() floats: (ndm, output_nsamps), row i
     * the DM plan.get_dm_grid_final()[i].
     * @throws std::invalid_argument on a size mismatch.
     */
    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) const;

    /**
     * @brief Searches one block given as one span per subband group (e.g.
     * one GUPPI node file each), in the config's subband_groups order.
     */
    template <IntegralDataType DataType>
    void execute(std::span<const std::span<const DataType>> groups,
                 std::span<float> dmt) const;

    /**
     * @brief Device-memory execute() (GPU backends), asynchronous on
     * @p stream.
     */
    template <IntegralDataType DataType>
    void execute(DeviceSpan<const DataType> d_data_in,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {}) const;

    /// @brief Device-memory execute() with one span per subband group.
    template <IntegralDataType DataType>
    void execute(std::span<const DeviceSpan<const DataType>> d_groups,
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

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
