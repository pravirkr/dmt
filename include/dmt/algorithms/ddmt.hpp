#pragma once

#include <cstdint>
#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

/**
 * @file ddmt.hpp
 * @brief Direct Dispersion Measure Transform (DDMT) brute-force delay-and-sum dedispersion.
 */

namespace dmt::algorithms {

/**
 * @brief Direct Dispersion Measure Transform (DDMT) engine for CPU execution.
 *
 * Implements classical brute-force delay-and-sum incoherent dedispersion:
 * @f[
 * D[\text{DM}, t] = \sum_{\nu} I[\nu, t + \Delta t(\nu, \text{DM})]
 * @f]
 * Supports 32-bit float waterfalls as well as packed low-bit integers (1, 2, 4, 8, 16 bits),
 * per-channel kill masks for RFI mitigation, multi-beam batching, and overlap-save streaming.
 */
class DDMTCPU {
public:
    /**
     * @brief Constructs a DDMTCPU engine with a linear DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_max Maximum trial DM in pc/cm^3.
     * @param dm_step Linear spacing between DM trials in pc/cm^3.
     * @param dm_min Minimum trial DM in pc/cm^3 (default: 0).
     * @param nthreads Number of OpenMP worker threads (default: 1).
     * @param nbits Input precision: 32 (float), or 1, 2, 4, 8, 16 (packed integer).
     * @param kill_mask Optional per-channel mask (size nchans, 1=keep, 0=exclude from sum).
     * @param nbeams Number of independent beams batched through one instance (default: 1).
     */
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min                       = 0.0F,
            int nthreads                       = 1,
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMTCPU engine with an explicit DM trial array.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_arr Explicit span of trial DMs in pc/cm^3.
     * @param nthreads Number of OpenMP worker threads (default: 1).
     * @param nbits Input precision (default: 32).
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams (default: 1).
     */
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            std::span<const float> dm_arr,
            int nthreads                       = 1,
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMTCPU engine with an optimal Lina Levin DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param levin LevinConfig specifying dm_start, dm_end, pulse_width, and tol.
     * @param nthreads Number of OpenMP threads.
     * @param nbits Input bit precision.
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams.
     */
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            const plans::LevinConfig& levin,
            int nthreads                       = 1,
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMTCPU engine directly from a pre-configured DDMTPlan.
     * @param plan Pre-initialized DDMT plan.
     * @param nthreads Number of OpenMP worker threads.
     * @param nbeams Number of batched beams.
     */
    explicit DDMTCPU(const plans::DDMTPlan& plan,
                     int nthreads    = 1,
                     SizeType nbeams = 1);

    ~DDMTCPU();
    DDMTCPU(DDMTCPU&&) noexcept;
    DDMTCPU& operator=(DDMTCPU&&) noexcept;
    DDMTCPU(const DDMTCPU&)            = delete;
    DDMTCPU& operator=(const DDMTCPU&) = delete;

    /// @brief Read-only reference to underlying DDMT execution plan
    [[nodiscard]] const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Dedisperses a beam-major float32 waterfall (nbeams, nchans, nsamps).
     *
     * Requires get_plan().get_nbits() == 32.
     *
     * @param waterfall Input waterfall buffer.
     * @param dmt Output DM-time buffer: shape (nbeams, ndm, get_output_nsamps(nsamps)).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Dedisperses a beam-major packed-integer waterfall (nbeams, nchans, nsamps).
     *
     * Each channel row is packed at get_plan().get_nbits() bits per sample, LSB-first.
     * Summation accumulates in 32-bit integer arithmetic without float conversions.
     *
     * @param waterfall_packed Input packed bytes.
     * @param nsamps Explicit time sample count per channel.
     * @param dmt Output int32 DM-time buffer: shape (nbeams, ndm, get_output_nsamps(nsamps)).
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt);

    /**
     * @brief Dedisperses a time-major packed filterbank of shape (nbeams, nsamps, nchans).
     *
     * Matches standard SIGPROC `.fil` disk format and telescope streaming DAQ buffers.
     *
     * @param filterbank_packed Input packed bytes in time-major order.
     * @param nsamps Sample count in time dimension.
     * @param dmt Output int32 DM-time buffer.
     */
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt);

    /**
     * @brief Computes output sample count for an incoming chunk of input_nsamps.
     *
     * On cold start: returns input_nsamps - max_delay.
     * Once stream is warm: returns input_nsamps.
     *
     * @param input_nsamps Incoming chunk size in samples.
     * @return Number of valid output time samples produced.
     */
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept;

    /// @brief Discards retained cross-call history, returning execute to cold-start mode
    void reset_history() noexcept;

    /// @brief Size of the warmed-up streaming history state buffer
    [[nodiscard]] SizeType history_state_size() const noexcept;

    /// @brief Saves current float history state to caller-owned storage (nbits == 32)
    bool save_history(std::span<float> out) const;

    /// @brief Saves current packed integer history state to caller-owned storage (nbits in {1,2,4,8,16})
    bool save_history(std::span<uint8_t> out) const;

    /// @brief Restores previously saved float history state (nbits == 32)
    bool load_history(std::span<const float> in);

    /// @brief Restores previously saved packed history state (nbits in {1,2,4,8,16})
    bool load_history(std::span<const uint8_t> in);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Direct Dispersion Measure Transform (DDMT) engine for CUDA GPU execution.
 *
 * Accelerates brute-force delay-and-sum dedispersion using massively parallel CUDA kernels.
 * Supports host memory streaming with multi-stream pipelined memory transfers (H2D -> Kernel -> D2H)
 * as well as direct zero-copy device span execution.
 */
class DDMTCUDA {
public:
    /**
     * @brief Constructs a DDMTCUDA engine with a linear DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_max Maximum trial DM in pc/cm^3.
     * @param dm_step Linear spacing between DM trials in pc/cm^3.
     * @param dm_min Minimum trial DM in pc/cm^3 (default: 0).
     * @param device_id Target CUDA device ID (default: 0).
     * @param nbits Precision: 32 (float), or 1, 2, 4, 8, 16 (packed integer).
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams (default: 1).
     */
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min                       = 0.0F,
             int device_id                      = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {},
             SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMTCUDA engine with an explicit DM array.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_arr Explicit vector of DM trials in pc/cm^3.
     * @param device_id Target CUDA device ID.
     * @param nbits Input precision.
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams.
     */
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const std::vector<float>& dm_arr,
             int device_id                      = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {},
             SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMTCUDA engine with an optimal Lina Levin DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels.
     * @param tsamp Sampling interval in seconds.
     * @param levin LevinConfig structure.
     * @param device_id Target CUDA device ID.
     * @param nbits Input precision.
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams.
     */
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const plans::LevinConfig& levin,
             int device_id                      = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {},
             SizeType nbeams                    = 1);

    /**
     * @brief Constructs directly from a pre-configured DDMTPlan.
     * @param plan Pre-initialized DDMT plan.
     * @param device_id CUDA device ID.
     * @param nbeams Number of batched beams.
     */
    explicit DDMTCUDA(const plans::DDMTPlan& plan,
                      int device_id   = 0,
                      SizeType nbeams = 1);

    ~DDMTCUDA();
    DDMTCUDA(DDMTCUDA&&) noexcept;
    DDMTCUDA& operator=(DDMTCUDA&&) noexcept;
    DDMTCUDA(const DDMTCUDA&)            = delete;
    DDMTCUDA& operator=(const DDMTCUDA&) = delete;

    /// @brief Read-only reference to underlying DDMT execution plan
    [[nodiscard]] const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Dedisperses a float32 waterfall from host memory with pipelined H2D/kernel/D2H transfers.
     * @param waterfall Host input waterfall span.
     * @param dmt Host destination DMT array span.
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Dedisperses a float32 waterfall resident in GPU device memory.
     * @param d_waterfall Device pointer/span to input waterfall.
     * @param d_dmt Device pointer/span to destination DMT buffer.
     * @param stream CUDA stream for non-blocking asynchronous execution.
     */
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    /// @brief Dedisperse a packed-integer waterfall (host memory); see
    /// DDMTCPU::execute's packed-integer overload for the exact layout.
    /// Requires get_plan().get_nbits() in {1,2,4,8,16}.
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt);

    /// @brief Dedisperse a packed-integer waterfall already resident on the
    /// device, as a single kernel launch on `stream` with no internal
    /// chunking.
    void execute(cuda::std::span<const uint8_t> d_waterfall_packed,
                 SizeType nsamps,
                 cuda::std::span<int32_t> d_dmt,
                 cudaStream_t stream = nullptr);

    /// @brief Dedisperse a time-major packed filterbank of shape (nsamps,
    /// nchans), where at each time sample, channels are packed together
    /// consecutively. Unlike the channel-major execute() overloads, this one
    /// does not gulp: it allocates one device buffer sized to the whole
    /// input/output and does a single synchronous copy in, kernel launch, copy
    /// out. Fine for inputs that fit in device memory; for larger streams,
    /// chunk the input yourself and call this once per chunk (each call is a
    /// self-contained, cold-start transform -- no cross-call history).
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt);

    void execute_time_major(cuda::std::span<const uint8_t> d_filterbank_packed,
                            SizeType nsamps,
                            cuda::std::span<int32_t> d_dmt,
                            cudaStream_t stream = nullptr);

    /// @brief Given the size of the next block (in samples per channel), the
    /// number of output samples the next execute() call will produce.
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept;

    /// @brief Discard retained cross-call history.
    void reset_history() noexcept;

    /// @brief Size of a fully-warmed-up history state: in floats (nbeams *
    /// nchans * max_delay) when nbits == 32, or in bytes (nbeams * nchans *
    /// packed_row_bytes(max_delay, nbits)) when nbits is a packed integer width.
    [[nodiscard]] SizeType history_state_size() const noexcept;

    /// @brief Save the current float history state to host memory (nbits == 32).
    bool save_history(std::span<float> out) const;

    /// @brief Save the current float history state to device memory (nbits == 32).
    bool save_history(cuda::std::span<float> d_out,
                      cudaStream_t stream = nullptr) const;

    /// @brief Save the current packed history state to host memory (nbits in {1,2,4,8,16}).
    bool save_history(std::span<uint8_t> out) const;

    /// @brief Save the current packed history state to device memory (nbits in {1,2,4,8,16}).
    bool save_history(cuda::std::span<uint8_t> d_out,
                      cudaStream_t stream = nullptr) const;

    /// @brief Restore a previously saved float history state from host memory (nbits == 32).
    bool load_history(std::span<const float> in);

    /// @brief Restore a previously saved float history state from device memory (nbits == 32).
    bool load_history(cuda::std::span<const float> d_in,
                      cudaStream_t stream = nullptr);

    /// @brief Restore a previously saved packed history state from host memory (nbits in {1,2,4,8,16}).
    bool load_history(std::span<const uint8_t> in);

    /// @brief Restore a previously saved packed history state from device memory (nbits in {1,2,4,8,16}).
    bool load_history(cuda::std::span<const uint8_t> d_in,
                      cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA
} // namespace dmt::algorithms
