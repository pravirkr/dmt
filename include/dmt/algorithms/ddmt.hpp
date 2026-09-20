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

namespace dmt::algorithms {

class DDMTCPU {
public:
    /**
     * @param nbits Input precision: 32 (default) selects the float
     * waterfall execute() overload; 1, 2, 4, 8, or 16 selects the
     * packed-integer execute() overload, with samples packed LSB-first
     * within each byte (matching dedisp's packed-integer convention).
     * @param kill_mask Optional per-channel mask (size == nchans, 1 = keep,
     * 0 = exclude from the sum); empty means all channels are kept.
     * @param nbeams Number of independent beams batched through one
     * instance (default 1). All execute() overloads become beam-major:
     * input shape (nbeams, nchans, nsamps[_packed]), output shape (nbeams,
     * ndm, nsamps_reduced); the delay table, kill mask, and DM grid are
     * shared across beams (dispersion doesn't depend on beam). At
     * nbeams == 1 every buffer layout is identical to the single-beam case.
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

    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            std::span<const float> dm_arr,
            int nthreads                       = 1,
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1);

    /// @brief Construct DDMTCPU with an optimal Lina Levin DM grid.
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            const plans::LevinConfig& levin,
            int nthreads                       = 1,
            SizeType nbits                     = 32,
            std::span<const uint8_t> kill_mask = {},
            SizeType nbeams                    = 1);

    /// @brief Construct directly from a pre-configured DDMTPlan.
    explicit DDMTCPU(const plans::DDMTPlan& plan,
                     int nthreads    = 1,
                     SizeType nbeams = 1);

    ~DDMTCPU();
    DDMTCPU(DDMTCPU&&) noexcept;
    DDMTCPU& operator=(DDMTCPU&&) noexcept;
    DDMTCPU(const DDMTCPU&)            = delete;
    DDMTCPU& operator=(const DDMTCPU&) = delete;

    const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call (see the
    /// constructor's `nbeams` doc comment).
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /// @brief Dedisperse a beam-major float waterfall (nbeams, nchans,
    /// nsamps). Requires get_plan().get_nbits() == 32; otherwise logs an
    /// error and leaves `dmt` untouched.
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /// @brief Dedisperse a beam-major packed-integer waterfall (nbeams,
    /// nchans, nsamps), each channel row packed at get_plan().get_nbits()
    /// bits per sample, LSB-first, byte-aligned per channel row (a row is
    /// ceil(nsamps * nbits / 8) bytes). `nsamps` must be given explicitly
    /// since it isn't always recoverable from the packed byte count alone
    /// (sub-byte widths round a row up to a whole byte). Requires
    /// get_plan().get_nbits() in {1,2,4,8,16}; otherwise logs an error and
    /// leaves `dmt` untouched. Summation is done in int32_t, avoiding a
    /// float conversion per sample.
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt);

    /// @brief Dedisperse a beam-major time-major packed filterbank of shape
    /// (nbeams, nsamps, nchans), where at each time sample, channels are
    /// packed together consecutively (matching SIGPROC .fil format and
    /// telescope streaming buffers).
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt);

    /// @brief Given the size of the next block (in samples per channel), the
    /// number of output samples the next execute() call (float or packed) will
    /// produce: 0 while still warming up (retained history + input shorter than
    /// the maximum delay), input_nsamps - max_delay on a cold start, or
    /// input_nsamps once the stream is warm.
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept;

    /// @brief Discard retained cross-call history and return execute
    /// to its one-shot, cold-start behavior (output size == input_nsamps -
    /// max_delay).
    void reset_history() noexcept;

    /// @brief Size of a fully-warmed-up history state: in floats (nbeams *
    /// nchans * max_delay) when nbits == 32, or in bytes (nbeams * nchans *
    /// packed_row_bytes(max_delay, nbits)) when nbits is a packed integer width.
    [[nodiscard]] SizeType history_state_size() const noexcept;

    /// @brief Save the current float history state (nbits == 32).
    /// Requires out.size() == history_state_size(); otherwise returns false.
    bool save_history(std::span<float> out) const;

    /// @brief Save the current packed history state (nbits in {1,2,4,8,16}).
    /// Requires out.size() == history_state_size(); otherwise returns false.
    bool save_history(std::span<uint8_t> out) const;

    /// @brief Restore a previously saved float history state (nbits == 32).
    /// Requires in.size() == history_state_size(); otherwise returns false.
    bool load_history(std::span<const float> in);

    /// @brief Restore a previously saved packed history state (nbits in {1,2,4,8,16}).
    /// Requires in.size() == history_state_size(); otherwise returns false.
    bool load_history(std::span<const uint8_t> in);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
/**
 * @brief CUDA direct dedispersion.
 * @details
 * Mirrors DDMTCPU's public API (get_plan, the float and packed-integer
 * execute() overloads, nbits/kill_mask). The host-span execute() overloads
 * internally chunk large inputs into gulps and pipeline H2D copy / kernel /
 * D2H copy across CUDA streams (see lib/ddmt_cuda.cu); the device-span
 * overloads are a single kernel launch with no chunking, for callers that
 * already manage device memory and want to pipeline across their own calls.
 */
class DDMTCUDA {
public:
    /**
     * @param nbits Input precision: 32 (default) selects the float
     * waterfall execute() overload; 1, 2, 4, 8, or 16 selects the
     * packed-integer execute() overload, with samples packed LSB-first
     * within each byte (matching dedisp's tdd convention).
     * @param kill_mask Optional per-channel mask (size == nchans, 1 = keep,
     * 0 = exclude from the sum); empty means all channels are kept.
     * @param nbeams Number of independent beams batched through one
     * instance (default 1); see DDMTCPU's constructor doc comment for the
     * beam-major layout contract, identical here.
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

    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const std::vector<float>& dm_arr,
             int device_id                      = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {},
             SizeType nbeams                    = 1);

    /// @brief Construct DDMTCUDA with an optimal Lina Levin DM grid.
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const plans::LevinConfig& levin,
             int device_id                      = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {},
             SizeType nbeams                    = 1);

    /// @brief Construct directly from a pre-configured DDMTPlan.
    explicit DDMTCUDA(const plans::DDMTPlan& plan,
                      int device_id   = 0,
                      SizeType nbeams = 1);

    ~DDMTCUDA();
    DDMTCUDA(DDMTCUDA&&) noexcept;
    DDMTCUDA& operator=(DDMTCUDA&&) noexcept;
    DDMTCUDA(const DDMTCUDA&)            = delete;
    DDMTCUDA& operator=(const DDMTCUDA&) = delete;

    const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call.
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /// @brief Dedisperse a float waterfall (host memory). Requires
    /// get_plan().get_nbits() == 32. Internally gulps large inputs and
    /// pipelines H2D/kernel/D2H across streams; see the class doc comment.
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /// @brief Dedisperse a float waterfall already resident on the device,
    /// as a single kernel launch on `stream` with no internal chunking.
    /// Requires get_plan().get_nbits() == 32.
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
