#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

/**
 * @file ddmt.hpp
 * @brief Direct Dispersion Measure Transform (DDMT) brute-force delay-and-sum
 * dedispersion.
 */

namespace dmt::algorithms {

/**
 * @brief Direct Dispersion Measure Transform (DDMT).
 *
 * Implements classical brute-force delay-and-sum incoherent dedispersion:
 * @f[
 * D[\text{DM}, t] = \sum_{\nu} I[\nu, t + \Delta t(\nu, \text{DM})]
 * @f]
 * Supports 32-bit float waterfalls as well as packed low-bit integers (1, 2, 4,
 * 8, 16 bits), per-channel kill masks for RFI mitigation, multi-beam batching,
 * and overlap-save streaming.
 *
 * Every trial sums every active channel directly: the cost is
 * nchans x ndm additions per output sample on every backend. SDMT computes
 * the same sums with fewer additions on the CPU.
 *
 * Runs on the backend chosen by the `Exec` constructor argument. Host memory
 * (`std::span`) works on every backend; a GPU backend streams it through
 * pipelined H2D/kernel/D2H transfers and blocks until the result is on the
 * host. Device memory (`DeviceSpan`) works on GPU backends only (the CPU
 * backend throws std::invalid_argument), as one asynchronous launch on the
 * given Stream.
 */
class DDMT {
public:
    /**
     * @brief Constructs a DDMT engine with a linear DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_max Maximum trial DM in pc/cm^3.
     * @param dm_step Linear spacing between DM trials in pc/cm^3.
     * @param dm_min Minimum trial DM in pc/cm^3 (default: 0).
     * @param exec Backend and its resources (default: CPU, 1 thread), e.g.
     * `Exec::cpu(8)` or `Exec::cuda(0)`.
     * @param nbits Input precision: 32 (float), or 1, 2, 4, 8, 16 (packed
     * integer).
     * @param kill_mask Optional per-channel mask (size nchans, 1=keep,
     * 0=exclude from sum).
     * @param nbeams Number of independent beams batched through one instance
     * (default: 1).
     */
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min                       = 0.0F,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMT engine with an explicit DM trial array.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_arr Explicit span of trial DMs in pc/cm^3.
     * @param exec Backend and its resources (default: CPU, 1 thread).
     * @param nbits Input precision (default: 32).
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams (default: 1).
     */
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         std::span<const float> dm_arr,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMT engine with an optimal Lina Levin DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param levin LevinConfig specifying dm_start, dm_end, pulse_width, and
     * tol.
     * @param exec Backend and its resources (default: CPU, 1 thread).
     * @param nbits Input bit precision.
     * @param kill_mask Optional per-channel mask.
     * @param nbeams Number of batched beams.
     */
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const plans::LevinConfig& levin,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /**
     * @brief Constructs a DDMT engine directly from a pre-configured
     * DDMTPlan.
     * @param plan Pre-initialized DDMT plan.
     * @param exec Backend and its resources (default: CPU, 1 thread).
     * @param nbeams Number of batched beams.
     */
    explicit DDMT(const plans::DDMTPlan& plan,
                  Exec exec       = {},
                  SizeType nbeams = 1);

    virtual ~DDMT();
    DDMT(DDMT&&) noexcept;
    DDMT& operator=(DDMT&&) noexcept;
    DDMT(const DDMT&)            = delete;
    DDMT& operator=(const DDMT&) = delete;

    /// @brief Read-only reference to underlying DDMT execution plan
    [[nodiscard]] const plans::DDMTPlan& get_plan() const noexcept;
    /// @brief Number of beams batched per execute() call
    [[nodiscard]] SizeType get_nbeams() const noexcept;
    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;
    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;
    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;

    /**
     * @brief Dedisperses a beam-major float32 waterfall (nbeams, nchans,
     * nsamps).
     *
     * Requires get_plan().get_nbits() == 32.
     *
     * @param waterfall Input waterfall buffer.
     * @param dmt Output DM-time buffer: shape (nbeams, ndm,
     * get_output_nsamps(nsamps)).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Dedisperses a float32 waterfall resident in device memory (GPU
     * backends), as one launch on @p stream with no internal chunking.
     *
     * The stream history is kept on the device, so the call is fully
     * asynchronous with respect to the host; calls made on different streams
     * are ordered after each other on the device automatically.
     */
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

    /**
     * @brief Dedisperses a beam-major packed-integer waterfall (nbeams, nchans,
     * nsamps).
     *
     * Each channel row is packed at get_plan().get_nbits() bits per sample,
     * LSB-first. Summation accumulates in 32-bit integer arithmetic without
     * float conversions.
     *
     * @param waterfall_packed Input packed bytes.
     * @param nsamps Explicit time sample count per channel.
     * @param dmt Output int32 DM-time buffer: shape (nbeams, ndm,
     * get_output_nsamps(nsamps)).
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt);

    /// @brief Packed-integer analogue of the device-memory execute().
    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nsamps,
                 DeviceSpan<int32_t> d_dmt,
                 Stream stream = {});

    /**
     * @brief Dedisperses a time-major packed filterbank of shape (nbeams,
     * nsamps, nchans).
     *
     * Matches standard SIGPROC `.fil` disk format and telescope streaming DAQ
     * buffers. The block is transposed to channel-major form and then
     * streamed exactly like the channel-major packed execute(): it shares the
     * same cross-call history (get_output_nsamps() applies, and the two
     * overloads may be mixed within one stream). On a GPU backend the host
     * overload streams the input through the device in chunks (see
     * set_gulp_size()) and the transpose runs on the device.
     *
     * @param filterbank_packed Input packed bytes in time-major order.
     * @param nsamps Sample count in time dimension.
     * @param dmt Output int32 DM-time buffer: shape (nbeams, ndm,
     * get_output_nsamps(nsamps)).
     */
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt);

    /// @brief Device-memory analogue of execute_time_major() (GPU backends).
    void execute_time_major(DeviceSpan<const uint8_t> d_filterbank_packed,
                            SizeType nsamps,
                            DeviceSpan<int32_t> d_dmt,
                            Stream stream = {});

    /**
     * @brief Computes output sample count for an incoming chunk of
     * input_nsamps.
     *
     * On cold start: returns input_nsamps - max_delay.
     * Once stream is warm: returns input_nsamps.
     *
     * @param input_nsamps Incoming chunk size in samples.
     * @return Number of valid output time samples produced.
     */
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept;

    /// @brief Discards retained cross-call history, returning execute to
    /// cold-start mode
    void reset_history() noexcept;

    /**
     * @brief Sets the chunk length, in input samples, that host-memory
     * execute() calls on a GPU backend stream through the device (pinned
     * staging, overlapped copy/kernel/copy). Larger chunks use more pinned
     * and device memory (about nbeams * ndm * gulp_size output values per
     * buffer, double buffered). 0 restores the default (65536). Results do
     * not depend on it. The CPU backend stores it but does not use it.
     */
    void set_gulp_size(SizeType gulp_size);
    /// @brief Current host-path chunk length in input samples.
    [[nodiscard]] SizeType get_gulp_size() const noexcept;

    /// @brief Size of a fully-warmed-up history state: in floats (nbeams *
    /// nchans * max_delay) when nbits == 32, or in bytes (nbeams * nchans *
    /// packed_row_bytes(max_delay, nbits)) when nbits is a packed width.
    [[nodiscard]] SizeType history_state_size() const noexcept;

    // Errors: every execute()/save_history()/load_history() overload throws
    // std::invalid_argument for the wrong overload (float vs packed nbits) or
    // a buffer size mismatch; save_history() throws std::logic_error before
    // the stream holds a full history (fewer than max-delay samples seen).

    /// @brief Saves current float history state to caller-owned storage (nbits
    /// == 32)
    void save_history(std::span<float> out) const;

    /// @brief Saves current packed integer history state to caller-owned
    /// storage (nbits in {1,2,4,8,16})
    void save_history(std::span<uint8_t> out) const;

    /// @brief Restores previously saved float history state (nbits == 32)
    void load_history(std::span<const float> in);

    /// @brief Restores previously saved packed history state (nbits in
    /// {1,2,4,8,16})
    void load_history(std::span<const uint8_t> in);

    /// @brief Device-memory analogues of save_history()/load_history() (GPU
    /// backends), enqueued on @p stream.
    void save_history(DeviceSpan<float> d_out, Stream stream = {}) const;
    void save_history(DeviceSpan<uint8_t> d_out, Stream stream = {}) const;
    void load_history(DeviceSpan<const float> d_in, Stream stream = {});
    void load_history(DeviceSpan<const uint8_t> d_in, Stream stream = {});

    template <typename Alloc1 = std::allocator<float>,
              typename Alloc2 = std::allocator<float>>
    void execute(const std::vector<float, Alloc1>& waterfall,
                 std::vector<float, Alloc2>& dmt) {
        execute(std::span<const float>(waterfall), std::span<float>(dmt));
    }

    template <typename Alloc1 = std::allocator<uint8_t>,
              typename Alloc2 = std::allocator<int32_t>>
    void execute(const std::vector<uint8_t, Alloc1>& waterfall_packed,
                 SizeType nsamps,
                 std::vector<int32_t, Alloc2>& dmt) {
        execute(std::span<const uint8_t>(waterfall_packed), nsamps,
                std::span<int32_t>(dmt));
    }

    template <typename Alloc1 = std::allocator<uint8_t>,
              typename Alloc2 = std::allocator<int32_t>>
    void
    execute_time_major(const std::vector<uint8_t, Alloc1>& filterbank_packed,
                       SizeType nsamps,
                       std::vector<int32_t, Alloc2>& dmt) {
        execute_time_major(std::span<const uint8_t>(filterbank_packed), nsamps,
                           std::span<int32_t>(dmt));
    }

    template <typename Alloc = std::allocator<float>>
    void save_history(std::vector<float, Alloc>& out) const {
        save_history(std::span<float>(out));
    }

    template <typename Alloc = std::allocator<uint8_t>>
    void save_history(std::vector<uint8_t, Alloc>& out) const {
        save_history(std::span<uint8_t>(out));
    }

    template <typename Alloc = std::allocator<float>>
    void load_history(const std::vector<float, Alloc>& in) {
        load_history(std::span<const float>(in));
    }

    template <typename Alloc = std::allocator<uint8_t>>
    void load_history(const std::vector<uint8_t, Alloc>& in) {
        load_history(std::span<const uint8_t>(in));
    }

protected:
    /// Engine an instance runs: DDMT always sums directly; SDMT (a thin
    /// subclass) selects the shared-partial-sum CPU engine.
    enum class EngineKind : uint8_t { kDirect, kSharedSums };
    DDMT(const plans::DDMTPlan& plan,
         Exec exec,
         SizeType nbeams,
         EngineKind kind);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
