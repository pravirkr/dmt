#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"

namespace dmt::algorithms {

/// Allocated memory in bytes. The full DM-time output is never allocated.
struct FDMTBoxcarMemoryUsage {
    SizeType plan       = 0;
    SizeType state      = 0;
    SizeType history    = 0; ///< FDMT and boxcar sample history.
    SizeType peaks      = 0;
    SizeType workspace  = 0;
    SizeType candidates = 0; ///< Includes host result storage on GPUs.
    SizeType staging    = 0;
    [[nodiscard]] SizeType total() const noexcept {
        return plan + state + history + peaks + workspace + candidates +
               staging;
    }
};

/**
 * @brief Streaming valid-mode FDMT followed by analytical boxcar detection.
 *
 * Input is finite float32, (nbeams, nchans, nsamps), low frequency first,
 * normalized to zero mean and known common noise sigma. Analytical S/N
 * assumes independent channel/time noise before the modeled FDMT filtering;
 * masking other than kill_mask (below) or additional correlations require
 * upstream calibration.
 * For packed unsigned input set config.input_nbits (1/2/4/8/16), input_mean
 * and noise_sigma in code units, and execute byte spans. Packed rows follow
 * FDMT byte order with independent row padding; 32 denotes float32 input.
 *
 * Subband search: a plan built with subband_depth d is searched at depths
 * 0..d. Depth k covers 2^k equal tree subbands and the 2^k - 1 staggered
 * subbands of the same width centred on their boundaries, so a narrowband
 * burst anywhere in the band falls mostly inside one searched subband
 * (d = 1: full band, two halves and the central half). Every subband row
 * has its own analytical noise model; candidates carry the subband's depth,
 * index, channel extent and staggered flag.
 *
 * Consecutive non-overlapping blocks form one stream. Root samples are
 * merged directly into the search without a DM-time output plane. Pending
 * contiguous excursions cross blocks; drain() ends the stream (it searches
 * any samples still behind the search latency and emits the open
 * excursions); flush() only emits them.
 * Host calls block on every backend. Device calls are asynchronous; order
 * operations on one stream, or synchronize explicitly when changing streams.
 * GPU reset_history() is also ordered asynchronously on its supplied stream.
 * An instance is not safe for concurrent calls. CPU device calls throw.
 *
 * Memory: max_memory_bytes (default kFDMTDefaultMaxMemory, 10 GiB) bounds
 * everything the instance allocates (get_memory_usage().total()). The
 * boxcar carry, peaks and candidate buffers are fixed by the plan and
 * config; the FDMT takes its fastest path within the rest (see FDMT).
 * Construction throws std::invalid_argument when nothing fits. On the
 * skewed-window path (plans whose tree history exceeds the budget) the rows
 * of a subband search are searched with a fixed per-row latency of up to
 * the dispersion delay from the subband to the bottom of the band (that of
 * the full-band search, get_search_latency()); flush() adds no samples,
 * so at end of stream call drain(), which searches those samples too.
 *
 * Kill mask: kill_mask is empty (keep every channel) or has nchans entries,
 * low frequency first; channels with kill_mask[c] == 0 read as zero (code
 * 0 for packed input) and are left out of every row's noise model and
 * packed mean, so S/N stays calibrated on the remaining channels (a pulse
 * scores the S/N of its signal in them). Zeroing channels upstream instead
 * leaves sigma too high by sqrt(N / (N - k)). Each block is copied to a
 * staging buffer to zero its killed rows (counted in
 * get_memory_usage().staging and the memory budget).
 *
 * Per-call kill mask (float32 input): execute(waterfall, kill_mask) also
 * takes a mask for that block (same format and convention; nonzero is
 * active). It applies to every beam and sample of the block and is ANDed
 * with the constructor mask; an empty mask, or the overloads without one,
 * mean the constructor mask alone. The mask is read before the call
 * returns (device calls stay asynchronous). Killed input never reaches the
 * transform, and samples keep the mask they were ingested with. When the
 * mask changes, the endpoints whose input windows reach back into the
 * previous block (the first S + W - 1 of each row, S its FDMT support with
 * every channel active, W the width) are thresholded and scored with the
 * variance of the OR of both masks: no extra false alarms, S/N there
 * slightly low (at most sqrt(V_or / V_true)). Elsewhere S/N is exact for
 * the block's mask. The first per-call mask requires blocks of at least
 * max(S) + max_boxcar - 1 samples (so that a window spans at most two
 * blocks) and the memory for the noise tables within max_memory_bytes;
 * either failure throws std::invalid_argument before any state changes.
 * The noise tables of a new mask are computed on the host with
 * exec.nthreads threads (on GPUs too); the last three masks are cached.
 * skip_cold_prefix keeps the constructor mask's warm-up. reset_history()
 * clears the mask state; drain() pads with the last block's mask.
 * get_search_plans() shows the constructor mask's tables only. Packed
 * input takes no per-call mask.
 */
class FDMTBoxcar {
public:
    explicit FDMTBoxcar(plans::FDMTPlan plan,
                        BoxcarConfig config       = {},
                        bool use_box_smearing     = true,
                        Exec exec                 = {},
                        SizeType nbeams           = 1,
                        SizeType fuse_levels      = kFDMTAutoFuse,
                        SizeType chunk_nsamps     = 0,
                        SizeType max_memory_bytes = kFDMTDefaultMaxMemory,
                        std::span<const uint8_t> kill_mask = {});
    ~FDMTBoxcar();
    FDMTBoxcar(FDMTBoxcar&&) noexcept;
    FDMTBoxcar& operator=(FDMTBoxcar&&) noexcept;
    FDMTBoxcar(const FDMTBoxcar&)            = delete;
    FDMTBoxcar& operator=(const FDMTBoxcar&) = delete;

    [[nodiscard]] CandidateResultView execute(std::span<const float> waterfall);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const float> waterfall, Stream stream = {});
    /// One block with its own kill mask (empty or nchans entries, 0 =
    /// killed), ANDed with the constructor mask; see the class comment.
    [[nodiscard]] CandidateResultView
    execute(std::span<const float> waterfall,
            std::span<const uint8_t> kill_mask);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const float> waterfall,
            std::span<const uint8_t> kill_mask,
            Stream stream = {});
    /// Packed rows follow FDMT byte order and have independent byte padding.
    [[nodiscard]] CandidateResultView
    execute(std::span<const uint8_t> waterfall);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const uint8_t> waterfall, Stream stream = {});
    [[nodiscard]] CandidateResultView flush();
    [[nodiscard]] CandidateDeviceResultView flush_device(Stream stream = {});
    /**
     * End of stream: searches the stream up to its last sample, then emits
     * the open excursions as flush() does. Without it the last
     * get_search_latency() samples are never searched. Endpoints after
     * the last sample are not searched; the padding pushed through the
     * transform does not change the rows before it (the FDMT is causal).
     * A host call: after device calls, synchronize their stream first. It
     * ends the stream: execute() throws until reset_history().
     */
    [[nodiscard]] CandidateResultView drain();
    /// Stream samples [0, n) whose trailing endpoints every row has searched
    /// (after drain(), the whole stream).
    [[nodiscard]] uint64_t get_searched_nsamps() const noexcept;
    /**
     * Every candidate emitted by a later execute(), flush() or drain() has
     * begin_sample (hence peak_sample and end_sample) >= this value: the
     * smallest begin_sample of the excursions still open (any beam, row,
     * width), or get_searched_nsamps() when none is open. Non-decreasing
     * between reset_history() calls; get_searched_nsamps() after flush() or
     * drain(), 0 after reset_history(). A pipeline can release candidate
     * groups that end before it, however long an excursion stays open. A
     * host call; after device calls, synchronize their stream first (as
     * for reading a device result).
     */
    [[nodiscard]] uint64_t get_candidate_frontier() const noexcept;
    /// Input samples that a stream sample waits for before every searched
    /// row has searched it: the largest subband row lag on the skewed paths,
    /// 0 otherwise.
    [[nodiscard]] uint64_t get_search_latency() const noexcept;
    [[nodiscard]] CandidateResultView copy_result(Stream stream = {});
    /// Discard all histories and pending peaks; restart stream sample 0.
    void reset_history(Stream stream = {});

    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    [[nodiscard]] const BoxcarConfig& get_config() const noexcept;
    [[nodiscard]] std::span<const plans::BoxcarPlanePlan>
    get_search_plans() const noexcept;
    [[nodiscard]] SizeType get_nbeams() const noexcept;
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept;
    [[nodiscard]] SizeType get_fuse_levels() const noexcept;
    [[nodiscard]] Backend backend() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] int nthreads() const noexcept;
    [[nodiscard]] FDMTBoxcarMemoryUsage get_memory_usage() const noexcept;
    [[nodiscard]] std::string summary() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
