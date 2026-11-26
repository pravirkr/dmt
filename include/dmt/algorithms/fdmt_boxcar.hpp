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
    /// Of total(), the bytes in host memory on a GPU backend (host plan
    /// copies, host result storage, the per-call mask noise model). 0 on
    /// the CPU, where every byte is host memory and counts in the budget.
    SizeType host = 0;
    [[nodiscard]] SizeType total() const noexcept {
        return plan + state + history + peaks + workspace + candidates +
               staging;
    }
    /// The part of total() that max_memory_bytes bounds: device memory on
    /// a GPU, everything on the CPU.
    [[nodiscard]] SizeType budgeted() const noexcept { return total() - host; }
};

/**
 * @brief What an FDMTBoxcar built with the same arguments would allocate,
 * without allocating it (FDMTBoxcar::estimate_memory()).
 */
struct FDMTBoxcarMemoryEstimate {
    /// Equal to get_memory_usage() of that instance after construction.
    FDMTBoxcarMemoryUsage usage{};
    /// Its get_chunk_nsamps().
    SizeType chunk_nsamps{0};
    /// The transform's path, as summary() shows it ("Transform: ..."; empty
    /// for the tree path of FDMTBoxcar).
    std::string search_path;
    /// Its get_search_latency() (FDMTFFTBoxcar: get_output_latency()):
    /// samples a stream sample waits for before every row has searched it,
    /// and what drain() pads (in whole blocks).
    uint64_t search_latency{0};
    /// The max_memory_bytes it was made with; from
    /// FDMTBoxcar::estimate_minimum_memory(), the smallest that builds (at
    /// least usage.budgeted(): CPU path choices keep some headroom).
    SizeType max_memory_bytes{0};
};

/**
 * @brief Streaming valid-mode FDMT followed by analytical boxcar detection.
 *
 * Input is finite float32, (nbeams, nchans, nsamps), low frequency first.
 * FDMTBoxcar does no preprocessing: the noise model needs each channel
 * normalized to zero mean and the common noise sigma (noise_sigma, 1 by
 * default), and white in time over at least max_boxcar samples. Analytical
 * S/N assumes independent channel/time noise before the modeled FDMT
 * filtering;
 * masking other than kill_mask (below) or additional correlations require
 * upstream calibration. Red noise (slow drifts correlated over many
 * samples) inflates the S/N of wide boxes, and the search keeps no
 * baseline beyond max_boxcar samples, so detrend each channel upstream on
 * timescales several times the widest box and keep max_boxcar at about
 * 100 ms or less.
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
 * Memory: max_memory_bytes is the only memory setting (the block length
 * nsamps is the caller's I/O unit and does not change the FDMT's memory,
 * though a CPU kill mask copies the block into budgeted staging;
 * chunk_nsamps is an expert override, 0 = automatic, and a forced
 * chunk can throw where the automatic one fits). Plan with
 * estimate_memory() (usage, chunk, path and search latency of a budget)
 * and estimate_minimum_memory() (the smallest budget that builds); they
 * report what the instance owns, so add the caller's input blocks. A
 * larger budget can buy a faster configuration (a simple measured rule,
 * close to the fastest that fits, not faster at every budget); every budget
 * from the minimum up builds, with the same candidates to float precision. On a
 * GPU, construction also checks the chosen configuration against the device's
 * free memory and throws std::bad_alloc before allocating when it does not fit
 * (the choice never depends on free memory). max_memory_bytes (default
 * kFDMTDefaultMaxMemory, 10 GiB) bounds what the instance allocates where it
 * runs (get_memory_usage().budgeted()): everything on the CPU; device memory on
 * a GPU, where the host plan copies, host result storage and per-call mask
 * noise model are reported (get_memory_usage().host) but not budgeted. The
 * boxcar carry, peaks and candidate buffers are fixed by the plan and config;
 * the FDMT takes a fast path within the rest (see FDMT). Construction
 * throws std::invalid_argument when nothing fits. On a GPU, host-memory input
 * is copied to the device one chunk at a time into a buffer reserved at
 * construction inside that budget (a few MiB to ~100 MiB, held whether
 * or not host input is used), so host calls allocate nothing. On the
 * skewed-window path (plans whose tree history exceeds the budget; on the
 * CPU also packed input with blocks over 2048 samples) the rows
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
 * leaves sigma too high by sqrt(N / (N - k)). On the CPU each block is
 * copied to a staging buffer to zero its killed rows (counted in
 * get_memory_usage().staging and the memory budget); on a GPU the
 * transform reads killed rows as zero (one byte per beam and channel, no
 * copy of the block).
 *
 * Per-call kill mask: execute(waterfall, kill_mask) also takes a mask for that
 * block (same format and convention; nonzero is active). Packed input (CPU and
 * GPU, every screen): channels it kills but the constructor mask keeps read the
 * code input_mean (which must be an integer), so the expected sums stay the
 * constructor plan's and only the cutoffs and S/N scales follow the rules
 * below; per-width peaks spanning a change of tables are ranked by S/N. It
 * applies to every beam and sample of the block and is ANDed with the
 * constructor mask; an empty mask, or the overloads without one, mean the
 * constructor mask alone. The mask is read before the call returns (device
 * calls stay asynchronous). Killed input never reaches the transform, and
 * samples keep the mask they were ingested with.
 * Endpoints whose input windows reach back before their block (the first S + W
 * - 1 of each row, S its FDMT support with every channel active, W the width)
 * are thresholded and scored with the variance of the history mask when
 * it differs from the block's: the block's channels and every channel
 * live within R = max(S) + max_boxcar - 1 samples before it (for blocks of
 * at least R samples, the OR of this and the previous block's masks). No
 * extra false alarms; S/N there slightly low (at most
 * sqrt(V_history / V_true)). Elsewhere S/N is exact for the block's mask.
 * Any block length works: the masks' memory does not grow with the number
 * of blocks a window or the search latency spans. When blocks are shorter
 * than the latency needs (more than the blocks of R samples would), the
 * oldest entry kept stands for the older blocks still searched with one
 * catch-all table: every channel live within R samples before the oldest
 * endpoint still searched, up to the newest block (conservative, as the
 * history mask). The first per-call mask requires the memory for the
 * noise tables within max_memory_bytes (on a GPU: their device copies),
 * else throws std::invalid_argument before any state changes.
 * With BoxcarConfig::per_call_masks the memory is checked at construction
 * instead (and by estimate_memory()), reserved before the transform's path
 * is chosen, and get_memory_usage() reports the reserved table sets (2 per
 * block kept, plus 3 cached, plus a catch-all when blocks are short) from
 * the start, so it stays at its estimate.
 * The noise tables of a new mask are computed on the host with
 * exec.nthreads threads (on GPUs too); the last three masks are cached.
 * skip_cold_prefix keeps the constructor mask's warm-up. reset_history()
 * clears the mask state; drain() pads with the last block's mask.
 * get_search_plans() shows the constructor mask's tables only.
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
    /**
     * @brief Memory an instance built with these arguments would allocate,
     * and its chunk, computed without allocating it: the constructor's
     * decisions and tables, with every working buffer sized but not
     * allocated (exact: the constructor allocates exactly these sizes).
     * Throws as the constructor would when nothing fits max_memory_bytes.
     * A GPU backend needs the device but allocates no device memory; the
     * CUDA context is not counted (nor in get_memory_usage()). Per-call
     * kill masks are included when declared (BoxcarConfig::per_call_masks);
     * otherwise the first per-call mask adds its noise model, tables and
     * buffers then (not included).
     */
    [[nodiscard]] static FDMTBoxcarMemoryEstimate
    estimate_memory(const plans::FDMTPlan& plan,
                    BoxcarConfig config                = {},
                    bool use_box_smearing              = true,
                    Exec exec                          = {},
                    SizeType nbeams                    = 1,
                    SizeType fuse_levels               = kFDMTAutoFuse,
                    SizeType chunk_nsamps              = 0,
                    SizeType max_memory_bytes          = kFDMTDefaultMaxMemory,
                    std::span<const uint8_t> kill_mask = {});
    /**
     * @brief The smallest configuration of these arguments: estimate_memory()
     * at the smallest max_memory_bytes an instance can be built with (the
     * result's max_memory_bytes; one byte less throws). A few estimates
     * long. The minimum is set by the boxcar's buffers (rows x
     * max_boxcar carry, peaks, candidates) and the FDMT's smallest path at
     * its shortest chunk; any larger budget also builds. Throws as
     * estimate_memory() for anything but the budget (a configuration that
     * is invalid at its smallest path throws here too).
     */
    [[nodiscard]] static FDMTBoxcarMemoryEstimate
    estimate_minimum_memory(const plans::FDMTPlan& plan,
                            BoxcarConfig config                = {},
                            bool use_box_smearing              = true,
                            Exec exec                          = {},
                            SizeType nbeams                    = 1,
                            SizeType fuse_levels               = kFDMTAutoFuse,
                            SizeType chunk_nsamps              = 0,
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
    /// Packed input with a per-call kill mask (CPU and GPU, every screen):
    /// channels it kills read the code input_mean.
    [[nodiscard]] CandidateResultView
    execute(std::span<const uint8_t> waterfall,
            std::span<const uint8_t> kill_mask);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const uint8_t> waterfall,
            std::span<const uint8_t> kill_mask,
            Stream stream = {});
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
     * The padding reads, stages and allocates no input block (on a GPU,
     * device input never needs host memory or the host-input staging), so
     * get_memory_usage() stays at its estimate.
     * A host call: after device calls, synchronize their stream first. It
     * ends the stream: execute() throws until reset_history().
     */
    [[nodiscard]] CandidateResultView drain();
    /**
     * End of stream with a short last block: searches the first @p n_tail
     * samples of each channel of @p tail, then drains as drain() does.
     * @p tail is a whole block, laid out as for execute() (float or packed
     * by config.input_nbits; host or device memory); its samples from
     * @p n_tail on are padding, never searched, and (the FDMT being causal)
     * they change no row before the stream's end. get_searched_nsamps()
     * then ends at the tail's n_tail-th sample, and no candidate reaches
     * past it. Uses the constructor's kill mask. Device input is read on
     * @p stream; the call returns host results, as drain().
     * @throws std::invalid_argument unless 1 <= n_tail <= nsamps.
     */
    [[nodiscard]] CandidateResultView drain(std::span<const float> tail,
                                            SizeType n_tail);
    [[nodiscard]] CandidateResultView drain(std::span<const uint8_t> tail,
                                            SizeType n_tail);
    [[nodiscard]] CandidateResultView
    drain(DeviceSpan<const float> tail, SizeType n_tail, Stream stream = {});
    [[nodiscard]] CandidateResultView
    drain(DeviceSpan<const uint8_t> tail, SizeType n_tail, Stream stream = {});
    /// FDMT output samples [0, n) whose trailing endpoints every row has
    /// searched (after drain(), the whole stream; for a chan_dm plan this is
    /// the raw FDMT count without subtracting
    /// FDMTPlan::get_level0_causal_delay()).
    [[nodiscard]] uint64_t get_searched_nsamps() const noexcept;
    /**
     * Every candidate emitted by a later execute(), flush() or drain() has
     * begin_sample (hence peak_sample and end_sample) >= this value: the
     * smallest begin_sample of the excursions still open (any beam, row,
     * width), or get_searched_nsamps() when none is open — for a chan_dm plan
     * minus FDMTPlan::get_level0_causal_delay() clamped at 0, matching
     * candidate times. Non-decreasing between reset_history() calls, 0 after
     * reset_history(). A pipeline can release candidate groups that end before
     * it, however long an excursion stays open. A host call; after device
     * calls, synchronize their stream first (as for reading a device result).
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
