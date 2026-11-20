#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dmt/algorithms/fdmt_boxcar.hpp"

#include "dmt/boxcar_noise.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::algorithms::detail {

/// Search planes of `plan`. Channels with kill_mask[c] == 0 are excluded
/// from the noise model (empty: keep all); `extra_warmup` is added to every
/// row's cold support (FDMT-FFT's interpolation guard).
[[nodiscard]] std::vector<plans::BoxcarPlanePlan>
make_boxcar_plans(const plans::FDMTPlan& plan,
                  const BoxcarConfig& config,
                  bool smear,
                  std::span<const uint8_t> kill_mask = {},
                  uint64_t extra_warmup              = 0);

/// Channels whose input FDMTBoxcar zeroes (kill_mask[c] == 0), as runs
/// [begin, end) of consecutive channels; none for an empty mask.
[[nodiscard]] inline std::vector<std::pair<SizeType, SizeType>>
boxcar_kill_runs(std::span<const uint8_t> kill_mask) {
    std::vector<std::pair<SizeType, SizeType>> runs;
    for (SizeType c = 0; c < kill_mask.size(); ++c) {
        if (kill_mask[c] != 0) {
            continue;
        }
        if (!runs.empty() && runs.back().second == c) {
            ++runs.back().second;
        } else {
            runs.emplace_back(c, c + 1);
        }
    }
    return runs;
}

/**
 * @brief A group of the width screen: consecutive widths [w_lo, w_hi] with
 * w_hi < 2 w_lo. For each width w of the group the boxcar sum
 * P(t) - P(t - w) is at most P(t) - min(P[t - w_hi .. t - w_lo]), with the
 * same rounding (it is monotonic), so a tile whose bound stays under the
 * group's smallest threshold has no hit for any of its widths. The range
 * minimum is two overlapping runs of 2^level entries.
 */
struct BoxcarScreenGroup {
    uint32_t f_lo;  ///< first filter index
    uint32_t f_hi;  ///< one past the last filter index
    uint32_t w_lo;  ///< smallest width
    uint32_t w_hi;  ///< largest width
    uint32_t level; ///< floor(log2(w_hi - w_lo + 1))
};

/**
 * @brief Width groups of the screen for `widths` (ascending), or none when
 * grouping would not pay: each group check and each range-minimum level
 * costs about one width check, so the screen is used only when groups plus
 * levels are under half the widths (163 widths form 8 groups; 12 widths at
 * spacing 1.5 form 7 and run faster ungrouped). Trapezoid widths are not
 * bounded by the range minimum and stay single.
 */
[[nodiscard]] inline std::vector<BoxcarScreenGroup>
boxcar_screen_groups(std::span<const uint32_t> widths, bool use_trapezoid) {
    std::vector<BoxcarScreenGroup> groups;
    uint32_t levels      = 0;
    const auto trapezoid = [&](uint32_t w) {
        return use_trapezoid && trapezoid_edge_width(w) > 0;
    };
    for (SizeType f = 0; f < widths.size();) {
        const uint32_t w_lo = widths[f];
        SizeType g          = f + 1;
        while (!trapezoid(w_lo) && g < widths.size() && widths[g] < 2 * w_lo &&
               !trapezoid(widths[g])) {
            ++g;
        }
        const uint32_t w_hi = widths[g - 1];
        const auto level =
            static_cast<uint32_t>(std::bit_width(w_hi - w_lo + 1U) - 1);
        groups.push_back({static_cast<uint32_t>(f), static_cast<uint32_t>(g),
                          w_lo, w_hi, level});
        levels = std::max(levels, level);
        f      = g;
    }
    if (2 * (groups.size() + levels) >= widths.size()) {
        groups.clear();
    }
    // Ascending levels, so a range-minimum table raised in place serves
    // every group (the last group, cut short by max_width, can have a lower
    // level than the one before). Order does not change the result.
    std::ranges::stable_sort(groups, {}, &BoxcarScreenGroup::level);
    return groups;
}

/// Per-plane row description, shared by CPU and device root sample readers.
struct BoxcarRow {
    SizeType row        = 0;
    SizeType tail       = 0;
    SizeType head       = SIZE_MAX;
    SizeType delay      = 0;
    SizeType hist       = 0;
    uint64_t warmup     = 0;
    bool unmaterialized = false;
    uint32_t plane      = 0; ///< index of the row's search plane
    RawCandidate candidate{};
};

/// BoxcarConfig::max_candidates_per_plane: the planes' shares must fit
/// max_candidates.
inline void check_plane_share(const BoxcarConfig& config, SizeType nplanes) {
    const SizeType cap = config.max_candidates_per_plane;
    if (cap != 0 && (nplanes > config.max_candidates / cap)) {
        throw std::invalid_argument(
            "FDMTBoxcar: max_candidates_per_plane times the number of "
            "searched planes (" +
            std::to_string(nplanes) + ") must not exceed max_candidates");
    }
}

/// Per-plane counts: the first plane of each subband_level, so that a
/// candidate's plane is offsets[subband_level] + subband_idx. Planes come
/// grouped by subband_level with subband_idx 0, 1, ... (make_boxcar_plans);
/// anything else is a logic error.
[[nodiscard]] inline std::vector<uint32_t>
boxcar_level_offsets(std::span<const plans::BoxcarPlanePlan> planes) {
    std::vector<uint32_t> offsets;
    for (SizeType i = 0; i < planes.size(); ++i) {
        const auto& p = planes[i];
        if (p.subband_level >= offsets.size()) {
            offsets.resize(p.subband_level + 1, UINT32_MAX);
        }
        if (offsets[p.subband_level] == UINT32_MAX) {
            offsets[p.subband_level] = static_cast<uint32_t>(i);
        }
        if (offsets[p.subband_level] + p.subband_idx != i) {
            throw std::logic_error(
                "FDMTBoxcar: search planes out of (level, subband) order");
        }
    }
    return offsets;
}

[[nodiscard]] std::vector<BoxcarRow>
make_boxcar_rows(const plans::FDMTPlan& plan,
                 std::span<const plans::BoxcarPlanePlan> planes);

/// Ring slot of the call's first sample in an unmaterialized row's tree
/// history (sample_begin mod delay), so readers of many samples take one
/// 64-bit remainder per row instead of one per sample. SIZE_MAX when the
/// row reads no ring.
DMT_HD inline SizeType boxcar_ring0(const FDMTSearchInput& input,
                                    const BoxcarRow& row) noexcept {
    return input.history_ring && row.head != SIZE_MAX && row.delay > 0
               ? input.sample_begin % row.delay
               : SIZE_MAX;
}

/// History slot of sample t < row.delay: (sample_begin + t) mod delay from
/// ring0 = boxcar_ring0() (SIZE_MAX: compute it), or t for a FIFO history.
DMT_HD inline SizeType boxcar_history_slot(const FDMTSearchInput& input,
                                           const BoxcarRow& row,
                                           SizeType t,
                                           SizeType ring0) noexcept {
    if (!input.history_ring) {
        return t;
    }
    if (ring0 == SIZE_MAX) {
        return (input.sample_begin + t) % row.delay;
    }
    const SizeType slot = ring0 + t;
    return slot >= row.delay ? slot - row.delay : slot;
}

/// The same merge addition as FDMT for unmaterialized rows (root and staggered
/// subbands); history is not mutated by the reader. `ring0`: see
/// boxcar_history_slot().
DMT_HD inline float boxcar_sample(const FDMTSearchInput& input,
                                  const BoxcarRow& row,
                                  SizeType beam,
                                  SizeType t,
                                  bool root      = false,
                                  SizeType ring0 = SIZE_MAX) noexcept {
    if (!(row.unmaterialized || root) || input.materialized_root) {
        const float* state = input.state + beam * input.beam_stride;
        return state[row.row * input.row_stride + t];
    }
    const float* prev =
        (input.prev_state != nullptr ? input.prev_state : input.state) +
        beam * input.beam_stride;
    const float tail = prev[row.tail * input.row_stride + t];
    if (row.head == SIZE_MAX) {
        return tail;
    }
    float head;
    if (t >= row.delay) {
        head = prev[row.head * input.row_stride + t - row.delay];
    } else {
        const SizeType slot = boxcar_history_slot(input, row, t, ring0);
        head = input.history[beam * input.history_stride + row.hist -
                             input.history_base + slot];
    }
    return tail + head;
}

struct BoxcarPeak {
    uint64_t begin   = 0;
    uint64_t end     = 0;
    uint64_t peak    = 0;
    float snr        = 0.0F;
    uint32_t filter  = 0;
    bool open        = false;
    uint64_t raw_sum = 0;
};

DMT_HD inline bool boxcar_better(float snr,
                                 uint64_t sample,
                                 uint32_t filter,
                                 const BoxcarPeak& peak) noexcept {
    return !peak.open || snr > peak.snr ||
           (snr == peak.snr && (sample < peak.peak ||
                                (sample == peak.peak && filter < peak.filter)));
}

/// Feed one point; emission is supplied by the CPU/GPU buffer owner.
template <typename Emit>
DMT_HD void boxcar_peak_step(BoxcarPeak& peak,
                             bool above,
                             float snr,
                             uint64_t sample,
                             uint32_t filter,
                             Emit emit) {
    if (!above) {
        if (peak.open) {
            emit(peak);
            peak.open = false;
        }
        return;
    }
    const bool better = boxcar_better(snr, sample, filter, peak);
    if (!peak.open) {
        peak.begin = sample;
    }
    peak.end = sample;
    if (better) {
        peak.snr    = snr;
        peak.peak   = sample;
        peak.filter = filter;
    }
    peak.open = true;
}

template <typename TInt>
DMT_HD inline uint64_t
boxcar_integer_sample_typed(const TInt* state_base,
                            const TInt* prev_base,
                            const TInt* hist_base,
                            const FDMTSearchInput& input,
                            const BoxcarRow& row,
                            SizeType beam,
                            SizeType t,
                            bool root      = false,
                            SizeType ring0 = SIZE_MAX) noexcept {
    if (!(row.unmaterialized || root) || input.materialized_root) {
        const TInt* state = state_base + beam * input.beam_stride;
        return static_cast<uint64_t>(state[row.row * input.row_stride + t]);
    }
    const TInt* prev = (prev_base != nullptr ? prev_base : state_base) +
                       beam * input.beam_stride;
    const uint64_t tail =
        static_cast<uint64_t>(prev[row.tail * input.row_stride + t]);
    if (row.head == SIZE_MAX) {
        return tail;
    }
    const SizeType slot =
        t < row.delay ? boxcar_history_slot(input, row, t, ring0) : t;
    const uint64_t head = static_cast<uint64_t>(
        t >= row.delay ? prev[row.head * input.row_stride + t - row.delay]
                       : hist_base[beam * input.history_stride + row.hist -
                                   input.history_base + slot]);
    return tail + head;
}

DMT_HD inline uint64_t
boxcar_integer_sample(const FDMTSearchInput& input,
                      const BoxcarRow& row,
                      SizeType beam,
                      SizeType t,
                      bool root      = false,
                      SizeType ring0 = SIZE_MAX) noexcept {
    if (input.integer32_state != nullptr) {
        return boxcar_integer_sample_typed<uint32_t>(
            input.integer32_state, input.integer32_prev_state,
            input.integer32_history, input, row, beam, t, root, ring0);
    }
    return boxcar_integer_sample_typed<uint64_t>(
        input.integer_state, input.integer_prev_state, input.integer_history,
        input, row, beam, t, root, ring0);
}

DMT_HD inline bool boxcar_integer_better(uint64_t sum,
                                         uint64_t sample,
                                         uint32_t filter,
                                         const BoxcarPeak& peak) noexcept {
    return !peak.open || sum > peak.raw_sum ||
           (sum == peak.raw_sum &&
            (sample < peak.peak ||
             (sample == peak.peak && filter < peak.filter)));
}

template <typename Emit>
DMT_HD void boxcar_integer_peak_step(BoxcarPeak& peak,
                                     bool above,
                                     uint64_t sum,
                                     uint64_t sample,
                                     uint32_t filter,
                                     Emit emit) {
    if (!above) {
        if (peak.open) {
            emit(peak);
            peak.open = false;
        }
        return;
    }
    const bool better = boxcar_integer_better(sum, sample, filter, peak);
    if (!peak.open) {
        peak.begin = sample;
    }
    peak.end = sample;
    if (better) {
        peak.raw_sum = sum;
        peak.peak    = sample;
        peak.filter  = filter;
    }
    peak.open = true;
}

/// Packed input under per-call masks: S/N first (its sum's own table), then
/// today's raw-sum order. Within one table the S/N is monotone in the sum,
/// so the order equals boxcar_integer_better()'s.
DMT_HD inline bool boxcar_integer_snr_better(float snr,
                                             uint64_t sum,
                                             uint64_t sample,
                                             uint32_t filter,
                                             const BoxcarPeak& peak) noexcept {
    return !peak.open || snr > peak.snr ||
           (snr == peak.snr &&
            (sum > peak.raw_sum ||
             (sum == peak.raw_sum &&
              (sample < peak.peak ||
               (sample == peak.peak && filter < peak.filter)))));
}

template <typename Emit>
DMT_HD void boxcar_integer_snr_peak_step(BoxcarPeak& peak,
                                         bool above,
                                         float snr,
                                         uint64_t sum,
                                         uint64_t sample,
                                         uint32_t filter,
                                         Emit emit) {
    if (!above) {
        if (peak.open) {
            emit(peak);
            peak.open = false;
        }
        return;
    }
    const bool better =
        boxcar_integer_snr_better(snr, sum, sample, filter, peak);
    if (!peak.open) {
        peak.begin = sample;
    }
    peak.end = sample;
    if (better) {
        peak.snr     = snr;
        peak.raw_sum = sum;
        peak.peak    = sample;
        peak.filter  = filter;
    }
    peak.open = true;
}

/// One kept candidate of a row's kKeepBrightest store on a GPU (the row
/// supplies the rest of its RawCandidate).
struct BoxcarKeepEntry {
    uint64_t peak   = 0;
    uint64_t begin  = 0;
    uint64_t end    = 0;
    float snr       = 0.0F;
    uint32_t filter = 0;
};

/// OverflowPolicy::kKeepBrightest order: higher S/N first; ties to the
/// earlier peak, then beam, subband, DM row, width, begin sample and end
/// sample. A strict total order on distinct candidates of one call.
DMT_HD inline bool brighter(const RawCandidate& a, const RawCandidate& b) {
    if (a.snr != b.snr) {
        return a.snr > b.snr;
    }
    if (a.peak_sample != b.peak_sample) {
        return a.peak_sample < b.peak_sample;
    }
    if (a.beam_idx != b.beam_idx) {
        return a.beam_idx < b.beam_idx;
    }
    if (a.subband_level != b.subband_level) {
        return a.subband_level < b.subband_level;
    }
    if (a.subband_idx != b.subband_idx) {
        return a.subband_idx < b.subband_idx;
    }
    if (a.staggered != b.staggered) {
        return a.staggered < b.staggered;
    }
    if (a.dm_idx != b.dm_idx) {
        return a.dm_idx < b.dm_idx;
    }
    if (a.filter_idx != b.filter_idx) {
        return a.filter_idx < b.filter_idx;
    }
    if (a.begin_sample != b.begin_sample) {
        return a.begin_sample < b.begin_sample;
    }
    if (a.end_sample != b.end_sample) {
        return a.end_sample < b.end_sample;
    }
    return a.width < b.width;
}

/// The transform feeding a boxcar engine: FDMT (merged root rows) or
/// FDMT-FFT (materialized root rows per overlap-save segment).
class BoxcarDriver {
public:
    BoxcarDriver()                               = default;
    virtual ~BoxcarDriver()                      = default;
    BoxcarDriver(const BoxcarDriver&)            = delete;
    BoxcarDriver& operator=(const BoxcarDriver&) = delete;
    BoxcarDriver(BoxcarDriver&&)                 = delete;
    BoxcarDriver& operator=(BoxcarDriver&&)      = delete;

    virtual void search(std::span<const float>, FDMTSearchSink&)          = 0;
    virtual void search(DeviceSpan<const float>, FDMTSearchSink&, Stream) = 0;
    virtual void search(std::span<const uint8_t> /*unused*/,
                        SizeType /*unused*/,
                        FDMTSearchSink& /*unused*/) {
        throw std::invalid_argument("boxcar search: packed input unsupported");
    }
    virtual void search(DeviceSpan<const uint8_t> /*unused*/,
                        SizeType /*unused*/,
                        FDMTSearchSink& /*unused*/,
                        Stream /*unused*/) {
        throw std::invalid_argument("boxcar search: packed input unsupported");
    }
    virtual void reset(Stream) = 0;
    /// FDMTEngine::allocate(): the transform's deferred buffers.
    virtual void allocate() = 0;
    /// GPU: the host-memory search() overloads stage the input inside the
    /// transform's own (budgeted) memory, chunk by chunk. Otherwise the
    /// boxcar engine copies the block to the device first.
    [[nodiscard]] virtual bool stages_host_input() const noexcept {
        return false;
    }
    /// FDMTEngine::set_search_keep(): kill-mask rows read as zero by the
    /// transform itself, without a staging copy. False where unsupported.
    virtual bool set_search_keep(const uint8_t* /*keep*/, float /*fill*/ = 0) {
        return false;
    }
    /// FDMTEngine::execute_search_zero(): host search of one all-zero block.
    [[nodiscard]] virtual bool search_zero_supported() const noexcept {
        return false;
    }
    virtual void search_zero(FDMTSearchSink& /*unused*/) {
        throw std::logic_error("boxcar search: no zero-block search here");
    }
    [[nodiscard]] virtual FDMTMemoryUsage memory() const noexcept = 0;
    [[nodiscard]] virtual SizeType chunk_nsamps() const noexcept  = 0;
    [[nodiscard]] virtual SizeType fuse_levels() const noexcept   = 0;
    [[nodiscard]] virtual std::string search_path() const { return {}; }
    /// Input samples a searched sample waits for (FDMTEngine::search_latency).
    [[nodiscard]] virtual uint64_t search_latency() const noexcept { return 0; }
};

[[nodiscard]] std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTEngine> fdmt);
[[nodiscard]] std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTFFTEngine> fft);

/**
 * @brief Noise model of one block under FDMTBoxcar's per-call kill mask.
 *
 * Endpoints of the block (stream samples from first_sample to the next
 * block's) use the `steady` tables, except, when `transition` is set, the
 * first S_r + W - 1 of each row r and width W (S_r: the row's support with
 * every channel active), whose input windows also reach earlier blocks:
 * they use `transition_tables`, the variance of the history mask (the
 * block's channels and every channel live within the largest reach before
 * it; for blocks at least that long, the OR of this and the previous
 * block's masks). A null table pointer means the constructor tables (the
 * search planes). The block's input rows of the `kill_runs` channels are
 * zeroed.
 *
 * `catch_all`: set when blocks are shorter than the engine's ring needs
 * (FDMTBoxcarEngine::noise_catch_all_begin()). The engine keeps the latest
 * one and gives it, as steady and transition tables, to the oldest entry
 * of a full ring, which then also stands for the evicted blocks still
 * searched. It covers every channel live from that oldest endpoint's reach
 * up to this block's end, so it stays valid for drain() padding.
 */
struct BlockNoise {
    uint64_t first_sample = 0;
    std::shared_ptr<const NoiseTables> steady;
    bool transition = false;
    std::shared_ptr<const NoiseTables> transition_tables;
    std::vector<std::pair<SizeType, SizeType>> kill_runs;
    bool has_catch_all = false; ///< catch_all set (null: constructor)
    std::shared_ptr<const NoiseTables> catch_all;
};

class FDMTBoxcarEngine : public FDMTSearchSink {
public:
    virtual CandidateResultView execute(std::span<const float>)   = 0;
    virtual CandidateDeviceResultView execute(DeviceSpan<const float>,
                                              Stream)             = 0;
    virtual CandidateResultView execute(std::span<const uint8_t>) = 0;
    virtual CandidateDeviceResultView execute(DeviceSpan<const uint8_t>,
                                              Stream)             = 0;
    virtual CandidateResultView flush()                           = 0;
    virtual CandidateDeviceResultView flush_device(Stream)        = 0;
    virtual CandidateResultView copy_result(Stream)               = 0;
    virtual void reset_history(Stream)                            = 0;
    virtual FDMTBoxcarMemoryUsage memory_usage() const noexcept   = 0;
    /// Allocates what a construction with
    /// FDMTEngineConfig::defer_allocation planned (the transform's too);
    /// memory_usage() is unchanged. A no-op once allocated.
    virtual void allocate() = 0;
    /// drain() padding: one block of zero input, searched without reading,
    /// staging or allocating an input block (a host call; candidates as
    /// execute() on zeros), where zero_blocks(). Otherwise drain() pushes a
    /// zero block through the host execute().
    [[nodiscard]] virtual bool zero_blocks() const noexcept { return false; }
    virtual CandidateResultView execute_zero() {
        throw std::logic_error("FDMTBoxcar: no zero-block execute here");
    }
    virtual SizeType chunk_nsamps() const noexcept = 0;
    virtual SizeType fuse_levels() const noexcept  = 0;
    virtual std::string search_path() const        = 0;
    /// drain(): endpoints at or after `stop` are not searched (UINT64_MAX:
    /// none); reset_history() clears it.
    virtual void set_search_stop(uint64_t stop) noexcept = 0;
    /// Stream samples consumed so far (the next block's first sample).
    [[nodiscard]] virtual uint64_t stream_position() const noexcept = 0;
    /// Input samples a sample waits for before every row has searched it.
    [[nodiscard]] virtual uint64_t search_latency() const noexcept = 0;
    /// Smallest begin sample of the excursions still open after the last
    /// call (UINT64_MAX: none). GPU: valid once the last call's stream has
    /// completed.
    [[nodiscard]] virtual uint64_t open_begin_min() const noexcept = 0;

    // Per-call kill masks (FDMTBoxcar, float or packed input). Until
    // enabled, every
    // block uses the constructor tables and kill runs.
    /// Bytes that enable_block_noise() would add to memory_usage().
    [[nodiscard]] virtual SizeType block_noise_bytes() const noexcept {
        return 0;
    }
    /// Starts per-block noise: `support` holds each row's support with
    /// every channel active. The ring starts with the constructor entry.
    virtual void enable_block_noise(
        std::shared_ptr<const std::vector<uint64_t>> /*unused*/) {
        throw std::invalid_argument(
            "FDMTBoxcar: per-call kill masks are not supported here");
    }
    /// The next execute()'s block; the entry is committed only once that
    /// call passes its checks. An execute() without one (drain() padding)
    /// repeats the last block's steady tables, with no transition and the
    /// constructor kill runs.
    virtual void push_block_noise(BlockNoise /*unused*/) {
        throw std::invalid_argument(
            "FDMTBoxcar: per-call kill masks are not supported here");
    }
    /// Drops an entry pushed for an execute() that threw.
    virtual void drop_block_noise() noexcept {}
    /// With blocks too short for the engine's ring: the first stream sample
    /// the catch-all of the next block must cover (the oldest endpoint
    /// still searched once that block is in); UINT64_MAX when the ring
    /// holds every block it needs.
    [[nodiscard]] virtual uint64_t noise_catch_all_begin() const noexcept {
        return UINT64_MAX;
    }
};

/// BoxcarScreenMode::kAuto resolved for a backend: kKadane on the CPU (its
/// gate beats the prefix screen there), kPrefix on a GPU (where the gate's
/// Stage 2 latency outweighs the saved screen). Other modes pass through.
[[nodiscard]] constexpr BoxcarScreenMode
resolve_screen_mode(BoxcarScreenMode mode, bool gpu) noexcept {
    if (mode != BoxcarScreenMode::kAuto) {
        return mode;
    }
    return gpu ? BoxcarScreenMode::kPrefix : BoxcarScreenMode::kKadane;
}

/// `config` with its screen mode resolved (resolve_screen_mode).
[[nodiscard]] constexpr BoxcarConfig resolve_screen_mode(BoxcarConfig config,
                                                         bool gpu) noexcept {
    config.screen_mode = resolve_screen_mode(config.screen_mode, gpu);
    return config;
}

/// The boxcar engines own `driver`, which calls them back per chunk.
/// Input rows of the `kill_runs` channels (boxcar_kill_runs()) are zeroed
/// in a staging copy of each block before the driver reads it (FDMTBoxcar's
/// kill mask; FDMT-FFT drivers mask inside the transform and pass none).
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_cpu(const plans::FDMTPlan& /*plan*/,
                     const BoxcarConfig& /*config*/,
                     const std::vector<plans::BoxcarPlanePlan>& /*planes*/,
                     const FDMTEngineConfig& /*cfg*/,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs = {});
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan&,
                     const BoxcarConfig&,
                     const std::vector<plans::BoxcarPlanePlan>&,
                     const FDMTEngineConfig&,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs = {});

} // namespace dmt::algorithms::detail
