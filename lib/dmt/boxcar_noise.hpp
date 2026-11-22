#pragma once

// Noise tables of FDMTBoxcar for a kill mask given per call (FDMTBoxcar's
// per-gulp kill mask): the thresholds and inverse sigmas that
// make_boxcar_plans() computes for a constructor mask, recomputed for any
// mask without tracing every row.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/dm_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::algorithms::detail {

/// Ascending (leaf box length, active channel count) entry of a row's
/// noise histogram.
struct BoxcarLengthBin {
    uint32_t len;
    uint32_t count;
};

/// Merges two ascending BoxcarLengthBin histograms into `dst` (counting only
/// when `dst == nullptr`). Returns the merged entry count.
inline SizeType merge_length_bins(std::span<const BoxcarLengthBin> a,
                                  std::span<const BoxcarLengthBin> b,
                                  BoxcarLengthBin* dst) noexcept {
    const BoxcarLengthBin* x  = a.data();
    const BoxcarLengthBin* xe = x + a.size();
    const BoxcarLengthBin* y  = b.data();
    const BoxcarLengthBin* ye = y + b.size();
    SizeType k                = 0;
    while (x != xe || y != ye) {
        BoxcarLengthBin v{};
        if (y == ye || (x != xe && x->len < y->len)) {
            v = *x++;
        } else if (x == xe || y->len < x->len) {
            v = *y++;
        } else {
            v = {x->len, x->count + y->count};
            ++x;
            ++y;
        }
        if (dst != nullptr) {
            dst[k] = v;
        }
        ++k;
    }
    return k;
}

/// Σ count · len · weight over a row's ascending length histogram (skipping
/// zero counts): the row's mean per unit code and filter weight.
inline double row_mean_weight(std::span<const BoxcarLengthBin> hist,
                              bool smear,
                              SizeType frac_bits,
                              double unit) noexcept {
    double sum = 0.0;
    for (const auto& e : hist) {
        if (e.count == 0) {
            continue;
        }
        const SizeType len = e.len;
        const double weight =
            smear ? box_weight(len, frac_bits) * unit : 1.0;
        sum += static_cast<double>(e.count) * static_cast<double>(len) * weight;
    }
    return sum;
}

struct RowWidthNoise {
    float inv_sigma         = 0.0F;
    float thresh            = std::numeric_limits<float>::max();
    double integer_mean     = 0.0;
    uint64_t integer_thresh = UINT64_MAX;
    bool ok_float           = true;
    bool ok_int             = true;
};

/// Noise parameters of one (row, width) pair from its ascending leaf-length
/// histogram. Shared by make_boxcar_plans() and BoxcarNoiseModel::compute()
/// so constructor tables and per-call mask tables stay bit-identical.
/// When `shift_to_ref` is true (packed per-call masks), the integer cutoff is
/// shifted by the exact difference between `mw_ref` and `mw`.
inline RowWidthNoise row_width_noise(std::span<const BoxcarLengthBin> hist,
                                     SizeType width,
                                     const BoxcarConfig& config,
                                     bool smear,
                                     SizeType frac_bits,
                                     double unit,
                                     bool packed,
                                     double mw,
                                     double mw_ref     = 0.0,
                                     bool shift_to_ref = false) noexcept {
    const SizeType w_edge =
        config.use_trapezoid
            ? trapezoid_edge_width(static_cast<uint32_t>(width))
            : 0;
    double variance = 0.0;
    for (const auto& e : hist) {
        if (e.count == 0) {
            continue;
        }
        const SizeType len = e.len;
        const double weight =
            smear ? box_weight(len, frac_bits) * unit : 1.0;
        const auto n = static_cast<double>(e.count);
        variance += n * fdmt_trapezoid_variance(len, width, w_edge, weight);
    }
    RowWidthNoise out;
    if (variance == 0.0) {
        return out;
    }
    const double sigma    = config.noise_sigma * std::sqrt(variance);
    const float inv_sigma = static_cast<float>(1.0 / sigma);
    const float threshold = static_cast<float>(config.detect_thresh * sigma);
    out.ok_float          = utils::is_finite_bits(inv_sigma) &&
                   utils::is_finite_bits(threshold) && inv_sigma > 0 &&
                   threshold > 0;
    out.inv_sigma = inv_sigma;
    out.thresh    = threshold;
    if (packed) {
        const double scale = smear ? 1.0 / unit : 1.0;
        const double filter_weight =
            w_edge == 0 ? static_cast<double>(width)
                        : static_cast<double>(2 * (width - w_edge));
        const double mean = config.input_mean * mw * filter_weight * scale;
        const double cutoff =
            std::ceil(mean + (config.detect_thresh * sigma * scale));
        if (!shift_to_ref) {
            out.ok_int = cutoff < std::ldexp(1.0, 64);
            if (out.ok_int) {
                out.integer_mean   = mean;
                out.integer_thresh = static_cast<uint64_t>(cutoff);
            }
        } else {
            const double mean_ref =
                config.input_mean * mw_ref * filter_weight * scale;
            const double shift = mean_ref - mean;
            // A whole number (integer mean code, dyadic weights).
            const bool fits = cutoff < std::ldexp(1.0, 64) && shift >= 0 &&
                              shift < std::ldexp(1.0, 63) &&
                              shift == std::floor(shift);
            out.ok_int = fits;
            if (fits) {
                const auto c0      = static_cast<uint64_t>(cutoff);
                const auto sh      = static_cast<uint64_t>(shift);
                out.ok_int         = c0 <= UINT64_MAX - sh;
                out.integer_thresh = c0 + sh;
            }
        }
    }
    return out;
}

/// Raw thresholds and inverse sigmas of every searched row and width (rows
/// in plane order, then DM; widths fastest), as BoxcarPlanePlan's
/// raw_thresh_lut / inv_sigma_lut; packed input adds the integer cutoffs
/// (integer_thresh_lut against the model's reference mean). `live` counts
/// the bytes of the tables alive (the memory usage of the per-call mask).
struct NoiseTables {
    std::vector<float> thresh;
    std::vector<float> inv_sigma;
    std::vector<uint64_t> integer_thresh; ///< Packed input only.
    std::shared_ptr<std::atomic<SizeType>> live;
    uint64_t id = 0; ///< unique per model (device copies are keyed by it)

    NoiseTables()                              = default;
    NoiseTables(const NoiseTables&)            = delete;
    NoiseTables& operator=(const NoiseTables&) = delete;
    NoiseTables(NoiseTables&&)                 = delete;
    NoiseTables& operator=(NoiseTables&&)      = delete;
    ~NoiseTables() {
        if (live) {
            live->fetch_sub(bytes(), std::memory_order_relaxed);
        }
    }
    [[nodiscard]] SizeType bytes() const noexcept {
        return ((thresh.size() + inv_sigma.size()) * sizeof(float)) +
               (integer_thresh.size() * sizeof(uint64_t));
    }
};

/**
 * @brief Row noise of a plan for any channel mask, bit-identical to
 * make_boxcar_plans().
 *
 * A row's variance depends only on how many active channels enter it with
 * each leaf box length. Box lengths alternate between neighbouring
 * channels (runs per row are about half the channels), so instead of
 * storing per-row channel lists the model keeps, per FDMT tree
 * coordinate, a histogram of (length, active count), merged bottom-up from
 * level 0 (one leaf, count mask[c]) through the plan's tail/head
 * coordinates; a row reads the histogram of its coordinate. The histograms'
 * lengths and sizes do not depend on the mask, so their layout is built
 * once (with every channel active, which also gives each row's support)
 * and each mask only refills the counts. The variance then accumulates in
 * ascending length over the nonzero counts, exactly as make_boxcar_plans().
 *
 * Packed input (config.input_nbits != 32): channels a per-call mask kills
 * but the `reference` mask (the constructor's) keeps are read as the code
 * input_mean, so every row's expected sum stays the reference's
 * (integer_mean_lut of make_boxcar_plans() with `reference`), whatever the
 * mask. A mask's integer cutoff is make_boxcar_plans()'s cutoff for that
 * mask plus the exact integer difference of the two means: bit-identical to
 * the reference plan's for the reference mask itself, and for any other
 * mask the candidates of a constructor mask equal to it, shifted by that
 * offset. Needs an integer input_mean (the substitution is exact), and
 * masks within the reference (killed channels stay killed).
 */
class BoxcarNoiseModel {
public:
    /// `reference`: packed input's mean mask (nchans entries, 0 = killed;
    /// empty: every channel); ignored for float input.
    BoxcarNoiseModel(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     bool smear,
                     int nthreads,
                     std::span<const uint8_t> reference = {});

    /// Tables for `mask` (nchans entries, 0 = killed). Throws
    /// std::invalid_argument when a sigma or threshold leaves float range
    /// (packed: or an integer cutoff leaves uint64, or `mask` keeps a
    /// channel the reference kills).
    [[nodiscard]] std::shared_ptr<const NoiseTables>
    compute(std::span<const uint8_t> mask);

    /// Each row's FDMT support with every channel active (samples before
    /// the endpoint its leaves reach, before the boxcar's width - 1).
    [[nodiscard]] const std::shared_ptr<const std::vector<uint64_t>>&
    support() const noexcept {
        return m_support;
    }
    [[nodiscard]] uint64_t max_support() const noexcept {
        return m_max_support;
    }
    [[nodiscard]] SizeType rows() const noexcept { return m_row_node.size(); }
    /// Bytes of one table set.
    [[nodiscard]] SizeType table_bytes() const noexcept {
        return rows() * m_widths.size() *
               ((2 * sizeof(float)) + (m_packed ? sizeof(uint64_t) : 0));
    }
    /// Bytes the model keeps: the row supports and node table, and the
    /// LiveHistory its owner keeps (plan); the histogram layout and counts
    /// (workspace).
    [[nodiscard]] SizeType plan_bytes() const noexcept;
    [[nodiscard]] SizeType workspace_bytes() const noexcept;
    /// Bytes of the table sets returned by compute() still alive.
    [[nodiscard]] SizeType live_table_bytes() const noexcept {
        return m_live->load(std::memory_order_relaxed);
    }

private:
    using Bin = BoxcarLengthBin;
    /// Fills the bins of every level from level-0 counts `mask` (empty:
    /// all active); with `layout`, also builds the offsets first.
    void fill(std::span<const uint8_t> mask, bool layout);
    /// Σ count · len · weight over row `r`'s bins: the row's mean per unit
    /// code and filter weight, in make_boxcar_plans()'s order.
    [[nodiscard]] double mean_weight(SizeType r) const noexcept;
    [[nodiscard]] std::vector<Bin>& bins(SizeType level) noexcept;

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    bool m_smear;
    bool m_packed;
    int m_threads;
    SizeType m_root;
    SizeType m_first_searched;
    std::vector<uint32_t> m_widths;
    /// Per row: (level, coordinate) of its node, in plane then DM order.
    std::vector<std::pair<uint32_t, uint32_t>> m_row_node;
    std::shared_ptr<const std::vector<uint64_t>> m_support;
    uint64_t m_max_support = 0;
    /// Per level, bin offsets of each coordinate (coordinates + 1).
    std::vector<std::vector<uint64_t>> m_offsets;
    /// Bins of the searched levels (kept for the rows) and two alternating
    /// buffers for the levels below them.
    std::vector<std::vector<Bin>> m_searched;
    std::vector<Bin> m_below[2];
    /// Packed: each row's mean_weight() under the reference mask.
    std::vector<double> m_reference_mean;
    std::vector<uint8_t> m_reference;
    std::shared_ptr<std::atomic<SizeType>> m_live;
    uint64_t m_next_id = 1;
};

/**
 * @brief When each channel was last live, for the history masks of
 * FDMTBoxcar's per-call kill masks: one past the last stream sample a
 * channel was live in (0: never), nchans entries whatever the stream.
 */
class LiveHistory {
public:
    /// The history at stream position `position`: the channels `mask`
    /// keeps were live in every earlier sample (none at position 0).
    void reset(std::span<const uint8_t> mask, uint64_t position) {
        m_until.assign(mask.size(), 0);
        if (position > 0) {
            for (SizeType c = 0; c < mask.size(); ++c) {
                m_until[c] = mask[c] != 0 ? position : 0;
            }
        }
    }
    /// `out` = `mask` OR every channel live at or after sample `from`.
    void since(std::span<const uint8_t> mask,
               uint64_t from,
               std::vector<uint8_t>& out) const {
        out.resize(mask.size());
        for (SizeType c = 0; c < mask.size(); ++c) {
            out[c] = (mask[c] != 0 || m_until[c] > from) ? 1 : 0;
        }
    }
    /// A block ending before sample `end` kept the channels of `mask`.
    void update(std::span<const uint8_t> mask, uint64_t end) {
        for (SizeType c = 0; c < mask.size(); ++c) {
            if (mask[c] != 0) {
                m_until[c] = end;
            }
        }
    }
    [[nodiscard]] SizeType bytes() const noexcept {
        return m_until.capacity() * sizeof(uint64_t);
    }

private:
    std::vector<uint64_t> m_until;
};

/// The masks of a block's noise entry (FDMTBoxcar's per-call kill masks).
struct BlockMasks {
    bool transition = false;      ///< windows before the block need history
    std::vector<uint8_t> history; ///< their mask (when `transition`)
    bool catch_all = false;       ///< the engine asked for a catch-all
    std::vector<uint8_t> catch_mask;
};

/**
 * The masks of the block at stream sample `first` with effective mask
 * `mask` (`prev`: the previous block's), from the live history `live`:
 * - history: `mask` OR every channel live within `reach` samples before
 *   `first` (for blocks of at least `reach` samples, OR the previous
 *   block's mask); used when the mask changed or the history differs
 *   (never for the first block of a stream);
 * - catch-all (`catch_begin` != UINT64_MAX, from
 *   FDMTBoxcarEngine::noise_catch_all_begin()): `mask` OR every channel
 *   live from `catch_begin - reach` on.
 */
inline void block_masks(const LiveHistory& live,
                        std::span<const uint8_t> mask,
                        std::span<const uint8_t> prev,
                        uint64_t first,
                        uint64_t reach,
                        uint64_t catch_begin,
                        BlockMasks& out) {
    out.transition = false;
    out.catch_all  = false;
    if (first > 0) {
        live.since(mask, first > reach ? first - reach : 0, out.history);
        out.transition =
            !std::equal(mask.begin(), mask.end(), prev.begin(), prev.end()) ||
            !std::equal(mask.begin(), mask.end(), out.history.begin(),
                        out.history.end());
    }
    if (catch_begin != UINT64_MAX) {
        live.since(mask, catch_begin > reach ? catch_begin - reach : 0,
                   out.catch_mask);
        out.catch_all = true;
    }
}

/**
 * @brief Noise tables and kill runs for one call (FDMTBoxcar per-call masks).
 *
 * `steady` / `transition_tables` null means the constructor's tables. When
 * blocks are shorter than the engine's search latency, the facade also
 * supplies a catch-all table set covering every channel live since
 * `FDMTBoxcarEngine::noise_catch_all_begin()`. The engine keeps the latest
 * one and gives it, as steady and transition tables, to the oldest entry of
 * a full ring, which then also stands for the evicted blocks still searched.
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

/**
 * @brief Host-side ring buffer and rollback state for per-call kill masks,
 * shared by the CPU and GPU FDMTBoxcar engines.
 */
class BlockNoiseRing {
public:
    void enable(
        SizeType ring_needed,
        SizeType ring_size,
        const std::vector<std::pair<SizeType, SizeType>>& default_kill_runs) {
        m_ring_needed = ring_needed;
        m_ring_size   = ring_size;
        m_ring.reserve(m_ring_size + 1);
        m_enabled = true;
        reset(default_kill_runs);
    }

    [[nodiscard]] bool enabled() const noexcept { return m_enabled; }
    [[nodiscard]] SizeType ring_needed() const noexcept {
        return m_ring_needed;
    }
    [[nodiscard]] SizeType ring_size() const noexcept { return m_ring_size; }
    [[nodiscard]] std::span<const BlockNoise> entries() const noexcept {
        return m_ring;
    }
    [[nodiscard]] const std::vector<std::pair<SizeType, SizeType>>&
    active_kill_runs(
        const std::vector<std::pair<SizeType, SizeType>>& default_runs)
        const noexcept {
        return m_enabled ? m_ring.back().kill_runs : default_runs;
    }

    void push(BlockNoise entry) { m_pending = std::move(entry); }

    void reset(
        const std::vector<std::pair<SizeType, SizeType>>& default_kill_runs) {
        m_pending.reset();
        m_evicted.reset();
        m_prev_evicted.reset();
        m_commit_open   = false;
        m_has_catch_all = false;
        m_catch_all.reset();
        m_catch_undo = {};
        m_ring.clear();
        BlockNoise first; // the constructor tables from stream sample 0
        first.kill_runs = default_kill_runs;
        m_ring.push_back(std::move(first));
    }

    void commit(
        uint64_t position,
        const std::vector<std::pair<SizeType, SizeType>>& default_kill_runs) {
        if (!m_enabled) {
            return;
        }
        BlockNoise entry;
        if (m_pending) {
            entry = std::move(*m_pending);
            m_pending.reset();
        } else {
            entry.steady    = m_ring.back().steady;
            entry.kill_runs = default_kill_runs;
        }
        entry.first_sample = position;
        m_prev_evicted             = std::move(m_evicted);
        m_catch_undo               = {};
        m_catch_undo.has_catch_all = m_has_catch_all;
        m_catch_undo.catch_all     = m_catch_all;
        if (entry.has_catch_all) {
            m_catch_all         = std::move(entry.catch_all);
            m_has_catch_all     = true;
            entry.has_catch_all = false;
        }
        m_evicted.reset();
        if (m_ring.size() == m_ring_size) {
            m_evicted = std::move(m_ring.front());
            m_ring.erase(m_ring.begin());
        }
        m_ring.push_back(std::move(entry));
        // A short block evicted an entry still searched: the oldest kept one
        // stands for it with the catch-all.
        if (m_evicted && m_ring_needed > m_ring_size && m_has_catch_all) {
            auto& front                    = m_ring.front();
            m_catch_undo.front_set         = true;
            m_catch_undo.steady            = front.steady;
            m_catch_undo.transition_tables = front.transition_tables;
            m_catch_undo.transition        = front.transition;
            front.steady                   = m_catch_all;
            front.transition_tables        = m_catch_all;
            front.transition               = true;
        }
        m_commit_open = true;
    }

    void finish_commit() noexcept {
        m_commit_open = false;
        m_catch_undo  = {};
        m_prev_evicted.reset();
    }

    // m_ring is pre-reserved to m_ring_size + 1 in enable().
    // NOLINTNEXTLINE(bugprone-exception-escape)
    bool undo_commit() noexcept {
        if (!m_commit_open) {
            return false;
        }
        auto& u = m_catch_undo;
        if (u.front_set) {
            auto& front             = m_ring.front();
            front.steady            = std::move(u.steady);
            front.transition_tables = std::move(u.transition_tables);
            front.transition        = u.transition;
        }
        m_has_catch_all = u.has_catch_all;
        m_catch_all     = std::move(u.catch_all);
        u               = {};
        m_ring.pop_back();
        if (m_evicted) {
            m_ring.insert(m_ring.begin(), std::move(*m_evicted));
        }
        m_evicted     = std::move(m_prev_evicted);
        m_commit_open = false;
        return true;
    }

    bool drop() noexcept {
        m_pending.reset();
        return undo_commit();
    }

    [[nodiscard]] uint64_t catch_all_begin(uint64_t position,
                                           uint64_t block_nsamps) const
        noexcept {
        if (!m_enabled || m_ring_needed <= m_ring_size) {
            return UINT64_MAX;
        }
        // The first sample of the oldest block the next call still searches.
        const uint64_t back = (m_ring_needed - 1) * block_nsamps;
        return position > back ? position - back : 0;
    }

    [[nodiscard]] SizeType index_for(uint64_t sample) const noexcept {
        SizeType i = m_ring.size() - 1;
        while (i > 0 && m_ring[i].first_sample > sample) {
            --i;
        }
        return i;
    }

    [[nodiscard]] bool references_table(uint64_t id) const noexcept {
        const auto has = [id](const BlockNoise& e) {
            return (e.steady && e.steady->id == id) ||
                   (e.transition_tables && e.transition_tables->id == id);
        };
        if (m_evicted && has(*m_evicted)) {
            return true;
        }
        for (const auto& e : m_ring) {
            if (has(e)) {
                return true;
            }
        }
        // The tables the catch-all replaced: an undo restores them.
        const auto& u = m_catch_undo;
        return u.front_set &&
               ((u.steady && u.steady->id == id) ||
                (u.transition_tables && u.transition_tables->id == id));
    }

private:
    /// What the open commit replaced (restored by undo_commit()).
    struct CatchUndo {
        bool has_catch_all = false;
        std::shared_ptr<const NoiseTables> catch_all;
        bool front_set = false;
        std::shared_ptr<const NoiseTables> steady, transition_tables;
        bool transition = false;
    };

    bool m_enabled         = false;
    SizeType m_ring_size   = 1;
    SizeType m_ring_needed = 1;
    std::vector<BlockNoise> m_ring; ///< oldest first, at most m_ring_size
    std::optional<BlockNoise> m_pending;
    /// The entry the last commit evicted, kept until its call completes.
    std::optional<BlockNoise> m_evicted;
    std::optional<BlockNoise> m_prev_evicted;
    bool m_commit_open   = false;
    bool m_has_catch_all = false; ///< m_catch_all set (null: constructor)
    std::shared_ptr<const NoiseTables> m_catch_all;
    CatchUndo m_catch_undo;
};

} // namespace dmt::algorithms::detail
