#pragma once

// Noise tables of FDMTBoxcar for a kill mask given per call (FDMTBoxcar's
// per-gulp kill mask): the thresholds and inverse sigmas that
// make_boxcar_plans() computes for a constructor mask, recomputed for any
// mask without tracing every row.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

namespace dmt::algorithms::detail {

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
    struct Bin {
        uint32_t len;
        uint32_t count;
    };
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

} // namespace dmt::algorithms::detail
