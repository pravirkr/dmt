#pragma once

// Noise tables of FDMTBoxcar for a kill mask given per call (FDMTBoxcar's
// per-gulp kill mask): the thresholds and inverse sigmas that
// make_boxcar_plans() computes for a constructor mask, recomputed for any
// mask without tracing every row.

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
/// raw_thresh_lut / inv_sigma_lut. `live` counts the bytes of the tables
/// alive (the memory usage of the per-call mask).
struct NoiseTables {
    std::vector<float> thresh;
    std::vector<float> inv_sigma;
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
        return (thresh.size() + inv_sigma.size()) * sizeof(float);
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
 */
class BoxcarNoiseModel {
public:
    BoxcarNoiseModel(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     bool smear,
                     int nthreads);

    /// Tables for `mask` (nchans entries, 0 = killed). Throws
    /// std::invalid_argument when a sigma or threshold leaves float range.
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
        return rows() * m_widths.size() * 2 * sizeof(float);
    }
    /// Bytes the model keeps: the row supports and node table (plan) and
    /// the histogram layout and counts (workspace).
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
    [[nodiscard]] std::vector<Bin>& bins(SizeType level) noexcept;

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    bool m_smear;
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
    std::shared_ptr<std::atomic<SizeType>> m_live;
    uint64_t m_next_id = 1;
};

} // namespace dmt::algorithms::detail
