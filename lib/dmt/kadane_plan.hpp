#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::kadane {

inline constexpr SizeType kMaxGateBuckets       = 3;
inline constexpr SizeType kKadaneRowAlign       = 16;
inline constexpr uint32_t kMaxKadaneGateWidth   = 1U << 20;
inline constexpr SizeType kMaxKadaneGateFilters = 4096;
inline constexpr int32_t kKadaneInt32Cap =
    std::numeric_limits<int32_t>::max() / 2;

/// Edge taper width of the symmetric {1, 2, 1} trapezoidal filter of total
/// support `width`. Widths 1 and 2 are pure boxcars (edge width 0); widths >= 3
/// use edge width floor((width + 1) / 4) and core width `width - 2 * w_edge`.
DMT_HD constexpr uint32_t trapezoid_edge_width(uint32_t width) noexcept {
    return width >= 3U ? (width + 1U) / 4U : 0U;
}

/// Contiguous filter-index bucket [f_lo, f_hi) in `widths`.
struct KadaneGateBucket {
    uint32_t f_lo         = 0; ///< First filter index (inclusive).
    uint32_t f_hi         = 0; ///< One past last filter index (exclusive).
    uint32_t w_lo         = 0; ///< Smallest width in bucket (widths[f_lo]).
    uint32_t w_hi         = 0; ///< Largest width in bucket (widths[f_hi - 1]).
    uint32_t max_edge_lag = 0; ///< Max trapezoid edge lag in bucket.
};

/**
 * @brief Transform-agnostic lossless multi-bucket Kadane Stage-1 gate plan.
 *
 * Partitions up to `kMaxKadaneGateFilters` (4096) strictly ascending `widths`
 * in `1..kMaxKadaneGateWidth` (1..2^20) into K (1..kMaxGateBuckets) contiguous
 * filter buckets shared across all rows of a plane, and computes per-row biases
 * and lossless trigger cuts in SIMD-friendly SoA layout `[k * row_stride + r]`,
 * plus optional exact `int32_t` gate tables (`ibias`, `icut`) with zero
 * rounding margin. Has zero dependency on `FDMTPlan`.
 */
struct KadaneGatePlan {
    SizeType num_buckets = 0;
    SizeType n_rows      = 0;
    SizeType row_stride  = 0; ///< Padded to a multiple of kKadaneRowAlign.
    bool use_trapezoid   = false;
    bool has_int32       = false;
    uint8_t int_shift    = 0; ///< Left-shift applied to integer samples x_t.

    std::vector<uint32_t> widths;
    std::array<KadaneGateBucket, kMaxGateBuckets> buckets{};
    std::array<uint32_t, kMaxGateBuckets> max_edge_lag{};

    /// Float32 gate slopes and cuts in SoA layout: [k * row_stride + r].
    std::vector<float> bias;
    std::vector<float> cut;

    /// Optional exact int32_t slopes and cuts: [k * row_stride + r], for
    /// clamped recurrence `c_t = max(0, min(c_{t-1}, kKadaneInt32Cap) +
    /// (static_cast<int32_t>(x_t << int_shift) - ibias))` and trigger
    /// `c_t >= icut` (zero rounding margin; `(x_t << int_shift) <=
    /// kKadaneInt32Cap`, `w_max * ibias <= INT32_MAX / 4`, and
    /// `1 <= icut <= INT32_MAX / 4` prevent signed i32 overflow on all ISAs
    /// via 1-cycle `min + add` and keep clamped states `>= icut` across
    /// `w_max` steps).
    std::vector<int32_t> ibias;
    std::vector<int32_t> icut;

    /// 1 when row r is dead / fully masked (never triggers), 0 when active.
    std::vector<uint8_t> disabled_row;

    KadaneGatePlan() = default;
    KadaneGatePlan(std::span<const uint32_t> widths,
                   bool use_trapezoid,
                   SizeType n_rows,
                   std::span<const float> raw_thresh_lut,
                   std::span<const double> integer_mean_lut     = {},
                   std::span<const uint64_t> integer_thresh_lut = {},
                   bool enable_int32_gate                       = false,
                   uint64_t max_int_sample                      = 0);

    [[nodiscard]] float row_bias(SizeType k, SizeType r) const noexcept {
        return bias[k * row_stride + r];
    }
    [[nodiscard]] float row_cut(SizeType k, SizeType r) const noexcept {
        return cut[k * row_stride + r];
    }
    [[nodiscard]] int32_t row_ibias(SizeType k, SizeType r) const noexcept {
        return ibias[k * row_stride + r];
    }
    [[nodiscard]] int32_t row_icut(SizeType k, SizeType r) const noexcept {
        return icut[k * row_stride + r];
    }

    /**
     * @brief Recompute lossless float32 cuts for `row` at its fixed `bias`
     * against an alternate/minimum threshold vector `min_thr`.
     *
     * Keeping `bias` fixed across per-block kill-mask transitions
     * (`BlockNoise`) preserves exact continuity of the running Kadane state
     * `c_{t,k}` without resetting at block boundaries.
     */
    void recompute_row_cuts(SizeType row,
                            std::span<const float> min_thr,
                            std::span<float> out_cuts) const;

    /**
     * @brief Integer counterpart of recompute_row_cuts() for the exact
     * int32 gate (`has_int32`): bucket cuts of `row` at its fixed `ibias`
     * and `int_shift` against alternate integer cutoffs `int_thr` (same
     * units as the plan's integer_thresh_lut; UINT64_MAX = never). Lossless
     * for any cutoffs; clamped to [0, INT32_MAX / 4] (a lower cut only
     * triggers more often), INT32_MAX when no width of a bucket can fire.
     * Valid because the row's mean (hence the bias' negative drift) is the
     * plan's.
     */
    void recompute_row_icuts(SizeType row,
                             std::span<const uint64_t> int_thr,
                             std::span<int32_t> out_cuts) const;

    [[nodiscard]] SizeType memory_bytes() const noexcept {
        return (widths.size() * sizeof(uint32_t)) +
               ((bias.size() + cut.size()) * sizeof(float)) +
               ((ibias.size() + icut.size()) * sizeof(int32_t)) +
               (disabled_row.size() * sizeof(uint8_t));
    }
};

[[nodiscard]] KadaneGatePlan
make_kadane_gate_plan(std::span<const uint32_t> widths,
                      bool use_trapezoid,
                      SizeType n_rows,
                      std::span<const float> raw_thresh_lut,
                      std::span<const double> integer_mean_lut     = {},
                      std::span<const uint64_t> integer_thresh_lut = {},
                      bool enable_int32_gate                       = false,
                      uint64_t max_int_sample                      = 0);

/// Recompute lossless float32 bucket cuts for explicit `row_biases` against
/// `min_thr` (size == `plan.widths.size()`).
void recompute_cuts_for_biases(const KadaneGatePlan& plan,
                               std::span<const float> row_biases,
                               std::span<const float> min_thr,
                               std::span<float> out_cuts);

} // namespace dmt::kadane
