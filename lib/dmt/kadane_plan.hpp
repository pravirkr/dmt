#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::kadane {

inline constexpr SizeType kMaxGateBuckets          = 3;
inline constexpr SizeType kMaxSurrogatePasses      = 3;
inline constexpr SizeType kKadaneRowAlign          = 16;
inline constexpr uint32_t kMaxKadaneGateWidth      = 1U << 20;
inline constexpr SizeType kMaxKadaneGateFilters    = 4096;
inline constexpr uint32_t kMaxKadaneSurrogateWidth = 4096U;
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

/// One width-strip pass of `KadaneSurrogatePlan`.
struct KadaneSurrogatePass {
    uint32_t w_lo            = 1; ///< Target strip minimum width (inclusive).
    uint32_t w_hi            = 1; ///< Target strip maximum width (inclusive).
    uint32_t decimate_factor = 1; ///< 1: full stream; 4: 4-sample block sum.
};

/**
 * @brief Transform-agnostic multi-bias / decimated Kadane surrogate plan for
 * `BoxcarScreenMode::kKadaneSurrogate`.
 *
 * Supports calibrated `widths` in `1..kMaxKadaneSurrogateWidth` (1..4096).
 * Plans up to `kMaxSurrogatePasses` (<= 3) passes (full-resolution passes plus
 * an optional fused f=4 decimated pass when `max_width >= 16` and
 * `!use_trapezoid`), strip-center tangent biases, excursion pre-filter trigger
 * thresholds, and O(1) arbitrary-width S/N evaluation.
 */
struct KadaneSurrogatePlan {
    SizeType num_passes = 0;
    SizeType n_rows     = 0;
    SizeType row_stride = 0;
    uint32_t max_width  = 0;
    bool use_trapezoid  = false;
    bool has_decimated  = false;
    bool is_integer     = false;
    double integer_unit = 1.0;

    std::vector<uint32_t> widths;
    std::array<KadaneSurrogatePass, kMaxSurrogatePasses> passes{};

    /// Per-pass step bias and excursion trigger threshold in SoA layout
    /// `[k * row_stride + r]`. When `decimate_factor == 4`, `bias` applies to
    /// unscaled 4-sample block sums `y_j = x_{4j} + x_{4j+1} + x_{4j+2} +
    /// x_{4j+3}`.
    std::vector<float> bias;
    std::vector<float> trigger;

    /// Effective rectangular boxcar inverse sigma [n_rows * widths.size()].
    std::vector<float> box_inv_sigma_lut;
    /// Per-row 1-sample baseline mean in integer units (empty when float).
    std::vector<double> row_mean_1;
    /// 1 when row r is dead / fully masked, 0 when active.
    std::vector<uint8_t> disabled_row;

    /// O(1) lookup tables indexed by width w in [0, max_width].
    std::vector<uint32_t> width_nearest_filter;
    std::vector<uint32_t> width_bracket_lo;
    std::vector<uint32_t> width_bracket_hi;
    std::vector<float> width_log_lerp;

    KadaneSurrogatePlan() = default;
    KadaneSurrogatePlan(std::span<const uint32_t> widths,
                        bool use_trapezoid,
                        SizeType n_rows,
                        std::span<const float> raw_thresh_lut,
                        std::span<const float> inv_sigma_lut,
                        std::span<const double> integer_mean_lut = {},
                        double integer_unit                      = 1.0,
                        bool allow_decimation                    = true);

    [[nodiscard]] float row_bias(SizeType k, SizeType r) const noexcept {
        return bias[k * row_stride + r];
    }
    [[nodiscard]] float row_trigger(SizeType k, SizeType r) const noexcept {
        return trigger[k * row_stride + r];
    }

    /// O(1) nearest filter index in `widths` for arbitrary width >= 1.
    [[nodiscard]] uint32_t nearest_filter_idx(uint32_t width) const noexcept;

    /// O(1) exact/interpolated rectangular boxcar inverse sigma for `row` and
    /// arbitrary width >= 1.
    [[nodiscard]] float interpolate_inv_sigma(SizeType row,
                                              uint32_t width) const noexcept;

    /// O(1) matched-filter S/N evaluation for a contiguous subarray sum
    /// `raw_sum` of length `width` on `row`. Writes nearest `filter_idx` in
    /// `widths` to `*out_filter_idx` when non-null.
    [[nodiscard]] float
    eval_surrogate_snr(SizeType row,
                       double raw_sum,
                       uint32_t width,
                       uint32_t* out_filter_idx = nullptr) const noexcept;

    [[nodiscard]] SizeType memory_bytes() const noexcept {
        return ((widths.size() + width_nearest_filter.size() +
                 width_bracket_lo.size() + width_bracket_hi.size()) *
                sizeof(uint32_t)) +
               ((bias.size() + trigger.size() + box_inv_sigma_lut.size() +
                 width_log_lerp.size()) *
                sizeof(float)) +
               (row_mean_1.size() * sizeof(double)) +
               (disabled_row.size() * sizeof(uint8_t));
    }
};

[[nodiscard]] KadaneSurrogatePlan
make_kadane_surrogate_plan(std::span<const uint32_t> widths,
                           bool use_trapezoid,
                           SizeType n_rows,
                           std::span<const float> raw_thresh_lut,
                           std::span<const float> inv_sigma_lut,
                           std::span<const double> integer_mean_lut = {},
                           double integer_unit                      = 1.0,
                           bool allow_decimation                    = true);

} // namespace dmt::kadane
