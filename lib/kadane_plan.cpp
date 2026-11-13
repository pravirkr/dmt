#include "dmt/kadane_plan.hpp"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

namespace dmt::kadane {

namespace {

constexpr float kDeadThreshFloor = 1.0e35F;

bool is_finite_f32(float x) noexcept {
    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
    return (bits & 0x7F800000U) != 0x7F800000U;
}

bool is_finite_f64(double x) noexcept {
    const volatile uint64_t bits = std::bit_cast<uint64_t>(x);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

/// Step a positive float down by 1 ULP using integer bit arithmetic so
/// `-ffast-math` / `-ffinite-math-only` cannot fold or warn on infinities.
float step_down_pos_f32(float x) noexcept {
    if (x <= 0.0F) {
        return 0.0F;
    }
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    return bits > 0U ? std::bit_cast<float>(bits - 1U) : 0.0F;
}

bool is_dead_row(std::span<const float> row_thr) noexcept {
    if (row_thr.empty()) {
        return true;
    }
    return std::ranges::any_of(row_thr, [](float v) {
        return !is_finite_f32(v) || v <= 0.0F || v >= kDeadThreshFloor;
    });
}

void validate_strictly_ascending(std::span<const uint32_t> widths) {
    for (SizeType i = 1; i < widths.size(); ++i) {
        if (widths[i] <= widths[i - 1]) {
            throw std::invalid_argument(
                "KadanePlan: widths must be strictly ascending");
        }
    }
}

void validate_gate_widths(std::span<const uint32_t> widths) {
    if (widths.empty() || widths.size() > kMaxKadaneGateFilters ||
        widths.front() == 0U || widths.back() > kMaxKadaneGateWidth) {
        throw std::invalid_argument(
            "KadaneGatePlan: widths must have 1..4096 entries within "
            "1..1048576");
    }
    validate_strictly_ascending(widths);
}

void validate_surrogate_widths(std::span<const uint32_t> widths) {
    if (widths.empty() || widths.front() == 0U ||
        widths.back() > kMaxKadaneSurrogateWidth) {
        throw std::invalid_argument(
            "KadaneSurrogatePlan: widths must be non-empty within 1..4096");
    }
    validate_strictly_ascending(widths);
}

struct HullPoint {
    double x = 0.0;
    double y = 0.0;
};

struct BiasOptimum {
    double bias     = 0.0;
    double exponent = 0.0; ///< Optimal b * tau(b).
};

/// Incremental lower convex hull of points (x, y) added in non-decreasing x
/// order, solving max_{b > 0} b * min_i (y_i - b * x_i) in O(log H) time.
class IncrementalLowerHull {
public:
    void reset(SizeType reserve_n) {
        m_pts.clear();
        m_pts.reserve(reserve_n);
        m_head = 0;
    }

    void push(double x, double y) {
        if (!m_pts.empty() && m_pts.back().x == x) {
            if (y >= m_pts.back().y) {
                return;
            }
            m_pts.pop_back();
        }
        while (m_pts.size() >= 2) {
            const auto& a = m_pts[m_pts.size() - 2];
            const auto& b = m_pts[m_pts.size() - 1];
            if ((b.y - a.y) * (x - b.x) >= (y - b.y) * (b.x - a.x)) {
                m_pts.pop_back();
            } else {
                break;
            }
        }
        m_pts.push_back({x, y});
        if (m_head >= m_pts.size()) {
            m_head = m_pts.size() - 1;
        }
        while (m_head + 1 < m_pts.size() &&
               m_pts[m_head + 1].y <= m_pts[m_head].y) {
            ++m_head;
        }
    }

    [[nodiscard]] BiasOptimum solve() const noexcept {
        const SizeType n_act = m_pts.size() - m_head;
        if (n_act == 0) {
            return {};
        }
        if (n_act == 1) {
            const auto& p  = m_pts[m_head];
            const double b = p.y / (2.0 * p.x);
            return {.bias = b, .exponent = 0.25 * p.y * p.y / p.x};
        }
        if (n_act == 2) {
            const auto& p0  = m_pts[m_head];
            const auto& p1  = m_pts[m_head + 1];
            const double s  = (p1.y - p0.y) / (p1.x - p0.x);
            const double v0 = p0.y / (2.0 * p0.x);
            if (v0 <= s) {
                return {.bias = v0, .exponent = 0.25 * p0.y * p0.y / p0.x};
            }
            const double v1 = p1.y / (2.0 * p1.x);
            if (v1 >= s) {
                return {.bias = v1, .exponent = 0.25 * p1.y * p1.y / p1.x};
            }
            return {.bias = s, .exponent = s * (p0.y - s * p0.x)};
        }

        SizeType lo = m_head;
        SizeType hi = m_pts.size() - 1;
        while (lo < hi) {
            const SizeType mid = lo + ((hi - lo) / 2);
            const auto& a      = m_pts[mid];
            const auto& b      = m_pts[mid + 1];
            const double s     = (b.y - a.y) / (b.x - a.x);
            const double tau   = a.y - (s * a.x);
            if (tau <= 0.0) {
                hi = mid;
                continue;
            }
            const double deriv_right = tau - (s * b.x);
            if (deriv_right > 0.0) {
                lo = mid + 1;
                continue;
            }
            const double deriv_left = tau - (s * a.x);
            if (deriv_left < 0.0) {
                hi = mid;
                continue;
            }
            return {.bias = s, .exponent = s * tau};
        }
        const auto& p  = m_pts[lo];
        const double b = p.y / (2.0 * p.x);
        return {.bias = b, .exponent = 0.25 * p.y * p.y / p.x};
    }

private:
    std::vector<HullPoint> m_pts;
    SizeType m_head = 0;
};

/// Compute exact lossless float32 bucket cut on `[bucket.f_lo, bucket.f_hi)`
/// for a fixed float32 `b_f32`, absorbing worst-case `bucket.w_hi`-step
/// float32 drift (for `bucket.w_hi` up to `2^20`) and clamping non-positive or
/// non-finite cuts safely to `0.0F`.
float compute_bucket_cut_f32(const KadaneGateBucket& bucket,
                             std::span<const uint32_t> widths,
                             bool use_trapezoid,
                             float b_f32,
                             std::span<const float> row_thr) noexcept {
    if (!is_finite_f32(b_f32) || b_f32 < 0.0F) {
        return 0.0F;
    }
    const double b_val    = static_cast<double>(b_f32);
    const uint32_t w0     = widths[0];
    const uint32_t we0    = use_trapezoid ? trapezoid_edge_width(w0) : 0U;
    const double tilde_w0 = static_cast<double>(w0 - we0);
    const double tilde_thr0 =
        static_cast<double>(row_thr[0]) * (we0 == 0U ? 1.0 : 0.5);
    const double scale_1 = tilde_thr0 / std::sqrt(tilde_w0);

    double tau_raw = std::numeric_limits<double>::max();
    double thr_max = 0.0;
    for (uint32_t f = bucket.f_lo; f < bucket.f_hi; ++f) {
        const uint32_t wf     = widths[f];
        const uint32_t we     = use_trapezoid ? trapezoid_edge_width(wf) : 0U;
        const double tilde_wf = static_cast<double>(wf - we);
        const auto thr_f      = static_cast<double>(row_thr[f]);
        const double tilde_th = thr_f * (we == 0U ? 1.0 : 0.5);
        tau_raw = std::min(tau_raw, tilde_th - (b_val * tilde_wf));
        thr_max = std::max(thr_max, thr_f);
    }
    if (!is_finite_f64(tau_raw) || tau_raw <= 0.0) {
        return 0.0F;
    }

    const auto w_hi = static_cast<double>(bucket.w_hi);
    const double fp_err =
        static_cast<double>(FLT_EPSILON) *
        ((w_hi * (tau_raw + (2.0 * b_val) + (2.0 * scale_1))) +
         (32.0 * (thr_max + (b_val * w_hi))));
    const double margin = std::max(0.008 * scale_1, 4.0 * fp_err);
    const double cut_d  = tau_raw - margin;
    if (!is_finite_f64(cut_d) || cut_d <= 0.0) {
        return 0.0F;
    }
    const auto cut_f32 = static_cast<float>(cut_d);
    if (!is_finite_f32(cut_f32) || cut_f32 <= 0.0F) {
        return 0.0F;
    }
    return step_down_pos_f32(cut_f32);
}

SizeType aligned_row_stride(SizeType n_rows) {
    if (n_rows == 0) {
        return 0;
    }
    if (n_rows > SIZE_MAX - (kKadaneRowAlign - 1)) {
        throw std::invalid_argument("KadanePlan: n_rows overflow");
    }
    return ((n_rows + kKadaneRowAlign - 1) / kKadaneRowAlign) * kKadaneRowAlign;
}

} // namespace

KadaneGatePlan::KadaneGatePlan(std::span<const uint32_t> widths_in,
                               bool use_trapezoid_in,
                               SizeType n_rows_in,
                               std::span<const float> raw_thresh_lut,
                               std::span<const double> integer_mean_lut,
                               std::span<const uint64_t> integer_thresh_lut,
                               bool enable_int32_gate,
                               uint64_t max_int_sample)
    : use_trapezoid(use_trapezoid_in),
      widths(widths_in.begin(), widths_in.end()) {
    validate_gate_widths(widths_in);
    const SizeType num_filters = widths.size();
    if (n_rows_in > SIZE_MAX / num_filters ||
        raw_thresh_lut.size() != n_rows_in * num_filters) {
        throw std::invalid_argument(fmt::format(
            "KadaneGatePlan: raw_thresh_lut size ({}) must equal n_rows * "
            "widths.size() ({})",
            raw_thresh_lut.size(), n_rows_in * num_filters));
    }
    const bool has_int_tables =
        !integer_mean_lut.empty() || !integer_thresh_lut.empty();
    if (has_int_tables &&
        (integer_mean_lut.size() != n_rows_in * num_filters ||
         integer_thresh_lut.size() != n_rows_in * num_filters)) {
        throw std::invalid_argument(
            "KadaneGatePlan: integer_mean_lut and integer_thresh_lut must "
            "both be empty or have n_rows * widths.size() entries");
    }

    n_rows     = n_rows_in;
    row_stride = aligned_row_stride(n_rows);

    std::vector<uint32_t> w_edge(num_filters, 0U);
    std::vector<uint32_t> tilde_w(num_filters, 0U);
    std::vector<uint32_t> int_weight(num_filters, 0U);
    std::vector<double> thr_scale(num_filters, 1.0);
    for (SizeType f = 0; f < num_filters; ++f) {
        w_edge[f]     = use_trapezoid ? trapezoid_edge_width(widths[f]) : 0U;
        tilde_w[f]    = widths[f] - w_edge[f];
        int_weight[f] = w_edge[f] == 0U ? widths[f] : 2U * tilde_w[f];
        thr_scale[f]  = w_edge[f] == 0U ? 1.0 : 0.5;
    }

    // Select bucket count K in 1..3 based on width span and bank cardinality.
    const uint32_t w_max = widths.back();
    SizeType target_k    = 3;
    if (num_filters == 1 || w_max <= 4U) {
        target_k = 1;
    } else if (num_filters == 2 || w_max <= 32U) {
        target_k = 2;
    }
    num_buckets = std::min({target_k, num_filters, kMaxGateBuckets});

    // Build representative normalized threshold curve across active rows
    // (taking the pointwise minimum normalized curve so universal bucket splits
    // protect the least-smeared / white-noise rows of the plane).
    disabled_row.assign(n_rows, 1U);
    std::vector<SizeType> active_rows;
    active_rows.reserve(n_rows);
    for (SizeType r = 0; r < n_rows; ++r) {
        const auto row_thr =
            raw_thresh_lut.subspan(r * num_filters, num_filters);
        const bool int_dead =
            has_int_tables && integer_thresh_lut[r * num_filters] == UINT64_MAX;
        if (!int_dead && !is_dead_row(row_thr)) {
            disabled_row[r] = 0U;
            active_rows.push_back(r);
        }
    }

    std::vector<double> rep_y(num_filters, std::numeric_limits<double>::max());
    if (!active_rows.empty()) {
        const SizeType sample_cnt = std::min<SizeType>(active_rows.size(), 64);
        for (SizeType idx = 0; idx < sample_cnt; ++idx) {
            const SizeType r =
                active_rows[(idx * active_rows.size()) / sample_cnt];
            const float* row_ptr = raw_thresh_lut.data() + (r * num_filters);
            const double norm0 = static_cast<double>(row_ptr[0]) * thr_scale[0];
            for (SizeType f = 0; f < num_filters; ++f) {
                const double y =
                    (static_cast<double>(row_ptr[f]) * thr_scale[f]) / norm0;
                rep_y[f] = std::min(rep_y[f], y);
            }
        }
    } else {
        for (SizeType f = 0; f < num_filters; ++f) {
            const auto wf = static_cast<double>(widths[f]);
            const auto we = static_cast<double>(w_edge[f]);
            rep_y[f]      = w_edge[f] == 0U
                                ? std::sqrt(wf)
                                : 0.5 * std::sqrt((4.0 * wf) - (6.0 * we));
        }
    }

    // Exact O(K * M^2) dynamic programming maximizing min_k (b_k * tau_k)
    // using an O(1)-amortized incremental lower convex hull per start index s.
    std::vector<double> dp1(num_filters, -1.0);
    std::vector<double> dp2(num_filters, -1.0);
    std::vector<double> dp3(num_filters, -1.0);
    std::vector<uint32_t> split2(num_filters, 0U);
    std::vector<uint32_t> split3(num_filters, 0U);

    IncrementalLowerHull hull;
    for (SizeType s = 0; s < num_filters; ++s) {
        hull.reset(num_filters - s);
        const double prev1 = s >= 1 ? dp1[s - 1] : -1.0;
        const double prev2 = s >= 2 ? dp2[s - 1] : -1.0;
        for (SizeType j = s; j < num_filters; ++j) {
            hull.push(static_cast<double>(tilde_w[j]), rep_y[j]);
            const double score = hull.solve().exponent;
            if (s == 0) {
                dp1[j] = score;
            }
            if (num_buckets >= 2 && s >= 1) {
                const double cand = std::min(prev1, score);
                if (cand > dp2[j]) {
                    dp2[j]    = cand;
                    split2[j] = static_cast<uint32_t>(s);
                }
            }
            if (num_buckets >= 3 && s >= 2) {
                const double cand = std::min(prev2, score);
                if (cand > dp3[j]) {
                    dp3[j]    = cand;
                    split3[j] = static_cast<uint32_t>(s);
                }
            }
        }
    }

    std::array<uint32_t, kMaxGateBuckets + 1> bounds{};
    bounds[0]           = 0U;
    bounds[num_buckets] = static_cast<uint32_t>(num_filters);
    if (num_buckets == 2) {
        bounds[1] = split2[num_filters - 1];
    } else if (num_buckets == 3) {
        const uint32_t s2 = split3[num_filters - 1];
        bounds[2]         = s2;
        bounds[1]         = split2[s2 - 1];
    }

    for (SizeType k = 0; k < num_buckets; ++k) {
        const uint32_t f_lo = bounds[k];
        const uint32_t f_hi = bounds[k + 1];
        uint32_t lag        = 0U;
        for (uint32_t f = f_lo; f < f_hi; ++f) {
            lag = std::max(lag, w_edge[f]);
        }
        buckets[k] = KadaneGateBucket{
            .f_lo         = f_lo,
            .f_hi         = f_hi,
            .w_lo         = widths[f_lo],
            .w_hi         = widths[f_hi - 1],
            .max_edge_lag = lag,
        };
        max_edge_lag[k] = lag;
    }

    // Per-row float32 biases and lossless cuts in SoA layout [k * row_stride +
    // r].
    bias.assign(num_buckets * row_stride, 0.0F);
    cut.assign(num_buckets * row_stride, std::numeric_limits<float>::max());

    for (const SizeType r : active_rows) {
        const auto row_thr =
            raw_thresh_lut.subspan(r * num_filters, num_filters);
        for (SizeType k = 0; k < num_buckets; ++k) {
            const auto& bkt = buckets[k];
            hull.reset(bkt.f_hi - bkt.f_lo);
            for (uint32_t f = bkt.f_lo; f < bkt.f_hi; ++f) {
                hull.push(static_cast<double>(tilde_w[f]),
                          static_cast<double>(row_thr[f]) * thr_scale[f]);
            }
            const double b_opt = hull.solve().bias;
            float b_f32        = static_cast<float>(b_opt);
            if (!is_finite_f32(b_f32) || b_f32 <= 0.0F) {
                b_f32 = std::numeric_limits<float>::min();
            }
            bias[(k * row_stride) + r] = b_f32;
            cut[(k * row_stride) + r]  = compute_bucket_cut_f32(
                bkt, widths, use_trapezoid, b_f32, row_thr);
        }
    }

    // Optional exact int32_t gate with zero rounding margin.
    // Requires an explicit nonzero `max_int_sample` upper bound from the caller
    // so bright pulses / RFI can never wrap `(x_t << int_shift)` in signed i32.
    if (enable_int32_gate && max_int_sample > 0U && has_int_tables &&
        !active_rows.empty()) {
        const auto w_max_u64 = static_cast<uint64_t>(widths.back());
        constexpr auto kInt32Max =
            static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        constexpr auto kSampleCap      = static_cast<uint64_t>(kKadaneInt32Cap);
        constexpr uint8_t kMaxIntShift = 8U;

        // Precompute continuous per-(active_row, bucket) excess slopes and
        // check basic table validity before choosing plane-uniform `int_shift`.
        bool tables_valid   = max_int_sample <= kSampleCap;
        double min_b_excess = std::numeric_limits<double>::max();
        std::vector<double> row_mu1(active_rows.size(), 0.0);
        std::vector<double> b_exc_tbl(active_rows.size() * num_buckets, 0.0);

        for (SizeType idx = 0; tables_valid && idx < active_rows.size();
             ++idx) {
            const SizeType r     = active_rows[idx];
            const double* mean_r = integer_mean_lut.data() + (r * num_filters);
            const uint64_t* thr_r =
                integer_thresh_lut.data() + (r * num_filters);

            const double mu_1 = mean_r[0] / static_cast<double>(int_weight[0]);
            if (!is_finite_f64(mu_1) || mu_1 < 0.0) {
                tables_valid = false;
                break;
            }
            row_mu1[idx] = mu_1;
            for (SizeType f = 0; f < num_filters; ++f) {
                if (!is_finite_f64(mean_r[f]) || mean_r[f] < 0.0) {
                    tables_valid = false;
                    break;
                }
            }
            if (!tables_valid) {
                break;
            }
            for (SizeType k = 0; k < num_buckets; ++k) {
                const auto& bkt = buckets[k];
                hull.reset(bkt.f_hi - bkt.f_lo);
                for (uint32_t f = bkt.f_lo; f < bkt.f_hi; ++f) {
                    const double excess =
                        (static_cast<double>(thr_r[f]) - mean_r[f]) *
                        thr_scale[f];
                    hull.push(static_cast<double>(tilde_w[f]),
                              std::max(1.0e-9, excess));
                }
                const double b_exc                 = hull.solve().bias;
                b_exc_tbl[(idx * num_buckets) + k] = b_exc;
                min_b_excess = std::min(min_b_excess, b_exc);
            }
        }

        uint8_t pref_shift = 0U;
        if (tables_valid && min_b_excess > 0.0) {
            while (pref_shift < kMaxIntShift &&
                   (min_b_excess * static_cast<double>(1ULL << pref_shift)) <
                       4.0) {
                if (max_int_sample > (kSampleCap >> (pref_shift + 1U))) {
                    break;
                }
                ++pref_shift;
            }
        }

        // Bounding `(x_t << shift_u) <= INT32_MAX / 2`, `ibias <=
        // INT32_MAX / (4 * w_max)`, and `1 <= icut <= INT32_MAX / 4`
        // guarantees `min(c, INT32_MAX / 2) + delta` cannot overflow signed
        // int32_t and stays `>= INT32_MAX / 4 >= icut` across `w_max` steps
        // after any clamp (for `w_max` up to 2^20).
        constexpr uint64_t kQuarterInt32Max =
            static_cast<uint64_t>(kInt32Max / 4);
        const auto b_cap = static_cast<int64_t>(kQuarterInt32Max / w_max_u64);

        bool found_shift = false;
        if (tables_valid && b_cap >= 1) {
            ibias.assign(num_buckets * row_stride, 0);
            icut.assign(num_buckets * row_stride,
                        std::numeric_limits<int32_t>::max());
        }
        for (int s = static_cast<int>(pref_shift);
             tables_valid && b_cap >= 1 && s >= 0; --s) {
            const auto shift_u       = static_cast<uint8_t>(s);
            const uint64_t scale_u64 = 1ULL << shift_u;
            const auto scale_f64     = static_cast<double>(scale_u64);
            if (max_int_sample > (kSampleCap >> shift_u)) {
                continue;
            }

            bool shift_ok = true;
            for (SizeType idx = 0; shift_ok && idx < active_rows.size();
                 ++idx) {
                const SizeType r = active_rows[idx];
                const uint64_t* thr_r =
                    integer_thresh_lut.data() + (r * num_filters);

                const double mu_s = row_mu1[idx] * scale_f64;
                if (!is_finite_f64(mu_s) ||
                    mu_s >= static_cast<double>(b_cap)) {
                    shift_ok = false;
                    break;
                }
                // Strictly negative drift requires ibias > (mu_1 << shift_u).
                const auto b_min = static_cast<int64_t>(std::floor(mu_s)) + 1;

                for (SizeType k = 0; k < num_buckets; ++k) {
                    const auto& bkt = buckets[k];
                    const double b_exc =
                        b_exc_tbl[(idx * num_buckets) + k] * scale_f64;
                    const double target_b = std::floor(mu_s + b_exc);
                    if (!is_finite_f64(target_b) ||
                        target_b > static_cast<double>(b_cap)) {
                        shift_ok = false;
                        break;
                    }
                    const int64_t cand0 = std::max<int64_t>(
                        b_min, static_cast<int64_t>(target_b));
                    const int64_t cand1 = cand0 + 1;

                    const auto eval_cut =
                        [&](int64_t b_int) noexcept -> int64_t {
                        int64_t min_cut = std::numeric_limits<int64_t>::max();
                        for (uint32_t f = bkt.f_lo; f < bkt.f_hi; ++f) {
                            if (thr_r[f] > (UINT64_MAX >> (shift_u + 1U))) {
                                return -1;
                            }
                            const uint64_t raw_t = thr_r[f] << shift_u;
                            const auto eff_t = w_edge[f] == 0U
                                                   ? static_cast<int64_t>(raw_t)
                                                   : static_cast<int64_t>(
                                                         (raw_t + 1ULL) / 2ULL);
                            const int64_t c_val =
                                eff_t -
                                (b_int * static_cast<int64_t>(tilde_w[f]));
                            min_cut = std::min(min_cut, c_val);
                        }
                        return min_cut;
                    };

                    int64_t chosen_b = cand0;
                    int64_t chosen_c = eval_cut(cand0);
                    if (cand1 <= b_cap) {
                        const int64_t alt_c = eval_cut(cand1);
                        if (alt_c > 0) {
                            const double score0 =
                                (static_cast<double>(cand0) - mu_s) *
                                static_cast<double>(chosen_c);
                            const double score1 =
                                (static_cast<double>(cand1) - mu_s) *
                                static_cast<double>(alt_c);
                            if (score1 > score0) {
                                chosen_b = cand1;
                                chosen_c = alt_c;
                            }
                        }
                    }

                    if (chosen_b <= 0 || chosen_b > b_cap ||
                        (static_cast<uint64_t>(chosen_b) * w_max_u64) >
                            kQuarterInt32Max ||
                        chosen_c < 1 || chosen_c > (kInt32Max / 4)) {
                        shift_ok = false;
                        break;
                    }
                    ibias[(k * row_stride) + r] =
                        static_cast<int32_t>(chosen_b);
                    icut[(k * row_stride) + r] = static_cast<int32_t>(chosen_c);
                }
            }

            if (shift_ok) {
                int_shift   = shift_u;
                has_int32   = true;
                found_shift = true;
                break;
            }
        }

        if (!found_shift) {
            ibias.clear();
            icut.clear();
            int_shift = 0U;
            has_int32 = false;
        }
    }
}

void KadaneGatePlan::recompute_row_cuts(SizeType row,
                                        std::span<const float> min_thr,
                                        std::span<float> out_cuts) const {
    if (row >= n_rows || min_thr.size() != widths.size() ||
        out_cuts.size() < num_buckets) {
        throw std::invalid_argument(
            "KadaneGatePlan::recompute_row_cuts: invalid row or span sizes");
    }
    if (disabled_row[row] != 0U || is_dead_row(min_thr)) {
        for (SizeType k = 0; k < num_buckets; ++k) {
            out_cuts[k] = std::numeric_limits<float>::max();
        }
        return;
    }
    for (SizeType k = 0; k < num_buckets; ++k) {
        out_cuts[k] = compute_bucket_cut_f32(buckets[k], widths, use_trapezoid,
                                             row_bias(k, row), min_thr);
    }
}

KadaneGatePlan
make_kadane_gate_plan(std::span<const uint32_t> widths,
                      bool use_trapezoid,
                      SizeType n_rows,
                      std::span<const float> raw_thresh_lut,
                      std::span<const double> integer_mean_lut,
                      std::span<const uint64_t> integer_thresh_lut,
                      bool enable_int32_gate,
                      uint64_t max_int_sample) {
    return KadaneGatePlan(widths, use_trapezoid, n_rows, raw_thresh_lut,
                          integer_mean_lut, integer_thresh_lut,
                          enable_int32_gate, max_int_sample);
}

void recompute_cuts_for_biases(const KadaneGatePlan& plan,
                               std::span<const float> row_biases,
                               std::span<const float> min_thr,
                               std::span<float> out_cuts) {
    if (row_biases.size() < plan.num_buckets ||
        min_thr.size() != plan.widths.size() ||
        out_cuts.size() < plan.num_buckets) {
        throw std::invalid_argument(
            "recompute_cuts_for_biases: invalid span sizes");
    }
    if (is_dead_row(min_thr)) {
        for (SizeType k = 0; k < plan.num_buckets; ++k) {
            out_cuts[k] = std::numeric_limits<float>::max();
        }
        return;
    }
    for (SizeType k = 0; k < plan.num_buckets; ++k) {
        out_cuts[k] =
            compute_bucket_cut_f32(plan.buckets[k], plan.widths,
                                   plan.use_trapezoid, row_biases[k], min_thr);
    }
}

KadaneSurrogatePlan::KadaneSurrogatePlan(
    std::span<const uint32_t> widths_in,
    bool use_trapezoid_in,
    SizeType n_rows_in,
    std::span<const float> raw_thresh_lut,
    std::span<const float> inv_sigma_lut,
    std::span<const double> integer_mean_lut,
    double integer_unit_in,
    bool allow_decimation)
    : n_rows(n_rows_in),
      row_stride(aligned_row_stride(n_rows_in)),
      use_trapezoid(use_trapezoid_in),
      is_integer(!integer_mean_lut.empty()),
      integer_unit(integer_unit_in),
      widths(widths_in.begin(), widths_in.end()) {
    validate_surrogate_widths(widths_in);
    const SizeType num_filters = widths.size();
    if (n_rows > SIZE_MAX / num_filters ||
        raw_thresh_lut.size() != n_rows * num_filters ||
        inv_sigma_lut.size() != n_rows * num_filters) {
        throw std::invalid_argument(
            "KadaneSurrogatePlan: raw_thresh_lut and inv_sigma_lut must have "
            "n_rows * widths.size() entries");
    }
    if (is_integer && (integer_mean_lut.size() != n_rows * num_filters ||
                       !is_finite_f64(integer_unit) || integer_unit <= 0.0)) {
        throw std::invalid_argument(
            "KadaneSurrogatePlan: invalid integer_mean_lut or integer_unit");
    }

    max_width = widths.back();

    // Precompute O(1) width -> nearest bank filter & log-log bracket weights.
    width_nearest_filter.assign(max_width + 1U, 0U);
    width_bracket_lo.assign(max_width + 1U, 0U);
    width_bracket_hi.assign(max_width + 1U, 0U);
    width_log_lerp.assign(max_width + 1U, 0.0F);

    SizeType cur_hi = 0;
    for (uint32_t w = 1U; w <= max_width; ++w) {
        while (cur_hi + 1 < num_filters && widths[cur_hi] < w) {
            ++cur_hi;
        }
        if (w <= widths[0]) {
            width_nearest_filter[w] = 0U;
            width_bracket_lo[w]     = 0U;
            width_bracket_hi[w]     = 0U;
            width_log_lerp[w]       = 0.0F;
        } else if (widths[cur_hi] == w) {
            const auto idx          = static_cast<uint32_t>(cur_hi);
            width_nearest_filter[w] = idx;
            width_bracket_lo[w]     = idx;
            width_bracket_hi[w]     = idx;
            width_log_lerp[w]       = 0.0F;
        } else {
            const SizeType lo      = cur_hi - 1;
            const uint32_t w_lo    = widths[lo];
            const uint32_t w_hi    = widths[cur_hi];
            const uint32_t dist_lo = w - w_lo;
            const uint32_t dist_hi = w_hi - w;
            width_nearest_filter[w] =
                static_cast<uint32_t>(dist_lo <= dist_hi ? lo : cur_hi);
            width_bracket_lo[w] = static_cast<uint32_t>(lo);
            width_bracket_hi[w] = static_cast<uint32_t>(cur_hi);
            const double log_num =
                std::log(static_cast<double>(w) / static_cast<double>(w_lo));
            const double log_den =
                std::log(static_cast<double>(w_hi) / static_cast<double>(w_lo));
            width_log_lerp[w] = static_cast<float>(log_num / log_den);
        }
    }

    // Build effective rectangular boxcar inv_sigma table per row.
    // When `use_trapezoid == false`, this copies `inv_sigma_lut` directly;
    // when `use_trapezoid == true`, widths >= 3 are converted from {1, 2, 1}
    // trapezoid sigma to rectangular subarray sigma so unweighted Kadane
    // excursion sums evaluate unbiased S/N.
    box_inv_sigma_lut.assign(n_rows * num_filters, 0.0F);
    disabled_row.assign(n_rows, 1U);
    if (is_integer) {
        row_mean_1.assign(n_rows, 0.0);
    }
    const uint32_t we0 = use_trapezoid ? trapezoid_edge_width(widths[0]) : 0U;
    const double g0 = we0 == 0U ? static_cast<double>(widths[0])
                                : static_cast<double>(2U * (widths[0] - we0));

    for (SizeType r = 0; r < n_rows; ++r) {
        const auto row_thr =
            raw_thresh_lut.subspan(r * num_filters, num_filters);
        const auto row_inv =
            inv_sigma_lut.subspan(r * num_filters, num_filters);
        if (is_dead_row(row_thr) || !is_finite_f32(row_inv[0]) ||
            row_inv[0] <= 0.0F) {
            continue;
        }
        bool inv_valid = true;
        for (const float inv : row_inv) {
            if (!is_finite_f32(inv) || inv <= 0.0F) {
                inv_valid = false;
                break;
            }
        }
        if (!inv_valid) {
            continue;
        }
        disabled_row[r] = 0U;
        if (is_integer) {
            const double m1 = integer_mean_lut[r * num_filters] / g0;
            if (!is_finite_f64(m1) || m1 < 0.0) {
                throw std::invalid_argument(
                    "KadaneSurrogatePlan: negative or non-finite integer mean");
            }
            row_mean_1[r] = m1;
        }

        const double w0_var_den =
            we0 == 0U ? static_cast<double>(widths[0])
                      : ((4.0 * static_cast<double>(widths[0])) -
                         (6.0 * static_cast<double>(we0)));
        const double sigma1_sq = w0_var_den / (static_cast<double>(row_inv[0]) *
                                               static_cast<double>(row_inv[0]));
        for (SizeType f = 0; f < num_filters; ++f) {
            const uint32_t wf = widths[f];
            const uint32_t we = use_trapezoid ? trapezoid_edge_width(wf) : 0U;
            if (we == 0U) {
                box_inv_sigma_lut[(r * num_filters) + f] = row_inv[f];
                continue;
            }
            const auto wd          = static_cast<double>(wf);
            const auto ed          = static_cast<double>(we);
            const double trap_var  = 1.0 / (static_cast<double>(row_inv[f]) *
                                            static_cast<double>(row_inv[f]));
            const double white_den = (4.0 * wd) - (6.0 * ed);
            const double rho_white = wd / white_den;
            const double half_wt   = 2.0 * (wd - ed);
            const double rho_smear = (wd * wd) / (half_wt * half_wt);
            const double obs_ratio = trap_var / (sigma1_sq * white_den);
            const double smear_den =
                std::max(1.0e-9, ((half_wt * half_wt) / white_den) - 1.0);
            const double alpha =
                std::clamp((obs_ratio - 1.0) / smear_den, 0.0, 1.0);
            const double rho_eff =
                ((1.0 - alpha) * rho_white) + (alpha * rho_smear);
            const double box_var = trap_var * rho_eff;
            box_inv_sigma_lut[(r * num_filters) + f] =
                static_cast<float>(1.0 / std::sqrt(box_var));
        }
    }

    // Partition [1, max_width] into K <= 3 geometric width strips, enabling
    // fused f=4 decimation on wide strips when max_width >= 16 &
    // !use_trapezoid.
    const bool use_decim =
        allow_decimation && (max_width >= 16U) && !use_trapezoid;
    has_decimated = use_decim;

    if (max_width <= 8U) {
        num_passes = 1;
        passes[0]  = {.w_lo = 1U, .w_hi = max_width, .decimate_factor = 1U};
    } else if (max_width <= 128U) {
        num_passes       = 2;
        uint32_t w_split = static_cast<uint32_t>(
            std::llround(std::sqrt(static_cast<double>(max_width))));
        if (use_decim) {
            w_split = std::clamp<uint32_t>(w_split, 12U, 16U);
            w_split = std::min<uint32_t>(w_split, max_width - 4U);
        } else {
            w_split = std::clamp<uint32_t>(w_split, 2U, max_width - 1U);
        }
        passes[0] = {.w_lo = 1U, .w_hi = w_split, .decimate_factor = 1U};
        passes[1] = {.w_lo            = w_split + 1U,
                     .w_hi            = max_width,
                     .decimate_factor = use_decim ? 4U : 1U};
    } else {
        num_passes        = 3;
        const auto w_d    = static_cast<double>(max_width);
        uint32_t w_split1 = static_cast<uint32_t>(std::llround(std::cbrt(w_d)));
        if (use_decim) {
            w_split1 = std::max<uint32_t>(w_split1, 12U);
        } else {
            w_split1 = std::max<uint32_t>(w_split1, 4U);
        }
        uint32_t w_split2 =
            static_cast<uint32_t>(std::llround(std::pow(w_d, 2.0 / 3.0)));
        w_split2 =
            std::clamp<uint32_t>(w_split2, w_split1 + 8U, max_width - 4U);
        passes[0] = {.w_lo = 1U, .w_hi = w_split1, .decimate_factor = 1U};
        passes[1] = {.w_lo            = w_split1 + 1U,
                     .w_hi            = w_split2,
                     .decimate_factor = use_decim ? 4U : 1U};
        passes[2] = {.w_lo            = w_split2 + 1U,
                     .w_hi            = max_width,
                     .decimate_factor = use_decim ? 4U : 1U};
    }

    // Compute per-row strip-center tangent biases and pre-filter triggers.
    bias.assign(num_passes * row_stride, 0.0F);
    trigger.assign(num_passes * row_stride, std::numeric_limits<float>::max());

    const double inv_unit = is_integer ? (1.0 / integer_unit) : 1.0;
    IncrementalLowerHull hull;
    std::vector<uint32_t> strip_widths;
    strip_widths.reserve(num_filters + 3);

    for (SizeType r = 0; r < n_rows; ++r) {
        if (disabled_row[r] != 0U) {
            continue;
        }
        const double detect_u =
            static_cast<double>(raw_thresh_lut[r * num_filters]) *
            static_cast<double>(inv_sigma_lut[r * num_filters]);
        const double mu_1 = is_integer ? row_mean_1[r] : 0.0;

        for (SizeType k = 0; k < num_passes; ++k) {
            const auto& p = passes[k];
            strip_widths.clear();
            strip_widths.push_back(p.w_lo);
            const auto w_mid = static_cast<uint32_t>(std::clamp<long long>(
                std::llround(std::sqrt(static_cast<double>(p.w_lo) * p.w_hi)),
                p.w_lo, p.w_hi));
            strip_widths.push_back(w_mid);
            for (const uint32_t wf : widths) {
                if (wf > p.w_lo && wf < p.w_hi) {
                    strip_widths.push_back(wf);
                }
            }
            strip_widths.push_back(p.w_hi);
            std::ranges::sort(strip_widths);
            const auto dup = std::ranges::unique(strip_widths);
            strip_widths.erase(dup.begin(), dup.end());

            hull.reset(strip_widths.size());
            for (const uint32_t w : strip_widths) {
                const double inv_s =
                    static_cast<double>(interpolate_inv_sigma(r, w));
                const double thr_w = detect_u / inv_s;
                hull.push(static_cast<double>(w), thr_w);
            }
            const double b_samp = hull.solve().bias;

            double tau_strip = std::numeric_limits<double>::max();
            for (const uint32_t w : strip_widths) {
                const double inv_s =
                    static_cast<double>(interpolate_inv_sigma(r, w));
                const double thr_w = detect_u / inv_s;
                tau_strip = std::min(tau_strip,
                                     thr_w - (b_samp * static_cast<double>(w)));
            }

            const auto dec_d         = static_cast<double>(p.decimate_factor);
            const double step_b      = (mu_1 + (b_samp * inv_unit)) * dec_d;
            const double trig_factor = p.decimate_factor == 1U ? 0.80 : 0.65;
            const double trig_val =
                std::max(0.0, trig_factor * tau_strip * inv_unit);

            bias[(k * row_stride) + r] = static_cast<float>(step_b);
            trigger[(k * row_stride) + r] =
                step_down_pos_f32(static_cast<float>(trig_val));
        }
    }
}

uint32_t
KadaneSurrogatePlan::nearest_filter_idx(uint32_t width) const noexcept {
    if (widths.empty() || width <= 1U) {
        return 0U;
    }
    if (width >= max_width) {
        return static_cast<uint32_t>(widths.size() - 1);
    }
    return width_nearest_filter[width];
}

float KadaneSurrogatePlan::interpolate_inv_sigma(
    SizeType row, uint32_t width) const noexcept {
    if (row >= n_rows || disabled_row[row] != 0U || width == 0U) {
        return 0.0F;
    }
    const SizeType num_filters = widths.size();
    const float* row_inv       = box_inv_sigma_lut.data() + (row * num_filters);
    if (width < widths[0]) {
        const float scale = std::sqrt(static_cast<float>(widths[0]) /
                                      static_cast<float>(width));
        return row_inv[0] * scale;
    }
    if (width >= max_width) {
        const float inv_max = row_inv[num_filters - 1];
        if (width == max_width) {
            return inv_max;
        }
        // Extrapolate wide excursions > max_width with asymptotic 1/sqrt(w)
        // noise growth so uncapped excursions never overestimate S/N.
        const float scale = std::sqrt(static_cast<float>(max_width) /
                                      static_cast<float>(width));
        return inv_max * scale;
    }
    const uint32_t lo = width_bracket_lo[width];
    const uint32_t hi = width_bracket_hi[width];
    const float t     = width_log_lerp[width];
    if (lo == hi || t <= 0.0F) {
        return row_inv[lo];
    }
    const float inv_lo = row_inv[lo];
    const float inv_hi = row_inv[hi];
    return inv_lo * std::exp2(t * std::log2(inv_hi / inv_lo));
}

float KadaneSurrogatePlan::eval_surrogate_snr(
    SizeType row,
    double raw_sum,
    uint32_t width,
    uint32_t* out_filter_idx) const noexcept {
    if (out_filter_idx != nullptr) {
        *out_filter_idx = nearest_filter_idx(width);
    }
    if (row >= n_rows || disabled_row[row] != 0U || width == 0U) {
        return 0.0F;
    }
    const float inv_s = interpolate_inv_sigma(row, width);
    if (inv_s <= 0.0F) {
        return 0.0F;
    }
    double signal = raw_sum;
    if (is_integer) {
        signal = (raw_sum - (row_mean_1[row] * static_cast<double>(width))) *
                 integer_unit;
    }
    return static_cast<float>(signal * static_cast<double>(inv_s));
}

KadaneSurrogatePlan
make_kadane_surrogate_plan(std::span<const uint32_t> widths,
                           bool use_trapezoid,
                           SizeType n_rows,
                           std::span<const float> raw_thresh_lut,
                           std::span<const float> inv_sigma_lut,
                           std::span<const double> integer_mean_lut,
                           double integer_unit,
                           bool allow_decimation) {
    return KadaneSurrogatePlan(widths, use_trapezoid, n_rows, raw_thresh_lut,
                               inv_sigma_lut, integer_mean_lut, integer_unit,
                               allow_decimation);
}

} // namespace dmt::kadane
