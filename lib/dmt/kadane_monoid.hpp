#pragma once

#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include "dmt/common/types.hpp"

#include "dmt/kadane_plan.hpp"

namespace dmt::kadane {

/// Non-reassociative float32 addition safe under `-ffast-math` /
/// `-use_fast_math` on both host (Clang/GCC) and device (CUDA/HIP).
[[nodiscard]] DMT_HD inline float monoid_add_f32(float a, float b) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(a, b);
#elif defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
    return a + b;
#else
    float res = a + b;
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
#if defined(__aarch64__)
    __asm__("" : "+w"(res));
#elif defined(__AVX512F__)
    __asm__("" : "+v"(res));
#elif defined(__x86_64__)
    __asm__("" : "+x"(res));
#endif
#endif
    return res;
#endif
}

/// Non-reassociative float32 subtraction safe under `-ffast-math` /
/// `-use_fast_math` on both host (Clang/GCC) and device (CUDA/HIP).
[[nodiscard]] DMT_HD inline float monoid_sub_f32(float a, float b) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    return __fsub_rn(a, b);
#elif defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
    return a - b;
#else
    float res = a - b;
#if defined(__GNUC__) && !defined(__CUDA_ARCH__)
#if defined(__aarch64__)
    __asm__("" : "+w"(res));
#elif defined(__AVX512F__)
    __asm__("" : "+v"(res));
#elif defined(__x86_64__)
    __asm__("" : "+x"(res));
#endif
#endif
    return res;
#endif
}

/// Exact float32 maximum (returns +0.0F when comparing 0.0F and -0.0F).
[[nodiscard]] DMT_HD constexpr float monoid_max_f32(float a, float b) noexcept {
    return a > b ? a : b;
}

/// `-ffast-math`-safe finite check for float32 on host and device.
[[nodiscard]] DMT_HD inline bool is_finite_monoid_f32(float x) noexcept {
    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
    return (bits & 0x7F800000U) != 0x7F800000U;
}

/// `-ffast-math`-safe finite check for float64 on host and device.
[[nodiscard]] DMT_HD inline bool is_finite_monoid_f64(double x) noexcept {
    const volatile uint64_t bits = std::bit_cast<uint64_t>(x);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

/// Step a positive float32 down by 1 ULP via bit arithmetic (`-ffast-math`
/// safe on host and device).
[[nodiscard]] DMT_HD constexpr float
step_down_pos_monoid_f32(float x) noexcept {
    if (x <= 0.0F) {
        return 0.0F;
    }
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    return bits > 0U ? std::bit_cast<float>(bits - 1U) : 0.0F;
}

[[nodiscard]] DMT_HD inline double monoid_sqrt_f64(double x) noexcept {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return ::sqrt(x);
#else
    return std::sqrt(x);
#endif
}

// ============================================================================
// Float32 Associative Tropical Kadane Monoid
// ============================================================================

/// Summary of an ordered contiguous float32 segment under biased Kadane steps
/// `d_i = x_i - bias`, with empty-subarray lower bound 0 so `{0, 0, 0, 0}` is
/// the exact two-sided monoid identity.
struct KadaneSegF32 {
    float sum  = 0.0F; ///< Total biased sum over the segment.
    float pfx  = 0.0F; ///< Max prefix sum (>= 0, includes empty prefix).
    float sfx  = 0.0F; ///< Max suffix sum (>= 0, includes empty suffix).
    float best = 0.0F; ///< Max contiguous subarray sum (>= 0).
};

struct KadaneApplyF32 {
    float c_out    = 0.0F; ///< Outgoing Kadane state at segment end (>= 0).
    float tile_max = 0.0F; ///< Max Kadane state across segment (>= 0).
};

[[nodiscard]] DMT_HD constexpr KadaneSegF32 kadane_identity_f32() noexcept {
    return {};
}

/// Construct a 1-sample `KadaneSegF32` (or identity `{0, 0, 0, 0}` when
/// `!valid`, e.g. cold-prefix `sample < warmup` or tail `local >= n`).
[[nodiscard]] DMT_HD inline KadaneSegF32
make_kadane_seg_f32(float x, float bias, bool valid = true) noexcept {
    if (!valid) {
        return {};
    }
    const float d = monoid_sub_f32(x, bias);
    const float p = d > 0.0F ? d : 0.0F;
    return {.sum = d, .pfx = p, .sfx = p, .best = p};
}

/// Associative tropical composition `L (+) R` in left-to-right time order.
[[nodiscard]] DMT_HD inline KadaneSegF32
kadane_combine(KadaneSegF32 L, KadaneSegF32 R) noexcept {
    const float sum   = monoid_add_f32(L.sum, R.sum);
    const float pfx   = monoid_max_f32(L.pfx, monoid_add_f32(L.sum, R.pfx));
    const float sfx   = monoid_max_f32(R.sfx, monoid_add_f32(L.sfx, R.sum));
    const float cross = monoid_add_f32(L.sfx, R.pfx);
    const float best  = monoid_max_f32(monoid_max_f32(L.best, R.best), cross);
    return {.sum = sum, .pfx = pfx, .sfx = sfx, .best = best};
}

/// Apply segment `seg` to incoming state `c_in >= 0`.
[[nodiscard]] DMT_HD inline KadaneApplyF32
kadane_apply(float c_in, KadaneSegF32 seg) noexcept {
    const float c0    = c_in > 0.0F ? c_in : 0.0F;
    const float c_out = monoid_max_f32(seg.sfx, monoid_add_f32(c0, seg.sum));
    const float tile_max =
        monoid_max_f32(seg.best, monoid_add_f32(c0, seg.pfx));
    return {.c_out = c_out, .tile_max = tile_max};
}

// ============================================================================
// Int64 Overflow-Proof Associative Tropical Kadane Monoid
// ============================================================================

/// Summary of an ordered contiguous integer-gate segment with `int64_t`
/// accumulators so 1024-sample intra-tile reductions never overflow or
/// underflow while cross-tile carry clamps to `[0, kKadaneInt32Cap]`.
struct KadaneSegI64 {
    int64_t sum  = 0; ///< Total biased sum over the segment.
    int64_t pfx  = 0; ///< Max prefix sum (>= 0, includes empty prefix).
    int64_t sfx  = 0; ///< Max suffix sum (>= 0, includes empty suffix).
    int64_t best = 0; ///< Max contiguous subarray sum (>= 0).
};

struct KadaneApplyI64 {
    int32_t c_out    = 0; ///< Clamped outgoing carry in [0, kKadaneInt32Cap].
    int64_t tile_max = 0; ///< Exact un-clamped max across segment (>= 0).
};

[[nodiscard]] DMT_HD constexpr KadaneSegI64 kadane_identity_i64() noexcept {
    return {};
}

[[nodiscard]] DMT_HD constexpr int32_t clamp_kadane_i32(int64_t v) noexcept {
    if (v <= 0) {
        return 0;
    }
    if (v >= static_cast<int64_t>(kKadaneInt32Cap)) {
        return kKadaneInt32Cap;
    }
    return static_cast<int32_t>(v);
}

/// Construct a 1-sample `KadaneSegI64` for `delta = (int64_t(x) << int_shift)
/// - ibias` (or `{0, 0, 0, 0}` when `!valid`).
[[nodiscard]] DMT_HD constexpr KadaneSegI64 make_kadane_seg_i64(
    uint64_t x, uint8_t int_shift, int32_t ibias, bool valid = true) noexcept {
    if (!valid) {
        return {};
    }
    const int64_t d =
        static_cast<int64_t>(x << int_shift) - static_cast<int64_t>(ibias);
    const int64_t p = d > 0 ? d : 0;
    return {.sum = d, .pfx = p, .sfx = p, .best = p};
}

/// Exact associative tropical composition `L (+) R` in left-to-right order.
[[nodiscard]] DMT_HD constexpr KadaneSegI64
kadane_combine(KadaneSegI64 L, KadaneSegI64 R) noexcept {
    const int64_t sum      = L.sum + R.sum;
    const int64_t l_plus_r = L.sum + R.pfx;
    const int64_t pfx      = L.pfx > l_plus_r ? L.pfx : l_plus_r;
    const int64_t s_plus_r = L.sfx + R.sum;
    const int64_t sfx      = R.sfx > s_plus_r ? R.sfx : s_plus_r;
    const int64_t lr_max   = L.best > R.best ? L.best : R.best;
    const int64_t cross    = L.sfx + R.pfx;
    const int64_t best     = lr_max > cross ? lr_max : cross;
    return {.sum = sum, .pfx = pfx, .sfx = sfx, .best = best};
}

/// Apply segment `seg` to incoming `int32_t` carry `c_in`, clamping outgoing
/// `c_out` to `[0, kKadaneInt32Cap]`.
[[nodiscard]] DMT_HD constexpr KadaneApplyI64
kadane_apply(int32_t c_in, KadaneSegI64 seg) noexcept {
    const int64_t c0     = c_in > 0 ? static_cast<int64_t>(c_in) : int64_t{0};
    const int64_t c_sum  = c0 + seg.sum;
    const int64_t c_full = seg.sfx > c_sum ? seg.sfx : c_sum;
    const int64_t c_pfx  = c0 + seg.pfx;
    const int64_t tmax   = seg.best > c_pfx ? seg.best : c_pfx;
    return {.c_out = clamp_kadane_i32(c_full), .tile_max = tmax};
}

// ============================================================================
// Shared Host/Device Monotone LagWindow & BlockNoise Cut Helper
// ============================================================================

/// Chunk-size-invariant monotonic ring deque tracking `max(c)` over the
/// trailing `[tile_start - max_edge_lag, tile_start)` window for `{1, 2, 1}`
/// trapezoidal filters across arbitrary streaming chunk sizes on CPU and GPU.
template <typename V> struct LagWindow {
    static constexpr uint8_t kCap = 16;
    uint64_t end_sample[kCap]{};
    V val[kCap]{};
    uint8_t head  = 0;
    uint8_t count = 0;

    DMT_HD void reset() noexcept {
        head  = 0;
        count = 0;
    }

    [[nodiscard]] DMT_HD V query_and_prune(uint64_t window_begin) noexcept {
        while (count > 0 && end_sample[head] <= window_begin) {
            head = static_cast<uint8_t>((head + 1U) & (kCap - 1U));
            --count;
        }
        return count > 0 ? val[head] : V{0};
    }

    DMT_HD void push(uint64_t tile_end, V tile_max) noexcept {
        if (tile_max <= V{0}) {
            return;
        }
        while (count > 0) {
            const auto back =
                static_cast<uint8_t>((head + count - 1U) & (kCap - 1U));
            if (val[back] <= tile_max) {
                --count;
            } else {
                break;
            }
        }
        if (count < kCap) {
            const auto back =
                static_cast<uint8_t>((head + count) & (kCap - 1U));
            end_sample[back] = tile_end;
            val[back]        = tile_max;
            ++count;
        } else {
            const auto back =
                static_cast<uint8_t>((head + kCap - 1U) & (kCap - 1U));
            end_sample[back] = tile_end;
        }
    }
};

/// Recompute exact lossless float32 bucket cut on `[bucket.f_lo, bucket.f_hi)`
/// at fixed `b_f32` against an on-the-fly threshold accessor `thr_fn(f)`,
/// matching `KadaneGatePlan::recompute_row_cuts` bit-for-bit up to
/// `bucket.w_hi = 2^20` on both host and device.
template <typename ThrFn>
[[nodiscard]] DMT_HD inline float
compute_bucket_cut_hd_f32(const KadaneGateBucket& bucket,
                          const uint32_t* widths,
                          SizeType nfilters,
                          bool use_trapezoid,
                          float b_f32,
                          ThrFn&& thr_fn) noexcept {
    constexpr float kDeadFloor = 1.0e35F;
    for (SizeType f = 0; f < nfilters; ++f) {
        const float v = thr_fn(f);
        if (!is_finite_monoid_f32(v) || v <= 0.0F || v >= kDeadFloor) {
            return std::numeric_limits<float>::max();
        }
    }
    if (!is_finite_monoid_f32(b_f32) || b_f32 < 0.0F) {
        return 0.0F;
    }
    const double b_val    = static_cast<double>(b_f32);
    const uint32_t w0     = widths[0];
    const uint32_t we0    = use_trapezoid ? trapezoid_edge_width(w0) : 0U;
    const double tilde_w0 = static_cast<double>(w0 - we0);
    const double tilde_thr0 =
        static_cast<double>(thr_fn(SizeType{0})) * (we0 == 0U ? 1.0 : 0.5);
    const double scale_1 = tilde_thr0 / monoid_sqrt_f64(tilde_w0);

    double tau_raw = std::numeric_limits<double>::max();
    double thr_max = 0.0;
    for (uint32_t f = bucket.f_lo; f < bucket.f_hi; ++f) {
        const uint32_t wf     = widths[f];
        const uint32_t we     = use_trapezoid ? trapezoid_edge_width(wf) : 0U;
        const double tilde_wf = static_cast<double>(wf - we);
        const auto thr_f      = static_cast<double>(thr_fn(SizeType{f}));
        const double tilde_th = thr_f * (we == 0U ? 1.0 : 0.5);
        const double cand     = tilde_th - (b_val * tilde_wf);
        tau_raw               = cand < tau_raw ? cand : tau_raw;
        thr_max               = thr_f > thr_max ? thr_f : thr_max;
    }
    if (!is_finite_monoid_f64(tau_raw) || tau_raw <= 0.0) {
        return 0.0F;
    }

    const auto w_hi = static_cast<double>(bucket.w_hi);
    const double fp_err =
        static_cast<double>(FLT_EPSILON) *
        ((w_hi * (tau_raw + (2.0 * b_val) + (2.0 * scale_1))) +
         (32.0 * (thr_max + (b_val * w_hi))));
    const double m1 = 0.008 * scale_1;
    const double m2 = 4.0 * fp_err;
    if (!is_finite_monoid_f64(fp_err) || !is_finite_monoid_f64(m1) ||
        !is_finite_monoid_f64(m2)) {
        return 0.0F;
    }
    const double margin = m1 > m2 ? m1 : m2;
    const double cut_d  = tau_raw - margin;
    if (!is_finite_monoid_f64(cut_d) || cut_d <= 0.0) {
        return 0.0F;
    }
    const auto cut_f32 = static_cast<float>(cut_d);
    if (!is_finite_monoid_f32(cut_f32) || cut_f32 <= 0.0F) {
        return 0.0F;
    }
    return step_down_pos_monoid_f32(cut_f32);
}

} // namespace dmt::kadane
