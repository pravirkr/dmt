#pragma once

#include <cmath>
#include <cstdint>

#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_surrogate.hpp"
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#include "dmt/gpu_compat.cuh"
#endif

namespace dmt::kadane {

/// Compact device descriptor for `BoxcarScreenMode::kKadane` (Design C).
struct KadaneGateDeviceView {
    SizeType num_buckets = 0;
    SizeType row_stride  = 0;
    uint8_t int_shift    = 0;
    bool has_int32       = false;
    bool any_edge_lag    = false;

    KadaneGateBucket buckets[kMaxGateBuckets]{};
    uint32_t max_edge_lag[kMaxGateBuckets]{};

    const float* bias           = nullptr; ///< [num_buckets * row_stride]
    const float* cut            = nullptr; ///< [num_buckets * row_stride]
    const int32_t* ibias        = nullptr; ///< [num_buckets * row_stride]
    const int32_t* icut         = nullptr; ///< [num_buckets * row_stride]
    const uint8_t* disabled_row = nullptr; ///< [n_rows]
    const double* row_int_m1    = nullptr; ///< [n_rows] when packed & !int32

    float* state_fc             = nullptr; ///< [tasks * num_buckets]
    int32_t* state_ic           = nullptr; ///< [tasks * num_buckets]
    LagWindow<float>* lag_f32   = nullptr; ///< [tasks * num_buckets] optional
    LagWindow<int32_t>* lag_i32 = nullptr; ///< [tasks * num_buckets] optional
};

/// Compact device descriptor & O(1) S/N evaluator for `kKadaneSurrogate`.
struct KadaneSurrogateDeviceView {
    SizeType num_passes = 0;
    SizeType n_rows     = 0;
    SizeType row_stride = 0;
    SizeType n_filters  = 0;
    uint32_t max_width  = 0;
    bool has_decimated  = false;
    bool is_integer     = false;
    bool int_u32_direct = false;
    double integer_unit = 1.0;

    uint32_t decimate_factors[kMaxSurrogatePasses]{};

    const float* bias            = nullptr; ///< [num_passes * row_stride]
    const float* trigger         = nullptr; ///< [num_passes * row_stride]
    const float* box_inv_sigma   = nullptr; ///< [n_rows * n_filters]
    const double* row_mean_1     = nullptr; ///< [n_rows] when is_integer
    const double* row_int_m1     = nullptr; ///< [n_rows] when packed centered
    const uint8_t* disabled_row  = nullptr; ///< [n_rows]
    const uint32_t* w_nearest    = nullptr; ///< [max_width + 1]
    const uint32_t* w_bracket_lo = nullptr; ///< [max_width + 1]
    const uint32_t* w_bracket_hi = nullptr; ///< [max_width + 1]
    const float* w_log_lerp      = nullptr; ///< [max_width + 1]

    KadaneSurrogateRowState* row_states = nullptr; ///< [tasks]
    uint64_t* row_pos                   = nullptr; ///< [tasks]

    [[nodiscard]] DMT_HD float row_bias(SizeType k, SizeType r) const noexcept {
        return bias[(k * row_stride) + r];
    }

    [[nodiscard]] DMT_HD float row_trigger(SizeType k,
                                           SizeType r) const noexcept {
        return trigger[(k * row_stride) + r];
    }

    [[nodiscard]] DMT_HD uint32_t
    nearest_filter_idx(uint32_t width) const noexcept {
        if (n_filters == 0 || width <= 1U) {
            return 0U;
        }
        if (width >= max_width) {
            return static_cast<uint32_t>(n_filters - 1);
        }
        return w_nearest[width];
    }

    [[nodiscard]] DMT_HD float interpolate_inv_sigma(
        SizeType row, uint32_t width, uint32_t min_width) const noexcept {
        if (row >= n_rows || disabled_row[row] != 0U || width == 0U) {
            return 0.0F;
        }
        const float* row_inv = box_inv_sigma + (row * n_filters);
        if (width < min_width) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            const float scale = sqrtf(static_cast<float>(min_width) /
                                      static_cast<float>(width));
#else
            const float scale = std::sqrt(static_cast<float>(min_width) /
                                          static_cast<float>(width));
#endif
            return row_inv[0] * scale;
        }
        if (width >= max_width) {
            const float inv_max = row_inv[n_filters - 1];
            if (width == max_width) {
                return inv_max;
            }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            const float scale = sqrtf(static_cast<float>(max_width) /
                                      static_cast<float>(width));
#else
            const float scale = std::sqrt(static_cast<float>(max_width) /
                                          static_cast<float>(width));
#endif
            return inv_max * scale;
        }
        const uint32_t lo = w_bracket_lo[width];
        const uint32_t hi = w_bracket_hi[width];
        const float t     = w_log_lerp[width];
        if (lo == hi || t <= 0.0F) {
            return row_inv[lo];
        }
        const float inv_lo = row_inv[lo];
        const float inv_hi = row_inv[hi];
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return inv_lo * exp2f(t * log2f(inv_hi / inv_lo));
#else
        return inv_lo * std::exp2(t * std::log2(inv_hi / inv_lo));
#endif
    }

    [[nodiscard]] DMT_HD float
    eval_surrogate_snr(SizeType row,
                       double raw_sum,
                       uint32_t width,
                       uint32_t min_width,
                       uint32_t& out_filter_idx) const noexcept {
        out_filter_idx = nearest_filter_idx(width);
        if (row >= n_rows || disabled_row[row] != 0U || width == 0U) {
            return 0.0F;
        }
        const float inv_s = interpolate_inv_sigma(row, width, min_width);
        if (inv_s <= 0.0F) {
            return 0.0F;
        }
        double signal = raw_sum;
        if (is_integer && row_mean_1 != nullptr) {
            signal =
                (raw_sum - (row_mean_1[row] * static_cast<double>(width))) *
                integer_unit;
        }
        return static_cast<float>(signal * static_cast<double>(inv_s));
    }
};

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)

/// Portable 64-bit logical-warp `shfl_up` built on 32-bit `gpu_compat::shfl_up`
/// so all CUDA and ROCm/HIP versions work without 64-bit intrinsic gaps.
[[nodiscard]] DMT_D inline int64_t shfl_up_i64(int64_t value, unsigned delta) {
    const auto u  = static_cast<uint64_t>(value);
    const auto lo = gpu_compat::shfl_up(static_cast<uint32_t>(u), delta);
    const auto hi = gpu_compat::shfl_up(static_cast<uint32_t>(u >> 32U), delta);
    return static_cast<int64_t>((static_cast<uint64_t>(hi) << 32U) |
                                static_cast<uint64_t>(lo));
}

[[nodiscard]] DMT_D inline KadaneSegF32 shfl_up_seg(KadaneSegF32 s,
                                                    unsigned delta) {
    return {
        .sum  = gpu_compat::shfl_up(s.sum, delta),
        .pfx  = gpu_compat::shfl_up(s.pfx, delta),
        .sfx  = gpu_compat::shfl_up(s.sfx, delta),
        .best = gpu_compat::shfl_up(s.best, delta),
    };
}

[[nodiscard]] DMT_D inline KadaneSegI64 shfl_up_seg(KadaneSegI64 s,
                                                    unsigned delta) {
    return {
        .sum  = shfl_up_i64(s.sum, delta),
        .pfx  = shfl_up_i64(s.pfx, delta),
        .sfx  = shfl_up_i64(s.sfx, delta),
        .best = shfl_up_i64(s.best, delta),
    };
}

/// Convergent intra-warp ordered tropical reduction over `kLogicalWarp = 32`
/// lanes. Lane `kLogicalWarp - 1` receives the exact total of the warp.
template <typename SegT>
[[nodiscard]] DMT_D inline SegT warp_kadane_reduce(SegT local_seg, int lane) {
    SegT acc = local_seg;
#pragma unroll
    for (unsigned offset = 1;
         offset < static_cast<unsigned>(gpu_compat::kLogicalWarp);
         offset *= 2U) {
        const SegT left = shfl_up_seg(acc, offset);
        if (lane >= static_cast<int>(offset)) {
            acc = kadane_combine(left, acc);
        }
    }
    return acc;
}

/// Convergent warp-0 reduction of `BlockWarps` (`2`, `4`, or `8`) per-warp
/// `KadaneSeg` totals. Lane `BlockWarps - 1` receives the whole-block total.
template <typename SegT, int BlockWarps>
[[nodiscard]] DMT_D inline SegT warp_totals_kadane_reduce(const SegT* warp_segs,
                                                          int lane) {
    SegT w = lane < BlockWarps ? warp_segs[lane] : SegT{};
#pragma unroll
    for (unsigned offset = 1; offset < static_cast<unsigned>(BlockWarps);
         offset *= 2U) {
        const SegT left = shfl_up_seg(w, offset);
        if (lane >= static_cast<int>(offset)) {
            w = kadane_combine(left, w);
        }
    }
    return w;
}

/// Convergent 32-lane logical-warp minimum broadcast to lane 0 (and all lanes).
[[nodiscard]] DMT_D inline float warp_min_f32(float val, int lane) {
    float acc = val;
#pragma unroll
    for (unsigned offset = 1;
         offset < static_cast<unsigned>(gpu_compat::kLogicalWarp);
         offset *= 2U) {
        const float left = gpu_compat::shfl_up(acc, offset);
        if (lane >= static_cast<int>(offset) && left < acc) {
            acc = left;
        }
    }
    return gpu_compat::shfl(acc, gpu_compat::kLogicalWarp - 1);
}

#endif // __CUDACC__ || __HIPCC__ || __HIP__

} // namespace dmt::kadane
