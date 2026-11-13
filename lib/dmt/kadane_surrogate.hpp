#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "dmt/common/types.hpp"

#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include "dmt/kadane_simd.hpp"
#endif

namespace dmt::kadane {

/// Fixed-capacity `DMT_HD` array avoiding host-only `std::array::operator[]`
/// issues under strict device-pass stdlib configurations.
template <typename T, SizeType Capacity> struct SurrogateFixedArray {
    T elems[Capacity]{};
    DMT_HD constexpr T& operator[](SizeType i) noexcept { return elems[i]; }
    DMT_HD constexpr const T& operator[](SizeType i) const noexcept {
        return elems[i];
    }
};

/// Closed or flushed positive Kadane excursion emitted by
/// `replay_surrogate_row`.
struct SurrogateExcursion {
    uint64_t seg_begin = 0; ///< Inclusive first sample of the excursion.
    uint64_t best_end  = 0; ///< Inclusive end sample where biased sum peaked.
    uint64_t close_end = 0; ///< Inclusive last positive sample before reset.
    double best_raw_sum =
        0.0; ///< Unbiased `double` sum on `[seg_begin, best_end]`.
    uint32_t best_len = 0; ///< Exact sample count `best_end - seg_begin + 1`.
    uint32_t pass_idx = 0; ///< Pass index `k` in `[0, num_passes)`.
    uint32_t best_filter = 0;     ///< Cached filter index when `has_snr`.
    float best_c         = 0.0F;  ///< Peak biased Kadane value in excursion.
    float best_snr       = 0.0F;  ///< Cached calibrated S/N when `has_snr`.
    bool has_snr         = false; ///< True if `best_snr`/`best_filter` cached.
};

/// Streaming per-pass excursion state for `replay_surrogate_row`.
struct KadaneSurrogatePassState {
    uint64_t seg_begin    = 0;
    uint64_t best_end     = 0;
    uint64_t last_pos_end = 0;
    double best_raw_sum   = 0.0;
    double cur_raw_sum    = 0.0;
    uint32_t best_len     = 0;
    uint32_t cur_len      = 0;
    uint32_t best_filter  = 0;
    float best_c          = 0.0F;
    float cur_c           = 0.0F;
    float best_snr        = 0.0F;
    bool open             = false;
    bool triggered        = false;
    bool has_snr          = false;
};

/// Streaming per-row state across tiles and chunks for `kKadaneSurrogate`.
struct KadaneSurrogateRowState {
    SurrogateFixedArray<KadaneSurrogatePassState, kMaxSurrogatePasses> passes{};
    uint64_t synced_sample     = 0;
    uint64_t dec_block_begin   = 0;
    double dec_partial_raw_sum = 0.0;
    float dec_partial_sum      = 0.0F;
    uint32_t dec_valid_len     = 0;
    bool initialized           = false;

    DMT_HD void reset() noexcept { *this = KadaneSurrogateRowState{}; }

    [[nodiscard]] DMT_HD bool has_any_open(SizeType num_passes) const noexcept {
        const SizeType np =
            num_passes < kMaxSurrogatePasses ? num_passes : kMaxSurrogatePasses;
        for (SizeType k = 0; k < np; ++k) {
            if (passes[k].open) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] DMT_HD bool
    has_triggered_open(SizeType num_passes,
                       const float* row_triggers) const noexcept {
        const SizeType np =
            num_passes < kMaxSurrogatePasses ? num_passes : kMaxSurrogatePasses;
        for (SizeType k = 0; k < np; ++k) {
            if (passes[k].open &&
                (passes[k].triggered || passes[k].best_c >= row_triggers[k])) {
                return true;
            }
        }
        return false;
    }
};

namespace detail {

[[nodiscard]] DMT_HD inline SurrogateExcursion
make_excursion(const KadaneSurrogatePassState& st, uint32_t pass_idx) noexcept {
    return SurrogateExcursion{
        .seg_begin    = st.seg_begin,
        .best_end     = st.best_end,
        .close_end    = st.last_pos_end,
        .best_raw_sum = st.best_raw_sum,
        .best_len     = st.best_len,
        .pass_idx     = pass_idx,
        .best_filter  = st.best_filter,
        .best_c       = st.best_c,
        .best_snr     = st.best_snr,
        .has_snr      = st.has_snr,
    };
}

template <typename EmitFn>
DMT_HD inline void step_surrogate_pass(uint32_t pass_idx,
                                       uint64_t step_beg,
                                       uint64_t step_end,
                                       uint32_t step_len,
                                       float y_f32,
                                       double y_f64,
                                       float bias,
                                       float trigger_cut,
                                       KadaneSurrogatePassState& st,
                                       EmitFn& emit_fn) {
    const float delta  = monoid_sub_f32(y_f32, bias);
    const float next_c = monoid_max_f32(0.0F, monoid_add_f32(st.cur_c, delta));
    if (next_c <= 0.0F) {
        if (st.open) {
            if (st.triggered || st.best_c >= trigger_cut) {
                emit_fn(make_excursion(st, pass_idx));
            }
            st = KadaneSurrogatePassState{};
        } else {
            st.cur_c = 0.0F;
        }
        return;
    }

    if (!st.open) {
        st           = KadaneSurrogatePassState{};
        st.seg_begin = step_beg;
        st.open      = true;
    }
    st.cur_c        = next_c;
    st.last_pos_end = step_end;
    st.cur_raw_sum += y_f64;
    st.cur_len += step_len;
    if (next_c >= trigger_cut) {
        st.triggered = true;
    }
    if (next_c > st.best_c) {
        st.best_c       = next_c;
        st.best_end     = step_end;
        st.best_len     = st.cur_len;
        st.best_raw_sum = st.cur_raw_sum;
        st.has_snr      = false;
    }
}

/// Step one stream sample `gt` across all `np` passes (including optional
/// `decimate_factor == 4` quad accumulation) on host or device.
template <typename EmitFn>
DMT_HD inline void step_surrogate_row_sample(uint64_t gt,
                                             float x_sample,
                                             bool valid,
                                             SizeType np,
                                             const uint32_t* dec,
                                             bool any_dec4,
                                             const float* row_biases,
                                             const float* row_triggers,
                                             KadaneSurrogateRowState& row_state,
                                             EmitFn& emit_excursion) {
    const float x_eff = valid ? x_sample : 0.0F;
    for (SizeType k = 0; k < np; ++k) {
        if (dec[k] == 1U) {
            if (!valid) {
                if (row_state.passes[k].open &&
                    (row_state.passes[k].triggered ||
                     row_state.passes[k].best_c >= row_triggers[k])) {
                    emit_excursion(make_excursion(row_state.passes[k],
                                                  static_cast<uint32_t>(k)));
                }
                row_state.passes[k] = KadaneSurrogatePassState{};
            } else {
                step_surrogate_pass(static_cast<uint32_t>(k), gt, gt, 1U, x_eff,
                                    static_cast<double>(x_eff), row_biases[k],
                                    row_triggers[k], row_state.passes[k],
                                    emit_excursion);
            }
        }
    }

    if (any_dec4) {
        const auto ph = static_cast<uint32_t>(gt & 3ULL);
        if (ph == 0U) {
            row_state.dec_partial_sum = x_eff;
            row_state.dec_partial_raw_sum =
                valid ? static_cast<double>(x_eff) : 0.0;
            row_state.dec_block_begin = valid ? gt : (gt + 1ULL);
            row_state.dec_valid_len   = valid ? 1U : 0U;
        } else if (ph < 3U) {
            row_state.dec_partial_sum =
                monoid_add_f32(row_state.dec_partial_sum, x_eff);
            if (valid) {
                if (row_state.dec_valid_len == 0U) {
                    row_state.dec_block_begin = gt;
                }
                row_state.dec_partial_raw_sum += static_cast<double>(x_eff);
                ++row_state.dec_valid_len;
            } else {
                row_state.dec_partial_raw_sum = 0.0;
                row_state.dec_valid_len       = 0U;
            }
        } else {
            const float y = monoid_add_f32(row_state.dec_partial_sum, x_eff);
            for (SizeType k = 0; k < np; ++k) {
                if (dec[k] != 4U) {
                    continue;
                }
                if (!valid) {
                    if (row_state.passes[k].open &&
                        (row_state.passes[k].triggered ||
                         row_state.passes[k].best_c >= row_triggers[k])) {
                        emit_excursion(make_excursion(
                            row_state.passes[k], static_cast<uint32_t>(k)));
                    }
                    row_state.passes[k] = KadaneSurrogatePassState{};
                } else {
                    const uint64_t beg = row_state.dec_valid_len == 0U
                                             ? gt
                                             : row_state.dec_block_begin;
                    const uint32_t len = row_state.dec_valid_len + 1U;
                    const double y_raw = row_state.dec_partial_raw_sum +
                                         static_cast<double>(x_eff);
                    step_surrogate_pass(static_cast<uint32_t>(k), beg, gt, len,
                                        y, y_raw, row_biases[k],
                                        row_triggers[k], row_state.passes[k],
                                        emit_excursion);
                }
            }
            row_state.dec_partial_sum     = 0.0F;
            row_state.dec_partial_raw_sum = 0.0;
            row_state.dec_valid_len       = 0U;
        }
    }
}

} // namespace detail

/// Flush any remaining open triggered excursions at end-of-stream (`flush()`).
template <typename EmitFn>
DMT_HD inline void flush_surrogate_row(SizeType num_passes,
                                       const float* row_triggers,
                                       KadaneSurrogateRowState& row_state,
                                       EmitFn&& emit_excursion) {
    const SizeType np =
        num_passes < kMaxSurrogatePasses ? num_passes : kMaxSurrogatePasses;
    for (SizeType k = 0; k < np; ++k) {
        auto& st = row_state.passes[k];
        if (st.open && (st.triggered || st.best_c >= row_triggers[k])) {
            emit_excursion(
                detail::make_excursion(st, static_cast<uint32_t>(k)));
        }
        st = KadaneSurrogatePassState{};
    }
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)

namespace detail {

template <SizeType K, typename Backend>
[[nodiscard]] inline uint32_t
scan_surrogate_batch_f32(SizeType n_rows,
                         SizeType tile_len,
                         const float* const* row_ptrs,
                         const uint32_t* decimate_factors,
                         const float* bias,
                         SizeType bias_stride,
                         const float* trigger_cut,
                         SizeType cut_stride,
                         float* state_c,
                         SizeType state_stride,
                         uint32_t dec_phase,
                         float* dec_partial_sum,
                         const SizeType* first_valid_t,
                         float* checkpoint_c_out,
                         SizeType checkpoint_c_stride,
                         float* checkpoint_dec_out) noexcept {
    static_assert(K >= 1 && K <= kMaxSurrogatePasses);
    constexpr SizeType kLanes  = Backend::kSimdLanes;
    constexpr SizeType kGroups = Backend::kRowGroups;
    constexpr SizeType kBatch  = Backend::kBatchRows;
    using VecF                 = Backend::VecF;
    using VecI                 = Backend::VecI;

    const VecF zero = Backend::zero_f32();
    if (n_rows == 0) {
        return 0U;
    }

    std::array<uint32_t, K> dec{};
    bool any_dec4 = false;
    for (SizeType k = 0; k < K; ++k) {
        dec[k]   = decimate_factors != nullptr ? decimate_factors[k] : 1U;
        any_dec4 = any_dec4 || (dec[k] == 4U);
    }
    const uint32_t phase0 = dec_phase & 3U;

    std::array<std::array<VecF, kGroups>, K> v_bias{};
    std::array<std::array<VecF, kGroups>, K> v_cut{};
    std::array<std::array<VecF, kGroups>, K> c{};
    std::array<std::array<VecF, kGroups>, K> m{};

    for (SizeType k = 0; k < K; ++k) {
        load_batch_param<Backend, float>(bias, bias_stride, k, n_rows, 0.0F,
                                         v_bias[k]);
        load_batch_param<Backend, float>(trigger_cut, cut_stride, k, n_rows,
                                         std::numeric_limits<float>::max(),
                                         v_cut[k]);
        load_batch_param<Backend, float>(state_c, state_stride, k, n_rows, 0.0F,
                                         c[k]);
        for (SizeType g = 0; g < kGroups; ++g) {
            c[k][g] = Backend::vmax_nonneg_f32(zero, c[k][g]);
            m[k][g] = zero;
        }
    }

    if (checkpoint_c_out != nullptr) {
        const SizeType chk_stride =
            checkpoint_c_stride != 0 ? checkpoint_c_stride : state_stride;
        for (SizeType k = 0; k < K; ++k) {
            store_batch_param<Backend, float>(c[k], checkpoint_c_out,
                                              chk_stride, k, n_rows);
        }
    }

    std::array<VecF, kGroups> acc4{};
    if (any_dec4 && phase0 != 0U && dec_partial_sum != nullptr) {
        load_batch_param<Backend, float>(dec_partial_sum, n_rows, 0, n_rows,
                                         0.0F, acc4);
    } else {
        for (SizeType g = 0; g < kGroups; ++g) {
            acc4[g] = zero;
        }
    }
    if (checkpoint_dec_out != nullptr) {
        store_batch_param<Backend, float>(acc4, checkpoint_dec_out, n_rows, 0,
                                          n_rows);
    }

    if (tile_len == 0) {
        return 0U;
    }

    std::array<const float*, kBatch> ptrs{};
    for (SizeType r = 0; r < kBatch; ++r) {
        ptrs[r] = row_ptrs[r < n_rows ? r : 0];
    }

    SizeType max_fv = 0;
    alignas(64) std::array<int32_t, kBatch> fv_clamped{};
    if (first_valid_t != nullptr) {
        for (SizeType r = 0; r < n_rows; ++r) {
            const SizeType fv = std::min(first_valid_t[r], tile_len);
            max_fv            = std::max(max_fv, fv);
            fv_clamped[r]     = static_cast<int32_t>(fv);
        }
    }

    if (!any_dec4) {
        // Pure full-resolution K-pass value-only SIMD loop.
        const auto step_full_unmasked = [&](SizeType g, VecF x) noexcept {
            for (SizeType k = 0; k < K; ++k) {
                const VecF delta = Backend::sub_f32(x, v_bias[k][g]);
                const VecF next  = Backend::vmax_nonneg_f32(
                    zero, Backend::add_f32(c[k][g], delta));
                m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                c[k][g] = next;
            }
        };

        SizeType t = 0;
        if (max_fv > 0) {
            std::array<VecI, kGroups> v_fv{};
            for (SizeType g = 0; g < kGroups; ++g) {
                v_fv[g] = Backend::loadu_i32(fv_clamped.data() + (g * kLanes));
            }
            const auto step_full_masked = [&](SizeType g, VecF x,
                                              SizeType sample_t) noexcept {
                const VecI vt =
                    Backend::set1_i32(static_cast<int32_t>(sample_t));
                for (SizeType k = 0; k < K; ++k) {
                    const VecF delta    = Backend::sub_f32(x, v_bias[k][g]);
                    const VecF raw_next = Backend::vmax_nonneg_f32(
                        zero, Backend::add_f32(c[k][g], delta));
                    const VecF next =
                        Backend::maskz_valid_f32(raw_next, vt, v_fv[g]);
                    m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                    c[k][g] = next;
                }
            };

            for (; t < max_fv && t + kLanes <= tile_len; t += kLanes) {
                for (SizeType g = 0; g < kGroups; ++g) {
                    const auto cols = Backend::load_transpose_f32(
                        ptrs.data() + (g * kLanes), t);
                    for (SizeType s = 0; s < kLanes; ++s) {
                        if (t + s < max_fv) {
                            step_full_masked(g, cols[s], t + s);
                        } else {
                            step_full_unmasked(g, cols[s]);
                        }
                    }
                }
            }
            if (t < tile_len && t < max_fv) {
                const SizeType t_rem = tile_len - t;
                for (SizeType g = 0; g < kGroups; ++g) {
                    const auto cols = Backend::load_transpose_tail_f32(
                        ptrs.data() + (g * kLanes), t, t_rem);
                    for (SizeType s = 0; s < t_rem; ++s) {
                        if (t + s < max_fv) {
                            step_full_masked(g, cols[s], t + s);
                        } else {
                            step_full_unmasked(g, cols[s]);
                        }
                    }
                }
                t = tile_len;
            }
        }

        for (; t + kLanes <= tile_len; t += kLanes) {
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols =
                    Backend::load_transpose_f32(ptrs.data() + (g * kLanes), t);
                for (SizeType s = 0; s < kLanes; ++s) {
                    step_full_unmasked(g, cols[s]);
                }
            }
        }
        if (t < tile_len) {
            const SizeType t_rem = tile_len - t;
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols = Backend::load_transpose_tail_f32(
                    ptrs.data() + (g * kLanes), t, t_rem);
                for (SizeType s = 0; s < t_rem; ++s) {
                    step_full_unmasked(g, cols[s]);
                }
            }
        }
    } else {
        // Fused full-resolution + f=4 decimated SIMD loop.
        std::array<VecI, kGroups> v_fv{};
        if (max_fv > 0) {
            for (SizeType g = 0; g < kGroups; ++g) {
                v_fv[g] = Backend::loadu_i32(fv_clamped.data() + (g * kLanes));
            }
        }

        const auto step_dec_sample_unmasked = [&](SizeType g, VecF x,
                                                  uint32_t ph) noexcept {
            for (SizeType k = 0; k < K; ++k) {
                if (dec[k] == 1U) {
                    const VecF delta = Backend::sub_f32(x, v_bias[k][g]);
                    const VecF next  = Backend::vmax_nonneg_f32(
                        zero, Backend::add_f32(c[k][g], delta));
                    m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                    c[k][g] = next;
                }
            }
            if (ph == 0U) {
                acc4[g] = x;
            } else if (ph < 3U) {
                acc4[g] = Backend::add_f32(acc4[g], x);
            } else {
                const VecF y = Backend::add_f32(acc4[g], x);
                acc4[g]      = zero;
                for (SizeType k = 0; k < K; ++k) {
                    if (dec[k] == 4U) {
                        const VecF delta = Backend::sub_f32(y, v_bias[k][g]);
                        const VecF next  = Backend::vmax_nonneg_f32(
                            zero, Backend::add_f32(c[k][g], delta));
                        m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                        c[k][g] = next;
                    }
                }
            }
        };

        const auto step_dec_sample_masked = [&](SizeType g, VecF x,
                                                SizeType sample_t,
                                                uint32_t ph) noexcept {
            const VecI vt = Backend::set1_i32(static_cast<int32_t>(sample_t));
            const VecF x_eff = Backend::maskz_valid_f32(x, vt, v_fv[g]);
            for (SizeType k = 0; k < K; ++k) {
                if (dec[k] == 1U) {
                    const VecF delta    = Backend::sub_f32(x_eff, v_bias[k][g]);
                    const VecF raw_next = Backend::vmax_nonneg_f32(
                        zero, Backend::add_f32(c[k][g], delta));
                    const VecF next =
                        Backend::maskz_valid_f32(raw_next, vt, v_fv[g]);
                    m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                    c[k][g] = next;
                }
            }
            if (ph == 0U) {
                acc4[g] = x_eff;
            } else if (ph < 3U) {
                acc4[g] = Backend::add_f32(acc4[g], x_eff);
            } else {
                const VecF y = Backend::add_f32(acc4[g], x_eff);
                acc4[g]      = zero;
                for (SizeType k = 0; k < K; ++k) {
                    if (dec[k] == 4U) {
                        const VecF delta    = Backend::sub_f32(y, v_bias[k][g]);
                        const VecF raw_next = Backend::vmax_nonneg_f32(
                            zero, Backend::add_f32(c[k][g], delta));
                        const VecF next =
                            Backend::maskz_valid_f32(raw_next, vt, v_fv[g]);
                        m[k][g] = Backend::vmax_nonneg_f32(m[k][g], next);
                        c[k][g] = next;
                    }
                }
            }
        };

        SizeType t = 0;
        if (max_fv > 0) {
            for (; t < max_fv && t + kLanes <= tile_len; t += kLanes) {
                for (SizeType g = 0; g < kGroups; ++g) {
                    const auto cols = Backend::load_transpose_f32(
                        ptrs.data() + (g * kLanes), t);
                    for (SizeType s = 0; s < kLanes; ++s) {
                        const uint32_t ph =
                            (phase0 + static_cast<uint32_t>(t + s)) & 3U;
                        if (t + s < max_fv) {
                            step_dec_sample_masked(g, cols[s], t + s, ph);
                        } else {
                            step_dec_sample_unmasked(g, cols[s], ph);
                        }
                    }
                }
            }
            if (t < tile_len && t < max_fv) {
                const SizeType t_rem = tile_len - t;
                for (SizeType g = 0; g < kGroups; ++g) {
                    const auto cols = Backend::load_transpose_tail_f32(
                        ptrs.data() + (g * kLanes), t, t_rem);
                    for (SizeType s = 0; s < t_rem; ++s) {
                        const uint32_t ph =
                            (phase0 + static_cast<uint32_t>(t + s)) & 3U;
                        if (t + s < max_fv) {
                            step_dec_sample_masked(g, cols[s], t + s, ph);
                        } else {
                            step_dec_sample_unmasked(g, cols[s], ph);
                        }
                    }
                }
                t = tile_len;
            }
        }

        // Fast unrolled 4-sample quad path when aligned (`(phase0 + t) % 4 ==
        // 0` and `kLanes >= 4`).
        if constexpr (kLanes >= 4) {
            if (((phase0 + static_cast<uint32_t>(t)) & 3U) == 0U) {
                for (; t + kLanes <= tile_len; t += kLanes) {
                    for (SizeType g = 0; g < kGroups; ++g) {
                        const auto cols = Backend::load_transpose_f32(
                            ptrs.data() + (g * kLanes), t);
                        for (SizeType q = 0; q < kLanes; q += 4) {
                            step_dec_sample_unmasked(g, cols[q + 0], 0U);
                            step_dec_sample_unmasked(g, cols[q + 1], 1U);
                            step_dec_sample_unmasked(g, cols[q + 2], 2U);
                            step_dec_sample_unmasked(g, cols[q + 3], 3U);
                        }
                    }
                }
            }
        }

        for (; t + kLanes <= tile_len; t += kLanes) {
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols =
                    Backend::load_transpose_f32(ptrs.data() + (g * kLanes), t);
                for (SizeType s = 0; s < kLanes; ++s) {
                    const uint32_t ph =
                        (phase0 + static_cast<uint32_t>(t + s)) & 3U;
                    step_dec_sample_unmasked(g, cols[s], ph);
                }
            }
        }

        if (t < tile_len) {
            const SizeType t_rem = tile_len - t;
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols = Backend::load_transpose_tail_f32(
                    ptrs.data() + (g * kLanes), t, t_rem);
                for (SizeType s = 0; s < t_rem; ++s) {
                    const uint32_t ph =
                        (phase0 + static_cast<uint32_t>(t + s)) & 3U;
                    step_dec_sample_unmasked(g, cols[s], ph);
                }
            }
        }

        if (dec_partial_sum != nullptr) {
            store_batch_param<Backend, float>(acc4, dec_partial_sum, n_rows, 0,
                                              n_rows);
        }
    }

    std::array<uint32_t, kGroups> group_hits{};
    for (SizeType k = 0; k < K; ++k) {
        store_batch_param<Backend, float>(c[k], state_c, state_stride, k,
                                          n_rows);
        for (SizeType g = 0; g < kGroups; ++g) {
            group_hits[g] |= Backend::cmpge_mask_f32(m[k][g], v_cut[k][g]);
        }
    }
    uint32_t hit_mask = 0U;
    for (SizeType g = 0; g < kGroups; ++g) {
        hit_mask |= (group_hits[g] << static_cast<uint32_t>(g * kLanes));
    }
    return hit_mask & active_row_mask(n_rows);
}

/// Reconstruct exact open excursion state at `tile_start_sample` when silent
/// tiles preceded a triggered tile and `checkpoint_c > 0.0F`, using upper/lower
/// Sandwich coalescing (`0 <= c_true <= trigger_cut`) or exact `synced_sample`
/// replay on `lookback` (`[tile_start_sample - L, tile_start_sample)`).
inline KadaneSurrogatePassState
reconstruct_fledgling_excursion(uint64_t tile_start_sample,
                                uint32_t dec_factor,
                                float bias,
                                float trigger_cut,
                                float target_c,
                                std::span<const float> lookback,
                                uint64_t row_warmup,
                                const KadaneSurrogateRowState& prev_state,
                                SizeType pass_idx) noexcept {
    if (target_c <= 0.0F) {
        return {};
    }
    const bool has_base =
        prev_state.initialized && prev_state.synced_sample <= tile_start_sample;
    const uint64_t base_sample = has_base ? prev_state.synced_sample : 0ULL;
    const uint64_t sync_dist   = tile_start_sample - base_sample;
    const SizeType max_avail =
        sync_dist > 0 ? static_cast<SizeType>(
                            std::min<uint64_t>(lookback.size(), sync_dist))
                      : 0;
    if (max_avail == 0) {
        if (sync_dist == 0 && has_base) {
            return prev_state.passes[pass_idx];
        }
        KadaneSurrogatePassState fallback{};
        fallback.seg_begin    = tile_start_sample;
        fallback.best_end     = tile_start_sample;
        fallback.last_pos_end = tile_start_sample;
        fallback.cur_c        = target_c;
        fallback.best_c       = target_c;
        fallback.open         = true;
        return fallback;
    }

    const auto noop_emit = [](const SurrogateExcursion&) noexcept {};
    SizeType win         = std::min<SizeType>(64, max_avail);
    while (true) {
        SizeType eff_win = win;
        if (dec_factor == 4U && eff_win < max_avail) {
            const uint64_t start_g = tile_start_sample - eff_win;
            const auto phase_rem   = static_cast<SizeType>(start_g & 3ULL);
            eff_win                = std::min(max_avail, eff_win + phase_rem);
        }

        const uint64_t lb_start = tile_start_sample - eff_win;
        const bool exact_seed   = (lb_start == base_sample);
        const float* lb_ptr     = lookback.data() + (lookback.size() - eff_win);
        KadaneSurrogatePassState cand = (exact_seed && has_base)
                                            ? prev_state.passes[pass_idx]
                                            : KadaneSurrogatePassState{};
        float hi_c                    = exact_seed ? 0.0F : trigger_cut;
        bool hi_hit_zero              = exact_seed || (hi_c <= 0.0F);

        if (dec_factor == 1U) {
            for (SizeType i = 0; i < eff_win; ++i) {
                const uint64_t gt = lb_start + i;
                if (gt < row_warmup) {
                    cand        = KadaneSurrogatePassState{};
                    hi_c        = 0.0F;
                    hi_hit_zero = true;
                    continue;
                }
                const float x = lb_ptr[i];
                if (!hi_hit_zero) {
                    const float d = ScalarBackend::sub_f32(x, bias);
                    hi_c          = ScalarBackend::vmax_nonneg_f32(
                        0.0F, ScalarBackend::add_f32(hi_c, d));
                    hi_hit_zero = (hi_c <= 0.0F);
                }
                step_surrogate_pass(0U, gt, gt, 1U, x, static_cast<double>(x),
                                    bias, std::numeric_limits<float>::max(),
                                    cand, noop_emit);
            }
        } else {
            const auto start_ph = static_cast<uint32_t>(lb_start & 3ULL);
            const SizeType skip =
                (exact_seed || start_ph == 0U)
                    ? 0
                    : std::min<SizeType>(eff_win, 4U - start_ph);
            float p_sum =
                (exact_seed && has_base) ? prev_state.dec_partial_sum : 0.0F;
            double p_raw =
                (exact_seed && has_base) ? prev_state.dec_partial_raw_sum : 0.0;
            uint64_t blk_beg = (exact_seed && has_base)
                                   ? prev_state.dec_block_begin
                                   : (lb_start + skip);
            uint32_t valid_cnt =
                (exact_seed && has_base) ? prev_state.dec_valid_len : 0U;
            for (SizeType i = skip; i < eff_win; ++i) {
                const uint64_t gt = lb_start + i;
                const uint32_t ph = static_cast<uint32_t>(gt & 3ULL);
                const bool valid  = gt >= row_warmup;
                const float x_eff = valid ? lb_ptr[i] : 0.0F;
                if (ph == 0U) {
                    p_sum     = x_eff;
                    p_raw     = valid ? static_cast<double>(x_eff) : 0.0;
                    blk_beg   = valid ? gt : (gt + 1ULL);
                    valid_cnt = valid ? 1U : 0U;
                } else if (ph < 3U) {
                    p_sum = ScalarBackend::add_f32(p_sum, x_eff);
                    if (valid) {
                        if (valid_cnt == 0U) {
                            blk_beg = gt;
                        }
                        p_raw += static_cast<double>(x_eff);
                        ++valid_cnt;
                    } else {
                        p_raw     = 0.0;
                        valid_cnt = 0U;
                    }
                } else {
                    const float y = ScalarBackend::add_f32(p_sum, x_eff);
                    if (!valid) {
                        cand        = KadaneSurrogatePassState{};
                        hi_c        = 0.0F;
                        hi_hit_zero = true;
                    } else {
                        if (!hi_hit_zero) {
                            const float d = ScalarBackend::sub_f32(y, bias);
                            hi_c          = ScalarBackend::vmax_nonneg_f32(
                                0.0F, ScalarBackend::add_f32(hi_c, d));
                            hi_hit_zero = (hi_c <= 0.0F);
                        }
                        const uint64_t beg = valid_cnt == 0U ? gt : blk_beg;
                        const uint32_t len = valid_cnt + 1U;
                        const double y_raw = p_raw + static_cast<double>(x_eff);
                        step_surrogate_pass(0U, beg, gt, len, y, y_raw, bias,
                                            std::numeric_limits<float>::max(),
                                            cand, noop_emit);
                    }
                    p_sum     = 0.0F;
                    p_raw     = 0.0;
                    valid_cnt = 0U;
                }
            }
        }

        if ((hi_hit_zero && cand.cur_c == target_c) || eff_win >= max_avail) {
            if (!cand.open) {
                cand.seg_begin    = tile_start_sample;
                cand.best_end     = tile_start_sample;
                cand.last_pos_end = tile_start_sample;
                cand.best_c       = target_c;
                cand.open         = true;
            }
            cand.cur_c = target_c;
            return cand;
        }
        win = std::min(max_avail, eff_win * 2);
    }
}

} // namespace detail

// ============================================================================
// Kernel 3: Surrogate Value-Only SIMD Tile Scanner + Exact Streaming Replay
// ============================================================================

/**
 * @brief Fast value-only SIMD tile scanner for `kKadaneSurrogate` (`K
 * in 1..3`).
 *
 * Scans `1 <= n_rows <= kMaxBatchRows` rows over `tile_len` samples, supporting
 * fused `decimate_factor == 4` 4-sample register accumulation across arbitrary
 * cross-tile `dec_phase = global_t % 4` and `dec_partial_sum` state. Optionally
 * checkpoints tile-start `(state_c, dec_partial_sum)` before scanning, updates
 * them to tile-end values, and returns a `uint32_t` bitmask (`m_k >=
 * trigger_cut_k`).
 */
template <SizeType K, typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_surrogate_tile_f32(SizeType n_rows,
                        SizeType tile_len,
                        const float* const* row_ptrs,
                        const uint32_t* decimate_factors,
                        const float* bias,
                        SizeType bias_stride,
                        const float* trigger_cut,
                        SizeType cut_stride,
                        float* state_c,
                        SizeType state_stride,
                        uint32_t dec_phase            = 0U,
                        float* dec_partial_sum        = nullptr,
                        const SizeType* first_valid_t = nullptr,
                        float* checkpoint_c_out       = nullptr,
                        SizeType checkpoint_c_stride  = 0,
                        float* checkpoint_dec_out     = nullptr) noexcept {
    constexpr SizeType kBatch = Backend::kBatchRows;
    if (n_rows <= kBatch) {
        return detail::scan_surrogate_batch_f32<K, Backend>(
            n_rows, tile_len, row_ptrs, decimate_factors, bias, bias_stride,
            trigger_cut, cut_stride, state_c, state_stride, dec_phase,
            dec_partial_sum, first_valid_t, checkpoint_c_out,
            checkpoint_c_stride, checkpoint_dec_out);
    }
    const SizeType cap_rows   = std::min(n_rows, kMaxBatchRows);
    uint32_t combined_hitmask = 0U;
    for (SizeType r0 = 0; r0 < cap_rows; r0 += kBatch) {
        const SizeType sub_n   = std::min(kBatch, cap_rows - r0);
        const uint32_t sub_hit = detail::scan_surrogate_batch_f32<K, Backend>(
            sub_n, tile_len, row_ptrs + r0, decimate_factors, bias + r0,
            bias_stride, trigger_cut + r0, cut_stride, state_c + r0,
            state_stride, dec_phase,
            dec_partial_sum != nullptr ? dec_partial_sum + r0 : nullptr,
            first_valid_t != nullptr ? first_valid_t + r0 : nullptr,
            checkpoint_c_out != nullptr ? checkpoint_c_out + r0 : nullptr,
            checkpoint_c_stride,
            checkpoint_dec_out != nullptr ? checkpoint_dec_out + r0 : nullptr);
        combined_hitmask |= (sub_hit << static_cast<uint32_t>(r0));
    }
    return combined_hitmask;
}

/// Runtime pass-count (`num_passes in {1, 2, 3}`) dispatcher for Kernel 3.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_surrogate_tile_f32(SizeType num_passes,
                        SizeType n_rows,
                        SizeType tile_len,
                        const float* const* row_ptrs,
                        const uint32_t* decimate_factors,
                        const float* bias,
                        SizeType bias_stride,
                        const float* trigger_cut,
                        SizeType cut_stride,
                        float* state_c,
                        SizeType state_stride,
                        uint32_t dec_phase            = 0U,
                        float* dec_partial_sum        = nullptr,
                        const SizeType* first_valid_t = nullptr,
                        float* checkpoint_c_out       = nullptr,
                        SizeType checkpoint_c_stride  = 0,
                        float* checkpoint_dec_out     = nullptr) noexcept {
    switch (num_passes) {
    case 1:
        return scan_surrogate_tile_f32<1, Backend>(
            n_rows, tile_len, row_ptrs, decimate_factors, bias, bias_stride,
            trigger_cut, cut_stride, state_c, state_stride, dec_phase,
            dec_partial_sum, first_valid_t, checkpoint_c_out,
            checkpoint_c_stride, checkpoint_dec_out);
    case 2:
        return scan_surrogate_tile_f32<2, Backend>(
            n_rows, tile_len, row_ptrs, decimate_factors, bias, bias_stride,
            trigger_cut, cut_stride, state_c, state_stride, dec_phase,
            dec_partial_sum, first_valid_t, checkpoint_c_out,
            checkpoint_c_stride, checkpoint_dec_out);
    case 3:
        return scan_surrogate_tile_f32<3, Backend>(
            n_rows, tile_len, row_ptrs, decimate_factors, bias, bias_stride,
            trigger_cut, cut_stride, state_c, state_stride, dec_phase,
            dec_partial_sum, first_valid_t, checkpoint_c_out,
            checkpoint_c_stride, checkpoint_dec_out);
    default:
        return 0U;
    }
}

/// `KadaneSurrogatePlan` convenience overload for `scan_surrogate_tile_f32`.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_surrogate_tile_f32(const KadaneSurrogatePlan& plan,
                        SizeType row_offset,
                        SizeType n_rows,
                        SizeType tile_len,
                        const float* const* row_ptrs,
                        float* state_c,
                        SizeType state_stride,
                        uint32_t dec_phase            = 0U,
                        float* dec_partial_sum        = nullptr,
                        const SizeType* first_valid_t = nullptr,
                        float* checkpoint_c_out       = nullptr,
                        SizeType checkpoint_c_stride  = 0,
                        float* checkpoint_dec_out     = nullptr) noexcept {
    if (row_offset >= plan.n_rows || plan.num_passes == 0) {
        return 0U;
    }
    const SizeType eff_rows = std::min(n_rows, plan.n_rows - row_offset);
    std::array<uint32_t, kMaxSurrogatePasses> dec_factors{};
    for (SizeType k = 0; k < plan.num_passes; ++k) {
        dec_factors[k] = plan.passes[k].decimate_factor;
    }
    return scan_surrogate_tile_f32<Backend>(
        plan.num_passes, eff_rows, tile_len, row_ptrs, dec_factors.data(),
        plan.bias.data() + row_offset, plan.row_stride,
        plan.trigger.data() + row_offset, plan.row_stride, state_c,
        state_stride, dec_phase, dec_partial_sum, first_valid_t,
        checkpoint_c_out, checkpoint_c_stride, checkpoint_dec_out);
}

/**
 * @brief Replay a triggered or open surrogate row across `tile_len` samples,
 * tracking exact streaming excursions `(seg_begin, best_end, best_len,
 * best_raw_sum, cur_c, cur_raw_sum, cur_len, open)` across tile/chunk
 * boundaries and emitting closed excursions via `emit_excursion`.
 *
 * @return True if any pass still holds an open excursion with `best_c >=
 * row_triggers[k]` at tile end (meaning the next tile must also replay this
 * row even if its tile max stays below `trigger_cut`).
 */
template <typename EmitFn>
inline bool replay_surrogate_row(uint64_t tile_start_sample,
                                 SizeType tile_len,
                                 const float* row_samples,
                                 SizeType num_passes,
                                 const uint32_t* decimate_factors,
                                 const float* row_biases,
                                 const float* row_triggers,
                                 KadaneSurrogateRowState& row_state,
                                 EmitFn&& emit_excursion,
                                 SizeType first_valid_t          = 0,
                                 const float* checkpoint_c       = nullptr,
                                 float checkpoint_dec_partial    = 0.0F,
                                 std::span<const float> lookback = {},
                                 uint64_t row_warmup             = 0) {
    const SizeType np = std::min(num_passes, kMaxSurrogatePasses);
    if (np == 0) {
        return false;
    }

    std::array<uint32_t, kMaxSurrogatePasses> dec{};
    bool any_dec4 = false;
    for (SizeType k = 0; k < np; ++k) {
        dec[k]   = decimate_factors != nullptr ? decimate_factors[k] : 1U;
        any_dec4 = any_dec4 || (dec[k] == 4U);
    }

    const uint64_t eff_warmup = std::max(
        row_warmup,
        first_valid_t > 0 ? (tile_start_sample + first_valid_t) : uint64_t{0});
    const auto phase0 = static_cast<uint32_t>(tile_start_sample & 3ULL);

    // Resync from tile-start checkpoint if silent tiles were skipped.
    if (!row_state.initialized ||
        row_state.synced_sample != tile_start_sample) {
        const KadaneSurrogateRowState prev_state = row_state;
        for (SizeType k = 0; k < np; ++k) {
            const float c0 = checkpoint_c != nullptr ? checkpoint_c[k] : 0.0F;
            if (c0 <= 0.0F) {
                row_state.passes[k] = KadaneSurrogatePassState{};
            } else {
                row_state.passes[k] = detail::reconstruct_fledgling_excursion(
                    tile_start_sample, dec[k], row_biases[k], row_triggers[k],
                    c0, lookback, eff_warmup, prev_state, k);
            }
        }
        row_state.dec_partial_sum     = 0.0F;
        row_state.dec_partial_raw_sum = 0.0;
        row_state.dec_block_begin     = tile_start_sample;
        row_state.dec_valid_len       = 0U;
        if (any_dec4 && phase0 != 0U) {
            if (lookback.size() >= phase0 && tile_start_sample >= phase0) {
                const uint64_t blk_start = tile_start_sample - phase0;
                const float* p_lb =
                    lookback.data() + (lookback.size() - phase0);
                for (uint32_t q = 0U; q < phase0; ++q) {
                    const uint64_t gt = blk_start + q;
                    const bool valid  = gt >= eff_warmup;
                    const float x_eff = valid ? p_lb[q] : 0.0F;
                    if (q == 0U) {
                        row_state.dec_partial_sum = x_eff;
                        row_state.dec_partial_raw_sum =
                            valid ? static_cast<double>(x_eff) : 0.0;
                        row_state.dec_block_begin = valid ? gt : (gt + 1ULL);
                        row_state.dec_valid_len   = valid ? 1U : 0U;
                    } else {
                        row_state.dec_partial_sum = ScalarBackend::add_f32(
                            row_state.dec_partial_sum, x_eff);
                        if (valid) {
                            if (row_state.dec_valid_len == 0U) {
                                row_state.dec_block_begin = gt;
                            }
                            row_state.dec_partial_raw_sum +=
                                static_cast<double>(x_eff);
                            ++row_state.dec_valid_len;
                        } else {
                            row_state.dec_partial_raw_sum = 0.0;
                            row_state.dec_valid_len       = 0U;
                        }
                    }
                }
            } else {
                row_state.dec_partial_sum = checkpoint_dec_partial;
                row_state.dec_partial_raw_sum =
                    static_cast<double>(checkpoint_dec_partial);
                row_state.dec_block_begin = tile_start_sample >= phase0
                                                ? (tile_start_sample - phase0)
                                                : 0ULL;
                row_state.dec_valid_len   = phase0;
            }
        }
        row_state.initialized   = true;
        row_state.synced_sample = tile_start_sample;
    }

    for (SizeType t = 0; t < tile_len; ++t) {
        const uint64_t gt = tile_start_sample + t;
        const bool valid  = (t >= first_valid_t) && (gt >= eff_warmup);
        detail::step_surrogate_row_sample(
            gt, valid ? row_samples[t] : 0.0F, valid, np, dec.data(), any_dec4,
            row_biases, row_triggers, row_state, emit_excursion);
    }

    row_state.synced_sample = tile_start_sample + tile_len;
    return row_state.has_triggered_open(np, row_triggers);
}

/// `KadaneSurrogatePlan` convenience overload for `replay_surrogate_row`.
template <typename EmitFn>
inline bool replay_surrogate_row(const KadaneSurrogatePlan& plan,
                                 SizeType row,
                                 uint64_t tile_start_sample,
                                 SizeType tile_len,
                                 const float* row_samples,
                                 KadaneSurrogateRowState& row_state,
                                 EmitFn&& emit_excursion,
                                 SizeType first_valid_t          = 0,
                                 const float* checkpoint_c       = nullptr,
                                 float checkpoint_dec_partial    = 0.0F,
                                 std::span<const float> lookback = {},
                                 uint64_t row_warmup             = 0) {
    if (row >= plan.n_rows || plan.disabled_row[row] != 0U ||
        plan.num_passes == 0) {
        return false;
    }
    std::array<uint32_t, kMaxSurrogatePasses> dec_factors{};
    std::array<float, kMaxSurrogatePasses> biases{};
    std::array<float, kMaxSurrogatePasses> triggers{};
    for (SizeType k = 0; k < plan.num_passes; ++k) {
        dec_factors[k] = plan.passes[k].decimate_factor;
        biases[k]      = plan.row_bias(k, row);
        triggers[k]    = plan.row_trigger(k, row);
    }
    return replay_surrogate_row(
        tile_start_sample, tile_len, row_samples, plan.num_passes,
        dec_factors.data(), biases.data(), triggers.data(), row_state,
        std::forward<EmitFn>(emit_excursion), first_valid_t, checkpoint_c,
        checkpoint_dec_partial, lookback, row_warmup);
}

/// `KadaneSurrogatePlan` convenience overload for `flush_surrogate_row`.
template <typename EmitFn>
inline void flush_surrogate_row(const KadaneSurrogatePlan& plan,
                                SizeType row,
                                KadaneSurrogateRowState& row_state,
                                EmitFn&& emit_excursion) {
    if (row >= plan.n_rows || plan.disabled_row[row] != 0U ||
        plan.num_passes == 0) {
        return;
    }
    std::array<float, kMaxSurrogatePasses> triggers{};
    for (SizeType k = 0; k < plan.num_passes; ++k) {
        triggers[k] = plan.row_trigger(k, row);
    }
    flush_surrogate_row(plan.num_passes, triggers.data(), row_state,
                        std::forward<EmitFn>(emit_excursion));
}

namespace scalar {

template <SizeType K>
[[nodiscard]] inline uint32_t
scan_surrogate_tile_f32(SizeType n_rows,
                        SizeType tile_len,
                        const float* const* row_ptrs,
                        const uint32_t* decimate_factors,
                        const float* bias,
                        SizeType bias_stride,
                        const float* trigger_cut,
                        SizeType cut_stride,
                        float* state_c,
                        SizeType state_stride,
                        uint32_t dec_phase            = 0U,
                        float* dec_partial_sum        = nullptr,
                        const SizeType* first_valid_t = nullptr,
                        float* checkpoint_c_out       = nullptr,
                        SizeType checkpoint_c_stride  = 0,
                        float* checkpoint_dec_out     = nullptr) noexcept {
    return dmt::kadane::scan_surrogate_tile_f32<K, ScalarBackend>(
        n_rows, tile_len, row_ptrs, decimate_factors, bias, bias_stride,
        trigger_cut, cut_stride, state_c, state_stride, dec_phase,
        dec_partial_sum, first_valid_t, checkpoint_c_out, checkpoint_c_stride,
        checkpoint_dec_out);
}

[[nodiscard]] inline uint32_t
scan_surrogate_tile_f32(SizeType num_passes,
                        SizeType n_rows,
                        SizeType tile_len,
                        const float* const* row_ptrs,
                        const uint32_t* decimate_factors,
                        const float* bias,
                        SizeType bias_stride,
                        const float* trigger_cut,
                        SizeType cut_stride,
                        float* state_c,
                        SizeType state_stride,
                        uint32_t dec_phase            = 0U,
                        float* dec_partial_sum        = nullptr,
                        const SizeType* first_valid_t = nullptr,
                        float* checkpoint_c_out       = nullptr,
                        SizeType checkpoint_c_stride  = 0,
                        float* checkpoint_dec_out     = nullptr) noexcept {
    return dmt::kadane::scan_surrogate_tile_f32<ScalarBackend>(
        num_passes, n_rows, tile_len, row_ptrs, decimate_factors, bias,
        bias_stride, trigger_cut, cut_stride, state_c, state_stride, dec_phase,
        dec_partial_sum, first_valid_t, checkpoint_c_out, checkpoint_c_stride,
        checkpoint_dec_out);
}

} // namespace scalar

#endif // !defined(__CUDACC__) && !defined(__HIPCC__)

} // namespace dmt::kadane
