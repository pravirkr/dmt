#pragma once

#if !defined(DMT_KADANE_FORCE_SCALAR) &&                                       \
    (defined(__AVX512F__) || defined(__AVX2__))
#include "dmt/x86_intrinsics.hpp"
#elif !defined(DMT_KADANE_FORCE_SCALAR) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>

#include "dmt/common/types.hpp"

#include "dmt/kadane_plan.hpp"

namespace dmt::kadane {

#if defined(__GNUC__) || defined(__clang__)
#define DMT_KADANE_INLINE inline __attribute__((always_inline))
#else
#define DMT_KADANE_INLINE inline
#endif

inline constexpr SizeType kMaxBatchRows = 32;

/// UB-safe bitmask with the lowest `n` bits set (`0 <= n <= 32`).
[[nodiscard]] DMT_KADANE_INLINE constexpr uint32_t
active_row_mask(SizeType n) noexcept {
    if (n == 0) {
        return 0U;
    }
    if (n >= 32) {
        return 0xFFFFFFFFU;
    }
    return (uint32_t{1} << static_cast<uint32_t>(n)) - 1U;
}

namespace detail {

/// Zero-instruction register barrier preventing GCC `-fassociative-math`
/// (`-ffast-math`) from reassociating `c + (x - b)` or `acc4` sums.
template <typename VecT>
DMT_KADANE_INLINE void strict_fp_barrier(VecT& v) noexcept {
#if defined(__GNUC__) && !defined(__clang__)
#if defined(__aarch64__)
    __asm__("" : "+w"(v));
#elif defined(__AVX512F__)
    __asm__("" : "+v"(v));
#elif defined(__x86_64__)
    __asm__("" : "+x"(v));
#endif
#else
    (void)v;
#endif
}

} // namespace detail

// ============================================================================
// Portable Scalar / 4-Row ILP Backend (always available for parity tests)
// ============================================================================

struct ScalarBackend {
    static constexpr SizeType kSimdLanes = 1;
    static constexpr SizeType kRowGroups = 4;
    static constexpr SizeType kBatchRows = kSimdLanes * kRowGroups;

    [[nodiscard]] static constexpr std::string_view isa_name() noexcept {
        return "scalar";
    }

    using VecF = float;
    using VecI = int32_t;

    [[nodiscard]] static DMT_KADANE_INLINE VecF zero_f32() noexcept {
        return 0.0F;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF set1_f32(float v) noexcept {
        return v;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    loadu_f32(const float* p) noexcept {
        return *p;
    }
    static DMT_KADANE_INLINE void storeu_f32(float* p, VecF v) noexcept {
        *p = v;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF add_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a + b;
#else
        VecF res = a + b;
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF sub_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a - b;
#else
        VecF res = a - b;
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    vmax_nonneg_f32(VecF a, VecF b) noexcept {
        return std::bit_cast<float>(
            std::max(std::bit_cast<int32_t>(a), std::bit_cast<int32_t>(b)));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    maskz_valid_f32(VecF v, VecI t_vec, VecI fv_vec) noexcept {
        return static_cast<uint32_t>(t_vec) >= static_cast<uint32_t>(fv_vec)
                   ? v
                   : 0.0F;
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_f32(VecF a, VecF b) noexcept {
        return a >= b ? 1U : 0U;
    }

    [[nodiscard]] static DMT_KADANE_INLINE VecI zero_i32() noexcept {
        return 0;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI set1_i32(int32_t v) noexcept {
        return v;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    loadu_i32(const int32_t* p) noexcept {
        return *p;
    }
    static DMT_KADANE_INLINE void storeu_i32(int32_t* p, VecI v) noexcept {
        *p = v;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI add_i32(VecI a,
                                                        VecI b) noexcept {
        return a + b;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI sub_i32(VecI a,
                                                        VecI b) noexcept {
        return a - b;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI min_i32(VecI a,
                                                        VecI b) noexcept {
        return std::min(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI max_i32(VecI a,
                                                        VecI b) noexcept {
        return std::max(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    shl_u32(VecI v, uint8_t shift) noexcept {
        return static_cast<int32_t>(static_cast<uint32_t>(v) << shift);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    maskz_valid_i32(VecI v, VecI t_vec, VecI fv_vec) noexcept {
        return static_cast<uint32_t>(t_vec) >= static_cast<uint32_t>(fv_vec)
                   ? v
                   : 0;
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_i32(VecI a, VecI b) noexcept {
        return a >= b ? 1U : 0U;
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    cvt_u32_to_f32(VecI v) noexcept {
        return static_cast<float>(static_cast<uint32_t>(v));
    }

    static DMT_KADANE_INLINE void
    transpose_f32(std::array<VecF, kSimdLanes>& /*r*/) noexcept {}
    static DMT_KADANE_INLINE void
    transpose_i32(std::array<VecI, kSimdLanes>& /*r*/) noexcept {}

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_f32(const float* const* row_ptrs,
                       SizeType t_offset) noexcept {
        return {row_ptrs[0][t_offset]};
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_tail_f32(const float* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        return {valid_cols > 0 ? row_ptrs[0][t_offset] : 0.0F};
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_i32(const int32_t* const* row_ptrs,
                       SizeType t_offset) noexcept {
        return {row_ptrs[0][t_offset]};
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_tail_i32(const int32_t* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        return {valid_cols > 0 ? row_ptrs[0][t_offset] : 0};
    }
};

// ============================================================================
// Pure AVX-512F Backend (16 lanes x 2 groups = 32 rows; zero BW/DQ/VL ops)
// ============================================================================

#if !defined(DMT_KADANE_FORCE_SCALAR) && defined(__AVX512F__)

/// Exact non-negative float32 max via signed-int32 `vpmaxsd`; exact whenever
/// at least one operand is `>= +0.0f` (canonicalizes `-0.0f` to `+0.0f`).
[[nodiscard]] DMT_KADANE_INLINE __m512 vmax_nonneg_f32(__m512 a,
                                                       __m512 b) noexcept {
    return _mm512_castsi512_ps(
        _mm512_max_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

/// Handwritten in-register 16x16 float32 transpose using pure AVX-512F ops.
DMT_KADANE_INLINE void transpose_16x16_f32(std::array<__m512, 16>& r) noexcept {
    __m512 t0 = _mm512_unpacklo_ps(r[0], r[1]);
    __m512 t1 = _mm512_unpackhi_ps(r[0], r[1]);
    __m512 t2 = _mm512_unpacklo_ps(r[2], r[3]);
    __m512 t3 = _mm512_unpackhi_ps(r[2], r[3]);
    __m512 t4 = _mm512_unpacklo_ps(r[4], r[5]);
    __m512 t5 = _mm512_unpackhi_ps(r[4], r[5]);
    __m512 t6 = _mm512_unpacklo_ps(r[6], r[7]);
    __m512 t7 = _mm512_unpackhi_ps(r[6], r[7]);
    __m512 t8 = _mm512_unpacklo_ps(r[8], r[9]);
    __m512 t9 = _mm512_unpackhi_ps(r[8], r[9]);
    __m512 ta = _mm512_unpacklo_ps(r[10], r[11]);
    __m512 tb = _mm512_unpackhi_ps(r[10], r[11]);
    __m512 tc = _mm512_unpacklo_ps(r[12], r[13]);
    __m512 td = _mm512_unpackhi_ps(r[12], r[13]);
    __m512 te = _mm512_unpacklo_ps(r[14], r[15]);
    __m512 tf = _mm512_unpackhi_ps(r[14], r[15]);

    __m512 s0 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 s1 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 s2 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 s3 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 s4 = _mm512_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 s5 = _mm512_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 s6 = _mm512_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 s7 = _mm512_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 s8 = _mm512_shuffle_ps(t8, ta, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 s9 = _mm512_shuffle_ps(t8, ta, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 sa = _mm512_shuffle_ps(t9, tb, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 sb = _mm512_shuffle_ps(t9, tb, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 sc = _mm512_shuffle_ps(tc, te, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 sd = _mm512_shuffle_ps(tc, te, _MM_SHUFFLE(3, 2, 3, 2));
    __m512 se = _mm512_shuffle_ps(td, tf, _MM_SHUFFLE(1, 0, 1, 0));
    __m512 sf = _mm512_shuffle_ps(td, tf, _MM_SHUFFLE(3, 2, 3, 2));

    t0 = _mm512_shuffle_f32x4(s0, s4, _MM_SHUFFLE(2, 0, 2, 0));
    t1 = _mm512_shuffle_f32x4(s8, sc, _MM_SHUFFLE(2, 0, 2, 0));
    t2 = _mm512_shuffle_f32x4(s1, s5, _MM_SHUFFLE(2, 0, 2, 0));
    t3 = _mm512_shuffle_f32x4(s9, sd, _MM_SHUFFLE(2, 0, 2, 0));
    t4 = _mm512_shuffle_f32x4(s2, s6, _MM_SHUFFLE(2, 0, 2, 0));
    t5 = _mm512_shuffle_f32x4(sa, se, _MM_SHUFFLE(2, 0, 2, 0));
    t6 = _mm512_shuffle_f32x4(s3, s7, _MM_SHUFFLE(2, 0, 2, 0));
    t7 = _mm512_shuffle_f32x4(sb, sf, _MM_SHUFFLE(2, 0, 2, 0));
    t8 = _mm512_shuffle_f32x4(s0, s4, _MM_SHUFFLE(3, 1, 3, 1));
    t9 = _mm512_shuffle_f32x4(s8, sc, _MM_SHUFFLE(3, 1, 3, 1));
    ta = _mm512_shuffle_f32x4(s1, s5, _MM_SHUFFLE(3, 1, 3, 1));
    tb = _mm512_shuffle_f32x4(s9, sd, _MM_SHUFFLE(3, 1, 3, 1));
    tc = _mm512_shuffle_f32x4(s2, s6, _MM_SHUFFLE(3, 1, 3, 1));
    td = _mm512_shuffle_f32x4(sa, se, _MM_SHUFFLE(3, 1, 3, 1));
    te = _mm512_shuffle_f32x4(s3, s7, _MM_SHUFFLE(3, 1, 3, 1));
    tf = _mm512_shuffle_f32x4(sb, sf, _MM_SHUFFLE(3, 1, 3, 1));

    r[0]  = _mm512_shuffle_f32x4(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));
    r[1]  = _mm512_shuffle_f32x4(t2, t3, _MM_SHUFFLE(2, 0, 2, 0));
    r[2]  = _mm512_shuffle_f32x4(t4, t5, _MM_SHUFFLE(2, 0, 2, 0));
    r[3]  = _mm512_shuffle_f32x4(t6, t7, _MM_SHUFFLE(2, 0, 2, 0));
    r[4]  = _mm512_shuffle_f32x4(t8, t9, _MM_SHUFFLE(2, 0, 2, 0));
    r[5]  = _mm512_shuffle_f32x4(ta, tb, _MM_SHUFFLE(2, 0, 2, 0));
    r[6]  = _mm512_shuffle_f32x4(tc, td, _MM_SHUFFLE(2, 0, 2, 0));
    r[7]  = _mm512_shuffle_f32x4(te, tf, _MM_SHUFFLE(2, 0, 2, 0));
    r[8]  = _mm512_shuffle_f32x4(t0, t1, _MM_SHUFFLE(3, 1, 3, 1));
    r[9]  = _mm512_shuffle_f32x4(t2, t3, _MM_SHUFFLE(3, 1, 3, 1));
    r[10] = _mm512_shuffle_f32x4(t4, t5, _MM_SHUFFLE(3, 1, 3, 1));
    r[11] = _mm512_shuffle_f32x4(t6, t7, _MM_SHUFFLE(3, 1, 3, 1));
    r[12] = _mm512_shuffle_f32x4(t8, t9, _MM_SHUFFLE(3, 1, 3, 1));
    r[13] = _mm512_shuffle_f32x4(ta, tb, _MM_SHUFFLE(3, 1, 3, 1));
    r[14] = _mm512_shuffle_f32x4(tc, td, _MM_SHUFFLE(3, 1, 3, 1));
    r[15] = _mm512_shuffle_f32x4(te, tf, _MM_SHUFFLE(3, 1, 3, 1));
}

/// Handwritten in-register 16x16 int32 transpose using pure AVX-512F ops.
DMT_KADANE_INLINE void
transpose_16x16_i32(std::array<__m512i, 16>& r) noexcept {
    __m512i t0 = _mm512_unpacklo_epi32(r[0], r[1]);
    __m512i t1 = _mm512_unpackhi_epi32(r[0], r[1]);
    __m512i t2 = _mm512_unpacklo_epi32(r[2], r[3]);
    __m512i t3 = _mm512_unpackhi_epi32(r[2], r[3]);
    __m512i t4 = _mm512_unpacklo_epi32(r[4], r[5]);
    __m512i t5 = _mm512_unpackhi_epi32(r[4], r[5]);
    __m512i t6 = _mm512_unpacklo_epi32(r[6], r[7]);
    __m512i t7 = _mm512_unpackhi_epi32(r[6], r[7]);
    __m512i t8 = _mm512_unpacklo_epi32(r[8], r[9]);
    __m512i t9 = _mm512_unpackhi_epi32(r[8], r[9]);
    __m512i ta = _mm512_unpacklo_epi32(r[10], r[11]);
    __m512i tb = _mm512_unpackhi_epi32(r[10], r[11]);
    __m512i tc = _mm512_unpacklo_epi32(r[12], r[13]);
    __m512i td = _mm512_unpackhi_epi32(r[12], r[13]);
    __m512i te = _mm512_unpacklo_epi32(r[14], r[15]);
    __m512i tf = _mm512_unpackhi_epi32(r[14], r[15]);

    __m512i s0 = _mm512_unpacklo_epi64(t0, t2);
    __m512i s1 = _mm512_unpackhi_epi64(t0, t2);
    __m512i s2 = _mm512_unpacklo_epi64(t1, t3);
    __m512i s3 = _mm512_unpackhi_epi64(t1, t3);
    __m512i s4 = _mm512_unpacklo_epi64(t4, t6);
    __m512i s5 = _mm512_unpackhi_epi64(t4, t6);
    __m512i s6 = _mm512_unpacklo_epi64(t5, t7);
    __m512i s7 = _mm512_unpackhi_epi64(t5, t7);
    __m512i s8 = _mm512_unpacklo_epi64(t8, ta);
    __m512i s9 = _mm512_unpackhi_epi64(t8, ta);
    __m512i sa = _mm512_unpacklo_epi64(t9, tb);
    __m512i sb = _mm512_unpackhi_epi64(t9, tb);
    __m512i sc = _mm512_unpacklo_epi64(tc, te);
    __m512i sd = _mm512_unpackhi_epi64(tc, te);
    __m512i se = _mm512_unpacklo_epi64(td, tf);
    __m512i sf = _mm512_unpackhi_epi64(td, tf);

    t0 = _mm512_shuffle_i32x4(s0, s4, _MM_SHUFFLE(2, 0, 2, 0));
    t1 = _mm512_shuffle_i32x4(s8, sc, _MM_SHUFFLE(2, 0, 2, 0));
    t2 = _mm512_shuffle_i32x4(s1, s5, _MM_SHUFFLE(2, 0, 2, 0));
    t3 = _mm512_shuffle_i32x4(s9, sd, _MM_SHUFFLE(2, 0, 2, 0));
    t4 = _mm512_shuffle_i32x4(s2, s6, _MM_SHUFFLE(2, 0, 2, 0));
    t5 = _mm512_shuffle_i32x4(sa, se, _MM_SHUFFLE(2, 0, 2, 0));
    t6 = _mm512_shuffle_i32x4(s3, s7, _MM_SHUFFLE(2, 0, 2, 0));
    t7 = _mm512_shuffle_i32x4(sb, sf, _MM_SHUFFLE(2, 0, 2, 0));
    t8 = _mm512_shuffle_i32x4(s0, s4, _MM_SHUFFLE(3, 1, 3, 1));
    t9 = _mm512_shuffle_i32x4(s8, sc, _MM_SHUFFLE(3, 1, 3, 1));
    ta = _mm512_shuffle_i32x4(s1, s5, _MM_SHUFFLE(3, 1, 3, 1));
    tb = _mm512_shuffle_i32x4(s9, sd, _MM_SHUFFLE(3, 1, 3, 1));
    tc = _mm512_shuffle_i32x4(s2, s6, _MM_SHUFFLE(3, 1, 3, 1));
    td = _mm512_shuffle_i32x4(sa, se, _MM_SHUFFLE(3, 1, 3, 1));
    te = _mm512_shuffle_i32x4(s3, s7, _MM_SHUFFLE(3, 1, 3, 1));
    tf = _mm512_shuffle_i32x4(sb, sf, _MM_SHUFFLE(3, 1, 3, 1));

    r[0]  = _mm512_shuffle_i32x4(t0, t1, _MM_SHUFFLE(2, 0, 2, 0));
    r[1]  = _mm512_shuffle_i32x4(t2, t3, _MM_SHUFFLE(2, 0, 2, 0));
    r[2]  = _mm512_shuffle_i32x4(t4, t5, _MM_SHUFFLE(2, 0, 2, 0));
    r[3]  = _mm512_shuffle_i32x4(t6, t7, _MM_SHUFFLE(2, 0, 2, 0));
    r[4]  = _mm512_shuffle_i32x4(t8, t9, _MM_SHUFFLE(2, 0, 2, 0));
    r[5]  = _mm512_shuffle_i32x4(ta, tb, _MM_SHUFFLE(2, 0, 2, 0));
    r[6]  = _mm512_shuffle_i32x4(tc, td, _MM_SHUFFLE(2, 0, 2, 0));
    r[7]  = _mm512_shuffle_i32x4(te, tf, _MM_SHUFFLE(2, 0, 2, 0));
    r[8]  = _mm512_shuffle_i32x4(t0, t1, _MM_SHUFFLE(3, 1, 3, 1));
    r[9]  = _mm512_shuffle_i32x4(t2, t3, _MM_SHUFFLE(3, 1, 3, 1));
    r[10] = _mm512_shuffle_i32x4(t4, t5, _MM_SHUFFLE(3, 1, 3, 1));
    r[11] = _mm512_shuffle_i32x4(t6, t7, _MM_SHUFFLE(3, 1, 3, 1));
    r[12] = _mm512_shuffle_i32x4(t8, t9, _MM_SHUFFLE(3, 1, 3, 1));
    r[13] = _mm512_shuffle_i32x4(ta, tb, _MM_SHUFFLE(3, 1, 3, 1));
    r[14] = _mm512_shuffle_i32x4(tc, td, _MM_SHUFFLE(3, 1, 3, 1));
    r[15] = _mm512_shuffle_i32x4(te, tf, _MM_SHUFFLE(3, 1, 3, 1));
}

struct Avx512fBackend {
    static constexpr SizeType kSimdLanes = 16;
    static constexpr SizeType kRowGroups = 2;
    static constexpr SizeType kBatchRows = kSimdLanes * kRowGroups;

    [[nodiscard]] static constexpr std::string_view isa_name() noexcept {
        return "avx512f";
    }

    using VecF = __m512;
    using VecI = __m512i;

    [[nodiscard]] static DMT_KADANE_INLINE VecF zero_f32() noexcept {
        return _mm512_setzero_ps();
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF set1_f32(float v) noexcept {
        return _mm512_set1_ps(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    loadu_f32(const float* p) noexcept {
        return _mm512_loadu_ps(p);
    }
    static DMT_KADANE_INLINE void storeu_f32(float* p, VecF v) noexcept {
        _mm512_storeu_ps(p, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF add_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a + b;
#else
        VecF res = _mm512_add_ps(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF sub_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a - b;
#else
        VecF res = _mm512_sub_ps(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    vmax_nonneg_f32(VecF a, VecF b) noexcept {
        return dmt::kadane::vmax_nonneg_f32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    maskz_valid_f32(VecF v, VecI t_vec, VecI fv_vec) noexcept {
        const __mmask16 valid = _mm512_cmpge_epu32_mask(t_vec, fv_vec);
        return _mm512_maskz_mov_ps(valid, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_f32(VecF a, VecF b) noexcept {
        return static_cast<uint32_t>(_mm512_cmp_ps_mask(a, b, _CMP_GE_OQ));
    }

    [[nodiscard]] static DMT_KADANE_INLINE VecI zero_i32() noexcept {
        return _mm512_setzero_si512();
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI set1_i32(int32_t v) noexcept {
        return _mm512_set1_epi32(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    loadu_i32(const int32_t* p) noexcept {
        return _mm512_loadu_si512(p);
    }
    static DMT_KADANE_INLINE void storeu_i32(int32_t* p, VecI v) noexcept {
        _mm512_storeu_si512(p, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI add_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm512_add_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI sub_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm512_sub_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI min_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm512_min_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI max_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm512_max_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    shl_u32(VecI v, uint8_t shift) noexcept {
        return _mm512_sll_epi32(v, _mm_cvtsi32_si128(static_cast<int>(shift)));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    maskz_valid_i32(VecI v, VecI t_vec, VecI fv_vec) noexcept {
        const __mmask16 valid = _mm512_cmpge_epu32_mask(t_vec, fv_vec);
        return _mm512_maskz_mov_epi32(valid, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_i32(VecI a, VecI b) noexcept {
        return static_cast<uint32_t>(_mm512_cmpge_epi32_mask(a, b));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    cvt_u32_to_f32(VecI v) noexcept {
        return _mm512_cvtepu32_ps(v);
    }

    static DMT_KADANE_INLINE void
    transpose_f32(std::array<VecF, kSimdLanes>& r) noexcept {
        transpose_16x16_f32(r);
    }
    static DMT_KADANE_INLINE void
    transpose_i32(std::array<VecI, kSimdLanes>& r) noexcept {
        transpose_16x16_i32(r);
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_f32(const float* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecF, kSimdLanes> r{};
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm512_loadu_ps(row_ptrs[i] + t_offset);
        }
        transpose_16x16_f32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_tail_f32(const float* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecF, kSimdLanes> r{};
        if (valid_cols == 0) {
            for (SizeType i = 0; i < kSimdLanes; ++i) {
                r[i] = _mm512_setzero_ps();
            }
            return r;
        }
        const auto mask = static_cast<__mmask16>(
            valid_cols >= kSimdLanes ? 0xFFFFU : ((1U << valid_cols) - 1U));
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm512_maskz_loadu_ps(mask, row_ptrs[i] + t_offset);
        }
        transpose_16x16_f32(r);
        return r;
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_i32(const int32_t* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecI, kSimdLanes> r{};
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm512_loadu_si512(row_ptrs[i] + t_offset);
        }
        transpose_16x16_i32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_tail_i32(const int32_t* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecI, kSimdLanes> r{};
        if (valid_cols == 0) {
            for (SizeType i = 0; i < kSimdLanes; ++i) {
                r[i] = _mm512_setzero_si512();
            }
            return r;
        }
        const auto mask = static_cast<__mmask16>(
            valid_cols >= kSimdLanes ? 0xFFFFU : ((1U << valid_cols) - 1U));
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm512_maskz_loadu_epi32(mask, row_ptrs[i] + t_offset);
        }
        transpose_16x16_i32(r);
        return r;
    }
};

using NativeBackend = Avx512fBackend;

// ============================================================================
// AVX2 Backend (8 lanes x 2 groups = 16 rows)
// ============================================================================

#elif !defined(DMT_KADANE_FORCE_SCALAR) && defined(__AVX2__)

/// Exact non-negative float32 max via signed-int32 `vpmaxsd`.
[[nodiscard]] DMT_KADANE_INLINE __m256 vmax_nonneg_f32(__m256 a,
                                                       __m256 b) noexcept {
    return _mm256_castsi256_ps(
        _mm256_max_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b)));
}

/// Handwritten in-register 8x8 float32 transpose.
DMT_KADANE_INLINE void transpose_8x8_f32(std::array<__m256, 8>& r) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]);
    const __m256 t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]);
    const __m256 t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]);
    const __m256 t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]);
    const __m256 t7 = _mm256_unpackhi_ps(r[6], r[7]);

    const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    r[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    r[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    r[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    r[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    r[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    r[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    r[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    r[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

/// Handwritten in-register 8x8 int32 transpose.
DMT_KADANE_INLINE void transpose_8x8_i32(std::array<__m256i, 8>& r) noexcept {
    const __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]);
    const __m256i t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]);
    const __m256i t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]);
    const __m256i t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]);
    const __m256i t7 = _mm256_unpackhi_epi32(r[6], r[7]);

    const __m256i s0 = _mm256_unpacklo_epi64(t0, t2);
    const __m256i s1 = _mm256_unpackhi_epi64(t0, t2);
    const __m256i s2 = _mm256_unpacklo_epi64(t1, t3);
    const __m256i s3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i s4 = _mm256_unpacklo_epi64(t4, t6);
    const __m256i s5 = _mm256_unpackhi_epi64(t4, t6);
    const __m256i s6 = _mm256_unpacklo_epi64(t5, t7);
    const __m256i s7 = _mm256_unpackhi_epi64(t5, t7);

    r[0] = _mm256_permute2x128_si256(s0, s4, 0x20);
    r[1] = _mm256_permute2x128_si256(s1, s5, 0x20);
    r[2] = _mm256_permute2x128_si256(s2, s6, 0x20);
    r[3] = _mm256_permute2x128_si256(s3, s7, 0x20);
    r[4] = _mm256_permute2x128_si256(s0, s4, 0x31);
    r[5] = _mm256_permute2x128_si256(s1, s5, 0x31);
    r[6] = _mm256_permute2x128_si256(s2, s6, 0x31);
    r[7] = _mm256_permute2x128_si256(s3, s7, 0x31);
}

struct Avx2Backend {
    static constexpr SizeType kSimdLanes = 8;
    static constexpr SizeType kRowGroups = 2;
    static constexpr SizeType kBatchRows = kSimdLanes * kRowGroups;

    [[nodiscard]] static constexpr std::string_view isa_name() noexcept {
        return "avx2";
    }

    using VecF = __m256;
    using VecI = __m256i;

    [[nodiscard]] static DMT_KADANE_INLINE VecF zero_f32() noexcept {
        return _mm256_setzero_ps();
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF set1_f32(float v) noexcept {
        return _mm256_set1_ps(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    loadu_f32(const float* p) noexcept {
        return _mm256_loadu_ps(p);
    }
    static DMT_KADANE_INLINE void storeu_f32(float* p, VecF v) noexcept {
        _mm256_storeu_ps(p, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF add_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a + b;
#else
        VecF res = _mm256_add_ps(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF sub_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a - b;
#else
        VecF res = _mm256_sub_ps(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    vmax_nonneg_f32(VecF a, VecF b) noexcept {
        return dmt::kadane::vmax_nonneg_f32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    maskz_valid_f32(VecF v, VecI t_vec, VecI fv_vec) noexcept {
        const __m256i invalid = _mm256_cmpgt_epi32(fv_vec, t_vec);
        return _mm256_andnot_ps(_mm256_castsi256_ps(invalid), v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_f32(VecF a, VecF b) noexcept {
        return static_cast<uint32_t>(
            _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_GE_OQ)));
    }

    [[nodiscard]] static DMT_KADANE_INLINE VecI zero_i32() noexcept {
        return _mm256_setzero_si256();
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI set1_i32(int32_t v) noexcept {
        return _mm256_set1_epi32(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    loadu_i32(const int32_t* p) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    static DMT_KADANE_INLINE void storeu_i32(int32_t* p, VecI v) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI add_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm256_add_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI sub_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm256_sub_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI min_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm256_min_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI max_i32(VecI a,
                                                        VecI b) noexcept {
        return _mm256_max_epi32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    shl_u32(VecI v, uint8_t shift) noexcept {
        return _mm256_sll_epi32(v, _mm_cvtsi32_si128(static_cast<int>(shift)));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    maskz_valid_i32(VecI v, VecI t_vec, VecI fv_vec) noexcept {
        const __m256i invalid = _mm256_cmpgt_epi32(fv_vec, t_vec);
        return _mm256_andnot_si256(invalid, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_i32(VecI a, VecI b) noexcept {
        const __m256i ge = _mm256_cmpeq_epi32(_mm256_max_epi32(a, b), a);
        return static_cast<uint32_t>(
            _mm256_movemask_ps(_mm256_castsi256_ps(ge)));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    cvt_u32_to_f32(VecI v) noexcept {
        const __m256 pos = _mm256_cvtepi32_ps(v);
        const __m256i q  = _mm256_or_si256(
            _mm256_srli_epi32(v, 1), _mm256_and_si256(v, _mm256_set1_epi32(1)));
        const __m256 half = _mm256_cvtepi32_ps(q);
        const __m256 neg  = _mm256_add_ps(half, half);
        return _mm256_blendv_ps(pos, neg, _mm256_castsi256_ps(v));
    }

    static DMT_KADANE_INLINE void
    transpose_f32(std::array<VecF, kSimdLanes>& r) noexcept {
        transpose_8x8_f32(r);
    }
    static DMT_KADANE_INLINE void
    transpose_i32(std::array<VecI, kSimdLanes>& r) noexcept {
        transpose_8x8_i32(r);
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_f32(const float* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecF, kSimdLanes> r{};
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm256_loadu_ps(row_ptrs[i] + t_offset);
        }
        transpose_8x8_f32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_tail_f32(const float* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecF, kSimdLanes> r{};
        if (valid_cols == 0) {
            for (SizeType i = 0; i < kSimdLanes; ++i) {
                r[i] = _mm256_setzero_ps();
            }
            return r;
        }
        const __m256i mask =
            _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(valid_cols)),
                               _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm256_maskload_ps(row_ptrs[i] + t_offset, mask);
        }
        transpose_8x8_f32(r);
        return r;
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_i32(const int32_t* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecI, kSimdLanes> r{};
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(row_ptrs[i] + t_offset));
        }
        transpose_8x8_i32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_tail_i32(const int32_t* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecI, kSimdLanes> r{};
        if (valid_cols == 0) {
            for (SizeType i = 0; i < kSimdLanes; ++i) {
                r[i] = _mm256_setzero_si256();
            }
            return r;
        }
        const __m256i mask =
            _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(valid_cols)),
                               _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        for (SizeType i = 0; i < kSimdLanes; ++i) {
            r[i] = _mm256_maskload_epi32(
                reinterpret_cast<const int*>(row_ptrs[i] + t_offset), mask);
        }
        transpose_8x8_i32(r);
        return r;
    }
};

using NativeBackend = Avx2Backend;

// ============================================================================
// ARM NEON Backend (4 lanes x 4 groups = 16 rows)
// ============================================================================

#elif !defined(DMT_KADANE_FORCE_SCALAR) && defined(__ARM_NEON)

/// Exact non-negative float32 max via signed-int32 `vmaxq_s32`.
[[nodiscard]] DMT_KADANE_INLINE float32x4_t
vmax_nonneg_f32(float32x4_t a, float32x4_t b) noexcept {
    return vreinterpretq_f32_s32(
        vmaxq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

/// Handwritten in-register 4x4 float32 transpose.
DMT_KADANE_INLINE void
transpose_4x4_f32(std::array<float32x4_t, 4>& r) noexcept {
    const float32x4x2_t t01 = vtrnq_f32(r[0], r[1]);
    const float32x4x2_t t23 = vtrnq_f32(r[2], r[3]);
    r[0] = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    r[1] = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    r[2] = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    r[3] = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
}

/// Handwritten in-register 4x4 int32 transpose.
DMT_KADANE_INLINE void transpose_4x4_i32(std::array<int32x4_t, 4>& r) noexcept {
    const int32x4x2_t t01 = vtrnq_s32(r[0], r[1]);
    const int32x4x2_t t23 = vtrnq_s32(r[2], r[3]);
    r[0] = vcombine_s32(vget_low_s32(t01.val[0]), vget_low_s32(t23.val[0]));
    r[1] = vcombine_s32(vget_low_s32(t01.val[1]), vget_low_s32(t23.val[1]));
    r[2] = vcombine_s32(vget_high_s32(t01.val[0]), vget_high_s32(t23.val[0]));
    r[3] = vcombine_s32(vget_high_s32(t01.val[1]), vget_high_s32(t23.val[1]));
}

namespace detail {

[[nodiscard]] DMT_KADANE_INLINE float32x4_t
neon_load_tail_f32(const float* p, SizeType valid_cols) noexcept {
    if (valid_cols == 0) {
        return vdupq_n_f32(0.0F);
    }
    if (valid_cols == 1) {
        float32x4_t v = vdupq_n_f32(0.0F);
        return vld1q_lane_f32(p, v, 0);
    }
    if (valid_cols == 2) {
        return vcombine_f32(vld1_f32(p), vdup_n_f32(0.0F));
    }
    if (valid_cols == 3) {
        const float32x2_t lo = vld1_f32(p);
        float32x2_t hi       = vdup_n_f32(0.0F);
        hi                   = vld1_lane_f32(p + 2, hi, 0);
        return vcombine_f32(lo, hi);
    }
    return vld1q_f32(p);
}

[[nodiscard]] DMT_KADANE_INLINE int32x4_t
neon_load_tail_i32(const int32_t* p, SizeType valid_cols) noexcept {
    if (valid_cols == 0) {
        return vdupq_n_s32(0);
    }
    if (valid_cols == 1) {
        int32x4_t v = vdupq_n_s32(0);
        return vld1q_lane_s32(p, v, 0);
    }
    if (valid_cols == 2) {
        return vcombine_s32(vld1_s32(p), vdup_n_s32(0));
    }
    if (valid_cols == 3) {
        const int32x2_t lo = vld1_s32(p);
        int32x2_t hi       = vdup_n_s32(0);
        hi                 = vld1_lane_s32(p + 2, hi, 0);
        return vcombine_s32(lo, hi);
    }
    return vld1q_s32(p);
}

[[nodiscard]] DMT_KADANE_INLINE uint32_t
neon_movemask_u32(uint32x4_t cmp) noexcept {
    alignas(16) static constexpr std::array<uint32_t, 4> kBitWeights = {1U, 2U,
                                                                        4U, 8U};
    const uint32x4_t bits = vandq_u32(cmp, vld1q_u32(kBitWeights.data()));
#if defined(__aarch64__)
    return vaddvq_u32(bits);
#else
    const uint32x2_t s2 = vpadd_u32(vget_low_u32(bits), vget_high_u32(bits));
    return vget_lane_u32(vpadd_u32(s2, s2), 0);
#endif
}

} // namespace detail

struct NeonBackend {
    static constexpr SizeType kSimdLanes = 4;
    static constexpr SizeType kRowGroups = 4;
    static constexpr SizeType kBatchRows = kSimdLanes * kRowGroups;

    [[nodiscard]] static constexpr std::string_view isa_name() noexcept {
        return "neon";
    }

    using VecF = float32x4_t;
    using VecI = int32x4_t;

    [[nodiscard]] static DMT_KADANE_INLINE VecF zero_f32() noexcept {
        return vdupq_n_f32(0.0F);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF set1_f32(float v) noexcept {
        return vdupq_n_f32(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    loadu_f32(const float* p) noexcept {
        return vld1q_f32(p);
    }
    static DMT_KADANE_INLINE void storeu_f32(float* p, VecF v) noexcept {
        vst1q_f32(p, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF add_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a + b;
#else
        VecF res = vaddq_f32(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF sub_f32(VecF a,
                                                        VecF b) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off) reassociate(off)
        return a - b;
#else
        VecF res = vsubq_f32(a, b);
        detail::strict_fp_barrier(res);
        return res;
#endif
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    vmax_nonneg_f32(VecF a, VecF b) noexcept {
        return dmt::kadane::vmax_nonneg_f32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    maskz_valid_f32(VecF v, VecI t_vec, VecI fv_vec) noexcept {
        const uint32x4_t valid = vcgeq_u32(vreinterpretq_u32_s32(t_vec),
                                           vreinterpretq_u32_s32(fv_vec));
        return vreinterpretq_f32_u32(
            vandq_u32(vreinterpretq_u32_f32(v), valid));
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_f32(VecF a, VecF b) noexcept {
        return detail::neon_movemask_u32(vcgeq_f32(a, b));
    }

    [[nodiscard]] static DMT_KADANE_INLINE VecI zero_i32() noexcept {
        return vdupq_n_s32(0);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI set1_i32(int32_t v) noexcept {
        return vdupq_n_s32(v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    loadu_i32(const int32_t* p) noexcept {
        return vld1q_s32(p);
    }
    static DMT_KADANE_INLINE void storeu_i32(int32_t* p, VecI v) noexcept {
        vst1q_s32(p, v);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI add_i32(VecI a,
                                                        VecI b) noexcept {
        return vaddq_s32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI sub_i32(VecI a,
                                                        VecI b) noexcept {
        return vsubq_s32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI min_i32(VecI a,
                                                        VecI b) noexcept {
        return vminq_s32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI max_i32(VecI a,
                                                        VecI b) noexcept {
        return vmaxq_s32(a, b);
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    shl_u32(VecI v, uint8_t shift) noexcept {
        return vreinterpretq_s32_u32(
            vshlq_u32(vreinterpretq_u32_s32(v),
                      vdupq_n_s32(static_cast<int32_t>(shift))));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecI
    maskz_valid_i32(VecI v, VecI t_vec, VecI fv_vec) noexcept {
        const uint32x4_t valid = vcgeq_u32(vreinterpretq_u32_s32(t_vec),
                                           vreinterpretq_u32_s32(fv_vec));
        return vreinterpretq_s32_u32(
            vandq_u32(vreinterpretq_u32_s32(v), valid));
    }
    [[nodiscard]] static DMT_KADANE_INLINE uint32_t
    cmpge_mask_i32(VecI a, VecI b) noexcept {
        return detail::neon_movemask_u32(vcgeq_s32(a, b));
    }
    [[nodiscard]] static DMT_KADANE_INLINE VecF
    cvt_u32_to_f32(VecI v) noexcept {
        return vcvtq_f32_u32(vreinterpretq_u32_s32(v));
    }

    static DMT_KADANE_INLINE void
    transpose_f32(std::array<VecF, kSimdLanes>& r) noexcept {
        transpose_4x4_f32(r);
    }
    static DMT_KADANE_INLINE void
    transpose_i32(std::array<VecI, kSimdLanes>& r) noexcept {
        transpose_4x4_i32(r);
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_f32(const float* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecF, kSimdLanes> r{
            vld1q_f32(row_ptrs[0] + t_offset),
            vld1q_f32(row_ptrs[1] + t_offset),
            vld1q_f32(row_ptrs[2] + t_offset),
            vld1q_f32(row_ptrs[3] + t_offset),
        };
        transpose_4x4_f32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecF, kSimdLanes>
    load_transpose_tail_f32(const float* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecF, kSimdLanes> r{
            detail::neon_load_tail_f32(row_ptrs[0] + t_offset, valid_cols),
            detail::neon_load_tail_f32(row_ptrs[1] + t_offset, valid_cols),
            detail::neon_load_tail_f32(row_ptrs[2] + t_offset, valid_cols),
            detail::neon_load_tail_f32(row_ptrs[3] + t_offset, valid_cols),
        };
        transpose_4x4_f32(r);
        return r;
    }

    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_i32(const int32_t* const* row_ptrs,
                       SizeType t_offset) noexcept {
        std::array<VecI, kSimdLanes> r{
            vld1q_s32(row_ptrs[0] + t_offset),
            vld1q_s32(row_ptrs[1] + t_offset),
            vld1q_s32(row_ptrs[2] + t_offset),
            vld1q_s32(row_ptrs[3] + t_offset),
        };
        transpose_4x4_i32(r);
        return r;
    }
    [[nodiscard]] static DMT_KADANE_INLINE std::array<VecI, kSimdLanes>
    load_transpose_tail_i32(const int32_t* const* row_ptrs,
                            SizeType t_offset,
                            SizeType valid_cols) noexcept {
        std::array<VecI, kSimdLanes> r{
            detail::neon_load_tail_i32(row_ptrs[0] + t_offset, valid_cols),
            detail::neon_load_tail_i32(row_ptrs[1] + t_offset, valid_cols),
            detail::neon_load_tail_i32(row_ptrs[2] + t_offset, valid_cols),
            detail::neon_load_tail_i32(row_ptrs[3] + t_offset, valid_cols),
        };
        transpose_4x4_i32(r);
        return r;
    }
};

using NativeBackend = NeonBackend;

#else

using NativeBackend = ScalarBackend;

#endif

inline constexpr SizeType kSimdLanes = NativeBackend::kSimdLanes;
inline constexpr SizeType kRowGroups = NativeBackend::kRowGroups;
inline constexpr SizeType kBatchRows = NativeBackend::kBatchRows;

[[nodiscard]] constexpr std::string_view active_isa_name() noexcept {
    return NativeBackend::isa_name();
}

/// Load an `L x valid_cols` (`0 <= valid_cols <= L`, `L = Backend::kSimdLanes`)
/// row-major block from `row_ptrs[0..L-1]` at `t_offset`, transpose in
/// registers, and store `[L][L]` sample-major vectors to `dst_sample_major`.
template <typename Backend = NativeBackend>
DMT_KADANE_INLINE void transpose_block_f32(const float* const* row_ptrs,
                                           SizeType t_offset,
                                           SizeType valid_cols,
                                           float* dst_sample_major) noexcept {
    constexpr SizeType kLanes = Backend::kSimdLanes;
    const auto cols =
        valid_cols >= kLanes
            ? Backend::load_transpose_f32(row_ptrs, t_offset)
            : Backend::load_transpose_tail_f32(row_ptrs, t_offset, valid_cols);
    for (SizeType s = 0; s < kLanes; ++s) {
        Backend::storeu_f32(dst_sample_major + (s * kLanes), cols[s]);
    }
}

/// Load an `L x valid_cols` (`0 <= valid_cols <= L`, `L = Backend::kSimdLanes`)
/// row-major int32 block from `row_ptrs[0..L-1]` at `t_offset`, transpose in
/// registers, and store `[L][L]` sample-major vectors to `dst_sample_major`.
template <typename Backend = NativeBackend>
DMT_KADANE_INLINE void transpose_block_i32(const int32_t* const* row_ptrs,
                                           SizeType t_offset,
                                           SizeType valid_cols,
                                           int32_t* dst_sample_major) noexcept {
    constexpr SizeType kLanes = Backend::kSimdLanes;
    const auto cols =
        valid_cols >= kLanes
            ? Backend::load_transpose_i32(row_ptrs, t_offset)
            : Backend::load_transpose_tail_i32(row_ptrs, t_offset, valid_cols);
    for (SizeType s = 0; s < kLanes; ++s) {
        Backend::storeu_i32(dst_sample_major + (s * kLanes), cols[s]);
    }
}

/// Vectorized unsigned 32-bit integer to IEEE-754 float32 conversion.
template <typename Backend = NativeBackend>
inline void convert_u32_to_f32(const uint32_t* __restrict__ src,
                               float* __restrict__ dst,
                               SizeType count) noexcept {
    constexpr SizeType kLanes = Backend::kSimdLanes;
    SizeType i                = 0;
    for (; i + kLanes <= count; i += kLanes) {
        const auto vi =
            Backend::loadu_i32(reinterpret_cast<const int32_t*>(src + i));
        Backend::storeu_f32(dst + i, Backend::cvt_u32_to_f32(vi));
    }
    for (; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

namespace detail {

template <typename Backend, typename T>
DMT_KADANE_INLINE void
load_batch_param(const T* src,
                 SizeType stride,
                 SizeType k,
                 SizeType n_rows,
                 T pad_val,
                 std::array<std::conditional_t<std::is_same_v<T, float>,
                                               typename Backend::VecF,
                                               typename Backend::VecI>,
                            Backend::kRowGroups>& out) noexcept {
    constexpr SizeType kLanes  = Backend::kSimdLanes;
    constexpr SizeType kGroups = Backend::kRowGroups;
    constexpr SizeType kBatch  = Backend::kBatchRows;
    const T* row_src           = src + (k * stride);
    if (n_rows == kBatch) {
        for (SizeType g = 0; g < kGroups; ++g) {
            if constexpr (std::is_same_v<T, float>) {
                out[g] = Backend::loadu_f32(row_src + (g * kLanes));
            } else {
                out[g] = Backend::loadu_i32(row_src + (g * kLanes));
            }
        }
        return;
    }
    alignas(64) std::array<T, kBatch> tmp{};
    tmp.fill(pad_val);
    std::copy_n(row_src, n_rows, tmp.data());
    for (SizeType g = 0; g < kGroups; ++g) {
        if constexpr (std::is_same_v<T, float>) {
            out[g] = Backend::loadu_f32(tmp.data() + (g * kLanes));
        } else {
            out[g] = Backend::loadu_i32(tmp.data() + (g * kLanes));
        }
    }
}

template <typename Backend, typename T>
DMT_KADANE_INLINE void
store_batch_param(const std::array<std::conditional_t<std::is_same_v<T, float>,
                                                      typename Backend::VecF,
                                                      typename Backend::VecI>,
                                   Backend::kRowGroups>& in,
                  T* dst,
                  SizeType stride,
                  SizeType k,
                  SizeType n_rows) noexcept {
    constexpr SizeType kLanes  = Backend::kSimdLanes;
    constexpr SizeType kGroups = Backend::kRowGroups;
    constexpr SizeType kBatch  = Backend::kBatchRows;
    T* row_dst                 = dst + (k * stride);
    if (n_rows == kBatch) {
        for (SizeType g = 0; g < kGroups; ++g) {
            if constexpr (std::is_same_v<T, float>) {
                Backend::storeu_f32(row_dst + (g * kLanes), in[g]);
            } else {
                Backend::storeu_i32(row_dst + (g * kLanes), in[g]);
            }
        }
        return;
    }
    alignas(64) std::array<T, kBatch> tmp{};
    for (SizeType g = 0; g < kGroups; ++g) {
        if constexpr (std::is_same_v<T, float>) {
            Backend::storeu_f32(tmp.data() + (g * kLanes), in[g]);
        } else {
            Backend::storeu_i32(tmp.data() + (g * kLanes), in[g]);
        }
    }
    std::copy_n(tmp.data(), n_rows, row_dst);
}

template <SizeType K, typename Backend>
[[nodiscard]] inline uint32_t
scan_gate_batch_f32(SizeType n_rows,
                    SizeType tile_len,
                    const float* const* row_ptrs,
                    const float* bias,
                    SizeType bias_stride,
                    const float* cut,
                    SizeType cut_stride,
                    float* state_c,
                    SizeType state_stride,
                    const SizeType* first_valid_t,
                    const float* lag_max_in,
                    SizeType lag_stride,
                    float* tile_max_out,
                    SizeType max_out_stride) noexcept {
    static_assert(K >= 1 && K <= kMaxGateBuckets);
    constexpr SizeType kLanes  = Backend::kSimdLanes;
    constexpr SizeType kGroups = Backend::kRowGroups;
    constexpr SizeType kBatch  = Backend::kBatchRows;
    using VecF                 = Backend::VecF;
    using VecI                 = Backend::VecI;

    const VecF zero = Backend::zero_f32();
    if (n_rows == 0 || tile_len == 0) {
        if (tile_max_out != nullptr && n_rows > 0) {
            const SizeType m_stride =
                max_out_stride != 0 ? max_out_stride : state_stride;
            for (SizeType k = 0; k < K; ++k) {
                std::fill_n(tile_max_out + (k * m_stride), n_rows, 0.0F);
            }
        }
        return 0U;
    }

    std::array<const float*, kBatch> ptrs{};
    for (SizeType r = 0; r < kBatch; ++r) {
        ptrs[r] = row_ptrs[r < n_rows ? r : 0];
    }

    std::array<std::array<VecF, kGroups>, K> v_bias{};
    std::array<std::array<VecF, kGroups>, K> v_cut{};
    std::array<std::array<VecF, kGroups>, K> c{};
    std::array<std::array<VecF, kGroups>, K> m{};

    for (SizeType k = 0; k < K; ++k) {
        load_batch_param<Backend, float>(bias, bias_stride, k, n_rows, 0.0F,
                                         v_bias[k]);
        load_batch_param<Backend, float>(cut, cut_stride, k, n_rows,
                                         std::numeric_limits<float>::max(),
                                         v_cut[k]);
        load_batch_param<Backend, float>(state_c, state_stride, k, n_rows, 0.0F,
                                         c[k]);
        for (SizeType g = 0; g < kGroups; ++g) {
            c[k][g] = Backend::vmax_nonneg_f32(zero, c[k][g]);
            m[k][g] = zero;
        }
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

    const auto step_unmasked = [&](SizeType g, VecF x) noexcept {
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
        const auto step_masked = [&](SizeType g, VecF x,
                                     SizeType sample_t) noexcept {
            const VecI vt = Backend::set1_i32(static_cast<int32_t>(sample_t));
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
                const auto cols =
                    Backend::load_transpose_f32(ptrs.data() + (g * kLanes), t);
                for (SizeType s = 0; s < kLanes; ++s) {
                    if (t + s < max_fv) {
                        step_masked(g, cols[s], t + s);
                    } else {
                        step_unmasked(g, cols[s]);
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
                        step_masked(g, cols[s], t + s);
                    } else {
                        step_unmasked(g, cols[s]);
                    }
                }
            }
            t = tile_len;
        }
    }

    // Zero-overhead unmasked fast loop across full `L x L` blocks.
    for (; t + kLanes <= tile_len; t += kLanes) {
        for (SizeType g = 0; g < kGroups; ++g) {
            const auto cols =
                Backend::load_transpose_f32(ptrs.data() + (g * kLanes), t);
            for (SizeType s = 0; s < kLanes; ++s) {
                step_unmasked(g, cols[s]);
            }
        }
    }

    // Safe masked tail (`1 <= t_rem < L`).
    if (t < tile_len) {
        const SizeType t_rem = tile_len - t;
        for (SizeType g = 0; g < kGroups; ++g) {
            const auto cols = Backend::load_transpose_tail_f32(
                ptrs.data() + (g * kLanes), t, t_rem);
            for (SizeType s = 0; s < t_rem; ++s) {
                step_unmasked(g, cols[s]);
            }
        }
    }

    const SizeType eff_lag_stride = lag_stride != 0 ? lag_stride : state_stride;
    const SizeType eff_max_stride =
        max_out_stride != 0 ? max_out_stride : state_stride;

    std::array<uint32_t, kGroups> group_hits{};
    for (SizeType k = 0; k < K; ++k) {
        store_batch_param<Backend, float>(c[k], state_c, state_stride, k,
                                          n_rows);
        if (tile_max_out != nullptr) {
            store_batch_param<Backend, float>(m[k], tile_max_out,
                                              eff_max_stride, k, n_rows);
        }
        std::array<VecF, kGroups> v_lag{};
        if (lag_max_in != nullptr) {
            load_batch_param<Backend, float>(lag_max_in, eff_lag_stride, k,
                                             n_rows, 0.0F, v_lag);
        }
        for (SizeType g = 0; g < kGroups; ++g) {
            const VecF eff_m = lag_max_in != nullptr
                                   ? Backend::vmax_nonneg_f32(m[k][g], v_lag[g])
                                   : m[k][g];
            group_hits[g] |= Backend::cmpge_mask_f32(eff_m, v_cut[k][g]);
        }
    }
    uint32_t hit_mask = 0U;
    for (SizeType g = 0; g < kGroups; ++g) {
        hit_mask |= (group_hits[g] << static_cast<uint32_t>(g * kLanes));
    }
    return hit_mask & active_row_mask(n_rows);
}

template <SizeType K, typename Backend>
[[nodiscard]] inline uint32_t
scan_gate_batch_i32(SizeType n_rows,
                    SizeType tile_len,
                    const uint32_t* const* row_ptrs,
                    uint8_t int_shift,
                    const int32_t* ibias,
                    SizeType ibias_stride,
                    const int32_t* icut,
                    SizeType icut_stride,
                    int32_t* state_c,
                    SizeType state_stride,
                    const SizeType* first_valid_t,
                    const int32_t* lag_max_in,
                    SizeType lag_stride,
                    int32_t* tile_max_out,
                    SizeType max_out_stride) noexcept {
    static_assert(K >= 1 && K <= kMaxGateBuckets);
    constexpr SizeType kLanes  = Backend::kSimdLanes;
    constexpr SizeType kGroups = Backend::kRowGroups;
    constexpr SizeType kBatch  = Backend::kBatchRows;
    using VecI                 = Backend::VecI;

    const VecI zero  = Backend::zero_i32();
    const VecI v_cap = Backend::set1_i32(kKadaneInt32Cap);
    if (n_rows == 0 || tile_len == 0) {
        if (tile_max_out != nullptr && n_rows > 0) {
            const SizeType m_stride =
                max_out_stride != 0 ? max_out_stride : state_stride;
            for (SizeType k = 0; k < K; ++k) {
                std::fill_n(tile_max_out + (k * m_stride), n_rows, int32_t{0});
            }
        }
        return 0U;
    }

    std::array<const int32_t*, kBatch> ptrs{};
    for (SizeType r = 0; r < kBatch; ++r) {
        ptrs[r] =
            reinterpret_cast<const int32_t*>(row_ptrs[r < n_rows ? r : 0]);
    }

    std::array<std::array<VecI, kGroups>, K> v_ibias{};
    std::array<std::array<VecI, kGroups>, K> v_icut{};
    std::array<std::array<VecI, kGroups>, K> c{};
    std::array<std::array<VecI, kGroups>, K> m{};

    for (SizeType k = 0; k < K; ++k) {
        load_batch_param<Backend, int32_t>(ibias, ibias_stride, k, n_rows, 0,
                                           v_ibias[k]);
        load_batch_param<Backend, int32_t>(icut, icut_stride, k, n_rows,
                                           std::numeric_limits<int32_t>::max(),
                                           v_icut[k]);
        load_batch_param<Backend, int32_t>(state_c, state_stride, k, n_rows, 0,
                                           c[k]);
        for (SizeType g = 0; g < kGroups; ++g) {
            c[k][g] = Backend::max_i32(zero, c[k][g]);
            m[k][g] = zero;
        }
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

    const auto step_unmasked = [&](SizeType g, VecI raw_x) noexcept {
        const VecI xs =
            int_shift != 0U ? Backend::shl_u32(raw_x, int_shift) : raw_x;
        for (SizeType k = 0; k < K; ++k) {
            const VecI delta   = Backend::sub_i32(xs, v_ibias[k][g]);
            const VecI clamped = Backend::min_i32(c[k][g], v_cap);
            const VecI next =
                Backend::max_i32(zero, Backend::add_i32(clamped, delta));
            m[k][g] = Backend::max_i32(m[k][g], next);
            c[k][g] = next;
        }
    };

    SizeType t = 0;
    if (max_fv > 0) {
        std::array<VecI, kGroups> v_fv{};
        for (SizeType g = 0; g < kGroups; ++g) {
            v_fv[g] = Backend::loadu_i32(fv_clamped.data() + (g * kLanes));
        }
        const auto step_masked = [&](SizeType g, VecI raw_x,
                                     SizeType sample_t) noexcept {
            const VecI xs =
                int_shift != 0U ? Backend::shl_u32(raw_x, int_shift) : raw_x;
            const VecI vt = Backend::set1_i32(static_cast<int32_t>(sample_t));
            for (SizeType k = 0; k < K; ++k) {
                const VecI delta   = Backend::sub_i32(xs, v_ibias[k][g]);
                const VecI clamped = Backend::min_i32(c[k][g], v_cap);
                const VecI raw_next =
                    Backend::max_i32(zero, Backend::add_i32(clamped, delta));
                const VecI next =
                    Backend::maskz_valid_i32(raw_next, vt, v_fv[g]);
                m[k][g] = Backend::max_i32(m[k][g], next);
                c[k][g] = next;
            }
        };

        for (; t < max_fv && t + kLanes <= tile_len; t += kLanes) {
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols =
                    Backend::load_transpose_i32(ptrs.data() + (g * kLanes), t);
                for (SizeType s = 0; s < kLanes; ++s) {
                    if (t + s < max_fv) {
                        step_masked(g, cols[s], t + s);
                    } else {
                        step_unmasked(g, cols[s]);
                    }
                }
            }
        }
        if (t < tile_len && t < max_fv) {
            const SizeType t_rem = tile_len - t;
            for (SizeType g = 0; g < kGroups; ++g) {
                const auto cols = Backend::load_transpose_tail_i32(
                    ptrs.data() + (g * kLanes), t, t_rem);
                for (SizeType s = 0; s < t_rem; ++s) {
                    if (t + s < max_fv) {
                        step_masked(g, cols[s], t + s);
                    } else {
                        step_unmasked(g, cols[s]);
                    }
                }
            }
            t = tile_len;
        }
    }

    for (; t + kLanes <= tile_len; t += kLanes) {
        for (SizeType g = 0; g < kGroups; ++g) {
            const auto cols =
                Backend::load_transpose_i32(ptrs.data() + (g * kLanes), t);
            for (SizeType s = 0; s < kLanes; ++s) {
                step_unmasked(g, cols[s]);
            }
        }
    }

    if (t < tile_len) {
        const SizeType t_rem = tile_len - t;
        for (SizeType g = 0; g < kGroups; ++g) {
            const auto cols = Backend::load_transpose_tail_i32(
                ptrs.data() + (g * kLanes), t, t_rem);
            for (SizeType s = 0; s < t_rem; ++s) {
                step_unmasked(g, cols[s]);
            }
        }
    }

    const SizeType eff_lag_stride = lag_stride != 0 ? lag_stride : state_stride;
    const SizeType eff_max_stride =
        max_out_stride != 0 ? max_out_stride : state_stride;

    std::array<uint32_t, kGroups> group_hits{};
    for (SizeType k = 0; k < K; ++k) {
        store_batch_param<Backend, int32_t>(c[k], state_c, state_stride, k,
                                            n_rows);
        if (tile_max_out != nullptr) {
            store_batch_param<Backend, int32_t>(m[k], tile_max_out,
                                                eff_max_stride, k, n_rows);
        }
        std::array<VecI, kGroups> v_lag{};
        if (lag_max_in != nullptr) {
            load_batch_param<Backend, int32_t>(lag_max_in, eff_lag_stride, k,
                                               n_rows, 0, v_lag);
        }
        for (SizeType g = 0; g < kGroups; ++g) {
            const VecI eff_m = lag_max_in != nullptr
                                   ? Backend::max_i32(m[k][g], v_lag[g])
                                   : m[k][g];
            group_hits[g] |= Backend::cmpge_mask_i32(eff_m, v_icut[k][g]);
        }
    }
    uint32_t hit_mask = 0U;
    for (SizeType g = 0; g < kGroups; ++g) {
        hit_mask |= (group_hits[g] << static_cast<uint32_t>(g * kLanes));
    }
    return hit_mask & active_row_mask(n_rows);
}

} // namespace detail

// ============================================================================
// Kernel 1: Lossless Float32 Tile Gate (`K in {1, 2, 3}`)
// ============================================================================

/**
 * @brief Lossless multi-bucket float32 Kadane Stage-1 tile gate (`K in 1..3`).
 *
 * Scans `1 <= n_rows <= kMaxBatchRows` (single pass when `n_rows <=
 * Backend::kBatchRows`) over `tile_len` samples, updating SoA running states
 * `state_c[k * state_stride + r]` and returning a `uint32_t` row-hit bitmask
 * where `max(m_k, lag_max_in) >= cut_k`.
 */
template <SizeType K, typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_f32(SizeType n_rows,
                   SizeType tile_len,
                   const float* const* row_ptrs,
                   const float* bias,
                   SizeType bias_stride,
                   const float* cut,
                   SizeType cut_stride,
                   float* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const float* lag_max_in       = nullptr,
                   SizeType lag_stride           = 0,
                   float* tile_max_out           = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    constexpr SizeType kBatch = Backend::kBatchRows;
    if (n_rows <= kBatch) {
        return detail::scan_gate_batch_f32<K, Backend>(
            n_rows, tile_len, row_ptrs, bias, bias_stride, cut, cut_stride,
            state_c, state_stride, first_valid_t, lag_max_in, lag_stride,
            tile_max_out, max_out_stride);
    }
    const SizeType cap_rows   = std::min(n_rows, kMaxBatchRows);
    uint32_t combined_hitmask = 0U;
    for (SizeType r0 = 0; r0 < cap_rows; r0 += kBatch) {
        const SizeType sub_n   = std::min(kBatch, cap_rows - r0);
        const uint32_t sub_hit = detail::scan_gate_batch_f32<K, Backend>(
            sub_n, tile_len, row_ptrs + r0, bias + r0, bias_stride, cut + r0,
            cut_stride, state_c + r0, state_stride,
            first_valid_t != nullptr ? first_valid_t + r0 : nullptr,
            lag_max_in != nullptr ? lag_max_in + r0 : nullptr, lag_stride,
            tile_max_out != nullptr ? tile_max_out + r0 : nullptr,
            max_out_stride);
        combined_hitmask |= (sub_hit << static_cast<uint32_t>(r0));
    }
    return combined_hitmask;
}

/// Runtime bucket-count (`num_buckets in {1, 2, 3}`) dispatcher for Kernel 1.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_f32(SizeType num_buckets,
                   SizeType n_rows,
                   SizeType tile_len,
                   const float* const* row_ptrs,
                   const float* bias,
                   SizeType bias_stride,
                   const float* cut,
                   SizeType cut_stride,
                   float* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const float* lag_max_in       = nullptr,
                   SizeType lag_stride           = 0,
                   float* tile_max_out           = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    switch (num_buckets) {
    case 1:
        return scan_gate_tile_f32<1, Backend>(
            n_rows, tile_len, row_ptrs, bias, bias_stride, cut, cut_stride,
            state_c, state_stride, first_valid_t, lag_max_in, lag_stride,
            tile_max_out, max_out_stride);
    case 2:
        return scan_gate_tile_f32<2, Backend>(
            n_rows, tile_len, row_ptrs, bias, bias_stride, cut, cut_stride,
            state_c, state_stride, first_valid_t, lag_max_in, lag_stride,
            tile_max_out, max_out_stride);
    case 3:
        return scan_gate_tile_f32<3, Backend>(
            n_rows, tile_len, row_ptrs, bias, bias_stride, cut, cut_stride,
            state_c, state_stride, first_valid_t, lag_max_in, lag_stride,
            tile_max_out, max_out_stride);
    default:
        return 0U;
    }
}

/// `KadaneGatePlan` convenience overload for `scan_gate_tile_f32`.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_f32(const KadaneGatePlan& plan,
                   SizeType row_offset,
                   SizeType n_rows,
                   SizeType tile_len,
                   const float* const* row_ptrs,
                   float* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const float* lag_max_in       = nullptr,
                   SizeType lag_stride           = 0,
                   float* tile_max_out           = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    if (row_offset >= plan.n_rows || plan.num_buckets == 0) {
        return 0U;
    }
    const SizeType eff_rows = std::min(n_rows, plan.n_rows - row_offset);
    return scan_gate_tile_f32<Backend>(
        plan.num_buckets, eff_rows, tile_len, row_ptrs,
        plan.bias.data() + row_offset, plan.row_stride,
        plan.cut.data() + row_offset, plan.row_stride, state_c, state_stride,
        first_valid_t, lag_max_in, lag_stride, tile_max_out, max_out_stride);
}

// ============================================================================
// Kernel 2: Lossless Exact `int32_t` Tile Gate (`K in {1, 2, 3}`)
// ============================================================================

/**
 * @brief Lossless multi-bucket exact `int32_t` Kadane Stage-1 tile gate.
 *
 * Implements the overflow-free clamped recurrence from `KadaneGatePlan`:
 * `delta = static_cast<int32_t>(x_u32 << int_shift) - ibias`,
 * `c = max(0, min(c, kKadaneInt32Cap) + delta)`, `m = max(m, c)`, returning
 * a `uint32_t` row-hit bitmask where `max(m_k, lag_max_in) >= icut_k`.
 */
template <SizeType K, typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_i32(SizeType n_rows,
                   SizeType tile_len,
                   const uint32_t* const* row_ptrs,
                   uint8_t int_shift,
                   const int32_t* ibias,
                   SizeType ibias_stride,
                   const int32_t* icut,
                   SizeType icut_stride,
                   int32_t* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const int32_t* lag_max_in     = nullptr,
                   SizeType lag_stride           = 0,
                   int32_t* tile_max_out         = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    constexpr SizeType kBatch = Backend::kBatchRows;
    if (n_rows <= kBatch) {
        return detail::scan_gate_batch_i32<K, Backend>(
            n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride, icut,
            icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
            lag_stride, tile_max_out, max_out_stride);
    }
    const SizeType cap_rows   = std::min(n_rows, kMaxBatchRows);
    uint32_t combined_hitmask = 0U;
    for (SizeType r0 = 0; r0 < cap_rows; r0 += kBatch) {
        const SizeType sub_n   = std::min(kBatch, cap_rows - r0);
        const uint32_t sub_hit = detail::scan_gate_batch_i32<K, Backend>(
            sub_n, tile_len, row_ptrs + r0, int_shift, ibias + r0, ibias_stride,
            icut + r0, icut_stride, state_c + r0, state_stride,
            first_valid_t != nullptr ? first_valid_t + r0 : nullptr,
            lag_max_in != nullptr ? lag_max_in + r0 : nullptr, lag_stride,
            tile_max_out != nullptr ? tile_max_out + r0 : nullptr,
            max_out_stride);
        combined_hitmask |= (sub_hit << static_cast<uint32_t>(r0));
    }
    return combined_hitmask;
}

/// Runtime bucket-count (`num_buckets in {1, 2, 3}`) dispatcher for Kernel 2.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_i32(SizeType num_buckets,
                   SizeType n_rows,
                   SizeType tile_len,
                   const uint32_t* const* row_ptrs,
                   uint8_t int_shift,
                   const int32_t* ibias,
                   SizeType ibias_stride,
                   const int32_t* icut,
                   SizeType icut_stride,
                   int32_t* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const int32_t* lag_max_in     = nullptr,
                   SizeType lag_stride           = 0,
                   int32_t* tile_max_out         = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    switch (num_buckets) {
    case 1:
        return scan_gate_tile_i32<1, Backend>(
            n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride, icut,
            icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
            lag_stride, tile_max_out, max_out_stride);
    case 2:
        return scan_gate_tile_i32<2, Backend>(
            n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride, icut,
            icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
            lag_stride, tile_max_out, max_out_stride);
    case 3:
        return scan_gate_tile_i32<3, Backend>(
            n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride, icut,
            icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
            lag_stride, tile_max_out, max_out_stride);
    default:
        return 0U;
    }
}

/// `KadaneGatePlan` convenience overload for `scan_gate_tile_i32`.
template <typename Backend = NativeBackend>
[[nodiscard]] inline uint32_t
scan_gate_tile_i32(const KadaneGatePlan& plan,
                   SizeType row_offset,
                   SizeType n_rows,
                   SizeType tile_len,
                   const uint32_t* const* row_ptrs,
                   int32_t* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const int32_t* lag_max_in     = nullptr,
                   SizeType lag_stride           = 0,
                   int32_t* tile_max_out         = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    if (!plan.has_int32 || row_offset >= plan.n_rows || plan.num_buckets == 0) {
        return 0U;
    }
    const SizeType eff_rows = std::min(n_rows, plan.n_rows - row_offset);
    return scan_gate_tile_i32<Backend>(
        plan.num_buckets, eff_rows, tile_len, row_ptrs, plan.int_shift,
        plan.ibias.data() + row_offset, plan.row_stride,
        plan.icut.data() + row_offset, plan.row_stride, state_c, state_stride,
        first_valid_t, lag_max_in, lag_stride, tile_max_out, max_out_stride);
}

namespace scalar {

template <SizeType K>
[[nodiscard]] inline uint32_t
scan_gate_tile_f32(SizeType n_rows,
                   SizeType tile_len,
                   const float* const* row_ptrs,
                   const float* bias,
                   SizeType bias_stride,
                   const float* cut,
                   SizeType cut_stride,
                   float* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const float* lag_max_in       = nullptr,
                   SizeType lag_stride           = 0,
                   float* tile_max_out           = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    return dmt::kadane::scan_gate_tile_f32<K, ScalarBackend>(
        n_rows, tile_len, row_ptrs, bias, bias_stride, cut, cut_stride, state_c,
        state_stride, first_valid_t, lag_max_in, lag_stride, tile_max_out,
        max_out_stride);
}

[[nodiscard]] inline uint32_t
scan_gate_tile_f32(SizeType num_buckets,
                   SizeType n_rows,
                   SizeType tile_len,
                   const float* const* row_ptrs,
                   const float* bias,
                   SizeType bias_stride,
                   const float* cut,
                   SizeType cut_stride,
                   float* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const float* lag_max_in       = nullptr,
                   SizeType lag_stride           = 0,
                   float* tile_max_out           = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    return dmt::kadane::scan_gate_tile_f32<ScalarBackend>(
        num_buckets, n_rows, tile_len, row_ptrs, bias, bias_stride, cut,
        cut_stride, state_c, state_stride, first_valid_t, lag_max_in,
        lag_stride, tile_max_out, max_out_stride);
}

template <SizeType K>
[[nodiscard]] inline uint32_t
scan_gate_tile_i32(SizeType n_rows,
                   SizeType tile_len,
                   const uint32_t* const* row_ptrs,
                   uint8_t int_shift,
                   const int32_t* ibias,
                   SizeType ibias_stride,
                   const int32_t* icut,
                   SizeType icut_stride,
                   int32_t* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const int32_t* lag_max_in     = nullptr,
                   SizeType lag_stride           = 0,
                   int32_t* tile_max_out         = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    return dmt::kadane::scan_gate_tile_i32<K, ScalarBackend>(
        n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride, icut,
        icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
        lag_stride, tile_max_out, max_out_stride);
}

[[nodiscard]] inline uint32_t
scan_gate_tile_i32(SizeType num_buckets,
                   SizeType n_rows,
                   SizeType tile_len,
                   const uint32_t* const* row_ptrs,
                   uint8_t int_shift,
                   const int32_t* ibias,
                   SizeType ibias_stride,
                   const int32_t* icut,
                   SizeType icut_stride,
                   int32_t* state_c,
                   SizeType state_stride,
                   const SizeType* first_valid_t = nullptr,
                   const int32_t* lag_max_in     = nullptr,
                   SizeType lag_stride           = 0,
                   int32_t* tile_max_out         = nullptr,
                   SizeType max_out_stride       = 0) noexcept {
    return dmt::kadane::scan_gate_tile_i32<ScalarBackend>(
        num_buckets, n_rows, tile_len, row_ptrs, int_shift, ibias, ibias_stride,
        icut, icut_stride, state_c, state_stride, first_valid_t, lag_max_in,
        lag_stride, tile_max_out, max_out_stride);
}

} // namespace scalar

} // namespace dmt::kadane
