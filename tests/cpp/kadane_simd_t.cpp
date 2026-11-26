#include "dmt/kadane_simd.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/types.hpp"

#include "dmt/kadane_plan.hpp"

#include "kadane_test_tables.hpp"

namespace dmt {
namespace {

using kadane::NativeBackend;
using kadane::ScalarBackend;
using kadane_test::exact_tile_has_hit_f32;
using kadane_test::exact_tile_has_hit_int;
using kadane_test::make_synthetic_tables;
using kadane_test::SyntheticPlaneTables;

std::mt19937 make_seeded_rng(uint32_t seed) { return std::mt19937(seed); }

template <typename Backend> void verify_transpose_backend() {
    constexpr SizeType kLanes = Backend::kSimdLanes;
    const SizeType t_offset   = 5;

    for (SizeType valid_cols = 0; valid_cols <= kLanes; ++valid_cols) {
        // Allocate exact-length row buffers so any OOB read past `valid_cols`
        // touches past end-of-allocation.
        std::vector<std::vector<float>> f_rows(kLanes);
        std::vector<std::vector<int32_t>> i_rows(kLanes);
        std::array<const float*, kLanes> f_ptrs{};
        std::array<const int32_t*, kLanes> i_ptrs{};

        for (SizeType r = 0; r < kLanes; ++r) {
            f_rows[r].resize(t_offset + valid_cols);
            i_rows[r].resize(t_offset + valid_cols);
            for (SizeType c = 0; c < t_offset + valid_cols; ++c) {
                const auto idx = static_cast<int32_t>((r * 37) + (c * 13) + 1);
                f_rows[r][c]   = ((r + c) % 3 == 0 ? -1.0F : 1.0F) *
                                 (static_cast<float>(idx) * 0.25F);
                if (r == 0 && c == t_offset) {
                    f_rows[r][c] = -0.0F;
                }
                i_rows[r][c] = ((r + c) % 2 == 0) ? idx : -idx * 100000;
            }
            f_ptrs[r] = f_rows[r].data();
            i_ptrs[r] = i_rows[r].data();
        }

        std::array<float, kLanes * kLanes> f_dst{};
        f_dst.fill(-999.0F);
        kadane::transpose_block_f32<Backend>(f_ptrs.data(), t_offset,
                                             valid_cols, f_dst.data());

        std::array<int32_t, kLanes * kLanes> i_dst{};
        i_dst.fill(-777777);
        kadane::transpose_block_i32<Backend>(i_ptrs.data(), t_offset,
                                             valid_cols, i_dst.data());

        for (SizeType s = 0; s < kLanes; ++s) {
            for (SizeType r = 0; r < kLanes; ++r) {
                const float exp_f =
                    s < valid_cols ? f_rows[r][t_offset + s] : 0.0F;
                const int32_t exp_i =
                    s < valid_cols ? i_rows[r][t_offset + s] : 0;
                CHECK(std::bit_cast<uint32_t>(f_dst[(s * kLanes) + r]) ==
                      std::bit_cast<uint32_t>(exp_f));
                CHECK(i_dst[(s * kLanes) + r] == exp_i);
            }
        }
    }
}

uint32_t reference_gate_tile_f32(SizeType num_buckets,
                                 SizeType n_rows,
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
                                 SizeType max_out_stride) {
    const SizeType eff_lag = lag_stride != 0 ? lag_stride : state_stride;
    const SizeType eff_max =
        max_out_stride != 0 ? max_out_stride : state_stride;
    if (n_rows == 0 || tile_len == 0) {
        if (tile_max_out != nullptr && n_rows > 0) {
            for (SizeType k = 0; k < num_buckets; ++k) {
                std::fill_n(tile_max_out + (k * eff_max), n_rows, 0.0F);
            }
        }
        return 0U;
    }

    uint32_t mask = 0U;
    for (SizeType r = 0; r < n_rows; ++r) {
        const SizeType fv = first_valid_t != nullptr ? first_valid_t[r] : 0;
        bool hit          = false;
        for (SizeType k = 0; k < num_buckets; ++k) {
            const float b  = bias[(k * bias_stride) + r];
            const float th = cut[(k * cut_stride) + r];
            float c        = ScalarBackend::vmax_nonneg_f32(
                0.0F, state_c[(k * state_stride) + r]);
            float m = 0.0F;
            for (SizeType t = 0; t < tile_len; ++t) {
                if (t < fv) {
                    c = 0.0F;
                } else {
                    const float d = ScalarBackend::sub_f32(row_ptrs[r][t], b);
                    c             = ScalarBackend::vmax_nonneg_f32(
                        0.0F, ScalarBackend::add_f32(c, d));
                }
                m = ScalarBackend::vmax_nonneg_f32(m, c);
            }
            state_c[(k * state_stride) + r] = c;
            if (tile_max_out != nullptr) {
                tile_max_out[(k * eff_max) + r] = m;
            }
            const float eff_m = lag_max_in != nullptr
                                    ? ScalarBackend::vmax_nonneg_f32(
                                          m, lag_max_in[(k * eff_lag) + r])
                                    : m;
            if (eff_m >= th) {
                hit = true;
            }
        }
        if (hit) {
            mask |= (uint32_t{1} << static_cast<uint32_t>(r));
        }
    }
    return mask;
}

uint32_t reference_gate_tile_i32(SizeType num_buckets,
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
                                 const SizeType* first_valid_t,
                                 const int32_t* lag_max_in,
                                 SizeType lag_stride,
                                 int32_t* tile_max_out,
                                 SizeType max_out_stride) {
    const SizeType eff_lag = lag_stride != 0 ? lag_stride : state_stride;
    const SizeType eff_max =
        max_out_stride != 0 ? max_out_stride : state_stride;
    if (n_rows == 0 || tile_len == 0) {
        if (tile_max_out != nullptr && n_rows > 0) {
            for (SizeType k = 0; k < num_buckets; ++k) {
                std::fill_n(tile_max_out + (k * eff_max), n_rows, int32_t{0});
            }
        }
        return 0U;
    }

    uint32_t mask = 0U;
    for (SizeType r = 0; r < n_rows; ++r) {
        const SizeType fv = first_valid_t != nullptr ? first_valid_t[r] : 0;
        bool hit          = false;
        for (SizeType k = 0; k < num_buckets; ++k) {
            const int32_t b  = ibias[(k * ibias_stride) + r];
            const int32_t th = icut[(k * icut_stride) + r];
            int32_t c = std::max(int32_t{0}, state_c[(k * state_stride) + r]);
            int32_t m = 0;
            for (SizeType t = 0; t < tile_len; ++t) {
                if (t < fv) {
                    c = 0;
                } else {
                    const uint32_t xs = row_ptrs[r][t] << int_shift;
                    const int32_t d   = static_cast<int32_t>(xs) - b;
                    const int64_t sum = static_cast<int64_t>(std::min(
                                            c, kadane::kKadaneInt32Cap)) +
                                        static_cast<int64_t>(d);
                    REQUIRE(sum >= static_cast<int64_t>(
                                       std::numeric_limits<int32_t>::min()));
                    REQUIRE(sum <= static_cast<int64_t>(
                                       std::numeric_limits<int32_t>::max()));
                    c = std::max(int32_t{0}, static_cast<int32_t>(sum));
                }
                m = std::max(m, c);
            }
            state_c[(k * state_stride) + r] = c;
            if (tile_max_out != nullptr) {
                tile_max_out[(k * eff_max) + r] = m;
            }
            const int32_t eff_m =
                lag_max_in != nullptr
                    ? std::max(m, lag_max_in[(k * eff_lag) + r])
                    : m;
            if (eff_m >= th) {
                hit = true;
            }
        }
        if (hit) {
            mask |= (uint32_t{1} << static_cast<uint32_t>(r));
        }
    }
    return mask;
}

} // namespace

// ============================================================================
// 1. ISA Dispatch, Bit Primitives & Unsigned Conversion
// ============================================================================

TEST_CASE("kadane_simd - bitmasks, vmax_nonneg_f32 and u32->f32",
          "[kadane_simd][kadane][boxcar][cpu]") {
    CHECK(kadane::active_row_mask(0) == 0U);
    CHECK(kadane::active_row_mask(1) == 0x1U);
    CHECK(kadane::active_row_mask(16) == 0xFFFFU);
    CHECK(kadane::active_row_mask(31) == 0x7FFFFFFFU);
    CHECK(kadane::active_row_mask(32) == 0xFFFFFFFFU);

    // `-0.0F` vs `+0.0F` canonicalization under signed int32 max.
    const auto z_sc = ScalarBackend::vmax_nonneg_f32(0.0F, -0.0F);
    CHECK(std::bit_cast<uint32_t>(z_sc) == 0U);
    alignas(64) std::array<float, NativeBackend::kSimdLanes> neg_zeros{};
    neg_zeros.fill(-0.0F);
    alignas(64) std::array<float, NativeBackend::kSimdLanes> out_zeros{};
    const auto vz = NativeBackend::vmax_nonneg_f32(
        NativeBackend::zero_f32(), NativeBackend::loadu_f32(neg_zeros.data()));
    NativeBackend::storeu_f32(out_zeros.data(), vz);
    for (float v : out_zeros) {
        CHECK(std::bit_cast<uint32_t>(v) == 0U);
    }

    // Unsigned 32-bit -> float32 conversion including high-bit `>= 2^31`
    // values.
    const std::array<SizeType, 8> counts = {0, 1, 3, 4, 7, 15, 16, 65};
    for (SizeType count : counts) {
        std::vector<uint32_t> src(count);
        for (SizeType i = 0; i < count; ++i) {
            switch (i % 7) {
            case 0:
                src[i] = 0U;
                break;
            case 1:
                src[i] = 1U;
                break;
            case 2:
                src[i] = 0x7FFFFFFFU;
                break;
            case 3:
                src[i] = 0x80000000U;
                break;
            case 4:
                src[i] = 0x80000001U;
                break;
            case 5:
                src[i] = 0xFFFFFF00U;
                break;
            default:
                src[i] = 0xFFFFFFFFU;
                break;
            }
        }
        std::vector<float> dst_nat(count, -1.0F);
        std::vector<float> dst_sca(count, -2.0F);
        kadane::convert_u32_to_f32<NativeBackend>(src.data(), dst_nat.data(),
                                                  count);
        kadane::convert_u32_to_f32<ScalarBackend>(src.data(), dst_sca.data(),
                                                  count);
        for (SizeType i = 0; i < count; ++i) {
            const auto expected = static_cast<float>(src[i]);
            CHECK(std::bit_cast<uint32_t>(dst_nat[i]) ==
                  std::bit_cast<uint32_t>(expected));
            CHECK(std::bit_cast<uint32_t>(dst_sca[i]) ==
                  std::bit_cast<uint32_t>(expected));
        }
    }
}

// ============================================================================
// 2. Handwritten Register Transpose Exactness & Safe Masked Tails
// ============================================================================

TEST_CASE("kadane_simd - transpose_LxL_f32 and transpose_LxL_i32 with tails",
          "[kadane_simd][kadane][boxcar][cpu]") {
    verify_transpose_backend<ScalarBackend>();
    verify_transpose_backend<NativeBackend>();
}

// ============================================================================
// 3. Kernel 1 (`scan_gate_tile_f32`) Bit-for-Bit Parity Across Rows/Lengths/K
// ============================================================================

TEST_CASE(
    "kadane_simd - scan_gate_tile_f32 bit-for-bit parity (Native vs Scalar)",
    "[kadane_simd][kadane][boxcar][cpu]") {
    auto rng = make_seeded_rng(20261113U);
    std::normal_distribution<float> noise_dist(120.0F, 2.5F);
    const std::array<SizeType, 10> tile_lens = {0,  1,  3,   7,   15,
                                                16, 17, 255, 256, 513};

    // Exhaustive `n_rows = 1..32` x `K = 1..3` parity check across unaligned
    // tiles, plus full `tile_len` sweep on boundary row counts.
    for (SizeType k_buckets = 1; k_buckets <= 3; ++k_buckets) {
        for (SizeType n_rows = 1; n_rows <= 32; ++n_rows) {
            const bool full_len_sweep =
                (n_rows == 1 || n_rows == 3 || n_rows == 4 || n_rows == 7 ||
                 n_rows == 8 || n_rows == 15 || n_rows == 16 || n_rows == 17 ||
                 n_rows == 31 || n_rows == 32);

            for (SizeType tile_len : tile_lens) {
                if (!full_len_sweep && tile_len != 3 && tile_len != 17) {
                    continue;
                }

                // Exact-sized per-row buffers (`tile_len` samples) to detect
                // OOB.
                std::vector<std::vector<float>> rows(
                    n_rows,
                    std::vector<float>(std::max<SizeType>(1, tile_len)));
                std::vector<const float*> ptrs(n_rows);
                for (SizeType r = 0; r < n_rows; ++r) {
                    for (SizeType t = 0; t < tile_len; ++t) {
                        rows[r][t] = noise_dist(rng);
                        if (((r + t) % 29) == 0) {
                            rows[r][t] = -0.0F;
                        }
                    }
                    ptrs[r] = rows[r].data();
                }

                // Tight non-16-aligned stride == n_rows tests OOB safety of
                // `load_batch_param` / `store_batch_param`!
                const SizeType stride = n_rows;
                std::vector<float> bias(k_buckets * stride);
                std::vector<float> cut(k_buckets * stride);
                std::vector<float> init_c(k_buckets * stride);
                std::vector<float> lag_in(k_buckets * stride);
                std::vector<SizeType> fv(n_rows, 0);

                for (SizeType r = 0; r < n_rows; ++r) {
                    fv[r] = (r % 5 == 0) ? 0 : ((r * 3) % 23);
                    for (SizeType b = 0; b < k_buckets; ++b) {
                        bias[(b * stride) + r] =
                            120.5F + (0.4F * static_cast<float>(b));
                        cut[(b * stride) + r] =
                            (r % 7 == 6)
                                ? std::numeric_limits<float>::max()
                                : (4.5F + (2.0F * static_cast<float>(b)) +
                                   (0.25F * static_cast<float>(r % 5)));
                        init_c[(b * stride) + r] =
                            (r % 3 == 0) ? 0.0F
                                         : (1.25F * static_cast<float>(b + 1));
                        lag_in[(b * stride) + r] =
                            (r % 11 == 0) ? cut[(b * stride) + r] : 0.5F;
                    }
                }

                for (int use_fv = 0; use_fv < 2; ++use_fv) {
                    const SizeType* fv_ptr = use_fv != 0 ? fv.data() : nullptr;
                    const float* lag_ptr =
                        (tile_len % 2 == 1) ? lag_in.data() : nullptr;

                    auto c_nat = init_c;
                    auto c_sca = init_c;
                    auto c_ns  = init_c;
                    auto c_ref = init_c;
                    std::vector<float> m_nat(k_buckets * stride, -1.0F);
                    std::vector<float> m_sca(k_buckets * stride, -2.0F);
                    std::vector<float> m_ns(k_buckets * stride, -3.0F);
                    std::vector<float> m_ref(k_buckets * stride, -4.0F);

                    const uint32_t hm_nat =
                        kadane::scan_gate_tile_f32<NativeBackend>(
                            k_buckets, n_rows, tile_len, ptrs.data(),
                            bias.data(), stride, cut.data(), stride,
                            c_nat.data(), stride, fv_ptr, lag_ptr, stride,
                            m_nat.data(), stride);
                    const uint32_t hm_sca =
                        kadane::scan_gate_tile_f32<ScalarBackend>(
                            k_buckets, n_rows, tile_len, ptrs.data(),
                            bias.data(), stride, cut.data(), stride,
                            c_sca.data(), stride, fv_ptr, lag_ptr, stride,
                            m_sca.data(), stride);
                    const uint32_t hm_ns = kadane::scalar::scan_gate_tile_f32(
                        k_buckets, n_rows, tile_len, ptrs.data(), bias.data(),
                        stride, cut.data(), stride, c_ns.data(), stride, fv_ptr,
                        lag_ptr, stride, m_ns.data(), stride);
                    const uint32_t hm_ref = reference_gate_tile_f32(
                        k_buckets, n_rows, tile_len, ptrs.data(), bias.data(),
                        stride, cut.data(), stride, c_ref.data(), stride,
                        fv_ptr, lag_ptr, stride, m_ref.data(), stride);

                    REQUIRE(hm_nat == hm_ref);
                    REQUIRE(hm_sca == hm_ref);
                    REQUIRE(hm_ns == hm_ref);
                    for (SizeType i = 0; i < k_buckets * stride; ++i) {
                        REQUIRE(std::bit_cast<uint32_t>(c_nat[i]) ==
                                std::bit_cast<uint32_t>(c_ref[i]));
                        REQUIRE(std::bit_cast<uint32_t>(c_sca[i]) ==
                                std::bit_cast<uint32_t>(c_ref[i]));
                        REQUIRE(std::bit_cast<uint32_t>(m_nat[i]) ==
                                std::bit_cast<uint32_t>(m_ref[i]));
                        REQUIRE(std::bit_cast<uint32_t>(m_sca[i]) ==
                                std::bit_cast<uint32_t>(m_ref[i]));
                    }
                }
            }
        }
    }
}

// ============================================================================
// 4. Kernel 2 (`scan_gate_tile_i32`) Bit-for-Bit Parity & Cap Saturation
// ============================================================================

TEST_CASE("kadane_simd - scan_gate_tile_i32 bit-for-bit parity and saturation",
          "[kadane_simd][kadane][boxcar][cpu]") {
    auto rng = make_seeded_rng(9876543U);
    std::uniform_int_distribution<uint32_t> sample_dist(80U, 180U);
    const std::array<SizeType, 10> tile_lens = {0,  1,  3,   7,   15,
                                                16, 17, 255, 256, 513};
    const std::array<uint8_t, 3> shifts      = {0U, 3U, 8U};

    for (SizeType k_buckets = 1; k_buckets <= 3; ++k_buckets) {
        for (uint8_t shift : shifts) {
            for (SizeType n_rows = 1; n_rows <= 32; ++n_rows) {
                const bool full_len_sweep =
                    (shift == 3U) &&
                    (n_rows == 1 || n_rows == 4 || n_rows == 8 ||
                     n_rows == 15 || n_rows == 16 || n_rows == 17 ||
                     n_rows == 32);

                for (SizeType tile_len : tile_lens) {
                    if (!full_len_sweep && tile_len != 17) {
                        continue;
                    }

                    std::vector<std::vector<uint32_t>> rows(
                        n_rows,
                        std::vector<uint32_t>(std::max<SizeType>(1, tile_len)));
                    std::vector<const uint32_t*> ptrs(n_rows);
                    for (SizeType r = 0; r < n_rows; ++r) {
                        for (SizeType t = 0; t < tile_len; ++t) {
                            rows[r][t] = sample_dist(rng);
                        }
                        ptrs[r] = rows[r].data();
                    }

                    const SizeType stride = n_rows;
                    std::vector<int32_t> ibias(k_buckets * stride);
                    std::vector<int32_t> icut(k_buckets * stride);
                    std::vector<int32_t> init_c(k_buckets * stride);
                    std::vector<int32_t> lag_in(k_buckets * stride);
                    std::vector<SizeType> fv(n_rows, 0);

                    for (SizeType r = 0; r < n_rows; ++r) {
                        fv[r] = (r % 4 == 0) ? 0 : ((r * 5) % 19);
                        for (SizeType b = 0; b < k_buckets; ++b) {
                            ibias[(b * stride) + r] = static_cast<int32_t>(
                                (132U + (4U * b)) << shift);
                            icut[(b * stride) + r] =
                                (r % 9 == 8)
                                    ? std::numeric_limits<int32_t>::max()
                                    : static_cast<int32_t>(
                                          (90U + (35U * b) + (5U * (r % 6)))
                                          << shift);
                            init_c[(b * stride) + r] =
                                (r % 3 == 0)
                                    ? 0
                                    : static_cast<int32_t>(20U << shift);
                            lag_in[(b * stride) + r] =
                                (r % 13 == 0) ? icut[(b * stride) + r] : 10;
                        }
                    }

                    auto c_nat = init_c;
                    auto c_sca = init_c;
                    auto c_ns  = init_c;
                    auto c_ref = init_c;
                    std::vector<int32_t> m_nat(k_buckets * stride, -1);
                    std::vector<int32_t> m_sca(k_buckets * stride, -2);
                    std::vector<int32_t> m_ns(k_buckets * stride, -3);
                    std::vector<int32_t> m_ref(k_buckets * stride, -4);

                    const SizeType* fv_ptr =
                        (n_rows % 2 == 1) ? fv.data() : nullptr;
                    const int32_t* lag_ptr =
                        (tile_len % 2 == 1) ? lag_in.data() : nullptr;

                    const uint32_t hm_nat =
                        kadane::scan_gate_tile_i32<NativeBackend>(
                            k_buckets, n_rows, tile_len, ptrs.data(), shift,
                            ibias.data(), stride, icut.data(), stride,
                            c_nat.data(), stride, fv_ptr, lag_ptr, stride,
                            m_nat.data(), stride);
                    const uint32_t hm_sca =
                        kadane::scan_gate_tile_i32<ScalarBackend>(
                            k_buckets, n_rows, tile_len, ptrs.data(), shift,
                            ibias.data(), stride, icut.data(), stride,
                            c_sca.data(), stride, fv_ptr, lag_ptr, stride,
                            m_sca.data(), stride);
                    const uint32_t hm_ns = kadane::scalar::scan_gate_tile_i32(
                        k_buckets, n_rows, tile_len, ptrs.data(), shift,
                        ibias.data(), stride, icut.data(), stride, c_ns.data(),
                        stride, fv_ptr, lag_ptr, stride, m_ns.data(), stride);
                    const uint32_t hm_ref = reference_gate_tile_i32(
                        k_buckets, n_rows, tile_len, ptrs.data(), shift,
                        ibias.data(), stride, icut.data(), stride, c_ref.data(),
                        stride, fv_ptr, lag_ptr, stride, m_ref.data(), stride);

                    REQUIRE(hm_nat == hm_ref);
                    REQUIRE(hm_sca == hm_ref);
                    REQUIRE(hm_ns == hm_ref);
                    REQUIRE(c_nat == c_ref);
                    REQUIRE(c_sca == c_ref);
                    REQUIRE(c_ns == c_ref);
                    REQUIRE(m_nat == m_ref);
                    REQUIRE(m_sca == m_ref);
                    REQUIRE(m_ns == m_ref);
                }
            }
        }
    }

    // Sustained bright saturation test around `kKadaneInt32Cap`:
    // `(x << int_shift) == kKadaneInt32Cap` for 600 samples followed by 600
    // quiet samples must never overflow signed int32 and must match reference.
    {
        constexpr SizeType kRows = 19;
        constexpr SizeType kLen  = 257;
        constexpr uint8_t kShift = 2U;
        const uint32_t max_sample =
            static_cast<uint32_t>(kadane::kKadaneInt32Cap) >> kShift;
        std::vector<uint32_t> bright(kLen, max_sample);
        std::vector<uint32_t> quiet(kLen, 0U);
        std::vector<const uint32_t*> b_ptrs(kRows, bright.data());
        std::vector<const uint32_t*> q_ptrs(kRows, quiet.data());

        std::vector<int32_t> ibias(3 * kRows, 100000);
        std::vector<int32_t> icut(3 * kRows,
                                  std::numeric_limits<int32_t>::max() / 4);
        std::vector<int32_t> c_nat(3 * kRows, 0);
        std::vector<int32_t> c_ref(3 * kRows, 0);

        for (int rep = 0; rep < 3; ++rep) {
            const uint32_t hm_nat =
                kadane::scan_gate_tile_i32<3, NativeBackend>(
                    kRows, kLen, b_ptrs.data(), kShift, ibias.data(), kRows,
                    icut.data(), kRows, c_nat.data(), kRows);
            const uint32_t hm_ref = reference_gate_tile_i32(
                3, kRows, kLen, b_ptrs.data(), kShift, ibias.data(), kRows,
                icut.data(), kRows, c_ref.data(), kRows, nullptr, nullptr, 0,
                nullptr, 0);
            REQUIRE(hm_nat == kadane::active_row_mask(kRows));
            REQUIRE(hm_nat == hm_ref);
            REQUIRE(c_nat == c_ref);
            for (int32_t v : c_nat) {
                CHECK(v > kadane::kKadaneInt32Cap);
            }
        }

        // Decay back to 0 under quiet samples without underflow.
        for (int rep = 0; rep < 60; ++rep) {
            const uint32_t hm_nat =
                kadane::scan_gate_tile_i32<3, NativeBackend>(
                    kRows, kLen, q_ptrs.data(), kShift, ibias.data(), kRows,
                    icut.data(), kRows, c_nat.data(), kRows);
            const uint32_t hm_ref = reference_gate_tile_i32(
                3, kRows, kLen, q_ptrs.data(), kShift, ibias.data(), kRows,
                icut.data(), kRows, c_ref.data(), kRows, nullptr, nullptr, 0,
                nullptr, 0);
            REQUIRE(hm_nat == hm_ref);
            REQUIRE(c_nat == c_ref);
        }
        for (int32_t v : c_nat) {
            CHECK(v == 0);
        }
    }
}

// ============================================================================
// 5. Stage 1 `KadaneGatePlan` Integration: 0 Missed Detections Across Tiles
// ============================================================================

TEST_CASE("kadane_simd - KadaneGatePlan lossless f32 & i32 pulse detection",
          "[kadane_simd][kadane][boxcar][cpu]") {
    const std::array<uint32_t, 12> widths = {1,  2,  3,  4,  6,  8,
                                             12, 16, 24, 32, 48, 64};
    std::vector<std::pair<SizeType, SizeType>> row_specs(21);
    for (SizeType r = 0; r < row_specs.size(); ++r) {
        row_specs[r] = (r == 10) ? std::make_pair(SizeType{1}, SizeType{0})
                                 : std::make_pair(SizeType{1} + (r % 4),
                                                  SizeType{32} + (r * 4));
    }

    for (bool use_trapezoid : {false, true}) {
        // Float32 lossless gate integration across multi-tile boundaries + lag.
        {
            const auto tables = make_synthetic_tables(
                widths, use_trapezoid, 6.0F, 1.0F, row_specs, false);
            const auto plan = kadane::make_kadane_gate_plan(
                widths, use_trapezoid, tables.n_rows, tables.raw_thresh);
            REQUIRE(plan.num_buckets >= 2);

            constexpr SizeType kStreamLen = 520;
            constexpr SizeType kTileLen   = 64;
            const SizeType n_tiles = (kStreamLen + kTileLen - 1) / kTileLen;

            for (uint32_t w : {uint32_t{1}, uint32_t{5}, uint32_t{16},
                               uint32_t{48}, uint32_t{64}}) {
                const uint32_t eff_w =
                    std::find(widths.begin(), widths.end(), w) != widths.end()
                        ? w
                        : uint32_t{4};
                // Place pulse straddling tile boundary `t = 192` so trapezoid
                // right taper also exercises cross-tile `lag_max_in`!
                const SizeType pulse_start = 192 - (eff_w / 2);

                std::vector<std::vector<float>> streams(
                    tables.n_rows, std::vector<float>(kStreamLen, 0.0F));
                const SizeType target_row = 7;
                const SizeType f_idx      = static_cast<SizeType>(
                    std::find(widths.begin(), widths.end(), eff_w) -
                    widths.begin());
                const float thr =
                    tables.raw_thresh[(target_row * widths.size()) + f_idx];
                const float amp = (thr * 1.02F) / static_cast<float>(eff_w);
                for (SizeType t = pulse_start; t < pulse_start + eff_w; ++t) {
                    streams[target_row][t] = amp;
                }

                std::vector<float> state_c(plan.num_buckets * plan.row_stride,
                                           0.0F);
                std::vector<float> prev_max(plan.num_buckets * plan.row_stride,
                                            0.0F);
                std::vector<float> cur_max(plan.num_buckets * plan.row_stride,
                                           0.0F);

                for (SizeType ti = 0; ti < n_tiles; ++ti) {
                    const SizeType t0 = ti * kTileLen;
                    const SizeType tl = std::min(kTileLen, kStreamLen - t0);
                    std::vector<const float*> ptrs(tables.n_rows);
                    for (SizeType r = 0; r < tables.n_rows; ++r) {
                        ptrs[r] = streams[r].data() + t0;
                    }
                    const float* lag_ptr =
                        (use_trapezoid && ti > 0) ? prev_max.data() : nullptr;

                    // Scan in batches of 16 rows via `KadaneGatePlan` overload.
                    uint32_t all_hits = 0U;
                    for (SizeType r0 = 0; r0 < tables.n_rows; r0 += 16) {
                        const SizeType sub_n =
                            std::min<SizeType>(16, tables.n_rows - r0);
                        const uint32_t sub_hm =
                            kadane::scan_gate_tile_f32<NativeBackend>(
                                plan, r0, sub_n, tl, ptrs.data() + r0,
                                state_c.data() + r0, plan.row_stride, nullptr,
                                lag_ptr != nullptr ? lag_ptr + r0 : nullptr,
                                plan.row_stride, cur_max.data() + r0,
                                plan.row_stride);
                        all_hits |= (sub_hm << static_cast<uint32_t>(r0));
                    }
                    prev_max = cur_max;

                    // Dead row 10 must NEVER trigger.
                    CHECK(((all_hits >> 10U) & 1U) == 0U);

                    const std::span<const float> row_thr(
                        tables.raw_thresh.data() + (target_row * widths.size()),
                        widths.size());
                    const bool exact_hit =
                        exact_tile_has_hit_f32(streams[target_row], t0, tl,
                                               widths, use_trapezoid, row_thr);
                    const bool gate_hit =
                        ((all_hits >> static_cast<uint32_t>(target_row)) &
                         1U) != 0U;
                    if (exact_hit) {
                        CHECK(gate_hit);
                    }
                }
            }

            // `recompute_row_cuts` mid-stream continuity test without state
            // reset across a `BlockNoise` threshold drop.
            std::vector<float> lower_thr(widths.size());
            for (SizeType f = 0; f < widths.size(); ++f) {
                lower_thr[f] =
                    tables.raw_thresh[(3 * widths.size()) + f] * 0.85F;
            }
            std::array<float, kadane::kMaxGateBuckets> updated_cuts{};
            plan.recompute_row_cuts(
                3, lower_thr,
                std::span<float>(updated_cuts.data(), plan.num_buckets));
            for (SizeType k = 0; k < plan.num_buckets; ++k) {
                CHECK(updated_cuts[k] <= plan.row_cut(k, 3));
            }
        }

        // Packed uint8/uint16 (`int32_t`) lossless gate integration.
        {
            const auto tables = make_synthetic_tables(
                widths, use_trapezoid, 6.0F, 2.0F, row_specs, true, 64.0, 0);
            const auto plan = kadane::make_kadane_gate_plan(
                widths, use_trapezoid, tables.n_rows, tables.raw_thresh,
                tables.int_mean, tables.int_thresh, true, 4096ULL);
            REQUIRE(plan.has_int32);

            constexpr SizeType kStreamLen = 384;
            constexpr SizeType kTileLen   = 64;
            const SizeType n_tiles    = (kStreamLen + kTileLen - 1) / kTileLen;
            const SizeType target_row = 5;

            for (SizeType f_idx : {SizeType{0}, SizeType{4}, SizeType{9}}) {
                const uint32_t w       = widths[f_idx];
                const SizeType n_chans = row_specs[target_row].second;
                const auto base_val    = static_cast<uint32_t>(64 * n_chans);
                std::vector<std::vector<uint32_t>> streams(
                    tables.n_rows, std::vector<uint32_t>(kStreamLen, base_val));

                const uint64_t thr =
                    tables.int_thresh[(target_row * widths.size()) + f_idx];
                const uint64_t need_per_sample = (thr + w - 1U) / w + 4ULL;
                const SizeType p0              = 128 - (w / 2);
                for (SizeType t = p0; t < p0 + w; ++t) {
                    streams[target_row][t] =
                        static_cast<uint32_t>(need_per_sample);
                }

                std::vector<int32_t> state_c(plan.num_buckets * plan.row_stride,
                                             0);
                std::vector<int32_t> prev_max(
                    plan.num_buckets * plan.row_stride, 0);
                std::vector<int32_t> cur_max(plan.num_buckets * plan.row_stride,
                                             0);

                for (SizeType ti = 0; ti < n_tiles; ++ti) {
                    const SizeType t0 = ti * kTileLen;
                    const SizeType tl = std::min(kTileLen, kStreamLen - t0);
                    std::vector<const uint32_t*> ptrs(tables.n_rows);
                    for (SizeType r = 0; r < tables.n_rows; ++r) {
                        ptrs[r] = streams[r].data() + t0;
                    }
                    const int32_t* lag_ptr =
                        (use_trapezoid && ti > 0) ? prev_max.data() : nullptr;

                    const uint32_t hm =
                        kadane::scan_gate_tile_i32<NativeBackend>(
                            plan, 0, tables.n_rows, tl, ptrs.data(),
                            state_c.data(), plan.row_stride, nullptr, lag_ptr,
                            plan.row_stride, cur_max.data(), plan.row_stride);
                    prev_max = cur_max;

                    CHECK(((hm >> 10U) & 1U) == 0U);
                    const std::span<const uint64_t> row_int_thr(
                        tables.int_thresh.data() + (target_row * widths.size()),
                        widths.size());
                    if (exact_tile_has_hit_int(streams[target_row], t0, tl,
                                               widths, use_trapezoid,
                                               row_int_thr)) {
                        CHECK(((hm >> static_cast<uint32_t>(target_row)) &
                               1U) != 0U);
                    }
                }
            }
        }
    }
}

TEST_CASE("kadane_simd - multi-tile streaming state_c carry",
          "[kadane_simd][kadane][boxcar][cpu]") {
    // Multi-tile streaming `state_c` carry: splitting 513 samples into
    // unaligned tiles `[1, 3, 7, 15, 16, 17, 64, ...]` must yield bit-for-bit
    // identical final `state_c` and overall max to a single 513-sample tile!
    constexpr SizeType kRows = 23;
    constexpr SizeType kLen  = 513;
    auto rng                 = make_seeded_rng(13579246U);
    std::normal_distribution<float> f_dist(50.0F, 3.0F);
    std::uniform_int_distribution<uint32_t> u_dist(40U, 110U);

    std::vector<std::vector<float>> f_streams(kRows, std::vector<float>(kLen));
    std::vector<std::vector<uint32_t>> u_streams(kRows,
                                                 std::vector<uint32_t>(kLen));
    std::vector<const float*> f_full_ptrs(kRows);
    std::vector<const uint32_t*> u_full_ptrs(kRows);
    for (SizeType r = 0; r < kRows; ++r) {
        for (SizeType t = 0; t < kLen; ++t) {
            f_streams[r][t] = f_dist(rng);
            u_streams[r][t] = u_dist(rng);
        }
        f_full_ptrs[r] = f_streams[r].data();
        u_full_ptrs[r] = u_streams[r].data();
    }

    std::vector<float> f_bias(3 * kRows, 50.8F);
    std::vector<float> f_cut(3 * kRows, 12.0F);
    std::vector<int32_t> i_bias(3 * kRows, 77 * 4);
    std::vector<int32_t> i_cut(3 * kRows, 90 * 4);
    std::vector<SizeType> warmup(kRows);
    for (SizeType r = 0; r < kRows; ++r) {
        warmup[r] = (r * 7) % 45;
    }

    std::vector<float> fc_single(3 * kRows, 0.0F);
    std::vector<float> fm_single(3 * kRows, 0.0F);
    const uint32_t fhm_single = kadane::scan_gate_tile_f32<3, NativeBackend>(
        kRows, kLen, f_full_ptrs.data(), f_bias.data(), kRows, f_cut.data(),
        kRows, fc_single.data(), kRows, warmup.data(), nullptr, 0,
        fm_single.data(), kRows);

    std::vector<int32_t> ic_single(3 * kRows, 0);
    std::vector<int32_t> im_single(3 * kRows, 0);
    const uint32_t ihm_single = kadane::scan_gate_tile_i32<3, NativeBackend>(
        kRows, kLen, u_full_ptrs.data(), 2U, i_bias.data(), kRows, i_cut.data(),
        kRows, ic_single.data(), kRows, warmup.data(), nullptr, 0,
        im_single.data(), kRows);

    const std::array<SizeType, 7> steps = {1, 3, 7, 15, 16, 17, 64};
    std::vector<float> fc_tiled(3 * kRows, 0.0F);
    std::vector<float> fm_tiled(3 * kRows, 0.0F);
    std::vector<int32_t> ic_tiled(3 * kRows, 0);
    std::vector<int32_t> im_tiled(3 * kRows, 0);
    uint32_t fhm_tiled = 0U;
    uint32_t ihm_tiled = 0U;

    SizeType pos = 0;
    SizeType idx = 0;
    while (pos < kLen) {
        const SizeType tl = std::min(steps[idx % steps.size()], kLen - pos);
        ++idx;
        std::vector<const float*> fp(kRows);
        std::vector<const uint32_t*> up(kRows);
        std::vector<SizeType> fv(kRows, 0);
        for (SizeType r = 0; r < kRows; ++r) {
            fp[r] = f_streams[r].data() + pos;
            up[r] = u_streams[r].data() + pos;
            if (warmup[r] > pos) {
                fv[r] = warmup[r] - pos;
            }
        }
        std::vector<float> fm_step(3 * kRows, 0.0F);
        std::vector<int32_t> im_step(3 * kRows, 0);
        fhm_tiled |= kadane::scan_gate_tile_f32<3, NativeBackend>(
            kRows, tl, fp.data(), f_bias.data(), kRows, f_cut.data(), kRows,
            fc_tiled.data(), kRows, fv.data(), nullptr, 0, fm_step.data(),
            kRows);
        ihm_tiled |= kadane::scan_gate_tile_i32<3, NativeBackend>(
            kRows, tl, up.data(), 2U, i_bias.data(), kRows, i_cut.data(), kRows,
            ic_tiled.data(), kRows, fv.data(), nullptr, 0, im_step.data(),
            kRows);
        for (SizeType i = 0; i < 3 * kRows; ++i) {
            fm_tiled[i] =
                ScalarBackend::vmax_nonneg_f32(fm_tiled[i], fm_step[i]);
            im_tiled[i] = std::max(im_tiled[i], im_step[i]);
        }
        pos += tl;
    }

    REQUIRE(fhm_tiled == fhm_single);
    REQUIRE(ihm_tiled == ihm_single);
    REQUIRE(ic_tiled == ic_single);
    REQUIRE(im_tiled == im_single);
    for (SizeType i = 0; i < 3 * kRows; ++i) {
        REQUIRE(std::bit_cast<uint32_t>(fc_tiled[i]) ==
                std::bit_cast<uint32_t>(fc_single[i]));
        REQUIRE(std::bit_cast<uint32_t>(fm_tiled[i]) ==
                std::bit_cast<uint32_t>(fm_single[i]));
    }
}

} // namespace dmt
