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
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_variance.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/kadane_surrogate.hpp"

namespace dmt {
namespace {

using algorithms::detail::fdmt_trapezoid_variance;
using kadane::KadaneSurrogatePlan;
using kadane::KadaneSurrogateRowState;
using kadane::NativeBackend;
using kadane::ScalarBackend;
using kadane::SurrogateExcursion;

std::mt19937 make_seeded_rng(uint32_t seed) { return std::mt19937(seed); }

struct SyntheticPlaneTables {
    std::vector<uint32_t> widths;
    std::vector<float> raw_thresh;
    std::vector<float> inv_sigma;
    std::vector<double> int_mean;
    std::vector<uint64_t> int_thresh;
    SizeType n_rows     = 0;
    double integer_unit = 1.0;
};

SyntheticPlaneTables
make_synthetic_tables(std::span<const uint32_t> widths,
                      bool use_trapezoid,
                      float detect_thresh,
                      float noise_sigma,
                      std::span<const std::pair<SizeType, SizeType>> row_specs,
                      bool packed       = false,
                      double input_mean = 0.0,
                      int scale_bits    = 0) {
    SyntheticPlaneTables out;
    out.widths.assign(widths.begin(), widths.end());
    out.n_rows         = row_specs.size();
    out.integer_unit   = std::ldexp(1.0, -scale_bits);
    const double scale = std::ldexp(1.0, scale_bits);
    const SizeType nf  = widths.size();

    out.raw_thresh.resize(out.n_rows * nf);
    out.inv_sigma.resize(out.n_rows * nf);
    if (packed) {
        out.int_mean.resize(out.n_rows * nf);
        out.int_thresh.resize(out.n_rows * nf);
    }

    for (SizeType r = 0; r < out.n_rows; ++r) {
        const auto [leaf_len, n_chans] = row_specs[r];
        if (n_chans == 0) {
            for (SizeType f = 0; f < nf; ++f) {
                out.inv_sigma[(r * nf) + f] = 0.0F;
                out.raw_thresh[(r * nf) + f] =
                    std::numeric_limits<float>::max();
                if (packed) {
                    out.int_mean[(r * nf) + f]   = 0.0;
                    out.int_thresh[(r * nf) + f] = UINT64_MAX;
                }
            }
            continue;
        }
        const double weight      = 1.0 / static_cast<double>(leaf_len);
        const double mean_weight = static_cast<double>(n_chans) *
                                   static_cast<double>(leaf_len) * weight;
        for (SizeType f = 0; f < nf; ++f) {
            const uint32_t w = widths[f];
            const uint32_t we =
                use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
            const double var = static_cast<double>(n_chans) *
                               fdmt_trapezoid_variance(leaf_len, w, we, weight);
            const double sigma =
                static_cast<double>(noise_sigma) * std::sqrt(var);
            out.inv_sigma[(r * nf) + f] = static_cast<float>(1.0 / sigma);
            out.raw_thresh[(r * nf) + f] =
                static_cast<float>(static_cast<double>(detect_thresh) * sigma);
            if (packed) {
                const double fw = we == 0U ? static_cast<double>(w)
                                           : static_cast<double>(2U * (w - we));
                const double mean = input_mean * mean_weight * fw * scale;
                const double cutoff =
                    std::ceil(mean + (static_cast<double>(detect_thresh) *
                                      sigma * scale));
                out.int_mean[(r * nf) + f]   = mean;
                out.int_thresh[(r * nf) + f] = static_cast<uint64_t>(cutoff);
            }
        }
    }
    return out;
}

bool exact_tile_has_hit_f32(std::span<const float> stream,
                            SizeType t0,
                            SizeType tile_len,
                            std::span<const uint32_t> widths,
                            bool use_trapezoid,
                            std::span<const float> row_thr,
                            SizeType warmup = 0) {
    std::vector<double> prefix(stream.size() + 1, 0.0);
    for (SizeType i = 0; i < stream.size(); ++i) {
        prefix[i + 1] = prefix[i] + static_cast<double>(stream[i]);
    }
    const SizeType t_end = std::min(stream.size(), t0 + tile_len);
    for (SizeType f = 0; f < widths.size(); ++f) {
        const uint32_t w = widths[f];
        const uint32_t we =
            use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
        const auto th = static_cast<double>(row_thr[f]);
        for (SizeType t = t0; t < t_end; ++t) {
            if (t + 1 < static_cast<SizeType>(w) || t < warmup + w - 1) {
                continue;
            }
            double sum = prefix[t + 1] - prefix[t + 1 - w];
            if (we > 0U) {
                sum += prefix[t + 1 - we] - prefix[t + 1 - w + we];
            }
            if (sum >= th) {
                return true;
            }
        }
    }
    return false;
}

bool exact_tile_has_hit_int(std::span<const uint32_t> stream,
                            SizeType t0,
                            SizeType tile_len,
                            std::span<const uint32_t> widths,
                            bool use_trapezoid,
                            std::span<const uint64_t> row_int_thr,
                            SizeType warmup = 0) {
    std::vector<uint64_t> prefix(stream.size() + 1, 0ULL);
    for (SizeType i = 0; i < stream.size(); ++i) {
        prefix[i + 1] = prefix[i] + static_cast<uint64_t>(stream[i]);
    }
    const SizeType t_end = std::min(stream.size(), t0 + tile_len);
    for (SizeType f = 0; f < widths.size(); ++f) {
        const uint32_t w = widths[f];
        const uint32_t we =
            use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
        const uint64_t thr = row_int_thr[f];
        for (SizeType t = t0; t < t_end; ++t) {
            if (t + 1 < static_cast<SizeType>(w) || t < warmup + w - 1) {
                continue;
            }
            uint64_t sum = prefix[t + 1] - prefix[t + 1 - w];
            if (we > 0U) {
                sum += prefix[t + 1 - we] - prefix[t + 1 - w + we];
            }
            if (sum >= thr) {
                return true;
            }
        }
    }
    return false;
}

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

std::vector<SurrogateExcursion>
run_continuous_surrogate_reference(const KadaneSurrogatePlan& plan,
                                   SizeType row,
                                   std::span<const float> stream,
                                   uint64_t row_warmup = 0) {
    std::vector<SurrogateExcursion> out;
    KadaneSurrogateRowState st{};
    const auto emit = [&](const SurrogateExcursion& ex) { out.push_back(ex); };
    static_cast<void>(kadane::replay_surrogate_row(
        plan, row, 0ULL, stream.size(), stream.data(), st, emit, 0, nullptr,
        0.0F, {}, row_warmup));
    kadane::flush_surrogate_row(plan, row, st, emit);
    return out;
}

std::vector<SurrogateExcursion>
run_tiled_surrogate_pipeline(const KadaneSurrogatePlan& plan,
                             SizeType row,
                             std::span<const float> stream,
                             std::span<const SizeType> tile_lengths,
                             uint64_t row_warmup = 0) {
    const SizeType n_rows = plan.n_rows;
    const SizeType stride = plan.row_stride;
    const SizeType np     = plan.num_passes;

    std::vector<float> dummy_zero(stream.size(), 0.0F);
    std::vector<const float*> base_ptrs(n_rows, dummy_zero.data());
    base_ptrs[row] = stream.data();

    std::vector<float> state_c(np * stride, 0.0F);
    std::vector<float> state_c_scalar(np * stride, 0.0F);
    std::vector<float> chk_c(np * stride, 0.0F);
    std::vector<float> chk_c_scalar(np * stride, 0.0F);
    std::vector<float> dec_part(n_rows, 0.0F);
    std::vector<float> dec_part_scalar(n_rows, 0.0F);
    std::vector<float> chk_dec(n_rows, 0.0F);
    std::vector<float> chk_dec_scalar(n_rows, 0.0F);

    KadaneSurrogateRowState row_st{};
    bool open_triggered = false;
    std::vector<SurrogateExcursion> out;
    const auto emit = [&](const SurrogateExcursion& ex) { out.push_back(ex); };

    SizeType t0      = 0;
    SizeType len_idx = 0;
    while (t0 < stream.size()) {
        const SizeType step_len = std::min(
            tile_lengths[len_idx % tile_lengths.size()], stream.size() - t0);
        ++len_idx;

        std::vector<const float*> tile_ptrs(n_rows);
        for (SizeType r = 0; r < n_rows; ++r) {
            tile_ptrs[r] = base_ptrs[r] + t0;
        }
        std::vector<SizeType> fv_vec(n_rows, 0);
        if (row_warmup > t0) {
            fv_vec[row] = static_cast<SizeType>(row_warmup - t0);
        }

        const auto phase = static_cast<uint32_t>(t0 & 3ULL);
        const uint32_t hit_native =
            kadane::scan_surrogate_tile_f32<NativeBackend>(
                plan, 0, n_rows, step_len, tile_ptrs.data(), state_c.data(),
                stride, phase, dec_part.data(), fv_vec.data(), chk_c.data(),
                stride, chk_dec.data());
        const uint32_t hit_scalar =
            kadane::scan_surrogate_tile_f32<ScalarBackend>(
                plan, 0, n_rows, step_len, tile_ptrs.data(),
                state_c_scalar.data(), stride, phase, dec_part_scalar.data(),
                fv_vec.data(), chk_c_scalar.data(), stride,
                chk_dec_scalar.data());
        REQUIRE(hit_native == hit_scalar);
        for (SizeType k = 0; k < np; ++k) {
            REQUIRE(
                std::bit_cast<uint32_t>(state_c[(k * stride) + row]) ==
                std::bit_cast<uint32_t>(state_c_scalar[(k * stride) + row]));
            REQUIRE(std::bit_cast<uint32_t>(chk_c[(k * stride) + row]) ==
                    std::bit_cast<uint32_t>(chk_c_scalar[(k * stride) + row]));
        }
        REQUIRE(std::bit_cast<uint32_t>(dec_part[row]) ==
                std::bit_cast<uint32_t>(dec_part_scalar[row]));

        const bool row_hit =
            ((hit_native >> static_cast<uint32_t>(row)) & 1U) != 0U;
        if (row_hit || open_triggered) {
            std::array<float, kadane::kMaxSurrogatePasses> row_chk{};
            for (SizeType k = 0; k < np; ++k) {
                row_chk[k] = chk_c[(k * stride) + row];
            }
            const std::span<const float> lookback(stream.data(), t0);
            open_triggered = kadane::replay_surrogate_row(
                plan, row, static_cast<uint64_t>(t0), step_len,
                stream.data() + t0, row_st, emit, fv_vec[row], row_chk.data(),
                chk_dec[row], lookback, row_warmup);

            // Verify that after replay, every pass's `cur_c` matches the SIMD
            // tile scanner's updated `state_c` bit-for-bit!
            for (SizeType k = 0; k < np; ++k) {
                REQUIRE(std::bit_cast<uint32_t>(row_st.passes[k].cur_c) ==
                        std::bit_cast<uint32_t>(state_c[(k * stride) + row]));
            }
        }

        t0 += step_len;
    }

    kadane::flush_surrogate_row(plan, row, row_st, emit);
    return out;
}

} // namespace

// ============================================================================
// 1. ISA Dispatch, Bit Primitives & Unsigned Conversion
// ============================================================================

TEST_CASE("kadane_simd - ISA metadata, bitmasks, vmax_nonneg_f32 and u32->f32",
          "[kadane_simd][kadane][boxcar][cpu]") {
    CHECK(ScalarBackend::isa_name() == "scalar");
#if defined(DMT_KADANE_FORCE_SCALAR)
    CHECK(kadane::active_isa_name() == "scalar");
#elif defined(__AVX512F__)
    CHECK(kadane::active_isa_name() == "avx512f");
    CHECK(kadane::kSimdLanes == 16);
    CHECK(kadane::kBatchRows == 32);
#elif defined(__AVX2__)
    CHECK(kadane::active_isa_name() == "avx2");
    CHECK(kadane::kSimdLanes == 8);
    CHECK(kadane::kBatchRows == 16);
#elif defined(__ARM_NEON)
    CHECK(kadane::active_isa_name() == "neon");
    CHECK(kadane::kSimdLanes == 4);
    CHECK(kadane::kBatchRows == 16);
#else
    CHECK(kadane::active_isa_name() == "scalar");
#endif

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

// ============================================================================
// 6. Kernel 3 (`scan_surrogate_tile_f32` & `replay_surrogate_row`) Parity
// ============================================================================

TEST_CASE(
    "kadane_simd - scan_surrogate_tile_f32 bit-for-bit parity (dec 1 & 4)",
    "[kadane_simd][kadane][boxcar][cpu]") {
    auto rng = make_seeded_rng(424242U);
    std::normal_distribution<float> dist(10.0F, 1.5F);
    const std::array<SizeType, 9> tile_lens = {0,  1,  3,   7,  15,
                                               16, 17, 255, 256};
    const std::array<std::array<uint32_t, 3>, 3> dec_configs = {{
        {1U, 1U, 1U},
        {1U, 4U, 1U},
        {1U, 1U, 4U},
    }};

    for (SizeType num_passes = 1; num_passes <= 3; ++num_passes) {
        const auto& dec = dec_configs[num_passes - 1];
        for (uint32_t phase : {0U, 1U, 2U, 3U}) {
            for (SizeType n_rows : {SizeType{1}, SizeType{3}, SizeType{4},
                                    SizeType{8}, SizeType{15}, SizeType{16},
                                    SizeType{17}, SizeType{31}, SizeType{32}}) {
                for (SizeType tile_len : tile_lens) {
                    if (phase != 1U && tile_len > 17 && n_rows != 17) {
                        continue;
                    }

                    std::vector<std::vector<float>> rows(
                        n_rows,
                        std::vector<float>(std::max<SizeType>(1, tile_len)));
                    std::vector<const float*> ptrs(n_rows);
                    for (SizeType r = 0; r < n_rows; ++r) {
                        for (SizeType t = 0; t < tile_len; ++t) {
                            rows[r][t] = dist(rng);
                        }
                        ptrs[r] = rows[r].data();
                    }

                    const SizeType stride = n_rows;
                    std::vector<float> bias(num_passes * stride);
                    std::vector<float> cut(num_passes * stride);
                    std::vector<float> c_init(num_passes * stride);
                    std::vector<float> dec_init(n_rows);
                    std::vector<SizeType> fv(n_rows);

                    for (SizeType r = 0; r < n_rows; ++r) {
                        fv[r]       = (r % 3 == 0) ? 0 : ((r * 3) % 14);
                        dec_init[r] = 20.0F + static_cast<float>(r);
                        for (SizeType k = 0; k < num_passes; ++k) {
                            bias[(k * stride) + r] =
                                dec[k] == 4U ? 42.0F : 10.5F;
                            cut[(k * stride) + r] =
                                5.0F + (2.0F * static_cast<float>(k));
                            c_init[(k * stride) + r] =
                                (r % 2 == 0) ? 0.0F : 1.5F;
                        }
                    }

                    auto c_nat  = c_init;
                    auto c_sca  = c_init;
                    auto c_ns   = c_init;
                    auto dp_nat = dec_init;
                    auto dp_sca = dec_init;
                    auto dp_ns  = dec_init;
                    std::vector<float> chkc_nat(num_passes * stride, -1.0F);
                    std::vector<float> chkc_sca(num_passes * stride, -2.0F);
                    std::vector<float> chkd_nat(n_rows, -1.0F);
                    std::vector<float> chkd_sca(n_rows, -2.0F);

                    const SizeType* fv_ptr =
                        (phase % 2 == 1U) ? fv.data() : nullptr;

                    const uint32_t hm_nat =
                        kadane::scan_surrogate_tile_f32<NativeBackend>(
                            num_passes, n_rows, tile_len, ptrs.data(),
                            dec.data(), bias.data(), stride, cut.data(), stride,
                            c_nat.data(), stride, phase, dp_nat.data(), fv_ptr,
                            chkc_nat.data(), stride, chkd_nat.data());
                    const uint32_t hm_sca =
                        kadane::scan_surrogate_tile_f32<ScalarBackend>(
                            num_passes, n_rows, tile_len, ptrs.data(),
                            dec.data(), bias.data(), stride, cut.data(), stride,
                            c_sca.data(), stride, phase, dp_sca.data(), fv_ptr,
                            chkc_sca.data(), stride, chkd_sca.data());
                    const uint32_t hm_ns =
                        kadane::scalar::scan_surrogate_tile_f32(
                            num_passes, n_rows, tile_len, ptrs.data(),
                            dec.data(), bias.data(), stride, cut.data(), stride,
                            c_ns.data(), stride, phase, dp_ns.data(), fv_ptr,
                            nullptr, 0, nullptr);

                    REQUIRE(hm_nat == hm_sca);
                    REQUIRE(hm_ns == hm_sca);
                    for (SizeType i = 0; i < num_passes * stride; ++i) {
                        REQUIRE(std::bit_cast<uint32_t>(c_nat[i]) ==
                                std::bit_cast<uint32_t>(c_sca[i]));
                        REQUIRE(std::bit_cast<uint32_t>(c_ns[i]) ==
                                std::bit_cast<uint32_t>(c_sca[i]));
                        REQUIRE(std::bit_cast<uint32_t>(chkc_nat[i]) ==
                                std::bit_cast<uint32_t>(chkc_sca[i]));
                    }
                    for (SizeType r = 0; r < n_rows; ++r) {
                        REQUIRE(std::bit_cast<uint32_t>(dp_nat[r]) ==
                                std::bit_cast<uint32_t>(dp_sca[r]));
                        REQUIRE(std::bit_cast<uint32_t>(dp_ns[r]) ==
                                std::bit_cast<uint32_t>(dp_sca[r]));
                        REQUIRE(std::bit_cast<uint32_t>(chkd_nat[r]) ==
                                std::bit_cast<uint32_t>(chkd_sca[r]));
                    }
                }
            }
        }
    }
}

TEST_CASE("kadane_simd - KadaneSurrogatePlan tiled SIMD + replay exact parity",
          "[kadane_simd][kadane][boxcar][cpu]") {
    const std::array<uint32_t, 16> widths = {1,  2,  3,  4,  6,  8,   12,  16,
                                             24, 32, 48, 64, 96, 128, 192, 256};
    const std::array<std::pair<SizeType, SizeType>, 9> row_specs = {{
        {1, 64},
        {2, 64},
        {3, 64},
        {4, 64},
        {1, 0},
        {2, 128},
        {5, 96},
        {1, 32},
        {3, 48},
    }};
    const auto tables =
        make_synthetic_tables(widths, false, 6.0F, 1.0F, row_specs, false);

    for (bool allow_decimation : {false, true}) {
        const auto plan = kadane::make_kadane_surrogate_plan(
            widths, false, tables.n_rows, tables.raw_thresh, tables.inv_sigma,
            {}, 1.0, allow_decimation);
        if (allow_decimation) {
            REQUIRE(plan.has_decimated);
        }

        const SizeType row      = 2;
        constexpr SizeType kLen = 1400;
        std::mt19937 rng(allow_decimation ? 777111U : 333999U);
        std::normal_distribution<float> bg(0.0F, 0.35F);
        std::vector<float> stream(kLen);
        for (float& v : stream) {
            v = bg(rng);
        }

        // 1) Adversarial fledgling ramp starting 110 samples before `t = 256`
        // with ULP-absorption jump at `t = 250`, crossing trigger after `t =
        // 256`!
        const float b0   = plan.row_bias(0, row);
        const float trig = plan.row_trigger(0, row);
        for (SizeType t = 130; t < 146; ++t) {
            stream[t] = -50.0F; // Guaranteed zero reset before `t = 146`.
        }
        stream[146] = b0 + 1.0e-6F;
        for (SizeType t = 147; t < 250; ++t) {
            stream[t] = b0;
        }
        stream[250] = b0 + (0.55F * trig);
        for (SizeType t = 251; t < 256; ++t) {
            stream[t] = b0;
        }
        for (SizeType t = 256; t < 264; ++t) {
            stream[t] = b0 + (0.35F * trig);
        }
        for (SizeType t = 264; t < 280; ++t) {
            stream[t] = -40.0F;
        }

        // 2) Wide pulse (`w = 96`) spanning multiple tiles `[610, 706)` to
        // exercise the `decimate_factor == 4` pass across arbitrary phases.
        const SizeType f_wide = 12; // width 96
        const float wide_thr =
            tables.raw_thresh[(row * widths.size()) + f_wide];
        const float wide_amp = (1.35F * wide_thr) / 96.0F;
        for (SizeType t = 610; t < 706; ++t) {
            stream[t] += wide_amp;
        }

        // 3) Open excursion extending right to end-of-stream
        // (`flush_surrogate_row`).
        for (SizeType t = 1340; t < kLen; ++t) {
            stream[t] = b0 + (0.25F * trig);
        }

        for (uint64_t warmup : {uint64_t{0}, uint64_t{19}}) {
            const auto expected =
                run_continuous_surrogate_reference(plan, row, stream, warmup);
            REQUIRE(expected.size() >= 2);

            // Verify evaluated S/N exceeds threshold on injected pulses!
            bool saw_wide_snr = false;
            for (const auto& ex : expected) {
                uint32_t f_nearest = 0U;
                const float snr    = plan.eval_surrogate_snr(
                    row, ex.best_raw_sum, ex.best_len, &f_nearest);
                if (ex.seg_begin >= 580ULL && ex.best_end <= 740ULL &&
                    snr >= 6.0F) {
                    saw_wide_snr = true;
                }
            }
            CHECK(saw_wide_snr);

            const std::array<std::vector<SizeType>, 5> schedules = {{
                {1},
                {7},
                {16},
                {64},
                {3, 17, 64, 15, 255, 16, 1},
            }};

            for (const auto& sched : schedules) {
                const auto actual = run_tiled_surrogate_pipeline(
                    plan, row, stream, sched, warmup);
                REQUIRE(actual.size() == expected.size());
                for (SizeType i = 0; i < expected.size(); ++i) {
                    CHECK(actual[i].seg_begin == expected[i].seg_begin);
                    CHECK(actual[i].best_end == expected[i].best_end);
                    CHECK(actual[i].close_end == expected[i].close_end);
                    CHECK(actual[i].best_len == expected[i].best_len);
                    CHECK(actual[i].pass_idx == expected[i].pass_idx);
                    CHECK(std::bit_cast<uint32_t>(actual[i].best_c) ==
                          std::bit_cast<uint32_t>(expected[i].best_c));
                    CHECK_THAT(actual[i].best_raw_sum,
                               Catch::Matchers::WithinRel(
                                   expected[i].best_raw_sum, 1e-12) ||
                                   Catch::Matchers::WithinAbs(
                                       expected[i].best_raw_sum, 1e-12));
                }
            }
        }
    }
}

TEST_CASE(
    "kadane_simd - multi-tile streaming state_c carry & integer surrogate",
    "[kadane_simd][kadane][boxcar][cpu]") {
    // 1) Multi-tile streaming `state_c` carry: splitting 513 samples into
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

    // 2) Integer surrogate row integration via `convert_u32_to_f32` +
    // `KadaneSurrogatePlan::eval_surrogate_snr`.
    const std::array<uint32_t, 8> widths = {1, 2, 4, 8, 16, 32, 64, 128};
    const std::array<std::pair<SizeType, SizeType>, 4> row_specs = {{
        {1, 32},
        {2, 32},
        {3, 32},
        {4, 32},
    }};
    const auto tables = make_synthetic_tables(widths, false, 6.0F, 2.0F,
                                              row_specs, true, 50.0, 0);
    const auto plan   = kadane::make_kadane_surrogate_plan(
        widths, false, tables.n_rows, tables.raw_thresh, tables.inv_sigma,
        tables.int_mean, tables.integer_unit, true);
    REQUIRE(plan.is_integer);

    std::vector<uint32_t> u_row(512, 50U * 32U);
    for (SizeType t = 200; t < 224; ++t) {
        u_row[t] += 45U;
    }
    std::vector<float> f_row(512);
    kadane::convert_u32_to_f32<NativeBackend>(u_row.data(), f_row.data(),
                                              u_row.size());
    const std::array<SizeType, 2> sched = {64, 17};
    const auto exs = run_tiled_surrogate_pipeline(plan, 0, f_row, sched, 0);
    REQUIRE(!exs.empty());
    bool detected = false;
    for (const auto& ex : exs) {
        if (plan.eval_surrogate_snr(0, ex.best_raw_sum, ex.best_len) >= 6.0F) {
            detected = true;
        }
    }
    CHECK(detected);
}

} // namespace dmt
