#include "dmt/kadane_plan.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/fdmt_variance.hpp"

namespace dmt {
namespace {

using algorithms::FDMTBoxcar;
using algorithms::detail::fdmt_trapezoid_variance;
using kadane::KadaneGatePlan;
using kadane::KadaneSurrogatePlan;

struct SyntheticPlaneTables {
    std::vector<uint32_t> widths;
    std::vector<float> raw_thresh;
    std::vector<float> inv_sigma;
    std::vector<double> int_mean;
    std::vector<uint64_t> int_thresh;
    SizeType n_rows     = 0;
    double integer_unit = 1.0;
};

/// Build exact analytical row tables matching `make_boxcar_plans()` for rows
/// with specified `(leaf_len, n_active_chans, scale_bits)`.
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

/// Exact Stage-2 reference check on tile `[t0, t0 + tile_len)` using `double`
/// prefix sums, matching `FDMTBoxcarCpuEngine::score_tile()`.
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

/// Simulate float32 Kadane gate over tiles of `tile_len` samples, returning
/// per-tile trigger flags. Holds `c = 0` for `t < warmup`.
std::vector<bool> run_kadane_gate_tiles_f32(const KadaneGatePlan& plan,
                                            SizeType row,
                                            std::span<const float> stream,
                                            SizeType tile_len,
                                            SizeType warmup = 0) {
    const SizeType n_tiles = (stream.size() + tile_len - 1) / tile_len;
    std::vector<bool> hits(n_tiles, false);
    if (plan.disabled_row[row] != 0U) {
        return hits;
    }

    const SizeType nb = plan.num_buckets;
    std::vector<std::vector<float>> c_hist(
        nb, std::vector<float>(stream.size(), 0.0F));
    for (SizeType k = 0; k < nb; ++k) {
        const float b = plan.row_bias(k, row);
        float c       = 0.0F;
        for (SizeType t = 0; t < stream.size(); ++t) {
            if (t < warmup) {
                c = 0.0F;
            } else {
                c = std::max(0.0F, c + (stream[t] - b));
            }
            c_hist[k][t] = c;
        }
    }

    for (SizeType ti = 0; ti < n_tiles; ++ti) {
        const SizeType t0 = ti * tile_len;
        const SizeType t1 = std::min(stream.size(), t0 + tile_len);
        bool triggered    = false;
        for (SizeType k = 0; k < nb && !triggered; ++k) {
            const uint32_t lag  = plan.max_edge_lag[k];
            const SizeType u_lo = t0 > lag ? t0 - lag : 0;
            float max_c         = 0.0F;
            for (SizeType u = u_lo; u < t1; ++u) {
                max_c = std::max(max_c, c_hist[k][u]);
            }
            if (max_c >= plan.row_cut(k, row)) {
                triggered = true;
            }
        }
        hits[ti] = triggered;
    }
    return hits;
}

/// Exact Stage-2 integer reference check on tile `[t0, t0 + tile_len)`.
bool exact_tile_has_hit_int(std::span<const uint32_t> stream,
                            SizeType t0,
                            SizeType tile_len,
                            std::span<const uint32_t> widths,
                            bool use_trapezoid,
                            std::span<const uint64_t> row_int_thr) {
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
            if (t + 1 < static_cast<SizeType>(w)) {
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

/// Simulate exact int32_t Kadane gate over tiles of `tile_len` samples.
std::vector<bool> run_kadane_gate_tiles_i32(const KadaneGatePlan& plan,
                                            SizeType row,
                                            std::span<const uint32_t> stream,
                                            SizeType tile_len) {
    const SizeType n_tiles = (stream.size() + tile_len - 1) / tile_len;
    std::vector<bool> hits(n_tiles, false);
    if (!plan.has_int32 || plan.disabled_row[row] != 0U) {
        return hits;
    }
    const SizeType nb = plan.num_buckets;
    std::vector<std::vector<int32_t>> c_hist(
        nb, std::vector<int32_t>(stream.size(), 0));
    for (SizeType k = 0; k < nb; ++k) {
        const int32_t b = plan.row_ibias(k, row);
        int32_t c       = 0;
        for (SizeType t = 0; t < stream.size(); ++t) {
            const uint32_t raw_sh = stream[t] << plan.int_shift;
            REQUIRE(raw_sh <= static_cast<uint32_t>(kadane::kKadaneInt32Cap));
            const int32_t delta   = static_cast<int32_t>(raw_sh) - b;
            const int32_t clamped = std::min(c, kadane::kKadaneInt32Cap);
            const int64_t check =
                static_cast<int64_t>(clamped) + static_cast<int64_t>(delta);
            REQUIRE(check >=
                    static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
            REQUIRE(check <=
                    static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
            c            = std::max<int32_t>(0, clamped + delta);
            c_hist[k][t] = c;
        }
    }
    for (SizeType ti = 0; ti < n_tiles; ++ti) {
        const SizeType t0 = ti * tile_len;
        const SizeType t1 = std::min(stream.size(), t0 + tile_len);
        bool triggered    = false;
        for (SizeType k = 0; k < nb && !triggered; ++k) {
            const uint32_t lag  = plan.max_edge_lag[k];
            const SizeType u_lo = t0 > lag ? t0 - lag : 0;
            int32_t max_c       = 0;
            for (SizeType u = u_lo; u < t1; ++u) {
                max_c = std::max(max_c, c_hist[k][u]);
            }
            if (max_c >= plan.row_icut(k, row)) {
                triggered = true;
            }
        }
        hits[ti] = triggered;
    }
    return hits;
}

std::mt19937 make_seeded_rng(uint32_t seed) {
    std::seed_seq seq{seed, seed ^ 0x9E3779B9U};
    return std::mt19937(seq);
}

} // namespace

TEST_CASE("BoxcarConfig screen_mode defaults to kPrefix and coexists with "
          "OverflowPolicy and wide boxcars",
          "[kadane_plan][kadane][boxcar][cpu]") {
    static_assert(kadane::trapezoid_edge_width(1) == 0U);
    static_assert(kadane::trapezoid_edge_width(2) == 0U);
    static_assert(kadane::trapezoid_edge_width(3) == 1U);
    static_assert(kadane::trapezoid_edge_width(4096) == 1024U);
    static_assert(kadane::trapezoid_edge_width(1048576) == 262144U);
    for (uint32_t w = 1U; w <= 4096U; ++w) {
        REQUIRE(kadane::trapezoid_edge_width(w) ==
                algorithms::detail::trapezoid_edge_width(w));
    }

    const BoxcarConfig default_cfg{};
    CHECK(default_cfg.overflow_policy == OverflowPolicy::kDropNew);
    CHECK(default_cfg.screen_mode == BoxcarScreenMode::kPrefix);

    // 11-argument construction stays 100% source-compatible with kDropNew +
    // kPrefix.
    const BoxcarConfig compat11_cfg(
        128, 1.5F, 6.0F, 1.0F, PeakFindingMode::kContiguousPeakAcrossWidths,
        8192, 256, true, 32, 0.0F, true);
    CHECK(compat11_cfg.overflow_policy == OverflowPolicy::kDropNew);
    CHECK(compat11_cfg.screen_mode == BoxcarScreenMode::kPrefix);
    CHECK(compat11_cfg.use_trapezoid);

    // 12-argument construction stays 100% source-compatible with OverflowPolicy
    // as the 12th parameter and screen_mode defaulting to kPrefix.
    const BoxcarConfig compat12_cfg(
        256, 2.0F, 6.0F, 1.0F, PeakFindingMode::kContiguousPeakPerWidth, 8192,
        16, true, 32, 0.0F, false, OverflowPolicy::kKeepBrightest);
    CHECK(compat12_cfg.overflow_policy == OverflowPolicy::kKeepBrightest);
    CHECK(compat12_cfg.screen_mode == BoxcarScreenMode::kPrefix);

    // 13-argument construction sets both overflow_policy (12th) and
    // screen_mode (13th).
    for (const auto policy :
         {OverflowPolicy::kDropNew, OverflowPolicy::kKeepBrightest}) {
        for (const auto mode :
             {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kKadane,
              BoxcarScreenMode::kKadaneSurrogate, BoxcarScreenMode::kNone}) {
            const BoxcarConfig cfg(
                256, 2.0F, 6.5F, 1.2F, PeakFindingMode::kContiguousPeakPerWidth,
                4096, 16, false, 32, 0.0F, false, policy, mode);
            CHECK(cfg.overflow_policy == policy);
            CHECK(cfg.screen_mode == mode);
            CHECK(cfg.max_boxcar == 256);
            CHECK(cfg.detect_thresh == 6.5F);
        }
    }

    // Plan / FDMTBoxcar validation:
    // - Rejects invalid screen_mode enumerator.
    // - Rejects kKadaneSurrogate when max_boxcar > 4096, accepts <= 4096.
    // - Accepts kPrefix, kKadane, and kNone up to 2^20 (1048576).
    // - Preserves OverflowPolicy::kKeepBrightest coexistence and validation.
    const plans::FDMTPlan fdmt_plan(1200.0F, 1500.0F, 8, 256, 0.001F, 10.0F);
    BoxcarConfig bad_cfg{};
    bad_cfg.max_boxcar  = 32;
    bad_cfg.screen_mode = static_cast<BoxcarScreenMode>(99);
    CHECK_THROWS_AS(FDMTBoxcar(fdmt_plan, bad_cfg), std::invalid_argument);

    for (const SizeType bad_surr_w :
         {SizeType{4097}, SizeType{16384}, SizeType{1048576}}) {
        BoxcarConfig surr_wide{};
        surr_wide.max_boxcar  = bad_surr_w;
        surr_wide.wtsp        = 2.0F;
        surr_wide.screen_mode = BoxcarScreenMode::kKadaneSurrogate;
        CHECK_THROWS_AS(FDMTBoxcar(fdmt_plan, surr_wide),
                        std::invalid_argument);
    }
    {
        BoxcarConfig surr_ok{};
        surr_ok.max_boxcar       = 4096;
        surr_ok.wtsp             = 2.0F;
        surr_ok.max_cands_per_dm = 16;
        surr_ok.overflow_policy  = OverflowPolicy::kKeepBrightest;
        surr_ok.screen_mode      = BoxcarScreenMode::kKadaneSurrogate;
        const FDMTBoxcar ok_boxcar(fdmt_plan, surr_ok);
        CHECK(ok_boxcar.get_search_plans().back().widths.back() == 4096U);
    }

    for (const auto wide_mode :
         {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kKadane,
          BoxcarScreenMode::kNone}) {
        BoxcarConfig wide_cfg{};
        wide_cfg.max_boxcar       = 1048576;
        wide_cfg.wtsp             = 2.0F;
        wide_cfg.max_cands_per_dm = 16;
        wide_cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
        wide_cfg.screen_mode      = wide_mode;
        const FDMTBoxcar wide_boxcar(fdmt_plan, wide_cfg);
        const auto& wp = wide_boxcar.get_search_plans().back();
        CHECK(wp.widths.back() == 1048576U);
        CHECK(wp.widths.size() <= 4096);

        // kKeepBrightest per-row cap (<= 1024) still enforced for every mode.
        wide_cfg.max_boxcar       = 256;
        wide_cfg.max_cands_per_dm = 4096;
        CHECK_THROWS_AS(FDMTBoxcar(fdmt_plan, wide_cfg), std::invalid_argument);
    }
}

TEST_CASE("KadaneGatePlan algebraic soundness across widths, thresholds, and "
          "FDMT noise models",
          "[kadane_plan][kadane][boxcar][cpu]") {
    const std::array<std::pair<SizeType, SizeType>, 4> row_specs{{
        {1, 64},   // White noise
        {4, 64},   // Mild FDMT box smearing
        {16, 128}, // Moderate FDMT box smearing
        {64, 256}, // Heavy FDMT box smearing (convex small-w regime)
    }};

    for (const SizeType max_boxcar : {SizeType{1}, SizeType{2}, SizeType{16},
                                      SizeType{200}, SizeType{4096}}) {
        for (const float wtsp : {2.0F, 1.5F, 1.0F}) {
            if (wtsp == 1.0F && max_boxcar == 200) {
                continue; // 4096 at wtsp=1.0 already covers dense bank
            }
            const auto widths =
                plans::generate_box_width_trials(max_boxcar, wtsp);
            for (const bool use_trap : {false, true}) {
                for (const float u : {4.5F, 6.0F, 8.0F, 10.0F}) {
                    const auto tables = make_synthetic_tables(
                        widths, use_trap, u, 1.25F, row_specs);
                    const auto plan = kadane::make_kadane_gate_plan(
                        widths, use_trap, tables.n_rows, tables.raw_thresh);

                    REQUIRE(plan.num_buckets >= 1);
                    REQUIRE(plan.num_buckets <= kadane::kMaxGateBuckets);
                    CHECK(plan.buckets[0].f_lo == 0U);
                    CHECK(plan.buckets[plan.num_buckets - 1].f_hi ==
                          widths.size());
                    for (SizeType k = 1; k < plan.num_buckets; ++k) {
                        CHECK(plan.buckets[k].f_lo == plan.buckets[k - 1].f_hi);
                    }

                    // Verify w = 1, 2 never introduce nonzero trapezoid lag or
                    // factor-of-2 threshold halving.
                    if (max_boxcar <= 2) {
                        CHECK(plan.max_edge_lag[0] == 0U);
                    }

                    const SizeType nf = widths.size();
                    for (SizeType r = 0; r < tables.n_rows; ++r) {
                        REQUIRE(plan.disabled_row[r] == 0U);
                        for (SizeType k = 0; k < plan.num_buckets; ++k) {
                            const float b   = plan.row_bias(k, r);
                            const float cut = plan.row_cut(k, r);
                            REQUIRE(b > 0.0F);
                            REQUIRE(cut >= 0.0F);
                            for (uint32_t f = plan.buckets[k].f_lo;
                                 f < plan.buckets[k].f_hi; ++f) {
                                const uint32_t w = widths[f];
                                const uint32_t we =
                                    use_trap ? kadane::trapezoid_edge_width(w)
                                             : 0U;
                                const double thr =
                                    tables.raw_thresh[(r * nf) + f];
                                const double exact_bound =
                                    we == 0U
                                        ? (thr - (static_cast<double>(b) * w))
                                        : (0.5 *
                                           (thr - (static_cast<double>(b) *
                                                   (2U * (w - we)))));
                                REQUIRE(static_cast<double>(cut) < exact_bound);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("KadaneGatePlan 0-false-negative pulse & Monte Carlo soundness",
          "[kadane_plan][kadane][boxcar][cpu]") {
    constexpr SizeType kTileLen = 256;
    auto rng                    = make_seeded_rng(20261112U);
    std::normal_distribution<float> norm(0.0F, 1.0F);

    // 1. Test real FDMTBoxcar planes (both smeared & unsmeared, boxcar & trap).
    const plans::FDMTPlan fdmt_plan(1200.0F, 1500.0F, 32, 1024, 0.001F, 120, 0,
                                    1, "valid", 1);
    for (const bool smear : {false, true}) {
        for (const bool use_trap : {false, true}) {
            BoxcarConfig cfg{};
            cfg.max_boxcar    = 512;
            cfg.wtsp          = 1.5F;
            cfg.detect_thresh = 6.0F;
            cfg.use_trapezoid = use_trap;
            FDMTBoxcar boxcar(fdmt_plan, cfg, smear);
            const auto& plane = boxcar.get_search_plans().back();
            const SizeType nr = plane.dm_grid.size();
            const SizeType nf = plane.widths.size();

            const KadaneGatePlan gate(plane.widths, use_trap, nr,
                                      plane.raw_thresh_lut);

            // Pick low, mid, and high DM rows and inject pulses at 1.0002x
            // threshold across all widths in the bank, straddling tile edges.
            const std::array<SizeType, 3> test_rows{0, nr / 2, nr - 1};
            for (const SizeType r : test_rows) {
                const auto row_thr = std::span<const float>(
                    plane.raw_thresh_lut.data() + (r * nf), nf);
                const float sigma_1 =
                    1.0F / plane.inv_sigma_lut[r * nf]; // w=1 sigma

                for (SizeType f = 0; f < nf; ++f) {
                    const uint32_t w = plane.widths[f];
                    const uint32_t we =
                        use_trap ? kadane::trapezoid_edge_width(w) : 0U;
                    const SizeType stream_len = 3 * kTileLen;
                    std::vector<float> stream(stream_len, 0.0F);
                    for (float& x : stream) {
                        x = 0.15F * sigma_1 * norm(rng);
                    }

                    // Place endpoint just inside tile 2 so wide windows
                    // straddle tile 0/1/2 boundaries and trapezoid lags.
                    const SizeType end_t = (2 * kTileLen) + (f % 17);
                    const SizeType beg_t = end_t + 1 - w;

                    // Compute current filter sum and top up to 1.0005 * thr.
                    double cur_sum = 0.0;
                    for (SizeType t = beg_t; t <= end_t; ++t) {
                        cur_sum += static_cast<double>(stream[t]);
                    }
                    if (we > 0U) {
                        for (SizeType t = beg_t + we; t <= end_t - we; ++t) {
                            cur_sum += static_cast<double>(stream[t]);
                        }
                    }
                    const double weight_sum =
                        we == 0U ? static_cast<double>(w)
                                 : static_cast<double>(2U * (w - we));
                    const double target =
                        static_cast<double>(row_thr[f]) * 1.0005;
                    const auto delta =
                        static_cast<float>((target - cur_sum) / weight_sum);
                    for (SizeType t = beg_t; t <= end_t; ++t) {
                        stream[t] += delta;
                    }

                    const SizeType warmup = 12;
                    const auto gate_hits  = run_kadane_gate_tiles_f32(
                        gate, r, stream, kTileLen, warmup);
                    for (SizeType ti = 0; ti < gate_hits.size(); ++ti) {
                        const bool exact_hit = exact_tile_has_hit_f32(
                            stream, ti * kTileLen, kTileLen, plane.widths,
                            use_trap, row_thr, warmup);
                        if (exact_hit) {
                            REQUIRE(gate_hits[ti]);
                        }
                    }
                }
            }
        }
    }

    // 2. Worst-case 4096-step constant-pulse float32 rounding check at W=4096.
    const auto widths_4096 = plans::generate_box_width_trials(4096, 1.5F);
    const std::array<std::pair<SizeType, SizeType>, 1> white_spec{{{1, 1}}};
    for (const bool use_trap : {false, true}) {
        const auto tab = make_synthetic_tables(widths_4096, use_trap, 10.0F,
                                               1.0F, white_spec);
        const KadaneGatePlan gate(widths_4096, use_trap, 1, tab.raw_thresh);
        const SizeType stream_len = 4096 + (2 * kTileLen);
        std::vector<float> stream(stream_len, 0.0F);
        const SizeType end_t = stream_len - 37;
        const SizeType beg_t = end_t + 1 - 4096;
        const uint32_t we = use_trap ? kadane::trapezoid_edge_width(4096) : 0U;
        const double denom =
            we == 0U ? 4096.0 : static_cast<double>(2U * (4096U - we));
        const float thr_max = tab.raw_thresh.back();
        const float amp =
            std::bit_cast<float>(std::bit_cast<uint32_t>(static_cast<float>(
                                     static_cast<double>(thr_max) / denom)) +
                                 2U);
        for (SizeType t = beg_t; t <= end_t; ++t) {
            stream[t] = amp;
        }
        const auto gate_hits =
            run_kadane_gate_tiles_f32(gate, 0, stream, kTileLen);
        bool any_exact = false;
        for (SizeType ti = 0; ti < gate_hits.size(); ++ti) {
            const bool exact_hit =
                exact_tile_has_hit_f32(stream, ti * kTileLen, kTileLen,
                                       widths_4096, use_trap, tab.raw_thresh);
            if (exact_hit) {
                any_exact = true;
                REQUIRE(gate_hits[ti]);
            }
        }
        REQUIRE(any_exact);
    }
}

TEST_CASE("KadaneGatePlan packed 1/2/4/8/16-bit int32 gate, dead rows, and "
          "mixed-mask cut recomputation",
          "[kadane_plan][kadane][boxcar][cpu]") {
    constexpr SizeType kTileLen = 256;
    auto rng                    = make_seeded_rng(987654U);

    // 1. Packed 1, 2, 4, 8, 16-bit tables across boxcar and {1,2,1} trapezoid,
    // both unsmeared and dyadic-smeared.
    for (const SizeType nbits :
         {SizeType{1}, SizeType{2}, SizeType{4}, SizeType{8}, SizeType{16}}) {
        const uint32_t maxval = (1U << nbits) - 1U;
        const SizeType nchans = nbits == 16 ? SizeType{4} : SizeType{256};
        const double mean_in  = 0.5 * static_cast<double>(maxval);
        const float sigma_in =
            std::max(0.5F, 0.15F * static_cast<float>(maxval));

        for (const int scale_bits : {0, 5}) {
            if (nbits == 16 && scale_bits > 0) {
                continue; // Tested separately for overflow rejection below.
            }
            for (const bool use_trap : {false, true}) {
                const auto widths = plans::generate_box_width_trials(256, 1.5F);
                const std::array<std::pair<SizeType, SizeType>, 2> specs{{
                    {1, nchans},
                    {8, nchans},
                }};
                const auto tab =
                    make_synthetic_tables(widths, use_trap, 6.0F, sigma_in,
                                          specs, true, mean_in, scale_bits);
                const uint64_t max_samp = static_cast<uint64_t>(maxval) *
                                          nchans * (1ULL << scale_bits);
                const auto plan         = kadane::make_kadane_gate_plan(
                    widths, use_trap, tab.n_rows, tab.raw_thresh, tab.int_mean,
                    tab.int_thresh, true, max_samp);

                REQUIRE(plan.has_int32);
                const SizeType nf = widths.size();

                for (SizeType r = 0; r < tab.n_rows; ++r) {
                    const double mu_1 =
                        tab.int_mean[r * nf] / static_cast<double>(widths[0]);
                    const auto shift_scale =
                        static_cast<double>(1ULL << plan.int_shift);
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const int32_t ib = plan.row_ibias(k, r);
                        const int32_t ic = plan.row_icut(k, r);
                        // Strictly negative drift & positive cut.
                        REQUIRE(static_cast<double>(ib) > (mu_1 * shift_scale));
                        REQUIRE(ic >= 1);
                    }

                    // Inject exact integer threshold pulses for every width.
                    const auto row_ithr = std::span<const uint64_t>(
                        tab.int_thresh.data() + (r * nf), nf);
                    const auto base_val =
                        static_cast<uint32_t>(std::llround(mu_1));
                    for (SizeType f = 0; f < nf; ++f) {
                        const uint32_t w = widths[f];
                        const uint32_t we =
                            use_trap ? kadane::trapezoid_edge_width(w) : 0U;
                        std::vector<uint32_t> istream(3 * kTileLen, base_val);
                        const SizeType end_t   = (2 * kTileLen) + 11;
                        const SizeType beg_t   = end_t + 1 - w;
                        const uint64_t fw      = we == 0U ? w : (2U * (w - we));
                        const uint64_t cur_sum = base_val * fw;
                        const uint64_t target  = row_ithr[f];
                        if (target > cur_sum) {
                            const uint64_t need = target - cur_sum;
                            const uint64_t step = (need + fw - 1ULL) / fw;
                            for (SizeType t = beg_t; t <= end_t; ++t) {
                                istream[t] += static_cast<uint32_t>(step);
                            }
                        }
                        const auto ihits = run_kadane_gate_tiles_i32(
                            plan, r, istream, kTileLen);
                        for (SizeType ti = 0; ti < ihits.size(); ++ti) {
                            if (exact_tile_has_hit_int(istream, ti * kTileLen,
                                                       kTileLen, widths,
                                                       use_trap, row_ithr)) {
                                REQUIRE(ihits[ti]);
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. Overflow & bound guards:
    // - max_int_sample == 0 (unknown upper bound) disables int32 gate.
    // - max_int_sample > kKadaneInt32Cap disables int32 gate.
    // - Wide 16-bit smeared rows disable int32 gate.
    // - 8-bit nchans = 2048, w_max = 4096, mean = 24 (max_int_sample = 522240)
    //   enables int32 gate and preserves 0-FN under sustained clamp saturation.
    {
        const auto widths = plans::generate_box_width_trials(4096, 2.0F);
        const std::array<std::pair<SizeType, SizeType>, 1> small_spec{
            {{1, 64}}};
        const auto small_tab = make_synthetic_tables(widths, false, 6.0F, 8.0F,
                                                     small_spec, true, 64.0, 0);
        const auto zero_max_plan = kadane::make_kadane_gate_plan(
            widths, false, 1, small_tab.raw_thresh, small_tab.int_mean,
            small_tab.int_thresh, true, 0U);
        CHECK(!zero_max_plan.has_int32);
        CHECK(zero_max_plan.ibias.empty());

        const auto over_cap_plan = kadane::make_kadane_gate_plan(
            widths, false, 1, small_tab.raw_thresh, small_tab.int_mean,
            small_tab.int_thresh, true,
            static_cast<uint64_t>(kadane::kKadaneInt32Cap) + 1ULL);
        CHECK(!over_cap_plan.has_int32);
        CHECK(over_cap_plan.ibias.empty());

        const std::array<std::pair<SizeType, SizeType>, 1> big_spec{{{8, 256}}};
        const auto tab  = make_synthetic_tables(widths, false, 6.0F, 4000.0F,
                                                big_spec, true, 32768.0, 6);
        const auto plan = kadane::make_kadane_gate_plan(
            widths, false, 1, tab.raw_thresh, tab.int_mean, tab.int_thresh,
            true, 65535ULL * 256ULL * 64ULL);
        CHECK(!plan.has_int32);
        CHECK(plan.ibias.empty());
        CHECK(plan.icut.empty());

        // 8-bit multi-channel (nchans = 2048, w_max = 4096, mean = 24,
        // max_int_sample = 522240) with sustained max_int_sample clamp run
        // followed by a marginal threshold pulse.
        const std::array<std::pair<SizeType, SizeType>, 1> ch2048_spec{{
            {1, 2048},
        }};
        constexpr uint32_t kMax2048 = 255U * 2048U; // 522,240
        for (const bool use_trap : {false, true}) {
            const auto tab2048 = make_synthetic_tables(
                widths, use_trap, 6.0F, 12.0F, ch2048_spec, true, 24.0, 0);
            const auto plan2048 = kadane::make_kadane_gate_plan(
                widths, use_trap, 1, tab2048.raw_thresh, tab2048.int_mean,
                tab2048.int_thresh, true, kMax2048);
            REQUIRE(plan2048.has_int32);
            CHECK(plan2048.memory_bytes() > 0);

            const SizeType nf   = widths.size();
            const auto row_ithr = std::span<const uint64_t>(tab2048.int_thresh);
            const uint32_t base_val = 24U * 2048U;
            // Stream: 8 tiles of saturated kMax2048 (forces c to clamp at
            // kKadaneInt32Cap), then quiet baseline, then a w=4096 threshold
            // pulse, plus a second pulse right on the tail of the saturated
            // run.
            const SizeType total_len = 32 * kTileLen;
            std::vector<uint32_t> istream(total_len, base_val);
            for (SizeType t = 0; t < 8 * kTileLen; ++t) {
                istream[t] = kMax2048;
            }
            const uint32_t w_last = widths[nf - 1];
            const uint32_t we_last =
                use_trap ? kadane::trapezoid_edge_width(w_last) : 0U;
            const uint64_t fw_last =
                we_last == 0U ? w_last : (2U * (w_last - we_last));
            const uint64_t need =
                row_ithr[nf - 1] - (static_cast<uint64_t>(base_val) * fw_last);
            const uint64_t step  = (need + fw_last - 1ULL) / fw_last;
            const SizeType end_t = (30 * kTileLen) + 19;
            const SizeType beg_t = end_t + 1 - w_last;
            for (SizeType t = beg_t; t <= end_t; ++t) {
                istream[t] += static_cast<uint32_t>(step);
            }

            const auto ihits =
                run_kadane_gate_tiles_i32(plan2048, 0, istream, kTileLen);
            for (SizeType ti = 0; ti < ihits.size(); ++ti) {
                if (exact_tile_has_hit_int(istream, ti * kTileLen, kTileLen,
                                           widths, use_trap, row_ithr)) {
                    REQUIRE(ihits[ti]);
                }
            }
        }
    }

    // 3. Dead / killed rows never trigger and store safe sentinels.
    {
        const auto widths = plans::generate_box_width_trials(128, 1.5F);
        const SizeType nf = widths.size();
        std::vector<float> thr(3 * nf, std::numeric_limits<float>::max());
        // Row 0: FLT_MAX (killed), Row 1: active, Row 2: 0.0F (dead).
        for (SizeType f = 0; f < nf; ++f) {
            thr[nf + f]       = 6.0F * std::sqrt(static_cast<float>(widths[f]));
            thr[(2 * nf) + f] = 0.0F;
        }
        const KadaneGatePlan plan(widths, false, 3, thr);
        CHECK(plan.disabled_row[0] == 1U);
        CHECK(plan.disabled_row[1] == 0U);
        CHECK(plan.disabled_row[2] == 1U);
        for (SizeType k = 0; k < plan.num_buckets; ++k) {
            CHECK(plan.row_bias(k, 0) == 0.0F);
            CHECK(plan.row_cut(k, 0) == std::numeric_limits<float>::max());
            CHECK(plan.row_bias(k, 2) == 0.0F);
            CHECK(plan.row_cut(k, 2) == std::numeric_limits<float>::max());
        }
    }

    // 4. Mixed-mask BlockNoise cut recomputation preserves c_{t,k} continuity
    // with 0 false negatives when half the channels are dynamically killed.
    {
        const auto widths = plans::generate_box_width_trials(256, 1.5F);
        const std::array<std::pair<SizeType, SizeType>, 1> full_spec{{{4, 64}}};
        const std::array<std::pair<SizeType, SizeType>, 1> half_spec{{{4, 24}}};
        const auto full_tab =
            make_synthetic_tables(widths, true, 6.0F, 1.0F, full_spec);
        const auto half_tab =
            make_synthetic_tables(widths, true, 6.0F, 1.0F, half_spec);

        const KadaneGatePlan plan(widths, true, 1, full_tab.raw_thresh);
        std::vector<float> min_thr(widths.size());
        for (SizeType f = 0; f < widths.size(); ++f) {
            min_thr[f] =
                std::min(full_tab.raw_thresh[f], half_tab.raw_thresh[f]);
        }
        std::vector<float> mixed_cuts(plan.num_buckets, 0.0F);
        plan.recompute_row_cuts(0, min_thr, mixed_cuts);

        KadaneGatePlan mixed_plan = plan;
        for (SizeType k = 0; k < plan.num_buckets; ++k) {
            CHECK(mixed_cuts[k] <= plan.row_cut(k, 0));
            mixed_plan.cut[k * mixed_plan.row_stride] = mixed_cuts[k];
        }

        // Stream transitions from full-channel noise in tile 0 to reduced-noise
        // threshold pulse in tile 1 WITHOUT resetting c_{t,k}.
        std::normal_distribution<float> norm(0.0F, 1.0F);
        const float sigma1_full = 1.0F / full_tab.inv_sigma[0];
        const float sigma1_half = 1.0F / half_tab.inv_sigma[0];
        for (SizeType f = 0; f < widths.size(); ++f) {
            std::vector<float> stream(2 * kTileLen, 0.0F);
            for (SizeType t = 0; t < kTileLen; ++t) {
                stream[t] = sigma1_full * norm(rng);
            }
            for (SizeType t = kTileLen; t < 2 * kTileLen; ++t) {
                stream[t] = 0.1F * sigma1_half * norm(rng);
            }
            const uint32_t w     = widths[f];
            const uint32_t we    = kadane::trapezoid_edge_width(w);
            const SizeType end_t = kTileLen + 60;
            const SizeType beg_t = end_t + 1 - w;
            double cur           = 0.0;
            for (SizeType t = beg_t; t <= end_t; ++t) {
                cur += static_cast<double>(stream[t]);
            }
            if (we > 0U) {
                for (SizeType t = beg_t + we; t <= end_t - we; ++t) {
                    cur += static_cast<double>(stream[t]);
                }
            }
            const double wt = we == 0U ? static_cast<double>(w)
                                       : static_cast<double>(2U * (w - we));
            const auto add  = static_cast<float>(
                ((static_cast<double>(half_tab.raw_thresh[f]) * 1.001) - cur) /
                wt);
            for (SizeType t = beg_t; t <= end_t; ++t) {
                stream[t] += add;
            }

            const auto hits =
                run_kadane_gate_tiles_f32(mixed_plan, 0, stream, kTileLen);
            REQUIRE(exact_tile_has_hit_f32(stream, kTileLen, kTileLen, widths,
                                           true, min_thr));
            REQUIRE(hits[1]);
        }
    }
}

TEST_CASE("KadaneGatePlan 256-sample noise tile trigger suppression < 0.5%",
          "[kadane_plan][kadane][boxcar][cpu]") {
    constexpr SizeType kTileLen  = 256;
    constexpr SizeType kNumTiles = 4000;
    const SizeType total_samps   = kTileLen * kNumTiles;

    auto rng = make_seeded_rng(424242U);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    std::poisson_distribution<int> poisson(64); // mean=64, sigma=8

    std::vector<float> g_stream(total_samps);
    for (float& x : g_stream) {
        x = gauss(rng);
    }

    std::vector<uint32_t> p_stream(total_samps);
    for (uint32_t& x : p_stream) {
        x = static_cast<uint32_t>(std::clamp(poisson(rng), 0, 255));
    }

    for (const SizeType w_max : {SizeType{200}, SizeType{4096}}) {
        const auto widths = plans::generate_box_width_trials(w_max, 1.5F);
        for (const bool use_trap : {false, true}) {
            // 1. Float32 Gaussian noise tiles.
            const std::array<std::pair<SizeType, SizeType>, 2> specs{{
                {1, 1},  // White noise
                {8, 64}, // Colored FDMT noise
            }};
            const auto f_tab =
                make_synthetic_tables(widths, use_trap, 6.0F, 1.0F, specs);
            const KadaneGatePlan f_plan(widths, use_trap, f_tab.n_rows,
                                        f_tab.raw_thresh);

            const auto g_hits =
                run_kadane_gate_tiles_f32(f_plan, 0, g_stream, kTileLen);
            const auto g_fired =
                static_cast<SizeType>(std::ranges::count(g_hits, true));
            const double g_frac =
                static_cast<double>(g_fired) / static_cast<double>(kNumTiles);
            INFO("Float32 noise w_max=" << w_max << " trap=" << use_trap
                                        << " trigger_frac=" << g_frac);
            CHECK(g_frac < 0.005);

            // 2. Poisson-quantized unsigned integer tiles (mean=64, sigma=8).
            const std::array<std::pair<SizeType, SizeType>, 1> ispec{{{1, 1}}};
            const auto i_tab = make_synthetic_tables(
                widths, use_trap, 6.0F, 8.0F, ispec, true, 64.0, 0);
            const auto i_plan = kadane::make_kadane_gate_plan(
                widths, use_trap, 1, i_tab.raw_thresh, i_tab.int_mean,
                i_tab.int_thresh, true, 255U);
            REQUIRE(i_plan.has_int32);

            const auto p_hits =
                run_kadane_gate_tiles_i32(i_plan, 0, p_stream, kTileLen);
            const auto p_fired =
                static_cast<SizeType>(std::ranges::count(p_hits, true));
            const double p_frac =
                static_cast<double>(p_fired) / static_cast<double>(kNumTiles);
            INFO("Poisson int32 w_max=" << w_max << " trap=" << use_trap
                                        << " trigger_frac=" << p_frac);
            CHECK(p_frac < 0.005);
        }
    }
}

TEST_CASE("KadaneSurrogatePlan strip coverage, O(1) S/N evaluation, and pulse "
          "recovery across 1..4096",
          "[kadane_plan][kadane][boxcar][cpu]") {
    const auto widths = plans::generate_box_width_trials(4096, 1.5F);
    const std::array<std::pair<SizeType, SizeType>, 2> specs{{
        {1, 1},   // White noise
        {16, 64}, // Colored FDMT box smearing
    }};
    const auto tab =
        make_synthetic_tables(widths, false, 6.0F, 1.0F, specs, false);

    const auto dec_plan = kadane::make_kadane_surrogate_plan(
        widths, false, tab.n_rows, tab.raw_thresh, tab.inv_sigma, {}, 1.0,
        true);
    const auto full_plan = kadane::make_kadane_surrogate_plan(
        widths, false, tab.n_rows, tab.raw_thresh, tab.inv_sigma, {}, 1.0,
        false);

    REQUIRE(dec_plan.num_passes == 3);
    CHECK(dec_plan.has_decimated);
    CHECK(dec_plan.passes[0].w_lo == 1U);
    CHECK(dec_plan.passes[0].decimate_factor == 1U);
    CHECK(dec_plan.passes[1].w_lo == dec_plan.passes[0].w_hi + 1U);
    CHECK(dec_plan.passes[1].decimate_factor == 4U);
    CHECK(dec_plan.passes[2].w_lo == dec_plan.passes[1].w_hi + 1U);
    CHECK(dec_plan.passes[2].w_hi == 4096U);
    CHECK(dec_plan.passes[2].decimate_factor == 4U);

    REQUIRE(full_plan.num_passes == 3);
    CHECK(!full_plan.has_decimated);
    for (SizeType k = 0; k < full_plan.num_passes; ++k) {
        CHECK(full_plan.passes[k].decimate_factor == 1U);
    }

    // Trapezoid disables decimation automatically.
    const auto trap_tab =
        make_synthetic_tables(widths, true, 6.0F, 1.0F, specs, false);
    const auto trap_plan = kadane::make_kadane_surrogate_plan(
        widths, true, trap_tab.n_rows, trap_tab.raw_thresh, trap_tab.inv_sigma,
        {}, 1.0, true);
    CHECK(!trap_plan.has_decimated);

    // Check O(1) arbitrary-width S/N evaluation on both white & colored rows.
    CHECK(dec_plan.memory_bytes() > 0);
    for (uint32_t w = 1U; w <= 4096U; ++w) {
        uint32_t f_idx     = UINT32_MAX;
        const float snr_w0 = dec_plan.eval_surrogate_snr(
            0, 10.0 * std::sqrt(static_cast<double>(w)), w, &f_idx);
        CHECK(f_idx < widths.size());
        CHECK_THAT(snr_w0, Catch::Matchers::WithinRel(10.0F, 1.0e-5F));

        // Colored FDMT row (leaf_len=16, n_chans=64) exact variance comparison.
        const double exact_var =
            64.0 * fdmt_trapezoid_variance(16, w, 0, 1.0 / 16.0);
        const double exact_sigma = std::sqrt(exact_var);
        const float snr_colored =
            dec_plan.eval_surrogate_snr(1, 10.0 * exact_sigma, w);
        CHECK_THAT(snr_colored, Catch::Matchers::WithinRel(10.0F, 0.015F));
    }

    // Check packed integer (scale_bits = 5, integer_unit = 1/32) S/N evaluation
    // and sub-minimum width (`width < widths[0]`) inv_sigma extrapolation.
    {
        const auto p_tab  = make_synthetic_tables(widths, true, 6.0F, 4.0F,
                                                  specs, true, 128.0, 5);
        const auto p_plan = kadane::make_kadane_surrogate_plan(
            widths, true, p_tab.n_rows, p_tab.raw_thresh, p_tab.inv_sigma,
            p_tab.int_mean, p_tab.integer_unit, true);
        CHECK(p_plan.is_integer);
        CHECK(p_plan.memory_bytes() > 0);
        const double scale = 32.0;
        for (const uint32_t w : {1U, 4U, 19U, 128U, 1024U, 4096U}) {
            const double mean_sum = p_plan.row_mean_1[0] * w;
            const double sig_w    = 4.0 * std::sqrt(static_cast<double>(w));
            const double raw_sum  = mean_sum + (8.5 * sig_w * scale);
            const float snr_p     = p_plan.eval_surrogate_snr(0, raw_sum, w);
            CHECK_THAT(snr_p, Catch::Matchers::WithinRel(8.5F, 1.0e-4F));
        }

        // Sub-minimum bank width (widths starting at 4, evaluating width = 1).
        const std::array<uint32_t, 3> hi_widths{4U, 8U, 16U};
        const std::array<std::pair<SizeType, SizeType>, 1> wspec{{{1, 1}}};
        const auto hi_tab =
            make_synthetic_tables(hi_widths, true, 6.0F, 2.0F, wspec, false);
        const auto hi_plan = kadane::make_kadane_surrogate_plan(
            hi_widths, true, 1, hi_tab.raw_thresh, hi_tab.inv_sigma);
        CHECK_THAT(hi_plan.eval_surrogate_snr(0, 20.0, 1U),
                   Catch::Matchers::WithinRel(10.0F, 1.0e-4F));
    }

    // Simulate multi-pass decimated surrogate Kadane across widths 1..4096
    // on noisy streams with injected S/N = 10 pulses.
    auto rng = make_seeded_rng(13579U);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    const std::array<uint32_t, 10> test_widths{1,  2,   5,   12,   20,
                                               48, 128, 400, 1200, 4096};

    for (const uint32_t pw : test_widths) {
        constexpr SizeType kStreamLen = 12288;
        std::vector<float> stream(kStreamLen);
        for (float& x : stream) {
            x = gauss(rng);
        }
        const SizeType p_start = 4096 + 3; // Intentionally unaligned mod 4!
        double cur             = 0.0;
        for (SizeType t = p_start; t < p_start + pw; ++t) {
            cur += static_cast<double>(stream[t]);
        }
        const double target = 10.0 * std::sqrt(static_cast<double>(pw));
        const auto add      = static_cast<float>((target - cur) / pw);
        for (SizeType t = p_start; t < p_start + pw; ++t) {
            stream[t] += add;
        }

        // Prefix sum for O(1) exact boxcar re-score of winning excursions.
        std::vector<double> pfx(kStreamLen + 1, 0.0);
        for (SizeType t = 0; t < kStreamLen; ++t) {
            pfx[t + 1] = pfx[t] + static_cast<double>(stream[t]);
        }

        float best_snr = 0.0F;
        for (SizeType k = 0; k < dec_plan.num_passes; ++k) {
            const uint32_t dec   = dec_plan.passes[k].decimate_factor;
            const float b        = dec_plan.row_bias(k, 0);
            const float trig     = dec_plan.row_trigger(k, 0);
            const SizeType steps = kStreamLen / dec;

            float c          = 0.0F;
            float exc_peak   = 0.0F;
            SizeType exc_beg = 0;
            SizeType peak_st = 0;
            SizeType peak_en = 0;

            const auto finish_exc = [&] {
                if (exc_peak >= trig) {
                    const SizeType s0 = peak_st * dec;
                    const SizeType s1 = ((peak_en + 1) * dec);
                    const auto w_cand = static_cast<uint32_t>(s1 - s0);
                    const double rsum = pfx[s1] - pfx[s0];
                    best_snr          = std::max(
                        best_snr, dec_plan.eval_surrogate_snr(0, rsum, w_cand));
                }
                exc_peak = 0.0F;
            };

            for (SizeType j = 0; j < steps; ++j) {
                float y = 0.0F;
                for (uint32_t q = 0; q < dec; ++q) {
                    y += stream[(j * dec) + q];
                }
                if (c == 0.0F) {
                    exc_beg = j;
                }
                c += (y - b);
                if (c <= 0.0F) {
                    finish_exc();
                    c = 0.0F;
                } else if (c > exc_peak) {
                    exc_peak = c;
                    peak_st  = exc_beg;
                    peak_en  = j;
                }
            }
            finish_exc();
        }
        INFO("Surrogate pulse width=" << pw << " recovered S/N=" << best_snr);
        CHECK(best_snr >= 8.0F);
    }
}

TEST_CASE("KadaneGatePlan wide boxcars up to 2^20: soundness, int32 bounds, "
          "and width validation",
          "[kadane_plan][kadane][boxcar][cpu]") {
    constexpr SizeType kTileLen = 256;
    const std::array<std::pair<SizeType, SizeType>, 3> row_specs{{
        {1, 1},    // White noise
        {8, 64},   // Moderate FDMT smearing
        {64, 256}, // Heavy FDMT smearing
    }};

    // 1. Validation boundaries on KadaneGatePlan vs KadaneSurrogatePlan.
    {
        const std::vector<float> one_thr{6.0F};
        const std::vector<float> one_inv{1.0F};
        CHECK_THROWS_AS(
            kadane::make_kadane_gate_plan(std::span<const uint32_t>{}, false, 1,
                                          std::span<const float>{}),
            std::invalid_argument);

        const std::array<uint32_t, 1> zero_w{0U};
        CHECK_THROWS_AS(
            kadane::make_kadane_gate_plan(zero_w, false, 1, one_thr),
            std::invalid_argument);

        const std::array<uint32_t, 2> dup_w{4U, 4U};
        const std::array<float, 2> two_thr{12.0F, 12.0F};
        CHECK_THROWS_AS(kadane::make_kadane_gate_plan(dup_w, false, 1, two_thr),
                        std::invalid_argument);

        // > 2^20 max width rejected by KadaneGatePlan.
        const std::array<uint32_t, 2> over_max_w{1U, (1U << 20) + 1U};
        CHECK_THROWS_AS(
            kadane::make_kadane_gate_plan(over_max_w, false, 1, two_thr),
            std::invalid_argument);

        // > 4096 widths (even within 1..2^20) rejected by KadaneGatePlan.
        std::vector<uint32_t> too_many_w(4097);
        std::vector<float> too_many_thr(4097);
        for (SizeType i = 0; i < too_many_w.size(); ++i) {
            too_many_w[i] = static_cast<uint32_t>(i + 1);
            too_many_thr[i] =
                6.0F * std::sqrt(static_cast<float>(too_many_w[i]));
        }
        CHECK_THROWS_AS(
            kadane::make_kadane_gate_plan(too_many_w, false, 1, too_many_thr),
            std::invalid_argument);

        // Exactly 4096 widths ending at 1048576 is accepted by KadaneGatePlan.
        std::vector<uint32_t> max_trials_w(4096);
        std::vector<float> max_trials_thr(4096);
        for (SizeType i = 0; i < 4095; ++i) {
            max_trials_w[i] = static_cast<uint32_t>(i + 1);
            max_trials_thr[i] =
                6.0F * std::sqrt(static_cast<float>(max_trials_w[i]));
        }
        max_trials_w[4095]   = 1048576U;
        max_trials_thr[4095] = 6.0F * 1024.0F;
        const auto ok_4096   = kadane::make_kadane_gate_plan(max_trials_w, true,
                                                             1, max_trials_thr);
        CHECK(ok_4096.widths.size() == 4096);
        CHECK(ok_4096.buckets[ok_4096.num_buckets - 1].w_hi == 1048576U);

        // KadaneSurrogatePlan remains strictly capped at max_width <= 4096.
        const std::array<uint32_t, 2> surr_4097{1U, 4097U};
        const std::array<float, 2> surr_inv{1.0F, 1.0F / 64.0F};
        CHECK_THROWS_AS(kadane::make_kadane_surrogate_plan(surr_4097, false, 1,
                                                           two_thr, surr_inv),
                        std::invalid_argument);
        const auto wide_8192 = plans::generate_box_width_trials(8192, 2.0F);
        const auto tab_8192 =
            make_synthetic_tables(wide_8192, false, 6.0F, 1.0F, row_specs);
        CHECK_THROWS_AS(kadane::make_kadane_surrogate_plan(
                            wide_8192, false, tab_8192.n_rows,
                            tab_8192.raw_thresh, tab_8192.inv_sigma),
                        std::invalid_argument);
    }

    // 2. Algebraic soundness & cut clamping across wide boxcars (16384 & 2^20).
    for (const SizeType max_boxcar : {SizeType{16384}, SizeType{1048576}}) {
        for (const float wtsp : {2.0F, 1.5F}) {
            const auto widths =
                plans::generate_box_width_trials(max_boxcar, wtsp);
            REQUIRE(widths.size() <= kadane::kMaxKadaneGateFilters);
            REQUIRE(widths.back() == max_boxcar);

            for (const bool use_trap : {false, true}) {
                const auto tab  = make_synthetic_tables(widths, use_trap, 6.0F,
                                                        1.0F, row_specs);
                const auto plan = kadane::make_kadane_gate_plan(
                    widths, use_trap, tab.n_rows, tab.raw_thresh);

                REQUIRE(plan.num_buckets == 3);
                CHECK(plan.buckets[2].w_hi == max_boxcar);
                if (use_trap) {
                    CHECK(plan.max_edge_lag[2] ==
                          kadane::trapezoid_edge_width(
                              static_cast<uint32_t>(max_boxcar)));
                } else {
                    CHECK(plan.max_edge_lag[2] == 0U);
                }

                const auto finite_f32 = [](float x) noexcept {
                    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
                    return (bits & 0x7F800000U) != 0x7F800000U;
                };
                const SizeType nf = widths.size();
                for (SizeType r = 0; r < tab.n_rows; ++r) {
                    REQUIRE(plan.disabled_row[r] == 0U);
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const float b   = plan.row_bias(k, r);
                        const float cut = plan.row_cut(k, r);
                        REQUIRE(finite_f32(b));
                        REQUIRE(b > 0.0F);
                        REQUIRE(finite_f32(cut));
                        REQUIRE(cut >= 0.0F);
                        for (uint32_t f = plan.buckets[k].f_lo;
                             f < plan.buckets[k].f_hi; ++f) {
                            const uint32_t w = widths[f];
                            const uint32_t we =
                                use_trap ? kadane::trapezoid_edge_width(w) : 0U;
                            const double thr = tab.raw_thresh[(r * nf) + f];
                            const double exact_bound =
                                we == 0U
                                    ? (thr - (static_cast<double>(b) * w))
                                    : (0.5 * (thr - (static_cast<double>(b) *
                                                     (2U * (w - we)))));
                            REQUIRE(static_cast<double>(cut) < exact_bound);
                        }
                    }
                }

                // Force margin > tau_raw on the 2^20 bucket via a steep fixed
                // bias in recompute_cuts_for_biases and verify safe clamp to 0.
                std::vector<float> steep_biases(plan.num_buckets);
                for (SizeType k = 0; k < plan.num_buckets; ++k) {
                    steep_biases[k] = plan.row_bias(k, 0);
                }
                const uint32_t f_last = static_cast<uint32_t>(nf - 1);
                const uint32_t w_last = widths[f_last];
                const uint32_t we_last =
                    use_trap ? kadane::trapezoid_edge_width(w_last) : 0U;
                const double tilde_last = static_cast<double>(w_last - we_last);
                const double tilde_thr =
                    static_cast<double>(tab.raw_thresh[f_last]) *
                    (we_last == 0U ? 1.0 : 0.5);
                // Bias leaving tiny positive tau_raw < rounding margin:
                steep_biases[plan.num_buckets - 1] =
                    static_cast<float>((tilde_thr - 1.0e-3) / tilde_last);
                std::vector<float> clamped_cuts(plan.num_buckets, -1.0F);
                kadane::recompute_cuts_for_biases(
                    plan, steep_biases,
                    std::span<const float>(tab.raw_thresh.data(), nf),
                    clamped_cuts);
                CHECK(clamped_cuts[plan.num_buckets - 1] == 0.0F);
            }
        }
    }

    // 3. Long-excursion float32 0-false-negative pulse soundness at W = 16384
    // (noisy + constant marginal) and W = 1048576 (2^20-step constant marginal)
    // across both pure boxcar and {1, 2, 1} trapezoid.
    auto rng = make_seeded_rng(20261113U);
    std::normal_distribution<float> gauss(0.0F, 1.0F);
    const std::array<std::pair<SizeType, SizeType>, 1> white_spec{{{1, 1}}};

    for (const bool use_trap : {false, true}) {
        // 3a. W = 16384 noisy pulses across wide filters + marginal constant.
        const auto widths_16k = plans::generate_box_width_trials(16384, 2.0F);
        const auto tab_16k =
            make_synthetic_tables(widths_16k, use_trap, 6.0F, 1.0F, white_spec);
        const KadaneGatePlan gate_16k(widths_16k, use_trap, 1,
                                      tab_16k.raw_thresh);

        for (const uint32_t target_w : {4096U, 8192U, 16384U}) {
            const auto it = std::ranges::find(widths_16k, target_w);
            REQUIRE(it != widths_16k.end());
            const auto f_idx =
                static_cast<SizeType>(std::distance(widths_16k.begin(), it));
            const SizeType stream_len = target_w + (3 * kTileLen);
            std::vector<float> stream(stream_len, 0.0F);
            for (float& x : stream) {
                x = 0.15F * gauss(rng);
            }
            const SizeType end_t = stream_len - kTileLen + 23;
            const SizeType beg_t = end_t + 1 - target_w;
            const uint32_t we =
                use_trap ? kadane::trapezoid_edge_width(target_w) : 0U;
            double cur = 0.0;
            for (SizeType t = beg_t; t <= end_t; ++t) {
                cur += static_cast<double>(stream[t]);
            }
            if (we > 0U) {
                for (SizeType t = beg_t + we; t <= end_t - we; ++t) {
                    cur += static_cast<double>(stream[t]);
                }
            }
            const double wt  = we == 0U
                                   ? static_cast<double>(target_w)
                                   : static_cast<double>(2U * (target_w - we));
            const auto delta = static_cast<float>(
                ((static_cast<double>(tab_16k.raw_thresh[f_idx]) * 1.0005) -
                 cur) /
                wt);
            for (SizeType t = beg_t; t <= end_t; ++t) {
                stream[t] += delta;
            }
            const auto gate_hits =
                run_kadane_gate_tiles_f32(gate_16k, 0, stream, kTileLen);
            bool any_exact = false;
            for (SizeType ti = 0; ti < gate_hits.size(); ++ti) {
                if (exact_tile_has_hit_f32(stream, ti * kTileLen, kTileLen,
                                           widths_16k, use_trap,
                                           tab_16k.raw_thresh)) {
                    any_exact = true;
                    REQUIRE(gate_hits[ti]);
                }
            }
            REQUIRE(any_exact);
        }

        // 3b. W = 1048576 (2^20 steps!) worst-case constant-pulse float32
        // accumulation test (2 ULPs above threshold per sample).
        constexpr uint32_t kWMax20 = 1U << 20;
        const auto widths_1m = plans::generate_box_width_trials(kWMax20, 2.0F);
        const auto tab_1m =
            make_synthetic_tables(widths_1m, use_trap, 8.0F, 1.0F, white_spec);
        const KadaneGatePlan gate_1m(widths_1m, use_trap, 1, tab_1m.raw_thresh);

        const SizeType stream_len = kWMax20 + (2 * kTileLen);
        std::vector<float> stream(stream_len, 0.0F);
        const SizeType end_t = stream_len - 29;
        const SizeType beg_t = end_t + 1 - kWMax20;
        const uint32_t we =
            use_trap ? kadane::trapezoid_edge_width(kWMax20) : 0U;
        const double denom  = we == 0U
                                  ? static_cast<double>(kWMax20)
                                  : static_cast<double>(2U * (kWMax20 - we));
        const float thr_max = tab_1m.raw_thresh.back();
        const float amp =
            std::bit_cast<float>(std::bit_cast<uint32_t>(static_cast<float>(
                                     static_cast<double>(thr_max) / denom)) +
                                 2U);
        for (SizeType t = beg_t; t <= end_t; ++t) {
            stream[t] = amp;
        }

        // Verify the endpoint tile in O(kWMax20) without scanning all 4098
        // tiles by quadratic reference prefix loops: check exact endpoint sum
        // directly and confirm the float32 Kadane gate triggers on end_t's
        // tile.
        const auto gate_hits =
            run_kadane_gate_tiles_f32(gate_1m, 0, stream, kTileLen);
        const SizeType hit_tile = end_t / kTileLen;
        REQUIRE(exact_tile_has_hit_f32(stream, hit_tile * kTileLen, kTileLen,
                                       widths_1m, use_trap, tab_1m.raw_thresh));
        REQUIRE(gate_hits[hit_tile]);
    }

    // 4. Wide-boxcar int32_t gate overflow & clean fallback checks.
    {
        constexpr uint32_t kWMax20 = 1U << 20;
        const auto widths_1m = plans::generate_box_width_trials(kWMax20, 2.0F);

        // 4a. At w_max = 1048576, b_cap = (INT32_MAX / 4) / 1048576 = 511.
        // With nchans = 64, 8-bit mean = 64 -> mu_1 = 4096 > 511, while
        // max_int_sample = 255 * 64 = 16320 <= kKadaneInt32Cap: w_max * ibias
        // exceeds INT32_MAX / 4, so has_int32 MUST cleanly fall back to false!
        const std::array<std::pair<SizeType, SizeType>, 1> ch64_spec{{{1, 64}}};
        const auto tab_over_bcap = make_synthetic_tables(
            widths_1m, false, 6.0F, 12.0F, ch64_spec, true, 64.0, 0);
        const auto plan_over_bcap = kadane::make_kadane_gate_plan(
            widths_1m, false, 1, tab_over_bcap.raw_thresh,
            tab_over_bcap.int_mean, tab_over_bcap.int_thresh, true,
            255ULL * 64ULL);
        CHECK(!plan_over_bcap.has_int32);
        CHECK(plan_over_bcap.ibias.empty());
        CHECK(plan_over_bcap.icut.empty());
        CHECK(plan_over_bcap.int_shift == 0U);

        // 4b. Extreme integer_mean_lut (> 2^63) never triggers UB on double ->
        // int64_t conversion and cleanly disables has_int32.
        auto huge_mean_tab = tab_over_bcap;
        for (double& m : huge_mean_tab.int_mean) {
            m = 1.0e25;
        }
        const auto plan_huge_mean = kadane::make_kadane_gate_plan(
            widths_1m, false, 1, huge_mean_tab.raw_thresh,
            huge_mean_tab.int_mean, huge_mean_tab.int_thresh, true, 1024ULL);
        CHECK(!plan_huge_mean.has_int32);
        CHECK(plan_huge_mean.ibias.empty());

        // 4c. Wide-boxcar int32_t gate when feasible (W = 16384 on 1-bit
        // 64-chan, and W = 1048576 on high-dynamic-range low-mean stream where
        // both chosen_c >= 1 and w_max * ibias <= INT32_MAX / 4 hold), plus
        // clean fallback when quantization forces chosen_c < 1 at W = 1048576.
        {
            const auto w_16k = plans::generate_box_width_trials(16384, 2.0F);
            const auto tab_16k_1bit = make_synthetic_tables(
                w_16k, true, 6.0F, 0.5F, ch64_spec, true, 0.5, 0);
            const auto plan_16k_1bit = kadane::make_kadane_gate_plan(
                w_16k, true, 1, tab_16k_1bit.raw_thresh, tab_16k_1bit.int_mean,
                tab_16k_1bit.int_thresh, true, 64ULL);
            REQUIRE(plan_16k_1bit.has_int32);
            for (SizeType k = 0; k < plan_16k_1bit.num_buckets; ++k) {
                const auto ib =
                    static_cast<uint64_t>(plan_16k_1bit.row_ibias(k, 0));
                const auto ic = plan_16k_1bit.row_icut(k, 0);
                CHECK(ib * 16384ULL <=
                      static_cast<uint64_t>(
                          std::numeric_limits<int32_t>::max() / 4));
                CHECK(ic >= 1);
                CHECK(ic <= (std::numeric_limits<int32_t>::max() / 4));
            }

            const SizeType nf = w_16k.size();
            const auto row_ithr =
                std::span<const uint64_t>(tab_16k_1bit.int_thresh);
            const SizeType total_len = 80 * kTileLen;
            std::vector<uint32_t> istream(total_len, 32U);
            for (SizeType t = 0; t < 4 * kTileLen; ++t) {
                istream[t] = 64U;
            }
            const uint32_t w_last  = w_16k[nf - 1];
            const uint32_t we_last = kadane::trapezoid_edge_width(w_last);
            const uint64_t fw_last = 2ULL * (w_last - we_last);
            const uint64_t need    = row_ithr[nf - 1] - (32ULL * fw_last);
            const uint64_t step    = (need + fw_last - 1ULL) / fw_last;
            const SizeType end_t   = total_len - 31;
            const SizeType beg_t   = end_t + 1 - w_last;
            for (SizeType t = beg_t; t <= end_t; ++t) {
                istream[t] += static_cast<uint32_t>(step);
            }
            const auto ihits =
                run_kadane_gate_tiles_i32(plan_16k_1bit, 0, istream, kTileLen);
            const SizeType hit_tile = end_t / kTileLen;
            REQUIRE(exact_tile_has_hit_int(istream, hit_tile * kTileLen,
                                           kTileLen, w_16k, true, row_ithr));
            REQUIRE(ihits[hit_tile]);

            // At W = 1048576, 1-bit 64-chan has chosen_c < 1 under b_cap = 511,
            // so it cleanly falls back to has_int32 == false:
            const auto tab_1m_1bit = make_synthetic_tables(
                widths_1m, true, 6.0F, 0.5F, ch64_spec, true, 0.5, 0);
            const auto plan_1m_1bit = kadane::make_kadane_gate_plan(
                widths_1m, true, 1, tab_1m_1bit.raw_thresh,
                tab_1m_1bit.int_mean, tab_1m_1bit.int_thresh, true, 64ULL);
            CHECK(!plan_1m_1bit.has_int32);
            CHECK(plan_1m_1bit.ibias.empty());

            // Whereas W = 1048576 with mu_1 = 32 and sigma_1 = 80 satisfies
            // both chosen_c >= 1 and 1048576 * ibias <= INT32_MAX / 4:
            const std::array<std::pair<SizeType, SizeType>, 1> ch4_spec{
                {{1, 4}}};
            const auto tab_1m_ok = make_synthetic_tables(
                widths_1m, true, 6.0F, 40.0F, ch4_spec, true, 8.0, 0);
            const auto plan_1m_ok = kadane::make_kadane_gate_plan(
                widths_1m, true, 1, tab_1m_ok.raw_thresh, tab_1m_ok.int_mean,
                tab_1m_ok.int_thresh, true, 1020ULL);
            REQUIRE(plan_1m_ok.has_int32);
            for (SizeType k = 0; k < plan_1m_ok.num_buckets; ++k) {
                const auto ib =
                    static_cast<uint64_t>(plan_1m_ok.row_ibias(k, 0));
                const auto ic = plan_1m_ok.row_icut(k, 0);
                CHECK(ib * kWMax20 <=
                      static_cast<uint64_t>(
                          std::numeric_limits<int32_t>::max() / 4));
                CHECK(ic >= 1);
                CHECK(ic <= (std::numeric_limits<int32_t>::max() / 4));
            }
        }
    }
}

} // namespace dmt
