#include "dmt/kadane_monoid.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_variance.hpp"
#include "dmt/kadane_cuda.cuh"
#include "dmt/kadane_plan.hpp"
#include "dmt/kadane_simd.hpp"

namespace dmt {
namespace {

using algorithms::detail::fdmt_trapezoid_variance;
using kadane::clamp_kadane_i32;
using kadane::compute_bucket_cut_hd_f32;
using kadane::kadane_apply;
using kadane::kadane_combine;
using kadane::kadane_identity_f32;
using kadane::kadane_identity_i64;
using kadane::KadaneGatePlan;
using kadane::KadaneSegF32;
using kadane::KadaneSegI64;
using kadane::KadaneSurrogateDeviceView;
using kadane::KadaneSurrogatePlan;
using kadane::kKadaneInt32Cap;
using kadane::LagWindow;
using kadane::make_kadane_seg_f32;
using kadane::make_kadane_seg_i64;
using kadane::ScalarBackend;
using kadane::trapezoid_edge_width;

constexpr int kItems = 4;
constexpr int kWarp  = 32;

std::mt19937 make_seeded_rng(uint32_t seed) { return std::mt19937(seed); }

// Exact host emulation of the GPU block/warp two-level tropical reduction used
// in `kernel_fdmt_boxcar_kadane` (`BlockThreads in {64, 128, 256}`, `kItems =
// 4`, intra-warp `shfl_up` power-of-2 tree + warp-0 `warp_totals` reduction).
template <int BlockThreads>
KadaneSegF32 emulate_gpu_block_scan_f32(std::span<const float> tile,
                                        float bias,
                                        uint64_t start,
                                        uint64_t warmup,
                                        bool skip_cold) {
    constexpr int kBlockWarps = BlockThreads / kWarp;
    std::array<KadaneSegF32, BlockThreads> thread_segs{};
    const auto n = tile.size();

    for (int tid = 0; tid < BlockThreads; ++tid) {
        KadaneSegF32 local_seg = kadane_identity_f32();
        for (int j = 0; j < kItems; ++j) {
            const auto local = static_cast<SizeType>((tid * kItems) + j);
            const bool valid =
                local < n && (!skip_cold || (start + local) >= warmup);
            local_seg = kadane_combine(
                local_seg,
                make_kadane_seg_f32(valid ? tile[local] : 0.0F, bias, valid));
        }
        thread_segs[static_cast<SizeType>(tid)] = local_seg;
    }

    std::array<KadaneSegF32, kBlockWarps> warp_totals{};
    for (int w = 0; w < kBlockWarps; ++w) {
        std::array<KadaneSegF32, kWarp> acc{};
        for (int lane = 0; lane < kWarp; ++lane) {
            acc[static_cast<SizeType>(lane)] =
                thread_segs[static_cast<SizeType>((w * kWarp) + lane)];
        }
        for (int offset = 1; offset < kWarp; offset *= 2) {
            std::array<KadaneSegF32, kWarp> next = acc;
            for (int lane = offset; lane < kWarp; ++lane) {
                next[static_cast<SizeType>(lane)] =
                    kadane_combine(acc[static_cast<SizeType>(lane - offset)],
                                   acc[static_cast<SizeType>(lane)]);
            }
            acc = next;
        }
        warp_totals[static_cast<SizeType>(w)] = acc[kWarp - 1];
    }

    std::array<KadaneSegF32, kBlockWarps> w_acc = warp_totals;
    for (int offset = 1; offset < kBlockWarps; offset *= 2) {
        std::array<KadaneSegF32, kBlockWarps> next = w_acc;
        for (int lane = offset; lane < kBlockWarps; ++lane) {
            next[static_cast<SizeType>(lane)] =
                kadane_combine(w_acc[static_cast<SizeType>(lane - offset)],
                               w_acc[static_cast<SizeType>(lane)]);
        }
        w_acc = next;
    }
    return w_acc[kBlockWarps - 1];
}

template <int BlockThreads>
KadaneSegI64 emulate_gpu_block_scan_i64(std::span<const uint32_t> tile,
                                        uint8_t int_shift,
                                        int32_t ibias,
                                        uint64_t start,
                                        uint64_t warmup,
                                        bool skip_cold) {
    constexpr int kBlockWarps = BlockThreads / kWarp;
    std::array<KadaneSegI64, BlockThreads> thread_segs{};
    const auto n = tile.size();

    for (int tid = 0; tid < BlockThreads; ++tid) {
        KadaneSegI64 local_seg = kadane_identity_i64();
        for (int j = 0; j < kItems; ++j) {
            const auto local = static_cast<SizeType>((tid * kItems) + j);
            const bool valid =
                local < n && (!skip_cold || (start + local) >= warmup);
            const uint64_t x = valid ? tile[local] : 0ULL;
            local_seg        = kadane_combine(
                local_seg, make_kadane_seg_i64(x, int_shift, ibias, valid));
        }
        thread_segs[static_cast<SizeType>(tid)] = local_seg;
    }

    std::array<KadaneSegI64, kBlockWarps> warp_totals{};
    for (int w = 0; w < kBlockWarps; ++w) {
        std::array<KadaneSegI64, kWarp> acc{};
        for (int lane = 0; lane < kWarp; ++lane) {
            acc[static_cast<SizeType>(lane)] =
                thread_segs[static_cast<SizeType>((w * kWarp) + lane)];
        }
        for (int offset = 1; offset < kWarp; offset *= 2) {
            std::array<KadaneSegI64, kWarp> next = acc;
            for (int lane = offset; lane < kWarp; ++lane) {
                next[static_cast<SizeType>(lane)] =
                    kadane_combine(acc[static_cast<SizeType>(lane - offset)],
                                   acc[static_cast<SizeType>(lane)]);
            }
            acc = next;
        }
        warp_totals[static_cast<SizeType>(w)] = acc[kWarp - 1];
    }

    std::array<KadaneSegI64, kBlockWarps> w_acc = warp_totals;
    for (int offset = 1; offset < kBlockWarps; offset *= 2) {
        std::array<KadaneSegI64, kBlockWarps> next = w_acc;
        for (int lane = offset; lane < kBlockWarps; ++lane) {
            next[static_cast<SizeType>(lane)] =
                kadane_combine(w_acc[static_cast<SizeType>(lane - offset)],
                               w_acc[static_cast<SizeType>(lane)]);
        }
        w_acc = next;
    }
    return w_acc[kBlockWarps - 1];
}

struct MonoidSyntheticTables {
    std::vector<uint32_t> widths;
    std::vector<float> raw_thresh;
    std::vector<float> inv_sigma;
    std::vector<double> int_mean;
    std::vector<uint64_t> int_thresh;
    SizeType n_rows     = 0;
    double integer_unit = 1.0;
};

MonoidSyntheticTables
make_tables(std::span<const uint32_t> widths,
            bool use_trapezoid,
            float detect_thresh,
            float noise_sigma,
            std::span<const std::pair<SizeType, SizeType>> row_specs,
            bool packed       = false,
            double input_mean = 0.0,
            int scale_bits    = 0) {
    MonoidSyntheticTables out;
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
        for (SizeType f = 0; f < nf; ++f) {
            const uint32_t w    = widths[f];
            const uint32_t edge = use_trapezoid ? trapezoid_edge_width(w) : 0U;
            const double var_1 =
                fdmt_trapezoid_variance(leaf_len, w, edge, 1.0) *
                static_cast<double>(n_chans) *
                static_cast<double>(noise_sigma) *
                static_cast<double>(noise_sigma);
            const auto inv_s = static_cast<float>(1.0 / std::sqrt(var_1));
            out.inv_sigma[(r * nf) + f]  = inv_s;
            out.raw_thresh[(r * nf) + f] = detect_thresh / inv_s;
            if (packed) {
                const uint32_t tw = edge == 0U ? w : 2U * (w - edge);
                const double mu   = input_mean * static_cast<double>(n_chans) *
                                    static_cast<double>(leaf_len) *
                                    static_cast<double>(tw) * scale;
                const double thr_d =
                    mu +
                    (static_cast<double>(out.raw_thresh[(r * nf) + f]) * scale);
                out.int_mean[(r * nf) + f] = mu;
                out.int_thresh[(r * nf) + f] =
                    static_cast<uint64_t>(std::max(1.0, std::ceil(thr_d)));
            }
        }
    }
    return out;
}

} // namespace

TEST_CASE("KadaneSegF32 and KadaneSegI64 satisfy tropical monoid laws",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    auto rng = make_seeded_rng(42U);
    std::normal_distribution<float> dist_f(0.0F, 4.0F);
    std::uniform_int_distribution<uint32_t> dist_u(0U, 4095U);

    const auto id_f = kadane_identity_f32();
    const auto id_i = kadane_identity_i64();

    for (int trial = 0; trial < 200; ++trial) {
        const float bias   = 1.25F;
        const int32_t ib   = 2100;
        const uint8_t shft = 1U;

        auto rand_seg_f = [&] {
            KadaneSegF32 s = id_f;
            for (int i = 0; i < 7; ++i) {
                s = kadane_combine(
                    s, make_kadane_seg_f32(dist_f(rng), bias, (i != 3)));
            }
            return s;
        };
        auto rand_seg_i = [&] {
            KadaneSegI64 s = id_i;
            for (int i = 0; i < 7; ++i) {
                s = kadane_combine(
                    s, make_kadane_seg_i64(dist_u(rng), shft, ib, (i != 2)));
            }
            return s;
        };

        const auto a_f = rand_seg_f();
        const auto b_f = rand_seg_f();
        const auto c_f = rand_seg_f();

        // Exact identity laws
        const auto lid_f = kadane_combine(id_f, a_f);
        const auto rid_f = kadane_combine(a_f, id_f);
        CHECK(lid_f.sum == a_f.sum);
        CHECK(lid_f.pfx == a_f.pfx);
        CHECK(lid_f.sfx == a_f.sfx);
        CHECK(lid_f.best == a_f.best);
        CHECK(rid_f.sum == a_f.sum);
        CHECK(rid_f.pfx == a_f.pfx);
        CHECK(rid_f.sfx == a_f.sfx);
        CHECK(rid_f.best == a_f.best);

        // Floating-point associativity within IEEE-754 rounding tolerance
        const auto lhs_f = kadane_combine(kadane_combine(a_f, b_f), c_f);
        const auto rhs_f = kadane_combine(a_f, kadane_combine(b_f, c_f));
        CHECK(std::abs(lhs_f.sum - rhs_f.sum) <= 1.0e-5F);
        CHECK(std::abs(lhs_f.pfx - rhs_f.pfx) <= 1.0e-5F);
        CHECK(std::abs(lhs_f.sfx - rhs_f.sfx) <= 1.0e-5F);
        CHECK(std::abs(lhs_f.best - rhs_f.best) <= 1.0e-5F);

        // Action homomorphic law: apply(apply(c0, A).c_out, B) == apply(c0, AB)
        const float c0_f  = std::max(0.0F, dist_f(rng));
        const auto ab_f   = kadane_combine(a_f, b_f);
        const auto step_a = kadane_apply(c0_f, a_f);
        const auto step_b = kadane_apply(step_a.c_out, b_f);
        const auto direct = kadane_apply(c0_f, ab_f);
        CHECK(std::abs(step_b.c_out - direct.c_out) <= 1.0e-5F);
        CHECK(std::abs(std::max(step_a.tile_max, step_b.tile_max) -
                       direct.tile_max) <= 1.0e-5F);

        // Exact bit-for-bit integer monoid laws
        const auto a_i   = rand_seg_i();
        const auto b_i   = rand_seg_i();
        const auto c_i   = rand_seg_i();
        const auto lid_i = kadane_combine(id_i, a_i);
        const auto rid_i = kadane_combine(a_i, id_i);
        CHECK(lid_i.sum == a_i.sum);
        CHECK(lid_i.pfx == a_i.pfx);
        CHECK(lid_i.sfx == a_i.sfx);
        CHECK(lid_i.best == a_i.best);
        CHECK(rid_i.sum == a_i.sum);
        CHECK(rid_i.pfx == a_i.pfx);
        CHECK(rid_i.sfx == a_i.sfx);
        CHECK(rid_i.best == a_i.best);

        const auto lhs_i = kadane_combine(kadane_combine(a_i, b_i), c_i);
        const auto rhs_i = kadane_combine(a_i, kadane_combine(b_i, c_i));
        CHECK(lhs_i.sum == rhs_i.sum);
        CHECK(lhs_i.pfx == rhs_i.pfx);
        CHECK(lhs_i.sfx == rhs_i.sfx);
        CHECK(lhs_i.best == rhs_i.best);

        const auto c0_i     = static_cast<int32_t>(dist_u(rng));
        const auto ab_i     = kadane_combine(a_i, b_i);
        const auto step_a_i = kadane_apply(c0_i, a_i);
        const auto step_b_i = kadane_apply(step_a_i.c_out, b_i);
        const auto direct_i = kadane_apply(c0_i, ab_i);
        CHECK(step_b_i.c_out == direct_i.c_out);
        CHECK(std::max(step_a_i.tile_max, step_b_i.tile_max) ==
              direct_i.tile_max);
    }
}

TEST_CASE("GPU block/warp KadaneSegF32 tree scan matches Stage-2 sequential "
          "gate across 64/128/256 threads, cold-prefix, pulses, and LagWindow",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    const std::vector<uint32_t> widths = {1,   2,   4,   8,    16,   32,  64,
                                          128, 256, 512, 1024, 2048, 4096};
    const std::vector<std::pair<SizeType, SizeType>> specs = {{1, 16}, {3, 32}};

    for (const bool use_trap : {false, true}) {
        const auto tbl  = make_tables(widths, use_trap, 5.5F, 1.0F, specs);
        const auto plan = kadane::make_kadane_gate_plan(
            widths, use_trap, specs.size(), tbl.raw_thresh);
        REQUIRE(plan.num_buckets >= 1);

        std::mt19937 rng(use_trap ? 1337U : 2026U);
        std::normal_distribution<float> gauss(0.0F, 4.0F);
        constexpr SizeType kStreamLen = 4300;
        std::vector<float> stream(kStreamLen);
        for (float& v : stream) {
            v = gauss(rng);
        }
        // Inject a narrow bright pulse and a wide pulse spanning tiles.
        stream[610] = 65.0F;
        for (SizeType i = 2100; i < 2420; ++i) {
            stream[i] += 6.0F;
        }

        auto run_emulated_stream = [&]<int BlockThreads>(uint64_t warmup) {
            constexpr SizeType kBlockTile =
                static_cast<SizeType>(BlockThreads * kItems);
            for (SizeType r = 0; r < specs.size(); ++r) {
                std::array<float, kadane::kMaxGateBuckets> gpu_c{};
                std::array<float, kadane::kMaxGateBuckets> seq_c{};
                std::array<LagWindow<float>, kadane::kMaxGateBuckets> gpu_lag{};
                std::array<LagWindow<float>, kadane::kMaxGateBuckets> seq_lag{};

                // Seed non-zero initial carry before warmup to verify cold
                // tile-start carry reset parity.
                if (warmup > 0) {
                    gpu_c.fill(25.0F);
                    seq_c.fill(25.0F);
                }

                for (SizeType t0 = 0; t0 < kStreamLen; t0 += kBlockTile) {
                    const SizeType n = std::min(kBlockTile, kStreamLen - t0);
                    const uint64_t start = t0;
                    const std::span<const float> tile(stream.data() + t0, n);

                    // Sequential reference
                    // (`ScalarBackend::scan_gate_tile_f32`)
                    const float* row_ptr = tile.data();
                    const SizeType fv =
                        warmup > start
                            ? std::min<SizeType>(
                                  n, static_cast<SizeType>(warmup - start))
                            : SizeType{0};
                    std::array<float, kadane::kMaxGateBuckets> lag_in{};
                    std::array<float, kadane::kMaxGateBuckets> seq_m{};
                    bool any_lag = false;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const uint32_t mlag = plan.max_edge_lag[k];
                        if (mlag > 0U) {
                            any_lag = true;
                            const uint64_t w_beg =
                                start > mlag ? start - mlag : 0ULL;
                            lag_in[k] = seq_lag[k].query_and_prune(w_beg);
                        }
                    }
                    const uint32_t seq_hit_mask =
                        kadane::scan_gate_tile_f32<ScalarBackend>(
                            plan, r, 1, n, &row_ptr, seq_c.data(), 1,
                            fv > 0 ? &fv : nullptr,
                            any_lag ? lag_in.data() : nullptr, 1, seq_m.data(),
                            1);
                    const bool seq_hit = (seq_hit_mask & 1U) != 0U;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        if (plan.max_edge_lag[k] > 0U) {
                            seq_lag[k].push(start + n, seq_m[k]);
                        }
                    }

                    // Emulated GPU block/warp tropical scan
                    bool gpu_hit = false;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const float b   = plan.row_bias(k, r);
                        const float cut = plan.row_cut(k, r);
                        const auto seg =
                            emulate_gpu_block_scan_f32<BlockThreads>(
                                tile, b, start, warmup, warmup > 0);
                        const float c_in =
                            (warmup > 0 && start < warmup) ? 0.0F : gpu_c[k];
                        const auto res      = kadane_apply(c_in, seg);
                        gpu_c[k]            = res.c_out;
                        float eff_m         = res.tile_max;
                        const uint32_t mlag = plan.max_edge_lag[k];
                        if (mlag > 0U) {
                            const uint64_t w_beg =
                                start > mlag ? start - mlag : 0ULL;
                            const float lmax =
                                gpu_lag[k].query_and_prune(w_beg);
                            eff_m = std::max(eff_m, lmax);
                            gpu_lag[k].push(start + n, res.tile_max);
                        }
                        if (eff_m >= cut) {
                            gpu_hit = true;
                        }
                        // Outgoing carry matches sequential reference within
                        // tight N-sample rounding bound; never under-reports
                        // hits!
                        const float tol =
                            static_cast<float>(n) * 4.0e-5F + 1.0e-5F;
                        CHECK(std::abs(gpu_c[k] - seq_c[k]) <= tol);
                        CHECK(res.tile_max + tol >= seq_m[k]);
                    }
                    if (seq_hit) {
                        CHECK(gpu_hit);
                    }
                }
            }
        };

        for (const uint64_t wu :
             {uint64_t{0}, uint64_t{256}, uint64_t{605}, uint64_t{5000}}) {
            run_emulated_stream.template operator()<64>(wu);
            run_emulated_stream.template operator()<128>(wu);
            run_emulated_stream.template operator()<256>(wu);
        }
    }
}

TEST_CASE("GPU block/warp KadaneSegI64 tree scan is bit-exact and "
          "saturation-safe vs Stage-2 int32 gate",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    const std::vector<uint32_t> widths = {1,  2,   4,   8,   16,  32,
                                          64, 128, 256, 512, 1024};
    const std::vector<std::pair<SizeType, SizeType>> specs = {{1, 8}, {2, 16}};

    for (const bool use_trap : {false, true}) {
        const auto tbl =
            make_tables(widths, use_trap, 5.0F, 15.0F, specs, true, 128.0, 0);
        const uint64_t max_samp = 255ULL * 32ULL;
        const auto plan         = kadane::make_kadane_gate_plan(
            widths, use_trap, specs.size(), tbl.raw_thresh, tbl.int_mean,
            tbl.int_thresh, true, max_samp);
        REQUIRE(plan.has_int32);

        std::mt19937 rng(use_trap ? 9001U : 7777U);
        constexpr SizeType kStreamLen = 3333;
        std::vector<uint32_t> stream(kStreamLen);
        for (SizeType r = 0; r < specs.size(); ++r) {
            const auto [leaf_len, n_chans] = specs[r];
            const double mean_u =
                128.0 * static_cast<double>(leaf_len * n_chans);
            std::normal_distribution<double> ndist(mean_u, 45.0);
            for (uint32_t& u : stream) {
                const double v = std::clamp(std::round(ndist(rng)), 0.0,
                                            static_cast<double>(max_samp));
                u              = static_cast<uint32_t>(v);
            }
            for (SizeType i = 1100; i < 1180; ++i) {
                stream[i] = static_cast<uint32_t>(max_samp);
            }

            auto check_threads = [&]<int BlockThreads>(uint64_t warmup) {
                constexpr SizeType kBlockTile =
                    static_cast<SizeType>(BlockThreads * kItems);
                std::array<int32_t, kadane::kMaxGateBuckets> gpu_c{};
                std::array<int32_t, kadane::kMaxGateBuckets> seq_c{};
                std::array<LagWindow<int32_t>, kadane::kMaxGateBuckets>
                    gpu_lag{};
                std::array<LagWindow<int32_t>, kadane::kMaxGateBuckets>
                    seq_lag{};

                for (SizeType t0 = 0; t0 < kStreamLen; t0 += kBlockTile) {
                    const SizeType n = std::min(kBlockTile, kStreamLen - t0);
                    const uint64_t start = t0;
                    const std::span<const uint32_t> tile(stream.data() + t0, n);
                    const uint32_t* row_ptr = tile.data();
                    const SizeType fv =
                        warmup > start
                            ? std::min<SizeType>(
                                  n, static_cast<SizeType>(warmup - start))
                            : SizeType{0};

                    std::array<int32_t, kadane::kMaxGateBuckets> lag_in{};
                    std::array<int32_t, kadane::kMaxGateBuckets> seq_m{};
                    bool any_lag = false;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const uint32_t mlag = plan.max_edge_lag[k];
                        if (mlag > 0U) {
                            any_lag = true;
                            const uint64_t w_beg =
                                start > mlag ? start - mlag : 0ULL;
                            lag_in[k] = seq_lag[k].query_and_prune(w_beg);
                        }
                    }

                    const uint32_t seq_mask =
                        kadane::scan_gate_tile_i32<ScalarBackend>(
                            plan, r, 1, n, &row_ptr, seq_c.data(), 1,
                            fv > 0 ? &fv : nullptr,
                            any_lag ? lag_in.data() : nullptr, 1, seq_m.data(),
                            1);
                    const bool seq_hit = (seq_mask & 1U) != 0U;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        if (plan.max_edge_lag[k] > 0U) {
                            seq_lag[k].push(start + n, seq_m[k]);
                        }
                    }

                    bool gpu_hit = false;
                    for (SizeType k = 0; k < plan.num_buckets; ++k) {
                        const int32_t ib = plan.row_ibias(k, r);
                        const int32_t ic = plan.row_icut(k, r);
                        const auto seg =
                            emulate_gpu_block_scan_i64<BlockThreads>(
                                tile, plan.int_shift, ib, start, warmup,
                                warmup > 0);
                        const int32_t c_in   = (warmup > 0 && start < warmup)
                                                   ? int32_t{0}
                                                   : gpu_c[k];
                        const auto res       = kadane_apply(c_in, seg);
                        gpu_c[k]             = clamp_kadane_i32(res.c_out);
                        const int32_t tile_m = clamp_kadane_i32(res.tile_max);
                        int32_t eff_m        = tile_m;
                        const uint32_t mlag  = plan.max_edge_lag[k];
                        if (mlag > 0U) {
                            const uint64_t w_beg =
                                start > mlag ? start - mlag : 0ULL;
                            const int32_t lmax =
                                gpu_lag[k].query_and_prune(w_beg);
                            eff_m = std::max(eff_m, lmax);
                            gpu_lag[k].push(start + n, tile_m);
                        }
                        if (eff_m >= ic) {
                            gpu_hit = true;
                        }
                        // Whenever no intra-tile kKadaneInt32Cap clamping
                        // occurred, int64_t monoid is 100% bit-exact with seq!
                        if (res.tile_max < kKadaneInt32Cap) {
                            CHECK(gpu_c[k] == seq_c[k]);
                            CHECK(tile_m >= seq_m[k]);
                        }
                    }
                    CHECK(gpu_hit == seq_hit);
                }
            };

            for (const uint64_t wu :
                 {uint64_t{0}, uint64_t{199}, uint64_t{1120}}) {
                check_threads.template operator()<64>(wu);
                check_threads.template operator()<128>(wu);
                check_threads.template operator()<256>(wu);
            }
        }
    }
}

TEST_CASE("KadaneSegI64 handles extreme kKadaneInt32Cap saturation without "
          "overflow across 1024 samples",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    constexpr int kBlockThreads = 256;
    constexpr SizeType kTileLen = 1024;
    // Construct extreme valid shifted samples near INT32_MAX / 2 that would
    // overflow signed 32-bit addition ~500x over a 1024-sample tile without
    // int64_t intra-tile segments.
    std::vector<uint32_t> extreme(kTileLen, uint32_t{1U << 29});
    const uint8_t int_shift = 1U;
    const int32_t ibias     = 1000;
    const int32_t icut      = kKadaneInt32Cap / 2;

    const auto seg = emulate_gpu_block_scan_i64<kBlockThreads>(
        extreme, int_shift, ibias, 0, 0, false);
    const auto applied  = kadane_apply(kKadaneInt32Cap, seg);
    const int32_t c_out = clamp_kadane_i32(applied.c_out);
    const int32_t m_out = clamp_kadane_i32(applied.tile_max);

    CHECK(applied.tile_max > static_cast<int64_t>(kKadaneInt32Cap) * 500LL);
    CHECK(c_out == kKadaneInt32Cap);
    CHECK(m_out == kKadaneInt32Cap);
    CHECK(m_out >= icut);
}

TEST_CASE("Shared DMT_HD LagWindow is chunk-size-invariant across small and "
          "large tiles",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    constexpr SizeType kTotal  = 2048;
    constexpr uint32_t kMaxLag = 512;
    auto rng                   = make_seeded_rng(5150U);
    std::uniform_real_distribution<float> dist(0.0F, 100.0F);
    std::vector<float> sample_peaks(kTotal);
    for (float& v : sample_peaks) {
        v = dist(rng);
    }

    for (const SizeType chunk : {SizeType{17}, SizeType{64}, SizeType{256},
                                 SizeType{512}, SizeType{1024}}) {
        LagWindow<float> win{};
        for (SizeType t0 = 0; t0 < kTotal; t0 += chunk) {
            const SizeType n = std::min(chunk, kTotal - t0);
            // Reference upper envelope of prior tile maxima whose tile_start >=
            // t0 - kMaxLag: LagWindow conservatively bounds any sample peak in
            // [t0 - kMaxLag, t0).
            float exact_needed = 0.0F;
            const SizeType lo =
                t0 > kMaxLag ? (t0 - static_cast<SizeType>(kMaxLag)) : 0;
            for (SizeType s = lo; s < t0; ++s) {
                exact_needed = std::max(exact_needed, sample_peaks[s]);
            }
            const float got = win.query_and_prune(lo);
            CHECK(got >= exact_needed);

            float tile_max = 0.0F;
            for (SizeType s = t0; s < t0 + n; ++s) {
                tile_max = std::max(tile_max, sample_peaks[s]);
            }
            win.push(t0 + n, tile_max);
        }
    }
}

TEST_CASE("DMT_HD BlockNoise cut recomputation and surrogate S/N view match "
          "host plans bit-for-bit",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    const std::vector<uint32_t> widths = {1,  2,  3,  5,   8,   13,  21,
                                          34, 55, 89, 144, 233, 377, 610};
    const std::vector<std::pair<SizeType, SizeType>> specs = {
        {1, 32}, {4, 64}, {2, 0}};

    for (const bool use_trap : {false, true}) {
        const auto tbl = make_tables(widths, use_trap, 6.0F, 1.25F, specs);
        const KadaneGatePlan gplan = kadane::make_kadane_gate_plan(
            widths, use_trap, specs.size(), tbl.raw_thresh);

        // Perturbed BlockNoise minimum thresholds (plus a dead masked row)
        std::mt19937 rng(use_trap ? 314U : 271U);
        std::uniform_real_distribution<float> scale_dist(0.65F, 1.35F);
        for (SizeType r = 0; r < specs.size(); ++r) {
            std::vector<float> min_thr(widths.size());
            for (SizeType f = 0; f < widths.size(); ++f) {
                const float base = tbl.raw_thresh[(r * widths.size()) + f];
                min_thr[f] = base >= 1.0e35F ? base : base * scale_dist(rng);
            }
            std::vector<float> host_cuts(gplan.num_buckets, 0.0F);
            gplan.recompute_row_cuts(r, min_thr, host_cuts);

            for (SizeType k = 0; k < gplan.num_buckets; ++k) {
                const float hd_cut =
                    gplan.disabled_row[r] != 0U
                        ? std::numeric_limits<float>::max()
                        : compute_bucket_cut_hd_f32(
                              gplan.buckets[k], widths.data(), widths.size(),
                              use_trap, gplan.row_bias(k, r),
                              [&](SizeType f) { return min_thr[f]; });
                CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                      std::bit_cast<uint32_t>(host_cuts[k]));
            }
        }

        // Surrogate device view S/N evaluator parity (float & integer)
        for (const bool packed : {false, true}) {
            const auto stbl = make_tables(widths, use_trap, 6.0F, 1.25F, specs,
                                          packed, 64.0, 2);
            const KadaneSurrogatePlan splan =
                packed ? kadane::make_kadane_surrogate_plan(
                             widths, use_trap, specs.size(), stbl.raw_thresh,
                             stbl.inv_sigma, stbl.int_mean, stbl.integer_unit)
                       : kadane::make_kadane_surrogate_plan(
                             widths, use_trap, specs.size(), stbl.raw_thresh,
                             stbl.inv_sigma);

            KadaneSurrogateDeviceView view{};
            view.num_passes    = splan.num_passes;
            view.n_rows        = splan.n_rows;
            view.row_stride    = splan.row_stride;
            view.n_filters     = widths.size();
            view.max_width     = splan.max_width;
            view.is_integer    = splan.is_integer;
            view.integer_unit  = splan.integer_unit;
            view.bias          = splan.bias.data();
            view.trigger       = splan.trigger.data();
            view.box_inv_sigma = splan.box_inv_sigma_lut.data();
            view.row_mean_1 =
                splan.row_mean_1.empty() ? nullptr : splan.row_mean_1.data();
            view.disabled_row = splan.disabled_row.data();
            view.w_nearest    = splan.width_nearest_filter.data();
            view.w_bracket_lo = splan.width_bracket_lo.data();
            view.w_bracket_hi = splan.width_bracket_hi.data();
            view.w_log_lerp   = splan.width_log_lerp.data();

            for (SizeType r = 0; r < specs.size(); ++r) {
                for (uint32_t w :
                     {1U, 2U, 4U, 7U, 19U, 64U, 200U, 610U, 750U}) {
                    const double raw_sum =
                        packed ? (64.0 * 4.0 * 64.0 * w + 500.0) : 42.5;
                    uint32_t host_f = 0U;
                    uint32_t dev_f  = 0U;
                    const float host_snr =
                        splan.eval_surrogate_snr(r, raw_sum, w, &host_f);
                    const float dev_snr = view.eval_surrogate_snr(
                        r, raw_sum, w, widths[0], dev_f);
                    CHECK(dev_f == host_f);
                    CHECK(std::abs(dev_snr - host_snr) <= 1.0e-5F);
                }
            }
        }
    }
}

TEST_CASE("DMT_HD wide-boxcar BlockNoise cut bit-parity up to 2^20 and "
          "LagWindow soundness beyond kCap",
          "[kadane_monoid][kadane][monoid][boxcar][cpu]") {
    const std::vector<std::pair<SizeType, SizeType>> specs = {
        {1, 1}, {8, 64}, {64, 256}, {4, 0}};

    // 1. Wide-boxcar (max_boxcar = 16384 and 1048576) bit-identical parity
    // between compute_bucket_cut_hd_f32 and KadaneGatePlan::recompute_row_cuts
    // / recompute_cuts_for_biases across normal, clamped-zero, near-overflow,
    // and dead-row threshold profiles.
    for (const SizeType max_boxcar : {SizeType{16384}, SizeType{1048576}}) {
        for (const float wtsp : {2.0F, 1.5F}) {
            const auto widths =
                plans::generate_box_width_trials(max_boxcar, wtsp);
            REQUIRE(widths.back() == max_boxcar);

            for (const bool use_trap : {false, true}) {
                const auto tbl =
                    make_tables(widths, use_trap, 6.0F, 1.0F, specs);
                const KadaneGatePlan gplan = kadane::make_kadane_gate_plan(
                    widths, use_trap, specs.size(), tbl.raw_thresh);
                REQUIRE(gplan.num_buckets == 3);
                CHECK(gplan.buckets[2].w_hi == max_boxcar);

                std::mt19937 rng(static_cast<uint32_t>(max_boxcar) ^
                                 (use_trap ? 0xA5A5U : 0x5A5AU));
                std::uniform_real_distribution<float> scale_dist(0.55F, 1.45F);

                for (SizeType r = 0; r < specs.size(); ++r) {
                    // (a) Perturbed BlockNoise minimum thresholds
                    std::vector<float> min_thr(widths.size());
                    for (SizeType f = 0; f < widths.size(); ++f) {
                        const float base =
                            tbl.raw_thresh[(r * widths.size()) + f];
                        min_thr[f] =
                            base >= 1.0e35F ? base : base * scale_dist(rng);
                    }
                    std::vector<float> host_cuts(gplan.num_buckets, -1.0F);
                    gplan.recompute_row_cuts(r, min_thr, host_cuts);
                    for (SizeType k = 0; k < gplan.num_buckets; ++k) {
                        const float hd_cut =
                            gplan.disabled_row[r] != 0U
                                ? std::numeric_limits<float>::max()
                                : compute_bucket_cut_hd_f32(
                                      gplan.buckets[k], widths.data(),
                                      widths.size(), use_trap,
                                      gplan.row_bias(k, r),
                                      [&](SizeType f) { return min_thr[f]; });
                        CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                              std::bit_cast<uint32_t>(host_cuts[k]));
                    }

                    // (b) Severe threshold drop clamping cut_d <= 0 to +0.0F
                    if (gplan.disabled_row[r] == 0U) {
                        std::vector<float> dropped_thr(widths.size());
                        for (SizeType f = 0; f < widths.size(); ++f) {
                            dropped_thr[f] =
                                tbl.raw_thresh[(r * widths.size()) + f] * 0.15F;
                        }
                        gplan.recompute_row_cuts(r, dropped_thr, host_cuts);
                        for (SizeType k = 0; k < gplan.num_buckets; ++k) {
                            const float hd_cut = compute_bucket_cut_hd_f32(
                                gplan.buckets[k], widths.data(), widths.size(),
                                use_trap, gplan.row_bias(k, r),
                                [&](SizeType f) { return dropped_thr[f]; });
                            CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                                  std::bit_cast<uint32_t>(host_cuts[k]));
                            CHECK(std::bit_cast<uint32_t>(hd_cut) == 0U);
                        }
                    }
                }

                // (c) Marginal bias (0 < tau_raw < margin) and extreme near-
                // kDeadFloor / huge-bias finiteness checks at w_hi = max_boxcar
                const SizeType nf     = widths.size();
                const uint32_t f_last = static_cast<uint32_t>(nf - 1);
                const uint32_t w_last = widths[f_last];
                const uint32_t we_last =
                    use_trap ? trapezoid_edge_width(w_last) : 0U;
                const auto tilde_last = static_cast<double>(w_last - we_last);
                const double tilde_thr =
                    static_cast<double>(tbl.raw_thresh[f_last]) *
                    (we_last == 0U ? 1.0 : 0.5);

                std::vector<float> test_biases = {
                    gplan.row_bias(0, 0),
                    gplan.row_bias(1, 0),
                    static_cast<float>((tilde_thr - 1.0e-3) / tilde_last),
                };
                const std::span<const float> row0_thr(tbl.raw_thresh.data(),
                                                      nf);
                std::vector<float> host_custom(gplan.num_buckets, -1.0F);
                kadane::recompute_cuts_for_biases(gplan, test_biases, row0_thr,
                                                  host_custom);
                for (SizeType k = 0; k < gplan.num_buckets; ++k) {
                    const float hd_cut = compute_bucket_cut_hd_f32(
                        gplan.buckets[k], widths.data(), nf, use_trap,
                        test_biases[k],
                        [&](SizeType f) { return row0_thr[f]; });
                    CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                          std::bit_cast<uint32_t>(host_custom[k]));
                }
                CHECK(std::bit_cast<uint32_t>(host_custom[2]) == 0U);

                // Near-kDeadFloor finite thresholds + huge bias overflow guard
                std::vector<float> huge_thr(nf, 9.0e34F);
                std::vector<float> huge_biases(gplan.num_buckets, 1.0e38F);
                kadane::recompute_cuts_for_biases(gplan, huge_biases, huge_thr,
                                                  host_custom);
                for (SizeType k = 0; k < gplan.num_buckets; ++k) {
                    const float hd_cut = compute_bucket_cut_hd_f32(
                        gplan.buckets[k], widths.data(), nf, use_trap,
                        huge_biases[k],
                        [&](SizeType f) { return huge_thr[f]; });
                    CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                          std::bit_cast<uint32_t>(host_custom[k]));
                    CHECK(std::bit_cast<uint32_t>(hd_cut) == 0U);
                }
            }
        }
    }

    // Full 4096-filter wide bank ending at 2^20
    {
        std::vector<uint32_t> widths_4096(4096);
        std::vector<float> thr_4096(4096);
        std::vector<float> perturbed_4096(4096);
        for (SizeType i = 0; i < 4095; ++i) {
            widths_4096[i] = static_cast<uint32_t>(i + 1);
        }
        widths_4096[4095] = 1048576U;
        for (SizeType i = 0; i < 4096; ++i) {
            thr_4096[i] = 6.0F * std::sqrt(static_cast<float>(widths_4096[i]));
            perturbed_4096[i] = thr_4096[i] * 0.91F;
        }
        const KadaneGatePlan plan_4k =
            kadane::make_kadane_gate_plan(widths_4096, true, 1, thr_4096);
        std::vector<float> host_cuts(plan_4k.num_buckets, -1.0F);
        plan_4k.recompute_row_cuts(0, perturbed_4096, host_cuts);
        for (SizeType k = 0; k < plan_4k.num_buckets; ++k) {
            const float hd_cut = compute_bucket_cut_hd_f32(
                plan_4k.buckets[k], widths_4096.data(), widths_4096.size(),
                true, plan_4k.row_bias(k, 0),
                [&](SizeType f) { return perturbed_4096[f]; });
            CHECK(std::bit_cast<uint32_t>(hd_cut) ==
                  std::bit_cast<uint32_t>(host_cuts[k]));
        }
    }

    // 2. LagWindow conservative upper-bound soundness when max_edge_lag spans
    // >> kCap (16) tiles (64 tiles for max_boxcar = 16384 @ tile = 64, and
    // 1024 tiles for max_boxcar = 1048576 @ tile = 256).
    struct LagCase {
        uint32_t max_boxcar;
        uint32_t tile_len;
    };
    for (const auto lc : {LagCase{16384U, 64U}, LagCase{1048576U, 256U}}) {
        const uint32_t max_lag   = trapezoid_edge_width(lc.max_boxcar);
        const SizeType win_tiles = max_lag / lc.tile_len;
        REQUIRE(win_tiles > LagWindow<float>::kCap);

        // (a) Strictly decreasing staircase over win_tiles + 48 tiles: forces
        // count == kCap tail-timestamp coalescing on every tile after kCap!
        const SizeType stair_tiles = win_tiles + 48;
        const SizeType quiet_tail  = win_tiles + 4;
        const SizeType total_stair = stair_tiles + quiet_tail;
        LagWindow<float> win_f{};
        LagWindow<int32_t> win_i{};
        std::vector<float> hist_f(total_stair, 0.0F);
        std::vector<int32_t> hist_i(total_stair, 0);

        for (SizeType ti = 0; ti < stair_tiles; ++ti) {
            hist_f[ti] = static_cast<float>(total_stair - ti) * 1.25F;
            hist_i[ti] = static_cast<int32_t>((total_stair - ti) * 10);
        }

        for (SizeType ti = 0; ti < total_stair; ++ti) {
            const uint64_t t0    = static_cast<uint64_t>(ti) * lc.tile_len;
            const uint64_t w_beg = t0 > max_lag ? (t0 - max_lag) : uint64_t{0};

            float exact_f   = 0.0F;
            int32_t exact_i = 0;
            const SizeType j_lo =
                ti > win_tiles ? (ti - win_tiles) : SizeType{0};
            for (SizeType j = j_lo; j < ti; ++j) {
                const uint64_t j_end =
                    static_cast<uint64_t>(j + 1) * lc.tile_len;
                if (j_end > w_beg) {
                    exact_f = std::max(exact_f, hist_f[j]);
                    exact_i = std::max(exact_i, hist_i[j]);
                }
            }

            const float got_f   = win_f.query_and_prune(w_beg);
            const int32_t got_i = win_i.query_and_prune(w_beg);
            CHECK(got_f >= exact_f);
            CHECK(got_i >= exact_i);

            const uint64_t t_end = t0 + lc.tile_len;
            win_f.push(t_end, hist_f[ti]);
            win_i.push(t_end, hist_i[ti]);
        }
        // After > max_edge_lag quiet samples past the last positive tile,
        // even coalesced entries expire cleanly back to zero.
        const uint64_t final_t0 =
            static_cast<uint64_t>(total_stair) * lc.tile_len;
        CHECK(win_f.query_and_prune(final_t0 - max_lag) == 0.0F);
        CHECK(win_i.query_and_prune(final_t0 - max_lag) == 0);

        // (b) Random + sawtooth bursts across 1500 tiles (>> kCap = 16)
        constexpr SizeType kRandTiles = 1500;
        std::mt19937 rng(lc.max_boxcar + 77U);
        std::uniform_real_distribution<float> fdist(0.0F, 500.0F);
        win_f.reset();
        win_i.reset();
        std::vector<float> rhist_f(kRandTiles, 0.0F);
        std::vector<int32_t> rhist_i(kRandTiles, 0);
        for (SizeType ti = 0; ti < kRandTiles; ++ti) {
            if ((ti % 37) < 28) {
                const float v = fdist(rng) / static_cast<float>(1 + (ti % 23));
                rhist_f[ti]   = v;
                rhist_i[ti]   = static_cast<int32_t>(v * 100.0F) + 1;
            }
        }
        for (SizeType ti = 0; ti < kRandTiles; ++ti) {
            const uint64_t t0    = static_cast<uint64_t>(ti) * lc.tile_len;
            const uint64_t w_beg = t0 > max_lag ? (t0 - max_lag) : uint64_t{0};
            float exact_f        = 0.0F;
            int32_t exact_i      = 0;
            const SizeType j_lo =
                ti > win_tiles ? (ti - win_tiles) : SizeType{0};
            for (SizeType j = j_lo; j < ti; ++j) {
                const uint64_t j_end =
                    static_cast<uint64_t>(j + 1) * lc.tile_len;
                if (j_end > w_beg) {
                    exact_f = std::max(exact_f, rhist_f[j]);
                    exact_i = std::max(exact_i, rhist_i[j]);
                }
            }
            CHECK(win_f.query_and_prune(w_beg) >= exact_f);
            CHECK(win_i.query_and_prune(w_beg) >= exact_i);
            win_f.push(t0 + lc.tile_len, rhist_f[ti]);
            win_i.push(t0 + lc.tile_len, rhist_i[ti]);
        }
    }
}

} // namespace dmt
