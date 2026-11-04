#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_skew.hpp"

// Skewed-window streaming FDMT prototype (lib/dmt/fdmt_skew.hpp): on the
// benchmark presets' geometry it must reproduce the streaming tree engine
// bit for bit across consecutive blocks, for float input holding integers
// and for packed 8-bit input, at any chunk length and switch level.

namespace dmt {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;
using algorithms::detail::FDMTSkewCpu;

namespace {

struct Preset {
    std::string name;
    float f_min;
    float f_max;
    float tsamp;
    SizeType nchans;
    float levin_tol;
};

// bench/suite/suite_common.hpp presets.
const Preset kCryopafLband{"cryopaf_lband", 1100.0F, 1714.0F,
                           455.6e-6F,       1152,    1.1F};
const Preset kMeerkatLband{"meerkat_lband", 856.0F, 1712.0F,
                           612.48e-6F,      4096,   1.1F};
const Preset kParkesUwl{"parkes_uwl", 704.0F, 4032.0F, 64e-6F, 6656, 1.6F};
const Preset kCryopaf{"cryopaf", 704.0F, 1216.0F, 5.12e-4F, 512, 1.1F};

plans::FDMTPlan make_plan(const Preset& p, SizeType nsamps, float dm_max) {
    const auto grid = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, dm_max, p.tsamp, p.tsamp, p.f_min, p.f_max, p.nchans,
        p.levin_tol);
    return {p.f_min, p.f_max, p.nchans, nsamps, p.tsamp, grid, "valid"};
}

// 8-bit filterbank values as floats, and the same values packed.
struct Block {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

Block random_block(SizeType nchans, SizeType nsamps, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 255);
    Block b{.values = std::vector<float>(nchans * nsamps),
            .packed = std::vector<uint8_t>(nchans * nsamps)};
    for (SizeType i = 0; i < b.values.size(); ++i) {
        const auto v = dis(gen);
        b.values[i]  = static_cast<float>(v);
        b.packed[i]  = static_cast<uint8_t>(v);
    }
    return b;
}

std::vector<float> run_tree(FDMT& fdmt, const Block& b, bool packed) {
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), -9.0F);
    if (packed) {
        fdmt.execute(std::span<const uint8_t>(b.packed), 8, dmt);
    } else {
        fdmt.execute(b.values, dmt);
    }
    return dmt;
}

std::vector<float> run_skew(FDMTSkewCpu& skew,
                            const plans::FDMTPlan& plan,
                            const Block& b,
                            bool packed) {
    std::vector<float> dmt(plan.get_dmt_size(), -9.0F);
    if (packed) {
        skew.execute(std::span<const uint8_t>(b.packed), 8, dmt);
    } else {
        skew.execute(b.values, dmt);
    }
    return dmt;
}

// Non-integer float input (double running sums): within float rounding.
void require_streams_close(const plans::FDMTPlan& plan,
                           SizeType nblocks,
                           int nthreads) {
    FDMT tree(plan, true, Exec::cpu(nthreads), 1, kFDMTAutoFuse, true, 1024);
    FDMTSkewCpu skew(plan, true, nthreads, 1024, 4);
    std::mt19937 gen(7);
    std::uniform_real_distribution<float> dis(-1.0F, 1.0F);
    for (unsigned blk = 0; blk < nblocks; ++blk) {
        std::vector<float> wf(plan.get_nchans() * plan.get_nsamps());
        for (auto& v : wf) {
            v = dis(gen);
        }
        std::vector<float> expected(plan.get_dmt_size(), -9.0F);
        std::vector<float> actual(plan.get_dmt_size(), -9.0F);
        tree.execute(wf, expected);
        skew.execute(wf, actual);
        // Each channel's box is summed in a different order (float shifted
        // adds against a double running sum): a few 1e-7 of its own
        // magnitude per channel, summed over nchans channels.
        float scale = 0.0F;
        for (const auto v : expected) {
            scale = std::max(scale, std::abs(v));
        }
        const double tol = (1e-5 * static_cast<double>(scale)) + 1e-4;
        double worst     = 0.0;
        for (SizeType i = 0; i < expected.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(actual[i]) -
                                             static_cast<double>(expected[i])) /
                                        tol);
        }
        INFO("block " << blk << " worst |skew - tree| / tol = " << worst
                      << " (scale " << scale << ")");
        REQUIRE(worst <= 1.0);
        REQUIRE(scale > 0.0F);
    }
}

// Streams `nblocks` blocks through both engines and requires equality.
void require_streams_equal(const plans::FDMTPlan& plan,
                           bool smear,
                           bool packed,
                           SizeType level,
                           SizeType chunk_tree,
                           SizeType chunk_skew,
                           SizeType nblocks,
                           int nthreads) {
    FDMT tree(plan, smear, Exec::cpu(nthreads), 1, kFDMTAutoFuse, true,
              chunk_tree);
    FDMTSkewCpu skew(plan, smear, nthreads, chunk_skew, level, packed ? 8 : 32);
    for (unsigned blk = 0; blk < nblocks; ++blk) {
        const auto b =
            random_block(plan.get_nchans(), plan.get_nsamps(), 100 + blk);
        const auto expected = run_tree(tree, b, packed);
        const auto actual   = run_skew(skew, plan, b, packed);
        INFO("block " << blk << " level " << level << " chunk " << chunk_skew
                      << " packed " << packed << " smear " << smear);
        // The tree wrote every sample (the fill is gone) and the sum is
        // the band's: a filterbank mean of 127.5 over nchans channels.
        REQUIRE(
            std::ranges::none_of(expected, [](float v) { return v == -9.0F; }));
        REQUIRE(expected.back() > 0.0F);
        REQUIRE_THAT(actual, Catch::Matchers::Equals(expected));
    }
}

void print_estimate(const Preset& p,
                    const plans::FDMTPlan& plan,
                    SizeType level,
                    SizeType chunk,
                    SizeType ring_elem,
                    int nthreads) {
    using algorithms::detail::estimate_skew_memory;
    const auto e =
        estimate_skew_memory(plan, level, chunk, ring_elem, nthreads);
    const auto gb = [](SizeType b) { return static_cast<double>(b) / 1.0e9; };
    std::printf("  %s K=%zu chunk=%zu ring_elem=%zu: ring %zu samples %.2f GB, "
                "state %.2f GB, rings %.3f GB, stripes %.3f GB, total %.2f GB "
                "| recompute %.1f%% | tree: state %.2f GB + history %.2f GB = "
                "%.2f GB\n",
                p.name.c_str(), level, chunk, ring_elem, e.ring_nsamps,
                gb(e.ring), gb(e.state), gb(e.rings), gb(e.stripes),
                gb(e.total()), 100.0 * e.recompute_fraction, gb(e.tree_state),
                gb(e.tree_history), gb(e.tree_total()));
}

} // namespace

TEST_CASE("FDMT skewed-window streaming is bit-exact with the tree engine",
          "[fdmt_cpu][cpu][skew]") {
    // Preset band, channel count and sampling with the DM range cut so the
    // case runs in seconds; the pruned piecewise Levin grid keeps the tree
    // structure of the full search. Delays span several chunks.
    struct Case {
        const Preset* preset;
        float dm_max;
        SizeType nsamps;
    };
    const std::vector<Case> cases = {{&kCryopafLband, 400.0F, 4096},
                                     {&kMeerkatLband, 300.0F, 4096}};
    for (const auto& c : cases) {
        const auto plan = make_plan(*c.preset, c.nsamps, c.dm_max);
        DYNAMIC_SECTION(c.preset->name << " dt_max " << plan.get_dt_max()
                                       << " trials " << plan.get_dmt_ndms()) {
            REQUIRE(plan.get_dt_max() > 1024);
            SECTION("float input holding integers, smear, chunk 1024") {
                require_streams_equal(plan, true, false, 4, 1024, 1024, 3, 4);
            }
            SECTION("packed 8-bit input, smear") {
                require_streams_equal(plan, true, true, 4, 1024, 1024, 3, 4);
            }
            SECTION("no box smearing, float and packed") {
                require_streams_equal(plan, false, false, 4, 1024, 1024, 2, 4);
                require_streams_equal(plan, false, true, 4, 1024, 1024, 2, 4);
            }
            SECTION("skew chunk 4096 vs tree chunk 1024, level 3 and 5") {
                require_streams_equal(plan, true, false, 3, 1024, 4096, 2, 4);
                require_streams_equal(plan, true, true, 5, 2048, 4096, 2, 4);
            }
            SECTION("uneven chunk") {
                require_streams_equal(plan, true, false, 4, 1024, 1000, 2, 4);
            }
            SECTION("non-integer float input agrees within rounding") {
                require_streams_close(plan, 2, 4);
            }
        }
    }
}

TEST_CASE("FDMT skewed-window: full presets", "[.skew_preset]") {
    for (const Preset* p : {&kCryopafLband, &kMeerkatLband}) {
        const auto plan = make_plan(*p, 32768, 2000.0F);
        DYNAMIC_SECTION(p->name) {
            require_streams_equal(plan, true, false, 4, 4096, 4096, 2, 8);
            require_streams_equal(plan, true, true, 4, 4096, 4096, 2, 8);
        }
    }
}

TEST_CASE("FDMT skewed-window: window statistics of the presets",
          "[.skew_report]") {
    for (const Preset* p :
         {&kCryopaf, &kCryopafLband, &kMeerkatLband, &kParkesUwl}) {
        const auto plan = make_plan(*p, 32768, 2000.0F);
        const auto w    = algorithms::detail::compute_skew_windows(plan, 1);
        std::printf("%s: nchans %zu dt_max %ld trials %zu niters %zu "
                    "tree_history %zu floats level0 history %zu floats\n",
                    p->name.c_str(), plan.get_nchans(), plan.get_dt_max(),
                    plan.get_dmt_ndms(), plan.get_niters(),
                    plan.get_tree_history_size(), plan.get_history_size());
        for (SizeType l = 1; l <= plan.get_niters(); ++l) {
            const auto st = w.stats(l);
            std::printf("  level %2zu: rows %9zu used %9zu spread max %8zu "
                        "mean %10.1f sum %12zu max_s_hi %8zu\n",
                        l, st.rows, st.used, st.max_spread,
                        st.used ? static_cast<double>(st.sum_spread) /
                                      static_cast<double>(st.used)
                                : 0.0,
                        st.sum_spread, st.max_s_hi);
        }
        for (const SizeType chunk : {SizeType{1024}, SizeType{4096}}) {
            print_estimate(*p, plan, 4, chunk, 1, 8);
            print_estimate(*p, plan, 4, chunk, 4, 8);
        }
        print_estimate(*p, plan, 3, 4096, 1, 8);
        print_estimate(*p, plan, 5, 4096, 1, 8);
        std::fflush(stdout);
    }
}

} // namespace dmt
