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
#include "dmt/boxcar.hpp"
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

// Streams `nblocks` blocks through execute_search of both make_fdmt_cpu
// (unskewed tree) and FDMTSkewCpu, requiring identical root samples.
void require_search_equal(const plans::FDMTPlan& plan,
                          bool smear,
                          bool packed,
                          bool search_u32,
                          SizeType chunk,
                          SizeType nblocks,
                          int nthreads) {
    using algorithms::detail::boxcar_integer_sample;
    using algorithms::detail::boxcar_ring0;
    using algorithms::detail::boxcar_sample;
    using algorithms::detail::BoxcarRow;
    using algorithms::detail::FDMTEngineConfig;
    using algorithms::detail::FDMTSearchInput;
    using algorithms::detail::FDMTSearchSink;
    using algorithms::detail::make_boxcar_plans;
    using algorithms::detail::make_boxcar_rows;
    using algorithms::detail::make_fdmt_cpu;

    const BoxcarConfig cfg(16, 1.5F, 6.0F, packed ? 12.0F : 1.0F,
                           PeakFindingMode::kContiguousPeakPerWidth, 4096, 256,
                           true, packed ? 8 : 32, packed ? 127.0F : 0.0F);
    const auto planes = make_boxcar_plans(plan, cfg, smear);
    const auto rows   = make_boxcar_rows(plan, planes);
    const FDMTEngineConfig ecfg{
        .use_box_smearing    = smear,
        .mode                = FDMTMode::kValid,
        .nbeams              = 1,
        .fuse_levels         = kFDMTAutoFuse,
        .int_tree            = false,
        .exec                = Exec::cpu(nthreads),
        .chunk_nsamps        = chunk,
        .max_fuse_levels     = plan.get_niters() - 1,
        .search_upper_levels = false,
        .search_nbits        = packed ? SizeType{8} : SizeType{32},
    };
    auto tree = make_fdmt_cpu(plan, ecfg);
    FDMTSkewCpu skew(plan, smear, nthreads, chunk, 4, packed ? 8 : 32, 0,
                     packed, search_u32);

    struct Collector final : FDMTSearchSink {
        const std::vector<BoxcarRow>& rows;
        SizeType root;
        SizeType nsamps;
        bool packed;
        std::vector<double> out;
        Collector(const std::vector<BoxcarRow>& r,
                  SizeType rt,
                  SizeType ns,
                  bool pk)
            : rows(r),
              root(rt),
              nsamps(ns),
              packed(pk),
              out(r.size() * ns) {}
        void consume(SizeType level, const FDMTSearchInput& in) override {
            if (level != root) {
                return;
            }
            const SizeType t_base =
                static_cast<SizeType>(in.sample_begin % nsamps);
            for (SizeType d = 0; d < rows.size(); ++d) {
                const auto& row      = rows[d];
                const SizeType ring0 = boxcar_ring0(in, row);
                for (SizeType t = 0; t < in.nsamps; ++t) {
                    out[(d * nsamps) + t_base + t] =
                        packed ? static_cast<double>(boxcar_integer_sample(
                                     in, row, 0, t, true, ring0))
                               : static_cast<double>(
                                     boxcar_sample(in, row, 0, t, true, ring0));
                }
            }
        }
    };

    for (unsigned blk = 0; blk < nblocks; ++blk) {
        const auto b =
            random_block(plan.get_nchans(), plan.get_nsamps(), 300 + blk);
        Collector expected(rows, plan.get_niters(), plan.get_nsamps(), packed);
        Collector actual(rows, plan.get_niters(), plan.get_nsamps(), packed);
        if (packed) {
            tree->execute_search(b.packed, 8, expected);
            skew.execute_search(b.packed, 8, actual);
        } else {
            tree->execute_search(b.values, expected);
            skew.execute_search(b.values, actual);
        }
        REQUIRE_THAT(actual.out, Catch::Matchers::Equals(expected.out));
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
            SECTION("execute_search float and packed integer (u32 and u64)") {
                require_search_equal(plan, true, false, true, 1024, 2, 4);
                require_search_equal(plan, true, true, true, 1024, 2, 4);
                require_search_equal(plan, true, true, false, 1024, 2, 4);
            }
        }
    }
}

namespace {

// make_fdmt_cpu engine on the skewed-window path (threshold 0) or the tree.
std::unique_ptr<algorithms::detail::FDMTEngine>
make_cpu_engine(const plans::FDMTPlan& plan, bool skew, SizeType chunk) {
    algorithms::detail::FDMTEngineConfig cfg{};
    cfg.exec               = Exec::cpu(4);
    cfg.chunk_nsamps       = chunk;
    cfg.skew_history_bytes = skew ? 0 : SIZE_MAX;
    auto e                 = algorithms::detail::make_fdmt_cpu(plan, cfg);
    REQUIRE(e->search_path().empty() == !skew);
    return e;
}

std::vector<float> run_engine(algorithms::detail::FDMTEngine& e,
                              const plans::FDMTPlan& plan,
                              const Block& b,
                              bool packed) {
    std::vector<float> dmt(plan.get_dmt_size(), -9.0F);
    if (packed) {
        e.execute(std::span<const uint8_t>(b.packed), 8, dmt);
    } else {
        e.execute(b.values, dmt);
    }
    return dmt;
}

std::vector<float> random_floats(SizeType n, unsigned seed, float offset) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0F, 1.0F);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = offset + dis(gen);
    }
    return v;
}

void require_close(const std::vector<float>& actual,
                   const std::vector<float>& expected) {
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
    INFO("worst |skew - tree| / tol = " << worst << " (scale " << scale << ")");
    REQUIRE(worst <= 1.0);
}

} // namespace

TEST_CASE("FDMT engine skewed path: wiring, history and input types",
          "[fdmt_cpu][cpu][skew]") {
    const auto plan    = make_plan(kCryopafLband, 4096, 400.0F);
    const SizeType nch = plan.get_nchans();
    const SizeType ns  = plan.get_nsamps();
    const auto blk     = [&](unsigned s) { return random_block(nch, ns, s); };
    REQUIRE(plan.get_dt_max() > 1024);

    SECTION("the engine path is bit-exact with the tree, float and packed") {
        for (const bool packed : {false, true}) {
            auto skew = make_cpu_engine(plan, true, 1024);
            auto tree = make_cpu_engine(plan, false, 1024);
            for (unsigned i = 0; i < 3; ++i) {
                const auto b = blk(10 + i);
                REQUIRE_THAT(run_engine(*skew, plan, b, packed),
                             Catch::Matchers::Equals(
                                 run_engine(*tree, plan, b, packed)));
            }
        }
    }

    SECTION("save/load resumes a stream exactly; layout is checked") {
        auto a = make_cpu_engine(plan, true, 1024);
        // The size is fixed before any input (the ring is allocated lazily).
        const SizeType hsize = a->history_state_size();
        REQUIRE(hsize > 0);
        for (const bool packed : {false, true}) {
            a->reset_history();
            run_engine(*a, plan, blk(20), packed);
            run_engine(*a, plan, blk(21), packed);
            REQUIRE(a->history_state_size() == hsize);
            std::vector<float> hist(hsize);
            a->save_history(hist);
            const auto ref = run_engine(*a, plan, blk(22), packed);
            // A fresh instance, and the same instance after another stream.
            auto c = make_cpu_engine(plan, true, 1024);
            c->load_history(hist);
            REQUIRE_THAT(run_engine(*c, plan, blk(22), packed),
                         Catch::Matchers::Equals(ref));
            a->reset_history();
            run_engine(*a, plan, blk(99), !packed);
            a->load_history(hist);
            REQUIRE_THAT(run_engine(*a, plan, blk(22), packed),
                         Catch::Matchers::Equals(ref));
            // Round trip: save after load gives the same buffer back.
            std::vector<float> again(hsize);
            c->load_history(hist);
            c->save_history(again);
            REQUIRE_THAT(again, Catch::Matchers::Equals(hist));
        }
        // A different skewed layout (chunk) or a non-skewed buffer throws.
        auto other = make_cpu_engine(plan, true, 2048);
        std::vector<float> hist(hsize);
        a->save_history(hist);
        if (other->history_state_size() == hsize) {
            REQUIRE_THROWS_AS(other->load_history(hist), std::invalid_argument);
        }
        std::vector<float> junk(hsize, 0.5F);
        REQUIRE_THROWS_AS(a->load_history(junk), std::invalid_argument);
    }

    SECTION("switching input type keeps the stream (matches the tree)") {
        // float (integers) -> packed -> float -> packed, and the reverse.
        // Rows straddling a switch mix the two box-mean conventions in the
        // tree (its stored rows keep the previous block's) but take the
        // current block's in the skewed engine (recomputed from samples):
        // they differ by the packed rounding, at most half an LSB per
        // channel, and only within dt_max samples of the switch. Every
        // later sample is bit-identical, so no history was lost.
        const auto t_edge  = static_cast<SizeType>(plan.get_dt_max()) + 1;
        const double bound = 0.5 * static_cast<double>(nch);
        for (const bool first_packed : {false, true}) {
            auto skew = make_cpu_engine(plan, true, 1024);
            auto tree = make_cpu_engine(plan, false, 1024);
            for (unsigned i = 0; i < 4; ++i) {
                const bool packed = first_packed == (i % 2 == 0);
                const auto b      = blk(30 + i);
                const auto xs     = run_engine(*skew, plan, b, packed);
                const auto xt     = run_engine(*tree, plan, b, packed);
                SizeType late     = 0;
                double worst      = 0.0;
                for (SizeType q = 0; q < xs.size(); ++q) {
                    const double d = std::abs(static_cast<double>(xs[q]) -
                                              static_cast<double>(xt[q]));
                    late += (q % ns >= t_edge && d != 0.0) ? 1 : 0;
                    worst = std::max(worst, d);
                }
                INFO("block " << i << " packed " << packed << " worst " << worst
                              << " bound " << bound);
                REQUIRE(late == 0);
                REQUIRE(worst <= (i == 0 ? 0.0 : bound));
            }
        }
    }

    SECTION("integer blocks after non-integer history stay within rounding") {
        // An integer block (here zeros: a dropped-data gap) must not take
        // the integer running sums while non-integer samples are in reach.
        auto skew = make_cpu_engine(plan, true, 1024);
        auto tree = make_cpu_engine(plan, false, 1024);
        std::vector<float> a(plan.get_dmt_size());
        std::vector<float> b(plan.get_dmt_size());
        const auto nonint = random_floats(nch * ns, 5, 3.0F);
        const std::vector<float> zeros(nch * ns, 0.0F);
        const auto ints = blk(40).values;
        for (const auto* wf : {&nonint, &zeros, &ints, &ints}) {
            skew->execute(*wf, a);
            tree->execute(*wf, b);
            require_close(a, b);
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
                        "mean %10.1f sum %12zu\n",
                        l, st.rows, st.used, st.max_spread,
                        st.used ? static_cast<double>(st.sum_spread) /
                                      static_cast<double>(st.used)
                                : 0.0,
                        st.sum_spread);
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
