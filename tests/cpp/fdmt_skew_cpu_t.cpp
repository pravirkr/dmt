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

#include "skew_subband_search.hpp"

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
const Preset kCryopaf{"cryopaf_low", 704.0F, 1216.0F, 5.12e-4F, 512, 1.1F};

plans::FDMTPlan make_plan(const Preset& p,
                          SizeType nsamps,
                          float dm_max,
                          SizeType subband_depth = 0) {
    const auto grid = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, dm_max, p.tsamp, p.tsamp, p.f_min, p.f_max, p.nchans,
        p.levin_tol);
    return {p.f_min, p.f_max, p.nchans, nsamps,
            p.tsamp, grid,    "valid",  subband_depth};
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
                           int nthreads,
                           SizeType leaf = 0) {
    FDMT tree(plan, true, Exec::cpu(nthreads), 1, kFDMTAutoFuse, true, 1024);
    FDMTSkewCpu skew(plan, true, nthreads, 1024, 4, 32, 0, false, true, 0,
                     leaf);
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
                           int nthreads,
                           SizeType leaf = 0) {
    FDMT tree(plan, smear, Exec::cpu(nthreads), 1, kFDMTAutoFuse, true,
              chunk_tree);
    FDMTSkewCpu skew(plan, smear, nthreads, chunk_skew, level, packed ? 8 : 32,
                     0, false, true, 0, leaf);
    if (leaf != 0) {
        REQUIRE(skew.get_leaf_level() == leaf);
        REQUIRE(skew.get_fuse_level() > leaf);
    }
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
                          int nthreads,
                          SizeType leaf = 0) {
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
                     packed, search_u32, 0, leaf);

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
            SECTION("leaf levels 1 and 2 below fusion from level 4") {
                // The budget-chosen leaf levels (FDMTCpuEngine).
                require_streams_equal(plan, true, false, 4, 1024, 4096, 2, 4,
                                      1);
                require_streams_equal(plan, true, true, 4, 1024, 2048, 2, 4, 2);
                require_streams_equal(plan, true, true, 4, 1024, 1024, 2, 4, 1);
                require_search_equal(plan, true, true, true, 1024, 2, 4, 1);
                require_search_equal(plan, true, false, true, 1024, 2, 4, 2);
            }
            SECTION("non-integer float input agrees within rounding") {
                require_streams_close(plan, 2, 4);
                require_streams_close(plan, 2, 4, 1);
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
        // The float size before any input (the ring is allocated lazily) and
        // once float input has arrived.
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
            // Round trip: save after load gives the same buffer back on an
            // instance whose ring holds float input (a). The fresh c took
            // the saved 8-bit samples into a packed ring (compact layout)
            // and kept it unless it then ran a float block.
            std::vector<float> again(hsize);
            a->load_history(hist);
            a->save_history(again);
            REQUIRE_THAT(again, Catch::Matchers::Equals(hist));
            REQUIRE((c->history_state_size() < hsize) == packed);
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

    SECTION("packed streams save a compact history; either layout loads") {
        // 8-bit: three samples per 24-bit float word; 16-bit: three in two.
        const auto pack16 = [&](unsigned seed) {
            std::mt19937 gen(seed);
            std::uniform_int_distribution<uint32_t> dis(0, 65535);
            std::vector<uint8_t> bytes(2 * nch * ns);
            for (SizeType i = 0; i < nch * ns; ++i) {
                const auto v     = dis(gen);
                bytes[2 * i]     = static_cast<uint8_t>(v & 0xFFU);
                bytes[2 * i + 1] = static_cast<uint8_t>(v >> 8);
            }
            return bytes;
        };
        for (const SizeType nbits : {SizeType{8}, SizeType{16}}) {
            CAPTURE(nbits);
            const auto run = [&](algorithms::detail::FDMTEngine& e,
                                 unsigned seed) {
                std::vector<float> dmt(plan.get_dmt_size(), -9.0F);
                if (nbits == 8) {
                    e.execute(std::span<const uint8_t>(blk(seed).packed), 8,
                              dmt);
                } else {
                    const auto bytes = pack16(seed);
                    e.execute(std::span<const uint8_t>(bytes), 16, dmt);
                }
                return dmt;
            };
            auto a                = make_cpu_engine(plan, true, 1024);
            const SizeType hfloat = a->history_state_size();
            run(*a, 60);
            run(*a, 61);
            const SizeType hsize = a->history_state_size();
            // N ring samples: N float words, ceil(N / 3) * nbits / 8 packed.
            const SizeType saved = hfloat - hsize;
            REQUIRE(hsize < hfloat);
            REQUIRE(saved > (nbits == 8 ? hfloat / 2 : hfloat / 4));
            std::vector<float> hist(hsize);
            a->save_history(hist);
            for (SizeType i = 16; i < 1024; ++i) { // packed ring words
                REQUIRE(hist[i] == std::floor(hist[i]));
                REQUIRE(hist[i] < 16777216.0F);
            }
            const auto ref = run(*a, 62);
            // A fresh instance (float-built, ring not yet allocated).
            auto c = make_cpu_engine(plan, true, 1024);
            c->load_history(hist);
            REQUIRE(c->history_state_size() == hsize);
            REQUIRE_THAT(run(*c, 62), Catch::Matchers::Equals(ref));
            // Round trip.
            c->load_history(hist);
            std::vector<float> again(hsize);
            c->save_history(again);
            REQUIRE_THAT(again, Catch::Matchers::Equals(hist));
            // An instance whose ring already holds float input keeps its
            // float layout and decodes the packed ring into it.
            auto f = make_cpu_engine(plan, true, 1024);
            run_engine(*f, plan, blk(63), false);
            REQUIRE(f->history_state_size() == hfloat);
            f->load_history(hist);
            REQUIRE(f->history_state_size() == hfloat);
            REQUIRE_THAT(run(*f, 62), Catch::Matchers::Equals(ref));
            // ... and its float-layout save loads back into a packed ring.
            f->load_history(hist);
            std::vector<float> fhist(hfloat);
            f->save_history(fhist);
            auto g = make_cpu_engine(plan, true, 1024);
            g->load_history(fhist);
            REQUIRE(g->history_state_size() == hsize);
            REQUIRE_THAT(run(*g, 62), Catch::Matchers::Equals(ref));
            // Altered words and wrong sizes throw, leaving the stream alone.
            for (const float bad : {0.5F, 16777216.0F, -1.0F}) {
                auto broken = hist;
                broken[17]  = bad;
                REQUIRE_THROWS_AS(c->load_history(broken),
                                  std::invalid_argument);
            }
            std::vector<float> shorter(hist.begin(), hist.end() - 1);
            REQUIRE_THROWS_AS(c->load_history(shorter), std::invalid_argument);
            if (nbits == 8) {
                // A packed-built (8-bit) engine is compact from the start.
                FDMTSkewCpu p8(plan, true, 4, 1024, 4, 8);
                REQUIRE(p8.history_state_size() < hfloat);
            }
        }
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

    SECTION("a packed-only skew engine rejects float input") {
        // Ring widening (packed -> float) happens only on float-capable
        // instances (nbits 32, as the engine builds them for transforms);
        // an instance built for packed input refuses float blocks rather
        // than widening its ring under 4-byte running-sum stripes.
        FDMTSkewCpu skew(plan, true, 4, 1024, 4, 8);
        const auto b0 = blk(50);
        std::vector<float> a(plan.get_dmt_size());
        skew.execute(std::span<const uint8_t>(b0.packed), 8, a);
        REQUIRE_THROWS_AS(skew.execute(b0.values, a), std::invalid_argument);
    }
}

TEST_CASE("FDMTBoxcar skewed path: subband search matches the tree",
          "[fdmt_cpu][cpu][skew][boxcar]") {
    using skew_search_test::require_same_before;
    using skew_search_test::run_subband_search;
    for (const SizeType depth : {SizeType{1}, SizeType{3}, SizeType{5}}) {
        const auto plan = make_plan(kCryopafLband, 4096, 600.0F, depth);
        for (const bool packed : {false, true}) {
            DYNAMIC_SECTION("depth " << depth << " packed " << packed) {
                constexpr SizeType kBlocks = 4;
                // Leaf level 1 (float) and 2 (packed), as the budget picks
                // them on wide plans.
                const auto skew =
                    run_subband_search(plan, Exec::cpu(4), true, packed,
                                       kBlocks, 1024, packed ? 2 : 1);
                const auto tree = run_subband_search(plan, Exec::cpu(4), false,
                                                     packed, kBlocks);
                REQUIRE_FALSE(skew.path.empty());
                REQUIRE(tree.path.empty());
                // Lags are at most the root's largest delay.
                const uint64_t cutoff =
                    (kBlocks * plan.get_nsamps()) -
                    static_cast<uint64_t>(plan.get_dt_max()) - 64;
                require_same_before(skew.candidates, tree.candidates, cutoff);
            }
        }
    }
}

TEST_CASE("FDMTBoxcar skewed path: drain() searches the lagged tail",
          "[fdmt_cpu][cpu][skew][boxcar]") {
    // flush() leaves each subband row's last `lag` samples unsearched on the
    // skewed path; drain() searches them, so the two paths then agree on
    // every candidate of the stream, and none lies past its end.
    using skew_search_test::require_same_before;
    using skew_search_test::run_subband_search;
    constexpr SizeType kBlocks = 3;
    for (const SizeType depth : {SizeType{1}, SizeType{3}}) {
        const auto plan  = make_plan(kCryopafLband, 4096, 600.0F, depth);
        const uint64_t n = kBlocks * plan.get_nsamps();
        for (const bool packed : {false, true}) {
            DYNAMIC_SECTION("depth " << depth << " packed " << packed) {
                const auto skew =
                    run_subband_search(plan, Exec::cpu(4), true, packed,
                                       kBlocks, 1024, packed ? 2 : 1, true);
                const auto tree = run_subband_search(
                    plan, Exec::cpu(4), false, packed, kBlocks, 1024, 0, true);
                REQUIRE(skew.latency > 0);
                REQUIRE(tree.latency == 0);
                REQUIRE(skew.searched == n);
                REQUIRE(tree.searched == n);
                for (const auto& c : skew.candidates) {
                    REQUIRE(c.end_sample < n);
                }
                require_same_before(skew.candidates, tree.candidates,
                                    UINT64_MAX);
                // Without drain() the skewed path misses some of them.
                const auto flushed =
                    run_subband_search(plan, Exec::cpu(4), true, packed,
                                       kBlocks, 1024, packed ? 2 : 1, false);
                REQUIRE(flushed.searched == n - skew.latency);
                REQUIRE(flushed.candidates.size() < skew.candidates.size());
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

#ifdef DMT_ENABLE_GPU
namespace dmt {

// Subband search on the GPU windowed top (fdmt_cuda.cu
// configure_windowed_top) against the CPU tree path. Host code, so the
// Catch2 macros stay out of nvcc.
TEST_CASE("FDMTBoxcar (gpu) windowed top: subband search matches CPU tree",
          "[fdmt_cuda][cuda][gpu][skew][boxcar]") {
    using skew_search_test::require_same_before;
    using skew_search_test::run_subband_search;
    for (const SizeType depth : {SizeType{1}, SizeType{3}, SizeType{5}}) {
        const auto plan = make_plan(kCryopafLband, 4096, 600.0F, depth);
        for (const bool packed : {false, true}) {
            DYNAMIC_SECTION("depth " << depth << " packed " << packed) {
                constexpr SizeType kBlocks = 4;
                const auto gpu = run_subband_search(plan, Exec::cuda(0), true,
                                                    packed, kBlocks);
                const auto cpu = run_subband_search(plan, Exec::cpu(4), false,
                                                    packed, kBlocks);
                INFO("gpu path: " << gpu.path);
                REQUIRE(gpu.path.find("windowed") != std::string::npos);
                const uint64_t cutoff =
                    (kBlocks * plan.get_nsamps()) -
                    static_cast<uint64_t>(plan.get_dt_max()) - 64;
                require_same_before(gpu.candidates, cpu.candidates, cutoff,
                                    packed ? 0.0F : 2e-5F);
            }
        }
    }
}

TEST_CASE("FDMTBoxcar (gpu) windowed top: drain() searches the lagged tail",
          "[fdmt_cuda][cuda][gpu][skew][boxcar]") {
    using skew_search_test::require_same_before;
    using skew_search_test::run_subband_search;
    constexpr SizeType kBlocks = 3;
    const auto plan            = make_plan(kCryopafLband, 4096, 600.0F, 3);
    const uint64_t n           = kBlocks * plan.get_nsamps();
    for (const bool packed : {false, true}) {
        DYNAMIC_SECTION("packed " << packed) {
            const auto gpu = run_subband_search(plan, Exec::cuda(0), true,
                                                packed, kBlocks, 1024, 0, true);
            const auto cpu = run_subband_search(plan, Exec::cpu(4), false,
                                                packed, kBlocks, 1024, 0, true);
            INFO("gpu path: " << gpu.path);
            REQUIRE(gpu.path.find("windowed") != std::string::npos);
            REQUIRE(gpu.latency > 0);
            REQUIRE(gpu.searched == n);
            for (const auto& c : gpu.candidates) {
                REQUIRE(c.end_sample < n);
            }
            require_same_before(gpu.candidates, cpu.candidates, UINT64_MAX,
                                packed ? 0.0F : 2e-5F);
        }
    }
}

} // namespace dmt
#endif

namespace dmt {

TEST_CASE("FDMT memory budget picks the path and rejects what cannot fit",
          "[fdmt_cpu][cpu][skew][budget]") {
    // DM 2000 on a 4096-sample block: the tree history is large next to
    // the skewed engine's ring and windows.
    const auto plan = make_plan(kCryopafLband, 16384, 2000.0F);
    FDMT tree(plan, true, Exec::cpu(4));
    REQUIRE(tree.get_memory_usage().total() > 0);
    const SizeType tree_bytes = tree.get_memory_usage().total();
    auto forced               = make_cpu_engine(plan, true, 1024);
    const SizeType skew_bytes = forced->get_memory_usage().total();
    INFO("tree " << tree_bytes << " B, skew (chunk 1024) " << skew_bytes
                 << " B");
    REQUIRE(skew_bytes < tree_bytes);

    SECTION("a budget below the tree path takes the skewed path") {
        const SizeType budget = (tree_bytes + skew_bytes) / 2;
        FDMT fdmt(plan, true, Exec::cpu(4), 1, kFDMTAutoFuse, true, 0, budget);
        CHECK(fdmt.get_memory_usage().total() <= budget);
        CHECK(fdmt.get_chunk_nsamps() < plan.get_nsamps());
        // Same output as the tree, bit for bit (integer-valued input).
        const auto b = random_block(plan.get_nchans(), plan.get_nsamps(), 3);
        REQUIRE_THAT(run_tree(fdmt, b, false),
                     Catch::Matchers::Equals(run_tree(tree, b, false)));
    }
    SECTION("nothing fits: FDMT and FDMTBoxcar throw") {
        REQUIRE_THROWS_AS(FDMT(plan, true, Exec::cpu(4), 1, kFDMTAutoFuse, true,
                               0, SizeType{1} << 20),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(algorithms::FDMTBoxcar(plan, BoxcarConfig(64), true,
                                                 Exec::cpu(4), 1, kFDMTAutoFuse,
                                                 0, SizeType{1} << 20),
                          std::invalid_argument);
    }
#ifdef DMT_ENABLE_GPU
    SECTION("GPU: nothing fits throws") {
        REQUIRE_THROWS_AS(FDMT(plan, true, Exec::cuda(0), 1, kFDMTAutoFuse,
                               true, 0, SizeType{1} << 20),
                          std::invalid_argument);
    }
#endif
}

} // namespace dmt
