#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/fdmt.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"

#include "test_helpers.hpp"
#include "test_random.hpp"

// Performance parameters (fuse_levels / int_tree), construction-time memory,
// and packed low-bit input. Every variant must be bit-identical to the
// original level-by-level float path (fuse_levels = 0, int_tree = false), so
// all comparisons here are exact.

namespace dmt {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

// fuse_levels / int_tree of the original unfused, all-float path.
constexpr SizeType kUnfused = 0;
constexpr bool kFloatTree   = false;

std::vector<float> random_waterfall(SizeType n, unsigned seed) {
    return dmt::test_random::uniform_real<float>(n, seed, -1.0F, 1.0F);
}

// Random nbits-wide unsigned samples, as float (reference) and packed rows.
struct PackedWaterfall {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

PackedWaterfall
random_packed(SizeType nrows, SizeType nsamps, SizeType nbits, unsigned seed) {
    const auto vals = dmt::test_random::uniform_int<uint32_t>(
        nrows * nsamps, seed, 0, bit_pack_utils::max_sample_value(nbits));
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    PackedWaterfall out{.values = std::vector<float>(nrows * nsamps),
                        .packed = std::vector<uint8_t>(nrows * row_bytes, 0)};
    for (SizeType r = 0; r < nrows; ++r) {
        uint8_t* row = out.packed.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const uint32_t v             = vals[(r * nsamps) + s];
            out.values[(r * nsamps) + s] = static_cast<float>(v);
            switch (nbits) {
            case 1:
                bit_pack_utils::write_packed_sample<1>(row, s, v);
                break;
            case 2:
                bit_pack_utils::write_packed_sample<2>(row, s, v);
                break;
            case 4:
                bit_pack_utils::write_packed_sample<4>(row, s, v);
                break;
            case 8:
                bit_pack_utils::write_packed_sample<8>(row, s, v);
                break;
            default:
                bit_pack_utils::write_packed_sample<16>(row, s, v);
                break;
            }
        }
    }
    return out;
}

std::vector<float> run(FDMT& fdmt, const std::vector<float>& wf) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(wf, dmt);
    return dmt;
}

std::vector<float>
run_packed(FDMT& fdmt, const std::vector<uint8_t>& wf, SizeType nbits) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const uint8_t>(wf), nbits, dmt);
    return dmt;
}

// Compares the leading dmt_size values of every beam.
void require_beams_exact(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         const FDMT& fdmt) {
    const auto bufsz = fdmt.get_plan().get_dmt_size(); // beam stride
    const auto n     = fdmt.get_plan().get_dmt_size();
    for (SizeType b = 0; b < fdmt.get_nbeams(); ++b) {
        const auto a = std::span(actual).subspan(b * bufsz, n);
        const auto e = std::span(expected).subspan(b * bufsz, n);
        REQUIRE_THAT(
            std::vector<float>(a.begin(), a.end()),
            Catch::Matchers::Equals(std::vector<float>(e.begin(), e.end())));
    }
}

// Largest |packed - float| of every trial with box smearing: packed input
// rounds each channel's box mean to f(L_c) fractional bits (at most
// 2^-(f + 1) each, see fdmt_box_weights.hpp); float input does not.
std::vector<double> packed_rounding_bound(const FDMT& fdmt) {
    const auto& plan    = fdmt.get_plan();
    const auto nchans   = plan.get_nchans();
    const auto smearing = plan.get_smearing_grid_final();
    std::vector<double> bound(plan.get_dmt_ndms(), 0.0);
    for (SizeType i = 0; i < bound.size(); ++i) {
        for (SizeType c = 0; c < nchans; ++c) {
            const auto len =
                static_cast<SizeType>(smearing[(i * nchans) + c]) + 1;
            bound[i] += std::ldexp(
                1.0,
                -static_cast<int>(algorithms::detail::box_mean_bits(len)) - 1);
        }
    }
    return bound;
}

// Compares the leading dmt_size values of every beam within the per-trial
// rounding bound of packed_rounding_bound() (plus float round-off).
void require_beams_near(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        const FDMT& fdmt) {
    const auto bufsz  = fdmt.get_plan().get_dmt_size(); // beam stride
    const auto ndms   = fdmt.get_plan().get_dmt_ndms();
    const auto nsamps = fdmt.get_plan().get_dmt_nsamps();
    const auto bound  = packed_rounding_bound(fdmt);
    double worst      = 0.0; // fraction of the bound used
    for (SizeType b = 0; b < fdmt.get_nbeams(); ++b) {
        for (SizeType i = 0; i < ndms; ++i) {
            for (SizeType t = 0; t < nsamps; ++t) {
                const auto k     = (b * bufsz) + (i * nsamps) + t;
                const double e   = expected[k];
                const double d   = std::abs(static_cast<double>(actual[k]) - e);
                const double tol = bound[i] + (1e-6 * std::abs(e)) + 1e-6;
                worst            = std::max(worst, d / tol);
            }
        }
    }
    INFO("worst |packed - float| / bound = " << worst);
    REQUIRE(worst <= 1.0);
}

} // namespace

TEST_CASE("FDMT performance parameters and memory usage", "[fdmt_cpu][cpu]") {
    FDMT fdmt(kFMin, kFMax, 32, 128, kTsamp, 16);
    const auto niters = fdmt.get_plan().get_niters();
    CHECK(fdmt.get_int_tree());
    CHECK(fdmt.get_fuse_levels() <= niters);

    FDMT clamped(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
                 Exec::cpu(1), 1, 99);
    CHECK(clamped.get_fuse_levels() == niters);
    FDMT orig(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
              Exec::cpu(1), 1, kUnfused, kFloatTree);
    CHECK(orig.get_fuse_levels() == 0);
    CHECK_FALSE(orig.get_int_tree());

    // Everything is accounted for, and scratch grows with the depth.
    // The state is two halves of the largest level below the root (the
    // root goes straight to the output, which is exactly dmt_size).
    const auto& plan = clamped.get_plan();
    const auto mem   = clamped.get_memory_usage();
    SizeType largest = 0;
    for (SizeType l = 0; l < plan.get_niters(); ++l) {
        largest =
            std::max(largest, plan.get_container().state_shape[l].nelements);
    }
    CHECK(mem.plan > plan.get_container().get_memory_usage());
    CHECK(mem.state == 2 * largest * sizeof(float));
    CHECK(mem.history == clamped.history_state_size() * sizeof(float));
    CHECK(mem.output == plan.get_dmt_size() * sizeof(float));
    CHECK(mem.total() == mem.plan + mem.state + mem.history + mem.workspace);
    CHECK(mem.workspace > orig.get_memory_usage().workspace);
    CHECK(orig.get_memory_usage().workspace > 0); // unpack + box rows

    // Per-thread scratch: exactly one slice per thread.
    FDMT one(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
             Exec::cpu(1), 1, 3);
    FDMT two(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
             Exec::cpu(2), 1, 3);
    CHECK(two.get_memory_usage().workspace ==
          2 * one.get_memory_usage().workspace);
}

TEST_CASE("FDMT automatic fusion depth", "[fdmt_cpu][cpu]") {
    // Modes "full" and "roll": kFDMTAutoFuse picks the deepest depth whose
    // two per-thread fusion buffers fit max(36 MiB / nthreads, 8 MiB): a
    // rule on the plan alone. The fusion buffers are what the workspace
    // holds beyond the three unpack/box rows of each thread.
    const SizeType nchans  = 1024;
    const SizeType nsamps  = 4096;
    const auto fused_bytes = [&](const FDMT& f, SizeType nthreads) {
        const auto rows = 3 * (((nsamps * sizeof(float)) + 63) / 64) * 64;
        return (f.get_memory_usage().workspace / nthreads) - rows;
    };
    SizeType prev_depth = SIZE_MAX;
    for (const int nthreads : {1, 2, 4, 8}) {
        DYNAMIC_SECTION("full, nthreads=" << nthreads) {
            const auto n = static_cast<SizeType>(nthreads);
            const SizeType budget =
                std::max<SizeType>((SizeType{36} << 20) / n, SizeType{8} << 20);
            FDMT fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 256, 0, 1, true,
                      "full", Exec::cpu(nthreads));
            const auto depth = fdmt.get_fuse_levels();
            REQUIRE(depth >= 1);
            REQUIRE(depth <= prev_depth); // more threads never fuse deeper
            prev_depth = depth;
            CHECK(fused_bytes(fdmt, n) <= budget);
            if (depth < fdmt.get_plan().get_niters()) {
                FDMT deeper(kFMin, kFMax, nchans, nsamps, kTsamp, 256, 0, 1,
                            true, "full", Exec::cpu(nthreads), 1, depth + 1);
                CHECK(fused_bytes(deeper, n) > budget);
            }
        }
    }
}

TEST_CASE("FDMT automatic fusion depth in valid mode (time tiles)",
          "[fdmt_cpu][cpu]") {
    // Mode "valid" fuses in time tiles: the deepest depth F whose mean
    // per-group working set with 256-sample tiles -- two scratch buffers of
    // the group's widest fused level, plus the tree-history rings of its
    // fused merges, in float -- is at most 1.75 MiB, with at least two
    // level-F groups per thread (not required of F = 1). Recomputed here
    // from the plan alone.
    constexpr double kBudget = 1792.0 * 1024.0;
    const auto working_set   = [](const plans::FDMTPlan& plan, SizeType fuse,
                                  SizeType tile) {
        const auto& pc         = plan.get_container();
        const SizeType ngroups = pc.state_shape[fuse].nchans;
        double total           = 0.0;
        for (SizeType g = 0; g < ngroups; ++g) {
            SizeType rows = 0;
            SizeType ring = 0;
            for (SizeType l = 0; l <= fuse; ++l) {
                const auto& grids    = pc.grids[l];
                const SizeType width = SizeType{1} << (fuse - l);
                const SizeType first = g * width;
                const SizeType last = std::min(first + width, grids.size()) - 1;
                const SizeType cb   = grids[first].coord_offset;
                const SizeType ce = grids[last].coord_offset + grids[last].ndt;
                if (l < fuse) {
                    rows = std::max(rows, ce - cb);
                }
                for (SizeType k = cb; l > 0 && k < ce; ++k) {
                    const auto& c = pc.coordinates[l][k];
                    ring += (c.i_coord_head != SIZE_MAX) ? c.delay : 0;
                }
            }
            total += static_cast<double>((2 * rows * tile) + ring) * 4.0;
        }
        return total / static_cast<double>(ngroups);
    };
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
    };
    const std::vector<Case> cases = {{1024, 4096, 256}, {256, 4096, 3000}};
    for (const auto& c : cases) {
        SizeType prev_depth = SIZE_MAX;
        for (const int nthreads : {1, 2, 4, 8}) {
            DYNAMIC_SECTION("nchans=" << c.nchans << " dt_max=" << c.dt_max
                                      << " nthreads=" << nthreads) {
                FDMT fdmt(kFMin, kFMax, c.nchans, c.nsamps, kTsamp, c.dt_max, 0,
                          1, true, "valid", Exec::cpu(nthreads));
                const auto& plan   = fdmt.get_plan();
                const auto& shapes = plan.get_container().state_shape;
                const auto depth   = fdmt.get_fuse_levels();
                REQUIRE(depth >= 1);
                REQUIRE(depth <= prev_depth);
                prev_depth = depth;
                CHECK(working_set(plan, depth, 256) <= kBudget);
                const auto next = depth + 1;
                if (next <= plan.get_niters()) {
                    CHECK((working_set(plan, next, 256) > kBudget ||
                           shapes[next].nchans <
                               2 * static_cast<SizeType>(nthreads)));
                }
                CHECK_THAT(fdmt.summary(),
                           Catch::Matchers::ContainsSubstring("time tiles of"));
            }
        }
    }
}

TEST_CASE("FDMT level-0 box rows match a float64 reference",
          "[fdmt_cpu][cpu]") {
    // A large DC offset over a long block is where the former serial
    // running sum (+= x[i]; -= x[i-w]) drifted; the direct box sum keeps each
    // output within a few ulps of its own (<= w-term) sum. Rows hold the
    // box sum times the box-mean weight round(2^b / L) * 2^-b.
    const SizeType nchans = 16;
    const SizeType nsamps = 1 << 16;
    for (const std::string mode : {"full", "roll", "valid"}) {
        DYNAMIC_SECTION("mode=" << mode) {
            FDMT fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 90, 60, 1, true,
                      mode);
            const auto wf = dmt::test_random::normal<float>(nchans * nsamps, 5,
                                                            1000.0F, 1.0F);
            std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
            fdmt.reset(wf, dmt);
            const auto level0 = fdmt.view_level_data();
            const auto& grids = fdmt.get_plan().get_container().grids[0];
            double worst_ulps = 0.0;
            for (SizeType c = 0; c < nchans; ++c) {
                const float* x = wf.data() + (c * nsamps);
                for (SizeType i_dt = 0; i_dt < grids[c].ndt; ++i_dt) {
                    const auto w =
                        static_cast<SizeType>(std::abs(grids[c].dt_grid[i_dt]));
                    REQUIRE(w >= 1);
                    const float* row =
                        level0.data() +
                        ((grids[c].coord_offset + i_dt) * nsamps);
                    const double weight =
                        test::box_mean_weight(fdmt.get_plan(), w + 1);
                    // Skip the block-start boundary (history / wrap / zero
                    // padding is covered by the streaming tests).
                    for (SizeType i = w; i < nsamps; ++i) {
                        double ref = 0.0;
                        for (SizeType k = 0; k <= w; ++k) {
                            ref += static_cast<double>(x[i - k]);
                        }
                        ref *= weight;
                        const double ulp =
                            std::nextafter(static_cast<float>(ref), 1e30F) -
                            static_cast<float>(ref);
                        worst_ulps = std::max(
                            worst_ulps,
                            std::abs(static_cast<double>(row[i]) - ref) / ulp);
                    }
                }
            }
            fdmt.finalize();
            INFO("worst error = " << worst_ulps << " ulp");
            // w <= ~7 terms here: a handful of roundings, never drift.
            REQUIRE(worst_ulps <= 8.0);
        }
    }
}

TEST_CASE("FDMT fused levels are bit-exact with the original path",
          "[fdmt_cpu][cpu]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    const std::vector<Case> cases = {
        {64, 256, 48, 0},   // even tree
        {37, 100, 40, 0},   // odd channel counts -> copy sub-bands
        {64, 256, 32, -32}, // symmetric DM range
        {32, 128, -4, -40}, // purely negative
        {16, 128, 90, 60},  // wide level-0 boxes
        {4, 64, 8, 0},      // fewer levels than the requested fusion depth
    };
    const std::vector<std::string> modes = {"full", "roll", "valid"};
    for (const auto& c : cases) {
        for (const auto& mode : modes) {
            for (const bool smearing : {true, false}) {
                DYNAMIC_SECTION("nchans=" << c.nchans << " dt=[" << c.dt_min
                                          << "," << c.dt_max << "] mode="
                                          << mode << " smearing=" << smearing) {
                    const auto wf = random_waterfall(2 * c.nchans * c.nsamps,
                                                     21 + c.nchans);
                    FDMT ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp, c.dt_max,
                             c.dt_min, 1, smearing, mode, Exec::cpu(1), 2,
                             kUnfused, kFloatTree);
                    const auto expected = run(ref, wf);
                    for (const SizeType fuse :
                         {SizeType{1}, SizeType{2}, SizeType{3}, SizeType{9},
                          kFDMTAutoFuse}) {
                        FDMT fdmt(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                  c.dt_max, c.dt_min, 1, smearing, mode,
                                  Exec::cpu(1), 2, fuse);
                        require_beams_exact(run(fdmt, wf), expected, fdmt);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMT fused levels: dt_grid, streaming, packed int tree",
          "[fdmt_cpu][cpu]") {
    SECTION("explicit dt_grid") {
        const std::vector<IndexType> dt_grid = {0, 1, 3, 7, 12, 20, 31, 45};
        const auto wf                        = random_waterfall(64 * 200, 8);
        FDMT ref(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid",
                 Exec::cpu(1), 1, kUnfused, kFloatTree);
        FDMT fdmt(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid",
                  Exec::cpu(1), 1, 3);
        require_beams_exact(run(fdmt, wf), run(ref, wf), fdmt);
    }
    SECTION("valid-mode streaming, fused and unfused engines alternating") {
        // One stream handed between a fused and an unfused engine through
        // save_history()/load_history(): the history is depth-independent.
        const SizeType nchans = 32;
        for (const SizeType block : {16, 64}) {
            FDMT ref(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                     "valid", Exec::cpu(1), 1, kUnfused, kFloatTree);
            FDMT fused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                       "valid", Exec::cpu(1), 1, 2);
            FDMT unfused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                         "valid", Exec::cpu(1), 1, kUnfused);
            std::vector<float> hist(ref.history_state_size(), 0.0F);
            ref.save_history(hist); // a cold stream's history
            for (SizeType b = 0; b < 10; ++b) {
                FDMT& engine  = (b % 3 == 2) ? unfused : fused;
                const auto wf = random_waterfall(nchans * block,
                                                 static_cast<unsigned>(60 + b));
                engine.load_history(hist);
                const auto got = run(engine, wf);
                engine.save_history(hist);
                require_beams_exact(got, run(ref, wf), engine);
            }
            std::vector<float> h_ref(ref.history_state_size());
            ref.save_history(h_ref);
            REQUIRE_THAT(hist, Catch::Matchers::Equals(h_ref));
        }
    }
    SECTION("packed input with int tree") {
        for (const SizeType nbits : {1, 4, 8}) {
            for (const std::string mode : {"full", "roll", "valid"}) {
                const auto wf = random_packed(256, 128, nbits, 30 + nbits);
                // Packed reference: the unfused float tree (same rounded box
                // means), bit-exact; float input within the rounding bound.
                FDMT ref(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true, mode,
                         Exec::cpu(1), 1, kUnfused, kFloatTree);
                FDMT fdmt(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true, mode,
                          Exec::cpu(1), 1, 3, true);
                const auto got = run_packed(fdmt, wf.packed, nbits);
                require_beams_exact(got, run_packed(ref, wf.packed, nbits),
                                    fdmt);
                ref.reset_history(); // a fresh stream for the float input
                require_beams_near(got, run(ref, wf.values), fdmt);
            }
        }
    }
    SECTION("stepper stays unfused and inspectable") {
        FDMT fdmt(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
                  Exec::cpu(1), 1, 3);
        const auto wf = random_waterfall(32 * 128, 2);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
        fdmt.reset(wf, dmt);
        REQUIRE(fdmt.current_level() == 0);
        REQUIRE_NOTHROW(fdmt.view_level_data());
        fdmt.finalize();
    }
}

TEST_CASE("FDMT packed input matches float input", "[fdmt_cpu][cpu]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    // nchans=256 with low nbits drives levels through uint8 -> uint16 ->
    // float; nsamps=203 exercises a partial trailing byte for nbits < 8.
    // dt_min > 0 / purely negative ranges give level-0 rows of width > 1,
    // exercising the exact vectorized box kernel's shifted-add passes.
    const std::vector<Case> cases        = {{64, 203, 40, 0},
                                            {37, 128, 24, -24},
                                            {256, 128, 64, 0},
                                            {16, 128, 90, 60},
                                            {16, 100, -30, -80}};
    const std::vector<std::string> modes = {"full", "roll", "valid"};
    for (const auto& c : cases) {
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            for (const auto& mode : modes) {
                for (const bool smearing : {true, false}) {
                    DYNAMIC_SECTION("nchans=" << c.nchans << " nbits=" << nbits
                                              << " mode=" << mode
                                              << " smearing=" << smearing) {
                        const auto wf = random_packed(c.nchans, c.nsamps, nbits,
                                                      17 + nbits);
                        FDMT ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                 c.dt_max, c.dt_min, 1, smearing, mode,
                                 Exec::cpu(1), 1, kUnfused, kFloatTree);
                        // Without box smearing packed input is exactly the
                        // float input; with it, the box means are rounded
                        // (fdmt_box_weights.hpp): within the rounding bound
                        // of float input, and bit-identical across int_tree
                        // and fusion.
                        const auto expected_f32 = run(ref, wf.values);
                        ref.reset_history(); // a fresh stream
                        const auto expected_pk =
                            run_packed(ref, wf.packed, nbits);
                        if (smearing) {
                            require_beams_near(expected_pk, expected_f32, ref);
                        } else {
                            require_beams_exact(expected_pk, expected_f32, ref);
                        }
                        for (const bool int_tree : {false, true}) {
                            for (const SizeType fuse :
                                 {SizeType{0}, kFDMTAutoFuse}) {
                                FDMT fdmt(kFMin, kFMax, c.nchans, c.nsamps,
                                          kTsamp, c.dt_max, c.dt_min, 1,
                                          smearing, mode, Exec::cpu(1), 1, fuse,
                                          int_tree);
                                require_beams_exact(
                                    run_packed(fdmt, wf.packed, nbits),
                                    expected_pk, fdmt);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMT packed valid-mode streaming and multiple beams",
          "[fdmt_cpu][cpu]") {
    struct Case {
        SizeType nchans;
        IndexType dt_max;
        IndexType dt_min;
    };
    const SizeType block  = 24;
    const SizeType nblks  = 8;
    const SizeType nbeams = 2;
    for (const auto& c : {Case{64, 40, 0}, Case{16, 90, 60}}) {
        for (const SizeType nbits : {1, 2, 8}) {
            DYNAMIC_SECTION("nchans=" << c.nchans << " dt_min=" << c.dt_min
                                      << " nbits=" << nbits) {
                const SizeType nchans = c.nchans;
                const auto row_bytes =
                    bit_pack_utils::packed_row_bytes(block, nbits);
                FDMT ref(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                         c.dt_min, 1, true, "valid", Exec::cpu(1), nbeams,
                         kUnfused, kFloatTree);
                // Default performance parameters: int tree + auto fusion.
                FDMT fdmt(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                          c.dt_min, 1, true, "valid", Exec::cpu(1), nbeams);
                // Float input on a third engine: within the rounding bound.
                FDMT ref_f32(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                             c.dt_min, 1, true, "valid", Exec::cpu(1), nbeams,
                             kUnfused, kFloatTree);
                for (SizeType b = 0; b < nblks; ++b) {
                    const auto wf =
                        random_packed(nbeams * nchans, block, nbits,
                                      static_cast<unsigned>(100 + b));
                    REQUIRE(wf.packed.size() == nbeams * nchans * row_bytes);
                    const auto expected = run_packed(ref, wf.packed, nbits);
                    const auto got      = run_packed(fdmt, wf.packed, nbits);
                    require_beams_exact(got, expected, fdmt);
                    require_beams_near(got, run(ref_f32, wf.values), fdmt);
                }
                std::vector<float> h_ref(ref.history_state_size());
                std::vector<float> h_pk(fdmt.history_state_size());
                ref.save_history(h_ref);
                fdmt.save_history(h_pk);
                REQUIRE_THAT(h_pk, Catch::Matchers::Equals(h_ref));
            }
        }
    }
}

TEST_CASE("FDMT packed input errors and integer-level inspection",
          "[fdmt_cpu][cpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 128;
    FDMT fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full");
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
    const auto wf = random_packed(nchans, nsamps, 1, 9);

    REQUIRE_THROWS_AS(fdmt.execute(std::span<const uint8_t>(wf.packed), 3, dmt),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fdmt.execute(std::span<const uint8_t>(wf.packed), 2, dmt),
                      std::invalid_argument);
    std::vector<float> small(8);
    REQUIRE_THROWS_AS(
        fdmt.execute(std::span<const uint8_t>(wf.packed), 1, small),
        std::invalid_argument);

    // Without int_tree every level is float and inspectable.
    FDMT flt(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full",
             Exec::cpu(1), 1, kFDMTAutoFuse, kFloatTree);
    flt.reset(std::span<const uint8_t>(wf.packed), 1, dmt);
    REQUIRE_NOTHROW(flt.view_level_data());
    flt.finalize();

    // With int_tree (the default) the 1-bit, 64-channel, no-smearing levels
    // start as uint8 (bound 1) and cannot be viewed; the root is always
    // float.
    fdmt.reset(std::span<const uint8_t>(wf.packed), 1, dmt);
    REQUIRE_THROWS_AS(fdmt.view_level_data(), std::logic_error);
    REQUIRE_THROWS_AS(fdmt.view_subband(0), std::logic_error);
    fdmt.advance_until_remaining(0);
    REQUIRE_NOTHROW(fdmt.view_level_data());
    fdmt.finalize();
}

} // namespace dmt
