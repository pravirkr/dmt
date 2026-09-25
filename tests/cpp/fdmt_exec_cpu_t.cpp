#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "test_helpers.hpp"

#include <catch2/matchers/catch_matchers_all.hpp>

// Performance parameters (fuse_levels / int_tree), construction-time memory,
// and packed low-bit input. Every variant must be bit-identical to the
// original level-by-level float path (fuse_levels = 0, int_tree = false), so
// all comparisons here are exact.

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::kFDMTAutoFuse;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

// fuse_levels / int_tree of the original unfused, all-float path.
constexpr SizeType kUnfused = 0;
constexpr bool kFloatTree   = false;

std::vector<float> random_waterfall(SizeType n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0F, 1.0F);
    std::vector<float> wf(n);
    for (auto& v : wf) {
        v = dis(gen);
    }
    return wf;
}

// Random nbits-wide unsigned samples, as float (reference) and packed rows.
struct PackedWaterfall {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

PackedWaterfall
random_packed(SizeType nrows, SizeType nsamps, SizeType nbits, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(
        0, bit_pack_utils::max_sample_value(nbits));
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    PackedWaterfall out{.values = std::vector<float>(nrows * nsamps),
                        .packed = std::vector<uint8_t>(nrows * row_bytes, 0)};
    for (SizeType r = 0; r < nrows; ++r) {
        uint8_t* row = out.packed.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const uint32_t v             = dis(gen);
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

std::vector<float> run(FDMTCPU& fdmt, const std::vector<float>& wf) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(wf, dmt);
    return dmt;
}

std::vector<float>
run_packed(FDMTCPU& fdmt, const std::vector<uint8_t>& wf, SizeType nbits) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const uint8_t>(wf), nbits, dmt);
    return dmt;
}

// Compares the leading dmt_size values of every beam.
void require_beams_exact(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         const FDMTCPU& fdmt) {
    const auto bufsz = fdmt.get_plan().get_buffer_size();
    const auto n     = fdmt.get_plan().get_dmt_size();
    for (SizeType b = 0; b < fdmt.get_nbeams(); ++b) {
        const auto a = std::span(actual).subspan(b * bufsz, n);
        const auto e = std::span(expected).subspan(b * bufsz, n);
        REQUIRE_THAT(
            std::vector<float>(a.begin(), a.end()),
            Catch::Matchers::Equals(std::vector<float>(e.begin(), e.end())));
    }
}

} // namespace

TEST_CASE("FDMTCPU performance parameters and memory usage",
          "[fdmt_cpu][cpu]") {
    FDMTCPU fdmt(kFMin, kFMax, 32, 128, kTsamp, 16);
    const auto niters = fdmt.get_plan().get_niters();
    CHECK(fdmt.get_int_tree());
    CHECK(fdmt.get_fuse_levels() <= niters);

    FDMTCPU clamped(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
                    false, 1, 1, 99);
    CHECK(clamped.get_fuse_levels() == niters);
    FDMTCPU orig(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid", false,
                 1, 1, kUnfused, kFloatTree);
    CHECK(orig.get_fuse_levels() == 0);
    CHECK_FALSE(orig.get_int_tree());

    // Everything is accounted for, and scratch grows with the depth.
    const auto& plan = clamped.get_plan();
    const auto mem   = clamped.get_memory_usage();
    CHECK(mem.plan == plan.get_container().get_memory_usage());
    CHECK(mem.state == plan.get_buffer_size() * sizeof(float));
    CHECK(mem.history == clamped.history_state_size() * sizeof(float));
    CHECK(mem.output == plan.get_buffer_size() * sizeof(float));
    CHECK(mem.total() == mem.plan + mem.state + mem.history + mem.workspace);
    CHECK(mem.workspace > orig.get_memory_usage().workspace);
    CHECK(orig.get_memory_usage().workspace > 0); // unpack + box rows

    // Per-thread scratch: exactly one slice per thread.
    FDMTCPU one(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid", false,
                1, 1, 3);
    FDMTCPU two(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid", false,
                2, 1, 3);
    CHECK(two.get_memory_usage().workspace ==
          2 * one.get_memory_usage().workspace);
}

TEST_CASE("FDMTCPU automatic fusion depth", "[fdmt_cpu][cpu]") {
    // kFDMTAutoFuse picks the deepest depth whose two per-thread fusion
    // buffers fit max(36 MiB / nthreads, 5 MiB): a rule on the plan alone.
    // The fusion buffers are what the workspace holds beyond the three
    // unpack/box rows of each thread.
    const SizeType nchans  = 1024;
    const SizeType nsamps  = 4096;
    const auto fused_bytes = [&](const FDMTCPU& f, SizeType nthreads) {
        const auto rows = 3 * (((nsamps * sizeof(float)) + 63) / 64) * 64;
        return (f.get_memory_usage().workspace / nthreads) - rows;
    };
    SizeType prev_depth = SIZE_MAX;
    for (const int nthreads : {1, 2, 4, 8}) {
        DYNAMIC_SECTION("nthreads=" << nthreads) {
            const auto n = static_cast<SizeType>(nthreads);
            const SizeType budget =
                std::max<SizeType>((SizeType{36} << 20) / n, SizeType{5} << 20);
            FDMTCPU fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 256, 0, 1, true,
                         "valid", false, nthreads);
            const auto depth = fdmt.get_fuse_levels();
            REQUIRE(depth >= 1);
            REQUIRE(depth <= prev_depth); // more threads never fuse deeper
            prev_depth = depth;
            CHECK(fused_bytes(fdmt, n) <= budget);
            if (depth < fdmt.get_plan().get_niters()) {
                FDMTCPU deeper(kFMin, kFMax, nchans, nsamps, kTsamp, 256, 0, 1,
                               true, "valid", false, nthreads, 1, depth + 1);
                CHECK(fused_bytes(deeper, n) > budget);
            }
        }
    }
}

TEST_CASE("FDMTCPU level-0 box rows match a float64 reference",
          "[fdmt_cpu][cpu]") {
    // A large DC offset over a long block is where the former serial
    // running sum (+= x[i]; -= x[i-w]) drifted; the direct box sum keeps each
    // output within a few ulps of its own (<= w-term) sum.
    const SizeType nchans = 16;
    const SizeType nsamps = 1 << 16;
    for (const std::string mode : {"full", "roll", "valid"}) {
        DYNAMIC_SECTION("mode=" << mode) {
            FDMTCPU fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 90, 60, 1, true,
                         mode);
            std::mt19937 gen(5);
            std::normal_distribution<float> dis(1000.0F, 1.0F);
            std::vector<float> wf(nchans * nsamps);
            for (auto& v : wf) {
                v = dis(gen);
            }
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
                    // Skip the block-start boundary (history / wrap / zero
                    // padding is covered by the streaming tests).
                    for (SizeType i = w; i < nsamps; ++i) {
                        double ref = 0.0;
                        for (SizeType k = 0; k <= w; ++k) {
                            ref += static_cast<double>(x[i - k]);
                        }
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

TEST_CASE("FDMTCPU fused levels are bit-exact with the original path",
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
                    FDMTCPU ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                c.dt_max, c.dt_min, 1, smearing, mode, false, 1,
                                2, kUnfused, kFloatTree);
                    const auto expected = run(ref, wf);
                    for (const SizeType fuse :
                         {SizeType{1}, SizeType{2}, SizeType{3}, SizeType{9},
                          kFDMTAutoFuse}) {
                        FDMTCPU fdmt(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                     c.dt_max, c.dt_min, 1, smearing, mode,
                                     false, 1, 2, fuse);
                        require_beams_exact(run(fdmt, wf), expected, fdmt);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTCPU fused levels: dt_grid, streaming, packed int tree",
          "[fdmt_cpu][cpu]") {
    SECTION("explicit dt_grid") {
        const std::vector<IndexType> dt_grid = {0, 1, 3, 7, 12, 20, 31, 45};
        const auto wf                        = random_waterfall(64 * 200, 8);
        FDMTCPU ref(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid",
                    false, 1, 1, kUnfused, kFloatTree);
        FDMTCPU fdmt(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid",
                     false, 1, 1, 3);
        require_beams_exact(run(fdmt, wf), run(ref, wf), fdmt);
    }
    SECTION("valid-mode streaming, fused and unfused engines alternating") {
        // One stream handed between a fused and an unfused engine through
        // save_history()/load_history(): the history is depth-independent.
        const SizeType nchans = 32;
        for (const SizeType block : {16, 64}) {
            FDMTCPU ref(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                        "valid", false, 1, 1, kUnfused, kFloatTree);
            FDMTCPU fused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                          "valid", false, 1, 1, 2);
            FDMTCPU unfused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1,
                            true, "valid", false, 1, 1, kUnfused);
            std::vector<float> hist(ref.history_state_size(), 0.0F);
            for (SizeType b = 0; b < 10; ++b) {
                FDMTCPU& engine = (b % 3 == 2) ? unfused : fused;
                const auto wf   = random_waterfall(nchans * block,
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
                FDMTCPU ref(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true,
                            mode, false, 1, 1, kUnfused, kFloatTree);
                FDMTCPU fdmt(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true,
                             mode, false, 1, 1, 3, true);
                require_beams_exact(run_packed(fdmt, wf.packed, nbits),
                                    run(ref, wf.values), fdmt);
            }
        }
    }
    SECTION("stepper stays unfused and inspectable") {
        FDMTCPU fdmt(kFMin, kFMax, 32, 128, kTsamp, 16, 0, 1, true, "valid",
                     false, 1, 1, 3);
        const auto wf = random_waterfall(32 * 128, 2);
        std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
        fdmt.reset(wf, dmt);
        REQUIRE(fdmt.current_level() == 0);
        REQUIRE_NOTHROW(fdmt.view_level_data());
        fdmt.finalize();
    }
}

TEST_CASE("FDMTCPU packed input matches float input", "[fdmt_cpu][cpu]") {
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
                        FDMTCPU ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                    c.dt_max, c.dt_min, 1, smearing, mode,
                                    false, 1, 1, kUnfused, kFloatTree);
                        const auto expected = run(ref, wf.values);
                        for (const bool int_tree : {false, true}) {
                            for (const SizeType fuse :
                                 {SizeType{0}, kFDMTAutoFuse}) {
                                FDMTCPU fdmt(kFMin, kFMax, c.nchans, c.nsamps,
                                             kTsamp, c.dt_max, c.dt_min, 1,
                                             smearing, mode, false, 1, 1, fuse,
                                             int_tree);
                                require_beams_exact(
                                    run_packed(fdmt, wf.packed, nbits),
                                    expected, fdmt);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTCPU packed valid-mode streaming and multiple beams",
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
                FDMTCPU ref(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                            c.dt_min, 1, true, "valid", false, 1, nbeams,
                            kUnfused, kFloatTree);
                // Default performance parameters: int tree + auto fusion.
                FDMTCPU fdmt(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                             c.dt_min, 1, true, "valid", false, 1, nbeams);
                for (SizeType b = 0; b < nblks; ++b) {
                    const auto wf =
                        random_packed(nbeams * nchans, block, nbits,
                                      static_cast<unsigned>(100 + b));
                    REQUIRE(wf.packed.size() == nbeams * nchans * row_bytes);
                    const auto expected = run(ref, wf.values);
                    require_beams_exact(run_packed(fdmt, wf.packed, nbits),
                                        expected, fdmt);
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

TEST_CASE("FDMTCPU packed input errors and integer-level inspection",
          "[fdmt_cpu][cpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 128;
    FDMTCPU fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full");
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
    FDMTCPU flt(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full",
                false, 1, 1, kFDMTAutoFuse, kFloatTree);
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
