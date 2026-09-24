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

// FDMTExecConfig (schedule / tiling / int_tree) and packed low-bit input.
// Every execution variant must be bit-identical to the default kCoord float
// path, so all comparisons here are exact.

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTExecConfig;
using algorithms::FDMTSchedule;
using algorithms::FDMTStreamingStores;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

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
    std::uniform_int_distribution<uint32_t> dis(0,
                                                utils::max_sample_value(nbits));
    const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
    PackedWaterfall out{.values = std::vector<float>(nrows * nsamps),
                        .packed = std::vector<uint8_t>(nrows * row_bytes, 0)};
    for (SizeType r = 0; r < nrows; ++r) {
        uint8_t* row = out.packed.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const uint32_t v             = dis(gen);
            out.values[(r * nsamps) + s] = static_cast<float>(v);
            switch (nbits) {
            case 1:
                utils::write_packed_sample<1>(row, s, v);
                break;
            case 2:
                utils::write_packed_sample<2>(row, s, v);
                break;
            case 4:
                utils::write_packed_sample<4>(row, s, v);
                break;
            case 8:
                utils::write_packed_sample<8>(row, s, v);
                break;
            default:
                utils::write_packed_sample<16>(row, s, v);
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

FDMTExecConfig tiled(SizeType tile_nsamps, SizeType tile_ndt) {
    return {.schedule    = FDMTSchedule::kTiled,
            .tile_nsamps = tile_nsamps,
            .tile_ndt    = tile_ndt};
}

const std::vector<FDMTExecConfig>& tiled_configs() {
    static const std::vector<FDMTExecConfig> configs = {
        tiled(37, 1), tiled(64, 3), tiled(0, 0), tiled(4096, 32)};
    return configs;
}

std::vector<float> column_block(const std::vector<float>& wf,
                                SizeType nrows,
                                SizeType total,
                                SizeType start,
                                SizeType len) {
    std::vector<float> block(nrows * len);
    for (SizeType r = 0; r < nrows; ++r) {
        std::copy_n(wf.data() + (r * total) + start, len,
                    block.data() + (r * len));
    }
    return block;
}

} // namespace

TEST_CASE("FDMTCPU exec config defaults and validation", "[fdmt_cpu][cpu]") {
    FDMTCPU fdmt(kFMin, kFMax, 32, 128, kTsamp, 16);
    CHECK(fdmt.get_exec_config() == FDMTExecConfig{});
    CHECK(fdmt.get_effective_tile_nsamps() >= 512);

    fdmt.set_exec_config(tiled(100, 5));
    CHECK(fdmt.get_exec_config().schedule == FDMTSchedule::kTiled);
    CHECK(fdmt.get_effective_tile_nsamps() == 100);

    // Cannot switch mid-transform.
    const auto wf = random_waterfall(32 * 128, 1);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
    fdmt.reset(wf, dmt);
    fdmt.advance(1);
    REQUIRE_THROWS_AS(fdmt.set_exec_config(FDMTExecConfig{}), std::logic_error);
    fdmt.finalize();
    REQUIRE_NOTHROW(fdmt.set_exec_config(FDMTExecConfig{}));
}

TEST_CASE("FDMTCPU tiled schedule is bit-exact with coord schedule",
          "[fdmt_cpu][cpu]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    const std::vector<Case> cases = {
        {32, 256, 48, 0},
        {37, 100, 40, 0},   // odd channels -> copy coordinates
        {64, 256, 32, -32}, // symmetric DM range
        {32, 128, -4, -40}, // purely negative
    };
    const std::vector<std::string> modes = {"full", "roll", "valid"};

    for (const auto& c : cases) {
        for (const auto& mode : modes) {
            for (const bool smearing : {true, false}) {
                DYNAMIC_SECTION("nchans=" << c.nchans << " nsamps=" << c.nsamps
                                          << " dt=[" << c.dt_min << ","
                                          << c.dt_max << "] mode=" << mode
                                          << " smearing=" << smearing) {
                    const auto wf =
                        random_waterfall(c.nchans * c.nsamps, 11 + c.nchans);
                    FDMTCPU ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                c.dt_max, c.dt_min, 1, smearing, mode);
                    const auto expected = run(ref, wf);
                    for (const auto& cfg : tiled_configs()) {
                        FDMTCPU fdmt(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                     c.dt_max, c.dt_min, 1, smearing, mode);
                        fdmt.set_exec_config(cfg);
                        require_beams_exact(run(fdmt, wf), expected, fdmt);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTCPU tiled schedule with explicit dt_grid and multiple beams",
          "[fdmt_cpu][cpu]") {
    const std::vector<IndexType> dt_grid = {0, 1, 3, 7, 12, 20, 31, 45};
    const SizeType nchans                = 64;
    const SizeType nsamps                = 200;
    const SizeType nbeams                = 3;
    for (const std::string mode : {"full", "valid"}) {
        DYNAMIC_SECTION("mode=" << mode) {
            const auto wf = random_waterfall(nbeams * nchans * nsamps, 5);
            FDMTCPU ref(kFMin, kFMax, nchans, nsamps, kTsamp, dt_grid, true,
                        mode, false, 1, nbeams);
            const auto expected = run(ref, wf);
            for (const auto& cfg : tiled_configs()) {
                FDMTCPU fdmt(kFMin, kFMax, nchans, nsamps, kTsamp, dt_grid,
                             true, mode, false, 1, nbeams);
                fdmt.set_exec_config(cfg);
                require_beams_exact(run(fdmt, wf), expected, fdmt);
            }
        }
    }
}

TEST_CASE("FDMTCPU tiled schedule valid-mode streaming", "[fdmt_cpu][cpu]") {
    // Includes blocks smaller than the delay (history spans several blocks)
    // and flipping the schedule between blocks of one stream.
    struct Case {
        IndexType dt_max;
        IndexType dt_min;
        SizeType block;
        SizeType nblocks;
    };
    const std::vector<Case> cases = {
        {48, 0, 64, 5}, {48, 0, 16, 12}, {32, -32, 17, 10}};
    const SizeType nchans = 32;
    for (const auto& c : cases) {
        for (const bool smearing : {true, false}) {
            DYNAMIC_SECTION("dt=[" << c.dt_min << "," << c.dt_max << "] block="
                                   << c.block << " smearing=" << smearing) {
                const SizeType total = c.block * c.nblocks;
                const auto wf        = random_waterfall(nchans * total, 3);
                FDMTCPU ref(kFMin, kFMax, nchans, c.block, kTsamp, c.dt_max,
                            c.dt_min, 1, smearing, "valid");
                FDMTCPU tiled_fdmt(kFMin, kFMax, nchans, c.block, kTsamp,
                                   c.dt_max, c.dt_min, 1, smearing, "valid");
                FDMTCPU flip(kFMin, kFMax, nchans, c.block, kTsamp, c.dt_max,
                             c.dt_min, 1, smearing, "valid");
                tiled_fdmt.set_exec_config(tiled(7, 2));
                for (SizeType b = 0; b < c.nblocks; ++b) {
                    const auto block =
                        column_block(wf, nchans, total, b * c.block, c.block);
                    const auto expected = run(ref, block);
                    require_beams_exact(run(tiled_fdmt, block), expected,
                                        tiled_fdmt);
                    flip.set_exec_config(b % 2 == 0 ? tiled(0, 0)
                                                    : FDMTExecConfig{});
                    require_beams_exact(run(flip, block), expected, flip);
                }
                // History state is schedule-independent.
                std::vector<float> h_ref(ref.history_state_size());
                std::vector<float> h_tiled(tiled_fdmt.history_state_size());
                ref.save_history(h_ref);
                tiled_fdmt.save_history(h_tiled);
                REQUIRE_THAT(h_tiled, Catch::Matchers::Equals(h_ref));
            }
        }
    }
}

TEST_CASE("FDMTCPU streaming stores are bit-exact", "[fdmt_cpu][cpu]") {
    // kAlways forces the non-temporal path on every float level (a no-op on
    // builds without streaming stores, where this still checks the switch).
    const std::vector<std::string> modes = {"full", "roll", "valid"};
    for (const auto& mode : modes) {
        for (const bool smearing : {true, false}) {
            DYNAMIC_SECTION("mode=" << mode << " smearing=" << smearing) {
                const SizeType nchans = 37;
                const SizeType block  = 300;
                FDMTCPU ref(kFMin, kFMax, nchans, block, kTsamp, 64, -20, 1,
                            smearing, mode, false, 1, 2);
                FDMTCPU fdmt(kFMin, kFMax, nchans, block, kTsamp, 64, -20, 1,
                             smearing, mode, false, 1, 2);
                for (int b = 0; b < 3; ++b) {
                    fdmt.set_exec_config(
                        {.streaming_stores =
                             (b == 1) ? FDMTStreamingStores::kAuto
                                      : FDMTStreamingStores::kAlways});
                    const auto wf = random_waterfall(
                        2 * nchans * block, static_cast<unsigned>(40 + b));
                    require_beams_exact(run(fdmt, wf), run(ref, wf), fdmt);
                }
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
                                2);
                    const auto expected = run(ref, wf);
                    for (const SizeType fuse : {1, 2, 3, 9}) {
                        FDMTCPU fdmt(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                     c.dt_max, c.dt_min, 1, smearing, mode,
                                     false, 1, 2);
                        fdmt.set_exec_config({.fuse_levels = fuse});
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
        FDMTCPU ref(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid");
        FDMTCPU fdmt(kFMin, kFMax, 64, 200, kTsamp, dt_grid, true, "valid");
        fdmt.set_exec_config({.fuse_levels = 3});
        require_beams_exact(run(fdmt, wf), run(ref, wf), fdmt);
    }
    SECTION("valid-mode streaming, fusion toggled between blocks") {
        const SizeType nchans = 32;
        for (const SizeType block : {16, 64}) {
            FDMTCPU ref(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                        "valid");
            FDMTCPU fdmt(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1, true,
                         "valid");
            for (SizeType b = 0; b < 10; ++b) {
                fdmt.set_exec_config({.fuse_levels = (b % 3 == 2) ? 0U : 2U});
                const auto wf = random_waterfall(nchans * block,
                                                 static_cast<unsigned>(60 + b));
                require_beams_exact(run(fdmt, wf), run(ref, wf), fdmt);
            }
            std::vector<float> h_ref(ref.history_state_size());
            std::vector<float> h_fused(fdmt.history_state_size());
            ref.save_history(h_ref);
            fdmt.save_history(h_fused);
            REQUIRE_THAT(h_fused, Catch::Matchers::Equals(h_ref));
        }
    }
    SECTION("packed input with int tree and tiled schedule") {
        for (const SizeType nbits : {1, 4, 8}) {
            for (const std::string mode : {"full", "valid"}) {
                const auto wf = random_packed(256, 128, nbits, 30 + nbits);
                FDMTCPU ref(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true,
                            mode);
                FDMTCPU fdmt(kFMin, kFMax, 256, 128, kTsamp, 64, 0, 1, true,
                             mode);
                fdmt.set_exec_config({.schedule    = FDMTSchedule::kTiled,
                                      .tile_nsamps = 40,
                                      .int_tree    = true,
                                      .fuse_levels = 3});
                require_beams_exact(run_packed(fdmt, wf.packed, nbits),
                                    run(ref, wf.values), fdmt);
            }
        }
    }
    SECTION("stepper stays unfused and inspectable") {
        FDMTCPU fdmt(kFMin, kFMax, 32, 128, kTsamp, 16);
        fdmt.set_exec_config({.fuse_levels = 3});
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
                                    c.dt_max, c.dt_min, 1, smearing, mode);
                        const auto expected = run(ref, wf.values);
                        for (const bool int_tree : {false, true}) {
                            for (const auto schedule :
                                 {FDMTSchedule::kCoord, FDMTSchedule::kTiled}) {
                                FDMTCPU fdmt(kFMin, kFMax, c.nchans, c.nsamps,
                                             kTsamp, c.dt_max, c.dt_min, 1,
                                             smearing, mode);
                                fdmt.set_exec_config({.schedule    = schedule,
                                                      .tile_nsamps = 50,
                                                      .tile_ndt    = 4,
                                                      .int_tree    = int_tree});
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
                const auto row_bytes  = utils::packed_row_bytes(block, nbits);
                FDMTCPU ref(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                            c.dt_min, 1, true, "valid", false, 1, nbeams);
                FDMTCPU fdmt(kFMin, kFMax, nchans, block, kTsamp, c.dt_max,
                             c.dt_min, 1, true, "valid", false, 1, nbeams);
                fdmt.set_exec_config({.schedule    = FDMTSchedule::kTiled,
                                      .tile_nsamps = 16,
                                      .int_tree    = true});
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
    fdmt.reset(std::span<const uint8_t>(wf.packed), 1, dmt);
    REQUIRE_NOTHROW(fdmt.view_level_data());
    fdmt.finalize();

    // With int_tree the 1-bit, 64-channel, no-smearing levels start as
    // uint8 (bound 1) and cannot be viewed; the root is always float.
    fdmt.set_exec_config({.int_tree = true});
    fdmt.reset(std::span<const uint8_t>(wf.packed), 1, dmt);
    REQUIRE_THROWS_AS(fdmt.view_level_data(), std::logic_error);
    REQUIRE_THROWS_AS(fdmt.view_subband(0), std::logic_error);
    fdmt.advance_until_remaining(0);
    REQUIRE_NOTHROW(fdmt.view_level_data());
    fdmt.finalize();
}

} // namespace dmt
