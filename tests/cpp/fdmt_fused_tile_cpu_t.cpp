#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/fdmt.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_fused_tile.hpp"
#include "dmt/fdmt_history.hpp"
#include "dmt/modes.hpp"

// The CUDA backend's fused-level tile (lib/dmt/fdmt_fused_tile.hpp), run
// sequentially on the host -- one "thread", barriers as no-ops -- over every
// (group, tile) of a block, against FDMT's original level-by-level path.
// This checks the tile geometry (halos, boundaries, history ownership)
// without a GPU; the GPU tests check the kernel launch itself. Integer-valued
// input keeps every sum exact whatever the host compiler's float flags, so
// all comparisons are bit-exact.

namespace dmt {

using algorithms::FDMT;
namespace detail = algorithms::detail;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

struct HostBlock {
    template <typename Row>
    void for_each_row(int nrows, int width, Row&& row) const {
        for (int r = 0; r < nrows; ++r) {
            auto cell = row(r);
            for (int j = 0; j < width; ++j) {
                cell(j);
            }
        }
    }
    void sync() const {}
};

struct Waterfall {
    std::vector<float> values;   // (nchans, nsamps)
    std::vector<uint8_t> packed; // 2-bit rows
};

Waterfall random_ints(SizeType nchans, SizeType nsamps, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 3);
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, 2);
    Waterfall wf{.values = std::vector<float>(nchans * nsamps),
                 .packed = std::vector<uint8_t>(nchans * row_bytes, 0)};
    for (SizeType c = 0; c < nchans; ++c) {
        for (SizeType s = 0; s < nsamps; ++s) {
            const auto v                = dis(gen);
            wf.values[(c * nsamps) + s] = static_cast<float>(v);
            bit_pack_utils::write_packed_sample<2>(
                wf.packed.data() + (c * row_bytes), s, v);
        }
    }
    return wf;
}

// Box means of `fplan` as the tile reads them (fdmt_box_weights.hpp): dyadic
// float weights, the packed rounding recipes on the scale 2^-F, and 2^-F.
struct BoxTables {
    std::vector<float> box_w;
    std::vector<detail::FDMTBoxRound> box_round;
    float int_unit{1.0F};

    [[nodiscard]] detail::FDMTBoxMeans means(bool smear) const {
        return {.w_f32     = box_w.data(),
                .round     = box_round.data(),
                .unit      = smear ? int_unit : 1.0F,
                .round_u64 = 0};
    }
};

BoxTables box_tables(const plans::FDMTPlan& fplan) {
    BoxTables t;
    const float w_unit =
        std::ldexp(1.0F, -static_cast<int>(fplan.get_box_fraction_bits()));
    const auto scale_f = fplan.get_box_mean_bits();
    SizeType len       = 1;
    for (const auto w : fplan.get_box_weights()) {
        t.box_w.push_back(static_cast<float>(w) * w_unit);
        t.box_round.push_back(detail::box_round(len++, scale_f));
    }
    t.int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
    return t;
}

// Runs every (group, tile) of the fused levels 0..fuse on the host and
// returns level `fuse` as float, in canonical units (integer level-F
// storage is rescaled by 2^-b). `thist_out` receives the fused levels'
// tree-history slots (valid mode).
template <FDMTMode Mode, bool Smear, typename TOut>
std::vector<float> run_tiles(const plans::FDMTPlanContainer& pc,
                             const detail::FDMTFusedTilePlan& plan,
                             const BoxTables& box,
                             int tile,
                             const Waterfall& wf,
                             bool packed,
                             const float* hist0,
                             const float* thist_in,
                             float* thist_out) {
    const auto nsamps    = static_cast<int>(pc.state_shape[0].nsamps);
    const auto hist_len  = static_cast<int>(pc.state_shape[0].dt_max);
    const float int_unit = Smear ? box.int_unit : 1.0F;
    const auto args      = detail::make_fused_tile_args(
        plan, tile, hist_len, plan.group_info.data(), plan.coords.data(),
        plan.dt_grid0.data(), plan.ndt0.data(), plan.coord_offset0.data(),
        box.means(Smear));
    const auto [cap_a, cap_b] = plan.smem_floats(tile);
    constexpr SizeType kGuard = 64;
    constexpr float kSentinel = -12345.0F;
    std::vector<float> smem(cap_a + cap_b + kGuard);
    std::vector<TOut> out(pc.state_shape[plan.fuse].nelements, TOut{0});
    const int ntiles = (plan.ntiles_nsamps + tile - 1) / tile;
    const detail::FDMTInputF32 in_f32{.data   = wf.values.data(),
                                      .nsamps = nsamps};
    const detail::FDMTInputPacked in_pk{
        .data = wf.packed.data(),
        .row_bytes =
            static_cast<int>(bit_pack_utils::packed_row_bytes(nsamps, 2)),
        .nbits = 2};
    for (int g = 0; g < plan.ngroups; ++g) {
        for (int t = 0; t < ntiles; ++t) {
            std::ranges::fill(smem, kSentinel);
            if (packed) {
                detail::fdmt_fused_tile<Mode, Smear>(
                    HostBlock{}, in_pk, 0, out.data(), args, t, g, smem.data(),
                    hist0, thist_in, thist_out);
            } else {
                detail::fdmt_fused_tile<Mode, Smear>(
                    HostBlock{}, in_f32, 0, out.data(), args, t, g, smem.data(),
                    hist0, thist_in, thist_out);
            }
            // Nothing written past the advertised scratch size.
            for (SizeType i = cap_a + cap_b; i < smem.size(); ++i) {
                REQUIRE(smem[i] == kSentinel);
            }
        }
    }
    std::vector<float> level(out.size());
    for (SizeType i = 0; i < out.size(); ++i) {
        level[i] = std::is_integral_v<TOut>
                       ? static_cast<float>(out[i]) * int_unit
                       : static_cast<float>(out[i]);
    }
    return level;
}

template <typename TOut>
std::vector<float> run_tiles_dispatch(FDMTMode mode,
                                      bool smear,
                                      const plans::FDMTPlanContainer& pc,
                                      const detail::FDMTFusedTilePlan& plan,
                                      const BoxTables& box,
                                      int tile,
                                      const Waterfall& wf,
                                      bool packed,
                                      const float* hist0,
                                      const float* thist_in,
                                      float* thist_out) {
    const auto go = [&]<FDMTMode Mode, bool S>() {
        return run_tiles<Mode, S, TOut>(pc, plan, box, tile, wf, packed, hist0,
                                        thist_in, thist_out);
    };
    if (mode == FDMTMode::kFull) {
        return smear ? go.template operator()<FDMTMode::kFull, true>()
                     : go.template operator()<FDMTMode::kFull, false>();
    }
    if (mode == FDMTMode::kRoll) {
        return smear ? go.template operator()<FDMTMode::kRoll, true>()
                     : go.template operator()<FDMTMode::kRoll, false>();
    }
    return smear ? go.template operator()<FDMTMode::kValid, true>()
                 : go.template operator()<FDMTMode::kValid, false>();
}

// Level `fuse` of one block through FDMT's original path.
std::vector<float>
reference_level(FDMT& ref, const std::vector<float>& block, SizeType fuse) {
    std::vector<float> dmt(ref.get_plan().get_buffer_size());
    ref.reset(block, dmt);
    ref.advance(fuse);
    const auto view = ref.view_level_data();
    std::vector<float> level(view.begin(), view.end());
    ref.finalize();
    return level;
}

// The same for packed 2-bit input (the rounded box means of packed input).
std::vector<float> reference_level_packed(FDMT& ref,
                                          const std::vector<uint8_t>& block,
                                          SizeType fuse) {
    std::vector<float> dmt(ref.get_plan().get_buffer_size());
    ref.reset(std::span<const uint8_t>(block), 2, dmt);
    ref.advance(fuse);
    const auto view = ref.view_level_data();
    std::vector<float> level(view.begin(), view.end());
    ref.finalize();
    return level;
}

// Tree-history slots [hist_offset, hist_offset + delay) of levels 1..fuse.
std::vector<SizeType> fused_history_slots(const plans::FDMTPlanContainer& pc,
                                          SizeType fuse) {
    std::vector<SizeType> slots;
    for (SizeType l = 1; l <= fuse; ++l) {
        for (const auto& c : pc.coordinates_sum[l]) {
            for (SizeType k = 0; k < c.delay; ++k) {
                slots.push_back(c.hist_offset + k);
            }
        }
    }
    return slots;
}

} // namespace

TEST_CASE("CUDA fused tile matches the original path (host emulation)",
          "[fdmt_fused_tile][cpu]") {
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
        {4, 64, 8, 0},      // fewer levels than the requested depth
    };
    for (const auto& c : cases) {
        for (const std::string mode : {"full", "roll", "valid"}) {
            for (const bool smear : {true, false}) {
                // Original level-by-level float path (stepper inspection).
                FDMT ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp, c.dt_max,
                         c.dt_min, 1, smear, mode, Exec::cpu(1), 1, 0, false);
                const auto& pc    = ref.get_plan().get_container();
                const auto niters = ref.get_plan().get_niters();
                const auto wf = random_ints(c.nchans, c.nsamps,
                                            static_cast<unsigned>(c.nchans));
                for (SizeType fuse = 1; fuse <= std::min<SizeType>(niters, 4);
                     ++fuse) {
                    ref.reset_history(); // first block of a stream
                    const auto expected = reference_level(ref, wf.values, fuse);
                    ref.reset_history();
                    const auto expected_pk =
                        reference_level_packed(ref, wf.packed, fuse);
                    const auto plan =
                        detail::build_fused_tile_plan(pc, niters, fuse);
                    for (const int tile : {32, 96, 1024}) {
                        DYNAMIC_SECTION(
                            "nchans=" << c.nchans << " dt=[" << c.dt_min << ","
                                      << c.dt_max << "] mode=" << mode
                                      << " smear=" << smear << " fuse=" << fuse
                                      << " tile=" << tile) {
                            // Fresh (zero) history: first block of a stream.
                            std::vector<float> hist0(
                                ref.get_plan().get_history_size(), 0.0F);
                            std::vector<float> thist_in(
                                ref.get_plan().get_tree_history_size(), 0.0F);
                            std::vector<float> thist_out(thist_in.size(), 0.0F);
                            const auto box = box_tables(ref.get_plan());
                            REQUIRE_THAT(run_tiles_dispatch<float>(
                                             parse_fdmt_mode(mode), smear, pc,
                                             plan, box, tile, wf, false,
                                             hist0.data(), thist_in.data(),
                                             thist_out.data()),
                                         Catch::Matchers::Equals(expected));
                            // Packed input (rounded box means), integer
                            // and float level-F storage.
                            REQUIRE_THAT(run_tiles_dispatch<uint16_t>(
                                             parse_fdmt_mode(mode), smear, pc,
                                             plan, box, tile, wf, true,
                                             hist0.data(), thist_in.data(),
                                             thist_out.data()),
                                         Catch::Matchers::Equals(expected_pk));
                            REQUIRE_THAT(run_tiles_dispatch<float>(
                                             parse_fdmt_mode(mode), smear, pc,
                                             plan, box, tile, wf, true,
                                             hist0.data(), thist_in.data(),
                                             thist_out.data()),
                                         Catch::Matchers::Equals(expected_pk));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("CUDA fused tile valid-mode streaming (host emulation)",
          "[fdmt_fused_tile][cpu]") {
    // Blocks shorter than the delays (history spans several blocks) and
    // longer ones; the history the tile writes for its fused levels must be
    // exactly the original path's, block after block.
    const SizeType nchans = 32;
    for (const SizeType block : {4, 5, 16, 40, 64}) {
        for (const bool smear : {true, false}) {
            for (const SizeType fuse : {1, 2, 3}) {
                DYNAMIC_SECTION("block=" << block << " smear=" << smear
                                         << " fuse=" << fuse) {
                    FDMT ref(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1,
                             smear, "valid", Exec::cpu(1), 1, 0, false);
                    const auto& plan_h = ref.get_plan();
                    const auto& pc     = plan_h.get_container();
                    const auto plan    = detail::build_fused_tile_plan(
                        pc, plan_h.get_niters(), fuse);
                    const auto h0_size = plan_h.get_history_size();
                    const auto slots   = fused_history_slots(pc, fuse);
                    REQUIRE(!slots.empty());
                    std::vector<float> before(ref.history_state_size());
                    std::vector<float> after(ref.history_state_size());
                    for (SizeType b = 0; b < 6; ++b) {
                        ref.save_history(before);
                        const auto wf = random_ints(
                            nchans, block, static_cast<unsigned>(70 + b));
                        const auto expected =
                            reference_level(ref, wf.values, fuse);
                        ref.save_history(after);
                        // Past the header: the level-0, then the tree
                        // history.
                        const float* hist0 =
                            before.data() + detail::kFDMTHistHeader;
                        const float* tree_in = hist0 + h0_size;
                        std::vector<float> tree_after(
                            after.begin() +
                                static_cast<std::ptrdiff_t>(
                                    detail::kFDMTHistHeader + h0_size),
                            after.end());
                        // Finite sentinel (release builds use -ffast-math,
                        // which folds NaN checks away); never a sum of
                        // non-negative integers.
                        std::vector<float> tree_template = tree_after;
                        for (const auto s : slots) {
                            tree_template[s] = -1.0e30F;
                        }
                        // Narrow tiles put tile boundaries inside every
                        // history window (production tiles are multiples of
                        // 32, but the geometry holds for any width).
                        for (const int tile : {3, 7, 32}) {
                            std::vector<float> tree_out = tree_template;
                            const auto got = run_tiles_dispatch<float>(
                                FDMTMode::kValid, smear, pc, plan,
                                box_tables(plan_h), tile, wf, false, hist0,
                                tree_in, tree_out.data());
                            REQUIRE_THAT(got,
                                         Catch::Matchers::Equals(expected));
                            REQUIRE_THAT(tree_out,
                                         Catch::Matchers::Equals(tree_after));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("CUDA fused tile plan sizing", "[fdmt_fused_tile][cpu]") {
    FDMT fdmt(704.0F, 1216.0F, 1024, 2048, 0.00008192F, 512);
    const auto& pc    = fdmt.get_plan().get_container();
    const auto niters = fdmt.get_plan().get_niters();
    REQUIRE_THROWS_AS(detail::build_fused_tile_plan(pc, niters, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(detail::build_fused_tile_plan(pc, niters, niters + 1),
                      std::invalid_argument);
    SizeType prev_floats = 0;
    for (SizeType fuse = 1; fuse <= 4; ++fuse) {
        const auto plan = detail::build_fused_tile_plan(pc, niters, fuse);
        REQUIRE(plan.ngroups == static_cast<int>(1024 >> fuse));
        // Deeper fusion needs more shared memory for the same tile.
        const auto [a, b] = plan.smem_floats(256);
        REQUIRE(a + b > prev_floats);
        prev_floats    = a + b;
        const int tile = plan.max_tile_nsamps(48 * 1024, 1024);
        REQUIRE(tile % 32 == 0);
        if (tile > 0) {
            const auto [ta, tb] = plan.smem_floats(tile);
            REQUIRE((ta + tb) * sizeof(float) <= 48 * 1024);
        }
    }
}

} // namespace dmt

namespace dmt {

TEST_CASE("CUDA fused tile chunked streaming (host emulation)",
          "[fdmt_fused_tile][cpu][chunk]") {
    // The GPU engine's chunked execution with every level fused (fuse =
    // niters): tables from the plan re-laid out for chunk rows, each chunk's
    // input and root output strided by the block length, the GPU's
    // double-buffered level-0 and tree histories swapped per chunk. Must
    // equal the CPU engine's chunked stream bit for bit.
    const SizeType nchans = 16;
    const SizeType nsamps = 500;
    for (const SizeType chunk : {64, 128, 448}) {
        for (const bool smear : {true, false}) {
            DYNAMIC_SECTION("chunk=" << chunk << " smear=" << smear) {
                FDMT cpu(kFMin, kFMax, nchans, nsamps, kTsamp, 160, -20, 1,
                         smear, "valid", Exec::cpu(1), 1, 0, false, chunk);
                const auto& fplan = cpu.get_plan();
                const auto niters = fplan.get_niters();
                const auto c      = cpu.get_chunk_nsamps();
                const auto pc     = detail::fdmt_relayout_container(
                    fplan.get_container(), niters, c);
                const auto plan =
                    detail::build_fused_tile_plan(pc, niters, niters);
                const auto box      = box_tables(fplan);
                const int tile      = 32;
                const auto hist_len = static_cast<int>(
                    fplan.get_container().state_shape[0].dt_max);
                std::vector<float> h0_a(fplan.get_history_size(), 0.0F);
                std::vector<float> h0_b(h0_a.size(), 0.0F);
                std::vector<float> th_a(fplan.get_tree_history_size(), 0.0F);
                std::vector<float> th_b(th_a.size(), 0.0F);
                const auto [cap_a, cap_b] = plan.smem_floats(tile);
                std::vector<float> smem(cap_a + cap_b);
                for (unsigned blk = 0; blk < 3; ++blk) {
                    const auto wf = random_ints(nchans, nsamps, 90 + blk);
                    std::vector<float> want(fplan.get_dmt_size());
                    cpu.execute(wf.values, want);
                    std::vector<float> got(want.size(), -1.0F);
                    for (SizeType t0 = 0; t0 < nsamps; t0 += c) {
                        const auto r = std::min(c, nsamps - t0);
                        auto args    = detail::make_fused_tile_args(
                            plan, tile, hist_len, plan.group_info.data(),
                            plan.coords.data(), plan.dt_grid0.data(),
                            plan.ndt0.data(), plan.coord_offset0.data(),
                            box.means(smear));
                        args.active = static_cast<int>(r);
                        const detail::FDMTInputF32 in{
                            .data   = wf.values.data() + t0,
                            .nsamps = static_cast<int>(nsamps)};
                        const int ntiles =
                            static_cast<int>((r + tile - 1) / tile);
                        for (int g = 0; g < plan.ngroups; ++g) {
                            for (int t = 0; t < ntiles; ++t) {
                                const auto run = [&]<bool S>() {
                                    detail::fdmt_fused_tile<FDMTMode::kValid,
                                                            S>(
                                        HostBlock{}, in, 0, got.data() + t0,
                                        args, t, g, smem.data(), h0_a.data(),
                                        th_a.data(), th_b.data());
                                };
                                if (smear) {
                                    run.template operator()<true>();
                                } else {
                                    run.template operator()<false>();
                                }
                            }
                        }
                        // Level-0 history: the last hist_len raw samples
                        // (kernel_advance_history_window), then swap.
                        for (SizeType ch = 0; ch < nchans; ++ch) {
                            for (int k = 0; k < hist_len; ++k) {
                                const auto from =
                                    static_cast<std::ptrdiff_t>(r) - hist_len +
                                    k;
                                h0_b[(ch * hist_len) + k] =
                                    from >= 0
                                        ? wf.values[(ch * nsamps) + t0 + from]
                                        : h0_a[(ch * hist_len) + k + r];
                            }
                        }
                        std::swap(h0_a, h0_b);
                        std::swap(th_a, th_b);
                    }
                    REQUIRE_THAT(got, Catch::Matchers::Equals(want));
                }
            }
        }
    }
}

} // namespace dmt
