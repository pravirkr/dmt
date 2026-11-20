#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/algorithms/fdmt_fft_boxcar.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"

namespace {

using namespace dmt; // NOLINT(google-build-using-namespace)
using algorithms::FDMTBoxcar;
using algorithms::FDMTFFTBoxcar;

struct StreamRun {
    std::vector<RawCandidate> candidates;
    CandidateCounts counts{};
};

auto canonical_key(const RawCandidate& c) {
    return std::tuple(c.beam_idx, c.subband_level, c.staggered, c.subband_idx,
                      c.dm_idx, c.filter_idx, c.begin_sample, c.end_sample,
                      c.peak_sample);
}

void append_view(StreamRun& run, const CandidateResultView& view) {
    REQUIRE(view.candidates.size() == view.emitted_count);
    run.counts.emitted_count += view.emitted_count;
    run.counts.total_detected += view.total_detected;
    run.counts.dropped_global += view.dropped_global;
    run.counts.dropped_per_dm += view.dropped_per_dm;
    run.candidates.insert(run.candidates.end(), view.candidates.begin(),
                          view.candidates.end());
}

void require_bit_exact(StreamRun actual, StreamRun expected) {
    CHECK(actual.counts.total_detected == expected.counts.total_detected);
    CHECK(actual.counts.emitted_count == expected.counts.emitted_count);
    CHECK(actual.counts.dropped_global == expected.counts.dropped_global);
    CHECK(actual.counts.dropped_per_dm == expected.counts.dropped_per_dm);
    std::ranges::sort(actual.candidates, {}, canonical_key);
    std::ranges::sort(expected.candidates, {}, canonical_key);
    REQUIRE(actual.candidates.size() == expected.candidates.size());
    for (SizeType i = 0; i < actual.candidates.size(); ++i) {
        const auto& a = actual.candidates[i];
        const auto& e = expected.candidates[i];
        REQUIRE(canonical_key(a) == canonical_key(e));
        CHECK(a.width == e.width);
        CHECK(a.chan_begin == e.chan_begin);
        CHECK(a.chan_end == e.chan_end);
        CHECK(std::bit_cast<uint32_t>(a.dm) == std::bit_cast<uint32_t>(e.dm));
        CHECK(std::bit_cast<uint32_t>(a.snr) == std::bit_cast<uint32_t>(e.snr));
    }
}

float next_gaussian(std::mt19937& rng) noexcept {
    constexpr double kInv2Pow32 = 1.0 / 4294967296.0;
    const double u1 = (static_cast<double>(rng()) + 0.5) * kInv2Pow32;
    const double u2 = (static_cast<double>(rng()) + 0.5) * kInv2Pow32;
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) *
                              std::cos(2.0 * std::numbers::pi * u2));
}

std::vector<std::vector<float>> make_float_blocks(const plans::FDMTPlan& plan,
                                                  SizeType beams,
                                                  SizeType blocks,
                                                  uint32_t seed) {
    std::mt19937 rng(seed);
    const SizeType nchans = plan.get_nchans();
    const SizeType nsamps = plan.get_nsamps();
    std::vector<std::vector<float>> out(
        blocks, std::vector<float>(beams * nchans * nsamps));
    for (SizeType k = 0; k < blocks; ++k) {
        for (float& x : out[k]) {
            x = next_gaussian(rng);
        }
        for (SizeType b = 0; b < beams; ++b) {
            for (SizeType c = 0; c < nchans; ++c) {
                for (SizeType t = 0; t < nsamps; ++t) {
                    const bool near_edge =
                        (t < 4) || (t + 5 >= nsamps) ||
                        (nsamps > 40 && t >= nsamps / 2 && t < nsamps / 2 + 7);
                    if (near_edge && ((c + b + k) % 2 == 0)) {
                        out[k][((b * nchans + c) * nsamps) + t] +=
                            (b == 0 ? 2.4F : 1.9F);
                    }
                }
            }
        }
    }
    return out;
}

std::vector<std::vector<uint8_t>>
make_packed_blocks(const plans::FDMTPlan& plan,
                   SizeType beams,
                   SizeType blocks,
                   SizeType nbits,
                   uint32_t seed) {
    std::mt19937 rng(seed);
    const uint32_t max_code   = (1U << nbits) - 1U;
    const SizeType nchans     = plan.get_nchans();
    const SizeType nsamps     = plan.get_nsamps();
    const SizeType row_bytes  = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    const SizeType total_rows = beams * nchans;
    std::vector<std::vector<uint8_t>> packed(
        blocks, std::vector<uint8_t>(total_rows * row_bytes, 0U));
    for (SizeType k = 0; k < blocks; ++k) {
        for (SizeType r = 0; r < total_rows; ++r) {
            for (SizeType t = 0; t < nsamps; ++t) {
                uint32_t code = rng() & max_code;
                if ((t + 4 >= nsamps || t < 3) && ((r + k) % 3 == 0)) {
                    code = max_code;
                }
                if (nbits == 8) {
                    packed[k][(r * row_bytes) + t] = static_cast<uint8_t>(code);
                } else {
                    packed[k][(r * row_bytes) + (2 * t)] =
                        static_cast<uint8_t>(code & 0xFFU);
                    packed[k][(r * row_bytes) + (2 * t) + 1] =
                        static_cast<uint8_t>(code >> 8U);
                }
            }
        }
    }
    return packed;
}

BoxcarConfig
make_cfg(SizeType max_boxcar, float wtsp, float thresh, bool trap = false) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = max_boxcar;
    cfg.wtsp             = wtsp;
    cfg.detect_thresh    = thresh;
    cfg.peak_mode        = PeakFindingMode::kContiguousPeakAcrossWidths;
    cfg.use_trapezoid    = trap;
    cfg.max_candidates   = 500000;
    cfg.max_cands_per_dm = 0;
    return cfg;
}

StreamRun run_float(const plans::FDMTPlan& plan,
                    BoxcarConfig cfg,
                    BoxcarScreenMode screen,
                    const std::vector<std::vector<float>>& blocks,
                    SizeType beams,
                    const std::vector<uint8_t>& ctor_mask              = {},
                    const std::vector<std::vector<uint8_t>>& call_mask = {}) {
    cfg.screen_mode = screen;
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(), beams, 0, 0,
                      algorithms::kFDMTDefaultMaxMemory, ctor_mask);
    StreamRun run;
    for (SizeType k = 0; k < blocks.size(); ++k) {
        if (!call_mask.empty() && !call_mask[k].empty()) {
            append_view(run,
                        search.execute(std::span<const float>(blocks[k]),
                                       std::span<const uint8_t>(call_mask[k])));
        } else {
            append_view(run, search.execute(blocks[k]));
        }
    }
    append_view(run, search.drain());
    return run;
}

StreamRun run_packed(const plans::FDMTPlan& plan,
                     BoxcarConfig cfg,
                     BoxcarScreenMode screen,
                     const std::vector<std::vector<uint8_t>>& blocks,
                     SizeType beams) {
    cfg.screen_mode = screen;
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(), beams);
    StreamRun run;
    for (const auto& block : blocks) {
        append_view(run, search.execute(std::span<const uint8_t>(block)));
    }
    append_view(run, search.drain());
    return run;
}

} // namespace

TEST_CASE("FDMTBoxcar CPU screen modes match bit-for-bit",
          "[boxcar][kadane][cpu]") {
    const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 400, 0.001F,
                               std::vector<IndexType>{-32, -7, 0, 11, 63},
                               "valid", 2);

    SECTION("float streaming (rectangular, trapezoid, and wide boxcar)") {
        const auto blocks = make_float_blocks(plan, 2, 3, 101);
        for (const auto& [max_w, wtsp, trap] :
             std::array<std::tuple<SizeType, float, bool>, 3>{
                 {{64, 1.0F, false}, {64, 1.25F, true}, {600, 1.25F, false}}}) {
            const auto cfg = make_cfg(max_w, wtsp, 3.0F, trap);
            const auto ref =
                run_float(plan, cfg, BoxcarScreenMode::kPrefix, blocks, 2);
            REQUIRE(ref.counts.total_detected > 0);
            for (const auto screen :
                 {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
                require_bit_exact(run_float(plan, cfg, screen, blocks, 2), ref);
            }
        }
    }

    SECTION("packed 8-bit and 16-bit streaming") {
        for (const SizeType nbits : {SizeType{8}, SizeType{16}}) {
            const uint32_t max_code = (1U << nbits) - 1U;
            auto cfg                = make_cfg(64, 1.0F, 3.0F);
            cfg.input_nbits         = nbits;
            cfg.input_mean          = static_cast<float>(max_code) * 0.5F;
            cfg.noise_sigma         = std::sqrt(static_cast<float>(max_code) *
                                                (max_code + 2.0F) / 12.0F);
            const auto blocks =
                make_packed_blocks(plan, 1, 3, nbits, 200 + nbits);
            const auto ref =
                run_packed(plan, cfg, BoxcarScreenMode::kPrefix, blocks, 1);
            REQUIRE(ref.counts.total_detected > 0);
            for (const auto screen :
                 {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
                require_bit_exact(run_packed(plan, cfg, screen, blocks, 1),
                                  ref);
            }
        }
    }

    SECTION("constructor and per-call kill masks") {
        const auto blocks = make_float_blocks(plan, 1, 4, 303);
        const std::vector<uint8_t> ctor_mask{1, 1, 0, 1, 1, 1, 1, 0,
                                             1, 1, 1, 1, 0, 1, 1, 1};
        const std::vector<std::vector<uint8_t>> masks = {
            {},
            {1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1},
            std::vector<uint8_t>(16, 0),
            {1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},
        };
        const auto cfg = make_cfg(64, 1.25F, 2.8F, true);
        const auto ref = run_float(plan, cfg, BoxcarScreenMode::kPrefix, blocks,
                                   1, ctor_mask, masks);
        REQUIRE(ref.counts.total_detected > 0);
        for (const auto screen :
             {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
            require_bit_exact(
                run_float(plan, cfg, screen, blocks, 1, ctor_mask, masks), ref);
        }
    }

    SECTION("kKeepBrightest overflow policy") {
        const auto blocks    = make_float_blocks(plan, 1, 3, 404);
        auto cfg             = make_cfg(64, 1.0F, 2.5F);
        cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
        cfg.max_candidates   = 32;
        cfg.max_cands_per_dm = 4;
        const auto ref =
            run_float(plan, cfg, BoxcarScreenMode::kPrefix, blocks, 1);
        REQUIRE(ref.counts.dropped_global + ref.counts.dropped_per_dm > 0);
        for (const auto screen :
             {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
            require_bit_exact(run_float(plan, cfg, screen, blocks, 1), ref);
        }
    }

    SECTION("FDMTFFTBoxcar") {
        const plans::FDMTPlan fft_plan(700.0F, 1200.0F, 8, 512, 0.001F,
                                       std::vector<IndexType>{0, 5, 19, 43});
        const auto blocks  = make_float_blocks(fft_plan, 1, 3, 505);
        const auto run_fft = [&](BoxcarScreenMode screen) {
            auto cfg        = make_cfg(48, 1.25F, 2.8F);
            cfg.screen_mode = screen;
            FDMTFFTBoxcar search(fft_plan, cfg, true, Exec::cpu(), 1, true);
            StreamRun run;
            for (const auto& block : blocks) {
                append_view(run, search.execute(block));
            }
            append_view(run, search.flush());
            return run;
        };
        const auto ref = run_fft(BoxcarScreenMode::kPrefix);
        REQUIRE(ref.counts.total_detected > 0);
        for (const auto screen :
             {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone}) {
            require_bit_exact(run_fft(screen), ref);
        }
    }
}
