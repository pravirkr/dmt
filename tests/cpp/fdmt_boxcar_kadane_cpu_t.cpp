#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <omp.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/algorithms/fdmt_fft_boxcar.hpp"
#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_drain.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_int_tree.hpp"

#include "fdmt_boxcar_overflow_reference.hpp"
#include "skew_subband_search.hpp"

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

void sort_canonical(std::vector<RawCandidate>& v) {
    std::ranges::sort(v, {}, canonical_key);
}

void append_checked(StreamRun& run,
                    const CandidateResultView& view,
                    uint64_t& frontier,
                    uint64_t next_frontier,
                    bool allow_overflow = false) {
    if (!allow_overflow) {
        REQUIRE_FALSE(view.overflowed());
    }
    for (const auto& c : view.candidates) {
        REQUIRE(c.begin_sample >= frontier);
    }
    REQUIRE(view.candidates.size() == view.emitted_count);
    run.counts.emitted_count += view.emitted_count;
    run.counts.total_detected += view.total_detected;
    run.counts.dropped_global += view.dropped_global;
    run.counts.dropped_per_dm += view.dropped_per_dm;
    run.candidates.insert(run.candidates.end(), view.candidates.begin(),
                          view.candidates.end());
    REQUIRE(next_frontier >= frontier);
    frontier = next_frontier;
}

void require_bit_exact_candidates(std::vector<RawCandidate> actual,
                                  std::vector<RawCandidate> expected) {
    sort_canonical(actual);
    sort_canonical(expected);
    REQUIRE(actual.size() == expected.size());
    for (SizeType i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        if (canonical_key(a) == canonical_key(e) && a.width == e.width &&
            a.chan_begin == e.chan_begin && a.chan_end == e.chan_end &&
            std::bit_cast<uint32_t>(a.dm) == std::bit_cast<uint32_t>(e.dm) &&
            std::bit_cast<uint32_t>(a.f_low_mhz) ==
                std::bit_cast<uint32_t>(e.f_low_mhz) &&
            std::bit_cast<uint32_t>(a.f_high_mhz) ==
                std::bit_cast<uint32_t>(e.f_high_mhz) &&
            std::bit_cast<uint32_t>(a.snr) == std::bit_cast<uint32_t>(e.snr)) {
            continue;
        }
        INFO(fmt::format(
            "idx {}: actual=(beam={} lvl={} stag={} band={} dm={} filt={} "
            "[{}..{}] pk={} snr={:.8g}) vs expected=(beam={} lvl={} stag={} "
            "band={} dm={} filt={} [{}..{}] pk={} snr={:.8g})",
            i, a.beam_idx, a.subband_level, a.staggered, a.subband_idx,
            a.dm_idx, a.filter_idx, a.begin_sample, a.end_sample, a.peak_sample,
            a.snr, e.beam_idx, e.subband_level, e.staggered, e.subband_idx,
            e.dm_idx, e.filter_idx, e.begin_sample, e.end_sample, e.peak_sample,
            e.snr));
        REQUIRE(canonical_key(a) == canonical_key(e));
        CHECK(a.width == e.width);
        CHECK(a.chan_begin == e.chan_begin);
        CHECK(a.chan_end == e.chan_end);
        CHECK(std::bit_cast<uint32_t>(a.dm) == std::bit_cast<uint32_t>(e.dm));
        CHECK(std::bit_cast<uint32_t>(a.f_low_mhz) ==
              std::bit_cast<uint32_t>(e.f_low_mhz));
        CHECK(std::bit_cast<uint32_t>(a.f_high_mhz) ==
              std::bit_cast<uint32_t>(e.f_high_mhz));
        CHECK(std::bit_cast<uint32_t>(a.snr) == std::bit_cast<uint32_t>(e.snr));
    }
}

void check_counts_eq(const CandidateCounts& actual,
                     const CandidateCounts& expected) {
    CHECK(actual.total_detected == expected.total_detected);
    CHECK(actual.emitted_count == expected.emitted_count);
    CHECK(actual.dropped_global == expected.dropped_global);
    CHECK(actual.dropped_per_dm == expected.dropped_per_dm);
}

void require_bit_exact(StreamRun actual,
                       StreamRun expected,
                       bool check_emitted_list = true) {
    check_counts_eq(actual.counts, expected.counts);
    if (!check_emitted_list) {
        return;
    }
    require_bit_exact_candidates(std::move(actual.candidates),
                                 std::move(expected.candidates));
}

// Portable standard-normal draw independent of libc++ vs libstdc++ algorithms.
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
        // Boundary-straddling bursts across channels and beams.
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

BoxcarConfig
make_cfg(SizeType max_boxcar,
         float wtsp,
         float thresh,
         bool trap          = false,
         PeakFindingMode pm = PeakFindingMode::kContiguousPeakPerWidth) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = max_boxcar;
    cfg.wtsp             = wtsp;
    cfg.detect_thresh    = thresh;
    cfg.peak_mode        = pm;
    cfg.use_trapezoid    = trap;
    cfg.max_candidates   = 500000;
    cfg.max_cands_per_dm = 0;
    return cfg;
}

BoxcarConfig
make_packed_cfg(SizeType max_boxcar,
                float wtsp,
                float thresh,
                SizeType nbits,
                float mean,
                float sigma,
                bool trap          = false,
                PeakFindingMode pm = PeakFindingMode::kContiguousPeakPerWidth) {
    auto cfg        = make_cfg(max_boxcar, wtsp, thresh, trap, pm);
    cfg.input_nbits = nbits;
    cfg.input_mean  = mean;
    cfg.noise_sigma = sigma;
    return cfg;
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
                } else if (nbits == 16) {
                    packed[k][(r * row_bytes) + (2 * t)] =
                        static_cast<uint8_t>(code & 0xFFU);
                    packed[k][(r * row_bytes) + (2 * t) + 1] =
                        static_cast<uint8_t>(code >> 8U);
                } else {
                    for (SizeType bit = 0; bit < nbits; ++bit) {
                        const SizeType off = (t * nbits) + bit;
                        packed[k][(r * row_bytes) + (off / 8)] |=
                            static_cast<uint8_t>(((code >> bit) & 1U)
                                                 << (off % 8));
                    }
                }
            }
        }
    }
    return packed;
}

StreamRun run_fdmt_boxcar_float(
    const plans::FDMTPlan& plan,
    BoxcarConfig cfg,
    BoxcarScreenMode mode,
    const std::vector<std::vector<float>>& blocks,
    bool smear                                               = true,
    Exec exec                                                = Exec::cpu(1),
    SizeType beams                                           = 1,
    SizeType chunk                                           = 0,
    bool use_drain                                           = false,
    std::span<const uint8_t> ctor_kill                       = {},
    const std::vector<std::vector<uint8_t>>* per_block_masks = nullptr,
    bool check_replay                                        = false,
    bool allow_overflow                                      = false) {
    cfg.screen_mode = mode;
    FDMTBoxcar search(plan, cfg, smear, exec, beams, algorithms::kFDMTAutoFuse,
                      chunk, algorithms::kFDMTDefaultMaxMemory, ctor_kill);
    CHECK(search.get_memory_usage().total() > 0);
    auto execute_pass = [&] {
        StreamRun run;
        uint64_t fr = search.get_candidate_frontier();
        CHECK(fr == 0);
        for (SizeType k = 0; k < blocks.size(); ++k) {
            append_checked(
                run,
                (per_block_masks != nullptr)
                    ? search.execute(blocks[k], (*per_block_masks)[k])
                    : search.execute(blocks[k]),
                fr, search.get_candidate_frontier(), allow_overflow);
        }
        append_checked(run, use_drain ? search.drain() : search.flush(), fr,
                       search.get_candidate_frontier(), allow_overflow);
        if (use_drain) {
            CHECK(search.get_candidate_frontier() ==
                  blocks.size() * plan.get_nsamps());
        } else {
            CHECK(search.flush().candidates.empty());
        }
        return run;
    };
    StreamRun first = execute_pass();
    if (check_replay) {
        search.reset_history();
        CHECK(search.get_candidate_frontier() == 0);
        require_bit_exact(execute_pass(), first,
                          !allow_overflow ||
                              cfg.overflow_policy ==
                                  OverflowPolicy::kKeepBrightest ||
                              exec.nthreads <= 1);
    }
    return first;
}

StreamRun
run_fdmt_boxcar_packed(const plans::FDMTPlan& plan,
                       BoxcarConfig cfg,
                       BoxcarScreenMode mode,
                       const std::vector<std::vector<uint8_t>>& blocks,
                       bool smear                         = true,
                       Exec exec                          = Exec::cpu(1),
                       SizeType beams                     = 1,
                       SizeType chunk                     = 0,
                       bool use_drain                     = true,
                       std::span<const uint8_t> ctor_kill = {},
                       bool check_replay                  = false) {
    cfg.screen_mode = mode;
    FDMTBoxcar search(plan, cfg, smear, exec, beams, algorithms::kFDMTAutoFuse,
                      chunk, algorithms::kFDMTDefaultMaxMemory, ctor_kill);
    CHECK(search.get_memory_usage().total() > 0);
    auto pass = [&] {
        StreamRun run;
        uint64_t fr = search.get_candidate_frontier();
        CHECK(fr == 0);
        for (const auto& blk : blocks) {
            append_checked(run, search.execute(std::span<const uint8_t>(blk)),
                           fr, search.get_candidate_frontier());
        }
        append_checked(run, use_drain ? search.drain() : search.flush(), fr,
                       search.get_candidate_frontier());
        return run;
    };
    StreamRun first = pass();
    if (check_replay) {
        search.reset_history();
        require_bit_exact(pass(), first);
    }
    return first;
}

StreamRun run_fdmt_fft_boxcar_float(const plans::FDMTPlan& plan,
                                    BoxcarConfig cfg,
                                    BoxcarScreenMode mode,
                                    const std::vector<std::vector<float>>& blks,
                                    bool smear                         = true,
                                    int nthreads                       = 1,
                                    SizeType beams                     = 1,
                                    bool frac                          = false,
                                    std::span<const uint8_t> ctor_kill = {},
                                    bool check_replay = false) {
    cfg.screen_mode = mode;
    FDMTFFTBoxcar search(plan, cfg, smear, Exec::cpu(nthreads), beams, frac,
                         ctor_kill);
    CHECK(search.get_memory_usage().total() > 0);
    auto pass = [&] {
        StreamRun run;
        uint64_t fr = search.get_candidate_frontier();
        CHECK(fr == 0);
        for (const auto& blk : blks) {
            append_checked(run, search.execute(blk), fr,
                           search.get_candidate_frontier());
        }
        append_checked(run, search.drain(), fr,
                       search.get_candidate_frontier());
        CHECK(search.get_candidate_frontier() ==
              blks.size() * plan.get_nsamps());
        return run;
    };
    StreamRun first = pass();
    if (check_replay) {
        search.reset_history();
        CHECK(search.get_candidate_frontier() == 0);
        require_bit_exact(pass(), first);
    }
    return first;
}

void check_modes_exact(const plans::FDMTPlan& plan,
                       const BoxcarConfig& cfg,
                       const std::vector<std::vector<float>>& blocks,
                       bool smear                                     = true,
                       SizeType beams                                 = 1,
                       SizeType chunk                                 = 0,
                       bool use_drain                                 = true,
                       bool replay                                    = false,
                       bool overflow                                  = false,
                       std::span<const uint8_t> ctor_kill             = {},
                       const std::vector<std::vector<uint8_t>>* masks = nullptr,
                       int alt_threads                                = 0,
                       bool check_none                                = true) {
    const auto ref = run_fdmt_boxcar_float(
        plan, cfg, BoxcarScreenMode::kPrefix, blocks, smear, Exec::cpu(1),
        beams, chunk, use_drain, ctor_kill, masks, replay, overflow);
    if (overflow) {
        REQUIRE(ref.counts.overflowed());
    } else {
        REQUIRE_FALSE(ref.candidates.empty());
    }
    require_bit_exact(
        run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kKadane, blocks,
                              smear, Exec::cpu(1), beams, chunk, use_drain,
                              ctor_kill, masks, replay, overflow),
        ref);
    if (alt_threads > 1) {
        require_bit_exact(
            run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kKadane, blocks,
                                  smear, Exec::cpu(alt_threads), beams, chunk,
                                  use_drain, ctor_kill, masks, false, overflow),
            ref);
    }
    if (check_none) {
        require_bit_exact(
            run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kNone, blocks,
                                  smear, Exec::cpu(1), beams, chunk, use_drain,
                                  ctor_kill, masks, false, overflow),
            ref);
    }
}

std::vector<std::vector<uint8_t>> make_changing_masks(SizeType nchans,
                                                      SizeType blocks) {
    std::vector<std::vector<uint8_t>> masks;
    masks.reserve(blocks);
    for (SizeType b = 0; b < blocks; ++b) {
        if (b == 2) {
            masks.emplace_back();
            continue;
        }
        std::mt19937 gen(500U + static_cast<uint32_t>(b));
        std::vector<uint8_t> m(nchans, 1U);
        for (SizeType c = 0; c < nchans; ++c) {
            // Portable deterministic draw (avoids libstdc++/libc++ diffs).
            if ((gen() % 100U) < (20U + (5U * static_cast<uint32_t>(b)))) {
                m[c] = 0U;
            }
        }
        masks.push_back(std::move(m));
    }
    return masks;
}

} // namespace

TEST_CASE("FDMTBoxcar kKadane and kNone match kPrefix bit-for-bit across peak "
          "modes, trapezoids, wtsp, and subbands",
          "[fdmt_boxcar_kadane][boxcar][kadane][cpu]") {
    for (const SizeType subband_depth : {SizeType{0}, SizeType{1}}) {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 8, 257, 0.001F, 48, 0, 1,
                                   "valid", subband_depth);
        const auto blocks = make_float_blocks(plan, 2, 4, 101U);
        for (const bool smear : {true, false}) {
            for (const bool trap : {false, true}) {
                for (const float wtsp : {2.0F, 1.5F, 1.0F}) {
                    if (wtsp == 1.0F && subband_depth == 1 && !smear) {
                        continue; // Keep combinatorial runtime snappy.
                    }
                    for (const auto peak_mode :
                         {PeakFindingMode::kContiguousPeakPerWidth,
                          PeakFindingMode::kContiguousPeakAcrossWidths,
                          PeakFindingMode::kAllAboveThreshold}) {
                        if (peak_mode == PeakFindingMode::kAllAboveThreshold &&
                            wtsp == 1.0F) {
                            continue;
                        }
                        DYNAMIC_SECTION("depth="
                                        << subband_depth << " smear=" << smear
                                        << " trap=" << trap << " wtsp=" << wtsp
                                        << " mode="
                                        << static_cast<int>(peak_mode)) {
                            const auto cfg =
                                make_cfg(36, wtsp, 2.8F, trap, peak_mode);
                            for (const SizeType chunk :
                                 {SizeType{64}, plan.get_nsamps()}) {
                                check_modes_exact(plan, cfg, blocks, smear, 2,
                                                  chunk, subband_depth == 1,
                                                  chunk == 64 && wtsp == 2.0F,
                                                  false, {}, nullptr, 0,
                                                  chunk == plan.get_nsamps() &&
                                                      wtsp != 1.0F);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar kKadane handles wide K=3 buckets, tiny streaming chunks "
          "with trapezoids, and candidate overflow",
          "[fdmt_boxcar_kadane][boxcar][kadane][cpu]") {
    SECTION("Wide max_boxcar >= 513 (K=3 gate buckets and edge lag > 256)") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 4, 257, 0.001F, 36);
        auto blocks = make_float_blocks(plan, 1, 6, 202U);
        // Add a sustained elevated plateau across blocks 2..3 to trigger wide
        // filters above w = 513.
        for (SizeType k = 2; k <= 3; ++k) {
            for (float& x : blocks[k]) {
                x += 0.45F;
            }
        }
        for (const bool trap : {false, true}) {
            for (const auto peak_mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths}) {
                check_modes_exact(plan,
                                  make_cfg(600, 1.5F, 2.9F, trap, peak_mode),
                                  blocks, true, 1, 257, true, true);
            }
        }
    }

    SECTION(
        "Tiny nsamps=13 streaming blocks with wide trapezoids (w=1,2..96)") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 4, 13, 0.001F, 24, 0, 1,
                                   "valid", 1);
        const auto blocks = make_float_blocks(plan, 2, 24, 303U);
        for (const bool trap : {false, true}) {
            check_modes_exact(plan, make_cfg(96, 1.5F, 2.6F, trap), blocks,
                              true, 2, 0, true, true);
        }
    }

    SECTION("Overflow accounting matches exactly") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 4, 64, 0.001F, 20);
        const auto blocks    = make_float_blocks(plan, 1, 2, 404U);
        auto cfg             = make_cfg(16, 1.5F, 1.8F, false,
                                        PeakFindingMode::kAllAboveThreshold);
        cfg.max_candidates   = 11;
        cfg.max_cands_per_dm = 3;
        check_modes_exact(plan, cfg, blocks, true, 1, 0, false, true, true);
    }
}

TEST_CASE("FDMTBoxcar kKadane matches kPrefix bit-for-bit on packed "
          "1/2/4/8/16-bit inputs",
          "[fdmt_boxcar_kadane][boxcar][kadane][packed][cpu]") {
    const plans::FDMTPlan plan(700.0F, 1200.0F, 8, 139, 0.001F, 40, 0, 1,
                               "valid", 1);
    const std::vector<uint8_t> kill{1, 0, 1, 1, 1, 0, 1, 1};
    auto check_pk = [&](const plans::FDMTPlan& p, const BoxcarConfig& c,
                        const auto& blks, SizeType bms, SizeType chk,
                        std::span<const uint8_t> km, bool rep, bool chk_none) {
        const auto ref =
            run_fdmt_boxcar_packed(p, c, BoxcarScreenMode::kPrefix, blks, true,
                                   Exec::cpu(1), bms, chk, true, km);
        REQUIRE_FALSE(ref.candidates.empty());
        require_bit_exact(
            run_fdmt_boxcar_packed(p, c, BoxcarScreenMode::kKadane, blks, true,
                                   Exec::cpu(1), bms, chk, true, km, rep),
            ref);
        if (chk_none) {
            require_bit_exact(
                run_fdmt_boxcar_packed(p, c, BoxcarScreenMode::kNone, blks,
                                       true, Exec::cpu(1), bms, chk, true, km),
                ref);
        }
    };
    for (const SizeType nbits :
         {SizeType{1}, SizeType{2}, SizeType{4}, SizeType{8}, SizeType{16}}) {
        for (const bool trap : {false, true}) {
            DYNAMIC_SECTION("nbits=" << nbits << " trap=" << trap) {
                const auto maxf   = static_cast<float>((1U << nbits) - 1U);
                const auto blocks = make_packed_blocks(
                    plan, 2, 4, nbits, 500U + static_cast<uint32_t>(nbits));
                check_pk(plan,
                         make_packed_cfg(
                             36, 1.5F, 2.65F, nbits, maxf * 0.5F,
                             std::sqrt(maxf * (maxf + 2.0F) / 12.0F), trap),
                         blocks, 2, 64, kill, nbits == 8, true);
            }
        }
    }

    SECTION("16-bit wide boxcar (!has_int32 centered-float gate fallback)") {
        const plans::FDMTPlan wide_plan(700.0F, 1200.0F, 16, 256, 0.001F, 48);
        const float maxf = 65535.0F;
        check_pk(wide_plan,
                 make_packed_cfg(600, 2.0F, 2.7F, 16, maxf * 0.5F,
                                 std::sqrt(maxf * (maxf + 2.0F) / 12.0F), true,
                                 PeakFindingMode::kContiguousPeakAcrossWidths),
                 make_packed_blocks(wide_plan, 1, 4, 16, 616U), 1, 256, {},
                 true, false);
    }

    SECTION("16-bit smearing plan exceeding uint32 (!m_carry32 uint64 carry)") {
        const plans::FDMTPlan u64_plan(700.0F, 1200.0F, 8, 128, 0.001F,
                                       std::vector<IndexType>{0, 32, 16384},
                                       "valid", 1);
        REQUIRE_FALSE(
            algorithms::detail::int_search_fits_u32(u64_plan, true, 16));
        const float maxf = 65535.0F;
        const auto pcfg =
            make_packed_cfg(32, 1.5F, 2.65F, 16, maxf * 0.5F,
                            std::sqrt(maxf * (maxf + 2.0F) / 12.0F), true);
        const auto pblks = make_packed_blocks(u64_plan, 1, 4, 16, 632U);
        check_pk(u64_plan, pcfg, pblks, 1, 64, {}, true, true);
        const auto surr = run_fdmt_boxcar_packed(
            u64_plan, pcfg, BoxcarScreenMode::kKadaneSurrogate, pblks, true,
            Exec::cpu(1), 1, 64, true, {}, true);
        CHECK_FALSE(surr.candidates.empty());
    }
}

TEST_CASE("FDMTBoxcar kKadane matches kPrefix bit-for-bit on skewed-window "
          "subband search",
          "[fdmt_boxcar_kadane][boxcar][kadane][skew][cpu]") {
    using namespace algorithms::detail; // NOLINT(google-build-using-namespace)
    const plans::FDMTPlan plan(856.0F, 1712.0F, 64, 128, 0.0005F, 64, 0, 1,
                               "valid", 1);
    const auto fblocks = make_float_blocks(plan, 1, 4, 707U);
    const auto pblocks = make_packed_blocks(plan, 1, 4, 8, 708U);

    for (const bool packed : {false, true}) {
        for (const bool trap : {false, true}) {
            for (const bool use_drain : {false, true}) {
                DYNAMIC_SECTION("packed=" << packed << " trap=" << trap
                                          << " drain=" << use_drain) {
                    auto base_cfg = make_cfg(24, 1.5F, 2.65F, trap);
                    if (packed) {
                        base_cfg.input_nbits = 8;
                        base_cfg.input_mean  = 127.5F;
                        base_cfg.noise_sigma =
                            std::sqrt(255.0F * 257.0F / 12.0F);
                    }
                    auto run_skew_mode = [&](BoxcarScreenMode mode) {
                        BoxcarConfig cfg  = base_cfg;
                        cfg.screen_mode   = mode;
                        const auto planes = make_boxcar_plans(plan, cfg, true);
                        const SizeType root = plan.get_niters();
                        FDMTEngineConfig ecfg{
                            .use_box_smearing = true,
                            .int_tree         = false,
                            .exec             = Exec::cpu(1),
                            .chunk_nsamps     = 64,
                            .max_fuse_levels  = std::min(
                                root - 1, root - plan.get_subband_depth()),
                            .search_upper_levels = true,
                            .search_nbits = packed ? SizeType{8} : SizeType{32},
                            .skew_history_bytes  = 0,
                            .force_skewed_direct = true,
                        };
                        auto engine = make_fdmt_boxcar_cpu(
                            plan, cfg, planes, ecfg,
                            make_boxcar_driver(make_fdmt_cpu(plan, ecfg)));
                        REQUIRE(engine->search_path().find("skew") !=
                                std::string::npos);
                        const uint64_t latency = engine->search_latency();
                        REQUIRE(latency > 0);
                        auto frontier_now = [&](const DrainState& ds) {
                            return std::min(
                                searched_nsamps(*engine, ds, latency),
                                engine->open_begin_min());
                        };
                        DrainState ds;
                        StreamRun run;
                        uint64_t frontier = 0;
                        for (SizeType b = 0; b < 4; ++b) {
                            const auto view =
                                packed
                                    ? engine->execute(
                                          std::span<const uint8_t>(pblocks[b]))
                                    : engine->execute(
                                          std::span<const float>(fblocks[b]));
                            append_checked(run, view, frontier,
                                           frontier_now(ds));
                        }
                        const auto tail =
                            use_drain
                                ? drain_stream(*engine, plan, 1,
                                               packed ? 8 : 32, latency, ds)
                                : engine->flush();
                        append_checked(run, tail, frontier, frontier_now(ds));
                        return run;
                    };

                    const auto ref  = run_skew_mode(BoxcarScreenMode::kPrefix);
                    const auto kad  = run_skew_mode(BoxcarScreenMode::kKadane);
                    const auto none = run_skew_mode(BoxcarScreenMode::kNone);
                    REQUIRE_FALSE(ref.candidates.empty());
                    require_bit_exact(kad, ref);
                    require_bit_exact(none, ref);
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar kKadane matches kPrefix bit-for-bit under dynamic "
          "per-call kill masks and multi-threading",
          "[fdmt_boxcar_kadane][boxcar][kadane][kill][cpu]") {
    const plans::FDMTPlan plan(700.0F, 1200.0F, 8, 192, 0.001F, 36, 0, 1,
                               "valid", 1);
    const auto blocks = make_float_blocks(plan, 2, 5, 808U);
    const std::vector<uint8_t> ctor_kill{1, 1, 1, 1, 1, 1, 1, 0};
    const std::vector<std::vector<uint8_t>> masks{
        {},
        {1, 0, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 0, 0, 0, 0}, // Kills entire upper half-band
        {1, 0, 1, 1, 1, 1, 1, 1},
        {},
    };

    for (const bool trap : {false, true}) {
        for (const auto peak_mode :
             {PeakFindingMode::kContiguousPeakPerWidth,
              PeakFindingMode::kContiguousPeakAcrossWidths}) {
            DYNAMIC_SECTION(
                "trap=" << trap << " mode=" << static_cast<int>(peak_mode)) {
                check_modes_exact(
                    plan, make_cfg(24, 1.5F, 2.6F, trap, peak_mode), blocks,
                    true, 2, 64, true, true, false, ctor_kill, &masks, 2);
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar kKadane and kNone match kPrefix bit-for-bit",
          "[fdmt_boxcar_kadane][boxcar][kadane][fft][cpu]") {
    const plans::FDMTPlan plan(700.0F, 1200.0F, 8, 256, 0.001F, 40);
    const auto blocks = make_float_blocks(plan, 2, 4, 909U);
    const std::vector<uint8_t> kill{1, 0, 1, 1, 1, 1, 0, 1};
    for (const bool frac : {false, true}) {
        for (const bool trap : {false, true}) {
            for (const auto peak_mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                DYNAMIC_SECTION("frac=" << frac << " trap=" << trap << " mode="
                                        << static_cast<int>(peak_mode)) {
                    const auto cfg = make_cfg(32, 1.5F, 2.75F, trap, peak_mode);
                    auto run       = [&](BoxcarScreenMode m, int nt, bool rep) {
                        return run_fdmt_fft_boxcar_float(
                            plan, cfg, m, blocks, true, nt, 2, frac, kill, rep);
                    };
                    const auto ref = run(BoxcarScreenMode::kPrefix, 1, false);
                    REQUIRE_FALSE(ref.candidates.empty());
                    require_bit_exact(
                        run(BoxcarScreenMode::kKadane, 1, frac && !trap), ref);
                    require_bit_exact(run(BoxcarScreenMode::kKadane, 2, false),
                                      ref);
                    require_bit_exact(run(BoxcarScreenMode::kNone, 1, false),
                                      ref);
                }
            }
        }
    }
}

TEST_CASE("kKadaneSurrogate end-to-end recovery, S/N calibration, streaming "
          "frontier, and zero false alarms",
          "[fdmt_boxcar_kadane][boxcar][kadane][surrogate][cpu]") {
    SECTION("Zero false positives on pure unit-variance Gaussian noise") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 512, 0.001F, 32);
        std::mt19937 rng(12345U); // NOLINT(bugprone-random-generator-seed)
        const SizeType blk_len = plan.get_nchans() * plan.get_nsamps();
        std::vector<std::vector<float>> fblocks(8, std::vector<float>(blk_len));
        std::vector<std::vector<uint8_t>> u8blocks(
            8, std::vector<uint8_t>(blk_len));
        for (SizeType k = 0; k < 8; ++k) {
            for (SizeType i = 0; i < blk_len; ++i) {
                const float g  = next_gaussian(rng);
                fblocks[k][i]  = g;
                u8blocks[k][i] = static_cast<uint8_t>(
                    std::clamp(std::round(128.0F + 16.0F * g), 0.0F, 255.0F));
            }
        }
        const auto fcfg = make_cfg(64, 1.5F, 6.5F);
        const auto frun = run_fdmt_boxcar_float(
            plan, fcfg, BoxcarScreenMode::kKadaneSurrogate, fblocks, true,
            Exec::cpu(1), 1, 128, true);
        CHECK(frun.candidates.empty());
        CHECK(frun.counts.total_detected == 0);
        CHECK(run_fdmt_fft_boxcar_float(plan, fcfg,
                                        BoxcarScreenMode::kKadaneSurrogate,
                                        fblocks, true, 1, 1, true)
                  .candidates.empty());
        CHECK(run_fdmt_boxcar_packed(
                  plan, make_packed_cfg(64, 1.5F, 6.5F, 8, 128.0F, 16.0F),
                  BoxcarScreenMode::kKadaneSurrogate, u8blocks, true,
                  Exec::cpu(1), 1, 128, true)
                  .candidates.empty());
    }

    SECTION("Narrow and wide dispersed pulse recovery on FDMTBoxcar and "
            "FDMTFFTBoxcar (float32, uint8, uint16)") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 512, 0.001F, 40, 0, 1,
                                   "valid", 1);
        const SizeType target_dm = 22;
        const SizeType nchans    = plan.get_nchans();
        const SizeType nsamps    = plan.get_nsamps();
        const SizeType blk_len   = nchans * nsamps;
        const plans::FDMTPlan full_plan(700.0F, 1200.0F, nchans, 4 * nsamps,
                                        0.001F, 40);
        const plans::FDMTPlan root_plan(700.0F, 1200.0F, nchans, nsamps, 0.001F,
                                        40);
        auto best_root_snr = [](const StreamRun& run) {
            float best = 0.0F;
            for (const auto& c : run.candidates) {
                if (c.subband_level == 0 && c.dm_idx == target_dm &&
                    c.peak_sample >= 950 && c.peak_sample <= 1120) {
                    best = std::max(best, c.snr);
                }
            }
            return best;
        };
        for (const SizeType pulse_w :
             {SizeType{1}, SizeType{4}, SizeType{24}, SizeType{64}}) {
            std::mt19937 rng(4000U + static_cast<uint32_t>(pulse_w));
            std::vector<float> full_wf(4 * blk_len);
            for (float& x : full_wf) {
                x = next_gaussian(rng);
            }
            algorithms::add_frb_track(
                full_wf, full_plan, target_dm,
                15.0F / std::sqrt(static_cast<float>(nchans * pulse_w)), 980,
                pulse_w);
            std::vector<std::vector<float>> fblocks(
                4, std::vector<float>(blk_len));
            std::vector<std::vector<uint8_t>> u8blocks(
                4, std::vector<uint8_t>(blk_len));
            std::vector<std::vector<uint8_t>> u16blocks(
                4, std::vector<uint8_t>(blk_len * 2));
            for (SizeType k = 0; k < 4; ++k) {
                for (SizeType c = 0; c < nchans; ++c) {
                    for (SizeType t = 0; t < nsamps; ++t) {
                        const float x =
                            full_wf[(c * 4 * nsamps) + (k * nsamps) + t];
                        const SizeType dst = (c * nsamps) + t;
                        fblocks[k][dst]    = x;
                        u8blocks[k][dst]   = static_cast<uint8_t>(std::clamp(
                            std::round(128.0F + 12.0F * x), 0.0F, 255.0F));
                        const auto u16v    = static_cast<uint16_t>(
                            std::clamp(std::round(32768.0F + 1000.0F * x), 0.0F,
                                       65535.0F));
                        u16blocks[k][2 * dst] = static_cast<uint8_t>(u16v);
                        u16blocks[k][(2 * dst) + 1] =
                            static_cast<uint8_t>(u16v >> 8U);
                    }
                }
            }
            for (const bool trap : {false, true}) {
                DYNAMIC_SECTION("pulse_w=" << pulse_w << " trap=" << trap) {
                    for (const auto peak_mode :
                         {PeakFindingMode::kContiguousPeakPerWidth,
                          PeakFindingMode::kContiguousPeakAcrossWidths,
                          PeakFindingMode::kAllAboveThreshold}) {
                        if (trap &&
                            peak_mode == PeakFindingMode::kAllAboveThreshold) {
                            continue;
                        }
                        const auto cfg =
                            make_cfg(64, 1.5F, 6.0F, trap, peak_mode);
                        const auto exact = run_fdmt_boxcar_float(
                            plan, cfg, BoxcarScreenMode::kPrefix, fblocks,
                            false, Exec::cpu(1), 1, 64, true);
                        const auto surr_1t = run_fdmt_boxcar_float(
                            plan, cfg, BoxcarScreenMode::kKadaneSurrogate,
                            fblocks, false, Exec::cpu(1), 1, 64, true, {},
                            nullptr,
                            peak_mode ==
                                PeakFindingMode::kContiguousPeakPerWidth);
                        require_bit_exact(
                            run_fdmt_boxcar_float(
                                plan, cfg, BoxcarScreenMode::kKadaneSurrogate,
                                fblocks, false, Exec::cpu(2), 1, 64, true),
                            surr_1t);
                        const float exact_snr = best_root_snr(exact);
                        const float surr_snr  = best_root_snr(surr_1t);
                        REQUIRE(exact_snr > 11.0F);
                        CHECK(surr_snr >= 0.90F * exact_snr);
                        CHECK(surr_snr <= 1.15F * exact_snr);
                    }
                    auto check_packed_surr = [&](const BoxcarConfig& pcfg,
                                                 const auto& pblks, bool rep) {
                        const auto s = run_fdmt_boxcar_packed(
                            plan, pcfg, BoxcarScreenMode::kKadaneSurrogate,
                            pblks, false, Exec::cpu(1), 1, 64, true, {}, rep);
                        const auto p = run_fdmt_boxcar_packed(
                            plan, pcfg, BoxcarScreenMode::kPrefix, pblks, false,
                            Exec::cpu(1), 1, 64, true);
                        CHECK(best_root_snr(s) >= 0.90F * best_root_snr(p));
                    };
                    check_packed_surr(
                        make_packed_cfg(64, 1.5F, 6.0F, 8, 128.0F, 12.0F, trap),
                        u8blocks, true);
                    check_packed_surr(make_packed_cfg(64, 1.5F, 6.0F, 16,
                                                      32768.0F, 1000.0F, trap),
                                      u16blocks, false);
                    const auto fft_cfg = make_cfg(64, 1.5F, 6.0F, trap);
                    for (const bool frac : {false, true}) {
                        if (frac && pulse_w == 1) {
                            continue;
                        }
                        const auto fft_exact = run_fdmt_fft_boxcar_float(
                            root_plan, fft_cfg, BoxcarScreenMode::kPrefix,
                            fblocks, false, 1, 1, frac);
                        const auto fft_surr = run_fdmt_fft_boxcar_float(
                            root_plan, fft_cfg,
                            BoxcarScreenMode::kKadaneSurrogate, fblocks, false,
                            1, 1, frac);
                        CHECK(best_root_snr(fft_surr) >=
                              0.90F * best_root_snr(fft_exact));
                    }
                }
            }
        }
    }

    SECTION("Dynamic per-call kill_mask, cross-block open excursions, and "
            "outer OpenMP team safety in surrogate") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 8, 256, 0.001F, 24, 0, 1,
                                   "valid", 1);
        auto blocks = make_float_blocks(plan, 1, 3, 999U);
        for (SizeType c = 4; c < 8; ++c) {
            for (SizeType t = 0; t < 256; ++t) {
                blocks[1][(c * 256) + t] = 80.0F;
            }
        }
        const std::vector<std::vector<uint8_t>> masks{
            {}, {1, 1, 1, 1, 0, 0, 0, 0}, {}};
        auto cfg        = make_cfg(32, 1.5F, 6.0F);
        cfg.screen_mode = BoxcarScreenMode::kKadaneSurrogate;

        const auto masked_run = run_fdmt_boxcar_float(
            plan, cfg, BoxcarScreenMode::kKadaneSurrogate, blocks, true,
            Exec::cpu(1), 1, 64, true, {}, &masks, true);
        for (const auto& c : masked_run.candidates) {
            if (c.chan_begin >= 4 && c.begin_sample >= 256 &&
                c.end_sample < 512) {
                FAIL("Killed upper subband emitted a surrogate candidate");
            }
        }

        // Open surrogate excursion peaking at end of Block 0 and closing in
        // Block 1 / Block 2 / flush() across 0%->100% and 75%->0% kill masks.
        const plans::FDMTPlan xplan(700.0F, 1200.0F, 8, 256, 0.001F, 16);
        const std::vector<uint8_t> m_none(8, 1U);
        const std::vector<uint8_t> m_all(8, 0U);
        const std::vector<uint8_t> m_75{1, 1, 0, 0, 0, 0, 0, 0};
        const std::vector<float> zero_blk(8 * 256, 0.0F);
        std::vector<float> b0_full(8 * 256, 0.0F);
        std::vector<float> b0_75(8 * 256, 0.0F);
        std::vector<float> b1_hold(8 * 256, 0.0F);
        std::vector<float> b1_nudge(8 * 256, 0.0F);
        for (SizeType c = 0; c < 8; ++c) {
            for (SizeType t = 248; t < 256; ++t) {
                b0_full[(c * 256) + t] = 14.0F / std::sqrt(8.0F * 8.0F);
                if (c < 2) {
                    b0_75[(c * 256) + t] = 9.5F / std::sqrt(2.0F * 8.0F);
                }
            }
            for (SizeType t = 0; t < 75; ++t) {
                b1_hold[(c * 256) + t] = 0.09F; // Decays past t=64 into steady!
            }
            for (SizeType t = 0; t < 3; ++t) {
                b1_nudge[(c * 256) + t] = 0.35F; // Nudges next_c under m_none!
            }
        }
        const std::vector<std::vector<uint8_t>> mk_ref0{m_none};
        const std::vector<std::vector<uint8_t>> mk_0to100{m_none, m_all, m_all};
        const std::vector<std::vector<uint8_t>> mk_ref75{m_75};
        const std::vector<std::vector<uint8_t>> mk_75to0{m_75, m_none};
        for (const auto pm : {PeakFindingMode::kContiguousPeakPerWidth,
                              PeakFindingMode::kContiguousPeakAcrossWidths,
                              PeakFindingMode::kAllAboveThreshold}) {
            const auto xcfg = make_cfg(32, 1.5F, 6.0F, false, pm);
            auto xsnr = [&](const std::vector<std::vector<float>>& blks,
                            const std::vector<std::vector<uint8_t>>& mks) {
                const auto r = run_fdmt_boxcar_float(
                    xplan, xcfg, BoxcarScreenMode::kKadaneSurrogate, blks,
                    false, Exec::cpu(1), 1, 64, false, {}, &mks);
                float best = 0.0F;
                for (const auto& c : r.candidates) {
                    if (c.dm_idx == 0 && c.peak_sample >= 244 &&
                        c.peak_sample < 256) {
                        best = std::max(best, c.snr);
                    }
                }
                return best;
            };
            const float ref_a = xsnr({b0_full}, mk_ref0);
            REQUIRE(ref_a > 12.0F);
            CHECK(std::bit_cast<uint32_t>(
                      xsnr({b0_full, zero_blk, zero_blk}, mk_0to100)) ==
                  std::bit_cast<uint32_t>(ref_a));
            const float ref_b = xsnr({b0_75}, mk_ref75);
            REQUIRE(ref_b > 8.0F);
            for (const auto* b1 : std::array<const std::vector<float>*, 3>{
                     &zero_blk, &b1_hold, &b1_nudge}) {
                CHECK(std::bit_cast<uint32_t>(xsnr({b0_75, *b1}, mk_75to0)) ==
                      std::bit_cast<uint32_t>(ref_b));
            }
        }

        // Outer OpenMP parallel team calling independent Exec::cpu(1) instances
        // where pos - 1 at flush() lands on a mixed transition+steady boundary.
        const plans::FDMTPlan tplan(700.0F, 1200.0F, 8, 47, 0.001F, 16);
        auto tblocks = make_float_blocks(tplan, 1, 2, 1212U);
        for (SizeType c = 1; c < 8; ++c) {
            for (SizeType t = 39; t < 47; ++t) {
                tblocks[1][(c * 47) + t] += 2.8F; // Open at pos=94 in flush()!
            }
        }
        const std::vector<std::vector<uint8_t>> tmasks{{0, 1, 1, 1, 1, 1, 1, 1},
                                                       m_none};
        const auto tcfg       = make_cfg(32, 1.5F, 5.0F);
        const auto serial_ref = run_fdmt_boxcar_float(
            tplan, tcfg, BoxcarScreenMode::kKadaneSurrogate, tblocks, false,
            Exec::cpu(1), 1, 47, false, {}, &tmasks);
        REQUIRE_FALSE(serial_ref.candidates.empty());
        std::array<StreamRun, 2> par_runs{};
        std::array<bool, 2> par_fr_ok{true, true};
#pragma omp parallel for num_threads(2) schedule(static, 1)
        for (int w = 0; w < 2; ++w) {
            bool& ok = par_fr_ok[static_cast<SizeType>(w)];
            try {
                BoxcarConfig wcfg = tcfg;
                wcfg.screen_mode  = BoxcarScreenMode::kKadaneSurrogate;
                FDMTBoxcar s(tplan, wcfg, false, Exec::cpu(1), 1,
                             algorithms::kFDMTAutoFuse, 47);
                auto& dst   = par_runs[static_cast<SizeType>(w)];
                uint64_t fr = 0;
                auto acc = [&](const CandidateResultView& v, uint64_t next_fr) {
                    for (const auto& c : v.candidates) {
                        ok = ok && (c.begin_sample >= fr);
                    }
                    ok = ok && (next_fr >= fr) && !v.overflowed();
                    fr = next_fr;
                    dst.counts.emitted_count += v.emitted_count;
                    dst.counts.total_detected += v.total_detected;
                    dst.candidates.insert(dst.candidates.end(),
                                          v.candidates.begin(),
                                          v.candidates.end());
                };
                for (SizeType k = 0; k < tblocks.size(); ++k) {
                    acc(s.execute(tblocks[k], tmasks[k]),
                        s.get_candidate_frontier());
                }
                acc(s.flush(), s.get_candidate_frontier());
            } catch (...) {
                ok = false;
            }
        }
        CHECK((par_fr_ok[0] && par_fr_ok[1]));
        require_bit_exact(par_runs[0], serial_ref);
        require_bit_exact(par_runs[1], serial_ref);

        // Candidate overflow bounds check (max_candidates = 5 per call)
        auto ov_cfg             = make_cfg(32, 1.5F, 2.2F, false,
                                           PeakFindingMode::kAllAboveThreshold);
        ov_cfg.max_candidates   = 5;
        ov_cfg.max_cands_per_dm = 2;
        const auto ov_run       = run_fdmt_boxcar_float(
            plan, ov_cfg, BoxcarScreenMode::kKadaneSurrogate, blocks, true,
            Exec::cpu(1), 1, 64, false, {}, nullptr, true, true);
        CHECK(ov_run.counts.overflowed());
        CHECK(ov_run.counts.emitted_count <=
              (blocks.size() + 1) * ov_cfg.max_candidates);
        CHECK(ov_run.counts.dropped_per_dm > 0);
        CHECK(ov_run.counts.dropped_global > 0);
        CHECK(ov_run.counts.total_detected == ov_run.counts.emitted_count +
                                                  ov_run.counts.dropped_per_dm +
                                                  ov_run.counts.dropped_global);
    }
}

TEST_CASE("FDMTBoxcar kKeepBrightest works identically across all 4 "
          "BoxcarScreenModes",
          "[fdmt_boxcar_kadane][boxcar][kadane][overflow][cpu]") {
    using algorithms::detail::brighter;
    using boxcar_test::keep_brightest;
    using boxcar_test::require_same;
    using boxcar_test::run_calls;

    SECTION("Per-call top-q per DM and global top-K across kPrefix, kKadane, "
            "kNone, and kKadaneSurrogate (float, packed, multi-beam, flush)") {
        constexpr SizeType kBeams = 2;
        for (const SizeType depth : {SizeType{0}, SizeType{1}}) {
            const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 256, 0.001F,
                                       std::vector<IndexType>{0, 3, 9, 20, 41},
                                       "valid", depth);
            auto fblocks = make_float_blocks(plan, kBeams, 3, 2100U);
            // Elevate the tail of the final block across both beams so open
            // excursions drain during flush() under kContiguousPeak* modes.
            for (SizeType b = 0; b < kBeams; ++b) {
                for (SizeType c = 0; c < plan.get_nchans(); ++c) {
                    for (SizeType t = plan.get_nsamps() - 8;
                         t < plan.get_nsamps(); ++t) {
                        fblocks.back()[(((b * plan.get_nchans()) + c) *
                                        plan.get_nsamps()) +
                                       t] += 3.2F;
                    }
                }
            }
            std::vector<std::vector<uint8_t>> pblocks;
            pblocks.reserve(fblocks.size());
            for (const auto& blk : fblocks) {
                std::vector<uint8_t> codes(blk.size());
                for (SizeType i = 0; i < blk.size(); ++i) {
                    codes[i] = static_cast<uint8_t>(std::clamp(
                        std::lround((blk[i] * 16.0F) + 64.0F), 0L, 255L));
                }
                pblocks.push_back(std::move(codes));
            }

            for (const bool packed : {false, true}) {
                for (const auto peak_mode :
                     {PeakFindingMode::kContiguousPeakPerWidth,
                      PeakFindingMode::kContiguousPeakAcrossWidths,
                      PeakFindingMode::kAllAboveThreshold}) {
                    DYNAMIC_SECTION("depth=" << depth << " packed=" << packed
                                             << " mode="
                                             << static_cast<int>(peak_mode)) {
                        BoxcarConfig base_cfg;
                        base_cfg.max_boxcar    = 16;
                        base_cfg.wtsp          = 1.5F;
                        base_cfg.detect_thresh = 2.1F;
                        base_cfg.peak_mode     = peak_mode;
                        if (packed) {
                            base_cfg.input_nbits = 8;
                            base_cfg.input_mean  = 64.0F;
                            base_cfg.noise_sigma = 16.0F;
                        }

                        const auto run_mode_calls =
                            [&](BoxcarConfig c, BoxcarScreenMode m, int nt,
                                std::vector<CandidateCounts>* counts,
                                bool replay = false) {
                                c.screen_mode = m;
                                FDMTBoxcar search(
                                    plan, c, true, Exec::cpu(nt), kBeams,
                                    algorithms::kFDMTAutoFuse, 64);
                                auto calls =
                                    packed ? run_calls(search, pblocks, counts)
                                           : run_calls(search, fblocks, counts);
                                if (replay) {
                                    search.reset_history();
                                    std::vector<CandidateCounts> rep_counts;
                                    const auto rep_calls =
                                        packed ? run_calls(search, pblocks,
                                                           &rep_counts)
                                               : run_calls(search, fblocks,
                                                           &rep_counts);
                                    REQUIRE(rep_calls.size() == calls.size());
                                    for (SizeType i = 0; i < calls.size();
                                         ++i) {
                                        require_bit_exact_candidates(
                                            rep_calls[i], calls[i]);
                                        if (counts != nullptr) {
                                            check_counts_eq(rep_counts[i],
                                                            (*counts)[i]);
                                        }
                                    }
                                }
                                return calls;
                            };

                        // 1. Exact modes: kPrefix == kKadane == kNone ==
                        // keep_brightest(uncapped).
                        auto open_cfg             = base_cfg;
                        open_cfg.max_candidates   = SizeType{1} << 20;
                        open_cfg.max_cands_per_dm = 0;
                        const auto all_prefix     = run_mode_calls(
                            open_cfg, BoxcarScreenMode::kPrefix, 1, nullptr);
                        SizeType total_exact = 0;
                        for (const auto& v : all_prefix) {
                            total_exact += v.size();
                        }
                        REQUIRE(total_exact > 200);
                        if (peak_mode != PeakFindingMode::kAllAboveThreshold) {
                            CHECK_FALSE(all_prefix.back().empty());
                        }

                        for (const auto& [q, k_cap] :
                             {std::pair<SizeType, SizeType>{3, 1U << 20},
                              {2, 80},
                              {1, 25}}) {
                            auto kcfg = base_cfg;
                            kcfg.overflow_policy =
                                OverflowPolicy::kKeepBrightest;
                            kcfg.max_cands_per_dm = q;
                            kcfg.max_candidates   = k_cap;

                            std::vector<CandidateCounts> pfx_counts;
                            std::vector<CandidateCounts> kad_counts;
                            std::vector<CandidateCounts> kad2_counts;
                            std::vector<CandidateCounts> none_counts;
                            const auto pfx_kept =
                                run_mode_calls(kcfg, BoxcarScreenMode::kPrefix,
                                               1, &pfx_counts);
                            const auto kad_kept =
                                run_mode_calls(kcfg, BoxcarScreenMode::kKadane,
                                               1, &kad_counts, q == 2);
                            const auto kad2_kept =
                                run_mode_calls(kcfg, BoxcarScreenMode::kKadane,
                                               2, &kad2_counts);
                            const auto none_kept = run_mode_calls(
                                kcfg, BoxcarScreenMode::kNone, 1, &none_counts);

                            REQUIRE(pfx_kept.size() == all_prefix.size());
                            SizeType dropped_sum = 0;
                            for (SizeType i = 0; i < all_prefix.size(); ++i) {
                                CandidateCounts want;
                                const auto expected = keep_brightest(
                                    all_prefix[i], q, k_cap, want);
                                require_same(pfx_kept[i], expected);
                                require_bit_exact_candidates(kad_kept[i],
                                                             pfx_kept[i]);
                                require_bit_exact_candidates(kad2_kept[i],
                                                             pfx_kept[i]);
                                require_bit_exact_candidates(none_kept[i],
                                                             pfx_kept[i]);
                                check_counts_eq(pfx_counts[i], want);
                                check_counts_eq(kad_counts[i], want);
                                check_counts_eq(kad2_counts[i], want);
                                check_counts_eq(none_counts[i], want);
                                CHECK(kad_counts[i].total_detected ==
                                      kad_counts[i].emitted_count +
                                          kad_counts[i].dropped_per_dm +
                                          kad_counts[i].dropped_global);
                                dropped_sum +=
                                    want.dropped_per_dm + want.dropped_global;
                            }
                            CHECK(dropped_sum > 0);
                        }

                        // 2. kKadaneSurrogate: keeps exact top-q per DM and
                        // global top-K of uncapped kKadaneSurrogate.
                        const auto all_surr = run_mode_calls(
                            open_cfg, BoxcarScreenMode::kKadaneSurrogate, 1,
                            nullptr);
                        SizeType total_surr = 0;
                        for (const auto& v : all_surr) {
                            total_surr += v.size();
                        }
                        REQUIRE(total_surr > 100);
                        if (peak_mode != PeakFindingMode::kAllAboveThreshold) {
                            CHECK_FALSE(all_surr.back().empty());
                        }

                        for (const auto& [q, k_cap] :
                             {std::pair<SizeType, SizeType>{3, 1U << 20},
                              {2, 70},
                              {1, 20}}) {
                            auto scfg = base_cfg;
                            scfg.overflow_policy =
                                OverflowPolicy::kKeepBrightest;
                            scfg.max_cands_per_dm = q;
                            scfg.max_candidates   = k_cap;
                            std::vector<CandidateCounts> s1_counts;
                            std::vector<CandidateCounts> s2_counts;
                            const auto s1_kept = run_mode_calls(
                                scfg, BoxcarScreenMode::kKadaneSurrogate, 1,
                                &s1_counts, q == 2);
                            const auto s2_kept = run_mode_calls(
                                scfg, BoxcarScreenMode::kKadaneSurrogate, 2,
                                &s2_counts);
                            REQUIRE(s1_kept.size() == all_surr.size());
                            SizeType dropped_sum = 0;
                            for (SizeType i = 0; i < all_surr.size(); ++i) {
                                CandidateCounts want;
                                const auto expected =
                                    keep_brightest(all_surr[i], q, k_cap, want);
                                require_same(s1_kept[i], expected);
                                require_bit_exact_candidates(s1_kept[i],
                                                             expected);
                                require_bit_exact_candidates(s2_kept[i],
                                                             expected);
                                check_counts_eq(s1_counts[i], want);
                                check_counts_eq(s2_counts[i], want);
                                CHECK(s1_counts[i].total_detected ==
                                      s1_counts[i].emitted_count +
                                          s1_counts[i].dropped_per_dm +
                                          s1_counts[i].dropped_global);
                                dropped_sum +=
                                    want.dropped_per_dm + want.dropped_global;
                            }
                            CHECK(dropped_sum > 0);
                        }
                    }
                }
            }
        }
    }

    SECTION("Early broadband RFI storm + late bright burst and memory "
            "accounting across all 4 modes") {
        const plans::FDMTPlan plan(1200.0F, 1500.0F, 64, 4096, 0.001F, 63, 0, 1,
                                   "valid", 1);
        std::mt19937 rng(777U); // NOLINT(bugprone-random-generator-seed)
        std::vector<float> block(plan.get_nchans() * plan.get_nsamps());
        for (SizeType c = 0; c < plan.get_nchans(); ++c) {
            for (SizeType t = 0; t < plan.get_nsamps(); ++t) {
                float v = next_gaussian(rng);
                if (t >= 100 && t < 2500) {
                    const SizeType phase = (t - 100) % 48;
                    v += (phase < 12) ? 3.2F : -0.8F;
                }
                if (t >= 3500 && t < 3504) {
                    v += 14.0F;
                }
                block[(c * plan.get_nsamps()) + t] = v;
            }
        }
        const auto has_late_pulse = [](const std::vector<RawCandidate>& v) {
            return std::ranges::any_of(v, [](const RawCandidate& c) {
                return c.peak_sample >= 3495 && c.peak_sample < 3525;
            });
        };

        std::vector<RawCandidate> pfx_kept_ref;
        CandidateCounts pfx_counts_ref{};
        for (const auto mode :
             {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kKadane,
              BoxcarScreenMode::kNone, BoxcarScreenMode::kKadaneSurrogate}) {
            BoxcarConfig cfg;
            cfg.max_boxcar       = 16;
            cfg.detect_thresh    = 6.0F;
            cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
            cfg.max_cands_per_dm = 8;
            cfg.max_candidates   = 120;
            cfg.screen_mode      = mode;

            FDMTBoxcar drop_search(plan, cfg, true, Exec::cpu(2));
            SizeType total_rows = 0;
            for (const auto& sp : drop_search.get_search_plans()) {
                total_rows += sp.dm_grid.size();
            }
            const auto dropped =
                drop_search.execute(std::span<const float>(block));
            REQUIRE(dropped.overflowed());
            CHECK_FALSE(has_late_pulse(
                {dropped.candidates.begin(), dropped.candidates.end()}));

            cfg.overflow_policy = OverflowPolicy::kKeepBrightest;
            FDMTBoxcar keep_search(plan, cfg, true, Exec::cpu(2));
            CHECK(
                keep_search.get_memory_usage().candidates ==
                drop_search.get_memory_usage().candidates +
                    (total_rows * cfg.max_cands_per_dm * sizeof(RawCandidate)));
            const auto kept =
                keep_search.execute(std::span<const float>(block));
            REQUIRE(kept.overflowed());
            const std::vector<RawCandidate> v(kept.candidates.begin(),
                                              kept.candidates.end());
            CHECK(v.size() == cfg.max_candidates);
            REQUIRE(has_late_pulse(v));
            const auto best = std::ranges::min_element(v, brighter);
            CHECK(best->peak_sample >= 3495);
            CHECK(best->peak_sample < 3525);
            CHECK(kept.total_detected == dropped.total_detected);
            CHECK(kept.total_detected == kept.emitted_count +
                                             kept.dropped_per_dm +
                                             kept.dropped_global);

            if (mode == BoxcarScreenMode::kPrefix) {
                pfx_kept_ref   = v;
                pfx_counts_ref = static_cast<const CandidateCounts&>(kept);
            } else if (mode != BoxcarScreenMode::kKadaneSurrogate) {
                require_bit_exact_candidates(v, pfx_kept_ref);
                check_counts_eq(static_cast<const CandidateCounts&>(kept),
                                pfx_counts_ref);
            }
        }
    }

    SECTION("FDMTFFTBoxcar kKeepBrightest across all 4 modes") {
        const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 256, 0.001F,
                                   std::vector<IndexType>{0, 3, 9, 20, 41});
        const auto fblocks = make_float_blocks(plan, 1, 3, 2200U);
        for (const auto mode :
             {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kKadane,
              BoxcarScreenMode::kNone, BoxcarScreenMode::kKadaneSurrogate}) {
            BoxcarConfig open_cfg;
            open_cfg.max_boxcar       = 16;
            open_cfg.wtsp             = 1.5F;
            open_cfg.detect_thresh    = 2.1F;
            open_cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
            open_cfg.max_candidates   = SizeType{1} << 20;
            open_cfg.max_cands_per_dm = 0;
            open_cfg.screen_mode      = mode;

            FDMTFFTBoxcar all_search(plan, open_cfg, true, Exec::cpu(2));
            const auto all = run_calls(all_search, fblocks);

            auto kcfg             = open_cfg;
            kcfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
            kcfg.max_cands_per_dm = 4;
            kcfg.max_candidates   = 50;
            FDMTFFTBoxcar kept_search(plan, kcfg, true, Exec::cpu(2));
            std::vector<CandidateCounts> counts;
            const auto kept = run_calls(kept_search, fblocks, &counts);
            REQUIRE(kept.size() == all.size());
            SizeType dropped = 0;
            for (SizeType i = 0; i < all.size(); ++i) {
                CandidateCounts want;
                const auto expected = keep_brightest(all[i], 4, 50, want);
                require_same(kept[i], expected);
                require_bit_exact_candidates(kept[i], expected);
                check_counts_eq(counts[i], want);
                dropped += want.dropped_per_dm + want.dropped_global;
            }
            CHECK(dropped > 0);
        }
    }
}

TEST_CASE("FDMTBoxcar drop_block_noise rollback on throw preserves skewed "
          "subband stream across kKadane, kNone, and kKadaneSurrogate",
          "[fdmt_boxcar_kadane][boxcar][kadane][skew][mask][cpu]") {
    using skew_search_test::run_subband_search;
    using skew_search_test::search_config;

    constexpr SizeType kBlocks = 4;
    const plans::FDMTPlan plan(856.0F, 1712.0F, 64, 256, 0.0005F, 64, 0, 1,
                               "valid", 1);
    const auto masks = make_changing_masks(plan.get_nchans(), kBlocks);

    for (const bool use_drain : {true, false}) {
        for (const bool trap : {false, true}) {
            DYNAMIC_SECTION("drain=" << use_drain << " trap=" << trap) {
                auto pfx_cfg          = search_config(false, 1.5F);
                pfx_cfg.detect_thresh = 3.2F;
                pfx_cfg.use_trapezoid = trap;
                pfx_cfg.screen_mode   = BoxcarScreenMode::kPrefix;
                const auto planes =
                    algorithms::detail::make_boxcar_plans(plan, pfx_cfg, true);

                const auto pfx_clean = run_subband_search(
                    plan, Exec::cpu(1), true, false, kBlocks, 64, 1, use_drain,
                    &masks, 1.5F, &planes, SIZE_MAX, &pfx_cfg);
                REQUIRE(pfx_clean.path.find("skew") != std::string::npos);
                REQUIRE(pfx_clean.latency > 0);
                REQUIRE(pfx_clean.candidates.size() > 150);

                // Confirm dynamic per-call kill masks actively alter output.
                const auto pfx_unmasked = run_subband_search(
                    plan, Exec::cpu(1), true, false, kBlocks, 64, 1, use_drain,
                    nullptr, 1.5F, &planes, SIZE_MAX, &pfx_cfg);
                REQUIRE(pfx_unmasked.candidates.size() !=
                        pfx_clean.candidates.size());

                for (const auto mode :
                     {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone,
                      BoxcarScreenMode::kKadaneSurrogate}) {
                    if (mode == BoxcarScreenMode::kNone && !use_drain) {
                        continue;
                    }
                    auto cfg         = pfx_cfg;
                    cfg.screen_mode  = mode;
                    const auto clean = run_subband_search(
                        plan, Exec::cpu(1), true, false, kBlocks, 64, 1,
                        use_drain, &masks, 1.5F, &planes, SIZE_MAX, &cfg);
                    REQUIRE(clean.latency > 0);
                    REQUIRE(clean.candidates.size() > 100);
                    if (mode != BoxcarScreenMode::kKadaneSurrogate) {
                        require_bit_exact_candidates(clean.candidates,
                                                     pfx_clean.candidates);
                    }

                    for (const SizeType fail_at : {SizeType{1}, SizeType{3}}) {
                        const auto failed = run_subband_search(
                            plan, Exec::cpu(1), true, false, kBlocks, 64, 1,
                            use_drain, &masks, 1.5F, &planes, fail_at, &cfg);
                        require_bit_exact_candidates(failed.candidates,
                                                     clean.candidates);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar kKadane and kNone match kPrefix bit-for-bit on wide "
          "boxcars (max_boxcar = 8192 and 32768)",
          "[fdmt_boxcar_kadane][boxcar][kadane][wide][cpu]") {
    SECTION("max_boxcar = 8192 across multi-block float32, packed 8/16-bit, "
            "trapezoids, and FDMTFFTBoxcar") {
        // nsamps = 2048 < max_boxcar = 8192: carry spans 4+ blocks.
        const plans::FDMTPlan plan(700.0F, 1200.0F, 4, 2048, 0.001F,
                                   std::vector<IndexType>{-4, 0, 5, 18},
                                   "valid", 1);
        auto fblocks = make_float_blocks(plan, 1, 6, 3100U);
        // Sustained elevated plateau across blocks 1..3 (6144 samples) so
        // boxcars with width > 4096 trigger and emit candidates.
        for (SizeType k = 1; k <= 3; ++k) {
            for (float& x : fblocks[k]) {
                x += 0.35F;
            }
        }

        for (const bool trap : {false, true}) {
            DYNAMIC_SECTION(
                "float32 FDMTBoxcar max_boxcar=8192 trap=" << trap) {
                const auto cfg =
                    make_cfg(8192, 1.5F, 2.9F, trap,
                             PeakFindingMode::kContiguousPeakAcrossWidths);
                const auto ref = run_fdmt_boxcar_float(
                    plan, cfg, BoxcarScreenMode::kPrefix, fblocks, true,
                    Exec::cpu(1), 1, 0, true);
                REQUIRE_FALSE(ref.candidates.empty());
                REQUIRE(std::ranges::any_of(
                    ref.candidates,
                    [](const RawCandidate& c) { return c.width > 4096; }));
                require_bit_exact(
                    run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kKadane,
                                          fblocks, true, Exec::cpu(1), 1, 0,
                                          true, {}, nullptr, true),
                    ref);
                require_bit_exact(run_fdmt_boxcar_float(
                                      plan, cfg, BoxcarScreenMode::kNone,
                                      fblocks, true, Exec::cpu(1), 1, 0, true),
                                  ref);
            }

            for (const SizeType nbits : {SizeType{8}, SizeType{16}}) {
                DYNAMIC_SECTION("packed "
                                << nbits
                                << "-bit max_boxcar=8192 trap=" << trap) {
                    const auto max_code = (1U << nbits) - 1U;
                    const auto maxf     = static_cast<float>(max_code);
                    auto pblocks        = make_packed_blocks(
                        plan, 1, 6, nbits,
                        3200U + static_cast<uint32_t>(nbits));
                    // Elevate blocks 1..3 slightly above mean to trigger
                    // widths > 4096 on both int32 (8-bit) and !has_int32
                    // centered-float32 (16-bit) Kadane gates.
                    const uint32_t bump = nbits == 8 ? 14U : 3600U;
                    for (SizeType k = 1; k <= 3; ++k) {
                        if (nbits == 8) {
                            for (uint8_t& v : pblocks[k]) {
                                v = static_cast<uint8_t>(
                                    std::min<uint32_t>(max_code, v + bump));
                            }
                        } else {
                            for (SizeType i = 0; i < pblocks[k].size();
                                 i += 2) {
                                const uint32_t code =
                                    static_cast<uint32_t>(pblocks[k][i]) |
                                    (static_cast<uint32_t>(pblocks[k][i + 1])
                                     << 8U);
                                const uint32_t next =
                                    std::min<uint32_t>(max_code, code + bump);
                                pblocks[k][i] =
                                    static_cast<uint8_t>(next & 0xFFU);
                                pblocks[k][i + 1] =
                                    static_cast<uint8_t>(next >> 8U);
                            }
                        }
                    }
                    const auto pcfg = make_packed_cfg(
                        8192, 1.5F, 2.8F, nbits, maxf * 0.5F,
                        std::sqrt(maxf * (maxf + 2.0F) / 12.0F), trap);
                    const auto ref = run_fdmt_boxcar_packed(
                        plan, pcfg, BoxcarScreenMode::kPrefix, pblocks, true,
                        Exec::cpu(1), 1, 0, true);
                    REQUIRE_FALSE(ref.candidates.empty());
                    REQUIRE(std::ranges::any_of(
                        ref.candidates,
                        [](const RawCandidate& c) { return c.width > 4096; }));
                    require_bit_exact(run_fdmt_boxcar_packed(
                                          plan, pcfg, BoxcarScreenMode::kKadane,
                                          pblocks, true, Exec::cpu(1), 1, 0,
                                          true, {}, nbits == 8),
                                      ref);
                    require_bit_exact(
                        run_fdmt_boxcar_packed(plan, pcfg,
                                               BoxcarScreenMode::kNone, pblocks,
                                               true, Exec::cpu(1), 1, 0, true),
                        ref);
                }
            }

            DYNAMIC_SECTION(
                "FDMTFFTBoxcar float32 max_boxcar=8192 trap=" << trap) {
                const plans::FDMTPlan fft_plan(700.0F, 1200.0F, 4, 4096, 0.001F,
                                               16);
                auto fft_blocks = make_float_blocks(fft_plan, 1, 3, 3300U);
                for (float& x : fft_blocks[1]) {
                    x += 0.35F;
                }
                for (SizeType i = 0; i < fft_blocks[2].size() / 2; ++i) {
                    fft_blocks[2][i] += 0.35F;
                }
                const auto cfg = make_cfg(8192, 1.5F, 2.9F, trap);
                const auto ref = run_fdmt_fft_boxcar_float(
                    fft_plan, cfg, BoxcarScreenMode::kPrefix, fft_blocks, true,
                    1, 1, false);
                REQUIRE_FALSE(ref.candidates.empty());
                REQUIRE(std::ranges::any_of(
                    ref.candidates,
                    [](const RawCandidate& c) { return c.width > 4096; }));
                require_bit_exact(run_fdmt_fft_boxcar_float(
                                      fft_plan, cfg, BoxcarScreenMode::kKadane,
                                      fft_blocks, true, 1, 1, false),
                                  ref);
                require_bit_exact(run_fdmt_fft_boxcar_float(
                                      fft_plan, cfg, BoxcarScreenMode::kNone,
                                      fft_blocks, true, 1, 1, false),
                                  ref);
            }
        }
    }

    SECTION("max_boxcar = 32768 across multi-block float32 and packed "
            "8/16-bit with pure boxcar and {1,2,1} trapezoid") {
        // nsamps = 8192 (> 4096 chunk size, < 32768 max_boxcar) over 6 blocks
        // = 49152 samples: tests both m_kadane_exact_prefix_cap > 4096 and
        // trapezoid max_edge_lag = 8192 (> 16 gate tiles, LagWindow coalesce).
        const plans::FDMTPlan plan(700.0F, 1200.0F, 4, 8192, 0.001F,
                                   std::vector<IndexType>{-3, 0, 6, 16},
                                   "valid", 1);
        auto fblocks = make_float_blocks(plan, 1, 6, 3400U);
        // Sustained 24576-sample plateau across blocks 1..3 so widths > 16384
        // trigger cleanly.
        for (SizeType k = 1; k <= 3; ++k) {
            for (float& x : fblocks[k]) {
                x += 0.22F;
            }
        }

        for (const bool trap : {false, true}) {
            DYNAMIC_SECTION(
                "float32 FDMTBoxcar max_boxcar=32768 trap=" << trap) {
                const auto cfg = make_cfg(32768, 1.5F, 2.9F, trap);
                const auto ref = run_fdmt_boxcar_float(
                    plan, cfg, BoxcarScreenMode::kPrefix, fblocks, true,
                    Exec::cpu(1), 1, 8192, true);
                REQUIRE_FALSE(ref.candidates.empty());
                REQUIRE(std::ranges::any_of(
                    ref.candidates,
                    [](const RawCandidate& c) { return c.width > 16384; }));
                require_bit_exact(
                    run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kKadane,
                                          fblocks, true, Exec::cpu(1), 1, 8192,
                                          true, {}, nullptr, !trap),
                    ref);
                require_bit_exact(
                    run_fdmt_boxcar_float(plan, cfg, BoxcarScreenMode::kNone,
                                          fblocks, true, Exec::cpu(1), 1, 8192,
                                          true),
                    ref);
            }

            for (const SizeType nbits : {SizeType{8}, SizeType{16}}) {
                DYNAMIC_SECTION("packed "
                                << nbits
                                << "-bit max_boxcar=32768 trap=" << trap) {
                    const auto max_code = (1U << nbits) - 1U;
                    const auto maxf     = static_cast<float>(max_code);
                    auto pblocks        = make_packed_blocks(
                        plan, 1, 6, nbits,
                        3500U + static_cast<uint32_t>(nbits));
                    const uint32_t bump = nbits == 8 ? 8U : 2000U;
                    for (SizeType k = 1; k <= 3; ++k) {
                        if (nbits == 8) {
                            for (uint8_t& v : pblocks[k]) {
                                v = static_cast<uint8_t>(
                                    std::min<uint32_t>(max_code, v + bump));
                            }
                        } else {
                            for (SizeType i = 0; i < pblocks[k].size();
                                 i += 2) {
                                const uint32_t code =
                                    static_cast<uint32_t>(pblocks[k][i]) |
                                    (static_cast<uint32_t>(pblocks[k][i + 1])
                                     << 8U);
                                const uint32_t next =
                                    std::min<uint32_t>(max_code, code + bump);
                                pblocks[k][i] =
                                    static_cast<uint8_t>(next & 0xFFU);
                                pblocks[k][i + 1] =
                                    static_cast<uint8_t>(next >> 8U);
                            }
                        }
                    }
                    const auto pcfg = make_packed_cfg(
                        32768, 1.5F, 2.8F, nbits, maxf * 0.5F,
                        std::sqrt(maxf * (maxf + 2.0F) / 12.0F), trap);
                    const auto ref = run_fdmt_boxcar_packed(
                        plan, pcfg, BoxcarScreenMode::kPrefix, pblocks, true,
                        Exec::cpu(1), 1, 8192, true);
                    REQUIRE_FALSE(ref.candidates.empty());
                    REQUIRE(std::ranges::any_of(
                        ref.candidates,
                        [](const RawCandidate& c) { return c.width > 16384; }));
                    require_bit_exact(run_fdmt_boxcar_packed(
                                          plan, pcfg, BoxcarScreenMode::kKadane,
                                          pblocks, true, Exec::cpu(1), 1, 8192,
                                          true),
                                      ref);
                    require_bit_exact(run_fdmt_boxcar_packed(
                                          plan, pcfg, BoxcarScreenMode::kNone,
                                          pblocks, true, Exec::cpu(1), 1, 8192,
                                          true),
                                      ref);
                }
            }
        }
    }
}
