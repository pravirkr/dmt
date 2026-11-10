// FDMTBoxcar per-call (per-gulp) kill mask on the CPU: execute(waterfall,
// kill_mask) against the old overload, the constructor mask and a direct
// reference with the conservative transition rule; validation, state and
// memory; the noise model against make_boxcar_plans() and brute force.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_noise.hpp"
#include "dmt/fdmt_variance.hpp"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;

namespace {

using Masks = std::vector<std::vector<uint8_t>>;

// 32 channels, 400 samples per block; supports up to ~120 samples, so a
// per-call mask fits with boxcars up to 37 (S + W - 1 <= 400).
plans::FDMTPlan mask_plan(SizeType depth) {
    return {700,     1200,   32,
            400,     0.001F, std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120},
            "valid", depth};
}

BoxcarConfig mask_config(PeakFindingMode mode, SizeType max_boxcar = 37) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = max_boxcar;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.peak_mode        = mode;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = 0;
    return cfg;
}

std::vector<uint8_t> random_mask(SizeType n, unsigned seed, double killed) {
    std::mt19937 gen(seed);
    std::bernoulli_distribution kill(killed);
    std::vector<uint8_t> m(n);
    for (auto& v : m) {
        v = kill(gen) ? 0 : static_cast<uint8_t>(1 + (gen() % 200));
    }
    return m;
}

// Exact equality of two candidate sets (every field, S/N bit for bit).
void compare_exact(std::vector<RawCandidate> a, std::vector<RawCandidate> b) {
    sort(a);
    sort(b);
    REQUIRE(a.size() == b.size());
    for (SizeType i = 0; i < a.size(); ++i) {
        INFO("candidate " << i);
        REQUIRE(key(a[i]) == key(b[i]));
        REQUIRE(std::bit_cast<uint32_t>(a[i].snr) ==
                std::bit_cast<uint32_t>(b[i].snr));
        REQUIRE(a[i].width == b[i].width);
        REQUIRE(a[i].dm == b[i].dm);
        REQUIRE(a[i].staggered == b[i].staggered);
    }
}

// Streams `input` through `search` with per-block masks (an empty entry:
// the old overload when `old_for_empty`, else an empty span), then ends
// with flush() or drain().
std::vector<RawCandidate> run(FDMTBoxcar& search,
                              const std::vector<std::vector<float>>& input,
                              const Masks& masks,
                              bool drain         = false,
                              bool old_for_empty = true) {
    std::vector<RawCandidate> out;
    for (SizeType k = 0; k < input.size(); ++k) {
        const auto& m = masks[k];
        if (m.empty() && old_for_empty) {
            append(out, search.execute(std::span<const float>(input[k])));
        } else {
            append(out, search.execute(std::span<const float>(input[k]),
                                       std::span<const uint8_t>(m)));
        }
    }
    append(out, drain ? search.drain() : search.flush());
    return out;
}

FDMTBoxcar make_search(const plans::FDMTPlan& plan,
                       const BoxcarConfig& cfg,
                       SizeType beams                = 1,
                       SizeType chunk                = 0,
                       std::span<const uint8_t> ctor = {},
                       SizeType max_memory           = 0,
                       bool smear                    = true) {
    return FDMTBoxcar(
        plan, cfg, smear, Exec::cpu(3), beams, algorithms::kFDMTAutoFuse, chunk,
        max_memory == 0 ? algorithms::kFDMTDefaultMaxMemory : max_memory, ctor);
}

} // namespace

TEST_CASE("Boxcar noise model equals make_boxcar_plans for any mask",
          "[boxcar][kill][mask][cpu]") {
    using algorithms::detail::BoxcarNoiseModel;
    using algorithms::detail::make_boxcar_plans;
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        const auto plan = mask_plan(depth);
        for (const bool smear : {true, false}) {
            for (const bool trap : {false, true}) {
                DYNAMIC_SECTION("depth " << depth << " smear " << smear
                                         << " trapezoid " << trap) {
                    auto cfg =
                        mask_config(PeakFindingMode::kContiguousPeakPerWidth);
                    cfg.use_trapezoid = trap;
                    BoxcarNoiseModel model(plan, cfg, smear, 2);
                    std::vector<std::vector<uint8_t>> masks = {
                        std::vector<uint8_t>(32, 1),
                        std::vector<uint8_t>(32, 0),
                        random_mask(32, 1, 0.3),
                        random_mask(32, 2, 0.8),
                    };
                    masks.push_back(std::vector<uint8_t>(32, 1));
                    std::fill_n(masks.back().begin(), 16, 0);
                    for (const auto& m : masks) {
                        const auto planes =
                            make_boxcar_plans(plan, cfg, smear, m);
                        const auto t = model.compute(m);
                        SizeType k   = 0;
                        for (const auto& p : planes) {
                            for (SizeType i = 0; i < p.raw_thresh_lut.size();
                                 ++i, ++k) {
                                REQUIRE(std::bit_cast<uint32_t>(
                                            p.raw_thresh_lut[i]) ==
                                        std::bit_cast<uint32_t>(t->thresh[k]));
                                REQUIRE(
                                    std::bit_cast<uint32_t>(
                                        p.inv_sigma_lut[i]) ==
                                    std::bit_cast<uint32_t>(t->inv_sigma[k]));
                            }
                        }
                        REQUIRE(k == t->thresh.size());
                    }
                    // Supports: the warm-up with every channel active.
                    const auto planes = make_boxcar_plans(plan, cfg, smear);
                    SizeType r        = 0;
                    for (const auto& p : planes) {
                        for (const auto w : p.warmup_samples) {
                            REQUIRE((*model.support())[r++] == w);
                        }
                    }
                    REQUIRE(model.live_table_bytes() == 0);
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar per-call mask: empty or all-ones is the old overload",
          "[boxcar][kill][mask][cpu]") {
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        const auto plan  = mask_plan(depth);
        const auto input = waterfalls(plan, 2, 4);
        for (const SizeType width : {SizeType{1}, SizeType{37}}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                for (const bool trap : {false, true}) {
                    DYNAMIC_SECTION("depth "
                                    << depth << " width " << width << " mode "
                                    << static_cast<int>(mode) << " trapezoid "
                                    << trap) {
                        auto cfg          = mask_config(mode, width);
                        cfg.use_trapezoid = trap;
                        auto old          = make_search(plan, cfg, 2);
                        const auto base = run(old, input, Masks(input.size()));
                        REQUIRE_FALSE(base.empty());
                        auto empty = make_search(plan, cfg, 2);
                        compare_exact(run(empty, input, Masks(input.size()),
                                          false, false),
                                      base);
                        auto ones = make_search(plan, cfg, 2);
                        const Masks all(input.size(),
                                        std::vector<uint8_t>(32, 1));
                        compare_exact(run(ones, input, all), base);
                        // Mixed with the old overload, and after a reset.
                        Masks mixed(input.size());
                        mixed[1] = all[1];
                        mixed[3] = all[3];
                        ones.reset_history();
                        compare_exact(run(ones, input, mixed), base);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar per-call mask: a constant mask is the constructor mask",
          "[boxcar][kill][mask][cpu]") {
    // Bit for bit without the cold-prefix skip; with it, the per-call mask
    // keeps the constructor mask's (longer) warm-up, which the reference
    // models.
    const auto plan  = mask_plan(2);
    const auto input = waterfalls(plan, 1, 4);
    const auto m     = random_mask(32, 7, 0.4);
    for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                            PeakFindingMode::kContiguousPeakAcrossWidths,
                            PeakFindingMode::kAllAboveThreshold}) {
        DYNAMIC_SECTION("mode " << static_cast<int>(mode)) {
            auto cfg             = mask_config(mode);
            cfg.skip_cold_prefix = false;
            auto ctor            = make_search(plan, cfg, 1, 0, m);
            const auto base      = run(ctor, input, Masks(input.size()));
            REQUIRE_FALSE(base.empty());
            auto call = make_search(plan, cfg);
            compare_exact(run(call, input, Masks(input.size(), m)), base);

            cfg.skip_cold_prefix = true;
            auto cold            = make_search(plan, cfg);
            const auto expected  = masked_reference(plan, cfg, true, input, {},
                                                    Masks(input.size(), m));
            compare(run(cold, input, Masks(input.size(), m)),
                    expected.candidates);
        }
    }
}

TEST_CASE("FDMTBoxcar per-call masks changing every block match the reference",
          "[boxcar][kill][mask][cpu]") {
    // Changes that kill, revive and swap channels (inside the dispersion
    // sweep and the boxcars of the pulses at every block boundary), an
    // omitted mask (the constructor mask alone), a wholly killed block, and
    // the constructor mask ANDed with all of them.
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        const auto plan  = mask_plan(depth);
        const auto input = waterfalls(plan, 1, 8);
        Masks masks      = {
            random_mask(32, 11, 0.2),    random_mask(32, 11, 0.2),
            random_mask(32, 12, 0.5),    {},
            std::vector<uint8_t>(32, 0), random_mask(32, 13, 0.1),
            std::vector<uint8_t>(32, 1), random_mask(32, 14, 0.6),
        };
        for (const bool with_ctor : {false, true}) {
            const auto ctor =
                with_ctor ? random_mask(32, 99, 0.25) : std::vector<uint8_t>{};
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                for (const bool trap : {false, true}) {
                    DYNAMIC_SECTION("depth " << depth << " ctor " << with_ctor
                                             << " mode "
                                             << static_cast<int>(mode)
                                             << " trapezoid " << trap) {
                        auto cfg            = mask_config(mode);
                        cfg.use_trapezoid   = trap;
                        const auto expected = masked_reference(
                            plan, cfg, true, input, ctor, masks);
                        REQUIRE(expected.candidates.size() > 50);
                        for (const SizeType chunk :
                             {SizeType{64}, plan.get_nsamps()}) {
                            auto search =
                                make_search(plan, cfg, 1, chunk, ctor);
                            compare(run(search, input, masks),
                                    expected.candidates);
                            // drain() pads with the last mask: same result
                            // off the skewed path.
                            search.reset_history();
                            compare(run(search, input, masks, true),
                                    expected.candidates);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar per-call mask: a killed block hides its values but "
          "closes earlier detections",
          "[boxcar][kill][mask][cpu]") {
    const auto plan   = mask_plan(0);
    const SizeType ns = plan.get_nsamps();
    auto input        = waterfalls(plan, 1, 4);
    // Large finite values in every row of block 2, which is wholly killed.
    std::ranges::fill(input[2], 1.0e6F);
    // A pulse in every channel ending at the last sample of block 1.
    for (SizeType c = 0; c < 32; ++c) {
        for (SizeType t = ns - 6; t < ns; ++t) {
            input[1][(c * ns) + t] += 4.0F;
        }
    }
    Masks masks(4);
    masks[2]       = std::vector<uint8_t>(32, 0);
    const auto cfg = mask_config(PeakFindingMode::kContiguousPeakPerWidth);
    auto search    = make_search(plan, cfg);
    std::vector<RawCandidate> during;
    for (SizeType k = 0; k < 4; ++k) {
        const auto r = search.execute(std::span<const float>(input[k]),
                                      std::span<const uint8_t>(masks[k]));
        if (k == 2) {
            append(during, r);
        }
    }
    // Runs of the pulse still open at the end of block 1 close in block 2:
    // their transition endpoints use the OR (here: previous) variance.
    REQUIRE_FALSE(during.empty());
    uint64_t support = 0;
    for (const auto& p : search.get_search_plans()) {
        for (const auto w : p.warmup_samples) {
            support = std::max(support, w);
        }
    }
    for (const auto& c : during) {
        INFO("peak " << c.peak_sample << " snr " << c.snr);
        CHECK(c.peak_sample < 2 * ns + support + cfg.max_boxcar - 1);
        CHECK(c.snr < 100.0F);
    }
    const auto expected = masked_reference(plan, cfg, true, input, {}, masks);
    search.reset_history();
    compare(run(search, input, masks), expected.candidates);
}

TEST_CASE("FDMTBoxcar per-call mask: rejected calls leave the stream unchanged",
          "[boxcar][kill][mask][cpu]") {
    const auto plan     = mask_plan(1);
    const auto input    = waterfalls(plan, 1, 4);
    const auto cfg      = mask_config(PeakFindingMode::kContiguousPeakPerWidth);
    const Masks masks   = {random_mask(32, 3, 0.3),
                           random_mask(32, 4, 0.3),
                           {},
                           random_mask(32, 5, 0.5)};
    const auto expected = masked_reference(plan, cfg, true, input, {}, masks);

    SECTION("wrong mask length, wrong waterfall size, device input") {
        auto search = make_search(plan, cfg);
        std::vector<RawCandidate> out;
        for (SizeType k = 0; k < input.size(); ++k) {
            const std::vector<uint8_t> short_mask(31, 1);
            CHECK_THROWS_AS(
                search.execute(std::span<const float>(input[k]),
                               std::span<const uint8_t>(short_mask)),
                std::invalid_argument);
            const std::span<const float> part(input[k].data(), 100);
            CHECK_THROWS_AS(
                search.execute(part, std::span<const uint8_t>(masks[0])),
                std::invalid_argument);
            CHECK_THROWS(search.execute(DeviceSpan<const float>(input[k].data(),
                                                                input[k].size(),
                                                                Device{}),
                                        std::span<const uint8_t>(masks[0])));
            const auto& m = masks[k];
            append(out, m.empty()
                            ? search.execute(std::span<const float>(input[k]))
                            : search.execute(std::span<const float>(input[k]),
                                             std::span<const uint8_t>(m)));
        }
        append(out, search.flush());
        compare(out, expected.candidates);
    }

    SECTION("a block shorter than the support needs") {
        // 100-sample blocks: the largest support plus max_boxcar - 1 does
        // not fit; the old overload keeps working.
        const plans::FDMTPlan short_plan(
            700, 1200, 32, 100, 0.001F,
            std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120}, "valid", 1);
        const auto blocks = waterfalls(short_plan, 1, 3);
        auto search       = make_search(short_plan, cfg);
        auto fresh        = make_search(short_plan, cfg);
        const auto before = search.get_memory_usage().total();
        CHECK_THROWS_AS(search.execute(std::span<const float>(blocks[0]),
                                       std::span<const uint8_t>(masks[0])),
                        std::invalid_argument);
        CHECK(search.get_memory_usage().total() == before);
        compare_exact(run(search, blocks, Masks(3)),
                      run(fresh, blocks, Masks(3)));
    }

    SECTION("the memory budget") {
        auto probe        = make_search(plan, cfg, 1, 128);
        const auto needed = probe.get_memory_usage().total();
        auto tight        = make_search(plan, cfg, 1, 128, {}, needed + 1024);
        REQUIRE(tight.get_memory_usage().total() == needed);
        CHECK_THROWS_AS(tight.execute(std::span<const float>(input[0]),
                                      std::span<const uint8_t>(masks[0])),
                        std::invalid_argument);
        CHECK(tight.get_memory_usage().total() == needed);
        auto fresh = make_search(plan, cfg, 1, 128);
        compare_exact(run(tight, input, Masks(4)), run(fresh, input, Masks(4)));
    }
}

TEST_CASE("FDMTBoxcar per-call mask: memory is accounted and reset clears the "
          "mask state",
          "[boxcar][kill][mask][cpu]") {
    const auto plan  = mask_plan(2);
    const auto input = waterfalls(plan, 1, 6);
    const auto cfg = mask_config(PeakFindingMode::kContiguousPeakAcrossWidths);
    auto search    = make_search(plan, cfg);
    const auto before = search.get_memory_usage();
    CHECK(before.staging == 0);
    const auto m = random_mask(32, 21, 0.3);
    (void)search.execute(std::span<const float>(input[0]),
                         std::span<const uint8_t>(m));
    const auto after = search.get_memory_usage();
    CHECK(after.staging == plan.get_nchans() * plan.get_nsamps() * 4);
    CHECK(after.plan > before.plan);
    CHECK(after.workspace > before.workspace);
    CHECK(after.total() <= algorithms::kFDMTDefaultMaxMemory);
    // Mid-stream reset: the next stream is a fresh one (no transition from
    // the old mask into its first block).
    (void)search.execute(std::span<const float>(input[1]),
                         std::span<const uint8_t>(random_mask(32, 22, 0.5)));
    search.reset_history();
    const Masks masks   = {random_mask(32, 23, 0.4),
                           random_mask(32, 24, 0.2),
                           {},
                           random_mask(32, 25, 0.4),
                           {},
                           {}};
    const auto expected = masked_reference(plan, cfg, true, input, {}, masks);
    compare(run(search, input, masks), expected.candidates);
    // Tables of masks no longer in the ring or cache are released.
    CHECK(search.get_memory_usage().total() <=
          after.total() + (8 * 1024 * 1024));
}

TEST_CASE("Boxcar noise model bounds the exact transition variance",
          "[boxcar][kill][mask][cpu]") {
    // Brute force on a small plan: impulse responses h[r, c, k] of every
    // searched row (unit impulses through the materialized FDMT), boxcar g =
    // h * box_W, and the exact variance of endpoint t when channel c's
    // sample t - k belongs to a block with mask m: sum m g^2. Steady
    // endpoints equal the model's variance; endpoints in a block's first
    // S_r + W - 1 lie between the AND and the OR masks' (the OR is used).
    using algorithms::detail::BoxcarNoiseModel;
    const plans::FDMTPlan plan(700, 1200, 8, 160, 0.001F,
                               std::vector<IndexType>{0, 5, 13, 30}, "valid",
                               1);
    const SizeType nchans = 8;
    const SizeType ns     = 160;
    auto cfg        = mask_config(PeakFindingMode::kContiguousPeakPerWidth, 9);
    cfg.noise_sigma = 1.0F;
    BoxcarNoiseModel model(plan, cfg, true, 1);
    const FDMTBoxcar meta(plan, cfg, true, Exec::cpu());
    const auto planes = meta.get_search_plans();
    const auto widths = planes.front().widths;
    // h[c][row][k]: response at lag k of each row to a unit impulse in c.
    const SizeType lags = 64;
    std::vector<std::vector<std::vector<double>>> h(nchans);
    for (SizeType c = 0; c < nchans; ++c) {
        std::vector<std::vector<float>> blocks(2,
                                               std::vector<float>(nchans * ns));
        blocks[0][(c * ns) + 50] = 1.0F;
        const auto rows =
            reference_rows(plan, true, 1, planes, blocks, Masks(2));
        for (const auto& row : rows) {
            std::vector<double> resp(lags, 0.0);
            for (SizeType k = 0; k < lags; ++k) {
                resp[k] = row[50 + k];
            }
            h[c].push_back(resp);
        }
    }
    const auto ma = random_mask(nchans, 5, 0.4);
    auto mb       = ma;
    mb[0]         = mb[0] == 0 ? 1 : 0;
    mb[5]         = mb[5] == 0 ? 1 : 0;
    std::vector<uint8_t> m_and(nchans), m_or(nchans);
    for (SizeType c = 0; c < nchans; ++c) {
        m_and[c] = (ma[c] != 0 && mb[c] != 0) ? 1 : 0;
        m_or[c]  = (ma[c] != 0 || mb[c] != 0) ? 1 : 0;
    }
    const auto ta  = model.compute(ma);
    const auto tb  = model.compute(mb);
    const auto tan = model.compute(m_and);
    const auto tor = model.compute(m_or);
    const auto var = [&](const auto& t, SizeType i) {
        const double inv = t->inv_sigma[i];
        return inv == 0 ? 0.0 : 1.0 / (inv * inv);
    };
    const auto& support = *model.support();
    SizeType checked    = 0;
    for (SizeType r = 0; r < support.size(); ++r) {
        for (SizeType f = 0; f < widths.size(); ++f) {
            const SizeType w = widths[f];
            // g[c][k] = sum_{j < w} h[c][r][k - j].
            std::vector<std::vector<double>> g(
                nchans, std::vector<double>(lags + w, 0.0));
            for (SizeType c = 0; c < nchans; ++c) {
                for (SizeType k = 0; k < lags + w; ++k) {
                    for (SizeType j = 0; j < w && j <= k; ++j) {
                        if (k - j < lags) {
                            g[c][k] += h[c][r][k - j];
                        }
                    }
                }
            }
            // Endpoint at offset u from the start of block b (mask mb),
            // previous block mask ma.
            const SizeType window = support[r] + w - 1;
            for (SizeType u = 0; u < window + 4; ++u) {
                double exact = 0;
                for (SizeType c = 0; c < nchans; ++c) {
                    for (SizeType k = 0; k < lags + w; ++k) {
                        const bool in_b = k <= u;
                        const auto& m   = in_b ? mb : ma;
                        exact += m[c] != 0 ? g[c][k] * g[c][k] : 0.0;
                    }
                }
                const SizeType i = (r * widths.size()) + f;
                INFO("row " << r << " width " << w << " offset " << u
                            << " exact " << exact);
                if (u >= window) {
                    REQUIRE_THAT(exact,
                                 Catch::Matchers::WithinRel(var(tb, i), 1e-5));
                } else {
                    REQUIRE(exact <= var(tor, i) * (1 + 1e-5));
                    REQUIRE(exact >= var(tan, i) * (1 - 1e-5));
                }
                ++checked;
            }
        }
    }
    REQUIRE(checked > 100);
    (void)ta;
}

TEST_CASE("FDMTBoxcar per-call mask: scores stay calibrated on white noise",
          "[boxcar][kill][mask][cpu]") {
    // Unselected signed scores of every endpoint (FDMT rows of white noise,
    // boxcar sums times the inverse sigma the transition rule picks): steady
    // endpoints have unit sigma, transition endpoints at most unit. 16
    // blocks of 400 samples; masks change every block.
    const auto plan   = mask_plan(1);
    const SizeType ns = plan.get_nsamps();
    const SizeType nb = 16;
    std::mt19937 gen(2026);
    std::normal_distribution<float> noise;
    std::vector<std::vector<float>> input(nb, std::vector<float>(32 * ns));
    for (auto& b : input) {
        for (auto& v : b) {
            v = noise(gen);
        }
    }
    Masks masks;
    for (SizeType k = 0; k < nb; ++k) {
        masks.push_back(random_mask(32, 300 + static_cast<unsigned>(k), 0.35));
    }
    const auto cfg = mask_config(PeakFindingMode::kAllAboveThreshold, 9);
    const FDMTBoxcar meta(plan, cfg, true, Exec::cpu());
    const auto planes = meta.get_search_plans();
    const auto widths = planes.front().widths;
    algorithms::detail::BoxcarNoiseModel model(plan, cfg, true, 2);
    std::vector<std::shared_ptr<const algorithms::detail::NoiseTables>> steady,
        trans;
    for (SizeType k = 0; k < nb; ++k) {
        steady.push_back(model.compute(masks[k]));
        std::vector<uint8_t> u(32);
        for (SizeType c = 0; c < 32; ++c) {
            u[c] =
                (masks[k][c] != 0 || (k > 0 && masks[k - 1][c] != 0)) ? 1 : 0;
        }
        trans.push_back(model.compute(u));
    }
    const auto rows     = reference_rows(plan, true, 1, planes, input, masks);
    const auto& support = *model.support();
    double ss[2]        = {0, 0};
    SizeType cnt[2]     = {0, 0};
    for (SizeType r = 0; r < rows.size(); ++r) {
        for (SizeType f = 0; f < widths.size(); ++f) {
            const SizeType w = widths[f];
            for (SizeType t = ns; t < nb * ns; ++t) {
                const SizeType k = t / ns;
                double sum       = 0;
                for (SizeType j = 0; j < w; ++j) {
                    sum += rows[r][t - j];
                }
                const bool tr    = t < (k * ns) + support[r] + w - 1;
                const auto& tab  = tr ? trans[k] : steady[k];
                const double inv = tab->inv_sigma[(r * widths.size()) + f];
                if (inv == 0) {
                    continue;
                }
                ss[tr ? 1 : 0] += (sum * inv) * (sum * inv);
                ++cnt[tr ? 1 : 0];
            }
        }
    }
    const double sigma_steady = std::sqrt(ss[0] / static_cast<double>(cnt[0]));
    const double sigma_trans  = std::sqrt(ss[1] / static_cast<double>(cnt[1]));
    INFO("steady " << sigma_steady << " over " << cnt[0] << ", transition "
                   << sigma_trans << " over " << cnt[1]);
    // Correlated samples: a few per cent of statistical spread.
    CHECK(std::abs(sigma_steady - 1.0) < 0.04);
    CHECK(sigma_trans < 1.0);
    CHECK(sigma_trans > 0.7);
}
