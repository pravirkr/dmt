#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_variance.hpp"

#include "fdmt_boxcar_reference.hpp"
#include "test_random.hpp"

using namespace boxcar_test;

TEST_CASE("Boxcar width generation and validation", "[boxcar][cpu]") {
    // Geometric widths, then max_width itself when the steps stop below it.
    CHECK(plans::generate_box_width_trials(9, 2) ==
          std::vector<uint32_t>{1, 2, 4, 8, 9});
    CHECK(plans::generate_box_width_trials(9, 1.5) ==
          std::vector<uint32_t>{1, 2, 3, 5, 8, 9});
    CHECK(plans::generate_box_width_trials(8, 2) ==
          std::vector<uint32_t>{1, 2, 4, 8});
    CHECK(plans::generate_box_width_trials(1, 2) == std::vector<uint32_t>{1});
    CHECK(plans::generate_box_width_trials(219, 2).back() == 219);
    CHECK(plans::generate_box_width_trials(4096, 1).size() == 4096);
    for (const float invalid :
         {0.0F, 0.9F, std::bit_cast<float>(uint32_t{0x7F800000U}),
          std::bit_cast<float>(uint32_t{0x7FC00000U})}) {
        CHECK_THROWS_AS(plans::generate_box_width_trials(32, invalid),
                        std::invalid_argument);
    }
    CHECK_THROWS_AS(plans::generate_box_width_trials(0, 2),
                    std::invalid_argument);
    CHECK(plans::generate_box_width_trials(4097, 2).back() == 4097);
    CHECK(plans::generate_box_width_trials(1U << 20, 2).back() == 1U << 20);
    CHECK_THROWS_AS(plans::generate_box_width_trials((1U << 20) + 1, 2),
                    std::invalid_argument);
}

TEST_CASE("FDMTBoxcar CPU searches boxes wider than 4096",
          "[boxcar][cpu]") {
    // Wider than a block: the carry spans blocks.
    const plans::FDMTPlan plan(700, 1200, 7, 2048, 0.001F,
                               std::vector<IndexType>{-5, 0, 7, 40}, "valid",
                               1);
    const auto input = waterfalls(plan, 1, 4);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 6000;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 3;
    cfg.max_cands_per_dm = 0;
    cfg.max_candidates   = 1U << 20;
    const auto expected  = reference(plan, cfg, true, 1, input);
    REQUIRE(expected.size() > 100);
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(2));
    std::vector<RawCandidate> actual;
    for (const auto& block : input) {
        append(actual, search.execute(block));
    }
    append(actual, search.flush());
    compare(actual, expected);
}

TEST_CASE("FDMTBoxcar matches materialized FDMT and direct windows",
          "[boxcar][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 7, 311, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    const auto input = waterfalls(plan, 2, 4);
    for (const bool smear : {true, false}) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            DYNAMIC_SECTION("smear=" << smear
                                     << " mode=" << static_cast<int>(mode)) {
                BoxcarConfig cfg;
                cfg.max_boxcar       = 37;
                cfg.wtsp             = 1.5F;
                cfg.detect_thresh    = 2.7F;
                cfg.peak_mode        = mode;
                cfg.max_candidates   = 500000;
                cfg.max_cands_per_dm = 0;
                const auto expected =
                    batched_reference(plan, cfg, smear, 2, input);
                REQUIRE_FALSE(expected.empty());
                for (const SizeType fuse :
                     {SizeType{0}, algorithms::kFDMTAutoFuse}) {
                    for (const SizeType chunk :
                         {SizeType{64}, plan.get_nsamps()}) {
                        FDMTBoxcar search(plan, cfg, smear, Exec::cpu(3), 2,
                                          fuse, chunk);
                        std::vector<RawCandidate> actual;
                        for (SizeType k = 0; k < input.size(); ++k) {
                            const auto result = search.execute(input[k]);
                            CHECK(result.input_sample_begin ==
                                  k * plan.get_nsamps());
                            append(actual, result);
                        }
                        append(actual, search.flush());
                        compare(actual, expected);
                        CHECK(search.flush().candidates.empty());
                        search.reset_history();
                        std::vector<RawCandidate> repeated;
                        for (const auto& block : input) {
                            append(repeated, search.execute(block));
                        }
                        append(repeated, search.flush());
                        compare(repeated, expected);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTBoxcar sigma uses raw sum and actual subband lineage",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 5, 128, 0.001F,
                         std::vector<IndexType>{-90, 0, 7, 100}, "valid", 2);
    for (const bool smear : {true, false}) {
        BoxcarConfig cfg;
        cfg.max_boxcar  = 32;
        cfg.noise_sigma = 2.5F;
        FDMTBoxcar search(plan, cfg, smear);
        const auto& root = search.get_search_plans().back();
        for (SizeType f = 0; f < root.widths.size(); ++f) {
            const auto sigma =
                plan.get_effective_sigma_grid(root.widths[f], smear);
            for (SizeType d = 0; d < sigma.size(); ++d) {
                CHECK_THAT(1.0F /
                               root.inv_sigma_lut[d * root.widths.size() + f],
                           Catch::Matchers::WithinRel(2.5F * sigma[d], 2e-6F));
            }
        }
        // Impulse energy of the actual FDMT subband response. Boxcar sums
        // are evaluated independently; each input channel is independent.
        const SizeType samples = 512;
        plans::FDMTPlan impulse_plan(700, 1200, 5, samples, 0.001F,
                                     plan.get_dt_grid_final(), "valid", 2);
        FDMTBoxcar impulse_meta(impulse_plan, cfg, smear);
        const auto planes = impulse_meta.get_search_plans();
        std::vector<std::vector<double>> energy;
        std::vector<std::vector<SizeType>> last_support;
        for (const auto& p : planes) {
            energy.emplace_back(p.dm_grid.size() * p.widths.size(), 0.0);
            last_support.emplace_back(p.dm_grid.size(), 0);
        }
        for (SizeType channel = 0; channel < 5; ++channel) {
            FDMT fdmt(impulse_plan, smear, {}, 1, 0, false, samples);
            std::vector<float> input(5 * samples);
            input[channel * samples + 128] = 1;
            std::vector<float> output(impulse_plan.get_dmt_size());
            fdmt.reset(input, output);
            for (SizeType p = 0; p < planes.size(); ++p) {
                fdmt.advance_until_remaining(planes[p].subband_level);
                const auto data = fdmt.view_subband(planes[p].subband_idx).data;
                for (SizeType d = 0; d < planes[p].dm_grid.size(); ++d) {
                    for (SizeType t = 128; t < samples; ++t) {
                        if (data[d * samples + t] != 0) {
                            last_support[p][d] =
                                std::max(last_support[p][d], t - 128);
                        }
                    }
                    for (SizeType f = 0; f < planes[p].widths.size(); ++f) {
                        for (SizeType t = 0; t < samples; ++t) {
                            double sum = 0;
                            for (SizeType k = 0;
                                 k < planes[p].widths[f] && k <= t; ++k) {
                                sum += data[d * samples + t - k];
                            }
                            energy[p][d * planes[p].widths.size() + f] +=
                                sum * sum;
                        }
                    }
                }
            }
            fdmt.finalize();
        }
        for (SizeType p = 0; p < planes.size(); ++p) {
            for (SizeType d = 0; d < planes[p].dm_grid.size(); ++d) {
                CHECK(planes[p].warmup_samples[d] == last_support[p][d]);
            }
            for (SizeType i = 0; i < energy[p].size(); ++i) {
                const double sigma = 1.0 / planes[p].inv_sigma_lut[i];
                CHECK_THAT(sigma * sigma, Catch::Matchers::WithinRel(
                                              energy[p][i] * 6.25, 3e-6));
            }
        }
    }
}

TEST_CASE("FDMTBoxcar validates contracts and bounds overflow",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 2, 16, 0.001F, 1);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 4;
    cfg.detect_thresh    = 1;
    cfg.skip_cold_prefix = false;
    cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
    cfg.max_candidates   = 3;
    cfg.max_cands_per_dm = 5;
    FDMTBoxcar search(plan, cfg);
    std::vector<float> input(32, 10);
    const auto result = search.execute(input);
    CHECK(result.emitted_count == 3);
    CHECK(result.total_detected == 96);
    CHECK(result.dropped_per_dm == 86);
    CHECK(result.dropped_global == 7);
    CHECK(result.overflowed());
    CHECK(search.copy_result().total_detected == 96);
    CHECK_THROWS_AS(search.execute(std::span<const float>{}),
                    std::invalid_argument);
    CHECK_THROWS_AS(search.execute(DeviceSpan<const float>{}),
                    std::invalid_argument);
    CHECK_THROWS_AS(search.flush_device(), std::invalid_argument);
    CHECK_THROWS_AS(search.reset_history(Stream{reinterpret_cast<void*>(1)}),
                    std::invalid_argument);
    // Two channels have one merge: no level below the root to search.
    CHECK_THROWS_AS(
        plans::FDMTPlan(700, 1200, 2, 16, 0.001F, 1, 0, 1, "valid", 1),
        std::invalid_argument);
    cfg.noise_sigma = std::bit_cast<float>(uint32_t{0x7FC00000U});
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    plans::FDMTPlan full(700, 1200, 2, 16, 0.001F, 1, 0, 1, "full");
    CHECK_THROWS_AS(FDMTBoxcar(full), std::invalid_argument);
}

TEST_CASE("FDMTBoxcar carries wide windows and peaks across small blocks",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 2, 17, 0.001F, 80);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 4096;
    cfg.wtsp             = 2;
    cfg.detect_thresh    = 3;
    cfg.peak_mode        = PeakFindingMode::kContiguousPeakAcrossWidths;
    cfg.max_cands_per_dm = 0;
    std::vector<std::vector<float>> input(250, std::vector<float>(34, 1.0F));
    for (SizeType k = 246; k < input.size(); ++k) {
        std::ranges::fill(input[k], 0.0F);
    }
    FDMTBoxcar search(plan, cfg);
    std::vector<RawCandidate> actual;
    for (const auto& block : input) {
        append(actual, search.execute(block));
    }
    append(actual, search.flush());
    compare(actual, reference(plan, cfg, true, 1, input));
    REQUIRE_FALSE(actual.empty());
    CHECK(search.flush().total_detected == 0);
}

TEST_CASE("Boxcar run summaries compose across worker and tile boundaries",
          "[boxcar][cpu]") {
    using namespace algorithms::detail;
    std::mt19937 random(39);
    for (int trial = 0; trial < 500; ++trial) {
        std::vector<bool> above(257);
        std::vector<float> snr(257);
        for (SizeType i = 0; i < above.size(); ++i) {
            above[i] = random() % 5 != 0;
            snr[i]   = static_cast<float>(random() % 5); // Deliberate ties.
        }
        BoxcarPeak carried;
        std::vector<BoxcarPeak> expected;
        for (SizeType i = 0; i < above.size(); ++i) {
            boxcar_peak_step(
                carried, above[i], snr[i], i, static_cast<uint32_t>(i % 3),
                [&](const BoxcarPeak& peak) { expected.push_back(peak); });
        }
        if (carried.open) {
            expected.push_back(carried);
        }
        std::vector<BoxcarRunSummary> summaries;
        for (SizeType first = 0; first < above.size(); first += 4) {
            BoxcarRunSummary s;
            for (SizeType i = first; i < std::min(first + 4, above.size());
                 ++i) {
                if (!above[i]) {
                    s.all           = false;
                    s.trailing.open = false;
                } else {
                    boxcar_peak_step(s.trailing, true, snr[i], i,
                                     static_cast<uint32_t>(i % 3),
                                     [](const BoxcarPeak&) {});
                }
            }
            summaries.push_back(s);
        }
        BoxcarRunSummary prefix;
        std::vector<BoxcarPeak> actual;
        for (SizeType worker = 0; worker < summaries.size(); ++worker) {
            BoxcarPeak incoming = prefix.trailing;
            for (SizeType i = worker * 4;
                 i < std::min(worker * 4 + 4, above.size()); ++i) {
                boxcar_peak_step(
                    incoming, above[i], snr[i], i, static_cast<uint32_t>(i % 3),
                    [&](const BoxcarPeak& peak) { actual.push_back(peak); });
            }
            prefix = boxcar_join(prefix, summaries[worker]);
        }
        if (prefix.trailing.open) {
            actual.push_back(prefix.trailing);
        }
        REQUIRE(actual.size() == expected.size());
        for (SizeType i = 0; i < actual.size(); ++i) {
            CHECK(actual[i].begin == expected[i].begin);
            CHECK(actual[i].end == expected[i].end);
            CHECK(actual[i].peak == expected[i].peak);
            CHECK(actual[i].snr == expected[i].snr);
            CHECK(actual[i].filter == expected[i].filter);
        }
        // Different association order must give the identical trailing run.
        auto right = BoxcarRunSummary{};
        for (SizeType i = summaries.size(); i > 0; --i) {
            right = boxcar_join(summaries[i - 1], right);
        }
        CHECK(right.trailing.open == prefix.trailing.open);
        CHECK(right.trailing.peak == prefix.trailing.peak);
        CHECK(right.trailing.begin == prefix.trailing.begin);
    }
}

TEST_CASE("FDMTBoxcar scores whole blocks over multiple prefix tiles",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 3, 2203, 0.001F,
                         std::vector<IndexType>{-7, 0, 9}, "valid", 1);
    const auto input = waterfalls(plan, 1, 3);
    for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                            PeakFindingMode::kContiguousPeakAcrossWidths,
                            PeakFindingMode::kAllAboveThreshold}) {
        BoxcarConfig cfg;
        cfg.max_boxcar       = 64;
        cfg.detect_thresh    = 2.7F;
        cfg.peak_mode        = mode;
        cfg.max_cands_per_dm = 0;
        FDMTBoxcar search(plan, cfg, true, Exec::cpu(2), 1, 0,
                          plan.get_nsamps());
        std::vector<RawCandidate> actual;
        for (const auto& block : input) {
            append(actual, search.execute(block));
        }
        append(actual, search.flush());
        compare(actual, reference(plan, cfg, true, 1, input));
    }
}

TEST_CASE("FDMTBoxcar analytical scaling predicts Gaussian noise",
          "[boxcar][cpu]") {
    const SizeType samples = 32768;
    plans::FDMTPlan plan(700, 1200, 4, samples, 0.001F,
                         std::vector<IndexType>{-20, 0, 19}, "valid", 1);
    BoxcarConfig cfg;
    cfg.max_boxcar = 7;
    FDMTBoxcar search(plan, cfg);
    FDMT fdmt(plan, true, {}, 1, 0, false, samples);
    const auto input = dmt::test_random::normal<float>(4 * samples, 32);
    std::vector<float> output(plan.get_dmt_size());
    fdmt.reset(input, output);
    for (const auto& p : search.get_search_plans()) {
        fdmt.advance_until_remaining(p.subband_level);
        const auto values = fdmt.view_subband(p.subband_idx).data;
        for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
            for (SizeType f = 0; f < p.widths.size(); ++f) {
                const SizeType first = p.warmup_samples[d] + p.widths[f] - 1;
                double sum = 0, square = 0;
                const SizeType count = samples - first;
                for (SizeType t = first; t < samples; ++t) {
                    double value = 0;
                    for (SizeType k = 0; k < p.widths[f]; ++k) {
                        value += values[d * samples + t - k];
                    }
                    value *= p.inv_sigma_lut[d * p.widths.size() + f];
                    sum += value;
                    square += value * value;
                }
                const double mean = sum / count;
                CHECK(std::abs(mean) < 0.08);
                CHECK(std::abs(square / count - mean * mean - 1.0) < 0.08);
            }
        }
    }
    fdmt.finalize();
}

TEST_CASE("FDMTBoxcar physical DM metadata and configuration are preserved",
          "[boxcar][cpu]") {
    const std::vector<float> dm{-6.25F, 0.0F, 8.125F};
    plans::FDMTPlan plan(700, 1200, 4, 128, 0.001F, dm);
    BoxcarConfig cfg;
    cfg.max_boxcar = 17;
    FDMTBoxcar search(plan, cfg);
    CHECK(search.get_search_plans().back().dm_grid == plan.get_dm_grid_final());
    CHECK(search.get_config().max_boxcar == 17);
    CHECK(search.backend() == Backend::kCPU);
    CHECK(search.device() == -1);
    auto moved = std::move(search);
    CHECK(moved.get_plan().get_nchans() == 4);
    CHECK(moved.get_memory_usage().total() > 0);
}

TEST_CASE("FDMTBoxcar detects a band-limited pulse in its matching subband",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 8, 64, 0.001F, 4, 0, 1, "valid", 2);
    BoxcarConfig cfg;
    cfg.max_boxcar    = 1;
    cfg.detect_thresh = 3;
    FDMTBoxcar search(plan, cfg);
    std::vector<float> input(8 * 64);
    input[32] = input[64 + 32] = 3;
    const auto result          = search.execute(input);
    bool quarter               = false;
    for (const auto& candidate : result.candidates) {
        if (candidate.dm_idx != 0) {
            continue;
        }
        CHECK(candidate.subband_level != 0);
        if (candidate.subband_level == 2 && candidate.subband_idx == 0) {
            quarter = true;
            CHECK(candidate.chan_begin == 0);
            CHECK(candidate.chan_end == 2);
            CHECK(candidate.peak_sample == 32);
            CHECK_THAT(candidate.snr, Catch::Matchers::WithinRel(
                                          6.0F / std::sqrt(2.0F), 1e-6F));
        }
    }
    CHECK(quarter);

    // Channels 3 and 4 straddle the quarter (and half) boundary at 4: only
    // the staggered quarter [3, 5) holds the whole pulse.
    std::vector<float> straddle(8 * 64);
    straddle[(3 * 64) + 32] = straddle[(4 * 64) + 32] = 3;
    search.reset_history();
    const auto across        = search.execute(straddle);
    const RawCandidate* best = nullptr;
    for (const auto& candidate : across.candidates) {
        if (candidate.dm_idx == 0 &&
            (best == nullptr || candidate.snr > best->snr)) {
            best = &candidate;
        }
    }
    REQUIRE(best != nullptr);
    CHECK(best->staggered == 1);
    CHECK(best->subband_level == 2);
    CHECK(best->subband_idx == 5);
    CHECK(best->chan_begin == 3);
    CHECK(best->chan_end == 5);
    CHECK(best->peak_sample == 32);
    CHECK_THAT(best->snr,
               Catch::Matchers::WithinRel(6.0F / std::sqrt(2.0F), 1e-6F));

    // Seven channels: quarters [0,2) [2,4) [4,6) [6,7), staggered quarters
    // [1,3) [3,5) [5,7); halves [0,4) [4,7), staggered half [2,6); root.
    plans::FDMTPlan odd(700, 1200, 7, 64, 0.001F, 4, 0, 1, "valid", 2);
    FDMTBoxcar odd_search(odd, cfg);
    const auto planes = odd_search.get_search_plans();
    REQUIRE(planes.size() == 11);
    const std::vector<std::array<uint32_t, 3>> extents{
        {0, 2, 0}, {2, 4, 0}, {4, 6, 0}, {6, 7, 0}, {1, 3, 1}, {3, 5, 1},
        {5, 7, 1}, {0, 4, 0}, {4, 7, 0}, {2, 6, 1}, {0, 7, 0}};
    for (SizeType p = 0; p < planes.size(); ++p) {
        INFO("plane " << p);
        CHECK(planes[p].chan_begin == extents[p][0]);
        CHECK(planes[p].chan_end == extents[p][1]);
        CHECK(planes[p].staggered == (extents[p][2] == 1));
    }
    CHECK(planes[3].f_high_mhz == odd.get_f_max());
}

TEST_CASE("FDMTBoxcar root reader agrees for CPU rings and GPU FIFOs",
          "[boxcar][cpu]") {
    using namespace algorithms::detail;
    for (const SizeType delay : {SizeType{0}, SizeType{1}, SizeType{9}}) {
        for (const SizeType n : {SizeType{3}, SizeType{17}}) {
            for (const uint64_t position :
                 {uint64_t{0}, uint64_t{4}, uint64_t{23}}) {
                std::vector<float> state(2 * n), fifo(delay), ring(delay);
                for (SizeType t = 0; t < n; ++t) {
                    state[t]     = static_cast<float>(position + t + 101);
                    state[n + t] = static_cast<float>(position + t + 1);
                }
                for (SizeType k = 0; k < delay; ++k) {
                    const int64_t time = static_cast<int64_t>(position + k) -
                                         static_cast<int64_t>(delay);
                    fifo[k] = time < 0 ? 0.0F : static_cast<float>(time + 1);
                    ring[(position + k) % delay] = fifo[k];
                }
                const BoxcarRow row{.tail = 0, .head = 1, .delay = delay};
                FDMTSearchInput input{.state        = state.data(),
                                      .history      = fifo.data(),
                                      .row_stride   = n,
                                      .beam_stride  = 2 * n,
                                      .nsamps       = n,
                                      .sample_begin = position};
                for (SizeType t = 0; t < n; ++t) {
                    const float device_value =
                        boxcar_sample(input, row, 0, t, true);
                    input.history      = ring.data();
                    input.history_ring = true;
                    CHECK(boxcar_sample(input, row, 0, t, true) ==
                          device_value);
                    input.history      = fifo.data();
                    input.history_ring = false;
                    const int64_t time = static_cast<int64_t>(position + t) -
                                         static_cast<int64_t>(delay);
                    CHECK(device_value ==
                          state[t] +
                              (time < 0 ? 0.0F : static_cast<float>(time + 1)));
                }
            }
        }
    }
}

TEST_CASE("Packed FDMTBoxcar agrees with centered float reference",
          "[boxcar][packed][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 7, 139, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    constexpr SizeType beams  = 2;
    constexpr SizeType blocks = 4;
    for (const SizeType bits : {1, 2, 4, 8, 16}) {
        const uint32_t maximum = (1U << bits) - 1;
        const float mean       = maximum * 0.5F;
        const float sigma =
            std::sqrt(static_cast<float>(maximum) * (maximum + 2.0F) / 12.0F);
        const SizeType row_bytes = (plan.get_nsamps() * bits + 7) / 8;
        std::vector<std::vector<uint8_t>> packed(
            blocks,
            std::vector<uint8_t>(beams * plan.get_nchans() * row_bytes));
        std::vector<std::vector<float>> centered(
            blocks,
            std::vector<float>(beams * plan.get_nchans() * plan.get_nsamps()));
        std::mt19937 random(121);
        for (SizeType k = 0; k < blocks; ++k) {
            for (SizeType row = 0; row < beams * plan.get_nchans(); ++row) {
                for (SizeType t = 0; t < plan.get_nsamps(); ++t) {
                    const auto value = random() & maximum;
                    centered[k][row * plan.get_nsamps() + t] = value - mean;
                    for (SizeType bit = 0; bit < bits; ++bit) {
                        const SizeType offset = t * bits + bit;
                        packed[k][row * row_bytes + offset / 8] |=
                            ((value >> bit) & 1U) << (offset % 8);
                    }
                }
            }
        }
        for (const bool smear : {true, false}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                DYNAMIC_SECTION("bits=" << bits << " smear=" << smear
                                        << " mode=" << static_cast<int>(mode)) {
                    BoxcarConfig cfg;
                    cfg.max_boxcar       = 37;
                    cfg.wtsp             = 1.5F;
                    cfg.detect_thresh    = 2.713F;
                    cfg.noise_sigma      = sigma;
                    cfg.peak_mode        = mode;
                    cfg.max_candidates   = 200000;
                    cfg.max_cands_per_dm = 0;
                    const auto expected =
                        batched_reference(plan, cfg, smear, beams, centered);
                    REQUIRE_FALSE(expected.empty());
                    cfg.input_nbits = bits;
                    cfg.input_mean  = mean;
                    for (const SizeType chunk :
                         {SizeType{64}, plan.get_nsamps()}) {
                        FDMTBoxcar search(plan, cfg, smear, Exec::cpu(3), beams,
                                          algorithms::kFDMTAutoFuse, chunk);
                        std::vector<RawCandidate> actual;
                        for (const auto& input : packed) {
                            append(actual, search.execute(input));
                        }
                        append(actual, search.flush());
                        compare(actual, expected);
                        search.reset_history();
                        std::vector<RawCandidate> repeated;
                        for (const auto& input : packed) {
                            append(repeated, search.execute(input));
                        }
                        append(repeated, search.flush());
                        compare(repeated, actual);
                    }
                }
            }
        }
    }
}

TEST_CASE("Packed FDMTBoxcar mean and input validation",
          "[boxcar][packed][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 7, 17, 0.001F, 97, 0, 1, "valid", 2);
    BoxcarConfig cfg;
    cfg.input_nbits = 8;
    cfg.input_mean  = 127;
    cfg.noise_sigma = 3;
    cfg.max_boxcar  = 4096;
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(2));
    std::vector<uint8_t> input(plan.get_nchans() * plan.get_nsamps(), 127);
    for (int i = 0; i < 260; ++i) {
        CHECK(search.execute(input).total_detected == 0);
    }
    CHECK(search.flush().total_detected == 0);
    CHECK_THROWS_AS(search.execute(std::vector<float>(input.size())),
                    std::invalid_argument);
    CHECK_THROWS_AS(search.execute(std::span<const uint8_t>(input.data(),
                                                            input.size() - 1)),
                    std::invalid_argument);
    for (const auto bits : {0, 3, 7, 31, 64}) {
        cfg.input_nbits = bits;
        CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    }
    cfg.input_nbits = 8;
    for (const float invalid :
         {-1.0F, 256.0F, std::bit_cast<float>(0x7F800000U),
          std::bit_cast<float>(0x7FC00000U)}) {
        cfg.input_mean = invalid;
        CHECK_THROWS_AS(FDMTBoxcar(plan, cfg), std::invalid_argument);
    }
}

TEST_CASE("Packed FDMTBoxcar wide windows exceed uint32 safely",
          "[boxcar][packed][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 7, 257, 0.001F,
                               std::vector<IndexType>{0, 1});
    const BoxcarConfig cfg(4096, 2.0F, 0.1F, 1.0F,
                           PeakFindingMode::kAllAboveThreshold, 10000, 0, true,
                           16, 65534.0F);
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(2), 1, 0, 64);
    std::vector<uint8_t> input(7 * 257 * 2, 255);
    bool found = false;
    for (int k = 0; k < 17; ++k) {
        const auto result = search.execute(input);
        REQUIRE_FALSE(result.overflowed());
        for (const auto& c : result.candidates) {
            if (c.width == 4096 && c.dm_idx == 0) {
                CHECK(c.peak_sample >= 4095);
                CHECK_THAT(c.snr, Catch::Matchers::WithinRel(
                                      std::sqrt(7.0F * 4096), 1e-6F));
                found = true;
            }
        }
    }
    CHECK(found);
    // This direct helper check also protects low-bit peak ordering from
    // accidentally going through float before candidate emission.
    algorithms::detail::BoxcarPeak peak;
    const auto ignore = [](const auto&) {};
    algorithms::detail::boxcar_integer_peak_step(
        peak, true, (uint64_t{1} << 54), 1, 0, ignore);
    algorithms::detail::boxcar_integer_peak_step(
        peak, true, (uint64_t{1} << 54) + 1, 2, 0, ignore);
    CHECK(peak.peak == 2);
}

TEST_CASE("Integer boxcar run summaries preserve exact peaks",
          "[boxcar][packed][cpu]") {
    using namespace algorithms::detail;
    const std::vector<uint64_t> sums{0,
                                     (uint64_t{1} << 54),
                                     (uint64_t{1} << 54) + 1,
                                     0,
                                     0,
                                     (uint64_t{1} << 54) + 2,
                                     (uint64_t{1} << 54) + 3,
                                     (uint64_t{1} << 54) + 3};
    const auto segment = [&](SizeType begin, SizeType end) {
        BoxcarRunSummary result;
        for (SizeType t = begin; t < end; ++t) {
            if (sums[t] == 0) {
                result.all           = false;
                result.trailing.open = false;
            } else {
                boxcar_integer_peak_step(result.trailing, true, sums[t], t, 0,
                                         [](const auto&) {});
            }
        }
        return result;
    };
    const auto expected = segment(0, sums.size());
    for (SizeType i = 0; i <= sums.size(); ++i) {
        for (SizeType j = i; j <= sums.size(); ++j) {
            const auto a = segment(0, i), b = segment(i, j),
                       c = segment(j, sums.size());
            for (const auto result :
                 {boxcar_join(boxcar_join(a, b, true), c, true),
                  boxcar_join(a, boxcar_join(b, c, true), true)}) {
                CHECK(result.all == expected.all);
                CHECK(result.trailing.begin == expected.trailing.begin);
                CHECK(result.trailing.end == expected.trailing.end);
                CHECK(result.trailing.peak == expected.trailing.peak);
                CHECK(result.trailing.raw_sum == expected.trailing.raw_sum);
            }
        }
    }
}

TEST_CASE("Trapezoidal FDMTBoxcar variance, sensitivity, and CPU/GPU parity",
          "[boxcar][trapezoid][cpu]") {
    using namespace algorithms::detail;
    // 1. Verify exact analytical variance against explicit discrete
    // convolution.
    for (SizeType len :
         {SizeType{1}, SizeType{2}, SizeType{3}, SizeType{5}, SizeType{8}}) {
        for (uint32_t width : {1U, 2U, 3U, 4U, 5U, 8U, 12U, 16U, 32U}) {
            const uint32_t w_edge = trapezoid_edge_width(width);
            std::vector<double> filt(width, 1.0);
            for (uint32_t k = w_edge; k < width - w_edge; ++k) {
                filt[k] = w_edge > 0 ? 2.0 : 1.0;
            }
            const SizeType conv_len = len + width - 1;
            double exact_var        = 0.0;
            for (SizeType n = 0; n < conv_len; ++n) {
                double c = 0.0;
                for (SizeType i = 0; i < len; ++i) {
                    const SizeType j = n - i;
                    if (n >= i && j < width) {
                        c += 1.25 * filt[j];
                    }
                }
                exact_var += c * c;
            }
            CHECK_THAT(fdmt_trapezoid_variance(len, width, w_edge, 1.25),
                       Catch::Matchers::WithinRel(exact_var, 1e-12));
        }
    }
    // 2. Verify off-phase pulse sensitivity gain for [0.5, 1.0, 0.5] and
    // [0.5, 1.0, 1.0, 0.5] profiles at DM=0.
    const plans::FDMTPlan plan(700, 1200, 16, 128, 0.001F,
                               std::vector<IndexType>{0, 4});
    std::vector<float> pulse_w3(16 * 128, 0.0F);
    for (SizeType c = 0; c < 16; ++c) {
        pulse_w3[c * 128 + 40] = 0.5F;
        pulse_w3[c * 128 + 41] = 1.0F;
        pulse_w3[c * 128 + 42] = 0.5F;
    }
    BoxcarConfig cfg_rect(8, 1.5F, 1.0F, 1.0F);
    BoxcarConfig cfg_trap  = cfg_rect;
    cfg_trap.use_trapezoid = true;
    FDMTBoxcar search_rect(plan, cfg_rect, true, Exec::cpu(1));
    FDMTBoxcar search_trap(plan, cfg_trap, true, Exec::cpu(1));
    const auto res_rect = search_rect.execute(pulse_w3);
    const auto res_trap = search_trap.execute(pulse_w3);
    float max_snr_rect  = 0.0F;
    float max_snr_trap  = 0.0F;
    for (const auto& c : res_rect.candidates) {
        if (c.dm_idx == 0) {
            max_snr_rect = std::max(max_snr_rect, c.snr);
        }
    }
    for (const auto& c : res_trap.candidates) {
        if (c.dm_idx == 0) {
            max_snr_trap = std::max(max_snr_trap, c.snr);
        }
    }
    // Optimal SNR is 4 * sqrt(1.5) = 4.898979; rectangular boxcar W=3 gets
    // 4 * 2 / sqrt(3) = 4.618802 (+6.06% recovery).
    CHECK_THAT(max_snr_trap,
               Catch::Matchers::WithinRel(4.0F * std::sqrt(1.5F), 1e-5F));
    CHECK(max_snr_trap > max_snr_rect * 1.05F);
#ifdef DMT_ENABLE_GPU
    // 3. Verify CPU and GPU agree on multi-chunk float32 and packed 8-bit
    // (integer thresholds use the filter weight 2 (W - w_edge); exact).
    const auto wf = waterfalls(plan, 1, 3);
    std::vector<std::vector<uint8_t>> wf8;
    for (const auto& block : wf) {
        auto& q = wf8.emplace_back(block.size());
        for (SizeType i = 0; i < block.size(); ++i) {
            q[i] = static_cast<uint8_t>(std::clamp(
                std::nearbyint(128.0F + (16.0F * block[i])), 0.0F, 255.0F));
        }
    }
    for (const SizeType bits : {SizeType{32}, SizeType{8}}) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            BoxcarConfig cfg(16, 1.5F, 2.5F, 1.0F, mode);
            cfg.use_trapezoid = true;
            if (bits == 8) {
                cfg.input_nbits = 8;
                cfg.input_mean  = 128.0F;
                cfg.noise_sigma = 16.0F;
            }
            FDMTBoxcar cpu_s(plan, cfg, true, Exec::cpu(2), 1, 0, 64);
            FDMTBoxcar gpu_s(plan, cfg, true,
                             Exec{algorithms::detail::kGPUBackend, 0}, 1, 0,
                             64);
            std::vector<RawCandidate> c;
            std::vector<RawCandidate> g;
            for (SizeType k = 0; k < wf.size(); ++k) {
                if (bits == 8) {
                    append(g, gpu_s.execute(wf8[k]));
                    append(c, cpu_s.execute(wf8[k]));
                } else {
                    append(g, gpu_s.execute(wf[k]));
                    append(c, cpu_s.execute(wf[k]));
                }
            }
            append(g, gpu_s.flush());
            append(c, cpu_s.flush());
            REQUIRE_FALSE(c.empty());
            compare(g, c);
        }
    }
#endif
}

TEST_CASE("Boxcar width screen groups cover every width", "[boxcar][cpu]") {
    using algorithms::detail::boxcar_screen_groups;
    // Spacing 1.5 merges too few widths to pay: no groups.
    CHECK(
        boxcar_screen_groups(plans::generate_box_width_trials(163, 1.5F), false)
            .empty());
    for (const SizeType max_width : {SizeType{48}, SizeType{163}}) {
        const auto widths = plans::generate_box_width_trials(max_width, 1.0F);
        const auto groups = boxcar_screen_groups(widths, false);
        REQUIRE_FALSE(groups.empty());
        std::vector<int> covered(widths.size(), 0);
        uint32_t level = 0;
        for (const auto& g : groups) {
            CHECK(g.level >= level); // ascending: one table raised in place
            level = g.level;
            CHECK(g.w_lo == widths[g.f_lo]);
            CHECK(g.w_hi == widths[g.f_hi - 1]);
            CHECK(g.w_hi < 2 * g.w_lo);
            CHECK((1U << g.level) <= g.w_hi - g.w_lo + 1);
            CHECK((2U << g.level) > g.w_hi - g.w_lo + 1);
            for (uint32_t f = g.f_lo; f < g.f_hi; ++f) {
                ++covered[f];
            }
        }
        CHECK(std::ranges::all_of(covered, [](int c) { return c == 1; }));
        // Trapezoid widths (>= 3) stay single, so grouping does not pay.
        CHECK(boxcar_screen_groups(widths, true).empty());
    }
}

TEST_CASE("FDMTBoxcar width screen matches the reference at many widths",
          "[boxcar][packed][cpu]") {
    // Spacing 1 up to 48 samples: the CPU and GPU scorers screen tiles by
    // width group (boxcar_screen_groups), which must not change any result.
    const plans::FDMTPlan plan(700, 1200, 7, 311, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    constexpr SizeType blocks = 4;
    REQUIRE_FALSE(algorithms::detail::boxcar_screen_groups(
                      plans::generate_box_width_trials(48, 1.0F), false)
                      .empty());
    const auto input = waterfalls(plan, 1, blocks);
    // 8-bit copies of a uniform integer stream, and the centred floats.
    const SizeType size = plan.get_nchans() * plan.get_nsamps();
    std::vector<std::vector<uint8_t>> packed(blocks,
                                             std::vector<uint8_t>(size));
    std::vector<std::vector<float>> centered(blocks, std::vector<float>(size));
    std::mt19937 random(7);
    for (SizeType k = 0; k < blocks; ++k) {
        for (SizeType i = 0; i < size; ++i) {
            packed[k][i]   = static_cast<uint8_t>(random() & 255U);
            centered[k][i] = static_cast<float>(packed[k][i]) - 127.5F;
        }
    }
    const float sigma = std::sqrt(255.0F * 257.0F / 12.0F);
    for (const float thresh : {2.7F, 4.0F}) {
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            DYNAMIC_SECTION("thresh=" << thresh
                                      << " mode=" << static_cast<int>(mode)) {
                BoxcarConfig cfg;
                cfg.max_boxcar       = 48;
                cfg.wtsp             = 1.0F;
                cfg.detect_thresh    = thresh;
                cfg.peak_mode        = mode;
                cfg.max_candidates   = 2000000;
                cfg.max_cands_per_dm = 0;
                const auto expected =
                    batched_reference(plan, cfg, true, 1, input);
                REQUIRE_FALSE(expected.empty());
                BoxcarConfig pcfg = cfg;
                pcfg.noise_sigma  = sigma;
                const auto expected_packed =
                    batched_reference(plan, pcfg, true, 1, centered);
                REQUIRE_FALSE(expected_packed.empty());
                pcfg.input_nbits = 8;
                pcfg.input_mean  = 127.5F;
                std::vector<Exec> execs{Exec::cpu(3)};
#ifdef DMT_ENABLE_GPU
                execs.push_back(Exec{algorithms::detail::kGPUBackend, 0});
#endif
                for (const auto& exec : execs) {
                    for (const SizeType chunk :
                         {SizeType{64}, plan.get_nsamps()}) {
                        FDMTBoxcar search(plan, cfg, true, exec, 1,
                                          algorithms::kFDMTAutoFuse, chunk);
                        std::vector<RawCandidate> actual;
                        for (const auto& block : input) {
                            append(actual, search.execute(block));
                        }
                        append(actual, search.flush());
                        compare(actual, expected);
                        FDMTBoxcar packed_search(plan, pcfg, true, exec, 1,
                                                 algorithms::kFDMTAutoFuse,
                                                 chunk);
                        std::vector<RawCandidate> actual_packed;
                        for (const auto& block : packed) {
                            append(actual_packed, packed_search.execute(block));
                        }
                        append(actual_packed, packed_search.flush());
                        compare(actual_packed, expected_packed);
                    }
                }
            }
        }
    }
}

#ifdef DMT_ENABLE_GPU
TEST_CASE("GPU FDMTBoxcar direct levels and top agree with the CPU",
          "[boxcar][packed][gpu]") {
    // 512 channels: the GPU computes levels 0..3 directly and, for subband
    // depths 0 and 1, sums the trials (or level root - depth - 1) from a
    // level-k ring (the direct top); depth 2 has only the direct levels.
    // 2048 channels at depth 3: the top sums level 7 from level 4.
    // Packed input is exact on both backends (uint32_t state at 8 bits,
    // uint64_t at 16); float32 agrees within rounding near the threshold.
    // The packed search's summary names the path taken, so a size limit
    // cannot silently fall back to the plain tree. A reset_history() replays
    // the stream exactly.
    constexpr SizeType kBeams  = 2;
    constexpr SizeType kBlocks = 4;
    struct Case {
        SizeType nchans;
        SizeType depth;
        bool top;
    };
    for (const Case cs : {Case{512, 0, true}, Case{512, 1, true},
                          Case{512, 2, false}, Case{2048, 3, true}}) {
        const SizeType depth = cs.depth;
        const plans::FDMTPlan plan(1100.0F, 1714.0F, cs.nchans, 256, 455.6e-6F,
                                   300, 0, 1, "valid", depth);
        const SizeType size = kBeams * plan.get_nchans() * plan.get_nsamps();
        for (const SizeType bits : {SizeType{8}, SizeType{16}, SizeType{32}}) {
            DYNAMIC_SECTION("nchans=" << cs.nchans << " depth=" << depth
                                      << " bits=" << bits) {
                const uint32_t maximum = bits == 32 ? 0U : (1U << bits) - 1U;
                const uint64_t seed    = 17 + depth + bits;
                std::vector<std::vector<uint8_t>> packed(kBlocks);
                std::vector<std::vector<float>> floats(kBlocks);
                for (SizeType k = 0; k < kBlocks; ++k) {
                    if (bits == 32) {
                        floats[k] = dmt::test_random::normal<float>(
                            size, seed + (k << 8U));
                        continue;
                    }
                    const auto values = dmt::test_random::uniform_int<uint32_t>(
                        size, seed + (k << 8U), 0, maximum);
                    packed[k].resize(size * (bits / 8));
                    for (SizeType i = 0; i < size; ++i) {
                        const uint32_t v          = values[i];
                        packed[k][i * (bits / 8)] = static_cast<uint8_t>(v);
                        if (bits == 16) {
                            packed[k][(i * 2) + 1] =
                                static_cast<uint8_t>(v >> 8U);
                        }
                    }
                }
                BoxcarConfig cfg(24, 1.0F, 3.2F, 1.0F,
                                 PeakFindingMode::kContiguousPeakPerWidth);
                cfg.max_candidates   = 4000000;
                cfg.max_cands_per_dm = 0;
                if (bits != 32) {
                    cfg.input_nbits = bits;
                    cfg.input_mean  = static_cast<float>(maximum) * 0.5F;
                    cfg.noise_sigma = std::sqrt(static_cast<float>(maximum) *
                                                (maximum + 2.0F) / 12.0F);
                }
                for (const SizeType chunk : {SizeType{64}, SizeType{0}}) {
                    FDMTBoxcar cpu_s(plan, cfg, true, Exec::cpu(4), kBeams,
                                     algorithms::kFDMTAutoFuse, chunk);
                    FDMTBoxcar gpu_s(plan, cfg, true,
                                     Exec{algorithms::detail::kGPUBackend, 0},
                                     kBeams, algorithms::kFDMTAutoFuse, chunk);
                    std::vector<RawCandidate> c;
                    std::vector<RawCandidate> g;
                    for (SizeType k = 0; k < kBlocks; ++k) {
                        if (bits == 32) {
                            append(c, cpu_s.execute(floats[k]));
                            append(g, gpu_s.execute(floats[k]));
                        } else {
                            append(c, cpu_s.execute(packed[k]));
                            append(g, gpu_s.execute(packed[k]));
                        }
                    }
                    append(c, cpu_s.flush());
                    append(g, gpu_s.flush());
                    REQUIRE_FALSE(c.empty());
                    if (bits == 32) {
                        compare_near_threshold(g, c, cfg.detect_thresh);
                    } else {
                        compare(g, c);
                    }
                    const auto summary = gpu_s.summary();
                    INFO(summary);
                    // Only the packed search takes the direct path.
                    const bool direct = bits != 32;
                    CHECK((summary.find("direct levels") !=
                           std::string::npos) == direct);
                    CHECK((summary.find("direct top") != std::string::npos) ==
                          (direct && cs.top));
                    // Run, reset, run again: identical candidates.
                    gpu_s.reset_history();
                    std::vector<RawCandidate> again;
                    for (SizeType k = 0; k < kBlocks; ++k) {
                        if (bits == 32) {
                            append(again, gpu_s.execute(floats[k]));
                        } else {
                            append(again, gpu_s.execute(packed[k]));
                        }
                    }
                    append(again, gpu_s.flush());
                    compare(again, g);
                }
            }
        }
    }
}
#endif

TEST_CASE("FDMTBoxcar drain() ends the stream", "[boxcar][cpu]") {
    // Off the skewed paths every row is searched up to the last sample, so
    // drain() is flush() plus the end-of-stream contract.
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{0, 4, 17, 40, 97});
    const auto input = waterfalls(plan, 1, 3);
    const uint64_t n = input.size() * plan.get_nsamps();
    BoxcarConfig cfg;
    cfg.max_boxcar       = 37;
    cfg.wtsp             = 1;
    cfg.detect_thresh    = 2.7F;
    cfg.max_candidates   = 500000;
    cfg.max_cands_per_dm = 0;
    FDMTBoxcar flushed(plan, cfg, true, Exec::cpu(2));
    FDMTBoxcar drained(plan, cfg, true, Exec::cpu(2));
    CHECK(drained.get_search_latency() == 0);
    std::vector<RawCandidate> a;
    std::vector<RawCandidate> b;
    for (const auto& block : input) {
        append(a, flushed.execute(block));
        append(b, drained.execute(block));
    }
    CHECK(drained.get_searched_nsamps() == n);
    append(a, flushed.flush());
    append(b, drained.drain());
    compare(b, a);
    CHECK(drained.get_searched_nsamps() == n);
    CHECK_THROWS_AS(drained.execute(input.front()), std::logic_error);
    drained.reset_history();
    CHECK(drained.get_searched_nsamps() == 0);
    std::vector<RawCandidate> again;
    for (const auto& block : input) {
        append(again, drained.execute(block));
    }
    append(again, drained.drain());
    compare(again, a);
}

namespace {

// Rows of the killed channels of every beam and block set to `value`.
template <typename T>
void fill_killed(std::vector<std::vector<T>>& blocks,
                 std::span<const uint8_t> kill,
                 SizeType beams,
                 SizeType row,
                 T value) {
    for (auto& block : blocks) {
        for (SizeType b = 0; b < beams; ++b) {
            for (SizeType c = 0; c < kill.size(); ++c) {
                if (kill[c] == 0) {
                    std::fill_n(block.begin() +
                                    static_cast<std::ptrdiff_t>(
                                        ((b * kill.size()) + c) * row),
                                row, value);
                }
            }
        }
    }
}

} // namespace

TEST_CASE("FDMTBoxcar kill mask reads killed channels as zero",
          "[boxcar][kill][cpu]") {
    // Killed channels hold a strong signal that must not be seen; the search
    // equals the FDMT of the input with those channels zeroed, windowed
    // with the masked noise model. The second mask kills a whole half band:
    // its rows have no noise and never detect.
    const plans::FDMTPlan plan(700, 1200, 8, 96, 0.001F,
                               std::vector<IndexType>{0, 5, 17, 40}, "valid",
                               1);
    const auto input = waterfalls(plan, 2, 3);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 12;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.5F;
    cfg.max_candidates   = 200000;
    cfg.max_cands_per_dm = 0;
    const std::vector<uint8_t> scattered{1, 0, 1, 1, 1, 1, 0, 0};
    const std::vector<uint8_t> upper_half{1, 1, 1, 1, 0, 0, 0, 0};
    for (const bool half : {false, true}) {
        const auto& kill = half ? upper_half : scattered;
        DYNAMIC_SECTION("upper_half=" << half) {
            auto dirty = input;
            fill_killed<float>(dirty, kill, 2, plan.get_nsamps(), 50.0F);
            const auto expected =
                batched_reference(plan, cfg, true, 2, input, kill);
            REQUIRE_FALSE(expected.empty());
            for (const SizeType chunk : {SizeType{32}, SizeType{0}}) {
                FDMTBoxcar search(plan, cfg, true, Exec::cpu(3), 2,
                                  algorithms::kFDMTAutoFuse, chunk,
                                  algorithms::kFDMTDefaultMaxMemory, kill);
                std::vector<RawCandidate> actual;
                for (const auto& block : dirty) {
                    append(actual, search.execute(block));
                }
                append(actual, search.flush());
                compare(actual, expected);
                CHECK(search.get_memory_usage().staging ==
                      dirty.front().size() * sizeof(float));
            }
            // The noise model is the plan's, with the same mask.
            FDMTBoxcar search(plan, cfg, true, Exec::cpu(), 2,
                              algorithms::kFDMTAutoFuse, 0,
                              algorithms::kFDMTDefaultMaxMemory, kill);
            bool silent = false;
            for (const auto& p : search.get_search_plans()) {
                for (SizeType f = 0; f < p.widths.size(); ++f) {
                    const auto var = plan.get_subband_variance_grid(
                        p.level, p.subband_idx, p.widths[f], true, kill);
                    const auto full = plan.get_subband_variance_grid(
                        p.level, p.subband_idx, p.widths[f], true);
                    for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
                        const SizeType i = (d * p.widths.size()) + f;
                        CAPTURE(p.level, p.subband_idx, d, f);
                        CHECK(var[d] <= full[d]);
                        if (var[d] == 0) {
                            silent = true;
                            CHECK(p.inv_sigma_lut[i] == 0);
                            CHECK(p.raw_thresh_lut[i] ==
                                  std::numeric_limits<float>::max());
                            continue;
                        }
                        const double inv = p.inv_sigma_lut[i];
                        CHECK_THAT(1.0 / (inv * inv),
                                   Catch::Matchers::WithinRel(
                                       static_cast<double>(var[d]), 1e-5));
                    }
                }
            }
            CHECK(silent == half);
            if (half) {
                for (const auto& c : expected) {
                    CHECK(c.chan_begin < 4);
                }
            }
        }
    }
    FDMTBoxcar unmasked(plan, cfg, true, Exec::cpu(), 2);
    CHECK(unmasked.get_memory_usage().staging == 0);
    const std::vector<uint8_t> wrong(7, 1);
    CHECK_THROWS_AS(FDMTBoxcar(plan, cfg, true, Exec::cpu(), 1,
                               algorithms::kFDMTAutoFuse, 0,
                               algorithms::kFDMTDefaultMaxMemory, wrong),
                    std::invalid_argument);
}

TEST_CASE("Packed FDMTBoxcar kill mask reads killed channels as code zero",
          "[boxcar][kill][packed][cpu]") {
    // Killed rows hold the largest code; the packed search ignores them and
    // leaves them out of the mean, so it equals the centred float search
    // with those channels zeroed.
    const plans::FDMTPlan plan(700, 1200, 7, 139, 0.001F,
                               std::vector<IndexType>{-73, -12, 0, 4, 97},
                               "valid", 2);
    constexpr SizeType beams  = 2;
    constexpr SizeType blocks = 3;
    const std::vector<uint8_t> kill{0, 1, 1, 0, 1, 1, 1};
    for (const SizeType bits : {SizeType{2}, SizeType{8}}) {
        DYNAMIC_SECTION("bits=" << bits) {
            const uint32_t maximum   = (1U << bits) - 1;
            const float mean         = static_cast<float>(maximum) * 0.5F;
            const auto maxf          = static_cast<float>(maximum);
            const float sigma        = std::sqrt(maxf * (maxf + 2.0F) / 12.0F);
            const SizeType row_bytes = ((plan.get_nsamps() * bits) + 7) / 8;
            std::vector<std::vector<uint8_t>> packed(
                blocks,
                std::vector<uint8_t>(beams * plan.get_nchans() * row_bytes));
            std::vector<std::vector<float>> centered(
                blocks, std::vector<float>(beams * plan.get_nchans() *
                                           plan.get_nsamps()));
            std::mt19937 random(7);
            for (SizeType k = 0; k < blocks; ++k) {
                for (SizeType row = 0; row < beams * plan.get_nchans(); ++row) {
                    for (SizeType t = 0; t < plan.get_nsamps(); ++t) {
                        const auto value = random() & maximum;
                        centered[k][(row * plan.get_nsamps()) + t] =
                            static_cast<float>(value) - mean;
                        for (SizeType bit = 0; bit < bits; ++bit) {
                            const SizeType offset = (t * bits) + bit;
                            packed[k][(row * row_bytes) + (offset / 8)] |=
                                ((value >> bit) & 1U) << (offset % 8);
                        }
                    }
                }
            }
            fill_killed<uint8_t>(packed, kill, beams, row_bytes, 0xFF);
            BoxcarConfig cfg;
            cfg.max_boxcar       = 37;
            cfg.wtsp             = 1.5F;
            cfg.detect_thresh    = 2.713F;
            cfg.noise_sigma      = sigma;
            cfg.max_candidates   = 200000;
            cfg.max_cands_per_dm = 0;
            const auto expected =
                batched_reference(plan, cfg, true, beams, centered, kill);
            REQUIRE_FALSE(expected.empty());
            cfg.input_nbits = bits;
            cfg.input_mean  = mean;
            FDMTBoxcar search(plan, cfg, true, Exec::cpu(3), beams,
                              algorithms::kFDMTAutoFuse, 64,
                              algorithms::kFDMTDefaultMaxMemory, kill);
            std::vector<RawCandidate> actual;
            for (const auto& block : packed) {
                append(actual, search.execute(block));
            }
            append(actual, search.flush());
            compare(actual, expected);
            CHECK(search.get_memory_usage().staging == packed.front().size());
        }
    }
}

TEST_CASE("FDMTBoxcar kill mask keeps scores calibrated",
          "[boxcar][kill][cpu]") {
    // Gaussian noise: the masked scores have unit variance. A pulse of
    // amplitude a in every channel at DM 0 scores a sqrt(N - k) with k
    // channels killed; zeroing them upstream without a mask scores
    // a (N - k) / sqrt(N), lower by sqrt(N / (N - k)).
    const SizeType samples = 32768;
    const SizeType nchans  = 8;
    const plans::FDMTPlan plan(700, 1200, nchans, samples, 0.001F,
                               std::vector<IndexType>{-20, 0, 19}, "valid", 1);
    const std::vector<uint8_t> kill{1, 1, 0, 1, 1, 1, 0, 1};
    BoxcarConfig cfg;
    cfg.max_boxcar = 7;
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(), 1,
                      algorithms::kFDMTAutoFuse, 0,
                      algorithms::kFDMTDefaultMaxMemory, kill);
    FDMT fdmt(plan, true, {}, 1, 0, false, samples);
    auto input = dmt::test_random::normal<float>(nchans * samples, 33);
    std::vector<float> output(plan.get_dmt_size());
    for (SizeType c = 0; c < nchans; ++c) {
        if (kill[c] == 0) {
            std::fill_n(input.begin() +
                            static_cast<std::ptrdiff_t>(c * samples),
                        samples, 0.0F);
        }
    }
    fdmt.reset(input, output);
    for (const auto& p : search.get_search_plans()) {
        fdmt.advance_until_remaining(p.subband_level);
        const auto values = fdmt.view_subband(p.subband_idx).data;
        for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
            for (SizeType f = 0; f < p.widths.size(); ++f) {
                const SizeType first = p.warmup_samples[d] + p.widths[f] - 1;
                double sum = 0, square = 0;
                const SizeType count = samples - first;
                for (SizeType t = first; t < samples; ++t) {
                    double value = 0;
                    for (SizeType k = 0; k < p.widths[f]; ++k) {
                        value += values[(d * samples) + t - k];
                    }
                    value *= p.inv_sigma_lut[(d * p.widths.size()) + f];
                    sum += value;
                    square += value * value;
                }
                const double mean = sum / static_cast<double>(count);
                CAPTURE(p.subband_level, p.subband_idx, d, f);
                CHECK(std::abs(mean) < 0.08);
                CHECK(std::abs((square / static_cast<double>(count)) -
                               (mean * mean) - 1.0) < 0.08);
            }
        }
    }
    fdmt.finalize();

    const plans::FDMTPlan pulse_plan(700, 1200, nchans, 512, 0.001F,
                                     std::vector<IndexType>{0, 3, 9});
    BoxcarConfig pulse_cfg;
    pulse_cfg.max_boxcar    = 1;
    pulse_cfg.detect_thresh = 2;
    constexpr float kAmp    = 4;
    std::vector<float> pulse(nchans * 512, 0.0F);
    std::vector<float> zeroed(nchans * 512, 0.0F);
    for (SizeType c = 0; c < nchans; ++c) {
        pulse[(c * 512) + 300] = kAmp;
        if (kill[c] != 0) {
            zeroed[(c * 512) + 300] = kAmp;
        }
    }
    const auto root_snr = [](CandidateResultView view) {
        float best = 0;
        for (const auto& c : view.candidates) {
            if (c.dm_idx == 0 && c.subband_level == 0) {
                CHECK(c.peak_sample == 300);
                best = std::max(best, c.snr);
            }
        }
        return best;
    };
    FDMTBoxcar masked(pulse_plan, pulse_cfg, true, Exec::cpu(), 1,
                      algorithms::kFDMTAutoFuse, 0,
                      algorithms::kFDMTDefaultMaxMemory, kill);
    FDMTBoxcar unmasked(pulse_plan, pulse_cfg);
    const double n = nchans;
    const double k = 2;
    CHECK_THAT(root_snr(masked.execute(pulse)),
               Catch::Matchers::WithinRel(kAmp * std::sqrt(n - k), 1e-6));
    CHECK_THAT(root_snr(unmasked.execute(pulse)),
               Catch::Matchers::WithinRel(kAmp * std::sqrt(n), 1e-6));
    unmasked.reset_history();
    CHECK_THAT(root_snr(unmasked.execute(zeroed)),
               Catch::Matchers::WithinRel(kAmp * (n - k) / std::sqrt(n), 1e-6));
}
