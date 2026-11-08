#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "dmt/engines.hpp"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;
using algorithms::FDMTFFT;
using algorithms::FDMTFFTBoxcar;

namespace {

// Forces several overlap-save segments per block; restored at scope exit.
struct SegmentCap {
    explicit SegmentCap(SizeType n) {
        algorithms::detail::set_fft_segment_cap(n);
    }
    ~SegmentCap() { algorithms::detail::set_fft_segment_cap(0); }
    SegmentCap(const SegmentCap&)            = delete;
    SegmentCap& operator=(const SegmentCap&) = delete;
    SegmentCap(SegmentCap&&)                 = delete;
    SegmentCap& operator=(SegmentCap&&)      = delete;
};

} // namespace

TEST_CASE("FDMTFFTBoxcar matches materialized FDMTFFT and direct windows",
          "[boxcar][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{0, 4, 17, 40, 97});
    const auto input = waterfalls(plan, 1, 4);
    for (const SizeType cap : {SizeType{0}, SizeType{512}}) {
        const SegmentCap segments(cap);
        for (const bool smear : {true, false}) {
            for (const bool frac : {true, false}) {
                for (const auto mode :
                     {PeakFindingMode::kContiguousPeakPerWidth,
                      PeakFindingMode::kContiguousPeakAcrossWidths,
                      PeakFindingMode::kAllAboveThreshold}) {
                    CAPTURE(cap, smear, frac, mode);
                    BoxcarConfig cfg;
                    cfg.max_boxcar       = 37;
                    cfg.wtsp             = 1;
                    cfg.detect_thresh    = 2.7F;
                    cfg.peak_mode        = mode;
                    cfg.max_candidates   = 500000;
                    cfg.max_cands_per_dm = 0;
                    FDMTFFTBoxcar search(plan, cfg, smear, Exec::cpu(2), 1,
                                         frac);
                    CHECK(search.get_output_latency() == (frac ? 64U : 0U));
                    std::vector<RawCandidate> actual;
                    for (const auto& block : input) {
                        append(actual, search.execute(block));
                    }
                    append(actual, search.flush());
                    compare(actual, fft_reference(plan, cfg, smear, frac, {},
                                                  Exec::cpu(), input));
                    // A reset stream reproduces the same candidates.
                    search.reset_history();
                    std::vector<RawCandidate> again;
                    for (const auto& block : input) {
                        append(again, search.execute(block));
                    }
                    append(again, search.flush());
                    compare(again, actual);
                }
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar beams and kill mask match per-beam references",
          "[boxcar][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 8, 600, 0.001F,
                               std::vector<IndexType>{0, 9, 33, 71});
    const auto input = waterfalls(plan, 2, 3);
    const std::vector<uint8_t> kill{1, 0, 1, 1, 0, 1, 1, 1};
    BoxcarConfig cfg;
    cfg.max_boxcar       = 16;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.max_cands_per_dm = 0;
    FDMTFFTBoxcar search(plan, cfg, true, Exec::cpu(3), 2, true, kill);
    std::vector<RawCandidate> actual;
    for (const auto& block : input) {
        append(actual, search.execute(block));
    }
    append(actual, search.flush());
    std::vector<RawCandidate> expected;
    const SizeType size = plan.get_nchans() * plan.get_nsamps();
    for (SizeType b = 0; b < 2; ++b) {
        auto beam = input;
        for (SizeType k = 0; k < input.size(); ++k) {
            beam[k].assign(input[k].begin() + b * size,
                           input[k].begin() + (b + 1) * size);
        }
        auto c = fft_reference(plan, cfg, true, true, kill, Exec::cpu(), beam);
        for (auto& x : c) {
            x.beam_idx = static_cast<uint32_t>(b);
        }
        expected.insert(expected.end(), c.begin(), c.end());
    }
    compare(actual, expected);
}

TEST_CASE("FDMTFFTBoxcar analytical scaling predicts Gaussian noise",
          "[boxcar][cpu]") {
    // Fractional shifts are all-pass: FDMTBoxcar's leaf energies give each
    // row's boxcar variance, also with killed channels left out.
    const SizeType samples = 16384;
    const plans::FDMTPlan plan(700, 1200, 16, samples, 0.001F,
                               std::vector<IndexType>{0, 7, 30, 61});
    const std::vector<uint8_t> none;
    const std::vector<uint8_t> kill{1, 1, 0, 1, 1, 1, 1, 0,
                                    1, 1, 1, 1, 0, 1, 1, 1};
    for (const auto& mask : {none, kill}) {
        for (const bool smear : {true, false}) {
            CAPTURE(mask.size(), smear);
            BoxcarConfig cfg;
            cfg.max_boxcar = 9;
            FDMTFFTBoxcar search(plan, cfg, smear, Exec::cpu(), 1, true, mask);
            FDMTFFT fft(700, 1200, 16, samples, 0.001F,
                        plan.get_dt_grid_final(), smear, "valid", Exec::cpu(),
                        1, true, mask);
            std::mt19937 random(32);
            std::normal_distribution<float> noise;
            std::vector<float> input(16 * samples);
            std::vector<float> output(plan.get_dmt_size());
            // Two blocks: the second is fully warm.
            for (int block = 0; block < 2; ++block) {
                for (auto& x : input) {
                    x = noise(random);
                }
                fft.execute(input, output);
            }
            const auto& p = search.get_search_plans().front();
            for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
                for (SizeType f = 0; f < p.widths.size(); ++f) {
                    double sum = 0, square = 0;
                    SizeType count = 0;
                    for (SizeType t = p.widths[f] - 1; t < samples; ++t) {
                        double value = 0;
                        for (SizeType k = 0; k < p.widths[f]; ++k) {
                            value += output[d * samples + t - k];
                        }
                        value *= p.inv_sigma_lut[d * p.widths.size() + f];
                        sum += value;
                        square += value * value;
                        ++count;
                    }
                    const double mean = sum / static_cast<double>(count);
                    CHECK(std::abs(mean) < 0.08);
                    CHECK(std::abs(square / static_cast<double>(count) -
                                   mean * mean - 1.0) < 0.08);
                }
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar validates its configuration", "[boxcar][cpu]") {
    const plans::FDMTPlan plan(700, 1200, 4, 256, 0.001F,
                               std::vector<IndexType>{0, 5, 9});
    const plans::FDMTPlan subband_plan(
        700, 1200, 4, 256, 0.001F, std::vector<IndexType>{0, 5, 9}, "valid", 1);
    BoxcarConfig subbands;
    subbands.max_boxcar = 8;
    CHECK_THROWS_AS(FDMTFFTBoxcar(subband_plan, subbands),
                    std::invalid_argument);
    BoxcarConfig packed;
    packed.max_boxcar  = 8;
    packed.input_nbits = 8;
    CHECK_THROWS_AS(FDMTFFTBoxcar(plan, packed), std::invalid_argument);
    BoxcarConfig cfg;
    cfg.max_boxcar = 8;
    CHECK_THROWS_AS(FDMTFFTBoxcar(plan, cfg, true, Exec::cpu(), 1, true,
                                  std::vector<uint8_t>{1, 1, 1}),
                    std::invalid_argument);
    FDMTFFTBoxcar search(plan, cfg);
    CHECK_THROWS_AS(search.execute(std::vector<float>(10)),
                    std::invalid_argument);
    CHECK(search.get_memory_usage().total() > 0);
    CHECK(search.get_chunk_nsamps() > 0);
    CHECK_FALSE(search.summary().empty());
}

TEST_CASE("FDMTFFTBoxcar warmup_samples excludes killed edge channels",
          "[boxcar][cpu]") {
    const plans::FDMTPlan plan(700.0F, 1200.0F, 16, 256, 0.001F, 32, 0, 1,
                               "valid");
    BoxcarConfig cfg;
    cfg.max_boxcar = 8;

    FDMTFFTBoxcar unmasked(plan, cfg, true, Exec::cpu(), 1, true, {});
    // Kill the upper half of channels (highest frequencies, which carry the
    // largest lookback shift relative to f_min).
    std::vector<uint8_t> kill_mask(16, 1);
    for (size_t c = 8; c < 16; ++c) {
        kill_mask[c] = 0;
    }
    FDMTFFTBoxcar masked(plan, cfg, true, Exec::cpu(), 1, true, kill_mask);

    REQUIRE(unmasked.get_search_plans().size() == 1);
    REQUIRE(masked.get_search_plans().size() == 1);
    CHECK(masked.get_search_plans()[0].warmup_samples.back() <
          unmasked.get_search_plans()[0].warmup_samples.back());
}

TEST_CASE("FDMTFFTBoxcar warmup is each masked row's exact cold support",
          "[boxcar][cpu]") {
    // A cold stream (zero history) and the same stream after real data must
    // agree on every row from warmup_samples[d] on, and differ one sample
    // earlier: the warmup is safe and no longer than needed. Integer shifts
    // (no fractional guard) make the support exact. Positive input keeps
    // the contributions at the first unsupported sample from cancelling.
    const SizeType nchans = 16;
    const SizeType nsamps = 256;
    const plans::FDMTPlan plan(700.0F, 1200.0F, nchans, nsamps, 0.001F, 32, 0,
                               1, "valid");
    BoxcarConfig cfg;
    cfg.max_boxcar = 8;
    std::mt19937 random(5);
    std::uniform_real_distribution<float> value(1.0F, 2.0F);
    const auto blocks = [&](SizeType n) {
        std::vector<std::vector<float>> out(
            n, std::vector<float>(nchans * nsamps));
        for (auto& block : out) {
            for (auto& x : block) {
                x = value(random);
            }
        }
        return out;
    };
    const auto prefix = blocks(2);
    const auto stream = blocks(3);

    std::vector<uint8_t> top(nchans, 1);
    std::vector<uint8_t> bottom(nchans, 1);
    std::vector<uint8_t> scattered(nchans, 1);
    for (SizeType c = 0; c < nchans / 2; ++c) {
        bottom[c]             = 0;
        top[c + (nchans / 2)] = 0;
    }
    for (const SizeType c : {3, 9, 15}) {
        scattered[c] = 0;
    }
    const std::vector<std::vector<uint8_t>> masks{{}, top, bottom, scattered};
    for (SizeType m = 0; m < masks.size(); ++m) {
        const auto& mask = masks[m];
        for (const bool smear : {true, false}) {
            CAPTURE(m, smear);
            FDMTFFTBoxcar search(plan, cfg, smear, Exec::cpu(), 1, false, mask);
            REQUIRE(search.get_output_latency() == 0);
            const auto warmup =
                search.get_search_plans().front().warmup_samples;
            const auto rows = [&](bool warm) {
                FDMTFFT fft(plan, smear, Exec::cpu(), 1, false, mask);
                std::vector<float> output(plan.get_dmt_size());
                if (warm) {
                    for (const auto& block : prefix) {
                        fft.execute(block, output);
                    }
                }
                std::vector<std::vector<float>> out(plan.get_dmt_ndms());
                for (const auto& block : stream) {
                    fft.execute(block, output);
                    for (SizeType d = 0; d < out.size(); ++d) {
                        const auto* row = output.data() + (d * nsamps);
                        out[d].insert(out[d].end(), row, row + nsamps);
                    }
                }
                return out;
            };
            const auto cold = rows(false);
            const auto hot  = rows(true);
            REQUIRE(warmup.size() == cold.size());
            for (SizeType d = 0; d < cold.size(); ++d) {
                CAPTURE(d, warmup[d]);
                REQUIRE(warmup[d] < cold[d].size());
                float scale = 1e-30F;
                for (const auto v : hot[d]) {
                    scale = std::max(scale, std::abs(v));
                }
                const float tol = 1e-4F * scale;
                for (SizeType t = warmup[d]; t < cold[d].size(); ++t) {
                    REQUIRE(std::abs(cold[d][t] - hot[d][t]) <= tol);
                }
                if (warmup[d] > 0) {
                    const auto t = warmup[d] - 1;
                    CHECK(std::abs(cold[d][t] - hot[d][t]) > 10.0F * tol);
                }
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar drain() searches the last latency samples",
          "[boxcar][cpu]") {
    // With fractional delays the search lags the input by the latency, and
    // flush() never searches the last `latency` samples. drain() does, with
    // zero signal after the end: it matches the same stream continued by
    // zero blocks, on every endpoint before the end.
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{0, 4, 17, 40, 97});
    const auto input = waterfalls(plan, 1, 3);
    const uint64_t n = input.size() * plan.get_nsamps();
    BoxcarConfig cfg;
    cfg.max_boxcar       = 37;
    cfg.wtsp             = 1;
    cfg.detect_thresh    = 2.0F;
    cfg.peak_mode        = PeakFindingMode::kAllAboveThreshold;
    cfg.max_candidates   = 500000;
    cfg.max_cands_per_dm = 0;
    for (const bool frac : {true, false}) {
        CAPTURE(frac);
        FDMTFFTBoxcar search(plan, cfg, true, Exec::cpu(2), 1, frac);
        const uint64_t latency = search.get_output_latency();
        CHECK(latency == (frac ? 64U : 0U));
        std::vector<RawCandidate> drained;
        for (const auto& block : input) {
            append(drained, search.execute(block));
        }
        CHECK(search.get_searched_nsamps() == n - latency);
        const auto tail = search.drain();
        append(drained, tail);
        CHECK(tail.input_sample_begin == n);
        CHECK(search.get_searched_nsamps() == n);
        for (const auto& c : drained) {
            REQUIRE(c.end_sample < n);
        }
        // The stream has ended.
        CHECK_THROWS_AS(search.execute(input.front()), std::logic_error);
        CHECK_THROWS_AS(search.drain(), std::logic_error);

        // Reference: the stream continued with zero blocks, cut at the end.
        FDMTFFTBoxcar longer(plan, cfg, true, Exec::cpu(2), 1, frac);
        std::vector<RawCandidate> expected;
        for (const auto& block : input) {
            append(expected, longer.execute(block));
        }
        const std::vector<float> zeros(input.front().size(), 0.0F);
        append(expected, longer.execute(zeros));
        append(expected, longer.flush());
        std::erase_if(expected,
                      [&](const RawCandidate& c) { return c.end_sample >= n; });
        compare(drained, expected);
        if (frac) {
            // The tail was searched: candidates flush() alone never gives.
            const auto in_tail =
                std::ranges::count_if(drained, [&](const RawCandidate& c) {
                    return c.end_sample >= n - latency;
                });
            CHECK(in_tail > 0);
        }

        // reset_history() starts a new stream that reproduces the first.
        search.reset_history();
        CHECK(search.get_searched_nsamps() == 0);
        std::vector<RawCandidate> again;
        for (const auto& block : input) {
            append(again, search.execute(block));
        }
        append(again, search.drain());
        compare(again, drained);
    }
}
