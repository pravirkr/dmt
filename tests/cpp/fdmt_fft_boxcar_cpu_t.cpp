#include "fdmt_boxcar_reference.hpp"

#include <cstdint>
#include <random>
#include <vector>

#include "dmt/engines.hpp"

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
    BoxcarConfig subbands;
    subbands.max_boxcar     = 8;
    subbands.subband_levels = 1;
    CHECK_THROWS_AS(FDMTFFTBoxcar(plan, subbands), std::invalid_argument);
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
