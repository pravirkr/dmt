#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"

namespace boxcar_test {
using namespace dmt;
using algorithms::FDMT;
using algorithms::FDMTBoxcar;

inline auto key(const RawCandidate& c) {
    return std::tuple(c.beam_idx, c.subband_level, c.subband_idx, c.dm_idx,
                      c.filter_idx, c.begin_sample, c.end_sample,
                      c.peak_sample);
}
inline void sort(std::vector<RawCandidate>& candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return key(a) < key(b); });
}
inline void compare(std::vector<RawCandidate> actual,
                    std::vector<RawCandidate> expected) {
    sort(actual);
    sort(expected);
    REQUIRE(actual.size() == expected.size());
    for (SizeType i = 0; i < actual.size(); ++i) {
        INFO("candidate " << i);
        REQUIRE(key(actual[i]) == key(expected[i]));
        CHECK(actual[i].width == expected[i].width);
        CHECK(actual[i].chan_begin == expected[i].chan_begin);
        CHECK(actual[i].chan_end == expected[i].chan_end);
        CHECK(actual[i].dm == expected[i].dm);
        CHECK(actual[i].f_low_mhz == expected[i].f_low_mhz);
        CHECK(actual[i].f_high_mhz == expected[i].f_high_mhz);
        CHECK_THAT(actual[i].snr,
                   Catch::Matchers::WithinRel(expected[i].snr, 2e-5F));
    }
}
inline void append(std::vector<RawCandidate>& to, CandidateResultView view) {
    REQUIRE_FALSE(view.overflowed());
    REQUIRE(view.candidates.size() == view.emitted_count);
    to.insert(to.end(), view.candidates.begin(), view.candidates.end());
}

inline std::vector<std::vector<float>>
waterfalls(const plans::FDMTPlan& plan, SizeType beams, SizeType blocks) {
    std::mt19937 random(42);
    std::normal_distribution<float> noise;
    std::vector<std::vector<float>> input(
        blocks,
        std::vector<float>(beams * plan.get_nchans() * plan.get_nsamps()));
    for (SizeType k = 0; k < blocks; ++k) {
        for (SizeType b = 0; b < beams; ++b) {
            for (SizeType c = 0; c < plan.get_nchans(); ++c) {
                for (SizeType t = 0; t < plan.get_nsamps(); ++t) {
                    const SizeType time = k * plan.get_nsamps() + t;
                    // A pulse across each block boundary, stronger in the
                    // first input channels to exercise subband selection.
                    const float pulse =
                        time % plan.get_nsamps() < 6 ||
                                time % plan.get_nsamps() > plan.get_nsamps() - 9
                            ? (c < 3 ? 3.0F : 1.0F)
                            : 0.0F;
                    input[k]
                         [(b * plan.get_nchans() + c) * plan.get_nsamps() + t] =
                             std::round(noise(random) * 16.0F) / 16.0F + pulse;
                }
            }
        }
    }
    return input;
}

// Materialize existing FDMT's actual intermediate planes, then evaluate
// windows by direct float64 summation. No new scorer/prefix helpers used.
inline std::vector<RawCandidate>
reference(const plans::FDMTPlan& plan,
          const BoxcarConfig& config,
          bool smear,
          SizeType beams,
          const std::vector<std::vector<float>>& input) {
    FDMTBoxcar metadata(plan, config, smear, Exec::cpu(), beams);
    const auto planes = metadata.get_search_plans();
    std::vector<std::vector<float>> rows;
    SizeType nrows = 0;
    for (const auto& p : planes) {
        nrows += beams * p.dm_grid.size();
    }
    rows.resize(nrows);
    FDMT fdmt(plan.get_f_min(), plan.get_f_max(), plan.get_nchans(),
              plan.get_nsamps(), plan.get_tsamp(), plan.get_dt_grid_final(),
              smear, "valid", Exec::cpu(), beams, 0, false, plan.get_nsamps());
    std::vector<float> output(beams * plan.get_dmt_size());
    for (const auto& block : input) {
        fdmt.reset(block, output);
        SizeType index = 0;
        for (const auto& p : planes) {
            fdmt.advance_until_remaining(p.subband_level);
            const auto level = fdmt.view_level_data();
            const auto& pc   = plan.get_container();
            // Public level view is the first beam; the stepper's internal
            // halves include padding. Run separate references per beam.
            REQUIRE(beams == 1);
            for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
                const SizeType offset =
                    (p.coord_offset + d) * pc.state_shape[p.level].nsamps;
                rows[index].insert(rows[index].end(), level.begin() + offset,
                                   level.begin() + offset + plan.get_nsamps());
                ++index;
            }
        }
        fdmt.finalize();
    }
    std::vector<RawCandidate> expected;
    SizeType row_idx = 0;
    for (const auto& p : planes) {
        for (SizeType d = 0; d < p.dm_grid.size(); ++d, ++row_idx) {
            const auto& values = rows[row_idx];
            std::vector<RawCandidate> open(p.widths.size());
            std::vector<bool> active(p.widths.size());
            const auto point = [&](SizeType t, SizeType f, float snr) {
                RawCandidate c;
                c.peak_sample = c.begin_sample = c.end_sample = t;
                c.snr                                         = snr;
                c.dm                                          = p.dm_grid[d];
                c.dm_idx        = static_cast<uint32_t>(d);
                c.filter_idx    = static_cast<uint32_t>(f);
                c.width         = p.widths[f];
                c.subband_level = p.subband_level;
                c.subband_idx   = p.subband_idx;
                c.chan_begin    = p.chan_begin;
                c.chan_end      = p.chan_end;
                c.f_low_mhz     = p.f_low_mhz;
                c.f_high_mhz    = p.f_high_mhz;
                return c;
            };
            const auto feed = [&](SizeType state, bool above, RawCandidate c) {
                if (!above) {
                    if (active[state]) {
                        expected.push_back(open[state]);
                        active[state] = false;
                    }
                } else if (!active[state]) {
                    open[state]   = c;
                    active[state] = true;
                } else {
                    auto& old        = open[state];
                    const auto begin = old.begin_sample;
                    if (c.snr > old.snr) {
                        old              = c;
                        old.begin_sample = begin;
                    }
                    old.end_sample = c.end_sample;
                }
            };
            for (SizeType t = 0; t < values.size(); ++t) {
                bool any = false;
                RawCandidate best;
                for (SizeType f = 0; f < p.widths.size(); ++f) {
                    const SizeType w = p.widths[f];
                    double sum       = 0;
                    for (SizeType k = 0; k < w && k <= t; ++k) {
                        sum += values[t - k];
                    }
                    const bool valid = !config.skip_cold_prefix ||
                                       t >= p.warmup_samples[d] + w - 1;
                    const bool above =
                        valid &&
                        sum >= p.raw_thresh_lut[d * p.widths.size() + f];
                    const auto c = point(
                        t, f,
                        static_cast<float>(
                            sum * p.inv_sigma_lut[d * p.widths.size() + f]));
                    if (config.peak_mode ==
                        PeakFindingMode::kAllAboveThreshold) {
                        if (above) {
                            expected.push_back(c);
                        }
                    } else if (config.peak_mode ==
                               PeakFindingMode::kContiguousPeakPerWidth) {
                        feed(f, above, c);
                    } else if (above && (!any || c.snr > best.snr)) {
                        best = c;
                        any  = true;
                    }
                }
                if (config.peak_mode ==
                    PeakFindingMode::kContiguousPeakAcrossWidths) {
                    feed(0, any, best);
                }
            }
            for (SizeType f = 0; f < active.size(); ++f) {
                if (active[f]) {
                    expected.push_back(open[f]);
                }
            }
        }
    }
    return expected;
}

inline std::vector<RawCandidate>
batched_reference(const plans::FDMTPlan& plan,
                  const BoxcarConfig& config,
                  bool smear,
                  SizeType beams,
                  const std::vector<std::vector<float>>& input) {
    std::vector<RawCandidate> expected;
    const SizeType size = plan.get_nchans() * plan.get_nsamps();
    for (SizeType b = 0; b < beams; ++b) {
        auto beam = input;
        for (SizeType k = 0; k < input.size(); ++k) {
            beam[k].assign(input[k].begin() + b * size,
                           input[k].begin() + (b + 1) * size);
        }
        auto candidates = reference(plan, config, smear, 1, beam);
        for (auto& c : candidates) {
            c.beam_idx = static_cast<uint32_t>(b);
        }
        expected.insert(expected.end(), candidates.begin(), candidates.end());
    }
    return expected;
}
} // namespace boxcar_test
