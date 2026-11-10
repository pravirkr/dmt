#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/format.h>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/algorithms/fdmt_fft_boxcar.hpp"

#include "dmt/fdmt_variance.hpp"

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
    const auto show = [](const RawCandidate& c) {
        return fmt::format("beam {} level {} band {} dm {} filter {} [{}, {}] "
                           "peak {} snr {}",
                           c.beam_idx, c.subband_level, c.subband_idx, c.dm_idx,
                           c.filter_idx, c.begin_sample, c.end_sample,
                           c.peak_sample, c.snr);
    };
    for (SizeType i = 0; i < actual.size(); ++i) {
        INFO("candidate " << i << "\n  actual   " << show(actual[i])
                          << "\n  expected " << show(expected[i]));
        REQUIRE(key(actual[i]) == key(expected[i]));
        CHECK(actual[i].width == expected[i].width);
        CHECK(actual[i].chan_begin == expected[i].chan_begin);
        CHECK(actual[i].chan_end == expected[i].chan_end);
        CHECK(actual[i].staggered == expected[i].staggered);
        CHECK(actual[i].dm == expected[i].dm);
        CHECK(actual[i].f_low_mhz == expected[i].f_low_mhz);
        CHECK(actual[i].f_high_mhz == expected[i].f_high_mhz);
        CHECK_THAT(actual[i].snr,
                   Catch::Matchers::WithinRel(expected[i].snr, 2e-5F));
    }
}
// For rows that are not exact in float (FDMT-FFT): float rounding may move
// a point across the threshold, so run edges may differ and whole runs may
// appear or vanish, but only at S/N within `rtol` of the threshold; and
// plateau ties (equal S/N at several samples) may pick another peak sample.
// Runs are matched by (beam, plane, DM, filter) and overlapping extents.
inline void compare_near_threshold(std::vector<RawCandidate> actual,
                                   std::vector<RawCandidate> expected,
                                   float threshold,
                                   float rtol = 1e-4F) {
    const auto row = [](const RawCandidate& c) {
        return std::tuple(c.beam_idx, c.subband_level, c.subband_idx, c.dm_idx,
                          c.filter_idx);
    };
    const auto order = [&](const auto& a, const auto& b) {
        return std::tuple(row(a), a.begin_sample) <
               std::tuple(row(b), b.begin_sample);
    };
    std::sort(actual.begin(), actual.end(), order);
    std::sort(expected.begin(), expected.end(), order);
    const auto marginal = [&](const RawCandidate& c) {
        return std::abs(c.snr / threshold - 1.0F) < rtol;
    };
    const auto overlap = [&](const RawCandidate& a, const RawCandidate& b) {
        return row(a) == row(b) && a.begin_sample <= b.end_sample &&
               b.begin_sample <= a.end_sample;
    };
    SizeType i = 0, j = 0, moved = 0;
    while (i < actual.size() || j < expected.size()) {
        if (i < actual.size() && j < expected.size() &&
            overlap(actual[i], expected[j])) {
            INFO("run " << actual[i].begin_sample << ".."
                        << actual[i].end_sample << " vs "
                        << expected[j].begin_sample << ".."
                        << expected[j].end_sample);
            CHECK_THAT(actual[i].snr,
                       Catch::Matchers::WithinRel(expected[j].snr, 2e-5F));
            moved += actual[i].begin_sample != expected[j].begin_sample ||
                     actual[i].end_sample != expected[j].end_sample ||
                     actual[i].peak_sample != expected[j].peak_sample;
            ++i;
            ++j;
        } else if (j == expected.size() ||
                   (i < actual.size() && order(actual[i], expected[j]))) {
            INFO("extra run at " << actual[i].peak_sample << " snr "
                                 << actual[i].snr);
            CHECK(marginal(actual[i]));
            ++i;
        } else {
            INFO("missing run at " << expected[j].peak_sample << " snr "
                                   << expected[j].snr);
            CHECK(marginal(expected[j]));
            ++j;
        }
    }
    // Rounding moves are rare: at most one run in a hundred.
    CHECK(moved * 100 <= expected.size() + 100);
}

// For FDMT-FFT rows whose values differ by a bound relative to each row's
// peak (overlap-save segments of other lengths with fractional delays: the
// kernel truncation, below 0.5% of the peak). Per (beam, plane, DM,
// filter) row, overlapping runs of both sets form clusters (a point near
// the threshold may split or join runs); matched clusters' peak S/N agree
// within frac * the row's peak + abs_tol (S/N units: noise rows differ by
// a fraction of sigma), and a cluster on one side only peaks within that of
// the threshold.
inline void compare_clusters(const std::vector<RawCandidate>& actual,
                             const std::vector<RawCandidate>& expected,
                             float threshold,
                             double frac,
                             double abs_tol) {
    using Row =
        std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;
    struct Tagged {
        Row row;
        const RawCandidate* c;
        int side;
    };
    std::vector<Tagged> all;
    for (int side = 0; side < 2; ++side) {
        for (const auto& c : side == 0 ? actual : expected) {
            all.push_back({{c.beam_idx, c.subband_level, c.subband_idx,
                            c.staggered, c.dm_idx, c.filter_idx},
                           &c,
                           side});
        }
    }
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return std::tie(a.row, a.c->begin_sample) <
               std::tie(b.row, b.c->begin_sample);
    });
    SizeType i        = 0;
    SizeType clusters = 0;
    while (i < all.size()) {
        // One row at a time: its peak sets the tolerance.
        SizeType end    = i;
        double row_peak = 0;
        while (end < all.size() && all[end].row == all[i].row) {
            row_peak = std::max(row_peak, static_cast<double>(all[end].c->snr));
            ++end;
        }
        const double tol = (frac * row_peak) + abs_tol;
        for (SizeType k = i; k < end;) {
            uint64_t last  = all[k].c->end_sample;
            double peak[2] = {0, 0};
            for (; k < end && all[k].c->begin_sample <= last; ++k) {
                last              = std::max(last, all[k].c->end_sample);
                peak[all[k].side] = std::max(
                    peak[all[k].side], static_cast<double>(all[k].c->snr));
            }
            ++clusters;
            INFO("cluster ending at " << last << ": " << peak[0] << " vs "
                                      << peak[1] << ", row peak " << row_peak);
            if (peak[0] > 0 && peak[1] > 0) {
                CHECK(std::abs(peak[0] - peak[1]) <= tol);
            } else {
                CHECK(std::max(peak[0], peak[1]) <= threshold + tol);
            }
        }
        i = end;
    }
    CHECK(clusters > 0);
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

// Direct float64 windows over each plane row's stream (rows in plane, then
// DM order), with the same peak bookkeeping as the engines. `table(row, t,
// f)` gives the threshold and inverse sigma of endpoint t (default: the
// planes' tables).
template <typename Table>
inline std::vector<RawCandidate>
direct_windows(std::span<const plans::BoxcarPlanePlan> planes,
               const std::vector<std::vector<float>>& rows,
               const BoxcarConfig& config,
               Table&& table) {
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
                c.staggered     = p.staggered;
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
                    // Trapezoid {1, 2, 1}: the box plus its core.
                    const SizeType w_edge =
                        config.use_trapezoid
                            ? algorithms::detail::trapezoid_edge_width(
                                  static_cast<uint32_t>(w))
                            : 0;
                    for (SizeType k = w_edge;
                         w_edge > 0 && k < w - w_edge && k <= t; ++k) {
                        sum += values[t - k];
                    }
                    const bool valid      = !config.skip_cold_prefix ||
                                            t >= p.warmup_samples[d] + w - 1;
                    const auto [thr, inv] = table(row_idx, t, f);
                    const bool above      = valid && sum >= thr;
                    const auto c = point(t, f, static_cast<float>(sum * inv));
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
direct_windows(std::span<const plans::BoxcarPlanePlan> planes,
               const std::vector<std::vector<float>>& rows,
               const BoxcarConfig& config) {
    // Flat row index -> (plane, DM) for the planes' own tables.
    std::vector<std::pair<SizeType, SizeType>> where;
    for (SizeType ip = 0; ip < planes.size(); ++ip) {
        for (SizeType d = 0; d < planes[ip].dm_grid.size(); ++d) {
            where.emplace_back(ip, d);
        }
    }
    return direct_windows(
        planes, rows, config, [&](SizeType row, SizeType, SizeType f) {
            const auto& p    = planes[where[row].first];
            const SizeType i = (where[row].second * p.widths.size()) + f;
            return std::pair{p.raw_thresh_lut[i], p.inv_sigma_lut[i]};
        });
}

// Materialize existing FDMT's actual intermediate planes, then evaluate
// windows by direct float64 summation. No new scorer/prefix helpers used.
// With a kill mask the FDMT sees the killed channels as zero and the windows
// use the masked search planes.
// The rows of `planes` over the stream from the materialized FDMT (one
// beam); block k's channels with block_kill[k][c] == 0 read as zero (an
// empty entry: none).
inline std::vector<std::vector<float>>
reference_rows(const plans::FDMTPlan& plan,
               bool smear,
               SizeType beams,
               std::span<const plans::BoxcarPlanePlan> planes,
               const std::vector<std::vector<float>>& input,
               const std::vector<std::vector<uint8_t>>& block_kill) {
    std::vector<std::vector<float>> rows;
    SizeType nrows = 0;
    for (const auto& p : planes) {
        nrows += beams * p.dm_grid.size();
    }
    rows.resize(nrows);
    FDMT fdmt(plan, smear, Exec::cpu(), beams, 0, false, plan.get_nsamps());
    std::vector<float> output(beams * plan.get_dmt_size());
    for (SizeType k = 0; k < input.size(); ++k) {
        auto block       = input[k];
        const auto& kill = block_kill[k];
        for (SizeType c = 0; c < kill.size(); ++c) {
            for (SizeType b = 0; b < beams && kill[c] == 0; ++b) {
                std::fill_n(block.begin() + static_cast<std::ptrdiff_t>(
                                                ((b * plan.get_nchans()) + c) *
                                                plan.get_nsamps()),
                            plan.get_nsamps(), 0.0F);
            }
        }
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
    return rows;
}

inline std::vector<RawCandidate>
reference(const plans::FDMTPlan& plan,
          const BoxcarConfig& config,
          bool smear,
          SizeType beams,
          const std::vector<std::vector<float>>& input,
          std::span<const uint8_t> kill = {}) {
    FDMTBoxcar metadata(plan, config, smear, Exec::cpu(), beams,
                        algorithms::kFDMTAutoFuse, 0,
                        algorithms::kFDMTDefaultMaxMemory, kill);
    const auto planes = metadata.get_search_plans();
    const std::vector<std::vector<uint8_t>> block_kill(
        input.size(), std::vector<uint8_t>(kill.begin(), kill.end()));
    return direct_windows(
        planes, reference_rows(plan, smear, beams, planes, input, block_kill),
        config);
}

/**
 * Per-call kill masks (FDMTBoxcar::execute(waterfall, kill_mask)): block k
 * runs with effective mask ctor AND masks[k] (an empty masks[k]: ctor
 * alone). Killed input reads as zero per block. Endpoint t of block k uses
 * the tables of that mask, except, when the mask changed from block k - 1
 * (k > 0), the first S_r + W - 1 endpoints of each row (S_r its support
 * with every channel active), which use the tables of the OR of both masks.
 * Tables come from constructor-masked instances (independent of the
 * engines' noise model); the warm-up from the constructor mask. One beam.
 */
struct MaskedReference {
    std::vector<std::vector<uint8_t>> effective;
    std::vector<RawCandidate> candidates;
};
inline MaskedReference
masked_reference(const plans::FDMTPlan& plan,
                 const BoxcarConfig& config,
                 bool smear,
                 const std::vector<std::vector<float>>& input,
                 std::span<const uint8_t> ctor,
                 const std::vector<std::vector<uint8_t>>& masks) {
    const SizeType nchans = plan.get_nchans();
    const SizeType ns     = plan.get_nsamps();
    const auto metadata   = [&](std::span<const uint8_t> kill) {
        return FDMTBoxcar(plan, config, smear, Exec::cpu(), 1,
                          algorithms::kFDMTAutoFuse, 0,
                          algorithms::kFDMTDefaultMaxMemory, kill);
    };
    MaskedReference out;
    for (SizeType k = 0; k < input.size(); ++k) {
        std::vector<uint8_t> e(nchans, 1);
        for (SizeType c = 0; c < nchans; ++c) {
            const bool on = (ctor.empty() || ctor[c] != 0) &&
                            (masks[k].empty() || masks[k][c] != 0);
            e[c]          = on ? 1 : 0;
        }
        out.effective.push_back(e);
    }
    // Flat row tables of a mask, cached.
    struct Tables {
        std::vector<float> thr, inv;
    };
    std::map<std::vector<uint8_t>, Tables> cache;
    const auto tables = [&](const std::vector<uint8_t>& mask) -> const Tables& {
        auto it = cache.find(mask);
        if (it == cache.end()) {
            Tables t;
            const auto m = metadata(mask);
            for (const auto& p : m.get_search_plans()) {
                t.thr.insert(t.thr.end(), p.raw_thresh_lut.begin(),
                             p.raw_thresh_lut.end());
                t.inv.insert(t.inv.end(), p.inv_sigma_lut.begin(),
                             p.inv_sigma_lut.end());
            }
            it = cache.emplace(mask, std::move(t)).first;
        }
        return it->second;
    };
    std::vector<uint64_t> support;
    {
        const auto all = metadata({});
        for (const auto& p : all.get_search_plans()) {
            support.insert(support.end(), p.warmup_samples.begin(),
                           p.warmup_samples.end());
        }
    }
    std::vector<const Tables*> steady;
    std::vector<const Tables*> transition;
    for (SizeType k = 0; k < input.size(); ++k) {
        steady.push_back(&tables(out.effective[k]));
        const Tables* tr = nullptr;
        if (k > 0 && out.effective[k] != out.effective[k - 1]) {
            std::vector<uint8_t> u(nchans);
            for (SizeType c = 0; c < nchans; ++c) {
                u[c] = out.effective[k][c] | out.effective[k - 1][c];
            }
            tr = &tables(u);
        }
        transition.push_back(tr);
    }
    const auto ctor_meta = metadata(ctor);
    const auto planes    = ctor_meta.get_search_plans();
    const SizeType nf    = planes.front().widths.size();
    const auto rows =
        reference_rows(plan, smear, 1, planes, input, out.effective);
    out.candidates = direct_windows(
        planes, rows, config, [&](SizeType row, SizeType t, SizeType f) {
            const SizeType k  = t / ns;
            const Tables* tab = steady[k];
            if (transition[k] != nullptr &&
                t < (k * ns) + support[row] + planes.front().widths[f] - 1) {
                tab = transition[k];
            }
            return std::pair{tab->thr[(row * nf) + f],
                             tab->inv[(row * nf) + f]};
        });
    return out;
}

inline std::vector<RawCandidate>
batched_reference(const plans::FDMTPlan& plan,
                  const BoxcarConfig& config,
                  bool smear,
                  SizeType beams,
                  const std::vector<std::vector<float>>& input,
                  std::span<const uint8_t> kill = {}) {
    std::vector<RawCandidate> expected;
    const SizeType size = plan.get_nchans() * plan.get_nsamps();
    for (SizeType b = 0; b < beams; ++b) {
        auto beam = input;
        for (SizeType k = 0; k < input.size(); ++k) {
            beam[k].assign(input[k].begin() + b * size,
                           input[k].begin() + (b + 1) * size);
        }
        auto candidates = reference(plan, config, smear, 1, beam, kill);
        for (auto& c : candidates) {
            c.beam_idx = static_cast<uint32_t>(b);
        }
        expected.insert(expected.end(), candidates.begin(), candidates.end());
    }
    return expected;
}

// FDMT-FFT rows from the existing transform (one beam), moved to input-stream
// coordinates (output sample j of the stream is input sample j - latency),
// then direct windows with FDMTFFTBoxcar's planes.
inline std::vector<RawCandidate>
fft_reference(const plans::FDMTPlan& plan,
              const BoxcarConfig& config,
              bool smear,
              bool frac,
              std::span<const uint8_t> kill,
              Exec exec,
              const std::vector<std::vector<float>>& input) {
    using algorithms::FDMTFFT;
    using algorithms::FDMTFFTBoxcar;
    FDMTFFTBoxcar metadata(plan, config, smear, Exec::cpu(), 1, frac, kill);
    const auto planes = metadata.get_search_plans();
    REQUIRE(planes.size() == 1);
    FDMTFFT fft(plan.get_f_min(), plan.get_f_max(), plan.get_nchans(),
                plan.get_nsamps(), plan.get_tsamp(), plan.get_dt_grid_final(),
                smear, "valid", exec, 1, frac, kill);
    const SizeType latency = fft.get_output_latency();
    REQUIRE(latency == metadata.get_output_latency());
    const SizeType ndm = planes.front().dm_grid.size();
    std::vector<std::vector<float>> rows(ndm);
    std::vector<float> output(plan.get_dmt_size());
    for (const auto& block : input) {
        fft.execute(block, output);
        for (SizeType d = 0; d < ndm; ++d) {
            const auto* row = output.data() + d * plan.get_nsamps();
            rows[d].insert(rows[d].end(), row, row + plan.get_nsamps());
        }
    }
    for (auto& row : rows) {
        row.erase(row.begin(),
                  row.begin() + static_cast<std::ptrdiff_t>(
                                    std::min(latency, row.size())));
    }
    return direct_windows(planes, rows, config);
}
} // namespace boxcar_test
