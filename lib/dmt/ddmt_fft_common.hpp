#pragma once

// Backend-neutral model of DDMTFFT, shared by the CPU and GPU engines: the
// exact delay model (double precision), the streaming geometry and the
// overlap-save segment length. No runtime headers: nvcc includes it too.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"

namespace dmt::algorithms::ddmt_fft {

/// @brief Shortest uniformly spaced run of DM trials the NUFFT is used for
/// (kAuto): below it one NUFFT costs about as much as the brute-force sum.
inline constexpr SizeType kMinNufftRun = 32;

/// @brief Trials [begin, begin + count) spaced exactly dm0 + i * ddm.
struct UniformRun {
    SizeType begin{};
    SizeType count{};
    double dm0{};
    double ddm{};
};

/**
 * @brief Exact fractional-delay model of one plan.
 *
 * Delay of trial d in channel c, in samples: tau = dm[d] * rate[c], the
 * same dispersion law and channel frequencies (f_min + c * df, referenced to
 * the highest channel) as DDMTPlan's tables, evaluated in double.
 *
 * The DM grid is split into maximal uniformly spaced runs (a linear grid is
 * one run; a DDplan-style piecewise-uniform grid, e.g. from
 * DDMTPlan::generate_levin_dm_grid_piecewise(), a few). Trials of a run are
 * stored exactly as dm0 + i * ddm (the plan's float values rounded to that
 * form), which is what the NUFFT path needs and what both methods then use,
 * so they agree.
 */
struct DelayModel {
    std::vector<double> dm;            // per trial, pc/cm^3
    std::vector<double> rate;          // per channel, samples per pc/cm^3
    std::vector<std::uint32_t> active; // unmasked channel indices
    std::vector<UniformRun> runs;      // cover dm in order
    bool uniform{false};               // one run covers the whole grid
    double dm0{0.0};                   // uniform: first trial
    double ddm{0.0};                   // uniform: spacing
    double max_tau{0.0};               // samples
    SizeType guard{0};
    SizeType max_delay{0}; // ceil(max tau) + guard (look-ahead)

    [[nodiscard]] double tau(SizeType d, SizeType c) const noexcept {
        return dm[d] * rate[c];
    }
    /// Look-behind + look-ahead a transformed segment spends on context.
    [[nodiscard]] SizeType context() const noexcept {
        return guard + max_delay;
    }

    DelayModel(const plans::DDMTPlan& plan, SizeType guard_samples)
        : guard(guard_samples) {
        const auto& pc     = plan.get_container();
        const auto nchans  = plan.get_nchans();
        const double f_min = plan.get_f_min();
        const double df    = (static_cast<double>(plan.get_f_max()) - f_min) /
                             static_cast<double>(nchans);
        const double f_ref = f_min + (df * static_cast<double>(nchans - 1));
        const double k     = static_cast<double>(kDispConst) /
                             static_cast<double>(plan.get_tsamp());
        rate.resize(nchans);
        for (SizeType c = 0; c < nchans; ++c) {
            const double f = f_min + (df * static_cast<double>(c));
            rate[c]        = k * ((1.0 / (f * f)) - (1.0 / (f_ref * f_ref)));
        }
        for (SizeType c = 0; c < nchans; ++c) {
            if (pc.kill_mask.empty() || pc.kill_mask[c] != 0) {
                active.push_back(static_cast<std::uint32_t>(c));
            }
        }
        dm.assign(pc.dm_arr.begin(), pc.dm_arr.end());
        for (const auto v : dm) {
            if (!utils::is_finite_bits(v) || v < 0.0) {
                throw std::invalid_argument(
                    "DDMTFFT: DM trials must be finite and >= 0");
            }
        }
        detect_runs();
        double max_dm = 0.0;
        for (const auto v : dm) {
            max_dm = std::max(max_dm, v);
        }
        double max_rate = 0.0;
        for (const auto c : active) {
            max_rate = std::max(max_rate, rate[c]);
        }
        max_tau   = max_dm * max_rate;
        max_delay = static_cast<SizeType>(std::ceil(max_tau)) + guard;
    }

    /// @brief Runs summed by the NUFFT: at least kMinNufftRun trials, or the
    /// whole grid when the NUFFT was asked for explicitly.
    [[nodiscard]] bool nufft_run(const UniformRun& r,
                                 bool explicit_nufft) const noexcept {
        return r.count >= kMinNufftRun ||
               (explicit_nufft && r.count == dm.size() && r.count > 0);
    }

private:
    // Float plan grids are dm0 + i * step rounded to float: allow that
    // rounding (a few ulp of the largest DM) plus 1e-3 of the spacing.
    [[nodiscard]] static double run_tol(double step, double a, double b) {
        return std::max(
            1.0E-3 * step,
            8.0 * static_cast<double>(std::numeric_limits<float>::epsilon()) *
                std::max(std::abs(a), std::abs(b)));
    }

    // True if dm[i..j] lies within tolerance of its endpoint line.
    [[nodiscard]] bool fits(SizeType i, SizeType j) const {
        const double step = (dm[j] - dm[i]) / static_cast<double>(j - i);
        if (!(step > 0.0)) {
            return false;
        }
        const double tol = run_tol(step, dm[i], dm[j]);
        for (SizeType q = i; q <= j; ++q) {
            const double ideal = dm[i] + (step * static_cast<double>(q - i));
            if (std::abs(dm[q] - ideal) > tol) {
                return false;
            }
        }
        return true;
    }

    // Greedy maximal runs: extend while the spacing stays within tolerance
    // of the run's first spacing, then shrink to the longest prefix that
    // fits the endpoint line (a slowly varying spacing, as in a continuous
    // Levin grid, chains many near-equal steps without being uniform).
    void detect_runs() {
        const auto n = dm.size();
        runs.clear();
        SizeType i = 0;
        while (i < n) {
            SizeType j = i;
            if (i + 1 < n && dm[i + 1] > dm[i]) {
                const double s0 = dm[i + 1] - dm[i];
                j               = i + 1;
                while (j + 1 < n) {
                    const double s = dm[j + 1] - dm[j];
                    if (std::abs(s - s0) >
                        2.0 * run_tol(s0, dm[i], dm[j + 1])) {
                        break;
                    }
                    ++j;
                }
                if (!fits(i, j)) {
                    SizeType lo = i + 1; // always fits (two points)
                    SizeType hi = j;     // does not fit
                    while (hi - lo > 1) {
                        const SizeType mid       = lo + ((hi - lo) / 2);
                        (fits(i, mid) ? lo : hi) = mid;
                    }
                    j = lo;
                }
            }
            UniformRun r{.begin = i, .count = j - i + 1, .dm0 = dm[i]};
            if (r.count > 1) {
                r.ddm = (dm[j] - dm[i]) / static_cast<double>(j - i);
                for (SizeType q = 0; q < r.count; ++q) {
                    dm[i + q] = r.dm0 + (r.ddm * static_cast<double>(q));
                }
            }
            runs.push_back(r);
            i = j + 1;
        }
        uniform = runs.size() == 1;
        if (uniform) {
            dm0 = runs[0].dm0;
            ddm = runs[0].ddm;
        }
    }
};

/**
 * @brief kAuto -> NUFFT when at least one run qualifies (nufft_run()), the
 * remaining trials by brute force; else brute force. An explicit kNUFFT
 * needs at least one qualifying run.
 */
inline DDMTFFTMethod resolve_method(DDMTFFTMethod requested,
                                    const DelayModel& model) {
    const bool explicit_nufft = requested == DDMTFFTMethod::kNUFFT;
    bool any                  = false;
    for (const auto& r : model.runs) {
        any = any || model.nufft_run(r, explicit_nufft);
    }
    if (explicit_nufft && !any) {
        throw std::invalid_argument(std::format(
            "DDMTFFT: method 'nufft' needs a uniformly spaced DM grid or "
            "uniform runs of at least {} trials (see "
            "DDMTPlan::generate_levin_dm_grid_piecewise)",
            kMinNufftRun));
    }
    if (requested != DDMTFFTMethod::kAuto) {
        return requested;
    }
    return any ? DDMTFFTMethod::kNUFFT : DDMTFFTMethod::kBrute;
}

/// @brief Block length whose transforms keep >= 80% of their samples as
/// output: at least 4 * context, rounded so that block + context is an
/// FFT-friendly length.
inline SizeType suggested_nsamps(SizeType context) {
    const auto c = std::max<SizeType>(context, 1);
    return utils::next_fft_size(5 * c) - c;
}

/// @brief Longest transform whose channel spectra (nchans * (N/2 + 1)
/// complex floats) fit in `max_spectra_bytes`, but at least 8 * context.
inline SizeType max_segment_length(SizeType context,
                                   SizeType nchans,
                                   SizeType max_spectra_bytes) {
    const SizeType cap =
        (max_spectra_bytes / (std::max<SizeType>(nchans, 1) * 8)) * 2;
    return std::max(cap, 8 * std::max<SizeType>(context, 512));
}

/**
 * @brief Overlap-save plan of one call: n_out output samples with `context`
 * samples of look-behind + look-ahead per transform. The channel-sum cost is
 * proportional to the transformed samples, so one transform is used when it
 * fits `max_len`, else the fewest balanced segments that do.
 */
struct Segmentation {
    SizeType n_fft{};
    SizeType hop{}; // output samples per segment (last may be shorter)
    SizeType nseg{};
};

inline Segmentation
plan_segments(SizeType n_out, SizeType context, SizeType max_len) {
    Segmentation s;
    s.nseg = 1;
    while (true) {
        const SizeType per = (n_out + s.nseg - 1) / s.nseg;
        s.n_fft            = utils::next_fft_size(per + context);
        if (s.n_fft <= max_len || per <= 1) {
            s.hop  = s.n_fft - context;
            s.nseg = (n_out + s.hop - 1) / s.hop;
            return s;
        }
        ++s.nseg;
    }
}

} // namespace dmt::algorithms::ddmt_fft
