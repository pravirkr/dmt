#pragma once

// Backend-neutral transform geometry of FDMTFFT, shared by the CPU and GPU
// engines (nvcc includes it: no runtime headers).
//
// Fractional delays (the default): each merge shifts its head by a real-valued
// delay as a phase ramp, which is band-limited (periodic sinc) interpolation.
// The delay is not simply the unrounded dt * phi of the merge: the children are
// still built on integer delay grids, and in the integer tree their rounding is
// complementary to the merge's (a child's trial and the merge shift use the
// same rounded value), which keeps every channel's total shift consistent.
// Rounding only the merge shift away breaks that and loses sensitivity.
// Instead every node tracks the mean timing error of its channels (path
// shift minus true delay, at the node's own DM) and the merge shift aligns
// the head group's mean error with the tail group's: a least-squares
// alignment that the integer tree approximates by rounding
// (fractional_delays()).
//
// Kernel tails need context on both sides of every sample: an output needs
// the inputs from `support` (the longest total shift plus `guard`) samples
// before it to `guard` samples after it. In valid mode the output therefore
// lags the input by `guard` samples (output_latency()): output sample o of
// a block is stream time block_start - guard + o, and the newest `guard`
// input samples are its look-ahead. Every output then sees the complete
// interpolation kernel, so streaming equals one long call. Full mode sees
// zeros outside the block (as integer full mode does) and roll wraps.
//
// Integer delays (fractional_delays = false, FDMT-equivalence tests only):
// the plan's FFT length and overlap. Every tree shift is a whole number of
// samples, i.e. an exact circular shift, and the output equals FDMT's.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms::fdmt_fft {

/// Guard samples of fractional-delay FDMTFFT (see file comment).
inline constexpr SizeType kFractionalGuard = 64;

/// @brief Real-valued merge shifts of the fractional tree, indexed like
/// FDMTPlanContainer::coordinates_sum, and the resulting support.
struct FractionalDelays {
    std::vector<std::vector<double>> shift; // [level][sum coord]
    double support{0.0}; // longest total shift of any channel path
};

/**
 * @brief Least-squares merge shifts (see the file comment).
 *
 * With g(f) = 1/f^2 the dispersion delay between two frequencies is DM
 * times a difference of g. A node over [f_start, f_end) with trial dt
 * stands for DMg = dt / (g(f_start) - g(f_end)); for each node we keep, over
 * its channels, the mean path shift S and the mean g-distance K from the
 * channel's reference frequency to f_start, so its mean timing error at a
 * DMg is S - K * DMg. Level 0 (channel c, trial dt0) without box smearing
 * places the channel's samples at its upper edge and shifts by dt0 (S = dt0,
 * K = g(f_c) - g(f_c+1)); with box smearing the box spans shifts 0..dt0 and
 * the smeared pulse the whole channel, both counted by their centroids
 * (S = dt0/2, K half the channel's g-width). A merge of tail T and head H
 * (boundary f_mid) at the parent's DMg shifts the head by
 *   s = (g(f_start) - g(f_mid)) DMg + E_T - E_H,
 * the true delay across the tail band corrected for the children's mean
 * errors. Negative-DM merges (whose subtrees never feed positive ones) keep
 * their integer shifts.
 */
inline FractionalDelays fractional_delays(const plans::FDMTPlan& plan,
                                          bool box) {
    const auto& pc = plan.get_container();
    const auto g   = [](double f) { return 1.0 / (f * f); };
    struct Stat {
        double n{0.0};
        double s{0.0}; // mean path shift
        double k{0.0}; // mean g-distance to the node's lower edge
        double span{0.0};
    };
    std::vector<Stat> prev(pc.state_shape[0].ncoords);
    for (const auto& gr : pc.grids[0]) {
        const double w = g(gr.f_start) - g(gr.f_end);
        for (SizeType i = 0; i < gr.ndt; ++i) {
            const double dt0 = std::abs(static_cast<double>(gr.dt_grid[i]));
            prev[gr.coord_offset + i] = box ? Stat{1.0, 0.5 * dt0, 0.5 * w, dt0}
                                            : Stat{1.0, dt0, w, dt0};
        }
    }
    FractionalDelays out;
    out.shift.resize(plan.get_niters() + 1);
    for (SizeType l = 1; l <= plan.get_niters(); ++l) {
        std::vector<Stat> cur(pc.state_shape[l].ncoords);
        const auto& sums = pc.coordinates_sum[l];
        out.shift[l].resize(sums.size());
        for (SizeType i = 0; i < sums.size(); ++i) {
            const auto& c  = sums[i];
            const auto& t  = prev[c.tail_buf_offset / c.tail_nsamps];
            const auto& h  = prev[c.head_buf_offset / c.head_nsamps];
            const auto& gp = pc.grids[l][c.i_sub];
            const auto dt  = gp.dt_grid[c.i_dt];
            double shift   = static_cast<double>(c.delay);
            double dk      = 0.0; // g-distance tail edge -> head edge
            if (dt > 0) {
                const double fs   = gp.f_start;
                const double fmid = pc.grids[l - 1][2 * c.i_sub].f_end;
                const double dmg =
                    static_cast<double>(dt) / (g(fs) - g(gp.f_end));
                dk              = g(fs) - g(fmid);
                const double et = t.s - (t.k * dmg);
                const double eh = h.s - (h.k * dmg);
                shift           = std::max(0.0, (dk * dmg) + et - eh);
            }
            out.shift[l][i]              = shift;
            const double n               = t.n + h.n;
            cur[c.buf_offset / c.nsamps] = Stat{
                n,
                ((t.n * t.s) + (h.n * (h.s + shift))) / n,
                ((t.n * t.k) + (h.n * (h.k + dk))) / n,
                std::max(t.span, shift + h.span),
            };
        }
        for (const auto& c : pc.coordinates_copy[l]) {
            cur[c.buf_offset / c.nsamps] =
                prev[c.tail_buf_offset / c.tail_nsamps];
        }
        prev = std::move(cur);
    }
    for (const auto& st : prev) {
        out.support = std::max(out.support, st.span);
    }
    return out;
}

/// Transform geometry of one engine.
struct Geometry {
    SizeType n_fft{};   // single-transform length
    SizeType overlap{}; // valid-mode history length (0 otherwise)
    SizeType skip{};    // transform sample of output sample 0
    SizeType support{}; // context an output needs behind it (look-behind)
    SizeType guard{};   // context an output needs after it (look-ahead)
    SizeType latency{}; // valid mode: output lag behind the input (samples)
};

inline Geometry make_geometry(const plans::FDMTPlan& plan,
                              FDMTMode mode,
                              bool fractional,
                              bool box) {
    Geometry g;
    const auto nsamps = plan.get_nsamps();
    if (!fractional || mode == FDMTMode::kRoll) {
        // Roll is cyclic by definition: fractional shifts wrap within the
        // block, exactly like the integer ones.
        g.n_fft   = plan.get_fft_size();
        g.overlap = mode == FDMTMode::kValid ? plan.get_fft_overlap() : 0;
        g.skip    = g.overlap;
        g.support = plan.get_fft_support();
        g.guard   = 0;
        return g;
    }
    g.guard = kFractionalGuard;
    g.support =
        static_cast<SizeType>(std::ceil(fractional_delays(plan, box).support)) +
        g.guard;
    if (mode == FDMTMode::kValid) {
        // Window [history | block]: output o sits at skip + o, with support
        // samples behind it and guard ahead (the block's newest samples).
        g.latency = g.guard;
        g.overlap = std::max(plan.get_fft_overlap(), g.support) + g.guard;
        g.skip    = g.overlap - g.latency;
        g.n_fft   = utils::next_fft_size(nsamps + g.overlap);
    } else { // full: window [block | zeros]
        g.overlap = 0;
        g.skip    = 0;
        g.n_fft   = utils::next_fft_size(
            std::max(nsamps + g.support, plan.get_dmt_nsamps() + g.guard));
    }
    return g;
}

} // namespace dmt::algorithms::fdmt_fft
