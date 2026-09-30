#pragma once

// Host-side tables shared by the CohFDMT CPU and GPU engines.
//
// Chirp phases are exact 0.64 fixed-point turns (as in fourier_gpu.cuh): the
// in-channel dedispersion phase of coarse trial k at bin b of channel c is
//   frac(phi(c, b) * d_k) = base[c, b] + k * inc[c, b]  (mod 2^64),
// with d_k = d_0 + k * step, so every backend evaluates bit-identical phases
// with integer arithmetic only.

#include <cmath>
#include <cstdint>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::cfdmt {

/// frac(x) as a 0.64 fixed-point number of turns.
inline uint64_t fixed_turns(double x) noexcept {
    const double f  = x - std::floor(x); // [0, 1)
    const double hi = std::floor(std::ldexp(f, 32));
    const double lo = std::floor(std::ldexp(std::ldexp(f, 32) - hi, 32));
    return (static_cast<uint64_t>(hi) << 32U) + static_cast<uint64_t>(lo);
}

/// Signed turn in [-1/2, 1/2) of a 0.64 fixed-point phase.
inline float turn_of(uint64_t s) noexcept {
    return static_cast<float>(static_cast<int64_t>(s)) * 5.42101086242752217e-20F;
}

struct ChirpPhases {
    std::vector<uint64_t> base; // (nchans, mbin)
    std::vector<uint64_t> inc;  // (nchans, mbin)
};

/**
 * Dedispersion phase (turns) of every channel bin at the first coarse trial
 * and its increment per trial. Bin b of a channel is the frequency offset
 * f' = (b - mbin / 2) * bw_sub / nbin from the channel centre f_c; the
 * coherent dedispersion filter (Hankins & Rickett) is
 *   H(f') = exp(-2 pi i * 1e6 * K * d * f'^2 / (f_c^2 (f_c + f'))).
 */
inline ChirpPhases make_chirp_phases(const plans::CohFDMTPlan& plan) {
    const auto nchans    = plan.get_nchans();
    const auto mbin      = plan.get_mbin();
    const double f_min   = plan.get_f_min();
    const double bw_chan = static_cast<double>(plan.get_bw_sub()) /
                           static_cast<double>(plan.get_n_p());
    const double bw_bin =
        static_cast<double>(plan.get_bw_sub()) /
        static_cast<double>(plan.get_nbin());
    const double step = plan.get_dm_step_coh();
    const double d0   = static_cast<double>(plan.get_dm_min()) + (0.5 * step);
    ChirpPhases ph;
    ph.base.resize(nchans * mbin);
    ph.inc.resize(nchans * mbin);
    for (SizeType c = 0; c < nchans; ++c) {
        const double f_c = f_min + ((static_cast<double>(c) + 0.5) * bw_chan);
        for (SizeType b = 0; b < mbin; ++b) {
            const double fp =
                (static_cast<double>(b) - (static_cast<double>(mbin) / 2.0)) *
                bw_bin;
            const double phi = -1.0E6 * static_cast<double>(kDispConst) * fp *
                               fp / (f_c * f_c * (f_c + fp));
            ph.base[(c * mbin) + b] = fixed_turns(phi * d0);
            ph.inc[(c * mbin) + b]  = fixed_turns(phi * step);
        }
    }
    return ph;
}

} // namespace dmt::cfdmt
