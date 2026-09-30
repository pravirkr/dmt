#pragma once

/**
 * @file simd_math.hpp
 * @brief Vectorisable phase helpers for the Fourier-domain CPU engines.
 *
 * Phases are carried in turns (1 turn = 2 pi rad): the large-argument part
 * of the reduction happens exactly in double (frac of a turn count), the
 * rest in float on a quarter-turn grid, so sin/cos come out correctly
 * rounded to about 1 ulp for any delay and frequency bin. All functions are
 * branch-free and written for `#pragma omp simd` loops, so they vectorise on
 * every ISA the compiler targets (AVX-512, AVX2, NEON, ...).
 */

#include <cmath>
#include <cstdint>

namespace dmt::simd {

/// Fractional part of a turn count, in [-0.5, 0.5) turns.
#pragma omp declare simd
inline double wrap_turns(double t) noexcept { return t - std::nearbyint(t); }

/**
 * @brief sin and cos of 2 pi u for |u| <= 0.5 turns (wider arguments work
 * but lose accuracy linearly).
 *
 * u is split into a quadrant q (nearest quarter turn) and a remainder r in
 * [-1/8, 1/8] turns; r * pi/2 * 4 lies in [-pi/4, pi/4] where the Cephes
 * minimax polynomials of sinf/cosf are accurate to ~1 ulp.
 */
#pragma omp declare simd
inline void sincos_turns(float u, float& s, float& c) noexcept {
    const float x  = 4.0F * u;                           // quarter turns
    const float qf = std::nearbyint(x);                  // quadrant
    const float r  = (x - qf) * 1.57079632679489661923F; // [-pi/4, pi/4]
    const float r2 = r * r;
    const float sp =
        r + (r * r2 *
             (-1.6666654611e-1F +
              (r2 * (8.3321608736e-3F + (r2 * -1.9515295891e-4F)))));
    const float cp =
        1.0F - (0.5F * r2) +
        (r2 * r2 *
         (4.166664568298827e-2F +
          (r2 * (-1.388731625493765e-3F + (r2 * 2.443315711809948e-5F)))));
    const auto q   = static_cast<std::int32_t>(qf) & 3;
    const bool swp = (q & 1) != 0;
    const float sv = swp ? cp : sp;
    const float cv = swp ? sp : cp;
    s              = ((q & 2) != 0) ? -sv : sv;
    c              = ((q == 1) || (q == 2)) ? -cv : cv;
}

} // namespace dmt::simd
