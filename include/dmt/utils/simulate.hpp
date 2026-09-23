#pragma once

/**
 * @file simulate.hpp
 * @brief Utilities for generating synthetic dispersed astronomical pulses (FRBs / pulsars).
 */

#include <tuple>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::utils {

/**
 * @brief Injects a noise-free dispersed top-hat or fractional-delay pulse into a zero waterfall.
 *
 * Models physical intra-channel dispersion integration between the top and bottom edges
 * of each channel:
 * @f[
 * \Delta t(\nu) = \Delta t_{\text{total}} \cdot \frac{f_{\text{min}}^{-2} - \nu_{\text{bot}}^{-2}}{f_{\text{min}}^{-2} - f_{\text{max}}^{-2}}
 * @f]
 *
 * @param nchans Number of frequency channels.
 * @param nsamps Number of time samples.
 * @param f_min Bottom edge frequency in MHz.
 * @param f_max Top edge frequency in MHz.
 * @param dt Total dispersive delay across the band in time samples.
 * @param pulse_toa Pulse time of arrival at the lowest channel in samples.
 * @param amplitude Peak pulse amplitude (default: 1.0).
 * @return Tuple of:
 * - Flattened float32 waterfall array of length nchans * nsamps.
 * - Number of samples that received non-zero dispersed energy.
 */
std::tuple<std::vector<float>, SizeType>
generate_pure_frb(SizeType nchans,
                  SizeType nsamps,
                  float f_min,
                  float f_max,
                  SizeType dt,
                  float pulse_toa,
                  float amplitude = 1.0F);

} // namespace dmt::utils
