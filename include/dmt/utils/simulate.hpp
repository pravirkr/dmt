#pragma once

/**
 * @file simulate.hpp
 * @brief Utilities for generating synthetic dispersed astronomical pulses (FRBs
 * / pulsars).
 */

#include <cstdint>
#include <span>
#include <tuple>
#include <vector>

#include "dmt/common/baseband.hpp"
#include "dmt/common/types.hpp"

namespace dmt::utils {

/**
 * @brief Injects a noise-free dispersed top-hat or fractional-delay pulse into
 * a zero waterfall.
 *
 * Models physical intra-channel dispersion integration between the top and
 * bottom edges of each channel:
 * @f[
 * \Delta t(\nu) = \Delta t_{\text{total}} \cdot \frac{f_{\text{min}}^{-2} -
 * \nu_{\text{bot}}^{-2}}{f_{\text{min}}^{-2} - f_{\text{max}}^{-2}}
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

/**
 * @brief A dispersed pulse injected by simulate_baseband().
 */
struct BasebandPulse {
    /// Dispersion measure in pc cm^-3.
    double dm{0.0};
    /// Arrival time at infinite frequency, seconds after sample 0.
    double t_arrival{0.0};
    /// Energy per polarisation and subband (sum of |v|^2 over time).
    double fluence{1.0};
    /// Gaussian width (sigma) in seconds of the band-limited pulse within
    /// each subband; 0 = impulse.
    double width{0.0};
};

/**
 * @brief Simulates complex dual-polarisation baseband of @p nsub critically
 * sampled subbands (upper sideband, DC at each subband centre) holding
 * dispersed pulses in complex Gaussian noise.
 *
 * Each pulse is built in the Fourier domain of the whole stream (one
 * transform of length @p nsamps per subband, i.e. circular in time) with the
 * exact cold-plasma phase exp(2 pi i K DM / f) at every RF frequency f, so
 * its arrival time at f is t_arrival + K * DM / f^2 with K = kDispConst.
 *
 * @param f_center Centre of the whole band in MHz.
 * @param bw_sub Subband bandwidth in MHz (sampling interval 1 / bw_sub).
 * @param nsub Number of subbands.
 * @param nsamps Samples per subband.
 * @param pulses Pulses to inject.
 * @param noise_sigma Noise standard deviation per real component (0 = none).
 * @param seed Noise seed.
 * @param nthreads OpenMP threads for the transforms.
 * @return (pol=2, nsub, nsamps) complex voltages.
 */
std::vector<ComplexType> simulate_baseband(float f_center,
                                           float bw_sub,
                                           SizeType nsub,
                                           SizeType nsamps,
                                           std::span<const BasebandPulse> pulses,
                                           float noise_sigma = 0.0F,
                                           uint64_t seed     = 42,
                                           int nthreads      = 1);

/**
 * @brief Quantises (pol, nsub, nsamps) complex voltages into a baseband
 * block of @p format, for subbands [@p sub_begin, @p sub_begin + @p
 * sub_count) and samples [@p t_begin, @p t_begin + @p t_count).
 *
 * Each component v becomes round(v * scale) clipped to the format's range
 * (2-bit: the nearest of format.levels_2bit to v * scale).
 */
std::vector<uint8_t> pack_baseband(std::span<const ComplexType> voltages,
                                   SizeType nsub,
                                   SizeType nsamps,
                                   const BasebandFormat& format,
                                   float scale,
                                   SizeType sub_begin = 0,
                                   SizeType sub_count = 0,
                                   SizeType t_begin   = 0,
                                   SizeType t_count   = 0);

} // namespace dmt::utils
