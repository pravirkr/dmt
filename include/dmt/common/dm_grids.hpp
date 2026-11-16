#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

/**
 * @file dm_grids.hpp
 * @brief Trial DM grids for searches (convenience generators).
 *
 * Every engine takes any DM grid (FDMTPlan rounds DMs to integer delays);
 * these build common ones from the observation and a sensitivity target:
 *
 * - generate_dm_grid(): dmt's rule. Trials are spaced so a pulse halfway
 *   between two keeps 1 - max_snr_loss of its S/N after the boxcar search,
 *   modelled over every channel (dispersion misalignment, intra-channel
 *   smearing and FDMT box smearing at |DM - chan_dm|, sampling, the
 *   boxcar widths and the noise they add). Negative DMs, coherently
 *   dedispersed data (chan_dm) and dead channels (kill_mask).
 * - generate_levin_dm_grid(): Levin (2012), the classic rule as in
 *   Heimdall, kept as it is (the benchmark grids).
 *
 * Both can be made piecewise uniform for DDMTFFT's NUFFT path. See the
 * pipeline guide's DM-grid page for choosing one.
 */

namespace dmt::plans {

/// Observation, pulse and target of a DM grid.
struct DMGridConfig {
    double f_min{};        ///< Bottom edge of the band (MHz).
    double f_max{};        ///< Top edge of the band (MHz).
    SizeType nchans{};     ///< Number of channels.
    double tsamp{};        ///< Sampling interval (s).
    double pulse_width{0}; ///< Intrinsic pulse width (s).
    /// DM the channels were coherently dedispersed to (pc cm^-3); 0 for
    /// ordinary filterbank data. Intra-channel smearing grows with
    /// |DM - chan_dm|, so generate_dm_grid() gets denser near chan_dm.
    /// generate_levin_dm_grid() requires 0.
    double chan_dm{0};
    /// generate_dm_grid(): largest fractional S/N loss halfway between two
    /// trials, in (0, 1); ignored when tol is set.
    double max_snr_loss{0.05};
    /// Broadening tolerance (> 1), or 0 for the default. generate_dm_grid():
    /// max_snr_loss = 1 - 1 / tol when set. generate_levin_dm_grid(): the
    /// Levin tolerance, 1.25 when 0.
    double tol{0};
    /// generate_dm_grid() and calculate_snr_loss(): channels in use
    /// (nonzero = keep, 0 = dead), nchans long; empty keeps every channel.
    /// generate_levin_dm_grid() requires it empty.
    std::vector<uint8_t> kill_mask;
    /// Make the grid piecewise uniform for DDMTFFT: uniform segments split
    /// where the step halves or doubles, each at the smallest step inside
    /// it and at least min_run trials long (never coarser than the
    /// continuous grid).
    bool piecewise{false};
    /// Shortest uniform segment of a piecewise grid.
    SizeType min_run{32};
};

/**
 * @brief Trial DMs from @p dm_min to @p dm_max (pc cm^-3), ascending, by
 * dmt's S/N-loss rule.
 *
 * The step at each DM keeps a pulse halfway between two trials at
 * 1 - max_snr_loss of the S/N it would have on a trial (pulse_width = 0:
 * only the sampling time broadens it). The grid starts at dm_min and ends
 * at dm_max
 * (a piecewise grid at dm_max or just past it). Negative DMs mirror
 * positive ones when chan_dm is 0. Throws std::invalid_argument on invalid
 * input (non-finite values, an empty band, a kill_mask of the wrong length
 * or with every channel dead, ...).
 */
[[nodiscard]] std::vector<double>
generate_dm_grid(const DMGridConfig& config, double dm_min, double dm_max);

/**
 * @brief Trial DMs from @p dm_min to @p dm_max by the Levin (2012) rule,
 * computed in float as DDMTPlan::generate_levin_dm_grid() (piecewise:
 * DDMTPlan::generate_levin_dm_grid_piecewise()), which return the same
 * values.
 *
 * Smearing and sweep from the band centre (a straight-line sweep, so too
 * sparse on very wide bands), broadening bounded by tol. Ends at the first
 * trial at or past dm_max. Requires dm_min >= 0, chan_dm == 0 and an empty
 * kill_mask; max_snr_loss is not used.
 */
[[nodiscard]] std::vector<double> generate_levin_dm_grid(
    const DMGridConfig& config, double dm_min, double dm_max);

/**
 * @brief Expected fractional S/N loss of a pulse at each DM in @p dm_test
 * searched with the trials @p trial_dms (sorted ascending), on the model of
 * generate_dm_grid(): 1 - S/N at the nearest trial / S/N on a trial.
 *
 * max_snr_loss, tol, piecewise and min_run are not used. A grid from
 * generate_dm_grid() keeps every DM between dm_min and dm_max at or below
 * max_snr_loss (up to the interpolation of its step).
 */
[[nodiscard]] std::vector<double>
calculate_snr_loss(const DMGridConfig& config,
                   std::span<const double> trial_dms,
                   std::span<const double> dm_test);

} // namespace dmt::plans
