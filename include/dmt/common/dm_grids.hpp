#pragma once

#include <vector>

#include "dmt/common/types.hpp"

/**
 * @file dm_grids.hpp
 * @brief Trial DM and delay grids for searches (convenience generators).
 *
 * Every engine takes any DM or delay grid; these build common ones from the
 * observation and a sensitivity target. The rules are approximations of the
 * S/N a pulse keeps between two trials; see the pipeline guide's DM-grid
 * section for their limits (very wide bands, coherently dedispersed data).
 */

namespace dmt::plans {

/// How generate_dm_grid() spaces the trials.
enum class DMGridMethod {
    /// dmt's rule (default): the pulse's effective width (intrinsic,
    /// sampling, intra-channel smearing at the bottom of the band, at
    /// |DM - chan_dm|) may grow by at most tol = 1 / (1 - max_snr_loss)
    /// under the half-step DM error, which misaligns it by the band's exact
    /// dispersion sweep. Supports negative DMs and coherently dedispersed
    /// data (chan_dm); never collapses at high DM. On very wide bands its
    /// single smearing frequency makes it lose more than max_snr_loss at
    /// high DM (pipeline guide, DM grids).
    kSnrLoss,
    /// Levin (2012), the classic rule as in Heimdall: broadening bounded by
    /// tol, smearing and sweep from the band centre (straight-line sweep,
    /// so trials are too sparse on very wide bands). Non-negative DMs; no
    /// chan_dm.
    kLevin,
    /// kLevin made piecewise uniform (DDplan-style): uniform segments split
    /// where the Levin step doubles, each at its smallest Levin step and at
    /// least min_run trials long. The grid for DDMTFFT's NUFFT path.
    kLevinPiecewise,
};

/// Observation, pulse and rule of a DM grid.
struct DMGridConfig {
    double f_min{};        ///< Bottom edge of the band (MHz).
    double f_max{};        ///< Top edge of the band (MHz).
    SizeType nchans{};     ///< Number of channels.
    double tsamp{};        ///< Sampling interval (s).
    double pulse_width{0}; ///< Intrinsic pulse width (s).
    /// kSnrLoss: DM the channels were coherently dedispersed to
    /// (pc cm^-3); 0 for ordinary (incoherent) filterbank data. The
    /// intra-channel smearing grows with |DM - chan_dm|, so trials get
    /// denser near chan_dm. The Levin rules require 0.
    double chan_dm{0};
    DMGridMethod method{DMGridMethod::kSnrLoss};
    /// kSnrLoss: largest fractional S/N loss between trials, in (0, 1);
    /// ignored when tol is set.
    double max_snr_loss{0.05};
    /// Broadening tolerance (> 1), the knob of every rule; 0 takes the
    /// rule's default: 1.25 for the Levin rules, 1 / (1 - max_snr_loss)
    /// for kSnrLoss.
    double tol{0};
    /// kLevinPiecewise: shortest uniform segment.
    SizeType min_run{32};
};

/**
 * @brief Trial DMs from @p dm_min to @p dm_max (pc cm^-3), ascending.
 *
 * The grid starts at dm_min and ends at dm_max (kSnrLoss) or at the first
 * trial at or past it (Levin rules). Throws std::invalid_argument on
 * invalid input (non-finite values, an empty band, negative DMs with a
 * Levin rule, ...).
 */
[[nodiscard]] std::vector<double>
generate_dm_grid(const DMGridConfig& config, double dm_min, double dm_max);

/**
 * @brief Integer delay trials (samples across the band) from @p dt_min to
 * @p dt_max, for FDMTPlan's delay grid: generate_dm_grid() on the DM range
 * of those delays, converted to delays, rounded (half to even) and
 * deduplicated, with both ends kept.
 */
[[nodiscard]] std::vector<IndexType>
generate_dt_grid(const DMGridConfig& config, double dt_min, double dt_max);

} // namespace dmt::plans
