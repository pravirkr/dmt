"""Optimal sparse DM and dt grid generation for FDMT and DDMT.

Provides physically motivated non-uniform trial grids that guarantee a maximum
fractional sensitivity loss (e.g. <= 5%) across the entire DM search range
while minimizing redundant trials.

Algorithms supported:

1. ``method="snr_loss"`` (Default, recommended for FDMT):
   Evaluates intra-channel dispersion smearing at the band edge (f_min, the
   worst-case channel)::

       t_chan(DM) = 2 * K_DM * DM * df / f_min^3

   The effective pulse width including smearing and sampling interval is::

       W_eff(DM) = sqrt(W_pulse^2 + t_samp^2 + t_chan(DM)^2)

   The allowed delay mismatch for a maximum fractional S/N loss eta is::

       dt_step(DM) <= max(1.0, 2 * eta * W_eff(DM))

   where eta = sqrt(1 / (1 - max_snr_loss)^2 - 1).
   This method is strictly monotonic, never collapses at high DM, and
   supports negative and symmetric DM ranges.

2. ``method="levin"`` (Classical Lina Levin 2012 pulse-broadening formulation):
   Evaluates smearing at the center frequency f_center and bounds pulse broadening
   by a tolerance factor tol > 1.0 (typically 1.15 to 1.25). Calls into C++
   ``DDMTPlan.generate_levin_dm_grid``. Note that the quadratic equation can
   collapse at high DM when the discriminant becomes negative.
"""

from __future__ import annotations

import numpy as np

from .libdmt import DDMTPlan

# Dispersion constant (MHz^2 * s / (pc * cm^-3)) - Lorimer & Kramer (2005)
DM_CONSTANT = 4.148808e3


def _generate_positive_dm_grid(
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    dm_max: float,
    dm_min: float = 0.0,
    pulse_width: float = 1.0,
    max_snr_loss: float = 0.05,
) -> np.ndarray:
    """Generate a non-uniform DM grid for non-negative DM bounds [dm_min, dm_max]."""
    if dm_max < dm_min:
        msg = f"dm_max ({dm_max}) must be >= dm_min ({dm_min})"
        raise ValueError(msg)
    if dm_max == dm_min:
        return np.array([float(dm_min)], dtype=np.float64)

    df_chan = (f_max - f_min) / nchans
    k_disp = DM_CONSTANT * (f_min**-2 - f_max**-2) / tsamp
    # Fractional tolerance parameter
    eta = np.sqrt(1.0 / (1.0 - max_snr_loss) ** 2 - 1.0)

    dms = [float(dm_min)]
    cur_dm = float(dm_min)
    while cur_dm < dm_max:
        # Intra-channel smearing at bottom of the band (worst-case channel)
        t_chan = (2.0 * DM_CONSTANT * cur_dm * df_chan / (f_min**3)) / tsamp
        w_eff = np.sqrt(pulse_width**2 + 1.0 + t_chan**2)

        # Allow at most 2 * eta * w_eff delay mismatch across band
        dt_step = max(1.0, 2.0 * eta * w_eff)
        dm_step = dt_step / k_disp

        cur_dm += dm_step
        if cur_dm >= dm_max:
            dms.append(float(dm_max))
            break
        dms.append(cur_dm)

    return np.array(dms, dtype=np.float64)


def generate_optimal_dm_grid(
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    dm_max: float,
    *,
    dm_min: float = 0.0,
    pulse_width: float = 1.0,
    max_snr_loss: float = 0.05,
    method: str = "snr_loss",
    tol: float | None = None,
) -> np.ndarray:
    """Generate an optimal non-uniform DM trial grid (in pc cm^-3).

    Parameters
    ----------
    f_min : float
        Bottom edge of frequency band in MHz.
    f_max : float
        Top edge of frequency band in MHz.
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling interval in seconds.
    dm_max : float
        Maximum DM trial (pc cm^-3).
    dm_min : float, default=0.0
        Minimum DM trial (pc cm^-3). Supports negative values when
        ``method="snr_loss"``.
    pulse_width : float, default=1.0
        Intrinsic pulse width in units of sampling time (tsamp).
    max_snr_loss : float, default=0.05
        Maximum allowable fractional S/N loss (e.g. 0.05 = 5%). Used by
        ``method="snr_loss"``.
    method : str, default="snr_loss"
        Grid generation algorithm:
        - "snr_loss": S/N-loss-bounded, non-collapsing, edge-frequency smearing.
        - "levin": Classical Lina Levin (2012) broadening tolerance at band center.
    tol : float, optional
        Pulse broadening factor (tol > 1.0, e.g. 1.25) for ``method="levin"``.
        If omitted when using "levin", defaults to 1 / (1 - max_snr_loss).

    Returns
    -------
    np.ndarray
        Sorted 1-D array of trial DMs (float64).
    """
    if method == "levin":
        if dm_min < 0.0:
            msg = f"Levin method does not support negative dm_min ({dm_min})"
            raise ValueError(msg)
        effective_tol = (
            tol
            if tol is not None
            else (1.0 / (1.0 - max_snr_loss) if max_snr_loss < 1.0 else 1.25)
        )
        if effective_tol <= 1.0:
            msg = f"Levin tol must be > 1.0, got {effective_tol}"
            raise ValueError(msg)
        return np.asarray(
            DDMTPlan.generate_levin_dm_grid(
                dm_min,
                dm_max,
                tsamp,
                pulse_width,
                f_min,
                f_max,
                nchans,
                effective_tol,
            ),
            dtype=np.float64,
        )

    if method != "snr_loss":
        msg = f"Unknown grid generation method '{method}'; choose 'snr_loss' or 'levin'"
        raise ValueError(msg)

    if not 0.0 < max_snr_loss < 1.0:
        msg = f"max_snr_loss must be between 0 and 1, got {max_snr_loss}"
        raise ValueError(msg)
    if f_min <= 0 or f_max <= f_min:
        msg = f"Invalid frequency band: f_min={f_min}, f_max={f_max}"
        raise ValueError(msg)
    if nchans <= 0 or tsamp <= 0:
        msg = f"Invalid nchans={nchans} or tsamp={tsamp}"
        raise ValueError(msg)

    # Handle symmetric / negative ranges
    if dm_min < 0.0 and dm_max > 0.0:
        neg_half = _generate_positive_dm_grid(
            f_min,
            f_max,
            nchans,
            tsamp,
            -dm_min,
            0.0,
            pulse_width,
            max_snr_loss,
        )
        pos_half = _generate_positive_dm_grid(
            f_min,
            f_max,
            nchans,
            tsamp,
            dm_max,
            0.0,
            pulse_width,
            max_snr_loss,
        )
        # Combine: -neg_half (reversed, dropping the duplicated 0.0) + pos_half
        return np.concatenate([-neg_half[:0:-1], pos_half])

    if dm_max <= 0.0:
        pos_grid = _generate_positive_dm_grid(
            f_min,
            f_max,
            nchans,
            tsamp,
            -dm_min,
            -dm_max,
            pulse_width,
            max_snr_loss,
        )
        return -pos_grid[::-1]

    return _generate_positive_dm_grid(
        f_min,
        f_max,
        nchans,
        tsamp,
        dm_max,
        dm_min,
        pulse_width,
        max_snr_loss,
    )


def generate_optimal_dt_grid(  # noqa: PLR0913
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    dt_max: float,
    *,
    dt_min: float = 0.0,
    pulse_width: float = 1.0,
    max_snr_loss: float = 0.05,
    integer_grid: bool = True,
    method: str = "snr_loss",
    tol: float | None = None,
) -> np.ndarray:
    """Generate an optimal non-uniform delay grid (in samples) for FDMT.

    Parameters
    ----------
    f_min : float
        Bottom edge of frequency band in MHz.
    f_max : float
        Top edge of frequency band in MHz.
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling interval in seconds.
    dt_max : float
        Maximum total dispersive delay across the band in samples.
    dt_min : float, default=0.0
        Minimum dispersive delay in samples. Supports negative values.
    pulse_width : float, default=1.0
        Intrinsic pulse width in units of sampling time (tsamp).
    max_snr_loss : float, default=0.05
        Maximum allowable fractional S/N loss (e.g. 0.05 = 5%).
    integer_grid : bool, default=True
        If True, rounds and deduplicates trial delays to unique sorted integers
        suitable for direct passing to FDMT `dt_arr`. If False, returns floats.
    method : str, default="snr_loss"
        Grid generation algorithm: "snr_loss" or "levin".
    tol : float, optional
        Broadening factor for "levin".

    Returns
    -------
    np.ndarray
        Sorted 1-D array of trial delays (int64 if integer_grid=True, else float64).
    """
    k_disp = DM_CONSTANT * (f_min**-2 - f_max**-2) / tsamp
    dm_min = dt_min / k_disp
    dm_max = dt_max / k_disp

    dm_grid = generate_optimal_dm_grid(
        f_min,
        f_max,
        nchans,
        tsamp,
        dm_max,
        dm_min=dm_min,
        pulse_width=pulse_width,
        max_snr_loss=max_snr_loss,
        method=method,
        tol=tol,
    )

    dt_floats = dm_grid * k_disp
    if not integer_grid:
        return dt_floats

    int_grid = np.unique(np.round(dt_floats).astype(np.int64))
    # Ensure boundary endpoints are preserved
    int_grid[0] = round(dt_min)
    int_grid[-1] = round(dt_max)
    return np.unique(int_grid)


def calculate_snr_loss(
    dm_test: float | np.ndarray,
    trial_dms: np.ndarray,
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    pulse_width: float = 1.0,
) -> float | np.ndarray:
    """Calculate expected fractional S/N loss for given true DM(s).

    Parameters
    ----------
    dm_test : float or np.ndarray
        True DM(s) to evaluate.
    trial_dms : np.ndarray
        Sorted 1-D array of trial DMs in the search grid.
    f_min : float
        Bottom edge of frequency band in MHz.
    f_max : float
        Top edge of frequency band in MHz.
    nchans : int
        Number of frequency channels.
    tsamp : float
        Sampling interval in seconds.
    pulse_width : float, default=1.0
        Intrinsic pulse width in units of sampling time.

    Returns
    -------
    float or np.ndarray
        Expected fractional S/N loss (1 - SNR / SNR_0), where 0.0 means perfect
        recovery and 0.05 means 5% sensitivity loss.
    """
    is_scalar = np.isscalar(dm_test)
    dm_arr = np.atleast_1d(np.asarray(dm_test, dtype=np.float64))

    # Find nearest trial DM for each test DM
    idx = np.searchsorted(trial_dms, dm_arr)
    idx = np.clip(idx, 0, len(trial_dms) - 1)
    idx_prev = np.clip(idx - 1, 0, len(trial_dms) - 1)

    diff1 = np.abs(dm_arr - trial_dms[idx])
    diff2 = np.abs(dm_arr - trial_dms[idx_prev])
    closest_dm_diff = np.minimum(diff1, diff2)

    df_chan = (f_max - f_min) / nchans
    k_disp = DM_CONSTANT * (f_min**-2 - f_max**-2) / tsamp

    # Delay offset across band in samples
    delta_t = closest_dm_diff * k_disp

    # Intra-channel smearing at bottom of band
    t_chan = (2.0 * DM_CONSTANT * np.abs(dm_arr) * df_chan / (f_min**3)) / tsamp
    w_eff = np.sqrt(pulse_width**2 + 1.0 + t_chan**2)

    # Broadened effective width
    w_broad = np.sqrt(w_eff**2 + delta_t**2)
    recovery = w_eff / w_broad
    loss = 1.0 - recovery

    if is_scalar:
        return float(loss[0])
    return loss
