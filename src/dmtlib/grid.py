"""Trial DM and delay grids for FDMT and DDMT.

Wrappers of the C++ generators (``dmt/common/dm_grids.hpp``, also
:func:`dmtlib.generate_dm_grid` and :func:`dmtlib.generate_dt_grid`) that take
the pulse width in samples. Each rule bounds how much a pulse may broaden
between two trials (``tol``, or ``max_snr_loss = 1 - 1 / tol``).

1. ``method="snr_loss"`` (default, dmt's rule): the effective width::

       W_eff(DM) = sqrt(W_pulse^2 + t_samp^2 + t_chan(DM)^2),
       t_chan(DM) = 2 * K_DM * |DM - chan_dm| * df / f_min^3

   may grow by at most ``tol`` under the delay error of half a step, from the
   exact dispersion sweep across the band::

       dt_step(DM) <= max(1.0, 2 * eta * W_eff(DM)),  eta = sqrt(tol^2 - 1)

   Never collapses at high DM; supports negative DMs and coherently
   dedispersed data (``chan_dm``). On very wide bands the bottom-channel
   smearing makes it lose more than ``max_snr_loss`` at high DM.

2. ``method="levin"`` (Levin 2012, as in Heimdall): smearing and sweep from
   the band centre (a straight-line sweep, so too sparse on very wide bands),
   broadening bounded by ``tol``. The classic formula, kept as it is.

3. ``method="levin_piecewise"`` (DDplan-style): the Levin grid made piecewise
   uniform: the range is split where the Levin step doubles and each segment
   is uniformly spaced at its smallest Levin step. Typically 1.2-1.4x the
   Levin trials; the grid for :class:`DDMTFFT`'s NUFFT path.

See the pipeline guide's DM-grid page for choosing one.
"""

from __future__ import annotations

import numpy as np

from .libdmt import DMGridMethod, generate_dm_grid, generate_dt_grid

DM_CONSTANT = 1 / 2.41e-4  # TEMPO2, Manchester&Taylor (1972)


def _effective_tol(method: str, tol: float | None, max_snr_loss: float) -> float:
    """Return the C++ ``tol`` (0 lets the S/N-loss rule use ``max_snr_loss``)."""
    if tol is not None:
        return tol
    if method == "snr_loss":
        return 0.0
    # Levin: the broadening of the requested loss (1.25 if it is invalid).
    return 1.0 / (1.0 - max_snr_loss) if max_snr_loss < 1.0 else 1.25


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
    chan_dm: float = 0.0,
) -> np.ndarray:
    """Generate an optimal non-uniform DM trial grid (in pc cm^-3).

    A wrapper of the C++ ``dmt::plans::generate_dm_grid`` (also
    :func:`dmtlib.generate_dm_grid`, which takes ``pulse_width`` in seconds).

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
        - "levin_piecewise": DDplan-style piecewise-uniform Levin grid (uniform
          segments where the Levin step doubles, never coarser than "levin").
          Use it with :class:`DDMTFFT`, which runs one NUFFT per segment.
    tol : float, optional
        Pulse broadening factor (tol > 1.0, e.g. 1.25), the knob of every
        method. If omitted: 1 / (1 - max_snr_loss) for "levin" (the same
        for "snr_loss", whose ``max_snr_loss`` is that tolerance:
        ``max_snr_loss = 1 - 1 / tol``).
    chan_dm : float, default=0.0
        DM the channels were coherently dedispersed to (0: ordinary
        filterbank data); "snr_loss" only. Intra-channel smearing grows with
        ``|DM - chan_dm|``, so trials get denser near ``chan_dm``.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of trial DMs (float64).
    """
    methods = {
        "snr_loss": DMGridMethod.SNR_LOSS,
        "levin": DMGridMethod.LEVIN,
        "levin_piecewise": DMGridMethod.LEVIN_PIECEWISE,
    }
    if method not in methods:
        msg = (
            f"Unknown grid generation method '{method}'; choose 'snr_loss', "
            "'levin' or 'levin_piecewise'"
        )
        raise ValueError(msg)
    effective_tol = _effective_tol(method, tol, max_snr_loss)
    return np.asarray(
        generate_dm_grid(
            f_min,
            f_max,
            nchans,
            tsamp,
            dm_min,
            dm_max,
            method=methods[method],
            pulse_width=pulse_width * tsamp,
            chan_dm=chan_dm,
            max_snr_loss=max_snr_loss,
            tol=effective_tol,
        ),
        dtype=np.float64,
    )


def generate_optimal_dt_grid(
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
    chan_dm: float = 0.0,
) -> np.ndarray:
    """Generate an optimal non-uniform delay grid (in samples) for FDMT.

    A wrapper of the C++ ``dmt::plans::generate_dt_grid`` (integer grids).

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
    chan_dm : float, default=0.0
        DM the channels were coherently dedispersed to (0: none);
        "snr_loss" only.

    Returns
    -------
    np.ndarray
        Sorted 1-D array of trial delays (int64 if integer_grid=True, else float64).
    """
    k_disp = DM_CONSTANT * (f_min**-2 - f_max**-2) / tsamp
    if integer_grid:
        methods = {
            "snr_loss": DMGridMethod.SNR_LOSS,
            "levin": DMGridMethod.LEVIN,
            "levin_piecewise": DMGridMethod.LEVIN_PIECEWISE,
        }
        if method not in methods:
            msg = f"Unknown grid generation method '{method}'"
            raise ValueError(msg)
        effective_tol = _effective_tol(method, tol, max_snr_loss)
        return np.asarray(
            generate_dt_grid(
                f_min,
                f_max,
                nchans,
                tsamp,
                dt_min,
                dt_max,
                method=methods[method],
                pulse_width=pulse_width * tsamp,
                chan_dm=chan_dm,
                max_snr_loss=max_snr_loss,
                tol=effective_tol,
            ),
            dtype=np.int64,
        )
    dm_grid = generate_optimal_dm_grid(
        f_min,
        f_max,
        nchans,
        tsamp,
        dt_max / k_disp,
        dm_min=dt_min / k_disp,
        pulse_width=pulse_width,
        max_snr_loss=max_snr_loss,
        method=method,
        tol=tol,
        chan_dm=chan_dm,
    )
    return dm_grid * k_disp


def calculate_snr_loss(
    dm_test: float | np.ndarray,
    trial_dms: np.ndarray,
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    pulse_width: float = 1.0,
    chan_dm: float = 0.0,
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
    chan_dm : float, default=0.0
        DM the channels were coherently dedispersed to (0: none).

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
    t_chan = (
        2.0 * DM_CONSTANT * np.abs(dm_arr - chan_dm) * df_chan / (f_min**3)
    ) / tsamp
    w_eff = np.sqrt(pulse_width**2 + 1.0 + t_chan**2)

    # Broadened effective width
    w_broad = np.sqrt(w_eff**2 + delta_t**2)
    recovery = w_eff / w_broad
    loss = 1.0 - recovery

    if is_scalar:
        return float(loss[0])
    return loss
