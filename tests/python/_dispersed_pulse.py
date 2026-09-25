"""Continuous-time (sub-sample-accurate) dispersed pulse injection.

Unlike `add_frb_track` (which places a pulse exactly on one of FDMT's own
trial-grid coordinates by construction, see `bind_fdmt.cpp`), this places a
pulse at an arbitrary, continuous dispersive delay and arrival phase,
splitting each channel's unit flux fractionally across the integer samples
its dispersion sweep actually crosses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

_DISP_COEFF = -2.0  # f^-2 dispersion law -- matches dmt's kDispCoeff


def channel_edges(
    f_min: float, f_max: float, nchans: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel [f_lo, f_hi) edges, in dmt's fixed band-edge convention
    (`f_min`/`f_max` are the absolute edges of the outermost channels).
    """
    edges = np.linspace(f_min, f_max, nchans + 1)
    return edges[:-1], edges[1:]


def channel_delay_fraction(f: np.ndarray, f_min: float, f_max: float) -> np.ndarray:
    """Fraction of the total [f_min, f_max] dispersive delay accumulated by
    frequency `f`, referenced to zero delay at `f_max` -- dmt's own `cff`
    ratio (`lib/dm_utils.cpp`), inlined here since it's a two-line closed
    form and not worth a binding just for this.
    """
    num = f**_DISP_COEFF - f_max**_DISP_COEFF
    den = f_min**_DISP_COEFF - f_max**_DISP_COEFF
    return num / den


def make_dispersed_pulse(
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    dt_true: float,
    t0: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """A pure dispersion-curve pulse, continuous in (delay, arrival time).

    `dt_true`: total dispersive delay across the full band, in samples
    (fractional/continuous -- this is dmt's own dt unit, not physical DM).
    `t0`: arrival time (fractional sample index) of the pulse at `f_max`.
    Each channel's `amplitude` worth of energy is spread uniformly in time
    across its own dispersion-smearing window [t0 + delay(f_hi), t0 +
    delay(f_lo)] and split across the integer samples that window overlaps
    in proportion to the overlap length -- energy is conserved per channel
    regardless of how the window straddles sample boundaries.
    """
    f_lo, f_hi = channel_edges(f_min, f_max, nchans)
    delay_hi = t0 + dt_true * channel_delay_fraction(f_hi, f_min, f_max)
    raw_width = dt_true * (
        channel_delay_fraction(f_lo, f_min, f_max)
        - channel_delay_fraction(f_hi, f_min, f_max)
    )
    # Floor the window width away from exactly zero, and grow the window
    # itself (not just the normalizing divisor) by the same amount, so a
    # window that degenerates to a point still has a strictly positive
    # length to compute bin overlaps against -- otherwise a point sitting
    # exactly on an integer sample boundary would get zero overlap with
    # every bin instead of landing fully in the bin it's actually in.
    width = np.maximum(raw_width, 1e-6)
    delay_lo = delay_hi + width

    i_start = np.floor(delay_hi).astype(np.int64)
    i_end = np.floor(delay_lo).astype(np.int64)
    max_span = int((i_end - i_start).max()) + 1

    waterfall = np.zeros((nchans, nsamps), dtype=np.float32)
    for k in range(max_span):
        idx = i_start + k
        valid = (idx <= i_end) & (idx >= 0) & (idx < nsamps)
        if not np.any(valid):
            continue
        lo = np.maximum(delay_hi, idx.astype(np.float64))
        hi = np.minimum(delay_lo, (idx + 1).astype(np.float64))
        overlap = np.clip(hi - lo, 0.0, None)
        contrib = np.where(valid, amplitude * overlap / width, 0.0)
        rows = np.nonzero(valid)[0]
        waterfall[rows, idx[rows]] += contrib[rows].astype(np.float32)
    return waterfall


def sweep_recovery(
    execute_fn: Callable[[np.ndarray], np.ndarray],
    dt_values: Sequence[float],
    phases: Sequence[float],
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    t0_base: float | None = None,
    amplitude: float = 1.0,
    sigma_fn: Callable[[int], float] | None = None,
) -> np.ndarray:
    """Sweep (dt_values x phases): inject a continuous dispersed pulse for
    every combination, run it through `execute_fn`, and return a
    `(len(dt_values), len(phases))` array of recovery fractions.

    `execute_fn(waterfall) -> (n_trials, nsamps)` is any dedispersion engine
    with FDMT's output convention (dmt's `FDMTCPU.execute`, or a standalone
    reference merge).

    If `sigma_fn` is given, the recovery fraction is the achieved S/N (peak
    output value divided by the analytical noise sigma at the peak's trial
    index) as a fraction of the pulse's own matched-filter S/N ceiling
    `sqrt(sum(pulse**2))` -- the best any linear method could do against
    dmt's assumed unit-variance-per-raw-sample noise model (see
    `variance_from_smearing_row` in `lib/plans.cpp`). Without `sigma_fn`,
    the fraction is simply peak amplitude over the ideal `nchans *
    amplitude` (every channel landing constructively in one output sample)
    -- adequate for A/B-comparing two variants of an engine that has no
    analytical noise model of its own.
    """
    if t0_base is None:
        t0_base = nsamps / 2.0

    result = np.empty((len(dt_values), len(phases)))
    for i, dt_true in enumerate(dt_values):
        for j, phase in enumerate(phases):
            pulse = make_dispersed_pulse(
                nchans, nsamps, f_min, f_max, dt_true, t0_base + phase, amplitude
            )
            out = np.asarray(execute_fn(pulse))
            idm_star, _ = np.unravel_index(np.argmax(out), out.shape)
            peak = float(out[np.unravel_index(np.argmax(out), out.shape)])
            if sigma_fn is not None:
                achieved = peak / sigma_fn(int(idm_star))
                ceiling = float(np.sqrt(np.sum(pulse.astype(np.float64) ** 2)))
                result[i, j] = achieved / ceiling
            else:
                result[i, j] = peak / (nchans * amplitude)
    return result
