from __future__ import annotations

import numpy as np
from dmtlib import libdmt

try:
    from ._dispersed_pulse import sweep_recovery
except ImportError:
    from _dispersed_pulse import sweep_recovery

# --------------------------------------------------------------------------
# Minimal, standalone reference FDMT (dense, power-of-2 channels, positive
# DM only) with the ±dF/2 correction exposed as a flag.
# --------------------------------------------------------------------------


def _fdmt_initialization(
    image: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
) -> np.ndarray:
    nchans, nsamps = image.shape
    d_f = (f_max - f_min) / nchans
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1 / f_min**2 - 1 / (f_min + d_f) ** 2)
            / (1 / f_min**2 - 1 / f_max**2)
        )
    )
    state = np.zeros((nchans, delta_t + 1, nsamps))
    state[:, 0, :] = image
    for i_dt in range(1, delta_t + 1):
        state[:, i_dt, i_dt:] = state[:, i_dt - 1, i_dt:] + image[:, : nsamps - i_dt]
    return state


def _fdmt_iteration(
    state: np.ndarray,
    max_dt: int,
    nchans_total: int,
    f_min: float,
    f_max: float,
    iteration_num: int,
    correction: bool,
) -> np.ndarray:
    nchans_in, _, nsamps = state.shape
    d_f_iter = 2**iteration_num * (f_max - f_min) / nchans_total
    d_f = (f_max - f_min) / nchans_total
    delta_t = int(
        np.ceil(
            (max_dt - 1)
            * (1 / f_min**2 - 1 / (f_min + d_f_iter) ** 2)
            / (1 / f_min**2 - 1 / f_max**2)
        )
    )
    nchans_out = nchans_in // 2
    out = np.zeros((nchans_out, delta_t + 1, nsamps))

    corr = d_f / 2.0 if (correction and iteration_num > 0) else 0.0

    for i_f in range(nchans_out):
        f_start = (f_max - f_min) / nchans_out * i_f + f_min
        f_end = (f_max - f_min) / nchans_out * (i_f + 1) + f_min
        f_mid = (f_end - f_start) / 2 + f_start - corr
        f_mid_larger = (f_end - f_start) / 2 + f_start + corr
        delta_t_local = int(
            np.ceil(
                (max_dt - 1)
                * (1 / f_start**2 - 1 / f_end**2)
                / (1 / f_min**2 - 1 / f_max**2)
            )
        )

        for i_dt in range(delta_t_local + 1):
            dt_mid = round(
                i_dt * (1 / f_mid**2 - 1 / f_start**2) / (1 / f_end**2 - 1 / f_start**2)
            )
            dt_mid_larger = round(
                i_dt
                * (1 / f_mid_larger**2 - 1 / f_start**2)
                / (1 / f_end**2 - 1 / f_start**2)
            )
            dt_rest = i_dt - dt_mid_larger

            shift = dt_mid_larger
            out[i_f, i_dt, :shift] = state[2 * i_f, dt_mid, :shift]
            out[i_f, i_dt, shift:] = (
                state[2 * i_f, dt_mid, shift:]
                + state[2 * i_f + 1, dt_rest, : nsamps - shift]
            )
    return out


def reference_fdmt(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    correction: bool = False,
) -> np.ndarray:
    """Minimal recursive FDMT. `f_min`/`f_max` are always treated as
    absolute channel-edge frequencies by the algorithm itself -- exactly as
    the classic implementation always assumed, whether or not the caller's
    actual values really are edges (that potential mismatch is the
    experiment `test_correction_never_improves_recovery` runs)."""
    nchans, nsamps = waterfall.shape
    if nchans & (nchans - 1) != 0:
        raise ValueError("reference_fdmt requires a power-of-2 channel count")
    n_iters = int(np.log2(nchans))
    state = _fdmt_initialization(waterfall, f_min, f_max, max_dt)
    for iteration_num in range(1, n_iters + 1):
        state = _fdmt_iteration(
            state, max_dt, nchans, f_min, f_max, iteration_num, correction
        )
    nchans_out, ndt, ns = state.shape
    assert nchans_out == 1
    return state.reshape(ndt, ns)


# --------------------------------------------------------------------------
# Experiment setup
# --------------------------------------------------------------------------

TRUE_F_MIN, TRUE_F_MAX = 1000.0, 1500.0
NCHANS = 64
DT_MAX = 64
NSAMPS = 4 * DT_MAX
CHAN_WIDTH = (TRUE_F_MAX - TRUE_F_MIN) / NCHANS


def _algorithm_inputs(convention: str) -> tuple[float, float]:
    """What a caller following each convention would pass as f_min/f_max --
    the pulse itself is always injected using the TRUE physical band edges
    (`TRUE_F_MIN`/`TRUE_F_MAX`); only the algorithm's belief changes."""
    if convention == "edge":
        return TRUE_F_MIN, TRUE_F_MAX
    if convention == "center_of_extreme_channels":
        return TRUE_F_MIN + CHAN_WIDTH / 2, TRUE_F_MAX - CHAN_WIDTH / 2
    raise ValueError(convention)


class TestCorrectionRedundancy:
    def test_reference_matches_dmt_real_engine(self) -> None:
        fdmt = libdmt.FDMTCPU(TRUE_F_MIN, TRUE_F_MAX, NCHANS, NSAMPS, 1.0, DT_MAX)
        dt_values = np.linspace(1.0, DT_MAX - 1.0, 10)
        phases = np.linspace(0.0, 0.9, 5)

        frac_dmt = sweep_recovery(
            fdmt.execute, dt_values, phases, NCHANS, NSAMPS, TRUE_F_MIN, TRUE_F_MAX
        )
        frac_ref = sweep_recovery(
            lambda wf: reference_fdmt(
                wf, TRUE_F_MIN, TRUE_F_MAX, DT_MAX, correction=False
            ),
            dt_values,
            phases,
            NCHANS,
            NSAMPS,
            TRUE_F_MIN,
            TRUE_F_MAX,
        )
        np.testing.assert_allclose(frac_dmt, frac_ref, atol=0.06)

    def test_correction_never_improves_recovery(self) -> None:
        dt_values = np.linspace(1.0, DT_MAX - 1.0, 10)
        phases = np.linspace(0.0, 0.9, 6)

        for convention in ("edge", "center_of_extreme_channels"):
            f_min_a, f_max_a = _algorithm_inputs(convention)

            def execute_no_correction(
                wf: np.ndarray, f_min_a=f_min_a, f_max_a=f_max_a
            ) -> np.ndarray:
                return reference_fdmt(wf, f_min_a, f_max_a, DT_MAX, correction=False)

            def execute_with_correction(
                wf: np.ndarray, f_min_a=f_min_a, f_max_a=f_max_a
            ) -> np.ndarray:
                return reference_fdmt(wf, f_min_a, f_max_a, DT_MAX, correction=True)

            frac_no_corr = sweep_recovery(
                execute_no_correction,
                dt_values,
                phases,
                NCHANS,
                NSAMPS,
                TRUE_F_MIN,
                TRUE_F_MAX,
            )
            frac_with_corr = sweep_recovery(
                execute_with_correction,
                dt_values,
                phases,
                NCHANS,
                NSAMPS,
                TRUE_F_MIN,
                TRUE_F_MAX,
            )

            # A small positive tolerance, not a strict <=, since this is a
            # statistical (mean-over-a-grid) comparison, not a per-point
            # exact one.
            assert frac_with_corr.mean() <= frac_no_corr.mean() + 0.02, (
                f"correction unexpectedly improved mean recovery under "
                f"convention={convention!r}: "
                f"{frac_with_corr.mean():.4f} > {frac_no_corr.mean():.4f}"
            )
