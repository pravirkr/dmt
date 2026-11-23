"""Standalone study of the classic FDMT +/-dF/2 midpoint correction.

Compares a minimal pure-NumPy recursive FDMT with and without the midpoint
correction on dispersed pulses under both frequency-edge and channel-centre
conventions, normalized by measured white-noise row sigma.

Usage:
    python bench/scripts/fdmt_correction_redundancy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS_PYTHON = Path(__file__).resolve().parents[2] / "tests" / "python"
if str(_TESTS_PYTHON) not in sys.path:
    sys.path.insert(0, str(_TESTS_PYTHON))

from _dispersed_pulse import sweep_recovery  # noqa: E402

TRUE_F_MIN, TRUE_F_MAX = 1000.0, 1500.0
NCHANS = 64
DT_MAX = 64
NSAMPS = 4 * DT_MAX
CHAN_WIDTH = (TRUE_F_MAX - TRUE_F_MIN) / NCHANS


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


def _box_mean_weights(ndt: int) -> np.ndarray:
    """dmt's level-0 box-mean weights round(2**b / L) * 2**-b, b =
    ceil(log2 L) + 2, for boxes of L = 1 .. ndt samples."""
    length = np.arange(1, ndt + 1, dtype=np.float64)
    b = np.ceil(np.log2(length)) + 2
    return np.floor((2.0**b + np.floor(length / 2)) / length) * 2.0**-b


def reference_fdmt(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    max_dt: int,
    correction: bool = False,
) -> np.ndarray:
    """Minimal recursive FDMT with dmt's dyadic level-0 box-mean weights."""
    nchans, nsamps = waterfall.shape
    if nchans & (nchans - 1) != 0:
        raise ValueError("reference_fdmt requires a power-of-2 channel count")
    n_iters = int(np.log2(nchans))
    state = _fdmt_initialization(waterfall, f_min, f_max, max_dt)
    state = state * _box_mean_weights(state.shape[1])[None, :, None]
    for iteration_num in range(1, n_iters + 1):
        state = _fdmt_iteration(
            state, max_dt, nchans, f_min, f_max, iteration_num, correction
        )
    nchans_out, ndt, ns = state.shape
    assert nchans_out == 1
    return state.reshape(ndt, ns)


def _row_sigma(execute_fn, nreal: int = 60, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = [
        execute_fn(rng.standard_normal((NCHANS, NSAMPS)))[:, DT_MAX:]
        for _ in range(nreal)
    ]
    return np.concatenate(rows, axis=1).std(axis=1)


def _algorithm_inputs(convention: str) -> tuple[float, float]:
    if convention == "edge":
        return TRUE_F_MIN, TRUE_F_MAX
    if convention == "center_of_extreme_channels":
        return TRUE_F_MIN + CHAN_WIDTH / 2, TRUE_F_MAX - CHAN_WIDTH / 2
    raise ValueError(convention)


def main() -> None:
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

        sigma_no_corr = _row_sigma(execute_no_correction)
        sigma_with_corr = _row_sigma(execute_with_correction)
        frac_no_corr = sweep_recovery(
            execute_no_correction,
            dt_values,
            phases,
            NCHANS,
            NSAMPS,
            TRUE_F_MIN,
            TRUE_F_MAX,
            sigma_fn=sigma_no_corr.__getitem__,
        )
        frac_with_corr = sweep_recovery(
            execute_with_correction,
            dt_values,
            phases,
            NCHANS,
            NSAMPS,
            TRUE_F_MIN,
            TRUE_F_MAX,
            sigma_fn=sigma_with_corr.__getitem__,
        )
        print(
            f"{convention}: mean S/N recovery no_corr={frac_no_corr.mean():.4f}, "
            f"with_corr={frac_with_corr.mean():.4f}"
        )
        assert frac_with_corr.mean() <= frac_no_corr.mean() + 0.02


if __name__ == "__main__":
    main()
