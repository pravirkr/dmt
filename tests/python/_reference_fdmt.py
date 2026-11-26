"""Minimal pure-NumPy recursive FDMT, for tests and the correction study.

`reference_fdmt` applies dmt's dyadic level-0 box-mean weights and, with
`correction`, the classic +/-dF/2 midpoint correction
(bench/scripts/fdmt_correction_redundancy.py studies it).
"""

from __future__ import annotations

import numpy as np

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
