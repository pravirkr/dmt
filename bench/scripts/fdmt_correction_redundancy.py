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
from _reference_fdmt import (  # noqa: E402
    CHAN_WIDTH,
    DT_MAX,
    NCHANS,
    NSAMPS,
    TRUE_F_MAX,
    TRUE_F_MIN,
    reference_fdmt,
)


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
