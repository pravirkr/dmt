from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from dmtlib import libdmt

try:
    from ._dispersed_pulse import sweep_recovery
except ImportError:
    from _dispersed_pulse import sweep_recovery


def _load_reference():
    path = (
        Path(__file__).resolve().parents[2]
        / "bench"
        / "scripts"
        / "fdmt_correction_redundancy.py"
    )
    spec = importlib.util.spec_from_file_location("_fdmt_correction_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ref = _load_reference()
TRUE_F_MIN = _ref.TRUE_F_MIN
TRUE_F_MAX = _ref.TRUE_F_MAX
NCHANS = _ref.NCHANS
DT_MAX = _ref.DT_MAX
NSAMPS = _ref.NSAMPS
reference_fdmt = _ref.reference_fdmt


class TestCorrectionRedundancy:
    def test_reference_matches_dmt_real_engine(self) -> None:
        fdmt = libdmt.FDMT(TRUE_F_MIN, TRUE_F_MAX, NCHANS, NSAMPS, 1.0, DT_MAX)
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
