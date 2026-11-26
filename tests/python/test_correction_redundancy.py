from __future__ import annotations

import numpy as np

from dmtlib import libdmt

try:
    from ._dispersed_pulse import sweep_recovery
    from ._reference_fdmt import (
        DT_MAX,
        NCHANS,
        NSAMPS,
        TRUE_F_MAX,
        TRUE_F_MIN,
        reference_fdmt,
    )
except ImportError:
    from _dispersed_pulse import sweep_recovery
    from _reference_fdmt import (
        DT_MAX,
        NCHANS,
        NSAMPS,
        TRUE_F_MAX,
        TRUE_F_MIN,
        reference_fdmt,
    )


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
