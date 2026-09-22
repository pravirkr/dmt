"""Regression guard: FDMT's S/N-recovery fraction for an off-grid (arbitrary
dispersive delay, arbitrary sub-sample arrival phase) pulse must not fall
below an empirically-established floor.

This complements `tests/python/test_fdmt.py`'s `add_frb_track`/`trace_dm`
exact-recovery tests, which place a pulse exactly on one of FDMT's own
trial-grid coordinates by construction and therefore cannot see the more
realistic case of a continuous-time pulse landing off-grid.
"""

import numpy as np
from dmtlib import libdmt

try:
    from ._dispersed_pulse import sweep_recovery
except ImportError:
    from _dispersed_pulse import sweep_recovery

# Established from the coarse grid below (observed min ~0.66) with margin;
# a drop past this floor indicates a real sensitivity regression in the
# box-smearing/rounding logic, not sweep noise (the sweep is deterministic).
RECOVERY_FLOOR = 0.5


class TestFDMTSensitivity:
    def test_offgrid_recovery_fraction_floor(self) -> None:
        f_min, f_max, tsamp, nchans, dt_max = 1000.0, 1500.0, 1.0, 64, 64
        nsamps = 4 * dt_max
        fdmt = libdmt.FDMTCPU(f_min, f_max, nchans, nsamps, tsamp, dt_max)

        dt_values = np.linspace(1.0, dt_max - 1.0, 12)
        phases = np.linspace(0.0, 0.9, 5)

        fractions = sweep_recovery(
            fdmt.execute,
            dt_values,
            phases,
            nchans,
            nsamps,
            f_min,
            f_max,
            sigma_fn=lambda idm: float(fdmt.get_effective_sigma_grid(1)[idm]),
        )

        assert np.all(np.isfinite(fractions))
        assert fractions.min() >= RECOVERY_FLOOR
        assert fractions.max() <= 1.0 + 1e-6
