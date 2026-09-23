from __future__ import annotations

import numpy as np
import pytest
from dmtlib import (
    FDMTCPU,
    calculate_snr_loss,
    compute_fdmt,
    generate_optimal_dm_grid,
    generate_optimal_dt_grid,
)

try:
    from ._dispersed_pulse import sweep_recovery
except ImportError:
    from _dispersed_pulse import sweep_recovery


class TestGridGenerator:
    f_min = 1000.0
    f_max = 1500.0
    nchans = 64
    tsamp = 0.001

    def test_dm_grid_monotonicity_and_endpoints(self) -> None:
        dm_min = 0.0
        dm_max = 100.0
        grid = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=dm_max,
            dm_min=dm_min,
            max_snr_loss=0.05,
        )

        assert grid.ndim == 1
        assert len(grid) > 1
        assert np.isclose(grid[0], dm_min)
        assert np.isclose(grid[-1], dm_max)
        # Strictly monotonically increasing
        assert np.all(np.diff(grid) > 0)

    def test_dm_grid_single_point(self) -> None:
        grid = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=10.0,
            dm_min=10.0,
        )
        assert len(grid) == 1
        assert np.isclose(grid[0], 10.0)

    def test_negative_and_symmetric_dm_grid(self) -> None:
        # Symmetric range [-50, 50]
        sym_grid = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=50.0,
            dm_min=-50.0,
            max_snr_loss=0.05,
        )
        assert np.isclose(sym_grid[0], -50.0)
        assert np.isclose(sym_grid[-1], 50.0)
        assert np.any(np.isclose(sym_grid, 0.0))
        assert np.all(np.diff(sym_grid) > 0)
        # Symmetry check
        np.testing.assert_allclose(sym_grid, -sym_grid[::-1], atol=1e-10)

        # Strictly negative range [-100, -20]
        neg_grid = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=-20.0,
            dm_min=-100.0,
            max_snr_loss=0.05,
        )
        assert np.isclose(neg_grid[0], -100.0)
        assert np.isclose(neg_grid[-1], -20.0)
        assert np.all(np.diff(neg_grid) > 0)

    def test_dt_grid_generation(self) -> None:
        dt_max = 128
        # Integer grid
        dt_grid_int = generate_optimal_dt_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dt_max=dt_max,
            dt_min=0,
            integer_grid=True,
        )
        assert np.issubdtype(dt_grid_int.dtype, np.integer)
        assert dt_grid_int[0] == 0
        assert dt_grid_int[-1] == dt_max
        assert np.all(np.diff(dt_grid_int) > 0)

        # Float grid
        dt_grid_flt = generate_optimal_dt_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dt_max=dt_max,
            dt_min=0,
            integer_grid=False,
        )
        assert np.issubdtype(dt_grid_flt.dtype, np.floating)
        assert np.isclose(dt_grid_flt[0], 0.0)
        assert np.isclose(dt_grid_flt[-1], float(dt_max))
        assert np.all(np.diff(dt_grid_flt) > 0)

    def test_trial_reduction_and_snr_loss_bounding(self) -> None:
        dm_max = 500.0
        max_loss = 0.05
        grid = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=dm_max,
            dm_min=0.0,
            max_snr_loss=max_loss,
        )

        # Uniform step dt=1 length across band
        k_disp = 4.148808e3 * (self.f_min**-2 - self.f_max**-2) / self.tsamp
        uniform_trials = int(np.ceil(dm_max * k_disp)) + 1

        # Must reduce trials by at least 40%
        assert len(grid) < uniform_trials * 0.6

        # Test worst-case DM midpoints between trials
        midpoints = (grid[:-1] + grid[1:]) / 2.0
        losses = calculate_snr_loss(
            midpoints,
            grid,
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            pulse_width=1.0,
        )
        assert np.all(np.isfinite(losses))
        # Midpoints should be bounded by max_loss (allowing numerical tolerance)
        assert np.all(losses <= max_loss + 0.01)

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="max_snr_loss"):
            generate_optimal_dm_grid(
                self.f_min, self.f_max, self.nchans, self.tsamp, dm_max=10.0, max_snr_loss=1.5
            )

        with pytest.raises(ValueError, match="dm_max"):
            generate_optimal_dm_grid(
                self.f_min, self.f_max, self.nchans, self.tsamp, dm_max=0.0, dm_min=10.0
            )

        with pytest.raises(ValueError, match="frequency band"):
            generate_optimal_dm_grid(
                1500.0, 1000.0, self.nchans, self.tsamp, dm_max=10.0
            )

    def test_end_to_end_fdmt_execution_with_optimal_grid(self) -> None:
        nsamps = 512
        dt_max = 64
        dt_grid = generate_optimal_dt_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dt_max=dt_max,
            dt_min=0,
            integer_grid=True,
        )

        fdmt = FDMTCPU(
            self.f_min,
            self.f_max,
            self.nchans,
            nsamps,
            self.tsamp,
            dt_arr=dt_grid,
        )

        assert fdmt.plan.is_custom_grid
        np.testing.assert_array_equal(fdmt.dt_grid_final, dt_grid)

        rng = np.random.default_rng(12345)
        wf = rng.standard_normal((self.nchans, nsamps), dtype=np.float32)

        out_fdmt = fdmt.execute(wf)
        assert out_fdmt.shape == (len(dt_grid), fdmt.plan.dmt_nsamps)
        assert np.all(np.isfinite(out_fdmt))

        # Also verify with compute_fdmt
        out_buf, plan = compute_fdmt(
            wf,
            self.f_min,
            self.f_max,
            self.nchans,
            nsamps,
            self.tsamp,
            dt_arr=dt_grid,
        )
        assert plan.is_custom_grid
        out_compute = out_buf[: plan.dmt_size].reshape(plan.dt_grid_final.size, plan.dmt_nsamps)
        np.testing.assert_allclose(out_fdmt, out_compute, rtol=1e-5, atol=1e-5)

    def test_levin_method_parametrization(self) -> None:
        from dmtlib import DDMTPlan

        dm_max = 50.0
        tol = 1.25
        grid_levin = generate_optimal_dm_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dm_max=dm_max,
            method="levin",
            tol=tol,
        )
        expected = DDMTPlan.generate_levin_dm_grid(
            0.0, dm_max, self.tsamp, 1.0, self.f_min, self.f_max, self.nchans, tol
        )
        np.testing.assert_allclose(grid_levin, expected)

        # dt grid via levin method
        dt_levin = generate_optimal_dt_grid(
            self.f_min,
            self.f_max,
            self.nchans,
            self.tsamp,
            dt_max=64,
            method="levin",
            tol=tol,
        )
        assert dt_levin[0] == 0
        assert dt_levin[-1] == 64
        assert np.all(np.diff(dt_levin) > 0)

        # Negative DM under levin raises ValueError
        with pytest.raises(ValueError, match="negative"):
            generate_optimal_dm_grid(
                self.f_min,
                self.f_max,
                self.nchans,
                self.tsamp,
                dm_max=50.0,
                dm_min=-10.0,
                method="levin",
            )

        # Unknown method raises ValueError
        with pytest.raises(ValueError, match="Unknown"):
            generate_optimal_dm_grid(
                self.f_min,
                self.f_max,
                self.nchans,
                self.tsamp,
                dm_max=50.0,
                method="invalid",
            )

