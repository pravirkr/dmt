import numpy as np
import pytest
from dmtlib import libdmt

class TestCohFDMT:

    @staticmethod
    def _make_plan() -> libdmt.CohFDMTPlan:
        chan_per_sub = 4
        f_center = 1250.0
        bw_sub = 25.0
        nsub = 4
        tbin = 1.0e-6
        nbin = 1024
        nfft = 2
        t_p = tbin * chan_per_sub
        dm_max = 5.0
        dm_min = 0.0
        return libdmt.CohFDMTPlan(
            f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max, dm_min, 32, "PRITF"
        )

    def test_plan_properties(self) -> None:
        plan = self._make_plan()
        assert plan.ndm == len(plan.dm_grid_final)
        assert plan.dmt_ndms == plan.ndm
        assert plan.dmt_nsamps == plan.fdmt_plan.dmt_nsamps
        assert plan.dmt_size == plan.ndm * plan.dmt_nsamps

        var_grid = plan.get_effective_variance_grid()
        sig_grid = plan.get_effective_sigma_grid()
        cnt_grid = plan.get_cumulative_count_grid()

        assert len(var_grid) == plan.ndm
        assert len(sig_grid) == plan.ndm
        assert len(cnt_grid) == plan.ndm
        assert np.all(var_grid > 0.0)
        # sigma may be computed in C++ float32; sqrt(var) in float64 — allow ~1 ulp.
        np.testing.assert_allclose(sig_grid, np.sqrt(var_grid), rtol=1e-6, atol=1e-6)
        assert np.all(cnt_grid > 0.0)

    def test_execute_output_shape_2d(self) -> None:
        plan = self._make_plan()
        coh_fdmt = libdmt.CohFDMTCPU(
            plan.f_center,
            plan.bw_sub,
            plan.nsub,
            plan.tbin,
            plan.nbin,
            plan.nfft,
            plan.t_p,
            plan.dm_max,
            plan.dm_min,
            plan.noverlap,
        )

        in_size = 2 * 2 * plan.nsamp * plan.nsub
        rng = np.random.default_rng(42)
        data_u8 = rng.integers(0, 256, size=in_size, dtype=np.uint8)

        dmt_u8 = coh_fdmt.execute(data_u8)
        assert dmt_u8.ndim == 2
        assert dmt_u8.shape == (plan.ndm, plan.dmt_nsamps)
        assert np.all(np.isfinite(dmt_u8))
        assert np.any(dmt_u8 > 0.0)

        data_i8 = rng.integers(-128, 128, size=in_size, dtype=np.int8)
        dmt_i8 = coh_fdmt.execute(data_i8)
        assert dmt_i8.ndim == 2
        assert dmt_i8.shape == (plan.ndm, plan.dmt_nsamps)
        assert np.all(np.isfinite(dmt_i8))

    def test_multi_block_streaming_and_reset(self) -> None:
        plan = self._make_plan()
        coh_fdmt = libdmt.CohFDMTCPU(
            plan.f_center,
            plan.bw_sub,
            plan.nsub,
            plan.tbin,
            plan.nbin,
            plan.nfft,
            plan.t_p,
            plan.dm_max,
            plan.dm_min,
            plan.noverlap,
        )

        in_size = 2 * 2 * plan.nsamp * plan.nsub
        rng = np.random.default_rng(123)
        b1 = rng.integers(0, 256, size=in_size, dtype=np.uint8)
        b2 = rng.integers(0, 256, size=in_size, dtype=np.uint8)

        dmt_b1_1 = coh_fdmt.execute(b1)
        dmt_b2_streamed = coh_fdmt.execute(b2)

        coh_fdmt.reset_history()
        dmt_b2_cold = coh_fdmt.execute(b2)

        # Streamed output should carry inter-block history from b1, differing from cold
        assert not np.allclose(dmt_b2_streamed, dmt_b2_cold, atol=1e-4)

        # Cold execution on b1 after reset must match the first run
        coh_fdmt.reset_history()
        dmt_b1_2 = coh_fdmt.execute(b1)
        np.testing.assert_array_equal(dmt_b1_1, dmt_b1_2)
