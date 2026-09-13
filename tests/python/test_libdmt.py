import numpy as np
import pytest
from dmtlib import libdmt


class TestFDMT:
    def test_initialise_ones(self) -> None:
        nchans = 500
        nsamples = 1024
        dt_max = 512
        thefdmt = libdmt.FDMTCPU(1000, 1500, nchans, nsamples, 0.001, dt_max)
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt_output = thefdmt.execute(waterfall)
        np.testing.assert_equal(
            dmt_output.shape,
            (thefdmt.dt_grid_final.size, thefdmt.plan.dmt_nsamps),
        )

    def test_stepper_bit_exact(self) -> None:
        nchans = 128
        nsamples = 512
        dt_max = 64
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(42)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Synchronous execution
        sync_output = thefdmt.execute(waterfall)

        # Stepper execution
        thefdmt.reset(waterfall)
        thefdmt.advance_until_remaining(0)
        assert thefdmt.is_finished
        step_output = thefdmt.finalize()

        np.testing.assert_allclose(step_output, sync_output, rtol=1e-6, atol=1e-6)

    def test_subband_views_one_level_remaining(self) -> None:
        nchans = 256
        nsamples = 1024
        dt_max = 128
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(123)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        assert thefdmt.current_level == 0
        assert thefdmt.remaining_levels == thefdmt.total_levels - 1

        # Advance until 1 level before root -> 2 children subbands
        thefdmt.advance_until_remaining(1)
        assert thefdmt.remaining_levels == 1
        assert thefdmt.num_subbands == 2
        assert not thefdmt.is_finished

        sub0 = thefdmt.view_subband(0)
        sub1 = thefdmt.view_subband(1)

        # Verify subband metadata
        assert sub0.subband_idx == 0
        assert sub1.subband_idx == 1
        assert np.isclose(sub0.f_start, 1000.0)
        assert np.isclose(sub0.f_end, sub1.f_start)
        assert np.isclose(sub1.f_end, 1500.0)

        assert sub0.ndt > 0
        assert sub1.ndt > 0
        assert sub0.nsamps == sub1.nsamps

        # Check 2D data views
        assert sub0.data.shape == (sub0.ndt, sub0.nsamps)
        assert sub1.data.shape == (sub1.ndt, sub1.nsamps)

        # Check view_subband_data matches view_subband(s).data
        sub0_data = thefdmt.view_subband_data(0)
        np.testing.assert_array_equal(sub0_data, sub0.data)

        # Resume and finalize
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_subband_views_two_levels_remaining(self) -> None:
        nchans = 256
        nsamples = 1024
        dt_max = 128
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(456)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        thefdmt.advance_until_remaining(2)

        assert thefdmt.remaining_levels == 2
        assert thefdmt.num_subbands == 4

        # Verify contiguous frequency coverage across the 4 subbands
        subbands = [thefdmt.view_subband(s) for s in range(4)]
        assert np.isclose(subbands[0].f_start, 1000.0)
        for s in range(3):
            assert np.isclose(subbands[s].f_end, subbands[s + 1].f_start)
        assert np.isclose(subbands[3].f_end, 1500.0)

        # Check shapes
        for s in range(4):
            assert subbands[s].data.shape == (subbands[s].ndt, subbands[s].nsamps)

        # Finalize and verify exact match
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_step_by_step_advance(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(789)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        levels_stepped = 0
        while not thefdmt.is_finished:
            thefdmt.advance(1)
            levels_stepped += 1
            assert thefdmt.current_level == levels_stepped

        assert thefdmt.remaining_levels == 0
        assert thefdmt.is_finished

        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_custom_dmt_buffer(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(999)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Allocate custom buffer matching buffer_size
        custom_buf = np.zeros(thefdmt.plan.buffer_size, dtype=np.float32)
        thefdmt.reset(waterfall, custom_buf)
        thefdmt.advance_until_remaining(1)
        res = thefdmt.finalize()

        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)
        # Verify that the final result is in custom_buf directly
        np.testing.assert_array_equal(res.ravel(), custom_buf[: res.size])

    def test_stepper_error_handling(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(101)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Calling advance or view before reset should raise
        with pytest.raises(RuntimeError):
            thefdmt.advance()
        with pytest.raises(RuntimeError):
            thefdmt.view_subband(0)

        # Buffer too small
        undersized_buf = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError):
            thefdmt.reset(waterfall, undersized_buf)

        # Invalid waterfall dimension
        with pytest.raises(RuntimeError):
            thefdmt.reset(waterfall.ravel())

        # Subband index out of range
        thefdmt.reset(waterfall)
        with pytest.raises(IndexError):
            thefdmt.view_subband(9999)

        # Advancing beyond remaining levels clamps cleanly at root
        thefdmt.advance(100)
        assert thefdmt.is_finished
        assert thefdmt.remaining_levels == 0
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize("mode", ["full", "roll", "valid"])
    @pytest.mark.parametrize("use_box_smearing", [True, False])
    def test_modes_and_smearing_matrix(self, mode: str, use_box_smearing: bool) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        fdmt_sync = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max,
            dt_min=0, use_box_smearing=use_box_smearing, mode=mode
        )
        if mode == "full":
            assert fdmt_sync.plan.dmt_nsamps == nsamples + dt_max
        else:
            assert fdmt_sync.plan.dmt_nsamps == nsamples

        rng = np.random.default_rng(42)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        sync_output = fdmt_sync.execute(waterfall)

        fdmt_step = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max,
            dt_min=0, use_box_smearing=use_box_smearing, mode=mode
        )
        fdmt_step.reset(waterfall)
        fdmt_step.advance_until_remaining(0)
        assert fdmt_step.is_finished
        step_output = fdmt_step.finalize()

        np.testing.assert_allclose(step_output, sync_output, rtol=1e-5, atol=1e-5)

    def test_valid_mode_streaming_history(self) -> None:
        nchans = 32
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max,
            mode="valid"
        )
        block1 = np.ones((nchans, nsamples), dtype=np.float32)
        block2 = np.full((nchans, nsamples), 2.0, dtype=np.float32)

        dmt1 = thefdmt.execute(block1)
        dmt2 = thefdmt.execute(block2)

        assert dmt1.shape == (thefdmt.dt_grid_final.size, nsamples)
        assert dmt2.shape == (thefdmt.dt_grid_final.size, nsamples)
        assert np.sum(dmt2) > np.sum(dmt1)

    def test_smearing_grid_final(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)
        smearing_grid = thefdmt.plan.smearing_grid_final
        assert smearing_grid.size == thefdmt.plan.dmt_ndms * nchans
        assert np.all(smearing_grid >= 0.0)

    def test_plan_dt_min_and_dt_step(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 64
        dt_min = 16
        dt_step = 4
        plan = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001,
            dt_max=dt_max, dt_min=dt_min, dt_step=dt_step
        )
        assert plan.dt_min == dt_min
        assert plan.dt_max == dt_max
        assert plan.dt_step == dt_step

        expected_grid = np.arange(dt_min, dt_max + 1, dt_step)
        np.testing.assert_array_equal(plan.dt_grid_final, expected_grid)

        comp = plan.complexity
        assert comp.n_dt == len(expected_grid)
        assert comp.n_chans == nchans
        assert comp.brute_force_ops == len(expected_grid) * nchans
        assert comp.sum_additions > 0
        assert comp.total_tree_nodes >= comp.sum_additions
        assert comp.ops_ratio > 1.0
        assert "FDMT Complexity:" in repr(comp)
        assert "Theoretical Speedup Factor" in comp.to_string()
        plan.print_complexity_summary()

    def test_fdmt_cpu_sparse_execution(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 64
        dt_min = 16
        dt_step = 4

        fdmt_dense = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=dt_max, dt_min=0, dt_step=1
        )
        fdmt_sparse = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=dt_max, dt_min=dt_min, dt_step=dt_step
        )

        rng = np.random.default_rng(2024)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        sync_dense = fdmt_dense.execute(waterfall)
        sync_sparse = fdmt_sparse.execute(waterfall)

        assert sync_sparse.shape == (fdmt_sparse.dt_grid_final.size, fdmt_sparse.plan.dmt_nsamps)

        for s_idx, dt in enumerate(fdmt_sparse.dt_grid_final):
            d_idx = np.where(fdmt_dense.dt_grid_final == dt)[0][0]
            np.testing.assert_allclose(sync_sparse[s_idx], sync_dense[d_idx], rtol=1e-5, atol=1e-5)

    def test_compute_fdmt_function(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 64
        dt_min = 16
        dt_step = 4

        rng = np.random.default_rng(2025)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        dmt_buf, plan = libdmt.compute_fdmt(
            waterfall, 1000.0, 1500.0, nchans, nsamples, 0.001,
            dt_max=dt_max, dt_min=dt_min, dt_step=dt_step
        )
        assert plan.dt_min == dt_min
        assert plan.dt_max == dt_max
        assert plan.dt_step == dt_step
        dmt = dmt_buf[: plan.dmt_size].reshape(plan.dt_grid_final.size, plan.dmt_nsamps)
        assert dmt.shape == (plan.dt_grid_final.size, plan.dmt_nsamps)

        fdmt_sparse = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=dt_max, dt_min=dt_min, dt_step=dt_step
        )
        sync_sparse = fdmt_sparse.execute(waterfall)
        np.testing.assert_allclose(dmt, sync_sparse, rtol=1e-5, atol=1e-5)

    def test_plan_arbitrary_dt_and_dm_grids(self) -> None:
        nchans = 64
        nsamples = 256

        # Arbitrary dt_arr
        dt_arr = np.array([15, 30, 45, 60], dtype=np.uint64)
        plan_dt = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=dt_arr)
        assert plan_dt.is_custom_grid
        assert plan_dt.dt_min == 15
        assert plan_dt.dt_max == 60
        np.testing.assert_array_equal(plan_dt.dt_grid_final, dt_arr)

        # Standard plan is not custom grid
        plan_std = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=60)
        assert not plan_std.is_custom_grid

        # Arbitrary dm_arr
        dm_arr = np.array([5.0, 15.0, 30.0], dtype=np.float32)
        plan_dm = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, dm_arr=dm_arr)
        assert plan_dm.is_custom_grid
        assert len(plan_dm.dt_grid_final) > 0
        assert len(plan_dm.dm_grid_final) > 0

    def test_fdmt_cpu_arbitrary_grid(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 64

        dt_arr = np.array([12, 24, 36, 48, 64], dtype=np.uint64)

        fdmt_dense = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=dt_max, dt_min=0, dt_step=1,
            use_box_smearing=False
        )
        fdmt_custom = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=dt_arr,
            use_box_smearing=False
        )

        assert fdmt_custom.plan.is_custom_grid

        rng = np.random.default_rng(4242)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        sync_dense = fdmt_dense.execute(waterfall)
        sync_custom = fdmt_custom.execute(waterfall)

        assert sync_custom.shape == (fdmt_custom.dt_grid_final.size, fdmt_custom.plan.dmt_nsamps)

        for c_idx, dt in enumerate(fdmt_custom.dt_grid_final):
            d_idx = np.where(fdmt_dense.dt_grid_final == dt)[0][0]
            np.testing.assert_allclose(sync_custom[c_idx], sync_dense[d_idx], rtol=1e-5, atol=1e-5)

    def test_compute_fdmt_custom_grid(self) -> None:
        nchans = 64
        nsamples = 256

        rng = np.random.default_rng(999)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Test compute_fdmt with dt_arr
        dt_arr = np.array([10, 25, 40, 55], dtype=np.uint64)
        dmt_buf, plan = libdmt.compute_fdmt(
            waterfall, 1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=dt_arr
        )
        assert plan.is_custom_grid
        np.testing.assert_array_equal(plan.dt_grid_final, dt_arr)
        dmt = dmt_buf[: plan.dmt_size].reshape(plan.dt_grid_final.size, plan.dmt_nsamps)

        fdmt_dt = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=dt_arr
        )
        sync_dt = fdmt_dt.execute(waterfall)
        np.testing.assert_allclose(dmt, sync_dt, rtol=1e-5, atol=1e-5)

        # Test compute_fdmt with dm_arr
        dm_arr = np.array([5.0, 15.0, 30.0], dtype=np.float32)
        dmt_buf_dm, plan_dm = libdmt.compute_fdmt(
            waterfall, 1000.0, 1500.0, nchans, nsamples, 0.001, dm_arr=dm_arr
        )
        assert plan_dm.is_custom_grid
        dmt_dm = dmt_buf_dm[: plan_dm.dmt_size].reshape(plan_dm.dt_grid_final.size, plan_dm.dmt_nsamps)

        fdmt_dm = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dm_arr=dm_arr
        )
        sync_dm = fdmt_dm.execute(waterfall)
        np.testing.assert_allclose(dmt_dm, sync_dm, rtol=1e-5, atol=1e-5)

    def test_keyword_only_disambiguation_and_auto_sort(self) -> None:
        nchans = 64
        nsamples = 256

        # 1. Positional calling with grid must raise TypeError
        with pytest.raises(TypeError):
            libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, [5.0, 15.5, 30.0])

        with pytest.raises(TypeError):
            libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, [5.0, 15.5, 30.0])

        # 2. Both dt_grid and dt_arr are accepted keywords
        plan_dt_grid = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_grid=[60, 15, 45, 30]
        )
        assert plan_dt_grid.is_custom_grid
        np.testing.assert_array_equal(plan_dt_grid.dt_grid_final, [15, 30, 45, 60])
        np.testing.assert_array_equal(plan_dt_grid.get_dt_grid_final(), [15, 30, 45, 60])

        # 3. Auto-sorting and deduplication for dt_grid
        plan_unsorted = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=[50, 20, 100, 20]
        )
        np.testing.assert_array_equal(plan_unsorted.dt_grid_final, [20, 50, 100])
        assert plan_unsorted.dt_min == 20
        assert plan_unsorted.dt_max == 100

        # 4. Integer DM grid must be treated as DM, not dt
        # For 1000-1500 MHz, tsamp=0.001s: dm_conv is approx 0.435 -> dt for DM=10 is ~23, DM=30 is ~69
        plan_dm_int = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001, dm_grid=[30, 10, 20]
        )
        assert plan_dm_int.is_custom_grid
        # Verify it was converted from DM to dt (dt_min is around 23, NOT 10)
        assert plan_dm_int.dt_min > 15
        assert len(plan_dm_int.dt_grid_final) == 3
        # Monotonically increasing
        assert np.all(np.diff(plan_dm_int.dt_grid_final) > 0)
        assert np.all(np.diff(plan_dm_int.get_dm_grid_final()) > 0)

        # 5. FDMTCPU getters for dt and dm truth grids
        fdmt_cpu = libdmt.FDMTCPU(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_grid=[60, 15, 45, 30]
        )
        np.testing.assert_array_equal(fdmt_cpu.dt_grid_final, [15, 30, 45, 60])
        np.testing.assert_array_equal(fdmt_cpu.get_dt_grid_final(), [15, 30, 45, 60])
        assert len(fdmt_cpu.dm_grid_final) == 4
        assert len(fdmt_cpu.get_dm_grid_final()) == 4

        # 6. Error handling: conflicting options
        with pytest.raises(ValueError, match="Cannot provide both dt_grid and dm_grid"):
            libdmt.FDMTPlan(
                1000.0, 1500.0, nchans, nsamples, 0.001,
                dt_grid=[10, 20], dm_grid=[10.0, 20.0]
            )

        with pytest.raises(ValueError, match="Cannot provide both dt_grid and dt_arr"):
            libdmt.FDMTPlan(
                1000.0, 1500.0, nchans, nsamples, 0.001,
                dt_grid=[10, 20], dt_arr=[10, 20]
            )

        with pytest.raises(ValueError, match="Cannot provide both dm_grid and dm_arr"):
            libdmt.FDMTPlan(
                1000.0, 1500.0, nchans, nsamples, 0.001,
                dm_grid=[10.0, 20.0], dm_arr=[10.0, 20.0]
            )

    def test_negative_and_symmetric_dispersion(self) -> None:
        nchans = 64
        nsamples = 512
        f_min = 1000.0
        f_max = 1500.0
        tsamp = 0.001
        dt_max = 32
        dt_min = -32

        # 1. Symmetric range construction
        fdmt = libdmt.FDMTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min)
        assert fdmt.plan.dt_min == -32
        assert fdmt.plan.dt_max == 32
        grid = fdmt.dt_grid_final
        assert len(grid) == 65
        assert grid[0] == -32
        assert grid[-1] == 32

        # 2. Execution on constant input and symmetry check
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt = fdmt.execute(waterfall)
        assert dmt.shape == (65, fdmt.plan.dmt_nsamps)

        # In steady-state central region (e.g. t = 200), +dt and -dt should be equal by symmetry
        for i in range(len(grid) // 2):
            opp_i = len(grid) - 1 - i
            assert grid[i] == -grid[opp_i]
            np.testing.assert_allclose(dmt[i, 200], dmt[opp_i, 200], rtol=1e-5)

        # Without smearing, steady-state central value is strictly nchans
        fdmt_nosmear = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min,
            use_box_smearing=False
        )
        dmt_nosmear = fdmt_nosmear.execute(waterfall)
        for i in range(len(grid)):
            np.testing.assert_allclose(dmt_nosmear[i, 200], float(nchans), rtol=1e-5)

        # 3. Custom signed dt_grid
        signed_dts = [-20, -10, 0, 10, 20]
        fdmt_custom = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_grid=signed_dts
        )
        assert fdmt_custom.plan.is_custom_grid
        np.testing.assert_array_equal(fdmt_custom.dt_grid_final, signed_dts)
        dmt_custom = fdmt_custom.execute(waterfall)
        assert dmt_custom.shape[0] == 5

        # 4. Negative DM pulse recovery
        target_dt = -16
        t0 = 150
        df = (f_max - f_min) / nchans
        f_min_inv2 = f_min**-2
        f_max_inv2 = f_max**-2

        wf_pulse = np.zeros((nchans, nsamples), dtype=np.float32)
        for c in range(nchans):
            fc = f_min + (c + 0.5) * df
            tau = int(round(target_dt * (fc**-2 - f_max_inv2) / (f_min_inv2 - f_max_inv2)))
            t = t0 + tau
            if 0 <= t < nsamples:
                wf_pulse[c, t] = 10.0

        dmt_pulse = fdmt.execute(wf_pulse)
        best_dm_idx, best_t = np.unravel_index(np.argmax(dmt_pulse), dmt_pulse.shape)
        recovered_dt = grid[best_dm_idx]
        assert abs(recovered_dt - target_dt) <= 1
        assert abs(best_t - t0) <= 1
        assert dmt_pulse[best_dm_idx, best_t] >= nchans * 7.0

    def test_effective_variance_and_sigma(self) -> None:
        nchans = 64
        nsamples = 512
        f_min = 1000.0
        f_max = 1500.0
        tsamp = 0.001
        dt_max = 32
        dt_min = -32

        plan = libdmt.FDMTPlan(f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min)
        ndms = plan.dmt_ndms

        # 1. Un-smeared case (use_box_smearing=False)
        for w in [1, 2, 4, 8]:
            var_no_smear = plan.get_effective_variance(0, w, use_box_smearing=False)
            assert var_no_smear == pytest.approx(float(nchans * w))
            sig_no_smear = plan.get_effective_sigma(0, w, use_box_smearing=False)
            assert sig_no_smear == pytest.approx(np.sqrt(float(nchans * w)))

            var_grid = plan.get_effective_variance_grid(w, use_box_smearing=False)
            assert len(var_grid) == ndms
            np.testing.assert_allclose(var_grid, float(nchans * w))

        # FDMTCPU forwarding with use_box_smearing=False
        fdmt_no_smear = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min,
            use_box_smearing=False
        )
        assert fdmt_no_smear.get_effective_variance(0, 4) == pytest.approx(float(nchans * 4))
        np.testing.assert_allclose(
            fdmt_no_smear.get_effective_variance_grid(4),
            float(nchans * 4)
        )

        # 2. Smeared case (use_box_smearing=True)
        # At dt = 0 (index 32), smearing is 0 for all channels -> variance is nchans
        var_dt0 = plan.get_effective_variance(32, 1, use_box_smearing=True)
        assert var_dt0 == pytest.approx(float(nchans))

        # Across all DMs, variance >= nchans for W=1
        for i in range(ndms):
            assert plan.get_effective_variance(i, 1, use_box_smearing=True) >= float(nchans)
            var_w4 = plan.get_effective_variance(i, 4, use_box_smearing=True)
            assert 0 < var_w4 <= float(nchans * 4 * 4)

        # Symmetry: Var(+dt, W) == Var(-dt, W)
        for i in range(ndms // 2):
            opp_i = ndms - 1 - i
            var_neg = plan.get_effective_variance(i, 4, use_box_smearing=True)
            var_pos = plan.get_effective_variance(opp_i, 4, use_box_smearing=True)
            assert var_neg == pytest.approx(var_pos)

        # 3. Monte Carlo validation of variance formula
        fdmt_smeared = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min,
            use_box_smearing=True
        )
        rng = np.random.default_rng(12345)
        num_trials = 30
        emp_vars = []
        for _ in range(num_trials):
            noise = rng.standard_normal((nchans, nsamples), dtype=np.float32)
            out = fdmt_smeared.execute(noise)
            emp_vars.append(np.var(out[:, 100:400], axis=1))
        empirical_var = np.mean(emp_vars, axis=0)
        analytical_var = np.array(fdmt_smeared.get_effective_variance_grid(1))
        ratio = empirical_var / analytical_var
        np.testing.assert_allclose(ratio, 1.0, rtol=0.05)

        # 4. Error handling
        with pytest.raises((ValueError, RuntimeError)):
            plan.get_effective_variance(0, 0)
        with pytest.raises((ValueError, RuntimeError)):
            plan.get_effective_sigma(0, 0)
        with pytest.raises((IndexError, RuntimeError)):
            plan.get_effective_variance(1000, 1)
        with pytest.raises((IndexError, RuntimeError)):
            plan.get_effective_sigma(1000, 1)

    def test_purely_negative_dt_range(self) -> None:
        # Regression test for a range with no positive/zero trials at all --
        # promised in the original implementation plan but never delivered.
        f_min, f_max = 1000.0, 1500.0
        nchans, nsamples, tsamp = 64, 256, 0.001
        dt_max, dt_min = -10, -50

        fdmt = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min,
            use_box_smearing=False,
        )
        assert fdmt.plan.dt_min == -50
        assert fdmt.plan.dt_max == -10
        grid = fdmt.dt_grid_final
        assert len(grid) == 41
        assert grid[0] == -50
        assert grid[-1] == -10
        assert np.all(grid <= 0)

        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt = fdmt.execute(waterfall)
        np.testing.assert_allclose(dmt[:, 200], float(nchans), rtol=1e-5)

    @pytest.mark.parametrize("nchans", [2, 3, 4, 5, 7])
    def test_box_smearing_false_coarse_channelization(self, nchans: int) -> None:
        # Regression test: with few channels relative to dt_max, a single
        # channel's own bandwidth induces a level-0 dt range spanning more
        # than one row. An earlier refactor of the level-0 init dispatcher
        # accidentally copied the *unshifted* raw waterfall into every such
        # row when use_box_smearing=False, instead of shifting each row by
        # its own |dt| samples -- this exercises exactly that path via
        # add_frb_track's exact-recovery check.
        f_min, f_max = 1000.0, 1500.0
        nsamples, tsamp, dt_max = 256, 0.001, 16

        plan = libdmt.FDMTPlan(f_min, f_max, nchans, nsamples, tsamp, dt_max, 0)
        ndms = plan.dmt_ndms
        assert ndms > 1
        target_idx = ndms // 2
        toffset = 150

        waterfall = np.zeros((nchans, nsamples), dtype=np.float32)
        libdmt.add_frb_track(waterfall, plan, target_idx, amplitude=1.0, toffset=toffset)

        fdmt = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, 0,
            use_box_smearing=False,
        )
        dmt = fdmt.execute(waterfall)
        assert dmt[target_idx, toffset] == pytest.approx(float(nchans))

    @pytest.mark.parametrize(
        ("nchans", "dt_max", "dt_min", "target_dt"),
        [
            (2, 16, 0, 1),
            (8, 16, 0, 4),
            (64, 32, 0, 10),
            (65, 32, 0, 10),
            (64, 32, -32, -16),
            (64, 32, -32, 16),
            (64, 32, -32, 0),
            (64, 32, -32, -32),
            (64, 32, -32, 32),
        ],
    )
    @pytest.mark.parametrize("use_box_smearing", [False, True])
    def test_add_frb_track_and_trace_dm_exact_recovery(
        self, nchans: int, dt_max: int, dt_min: int, target_dt: int,
        use_box_smearing: bool,
    ) -> None:
        # add_frb_track places a unit impulse in every channel at exactly
        # the sample trace_dm computes; running FDMT on that waterfall must
        # reproduce a value of exactly nchans (all channels combining
        # constructively, no partial/rounding loss) at (dm_idx, toffset).
        f_min, f_max = 1000.0, 1500.0
        nsamples, tsamp, toffset = 512, 0.001, 150

        plan = libdmt.FDMTPlan(f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min)
        grid = plan.dt_grid_final
        dm_idx = int(np.where(grid == target_dt)[0][0])

        waterfall = np.zeros((nchans, nsamples), dtype=np.float32)
        libdmt.add_frb_track(waterfall, plan, dm_idx, amplitude=1.0, toffset=toffset)

        fdmt = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min,
            use_box_smearing=use_box_smearing,
        )
        dmt = fdmt.execute(waterfall)
        assert dmt[dm_idx, toffset] == pytest.approx(float(nchans))

    def test_add_frb_track_width(self) -> None:
        # A width-N boxcar dedisperses to a width-N plateau at the target
        # DM, not a taller single spike -- amplitude is per-sample, not
        # accumulated across the width.
        f_min, f_max = 1000.0, 1500.0
        nchans, nsamples, tsamp, dt_max = 64, 512, 0.001, 32
        toffset, width = 150, 3

        plan = libdmt.FDMTPlan(f_min, f_max, nchans, nsamples, tsamp, dt_max, 0)
        grid = plan.dt_grid_final
        dm_idx = int(np.where(grid == 5)[0][0])

        waterfall = np.zeros((nchans, nsamples), dtype=np.float32)
        libdmt.add_frb_track(
            waterfall, plan, dm_idx, amplitude=1.0, toffset=toffset, width=width,
        )
        fdmt = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, 0,
            use_box_smearing=False,
        )
        dmt = fdmt.execute(waterfall)
        np.testing.assert_allclose(
            dmt[dm_idx, toffset:toffset + width], float(nchans),
        )
        assert dmt[dm_idx, toffset - 1] == 0.0
        assert dmt[dm_idx, toffset + width] == 0.0

    def test_valid_mode_seamless_streaming(self) -> None:
        # Seamless cross-block streaming in mode="valid": m_tree_history caches
        # trailing samples at each tree merge node, enabling bit-exact match
        # with monolithic continuous execution across consecutive blocks.
        f_min, f_max = 1000.0, 1500.0
        nchans, tsamp = 32, 0.001
        dt_max, dt_min = 32, -32
        block_size, n_blocks = 128, 5
        total = block_size * n_blocks

        rng = np.random.default_rng(7)
        waterfall = rng.uniform(1.0, 24.0, size=(nchans, total)).astype(np.float32)

        fdmt_full = libdmt.FDMTCPU(
            f_min, f_max, nchans, total, tsamp, dt_max, dt_min,
            use_box_smearing=False, mode="full",
        )
        dmt_full = fdmt_full.execute(waterfall)

        fdmt_valid = libdmt.FDMTCPU(
            f_min, f_max, nchans, block_size, tsamp, dt_max, dt_min,
            use_box_smearing=False, mode="valid",
        )
        assert fdmt_valid.plan.tree_history_size > 0
        ndms = fdmt_valid.plan.dmt_ndms
        streamed = np.zeros((ndms, total), dtype=np.float32)
        for b in range(n_blocks):
            block = waterfall[:, b * block_size:(b + 1) * block_size]
            streamed[:, b * block_size:(b + 1) * block_size] = fdmt_valid.execute(block)

        abs_dt_max = max(abs(dt_min), abs(dt_max))
        np.testing.assert_allclose(
            streamed[:, abs_dt_max:],
            dmt_full[:, abs_dt_max:total],
            rtol=1e-5,
            atol=1e-5,
        )

        # Test reset_history()
        fdmt_valid.reset_history()

    @staticmethod
    def _stream_blocks(fdmt, waterfall, block_size, n_blocks):
        ndms = fdmt.plan.dmt_ndms
        total = block_size * n_blocks
        streamed = np.zeros((ndms, total), dtype=np.float32)
        for b in range(n_blocks):
            block = waterfall[:, b * block_size:(b + 1) * block_size]
            streamed[:, b * block_size:(b + 1) * block_size] = fdmt.execute(block)
        return streamed

    @pytest.mark.parametrize(
        ("nchans", "block_size", "n_blocks"),
        [
            (32, 128, 5),   # box smearing + tree history combined (baseline)
            (13, 128, 8),   # odd nchans
            (63, 100, 10),  # odd nchans, non-round block size
            (32, 128, 20),  # long chain
            (32, 40, 15),   # many small blocks
        ],
    )
    def test_valid_mode_streaming_stress(
        self, nchans: int, block_size: int, n_blocks: int,
    ) -> None:
        f_min, f_max, tsamp = 1000.0, 1500.0, 0.001
        dt_max, dt_min = 32, -32
        total = block_size * n_blocks

        rng = np.random.default_rng(7)
        waterfall = rng.uniform(1.0, 24.0, size=(nchans, total)).astype(np.float32)

        fdmt_full = libdmt.FDMTCPU(
            f_min, f_max, nchans, total, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="full",
        )
        dmt_full = fdmt_full.execute(waterfall)

        fdmt_valid = libdmt.FDMTCPU(
            f_min, f_max, nchans, block_size, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="valid",
        )
        streamed = self._stream_blocks(fdmt_valid, waterfall, block_size, n_blocks)

        abs_dt_max = max(abs(dt_min), abs(dt_max))
        np.testing.assert_allclose(
            streamed[:, abs_dt_max:], dmt_full[:, abs_dt_max:total],
            rtol=1e-5, atol=1e-3,
        )

    def test_reset_history_matches_fresh_instance(self) -> None:
        f_min, f_max = 1000.0, 1500.0
        nchans, nsamps, tsamp = 32, 128, 0.001
        dt_max, dt_min = 32, -32

        rng = np.random.default_rng(3)
        block1 = rng.uniform(1.0, 24.0, size=(nchans, nsamps)).astype(np.float32)
        block2 = rng.uniform(1.0, 24.0, size=(nchans, nsamps)).astype(np.float32)

        fdmt = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="valid",
        )
        fdmt.execute(block1)
        out_with_history = fdmt.execute(block2).copy()

        fdmt.reset_history()
        out_after_reset = fdmt.execute(block2)

        fdmt_fresh = libdmt.FDMTCPU(
            f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="valid",
        )
        out_fresh = fdmt_fresh.execute(block2)

        np.testing.assert_allclose(out_after_reset, out_fresh)
        # History must have actually been doing something (else this test
        # can't distinguish a working reset from a no-op history).
        assert not np.allclose(out_with_history, out_after_reset)

    @pytest.mark.parametrize("target_dt", [16, -16, 0, 32, -32])
    @pytest.mark.parametrize("toffset_delta", [-5, 0, 5, 125])
    def test_add_frb_track_across_valid_mode_block_boundary(
        self, target_dt: int, toffset_delta: int,
    ) -> None:
        f_min, f_max = 1000.0, 1500.0
        nchans, tsamp = 64, 0.001
        dt_max, dt_min = 32, -32
        block_size, n_blocks = 128, 4
        total = block_size * n_blocks
        toffset = block_size + toffset_delta

        plan = libdmt.FDMTPlan(f_min, f_max, nchans, total, tsamp, dt_max, dt_min)
        grid = plan.dt_grid_final
        dm_idx = int(np.where(grid == target_dt)[0][0])

        waterfall = np.zeros((nchans, total), dtype=np.float32)
        libdmt.add_frb_track(waterfall, plan, dm_idx, amplitude=1.0, toffset=toffset)

        fdmt_valid = libdmt.FDMTCPU(
            f_min, f_max, nchans, block_size, tsamp, dt_max, dt_min,
            use_box_smearing=False, mode="valid",
        )
        streamed = self._stream_blocks(fdmt_valid, waterfall, block_size, n_blocks)
        assert streamed[dm_idx, toffset] == pytest.approx(float(nchans))

    def test_stepper_partial_advance_across_streamed_blocks(self) -> None:
        # "Stop 1-2 levels before root to inspect sub-bands" must not break
        # cross-block tree history, as long as finalize() always completes
        # the block before the next reset().
        f_min, f_max = 1000.0, 1500.0
        nchans, tsamp = 32, 0.001
        dt_max, dt_min = 32, -32
        block_size, n_blocks = 128, 6
        total = block_size * n_blocks

        rng = np.random.default_rng(11)
        waterfall = rng.uniform(1.0, 24.0, size=(nchans, total)).astype(np.float32)

        fdmt_full = libdmt.FDMTCPU(
            f_min, f_max, nchans, total, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="full",
        )
        dmt_full = fdmt_full.execute(waterfall)

        fdmt_valid = libdmt.FDMTCPU(
            f_min, f_max, nchans, block_size, tsamp, dt_max, dt_min,
            use_box_smearing=True, mode="valid",
        )
        ndms = fdmt_valid.plan.dmt_ndms
        streamed = np.zeros((ndms, total), dtype=np.float32)
        for b in range(n_blocks):
            block = waterfall[:, b * block_size:(b + 1) * block_size]
            dmt_buf = np.zeros(fdmt_valid.plan.buffer_size, dtype=np.float32)
            fdmt_valid.reset(block, dmt_buf)
            fdmt_valid.advance_until_remaining(2)
            assert fdmt_valid.num_subbands == 4
            for s in range(4):
                _ = fdmt_valid.view_subband(s)
            fdmt_valid.finalize()
            streamed[:, b * block_size:(b + 1) * block_size] = (
                dmt_buf[:ndms * block_size].reshape(ndms, block_size)
            )

        abs_dt_max = max(abs(dt_min), abs(dt_max))
        np.testing.assert_allclose(
            streamed[:, abs_dt_max:], dmt_full[:, abs_dt_max:total],
            rtol=1e-5, atol=1e-3,
        )


