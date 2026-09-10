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



