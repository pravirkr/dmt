import numpy as np
import pytest

from dmtlib import libdmt


class TestPlans:
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
            1000.0,
            1500.0,
            nchans,
            nsamples,
            0.001,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_step=dt_step,
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

    def test_plan_arbitrary_dt_and_dm_grids(self) -> None:
        nchans = 64
        nsamples = 256

        # Arbitrary dt_arr
        dt_arr = np.array([15, 30, 45, 60], dtype=np.uint64)
        plan_dt = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001, dt_arr=dt_arr
        )
        assert plan_dt.is_custom_grid
        assert plan_dt.dt_min == 15
        assert plan_dt.dt_max == 60
        np.testing.assert_array_equal(plan_dt.dt_grid_final, dt_arr)

        # Standard plan is not custom grid
        plan_std = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max=60)
        assert not plan_std.is_custom_grid

        # Arbitrary dm_arr
        dm_arr = np.array([5.0, 15.0, 30.0], dtype=np.float32)
        plan_dm = libdmt.FDMTPlan(
            1000.0, 1500.0, nchans, nsamples, 0.001, dm_arr=dm_arr
        )
        assert plan_dm.is_custom_grid
        assert len(plan_dm.dt_grid_final) > 0
        assert len(plan_dm.dm_grid_final) > 0

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
        np.testing.assert_array_equal(
            plan_dt_grid.get_dt_grid_final(), [15, 30, 45, 60]
        )

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
                1000.0,
                1500.0,
                nchans,
                nsamples,
                0.001,
                dt_grid=[10, 20],
                dm_grid=[10.0, 20.0],
            )

        with pytest.raises(ValueError, match="Cannot provide both dt_grid and dt_arr"):
            libdmt.FDMTPlan(
                1000.0,
                1500.0,
                nchans,
                nsamples,
                0.001,
                dt_grid=[10, 20],
                dt_arr=[10, 20],
            )

        with pytest.raises(ValueError, match="Cannot provide both dm_grid and dm_arr"):
            libdmt.FDMTPlan(
                1000.0,
                1500.0,
                nchans,
                nsamples,
                0.001,
                dm_grid=[10.0, 20.0],
                dm_arr=[10.0, 20.0],
            )

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
            f_min,
            f_max,
            nchans,
            nsamples,
            tsamp,
            dt_max,
            dt_min,
            use_box_smearing=False,
        )
        assert fdmt_no_smear.get_effective_variance(0, 4) == pytest.approx(
            float(nchans * 4)
        )
        np.testing.assert_allclose(
            fdmt_no_smear.get_effective_variance_grid(4), float(nchans * 4)
        )

        # 2. Smeared case (use_box_smearing=True)
        # At dt = 0 (index 32), smearing is 0 for all channels -> variance is nchans
        var_dt0 = plan.get_effective_variance(32, 1, use_box_smearing=True)
        assert var_dt0 == pytest.approx(float(nchans))

        # Across all DMs, variance >= nchans for W=1
        for i in range(ndms):
            assert plan.get_effective_variance(i, 1, use_box_smearing=True) >= float(
                nchans
            )
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
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min, use_box_smearing=True
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

    def test_plan_introspection_and_complexity(self) -> None:
        nchans = 64
        nsamples = 1024
        tsamp = 0.001
        dt_max = 64
        plan = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, tsamp, dt_max)

        # 1. Iteration breakdowns
        ops_by_iter = plan.operations_by_iteration
        nodes_by_iter = plan.nodes_by_iteration
        assert len(ops_by_iter) == plan.niters + 1
        assert len(nodes_by_iter) == plan.niters + 1
        assert all(op >= 0 for op in ops_by_iter)
        assert all(node >= 0 for node in nodes_by_iter)

        # 2. Total operations and FLOPs
        assert plan.total_operations == sum(ops_by_iter)
        assert plan.total_flops == plan.total_operations * plan.nsamps
        expected_gflops = plan.total_operations / (plan.tsamp * 1e9)
        assert plan.theoretical_gflops == pytest.approx(expected_gflops, rel=1e-4)

        # 3. Box smearing toggle
        ops_no_smear = plan.get_operations_by_iteration(use_box_smearing=False)
        assert ops_no_smear[0] == 0
        assert plan.get_total_operations(use_box_smearing=False) == sum(ops_no_smear)

        # 4. State shapes
        shapes = plan.state_shapes
        assert len(shapes) == plan.niters + 1
        assert shapes[0]["nchans"] == nchans
        assert shapes[-1]["nchans"] == 1
        assert shapes[-1]["nsamps"] == nsamples

        # 5. Complexity struct
        comp = plan.complexity
        assert len(comp.operations_by_iteration) == plan.niters + 1
        assert len(comp.nodes_by_iteration) == plan.niters + 1
        assert "Operations by iteration" in comp.to_string()
        assert comp.sum_additions > 0
        assert comp.ops_ratio > 1.0

    def test_plan_introspection_level0_ops_with_nonzero_dt_min(self) -> None:
        # Regression test: level-0 box-smearing op count must reflect the
        # actual steady-state cost of fdmt_init_impl_row0/fdmt_init_impl_row
        # (lib/fdmt.cpp) -- (ndt_c - 1) additions per channel, plus 2 more
        # (a fixed-width sliding sum: 1 add + 1 sub) only when that
        # channel's own local dt_grid doesn't start at 0. A prior version of
        # this formula used abs(dt_grid.front()) directly as a per-sample
        # cost, which only happened to be correct at dt_min=0 (see
        # test_plan_introspection_and_complexity above).
        nchans = 64
        nsamples = 256
        dt_max = 64
        dt_min = 16
        plan = libdmt.FDMTPlan(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max, dt_min)

        expected_lvl0_ops = 0
        old_buggy_lvl0_ops = 0
        any_nonzero_front = False
        for grid in plan.container.dt_grids[0]:
            dt_grid = grid.dt_grid
            if len(dt_grid) == 0:
                continue
            front = dt_grid[0]
            any_nonzero_front |= front != 0
            expected_lvl0_ops += (len(dt_grid) - 1) + (2 if front != 0 else 0)
            old_buggy_lvl0_ops += abs(front) + (len(dt_grid) - 1)

        # Sanity: this dt_min actually exercises the nonzero-front case, and
        # the old (buggy) formula would have given a different answer here
        # -- otherwise this test wouldn't actually be covering the bug.
        assert any_nonzero_front
        assert expected_lvl0_ops != old_buggy_lvl0_ops

        ops_by_iter = plan.get_operations_by_iteration(use_box_smearing=True)
        assert ops_by_iter[0] == expected_lvl0_ops
        assert plan.get_total_operations(use_box_smearing=True) == sum(ops_by_iter)
