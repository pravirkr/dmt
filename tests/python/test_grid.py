from __future__ import annotations

import numpy as np
import pytest

import dmtlib
from dmtlib import (
    FDMT,
    DDMTPlan,
    FDMTPlan,
    calculate_snr_loss,
    generate_dm_grid,
    generate_levin_dm_grid,
    grid,
)

LBAND = (1100.0, 1714.0, 1152, 455.6e-6)
UWL = (704.0, 4032.0, 6656, 64e-6)


def midpoints(grid: np.ndarray) -> np.ndarray:
    return 0.5 * (grid[:-1] + grid[1:])


class TestGenerateDmGrid:
    def test_endpoints_and_monotonic(self) -> None:
        grid = generate_dm_grid(*LBAND, 0.0, 1000.0)
        assert grid.dtype == np.float64
        assert grid[0] == 0.0
        assert grid[-1] == 1000.0
        assert np.all(np.diff(grid) > 0)

    def test_single_point(self) -> None:
        np.testing.assert_array_equal(generate_dm_grid(*LBAND, 5.0, 5.0), [5.0])

    def test_reference_count(self) -> None:
        # DM 0-2000, pulse_width one sample, 5% loss.
        tsamp = UWL[3]
        grid = generate_dm_grid(*UWL, 0.0, 2000.0, pulse_width=tsamp)
        assert grid.size == 12209
        coh = generate_dm_grid(*UWL, 0.0, 2000.0, pulse_width=tsamp, chan_dm=413.0)
        assert coh.size == 17809

    @pytest.mark.parametrize("max_snr_loss", [0.05, 0.2])
    @pytest.mark.parametrize("chan_dm", [0.0, 413.0])
    def test_midpoints_lose_max_snr_loss(
        self, max_snr_loss: float, chan_dm: float
    ) -> None:
        grid = generate_dm_grid(
            *LBAND, 0.0, 1000.0, max_snr_loss=max_snr_loss, chan_dm=chan_dm
        )
        loss = calculate_snr_loss(midpoints(grid)[:-1], grid, *LBAND, chan_dm=chan_dm)
        assert loss.max() <= max_snr_loss * 1.03
        assert loss.min() >= max_snr_loss * 0.8

    def test_tol_overrides_max_snr_loss(self) -> None:
        by_tol = generate_dm_grid(*LBAND, 0.0, 500.0, tol=1.25, max_snr_loss=0.5)
        by_loss = generate_dm_grid(*LBAND, 0.0, 500.0, max_snr_loss=1 - 1 / 1.25)
        np.testing.assert_array_equal(by_tol, by_loss)

    def test_wider_pulses_need_fewer_trials(self) -> None:
        narrow = generate_dm_grid(*LBAND, 0.0, 1000.0)
        wide = generate_dm_grid(*LBAND, 0.0, 1000.0, pulse_width=5e-3)
        assert wide.size < narrow.size

    def test_chan_dm_makes_trials_denser_around_it(self) -> None:
        plain = generate_dm_grid(*UWL, 0.0, 1100.0)
        coh = generate_dm_grid(*UWL, 0.0, 1100.0, chan_dm=413.0)
        assert coh.size > plain.size

        def step(g: np.ndarray, dm: float) -> float:
            i = np.searchsorted(g, dm)
            return float(g[i + 1] - g[i])

        assert step(coh, 413.0) < 0.5 * step(plain, 413.0)
        assert step(coh, 213.0) == pytest.approx(step(coh, 613.0), rel=0.05)

    def test_negative_dms_mirror(self) -> None:
        both = generate_dm_grid(*LBAND, -100.0, 100.0)
        np.testing.assert_array_equal(both, -both[::-1])
        assert 0.0 in both
        pos = generate_dm_grid(*LBAND, 10.0, 300.0)
        neg = generate_dm_grid(*LBAND, -300.0, -10.0)
        np.testing.assert_array_equal(neg, -pos[::-1])
        to_zero = generate_dm_grid(*LBAND, -100.0, 0.0)
        assert to_zero[-1] == 0.0
        assert not np.signbit(to_zero[-1])

    def test_kill_mask(self) -> None:
        nchans = LBAND[2]
        plain = generate_dm_grid(*LBAND, 0.0, 1000.0)
        keep = np.ones(nchans, dtype=bool)
        np.testing.assert_array_equal(
            generate_dm_grid(*LBAND, 0.0, 1000.0, kill_mask=keep), plain
        )
        keep[: nchans // 2] = False
        top = generate_dm_grid(*LBAND, 0.0, 1000.0, kill_mask=keep)
        assert top.size < plain.size
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, 10.0, kill_mask=np.zeros(nchans, bool))
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, 10.0, kill_mask=np.ones(nchans - 1, bool))

    @pytest.mark.parametrize("chan_dm", [0.0, 413.0])
    def test_piecewise_runs_ddmt_fft_by_nufft(self, chan_dm: float) -> None:
        f_min, f_max, nchans, tsamp = LBAND
        cont = generate_dm_grid(*LBAND, 0.0, 2000.0, chan_dm=chan_dm)
        pw = generate_dm_grid(
            *LBAND, 0.0, 2000.0, chan_dm=chan_dm, piecewise=True, min_run=32
        )
        assert pw[0] == 0.0
        assert pw[-1] >= 2000.0
        # Never coarser than the continuous grid (up to its interpolation).
        lc = calculate_snr_loss(midpoints(pw), cont, *LBAND, chan_dm=chan_dm)
        lp = calculate_snr_loss(midpoints(pw), pw, *LBAND, chan_dm=chan_dm)
        assert np.all(lp <= np.maximum(lc, 0.05 * 1.001))
        steps = np.diff(pw)
        segments = np.split(steps, np.nonzero(np.abs(np.diff(steps)) > 1e-9)[0] + 1)
        assert all(s.size >= 32 for s in segments)
        eng = dmtlib.DDMTFFT(f_min, f_max, nchans, tsamp, dm_arr=pw.astype(np.float32))
        assert eng.method_used in ("nufft", "piecewise_nufft")

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 10.0, 5.0)
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, np.nan)
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, 10.0, max_snr_loss=1.0)
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, 10.0, tol=0.9)
        with pytest.raises(ValueError):
            generate_dm_grid(*LBAND, 0.0, 10.0, pulse_width=-1.0)
        with pytest.raises(TypeError):
            generate_dm_grid(*LBAND, 0.0, 10.0, 1e-3)  # pulse_width is keyword-only

    def test_fdmt_runs_on_the_grid(self) -> None:
        f_min, f_max, nchans, tsamp = 1000.0, 1500.0, 64, 1e-3
        dms = generate_dm_grid(f_min, f_max, nchans, tsamp, 0.0, 80.0)
        plan = FDMTPlan(f_min, f_max, nchans, 512, tsamp, dm_grid=dms)
        assert plan.is_custom_grid
        fdmt = FDMT(plan)
        wf = np.random.default_rng(1).standard_normal((nchans, 512), dtype=np.float32)
        out = fdmt.execute(wf)
        assert out.shape == (plan.dt_grid_final.size, plan.dmt_nsamps)
        assert np.all(np.isfinite(out))


class TestGenerateLevinDmGrid:
    @pytest.mark.parametrize("piecewise", [False, True])
    def test_matches_ddmt_plan(self, piecewise: bool) -> None:
        args = (0.0, 2000.0, 6.4e-5, 1e-4, 1100.0, 1500.0, 1024, 1.2)
        want = (
            DDMTPlan.generate_levin_dm_grid_piecewise(*args)
            if piecewise
            else DDMTPlan.generate_levin_dm_grid(*args)
        )
        got = generate_levin_dm_grid(
            1100.0,
            1500.0,
            1024,
            6.4e-5,
            0.0,
            2000.0,
            pulse_width=1e-4,
            tol=1.2,
            piecewise=piecewise,
        )
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, np.asarray(want, dtype=np.float64))

    def test_default_tol_and_limits(self) -> None:
        grid = generate_levin_dm_grid(*LBAND, 0.0, 500.0)
        np.testing.assert_array_equal(
            grid, generate_levin_dm_grid(*LBAND, 0.0, 500.0, tol=1.25)
        )
        assert grid[-1] >= 500.0
        assert generate_levin_dm_grid(*LBAND, 0.0, 500.0, tol=1.1).size > grid.size
        with pytest.raises(ValueError):
            generate_levin_dm_grid(*LBAND, -5.0, 500.0)


class TestCalculateSnrLoss:
    def test_scalar_and_array(self) -> None:
        trials = np.array([100.0, 110.0])
        assert calculate_snr_loss(100.0, trials, *LBAND) == 0.0
        loss = calculate_snr_loss(105.0, trials, *LBAND)
        assert isinstance(loss, float)
        assert 0.0 < loss < 1.0
        arr = calculate_snr_loss(
            np.array([[101.0, 103.0], [105.0, 107.0]]), trials, *LBAND
        )
        assert arr.shape == (2, 2)
        assert arr[0, 0] < arr[0, 1] < arr[1, 0]
        assert arr[1, 1] == pytest.approx(arr[0, 1], abs=1e-3)

    def test_wider_pulses_lose_less(self) -> None:
        trials = np.array([100.0, 110.0])
        narrow = calculate_snr_loss(105.0, trials, *LBAND)
        wide = calculate_snr_loss(105.0, trials, *LBAND, pulse_width=5e-3)
        assert wide < narrow

    def test_invalid(self) -> None:
        with pytest.raises(ValueError):
            calculate_snr_loss(105.0, np.array([110.0, 100.0]), *LBAND)
        with pytest.raises(ValueError):
            calculate_snr_loss(105.0, np.array([100.0]), 1100.0, 1714.0, 1, 455.6e-6)
        with pytest.raises(ValueError):
            calculate_snr_loss(
                105.0,
                np.array([100.0]),
                1100.0,
                1714.0,
                4,
                455.6e-6,
                kill_mask=[0, 1, 0, 0],
            )
        with pytest.raises(TypeError):
            calculate_snr_loss(105.0, np.array([100.0]), *LBAND, 1.0)  # keyword-only


def test_grid_edge_regressions() -> None:
    with pytest.raises(ValueError):
        generate_dm_grid(1100.0, 1714.0, 1, 455.6e-6, 0.0, 10.0)
    with pytest.raises(ValueError):
        generate_dm_grid(1100.0, 1714.0, 4, 455.6e-6, 0.0, 10.0, kill_mask=[0, 1, 0, 0])
    with pytest.raises(ValueError):
        generate_dm_grid(*LBAND, 0.0, 10.0, max_snr_loss=1e-18)

    for dm_min in (0.0, -0.01):
        pw = generate_dm_grid(*LBAND, dm_min, 200.0, piecewise=True, min_run=32)
        assert pw[-1] >= 200.0
        assert np.all(np.diff(pw.astype(np.float32)) > 0)

    plan_coh = FDMTPlan(1100.0, 1714.0, 32, 512, 455.6e-6, 32, 0, 1, "valid", 0, 100.0)
    with pytest.raises(ValueError, match="chan_dm == 0"):
        dmtlib.sensitivity.fdmt_alignment_error(plan_coh, 0)


def test_grid_module_and_removed_names() -> None:
    assert grid.generate_dm_grid is generate_dm_grid
    assert grid.generate_levin_dm_grid is generate_levin_dm_grid
    assert grid.calculate_snr_loss is calculate_snr_loss
    assert pytest.approx(1 / 2.41e-4) == grid.DM_CONSTANT
    for name in (
        "generate_optimal_dm_grid",
        "generate_optimal_dt_grid",
        "generate_dt_grid",
        "DMGridMethod",
    ):
        assert not hasattr(dmtlib, name)
