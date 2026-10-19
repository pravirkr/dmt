"""Tests of `dmtlib.sensitivity`: exact injector, noise models, single-trial
engines, the plan emulation of the FDMT and the detection statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest

import dmtlib
from dmtlib import sensitivity as sens

try:
    from ._dispersed_pulse import make_dispersed_pulse
except ImportError:
    from _dispersed_pulse import make_dispersed_pulse

F_MIN, F_MAX, NCHANS, TSAMP, DM_MAX = 704.0, 1216.0, 64, 1e-3, 50.0
CFG = sens.SearchConfig(F_MIN, F_MAX, NCHANS, TSAMP, dm_max=DM_MAX, name="test")
ALL = sens.ENGINES + sens.VARIANTS


def waterfall(nsamps: int) -> np.ndarray:
    return np.zeros((NCHANS, nsamps), dtype=np.float64)


class TestInjector:
    @pytest.mark.parametrize(
        ("dm", "t_arr", "width"),
        [(0.0, 40.0, 0.0), (7.3, 40.6, 0.0), (21.9, 40.2, 2.5), (DM_MAX, 41.0, 16)],
    )
    def test_every_channel_sums_to_fluence(
        self, dm: float, t_arr: float, width: float
    ) -> None:
        wf = waterfall(512)
        ssq = sens.inject_pulse(wf, F_MIN, F_MAX, TSAMP, dm, t_arr, width, 2.5)
        np.testing.assert_allclose(wf.sum(axis=1), 2.5, rtol=1e-9)
        assert ssq == pytest.approx(float(np.sum(wf**2)))

    def test_no_dispersion_is_a_boxcar(self) -> None:
        wf = waterfall(64)
        sens.inject_pulse(wf, F_MIN, F_MAX, TSAMP, 0.0, 10.0, 4.0)
        expected = np.zeros(64)
        expected[10:14] = 0.25
        np.testing.assert_allclose(wf, np.broadcast_to(expected, wf.shape), atol=1e-12)
        half = waterfall(64)
        sens.inject_pulse(half, F_MIN, F_MAX, TSAMP, 0.0, 10.5, 1.0)
        np.testing.assert_allclose(half[:, 10:12], 0.5, atol=1e-12)
        delta = waterfall(64)
        sens.inject_pulse(delta, F_MIN, F_MAX, TSAMP, 0.0, 10.0, 0.0)
        np.testing.assert_array_equal(np.flatnonzero(delta[0]), [10])

    def test_narrow_boxcar_tends_to_delta(self) -> None:
        a, b = waterfall(512), waterfall(512)
        sens.inject_pulse(a, F_MIN, F_MAX, TSAMP, 17.7, 30.3, 0.0)
        sens.inject_pulse(b, F_MIN, F_MAX, TSAMP, 17.7, 30.3, 1e-7)
        np.testing.assert_allclose(b, a, atol=1e-5)

    def test_integer_shift_is_a_roll(self) -> None:
        a, b = waterfall(512), waterfall(512)
        sens.inject_pulse(a, F_MIN, F_MAX, TSAMP, 15.5, 20.25, 3.0)
        sens.inject_pulse(b, F_MIN, F_MAX, TSAMP, 15.5, 27.25, 3.0)
        np.testing.assert_allclose(np.roll(a, 7, axis=1), b, atol=1e-12)

    def test_centroid_follows_the_dispersion_law(self) -> None:
        dm, t_arr, width = 30.0, 12.3, 2.0
        wf = waterfall(512)
        sens.inject_pulse(wf, F_MIN, F_MAX, TSAMP, dm, t_arr, width)
        edges = np.linspace(F_MIN, F_MAX, NCHANS + 1)
        f_lo, f_hi = edges[:-1], edges[1:]
        # Mean arrival over a flat-spectrum channel: k DM (1/(f_lo f_hi) -
        # f_max^-2) / tsamp after t_arr, plus half the intrinsic width.
        k = dmtlib.simulate.DM_CONSTANT * dm / TSAMP
        mean = k * (1 / (f_lo * f_hi) - F_MAX**-2)
        centroid = (wf * (np.arange(512) + 0.5)).sum(axis=1) / wf.sum(axis=1)
        np.testing.assert_allclose(centroid, t_arr + mean + width / 2, atol=0.02)
        _, smear = sens.channel_sweep(F_MIN, F_MAX, NCHANS, TSAMP, dm)
        assert smear[0] == pytest.approx(CFG.bottom_smear(dm))
        assert smear[0] > smear[-1] > 0

    def test_matches_first_order_model(self) -> None:
        dm = 25.0
        exact = waterfall(512)
        sens.inject_pulse(exact, F_MIN, F_MAX, TSAMP, dm, 30.4, 0.0)
        dt = sens.band_delay(dm, F_MIN, F_MAX, TSAMP)
        approx = make_dispersed_pulse(NCHANS, 512, F_MIN, F_MAX, dt, 30.4)
        # The first-order model spreads each channel uniformly in time; the
        # exact density varies by ~3 df / f across the sweep.
        np.testing.assert_allclose(exact, approx, atol=0.03)

    def test_constant_is_the_library_one(self) -> None:
        assert dmtlib.simulate.DM_CONSTANT == pytest.approx(1 / 2.41e-4)
        assert CFG.dm_per_dt * CFG.band_delay(1.0) == pytest.approx(1.0, rel=1e-6)

    def test_out_of_block_raises(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            sens.inject_pulse(waterfall(64), F_MIN, F_MAX, TSAMP, 40.0, 10.0)
        with pytest.raises(ValueError, match="dm must be"):
            sens.channel_sweep(F_MIN, F_MAX, NCHANS, TSAMP, -1.0)


class TestSearch:
    def test_boxcar_widths(self) -> None:
        widths = sens.boxcar_widths(40)
        assert list(widths[:16]) == list(range(1, 17))
        assert widths[-1] >= 40
        assert np.all(np.diff(widths) > 0)
        assert np.all(widths[17:] / widths[16:-1] < 1.25)

    def test_boxcar_peak(self) -> None:
        series = np.zeros(50)
        series[20:24] = 1.0
        widths = np.arange(1, 9)
        peak = sens.boxcar_peak(series, np.sqrt(widths.astype(float)), widths)
        assert (peak.t, peak.width) == (20, 4)
        assert peak.snr == pytest.approx(2.0)

    def test_gaussian_tail_and_effective_tests(self) -> None:
        assert sens.gaussian_isf(0.5) == pytest.approx(0.0, abs=1e-9)
        p3 = 0.5 * math.erfc(3 / math.sqrt(2))
        assert sens.gaussian_isf(p3) == pytest.approx(3)
        rng = np.random.default_rng(0)
        maxima = rng.standard_normal((2000, 300)).max(axis=1)
        assert sens.effective_tests(maxima) == pytest.approx(300, rel=0.15)


class TestTailThreshold:
    @staticmethod
    def _box_noise(n: int, length: int, seed: int) -> np.ndarray:
        x = np.random.default_rng(seed).standard_normal(n + length - 1)
        c = np.concatenate(([0.0], np.cumsum(x)))
        return (c[length:] - c[:-length]) / math.sqrt(length)

    def test_upcrossings(self) -> None:
        x = np.array([0.0, 3.0, 1.0, 5.0, 5.0, 0.0, np.inf])
        counts, n = sens.upcrossings([x], [2.0, 4.0])
        assert n == 6
        np.testing.assert_array_equal(counts, [3, 2])

    def test_independent_noise_gives_the_gaussian_tail(self) -> None:
        x = self._box_noise(1 << 21, 1, 1)
        counts, n = sens.upcrossings([x], sens.TAIL_LEVELS)
        c, beta = sens.tail_fit(sens.TAIL_LEVELS, counts, n)
        assert -1.3 < beta < -0.5
        for rate in (1e-6, 1e-9, 1e-11):
            # Up-crossings of independent samples: Q(u) (1 - Q(u)) ~ Q(u).
            want = sens.gaussian_isf(rate)
            assert sens.tail_threshold(c, beta, rate) == pytest.approx(want, abs=0.1)

    def test_correlated_noise_lowers_the_threshold(self) -> None:
        x = self._box_noise(1 << 21, 60, 2)
        counts, n = sens.upcrossings([x], sens.TAIL_LEVELS)
        c, beta = sens.tail_fit(sens.TAIL_LEVELS, counts, n)
        assert -0.4 < beta < 0.4  # Rice: a smooth process has beta = 0
        rate = 1e-9
        # Clustered crossings lower the threshold, but only a little at high
        # levels (excursions shorten: ~0.13 sigma at L = 60, u ~ 6); the
        # median-fit n_eff extrapolation overstates it (~0.4 sigma here).
        got = sens.tail_threshold(c, beta, rate)
        independent = sens.gaussian_isf(rate)
        assert independent - 0.25 < got < independent - 0.05
        maxima = x[: (x.size // 1500) * 1500].reshape(-1, 1500).max(axis=1)
        n_eff = sens.effective_tests(maxima)
        median_fit = sens.gaussian_isf(rate * 1500 / n_eff)
        assert median_fit < got - 0.15


def fdmt_plan(dms: list[float], nsamps: int) -> dmtlib.FDMTPlan:
    grid = np.asarray(dms, dtype=np.float32)
    return dmtlib.FDMTPlan(F_MIN, F_MAX, NCHANS, nsamps, TSAMP, dm_grid=grid)


class TestFDMTEmulation:
    @pytest.mark.parametrize("integer", [False, True])
    def test_plan_weights_reproduce_fdmt(self, integer: bool) -> None:
        nsamps = 2048
        dms = np.asarray(CFG.dm_trials[[3, 40, -1]], np.float32)
        fdmt = dmtlib.FDMT(F_MIN, F_MAX, NCHANS, nsamps, TSAMP, dm_grid=dms)
        rng = np.random.default_rng(3)
        if integer:  # integer-valued input: every sum is exact
            x = rng.integers(0, 16, (NCHANS, nsamps)).astype(np.float32)
        else:
            x = rng.standard_normal((NCHANS, nsamps)).astype(np.float32)
        out = np.asarray(fdmt.execute(x)).copy()
        for row in range(3):
            series, wts = sens.fdmt_rows(fdmt.plan, x, row=row)
            shifts = np.asarray(fdmt.plan.trace_dm(row))
            t0 = max(0, -int(shifts.min()))
            got = out[row, t0 : t0 + series.size]
            if integer:
                np.testing.assert_array_equal(got, series)
            else:
                np.testing.assert_allclose(got, series, rtol=1e-5, atol=1e-4)
            smearing = np.asarray(fdmt.plan.smearing_grid_final).reshape(3, -1)
            length = np.abs(smearing[row]) + 1
            np.testing.assert_allclose(wts * length, 1.0, atol=1 / 8)

    def test_weights(self) -> None:
        length = np.array([1.0, 2.0, 3.0, 17.0, 300.0])
        dyadic = sens.box_weights(length)
        b = np.ceil(np.log2(length)) + 2
        np.testing.assert_array_equal(dyadic * 2**b, np.round(2**b / length))
        assert np.max(np.abs(dyadic * length - 1)) <= 1 / 8

    def test_weights_match_library(self) -> None:
        plan = fdmt_plan(list(CFG.dm_trials), 1024)
        b = int(plan.box_fraction_bits)
        table = np.asarray(plan.box_weights)
        length = np.arange(1, table.size + 1)
        assert b == math.ceil(math.log2(table.size)) + 2
        np.testing.assert_array_equal(sens.box_weights(length), table * 2.0**-b)
        np.testing.assert_array_equal(
            sens.plan_box_weights(plan, length), table * 2.0**-b
        )

    def test_variance_model_matches_library(self) -> None:
        plan = fdmt_plan(list(CFG.dm_trials), 1024)
        smearing = np.asarray(plan.smearing_grid_final).reshape(-1, NCHANS)
        wts = sens.plan_box_weights(plan, np.abs(smearing) + 1)
        for w in (1, 2, 3, 7):
            ours = sens.box_smearing_variance(smearing, [w], wts)[:, 0]
            theirs = np.asarray(plan.get_effective_variance_grid(w))
            np.testing.assert_allclose(ours, theirs, rtol=1e-5)

    def test_weighted_variance_matches_noise(self) -> None:
        plan = fdmt_plan([float(CFG.dm_trials[-1]), float(CFG.dm_trials[-2])], 8)
        rng = np.random.default_rng(5)
        x = rng.standard_normal((NCHANS, 1 << 16))
        series, wts = sens.fdmt_rows(plan, x)
        smearing = np.asarray(plan.smearing_grid_final).reshape(-1, NCHANS)[:1]
        for w in (1, 8):
            box = np.convolve(series, np.ones(w), mode="valid")[:: 4 * w]
            var = sens.box_smearing_variance(smearing, [w], wts[None])
            assert box.std() / math.sqrt(var[0, 0]) == pytest.approx(1.0, abs=0.04)

    def test_alignment_error(self) -> None:
        plan = fdmt_plan(list(CFG.dm_trials), 1024)
        assert np.all(sens.fdmt_alignment_error(plan, 0) == 0)
        for row in (10, 40, len(CFG.dm_trials) - 1):
            err = sens.fdmt_alignment_error(plan, row)
            assert err.shape == (NCHANS,)
            assert np.abs(err).max() < 2.0
            assert err.std() < 0.7


class TestTrialRunner:
    def test_single_trial_equals_full_grid(self) -> None:
        nsamps = 4096
        dms = CFG.dm_trials
        rng = np.random.default_rng(7)
        x = rng.standard_normal((NCHANS, nsamps)).astype(np.float32)
        band = (F_MIN, F_MAX, NCHANS)
        full_fdmt = dmtlib.FDMT(*band, nsamps, TSAMP, dm_grid=dms)
        full_ddmt = dmtlib.DDMT(*band, TSAMP, dm_arr=dms)
        f_out = np.asarray(full_fdmt.execute(x)).copy()
        d_out = np.asarray(full_ddmt.execute(x)).copy()
        f_dms = np.asarray(full_fdmt.dm_grid_final)
        for dm in sens.select_trials(CFG, 5, seed=2)[1:]:
            fdmt = sens.TrialRunner("FDMT", CFG, float(dm), nsamps)
            row = int(np.abs(f_dms - dm).argmin())
            np.testing.assert_array_equal(fdmt.run(x), f_out[row])
        dm = float(dms[25])
        ddmt = sens.TrialRunner("DDMT", CFG, dm, nsamps)
        one = ddmt.run(x)
        n = min(one.size, d_out.shape[1])  # output length follows the max delay
        np.testing.assert_allclose(one[:n], d_out[25, :n], rtol=1e-6, atol=1e-5)

    def test_every_engine_integrates_the_same_dm(self) -> None:
        for dm in sens.select_trials(CFG, 4, seed=1):
            got = [sens.TrialRunner(n, CFG, float(dm), 2048).dm for n in ALL]
            np.testing.assert_allclose(got, dm, rtol=1e-5, atol=1e-6)
            assert dm / CFG.dm_per_dt == pytest.approx(round(dm / CFG.dm_per_dt))

    def test_aligned_pulse_is_fully_recovered(self) -> None:
        x = np.zeros((NCHANS, 1024), np.float32)
        sens.inject_pulse(x, F_MIN, F_MAX, TSAMP, 0.0, 300.0, 1.0)
        widths = sens.boxcar_widths(8)
        for name in ALL:
            runner = sens.TrialRunner(name, CFG, 0.0, 1024)
            peak = sens.boxcar_peak(runner.run(x), runner.sigma(widths), widths)
            assert peak.snr == pytest.approx(math.sqrt(NCHANS), rel=2e-3), name

    @pytest.mark.parametrize("name", ALL)
    def test_analytic_sigma_matches_noise(self, name: str) -> None:
        dm = float(CFG.dm_trials[-3])
        nsamps = 1 << 16
        runner = sens.TrialRunner(name, CFG, dm, nsamps)
        rng = np.random.default_rng(11)
        noise = rng.standard_normal((NCHANS, nsamps)).astype(np.float32)
        out = runner.run(noise)
        out = out[out.size // 4 :]  # past any cold start
        widths = np.array([1, 4])
        sigma = runner.sigma(widths)
        for j, w in enumerate(widths):
            box = np.convolve(out, np.ones(w), mode="valid")[:: 2 * w]
            assert box.std() / sigma[j] == pytest.approx(1.0, abs=0.04), name


@pytest.fixture(scope="module")
def recovery() -> sens.RecoveryResult:
    dms = sens.select_trials(CFG, 4, seed=4)
    return sens.measure_recovery(CFG, dms, (0, 0.5, 1, 4, 16), n_phase=4, seed=9)


class TestRecovery:
    def test_bounds(self, recovery: sens.RecoveryResult) -> None:
        assert np.all(recovery.snr_matched_filter <= recovery.snr_injected * 1.000001)
        for name in ALL:
            assert np.all(recovery.eta_mf(name) <= 1 + 1e-3), name
            assert np.all(recovery.eta_mf(name) > 0.6), name

    def test_equalities(self, recovery: sens.RecoveryResult) -> None:
        def snr(name: str) -> np.ndarray:
            return recovery.engines[name]["snr"]

        np.testing.assert_allclose(snr("SDMT"), snr("DDMT"), rtol=1e-5)

    def test_box_smearing_helps(self, recovery: sens.RecoveryResult) -> None:
        smeared = recovery.dm > 0.5 * DM_MAX
        mean = recovery.eta_mf("FDMT")[smeared]
        nobox = recovery.eta_mf("FDMT-nobox")[smeared]
        assert np.median(mean - nobox) > 0.02
        # Narrow pulses here are limited by the merge rounding, not the weights.
        assert np.median(mean) > 0.9

    def test_detection(self, recovery: sens.RecoveryResult) -> None:
        for name in ALL:
            cols = recovery.engines[name]
            assert np.all((cols["threshold"] > 4.5) & (cols["threshold"] < 8.5))
            # The median n_eff fit assumes independent tests: never higher.
            assert np.all(cols["threshold_median"] <= cols["threshold"] + 0.1), name
            assert np.all(np.abs(cols["sigma_ratio"] - 1) < 0.12), name
            curve = recovery.completeness(name, np.ones(recovery.dm.size, bool))
            assert np.all(np.diff(curve) >= -0.05), name
            assert curve[0] < 0.5 < curve[-1], name

    def test_save_load_roundtrip(
        self, recovery: sens.RecoveryResult, tmp_path: object
    ) -> None:
        path = recovery.save(tmp_path / "res")  # type: ignore[operator]
        back = sens.RecoveryResult.load(path)
        assert back.config == recovery.config
        np.testing.assert_array_equal(
            back.snr_matched_filter, recovery.snr_matched_filter
        )
        np.testing.assert_array_equal(
            back.engines["FDMT"]["snr_detect"], recovery.engines["FDMT"]["snr_detect"]
        )
        assert back.meta["seed"] == 9

    def test_detection_fluence(self) -> None:
        signal = np.array([0.0, 1.0, 2.0, 0.5])
        noise = np.array([1.0, 0.5, -1.0, 0.0])
        f = sens.detection_fluence(signal, noise, 5.0)
        assert np.max(f * signal + noise) == pytest.approx(5.0, abs=1e-9)
        assert sens.detection_fluence(signal, noise, 0.5) == 0.0

    def test_select_trials(self) -> None:
        dms = sens.select_trials(CFG, 8, seed=3)
        assert dms[0] == 0.0
        assert np.all(np.diff(dms) > 0)
        steps = dms / CFG.dm_per_dt
        np.testing.assert_allclose(steps, np.round(steps), atol=1e-6)
        assert dms[-1] <= DM_MAX * 1.01
