import numpy as np
import pytest

import dmtlib

F_MIN, F_MAX, NCHANS, TSAMP = 1200.0, 1600.0, 16, 1e-3


def _noise(shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_method_resolution_and_properties():
    lin = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 30.0, 0.5)
    assert lin.method == "nufft"
    assert lin.guard == 64
    assert lin.tolerance == pytest.approx(1e-6)
    assert lin.max_delay > lin.guard
    assert lin.get_output_nsamps(lin.max_delay + 5) == 5

    uneven = dmtlib.DDMTFFT(
        F_MIN, F_MAX, NCHANS, TSAMP, np.array([0.0, 1.0, 3.0, 7.0], np.float32)
    )
    assert uneven.method == "brute"
    with pytest.raises(ValueError, match="uniformly spaced"):
        dmtlib.DDMTFFT(
            F_MIN,
            F_MAX,
            NCHANS,
            TSAMP,
            np.array([0.0, 1.0, 3.0], np.float32),
            method="nufft",
        )
    with pytest.raises(ValueError, match="unknown method"):
        dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 10.0, 1.0, method="x")


def test_nufft_matches_brute_and_shapes():
    kw = dict(dm_min=0.0, nthreads=2, guard=16)
    brute = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 40.0, 0.5, method="brute", **kw)
    nufft = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 40.0, 0.5, method="nufft", **kw)
    wf = _noise((NCHANS, 1500))
    cold = brute.get_output_nsamps(1500)  # before the stream warms up
    a = brute.execute(wf)
    b = nufft.execute(wf)
    assert a.shape == (81, cold)
    assert brute.get_output_nsamps(1500) == 1500
    assert a.dtype == np.float32
    np.testing.assert_allclose(b, a, atol=1e-5 * np.abs(a).max())


def test_packed_history_and_beams():
    rng = np.random.default_rng(1)
    packed = rng.integers(0, 256, size=(NCHANS, 900), dtype=np.uint8)
    p8 = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 20.0, 0.5, nbits=8)
    pf = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 20.0, 0.5)
    np.testing.assert_array_equal(
        p8.execute(packed, 900), pf.execute(packed.astype(np.float32))
    )

    a = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 20.0, 0.5)
    b = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 20.0, 0.5)
    for seed in range(2):
        a.execute(_noise((NCHANS, 700), seed))
    hist = a.save_history()
    assert hist.size == a.history_state_size()
    b.load_history(hist)
    wf = _noise((NCHANS, 700), 9)
    np.testing.assert_array_equal(a.execute(wf), b.execute(wf))
    with pytest.raises(ValueError, match="float32"):
        b.load_history(hist.astype(np.float64))

    multi = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 20.0, 0.5, nbeams=2)
    out = multi.execute(_noise((2, NCHANS, 800), 4))
    assert out.shape[0] == 2


def test_fft_planner_and_wisdom(tmp_path):
    saved = dmtlib.get_fft_planner()
    dmtlib.set_fft_planner(dmtlib.FFTPlanner.MEASURE)
    assert dmtlib.get_fft_planner() == dmtlib.FFTPlanner.MEASURE
    dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 10.0, 1.0).execute(
        _noise((NCHANS, 600))
    )
    path = tmp_path / "wisdom.txt"
    exported = dmtlib.export_fft_wisdom(str(path))
    dmtlib.set_fft_planner(saved)
    if not exported:
        # Another library loaded first (e.g. NumPy linked to MKL) can export
        # FFTW-compatible symbols whose wisdom functions are no-ops.
        pytest.skip("FFTW symbols are provided by a non-FFTW library")
    dmtlib.forget_fft_wisdom()
    assert dmtlib.import_fft_wisdom(str(path))


def test_fdmt_fft_fractional_flag():
    exact = dmtlib.FDMTFFT(1100.0, 1500.0, 32, 512, TSAMP, 40)  # the default
    plain = dmtlib.FDMTFFT(1100.0, 1500.0, 32, 512, TSAMP, 40, fractional_delays=False)
    assert exact.fractional_delays and not plain.fractional_delays
    wf = _noise((32, 512), 2)
    a = exact.execute(wf)
    b = plain.execute(wf)
    assert a.shape == b.shape
    assert not np.allclose(a, b)
    dmt, _ = dmtlib.compute_fdmt_fft(
        wf, 1100.0, 1500.0, 32, 512, TSAMP, 40, fractional_delays=True
    )
    np.testing.assert_allclose(dmt.reshape(a.shape), a, rtol=1e-5, atol=1e-4)


def test_fdmt_fft_latency_and_suggested_block():
    exact = dmtlib.FDMTFFT(1100.0, 1500.0, 32, 512, TSAMP, 40)  # the default
    plain = dmtlib.FDMTFFT(1100.0, 1500.0, 32, 512, TSAMP, 40, fractional_delays=False)
    assert exact.output_latency == 64  # the interpolation look-ahead
    assert plain.output_latency == 0
    assert plain.suggested_nsamps >= 4 * 40


def test_piecewise_grid_and_method_used():
    lin = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, 30.0, 0.5)
    assert lin.method_used == "nufft"
    assert lin.suggested_nsamps >= 4 * lin.max_delay
    dms = np.concatenate(
        [
            0.5 * np.arange(40),
            [20.3, 21.7, 23.0],
            24.0 + 0.75 * np.arange(64),
        ]
    ).astype(np.float32)
    mixed = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, dms, guard=16)
    brute = dmtlib.DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, dms, guard=16, method="brute")
    assert mixed.method_used == "piecewise_nufft"
    assert brute.method_used == "brute"
    wf = _noise((NCHANS, 1500), 3)
    a = brute.execute(wf)
    b = mixed.execute(wf)
    np.testing.assert_allclose(b, a, atol=1e-5 * np.abs(a).max())

    # DDplan-style Levin grid: uniform segments, never coarser than Levin.
    args = (0.0, 2000.0, 6.4e-5, 1e-4, F_MIN, F_MAX, 1024, 1.2)
    levin = dmtlib.DDMTPlan.generate_levin_dm_grid(*args)
    pw = dmtlib.DDMTPlan.generate_levin_dm_grid_piecewise(*args)
    assert len(levin) <= len(pw) < 2 * len(levin)
    assert pw[-1] >= 2000.0
    steps = np.diff(pw)
    idx = np.clip(np.searchsorted(levin, pw[:-1], side="right") - 1, 0, len(levin) - 2)
    assert np.all(steps <= np.diff(levin)[idx] * 1.0001)
    grid = dmtlib.generate_optimal_dm_grid(
        F_MIN, F_MAX, NCHANS, TSAMP, 30.0, method="levin_piecewise", tol=1.2
    )
    assert np.all(np.diff(grid) > 0)
    cfg = dmtlib.LevinConfig(0.0, 2000.0, 1e-4, 1.2, piecewise_uniform=True)
    assert cfg.piecewise_uniform
    eng = dmtlib.DDMTFFT(F_MIN, F_MAX, 1024, 6.4e-5, cfg)
    assert eng.method_used == "piecewise_nufft"
    assert eng.plan.dm_arr.size == pw.size
