import numpy as np
import pytest

from dmtlib import libdmt


def test_generate_pure_frb_binding() -> None:
    nchans = 8
    nsamps = 32
    arr, n_disp = libdmt.generate_pure_frb(nchans, nsamps, 1000.0, 1500.0, 0, 10.0, 2.0)
    assert n_disp > 0
    flat = np.asarray(arr).reshape(-1)
    assert flat.size == nchans * nsamps
    assert float(np.sum(flat)) > 0.0


def test_trace_dm_and_add_frb_track() -> None:
    plan = libdmt.FDMTPlan(1000.0, 1500.0, 16, 64, 0.001, 8)
    shifts = plan.trace_dm(0)
    assert np.asarray(shifts).shape[0] == 16
    np.testing.assert_array_equal(np.asarray(shifts), 0)

    waterfall = np.zeros((16, 64), dtype=np.float32)
    libdmt.add_frb_track(waterfall, plan, 0, 1.0, 8, 1)
    np.testing.assert_allclose(waterfall[:, 8], 1.0)


def test_simulate_python_helpers() -> None:
    pytest.importorskip("numba")
    simulate = pytest.importorskip("dmtlib.simulate")

    delays = simulate.get_dmdelays(10.0, 1000.0, 1500.0, 0.001, 8)
    assert delays.shape == (8,)
    assert delays[0] == 0
    assert delays[-1] >= delays[0]

    arr, n_disp = simulate.generate_pure_frb(8, 32, 1000.0, 1500.0, 0, 10.0, 1.0)
    assert arr.shape == (8, 32)
    assert n_disp > 0
    assert float(np.sum(arr)) > 0.0
