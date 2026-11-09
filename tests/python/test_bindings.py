import numpy as np
import pytest

import dmtlib
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
    assert delays[-1] >= delays[0]
    # Channel centres (the library's convention), referenced to f_min.
    centres = 1000.0 + (np.arange(8) + 0.5) * 62.5
    expected = simulate.DM_CONSTANT * 10.0 * (1000.0**-2 - centres**-2) / 0.001
    np.testing.assert_array_equal(delays, np.round(expected).astype(np.int32))
    assert delays[0] > 0  # the half-channel delay of channel 0

    arr, n_disp = simulate.generate_pure_frb(8, 32, 1000.0, 1500.0, 0, 10.0, 1.0)
    assert arr.shape == (8, 32)
    assert n_disp > 0
    assert float(np.sum(arr)) > 0.0


class TestBackends:
    def test_available_backends(self) -> None:
        avail = dmtlib.available_backends()
        assert avail[0] == "cpu"
        assert set(avail) <= {"cpu", "cuda", "hip"}

    def test_build_info(self) -> None:
        info = dmtlib.build_info()
        assert set(info) >= {"version", "git_sha", "git_dirty", "compiler", "gpu"}
        assert info["git_sha"] == "unknown" or len(info["git_sha"]) == 40
        assert isinstance(info["git_dirty"], bool)
        assert info["fftw"]
        assert (info["gpu"] == "none") == (dmtlib.available_backends() == ["cpu"])

    @pytest.mark.parametrize(
        "cls", [dmtlib.FDMT, dmtlib.FDMTFFT], ids=["FDMT", "FDMTFFT"]
    )
    def test_default_backend_is_cpu(self, cls: type) -> None:
        engine = cls(1000.0, 1500.0, 16, 64, 0.001, 8, nthreads=2)
        assert engine.backend == "cpu"
        assert engine.nthreads == 2
        assert engine.device == -1

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            dmtlib.FDMT(1000.0, 1500.0, 16, 64, 0.001, 8, backend="opencl")

    def test_missing_backend_raises(self) -> None:
        missing = [b for b in ("cuda", "hip") if b not in dmtlib.available_backends()]
        for backend in missing:
            with pytest.raises(ValueError, match="not available"):
                dmtlib.FDMT(1000.0, 1500.0, 16, 64, 0.001, 8, backend=backend)
            with pytest.raises(ValueError, match="not available"):
                dmtlib.DDMT(1000.0, 1500.0, 16, 0.001, 10.0, 1.0, backend=backend)

    def test_backend_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            dmtlib.FDMT(
                1000.0,
                1500.0,
                16,
                64,
                0.001,
                8,
                0,
                1,
                True,
                "valid",
                1,
                1,
                None,
                True,
                "cpu",
            )
