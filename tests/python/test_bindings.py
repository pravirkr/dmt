import math

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
    from dmtlib import simulate  # noqa: PLC0415 - numba is optional

    delays = simulate.get_dmdelays(10.0, 1000.0, 1500.0, 0.001, 8)
    assert delays.shape == (8,)
    assert delays[-1] >= delays[0]
    # Channel centres (the library's convention), referenced to f_min.
    centres = 1000.0 + (np.arange(8) + 0.5) * 62.5
    expected = simulate.DM_CONSTANT * 10.0 * (1000.0**-2 - centres**-2) / 0.001
    np.testing.assert_array_equal(delays, np.round(expected).astype(np.int32))
    assert delays[0] > 0  # the half-channel delay of channel 0

    cases = [
        (8, 32, 0.0, 20.0, 1.0),
        (8, 32, 12.375, 20.0, 1.0),
        (64, 128, 40.5, 90.25, 2.0),
        (16, 64, 30.0, 10.5, 1.0),  # the top channels start before sample 0
        (16, 64, 50.3, 70.1, 1.0),  # the bottom channels end after the block
        (16, 64, 30.0, 0.5, 1.0),  # most channels sweep wholly before sample 0
        (16, 64, 30.0, -0.25, 1.0),  # every sweep ends in (-1, 0) or earlier
        (4, 16, 400.0, 40.0, 1.0),  # the bottom sweep covers the whole block
    ]
    for nchans, nsamps, dt, toa, amp in cases:
        arr, n_disp = simulate.generate_pure_frb(
            nchans, nsamps, 1000.0, 1500.0, dt, toa, amp
        )
        assert arr.shape == (nchans, nsamps)
        assert arr.dtype == np.float32
        expected, expected_disp = _pure_frb_reference(
            nchans, nsamps, 1000.0, 1500.0, dt, toa, amp
        )
        assert isinstance(n_disp, int)
        assert n_disp == expected_disp
        assert np.all(arr >= 0.0)
        np.testing.assert_allclose(arr, expected, rtol=1e-6, atol=1e-7 * amp)


def _pure_frb_reference(
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    dt: float,
    toa: float,
    amp: float,
) -> tuple[np.ndarray, int]:
    """Channel-edge integration of a dispersed pulse in float64.

    Channel c spans [f_lo, f_hi]; its sweep starts at toa - dt * k(f_min, f_lo)
    and lasts dt * k(f_lo, f_hi) samples (k: the ratio of dispersion delays to
    the full band's), spread evenly over the samples it covers, with the first
    and last sample weighted by the fraction covered. Sample indices floor the
    sweep times, and samples outside the block are dropped.
    """
    # The C++ simulator takes float32 arguments and computes in double.
    f_min, f_max, dt, toa, amp = (
        float(np.float32(v)) for v in (f_min, f_max, dt, toa, amp)
    )

    def k(a: float, b: float) -> float:
        return (a**-2 - b**-2) / (f_min**-2 - f_max**-2)

    arr = np.zeros((nchans, nsamps))
    foff = (f_max - f_min) / nchans
    ndisp = 0
    for c in range(nchans):
        lo, hi = f_min + c * foff, f_min + (c + 1) * foff
        tstart = toa - dt * k(f_min, lo)
        sweep = dt * k(lo, hi)
        tend = tstart - sweep
        s, e = math.floor(tstart), math.floor(tend)
        if s < 0 or e >= nsamps:
            continue
        if e == s:
            arr[c, s] = amp
            ndisp += 1
            continue
        row = np.zeros(s - e + 1)
        row[:] = amp / sweep
        row[0] *= 1.0 - (tend - e)
        row[-1] *= tstart - s
        first, last = max(e, 0), min(s, nsamps - 1)
        arr[c, first : last + 1] = row[first - e : last - e + 1]
        ndisp += last - first + 1
    return arr, ndisp


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
        engine = cls(1000.0, 1500.0, 16, 64, 0.001, 8, nthreads=1)
        assert engine.backend == "cpu"
        assert engine.nthreads == 1
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


class TestStrictWaterfallInput:
    """Every waterfall binding requires exact dtype and C-contiguous layout."""

    def test_fdmt_and_fdmt_fft_reject_wrong_dtype_and_non_contiguous(self) -> None:
        nchans, nsamps = 16, 64
        wf_f64 = np.zeros((nchans, nsamps), dtype=np.float64)
        wf_noncontig = np.zeros((nchans, nsamps * 2), dtype=np.float32)[:, :nsamps]

        for cls in (dmtlib.FDMT, dmtlib.FDMTFFT):
            engine = cls(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
            with pytest.raises(ValueError, match="dtype float32"):
                engine.execute(wf_f64)
            with pytest.raises(ValueError, match="C-contiguous"):
                engine.execute(wf_noncontig)

            out = np.zeros(engine.plan.buffer_size, dtype=np.float32)
            with pytest.raises(ValueError, match="dtype float32"):
                engine.reset(wf_f64, dmt=out)
            with pytest.raises(ValueError, match="C-contiguous"):
                engine.reset(wf_noncontig, dmt=out)

        fdmt_packed = dmtlib.FDMT(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
        packed_u16 = np.zeros((nchans, nsamps), dtype=np.uint16)
        packed_noncontig = np.zeros((nchans, nsamps * 2), dtype=np.uint8)[:, :nsamps]
        with pytest.raises(ValueError, match="dtype uint8"):
            fdmt_packed.execute(packed_u16, nbits=8)
        with pytest.raises(ValueError, match="C-contiguous"):
            fdmt_packed.execute(packed_noncontig, nbits=8)
        with pytest.raises(ValueError, match="dtype uint8"):
            fdmt_packed.reset(packed_u16, nbits=8)
        with pytest.raises(ValueError, match="C-contiguous"):
            fdmt_packed.reset(packed_noncontig, nbits=8)

        fft_eng = dmtlib.FDMTFFT(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
        tm_u16 = np.zeros((nsamps, nchans), dtype=np.uint16)
        tm_noncontig = np.zeros((nsamps, nchans * 2), dtype=np.uint8)[:, :nchans]
        with pytest.raises(ValueError, match="dtype uint8"):
            fft_eng.execute(packed_u16, nbits=8)
        with pytest.raises(ValueError, match="C-contiguous"):
            fft_eng.execute(packed_noncontig, nbits=8)
        with pytest.raises(ValueError, match="dtype uint8"):
            fft_eng.execute_time_major(tm_u16, nbits=8)
        with pytest.raises(ValueError, match="C-contiguous"):
            fft_eng.execute_time_major(tm_noncontig, nbits=8)
        with pytest.raises(ValueError, match="dtype uint8"):
            fft_eng.reset(packed_u16, nbits=8)
        with pytest.raises(ValueError, match="C-contiguous"):
            fft_eng.reset(packed_noncontig, nbits=8)

        hist_f64 = np.zeros(fft_eng.history_state_size(), dtype=np.float64)
        hist_noncontig = np.zeros(fft_eng.history_state_size() * 2, dtype=np.float32)[
            ::2
        ]
        with pytest.raises(ValueError, match="dtype float32"):
            fft_eng.load_history(hist_f64)
        with pytest.raises(ValueError, match="C-contiguous"):
            fft_eng.load_history(hist_noncontig)

    def test_host_load_history_rejects_wrong_dtype_and_non_contiguous(self) -> None:
        """Host load_history takes exact dtype, C order (FDMT, DDMT, DDMTFFT)."""
        nchans, nsamps = 16, 64
        dms = np.linspace(0.0, 50.0, 8, dtype=np.float32)
        engines = [
            (dmtlib.FDMT(1000.0, 1500.0, nchans, nsamps, 0.001, 8), np.float32),
            (dmtlib.DDMT(1000.0, 1500.0, nchans, 0.001, dms), np.float32),
            (dmtlib.DDMT(1000.0, 1500.0, nchans, 0.001, dms, nbits=8), np.uint8),
            (dmtlib.DDMTFFT(1000.0, 1500.0, nchans, 0.001, dms), np.float32),
        ]
        for engine, dtype in engines:
            size = engine.history_state_size()
            assert size > 0
            wrong = np.float64 if dtype == np.float32 else np.float32
            name = "float32" if dtype == np.float32 else "uint8"
            noncontig = np.zeros(2 * size, dtype=dtype)[::2]
            with pytest.raises(ValueError, match=f"dtype {name}"):
                engine.load_history(np.zeros(size, dtype=wrong))
            with pytest.raises(ValueError, match="C-contiguous"):
                engine.load_history(noncontig)
            # save_history needs a warmed-up stream (DDMT, DDMTFFT).
            if isinstance(engine, dmtlib.FDMT):
                engine.execute(np.zeros((nchans, nsamps), dtype=dtype))
            elif dtype == np.uint8:
                engine.execute(np.zeros((nchans, 256), dtype=dtype), nsamps=256)
            else:
                engine.execute(np.zeros((nchans, 256), dtype=dtype))
            saved = engine.save_history()
            assert saved.dtype == dtype
            assert saved.size == size
            engine.load_history(saved)

    def test_stepper_dmt_buffer_rejects_bad_dtype_layout_and_readonly(self) -> None:
        nchans, nsamps = 16, 64
        wf = np.zeros((nchans, nsamps), dtype=np.float32)
        packed = np.zeros((nchans, nsamps), dtype=np.uint8)

        for cls in (dmtlib.FDMT, dmtlib.FDMTFFT):
            engine = cls(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
            needed = engine.plan.buffer_size
            bad_dtype = np.zeros(needed, dtype=np.float64)
            bad_stride = np.zeros(needed * 2, dtype=np.float32)[::2]
            readonly = np.zeros(needed, dtype=np.float32)
            readonly.setflags(write=False)

            for call in (
                lambda buf, e=engine: e.reset(wf, dmt=buf),
                lambda buf, e=engine: e.reset(packed, nbits=8, dmt=buf),
            ):
                with pytest.raises(ValueError, match="dtype float32"):
                    call(bad_dtype)
                with pytest.raises(ValueError, match="C-contiguous"):
                    call(bad_stride)
                with pytest.raises(ValueError, match="writeable"):
                    call(readonly)

    def test_compute_fdmt_one_shots_reject_wrong_dtype_and_non_contiguous(self) -> None:
        nchans, nsamps = 16, 64
        wf_f64 = np.zeros((nchans, nsamps), dtype=np.float64)
        wf_noncontig = np.zeros((nchans, nsamps * 2), dtype=np.float32)[:, :nsamps]

        for fn in (libdmt.compute_fdmt, libdmt.compute_fdmt_fft):
            with pytest.raises(ValueError, match="dtype float32"):
                fn(wf_f64, 1000.0, 1500.0, nchans, nsamps, 0.001, 8)
            with pytest.raises(ValueError, match="C-contiguous"):
                fn(wf_noncontig, 1000.0, 1500.0, nchans, nsamps, 0.001, 8)
            with pytest.raises(ValueError, match="dtype float32"):
                fn(wf_f64, 1000.0, 1500.0, nchans, nsamps, 0.001, dt_grid=[0, 4, 8])
            with pytest.raises(ValueError, match="C-contiguous"):
                fn(
                    wf_noncontig,
                    1000.0,
                    1500.0,
                    nchans,
                    nsamps,
                    0.001,
                    dt_grid=[0, 4, 8],
                )

    def test_ddmt_and_ddmt_fft_reject_wrong_dtype_and_non_contiguous(self) -> None:
        nchans, nsamps = 16, 64
        dms = np.array([0.0, 5.0], dtype=np.float32)
        wf_f64 = np.zeros((nchans, nsamps), dtype=np.float64)
        wf_noncontig = np.zeros((nchans, nsamps * 2), dtype=np.float32)[:, :nsamps]
        packed_u16 = np.zeros((nchans, nsamps), dtype=np.uint16)
        packed_noncontig = np.zeros((nchans, nsamps * 2), dtype=np.uint8)[:, :nsamps]
        tm_u16 = np.zeros((nsamps, nchans), dtype=np.uint16)
        tm_noncontig = np.zeros((nsamps, nchans * 2), dtype=np.uint8)[:, :nchans]

        for cls in (dmtlib.DDMT, dmtlib.DDMTFFT):
            eng_f32 = cls(1000.0, 1500.0, nchans, 0.001, dms, nthreads=1)
            with pytest.raises(ValueError, match="dtype float32"):
                eng_f32.execute(wf_f64)
            with pytest.raises(ValueError, match="C-contiguous"):
                eng_f32.execute(wf_noncontig)

            eng_u8 = cls(1000.0, 1500.0, nchans, 0.001, dms, nbits=8, nthreads=1)
            with pytest.raises(ValueError, match="dtype uint8"):
                eng_u8.execute(packed_u16, nsamps=nsamps)
            with pytest.raises(ValueError, match="C-contiguous"):
                eng_u8.execute(packed_noncontig, nsamps=nsamps)
            with pytest.raises(ValueError, match="dtype uint8"):
                eng_u8.execute_time_major(tm_u16, nsamps=nsamps)
            with pytest.raises(ValueError, match="C-contiguous"):
                eng_u8.execute_time_major(tm_noncontig, nsamps=nsamps)

    def test_packed_second_argument_is_keyword_only(self) -> None:
        nchans, nsamps = 16, 64
        packed = np.zeros((nchans, nsamps), dtype=np.uint8)
        tm = np.zeros((nsamps, nchans), dtype=np.uint8)
        dms = np.array([0.0, 5.0], dtype=np.float32)

        fdmt = dmtlib.FDMT(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
        with pytest.raises(TypeError):
            fdmt.execute(packed, 8)
        with pytest.raises(TypeError):
            fdmt.reset(packed, 8)

        fdmt_fft = dmtlib.FDMTFFT(1000.0, 1500.0, nchans, nsamps, 0.001, 8, nthreads=1)
        with pytest.raises(TypeError):
            fdmt_fft.execute(packed, 8)
        with pytest.raises(TypeError):
            fdmt_fft.execute_time_major(tm, 8)
        with pytest.raises(TypeError):
            fdmt_fft.reset(packed, 8)

        # dmt= is keyword-only on the float overloads too.
        wf = np.zeros((nchans, nsamps), dtype=np.float32)
        for eng in (fdmt, fdmt_fft):
            with pytest.raises(TypeError):
                eng.reset(wf, np.zeros((nchans, nsamps), dtype=np.float32))

        for cls in (dmtlib.DDMT, dmtlib.SDMT, dmtlib.DDMTFFT):
            eng = cls(1000.0, 1500.0, nchans, 0.001, dms, nbits=8, nthreads=1)
            with pytest.raises(TypeError):
                eng.execute(packed, nsamps)
            with pytest.raises(TypeError):
                eng.execute_time_major(tm, nsamps)

    def test_pack_baseband_rejects_wrong_dtype_and_non_contiguous(self) -> None:
        v_c128 = np.zeros((2, 4, 16), dtype=np.complex128)
        v_noncontig = np.zeros((2, 4, 32), dtype=np.complex64)[:, :, :16]
        with pytest.raises(ValueError, match="dtype complex64"):
            libdmt.pack_baseband(v_c128)
        with pytest.raises(ValueError, match="C-contiguous"):
            libdmt.pack_baseband(v_noncontig)
