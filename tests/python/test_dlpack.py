"""Device-memory (DLPack) inputs and outputs of the engines.

Each test runs one engine on device arrays and compares with the same engine
type fed NumPy arrays on the same backend. CuPy provides the device arrays
(CUDA only); the PyTorch test runs when torch with CUDA is installed.
"""

import dmtlib
import numpy as np
import pytest

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, SDMT, CohFDMT

cp = pytest.importorskip("cupy")


@pytest.fixture
def cuda(gpu_backend: str) -> str:
    if gpu_backend != "cuda":
        pytest.skip("the device arrays here come from CuPy (CUDA)")
    return gpu_backend


def ptr(a: object) -> int:
    return a.data.ptr


FDMT_ARGS = (1000.0, 1500.0, 64, 256, 0.001, 40)


def pack2(values: np.ndarray) -> np.ndarray:
    """2-bit LSB-first packing (4 samples per byte) along the last axis."""
    out = np.zeros((*values.shape[:-1], values.shape[-1] // 4), dtype=np.uint8)
    for k in range(4):
        out |= values[..., k::4].astype(np.uint8) << (2 * k)
    return out


class TestFDMT:
    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_execute_matches_host(self, cls: type, cuda: str) -> None:
        wf = np.random.default_rng(0).standard_normal((64, 256), dtype=np.float32)
        want = cls(*FDMT_ARGS, backend=cuda).execute(wf)
        got = cls(*FDMT_ARGS, backend=cuda).execute(cp.asarray(wf))
        assert isinstance(got, cp.ndarray)
        assert got.shape == want.shape
        np.testing.assert_array_equal(got.get(), want)

    def test_out_is_written_in_place(self, cuda: str) -> None:
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        wf = cp.asarray(np.random.default_rng(1).standard_normal((64, 256), np.float32))
        out = cp.full(fdmt.plan.dmt_size + 7, -1.0, dtype=cp.float32)
        pool = cp.get_default_memory_pool()
        used = pool.used_bytes()
        got = fdmt.execute(wf, out=out)
        assert pool.used_bytes() == used  # no copy of the input or output
        assert ptr(got) == ptr(out)
        assert got.shape == (fdmt.plan.dmt_ndms, fdmt.plan.dmt_nsamps)
        assert float(out[-1]) == -1.0
        np.testing.assert_array_equal(got.get(), FDMT(*FDMT_ARGS).execute(wf.get()))

    def test_input_is_not_copied(self, cuda: str) -> None:
        # A view at an offset into a larger allocation is read in place.
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        big = cp.asarray(
            np.random.default_rng(2).standard_normal((65, 256), dtype=np.float32)
        )
        view = big[1:]
        np.testing.assert_array_equal(
            fdmt.execute(view).get(),
            FDMT(*FDMT_ARGS, backend=cuda).execute(big[1:].get()),
        )

    def test_beams_and_packed(self, cuda: str) -> None:
        values = np.random.default_rng(3).integers(0, 4, (2, 64, 256))
        packed = pack2(values)
        host = FDMT(*FDMT_ARGS, nbeams=2, backend=cuda)
        dev = FDMT(*FDMT_ARGS, nbeams=2, backend=cuda)
        got = dev.execute(cp.asarray(packed), 2)
        assert got.shape == (2, dev.plan.dmt_ndms, dev.plan.dmt_nsamps)
        np.testing.assert_array_equal(got.get(), host.execute(packed, 2))

    def test_streams(self, cuda: str) -> None:
        # "roll" mode keeps no history, so every call gives the same result.
        fdmt = FDMT(*FDMT_ARGS, mode="roll", backend=cuda)
        wf = np.random.default_rng(4).standard_normal((64, 256), dtype=np.float32)
        want = FDMT(*FDMT_ARGS, mode="roll", backend=cuda).execute(wf)
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:  # default: the current CuPy stream
            d_wf = cp.asarray(wf)
            got = fdmt.execute(d_wf)
        stream.synchronize()
        np.testing.assert_array_equal(got.get(), want)
        for s in (stream, stream.ptr, cp.cuda.Stream.null, 0):
            got = fdmt.execute(d_wf, stream=s)
            cp.cuda.Device().synchronize()
            np.testing.assert_array_equal(got.get(), want)

    def test_errors(self, cuda: str) -> None:
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        wf = cp.zeros((64, 256), dtype=cp.float32)
        with pytest.raises(TypeError, match="float32"):
            fdmt.execute(wf.astype(cp.float64))
        with pytest.raises(TypeError, match="C-contiguous"):
            fdmt.execute(cp.zeros((256, 64), dtype=cp.float32).T)
        with pytest.raises(TypeError, match="nbits"):
            fdmt.execute(wf.astype(cp.uint8))
        with pytest.raises(ValueError, match="cpu backend"):
            FDMT(*FDMT_ARGS).execute(wf)
        with pytest.raises(ValueError, match="needs at least"):
            fdmt.execute(wf, out=cp.empty(10, dtype=cp.float32))
        with pytest.raises(TypeError, match="out must be a float32"):
            fdmt.execute(wf, out=cp.empty(fdmt.plan.dmt_size, dtype=cp.int32))
        with pytest.raises(TypeError, match="out must be a device array"):
            fdmt.execute(wf, out=np.empty(fdmt.plan.dmt_size, dtype=np.float32))
        with pytest.raises(TypeError):
            fdmt.execute(wf, stream="not a stream")

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_stepper(self, cls: type, cuda: str) -> None:
        wf = np.random.default_rng(5).integers(0, 4, (64, 256)).astype(np.float32)
        want = cls(*FDMT_ARGS, backend=cuda).execute(wf)
        eng = cls(*FDMT_ARGS, backend=cuda)
        eng.reset(cp.asarray(wf))
        eng.advance(stream=cp.cuda.get_current_stream())
        eng.advance_until_remaining(0)
        got = eng.finalize()
        assert isinstance(got, cp.ndarray)
        np.testing.assert_allclose(got.get(), want, rtol=1e-5, atol=1e-4)
        # A host reset() afterwards goes back to NumPy results.
        eng.reset(wf)
        eng.advance_until_remaining(0)
        assert isinstance(eng.finalize(), np.ndarray)

    def test_stepper_dmt_buffer(self, cuda: str) -> None:
        fdmt = FDMT(*FDMT_ARGS, int_tree=False, backend=cuda)
        dmt = cp.empty(fdmt.plan.dmt_size, dtype=cp.float32)
        fdmt.reset(cp.zeros((64, 256), dtype=cp.float32), dmt)
        fdmt.advance_until_remaining(0)
        assert ptr(fdmt.finalize()) == ptr(dmt)

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_history_round_trip(self, cls: type, cuda: str) -> None:
        rng = np.random.default_rng(6)
        blocks = rng.standard_normal((2, 64, 256), dtype=np.float32)
        host = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        host.execute(blocks[0])
        want = host.execute(blocks[1])

        first = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        first.execute(cp.asarray(blocks[0]))
        hist = cp.empty(first.history_state_size(), dtype=cp.float32)
        assert first.save_history(hist) is hist
        np.testing.assert_array_equal(hist.get(), host_history(cls, blocks[0], cuda))

        second = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        second.load_history(hist)
        np.testing.assert_array_equal(second.execute(cp.asarray(blocks[1])).get(), want)


def host_history(cls: type, block: np.ndarray, backend: str) -> np.ndarray:
    eng = cls(*FDMT_ARGS, mode="valid", backend=backend)
    eng.execute(block)
    return eng.save_history()


DDMT_ARGS = (1000.0, 1500.0, 64, 0.001, 50.0, 1.0)


class TestDDMT:
    @pytest.mark.parametrize("cls", [DDMT, SDMT])
    def test_execute_matches_host(self, cls: type, cuda: str) -> None:
        rng = np.random.default_rng(7)
        host = cls(*DDMT_ARGS, backend=cuda)
        dev = cls(*DDMT_ARGS, backend=cuda)
        for _ in range(2):  # the second block uses the carried history
            wf = rng.standard_normal((64, 1024), dtype=np.float32)
            want = host.execute(wf)
            got = dev.execute(cp.asarray(wf))
            assert got.shape == want.shape
            np.testing.assert_allclose(got.get(), want, rtol=1e-5, atol=1e-4)

    def test_packed_and_time_major(self, cuda: str) -> None:
        rng = np.random.default_rng(8)
        packed = rng.integers(0, 256, (64, 1024), dtype=np.uint8)
        host = DDMT(*DDMT_ARGS, nbits=8, backend=cuda)
        dev = DDMT(*DDMT_ARGS, nbits=8, backend=cuda)
        got = dev.execute(cp.asarray(packed), 1024)
        assert got.dtype == cp.int32
        np.testing.assert_array_equal(got.get(), host.execute(packed, 1024))
        tm = np.ascontiguousarray(packed.T)
        out = cp.empty(got.shape[0] * dev.get_output_nsamps(1024), dtype=cp.int32)
        got = dev.execute_time_major(cp.asarray(tm), 1024, out=out)
        assert ptr(got) == ptr(out)
        np.testing.assert_array_equal(got.get(), host.execute_time_major(tm, 1024))

    @pytest.mark.parametrize(("nbits", "dtype"), [(32, np.float32), (8, np.uint8)])
    def test_history_round_trip(self, nbits: int, dtype: type, cuda: str) -> None:
        rng = np.random.default_rng(9)
        if nbits == 32:
            blocks = [rng.standard_normal((64, 512), dtype=np.float32) for _ in "ab"]
            run = lambda e, b: e.execute(b)  # noqa: E731
        else:
            blocks = [rng.integers(0, 256, (64, 512), dtype=np.uint8) for _ in "ab"]
            run = lambda e, b: e.execute(b, 512)  # noqa: E731
        host = DDMT(*DDMT_ARGS, nbits=nbits, backend=cuda)
        run(host, blocks[0])
        want = run(host, blocks[1])

        first = DDMT(*DDMT_ARGS, nbits=nbits, backend=cuda)
        run(first, cp.asarray(blocks[0]))
        hist = cp.empty(first.history_state_size(), dtype=dtype)
        first.save_history(hist)
        second = DDMT(*DDMT_ARGS, nbits=nbits, backend=cuda)
        second.load_history(hist)
        np.testing.assert_allclose(
            run(second, cp.asarray(blocks[1])).get(), want, rtol=1e-5, atol=1e-4
        )
        with pytest.raises(TypeError, match=np.dtype(dtype).name):
            first.save_history(cp.empty(hist.size, dtype=cp.int16))

    def test_errors(self, cuda: str) -> None:
        dev = DDMT(*DDMT_ARGS, backend=cuda)
        with pytest.raises(ValueError, match="channels"):
            dev.execute(cp.zeros((32, 128), dtype=cp.float32))
        with pytest.raises(ValueError, match="beam"):
            dev.execute(cp.zeros((2, 64, 128), dtype=cp.float32))

    @pytest.mark.parametrize("method", ["brute", "nufft"])
    def test_ddmt_fft(self, method: str, cuda: str) -> None:
        rng = np.random.default_rng(10)
        args = (1100.0, 1500.0, 64, 0.001, 30.0, 0.25)
        host = DDMTFFT(*args, nbeams=2, method=method, backend=cuda)
        dev = DDMTFFT(*args, nbeams=2, method=method, backend=cuda)
        for _ in range(2):
            wf = rng.standard_normal((2, 64, 2000), dtype=np.float32)
            want = host.execute(wf)
            got = dev.execute(cp.asarray(wf))
            np.testing.assert_allclose(
                got.get(), want, rtol=0, atol=1e-5 * np.abs(want).max()
            )
        hist = cp.empty(dev.history_state_size(), dtype=cp.float32)
        dev.save_history(hist)
        np.testing.assert_allclose(
            hist.get(), host.save_history(), rtol=0, atol=1e-5 * np.abs(want).max()
        )

    def test_ddmt_fft_packed(self, cuda: str) -> None:
        rng = np.random.default_rng(11)
        args = (1100.0, 1500.0, 64, 0.001, 30.0, 0.25)
        packed = rng.integers(0, 256, (64, 2000), dtype=np.uint8)
        want = DDMTFFT(*args, nbits=8, backend=cuda).execute(packed, 2000)
        got = DDMTFFT(*args, nbits=8, backend=cuda).execute(cp.asarray(packed), 2000)
        assert got.dtype == cp.float32
        np.testing.assert_allclose(
            got.get(), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )


class TestCohFDMT:
    def test_execute_matches_host(self, cuda: str) -> None:
        cfg = dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0)
        coh = CohFDMT(cfg, backend=cuda)
        rng = np.random.default_rng(12)
        data = rng.normal(0.0, 8.0, size=coh.input_size()).astype(np.int8)
        want = coh.execute(data)
        got = coh.execute(cp.asarray(data))
        assert isinstance(got, cp.ndarray)
        assert got.shape == want.shape
        np.testing.assert_allclose(
            got.get(), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )
        out = cp.empty(coh.dmt_size, dtype=cp.float32)
        again = coh.execute(cp.asarray(data.view(np.uint8)), out=out, stream=0)
        assert ptr(again) == ptr(out)
        cp.cuda.Device().synchronize()
        np.testing.assert_array_equal(again.get(), got.get())

    def test_subband_groups(self, cuda: str) -> None:
        cfg = dmtlib.CohFDMTConfig(
            400.0, 1.0, 16, 4.0e-6, 10.0, 13.0, subband_groups=[6, 10]
        )
        coh = CohFDMT(cfg, backend=cuda)
        rng = np.random.default_rng(13)
        groups = [
            rng.integers(0, 256, coh.input_size(g), dtype=np.uint8) for g in (0, 1)
        ]
        want = coh.execute(groups)
        got = coh.execute([cp.asarray(g) for g in groups])
        np.testing.assert_allclose(
            got.get(), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )

    def test_errors(self, cuda: str) -> None:
        cfg = dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0)
        coh = CohFDMT(cfg, backend=cuda)
        data = np.zeros(coh.input_size(), dtype=np.uint8)
        with pytest.raises(TypeError, match="stream"):
            coh.execute(data, stream=0)
        with pytest.raises(TypeError, match="uint8 or int8"):
            coh.execute(cp.zeros(coh.input_size() // 2, dtype=cp.int16))
        with pytest.raises(TypeError, match="device array"):
            coh.execute([cp.asarray(data), data])
        with pytest.raises(ValueError, match="cpu backend"):
            CohFDMT(cfg).execute(cp.asarray(data))


def test_torch(cuda: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    wf = np.random.default_rng(14).standard_normal((64, 256), dtype=np.float32)
    want = FDMT(*FDMT_ARGS).execute(wf)
    fdmt = FDMT(*FDMT_ARGS, backend=cuda)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        t = torch.from_numpy(wf).cuda()
        got = fdmt.execute(t)
    assert isinstance(got, torch.Tensor)
    stream.synchronize()
    np.testing.assert_array_equal(got.cpu().numpy(), want)
    out = torch.empty(fdmt.plan.dmt_size, device="cuda")
    got = fdmt.execute(t, out=out, stream=stream)
    assert got.data_ptr() == out.data_ptr()
    stream.synchronize()
    np.testing.assert_array_equal(got.cpu().numpy(), want)
