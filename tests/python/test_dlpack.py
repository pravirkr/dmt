"""Device-memory inputs and outputs of the engines.

Each test runs one engine on device arrays and compares with the same engine
type fed NumPy arrays on the same backend. The device arrays come from every
installed array library: CuPy and PyTorch (DLPack) and Numba (CUDA array
interface only); each test runs once per library through the small `Lib`
adapters below.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Any

import dmtlib
import numpy as np
import pytest

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, SDMT, CohFDMT

if TYPE_CHECKING:
    from collections.abc import Iterator


class Lib:
    """The parts of a device-array library the tests use (CUDA device 0)."""

    name = ""

    def asarray(self, a: np.ndarray) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def get(self, a: Any) -> np.ndarray:  # noqa: ANN401
        raise NotImplementedError

    def empty(self, shape: int | tuple[int, ...], dtype: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def astype(self, a: Any, dtype: Any) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def ptr(self, a: Any) -> int:  # noqa: ANN401
        raise NotImplementedError

    def is_array(self, a: Any) -> bool:  # noqa: ANN401
        raise NotImplementedError

    def stream(self) -> Any:  # noqa: ANN401
        """A new non-blocking stream."""
        raise NotImplementedError

    def use(self, stream: Any) -> contextlib.AbstractContextManager[Any]:  # noqa: ANN401
        """Make ``stream`` the library's current stream."""
        raise NotImplementedError

    def handle(self, stream: Any) -> int:  # noqa: ANN401
        raise NotImplementedError

    def default_stream(self) -> Any:  # noqa: ANN401
        raise NotImplementedError

    def synchronize(self) -> None:
        raise NotImplementedError

    def used_bytes(self) -> int:
        raise NotImplementedError

    def full(self, shape: int | tuple[int, ...], value: float, dtype: Any) -> Any:  # noqa: ANN401
        return self.asarray(np.full(shape, value, dtype=dtype))


class CuPy(Lib):
    name = "cupy"

    def __init__(self) -> None:
        import cupy  # noqa: PLC0415

        self.cp = cupy

    def asarray(self, a: np.ndarray) -> Any:  # noqa: ANN401
        return self.cp.asarray(a)

    def get(self, a: Any) -> np.ndarray:  # noqa: ANN401
        return a.get()

    def empty(self, shape: int | tuple[int, ...], dtype: Any) -> Any:  # noqa: ANN401
        return self.cp.empty(shape, dtype=dtype)

    def astype(self, a: Any, dtype: Any) -> Any:  # noqa: ANN401
        return a.astype(dtype)

    def ptr(self, a: Any) -> int:  # noqa: ANN401
        return a.data.ptr

    def is_array(self, a: Any) -> bool:  # noqa: ANN401
        return isinstance(a, self.cp.ndarray)

    def stream(self) -> Any:  # noqa: ANN401
        return self.cp.cuda.Stream(non_blocking=True)

    def use(self, stream: Any) -> contextlib.AbstractContextManager[Any]:  # noqa: ANN401
        return stream

    def handle(self, stream: Any) -> int:  # noqa: ANN401
        return stream.ptr

    def default_stream(self) -> Any:  # noqa: ANN401
        return self.cp.cuda.Stream.null

    def synchronize(self) -> None:
        self.cp.cuda.Device().synchronize()

    def used_bytes(self) -> int:
        return self.cp.get_default_memory_pool().used_bytes()


class Torch(Lib):
    name = "torch"

    def __init__(self) -> None:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            pytest.skip("torch has no CUDA device")
        self.torch = torch

    def dtype(self, dtype: Any) -> Any:  # noqa: ANN401
        return getattr(self.torch, np.dtype(dtype).name)

    def asarray(self, a: np.ndarray) -> Any:  # noqa: ANN401
        return self.torch.as_tensor(np.ascontiguousarray(a), device="cuda")

    def get(self, a: Any) -> np.ndarray:  # noqa: ANN401
        return a.cpu().numpy()

    def empty(self, shape: int | tuple[int, ...], dtype: Any) -> Any:  # noqa: ANN401
        return self.torch.empty(shape, dtype=self.dtype(dtype), device="cuda")

    def astype(self, a: Any, dtype: Any) -> Any:  # noqa: ANN401
        return a.to(self.dtype(dtype))

    def ptr(self, a: Any) -> int:  # noqa: ANN401
        return a.data_ptr()

    def is_array(self, a: Any) -> bool:  # noqa: ANN401
        return isinstance(a, self.torch.Tensor)

    def stream(self) -> Any:  # noqa: ANN401
        return self.torch.cuda.Stream()

    def use(self, stream: Any) -> contextlib.AbstractContextManager[Any]:  # noqa: ANN401
        return self.torch.cuda.stream(stream)

    def handle(self, stream: Any) -> int:  # noqa: ANN401
        return stream.cuda_stream

    def default_stream(self) -> Any:  # noqa: ANN401
        return self.torch.cuda.default_stream()

    def synchronize(self) -> None:
        self.torch.cuda.synchronize()

    def used_bytes(self) -> int:
        return self.torch.cuda.memory_allocated()


class Numba(Lib):
    """Numba device arrays: ``__cuda_array_interface__`` only, no DLPack."""

    name = "numba"

    def __init__(self) -> None:
        with warnings.catch_warnings():
            # numba-cuda 0.30 trips a NumPy 2.5 deprecation on import.
            warnings.simplefilter("ignore", DeprecationWarning)
            from numba import cuda  # noqa: PLC0415

        if not cuda.is_available():
            pytest.skip("numba has no CUDA device")
        self.cuda = cuda

    def asarray(self, a: np.ndarray) -> Any:  # noqa: ANN401
        return self.cuda.to_device(np.ascontiguousarray(a))

    def get(self, a: Any) -> np.ndarray:  # noqa: ANN401
        return a.copy_to_host()

    def empty(self, shape: int | tuple[int, ...], dtype: Any) -> Any:  # noqa: ANN401
        return self.cuda.device_array(shape, dtype=dtype)

    def astype(self, a: Any, dtype: Any) -> Any:  # noqa: ANN401
        return self.asarray(self.get(a).astype(dtype))

    def ptr(self, a: Any) -> int:  # noqa: ANN401
        return a.__cuda_array_interface__["data"][0]

    def is_array(self, a: Any) -> bool:  # noqa: ANN401
        return self.cuda.is_cuda_array(a) and not hasattr(a, "__dlpack__")

    def stream(self) -> Any:  # noqa: ANN401
        return self.cuda.stream()

    def use(self, stream: Any) -> contextlib.AbstractContextManager[Any]:  # noqa: ANN401, ARG002
        return contextlib.nullcontext()  # Numba has no current stream

    def handle(self, stream: Any) -> int:  # noqa: ANN401
        handle = stream.handle
        return int(getattr(handle, "value", handle) or 0)

    def default_stream(self) -> Any:  # noqa: ANN401
        return self.cuda.default_stream()

    def synchronize(self) -> None:
        self.cuda.synchronize()

    def used_bytes(self) -> int:
        free, total = self.cuda.current_context().get_memory_info()
        return total - free


LIBS = {"cupy": CuPy, "torch": Torch, "numba": Numba}


@pytest.fixture
def cuda(gpu_backend: str) -> str:
    if gpu_backend != "cuda":
        pytest.skip("the device arrays here are CUDA arrays")
    return gpu_backend


@pytest.fixture(params=list(LIBS))
def xp(request: pytest.FixtureRequest, cuda: str) -> Lib:  # noqa: ARG001
    """Each installed device-array library in turn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pytest.importorskip(request.param)
    return LIBS[request.param]()


class StreamProtocol:
    """A bare CUDA stream protocol (``__cuda_stream__``) object."""

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


FDMT_ARGS = (1000.0, 1500.0, 64, 256, 0.001, 40)


def pack2(values: np.ndarray) -> np.ndarray:
    """2-bit LSB-first packing (4 samples per byte) along the last axis."""
    out = np.zeros((*values.shape[:-1], values.shape[-1] // 4), dtype=np.uint8)
    for k in range(4):
        out |= values[..., k::4].astype(np.uint8) << (2 * k)
    return out


class TestFDMT:
    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_execute_matches_host(self, cls: type, cuda: str, xp: Lib) -> None:
        wf = np.random.default_rng(0).standard_normal((64, 256), dtype=np.float32)
        want = cls(*FDMT_ARGS, backend=cuda).execute(wf)
        got = cls(*FDMT_ARGS, backend=cuda).execute(xp.asarray(wf))
        assert xp.is_array(got)
        assert tuple(got.shape) == want.shape
        np.testing.assert_array_equal(xp.get(got), want)

    def test_out_is_written_in_place(self, cuda: str, xp: Lib) -> None:
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        wf = xp.asarray(np.random.default_rng(1).standard_normal((64, 256), np.float32))
        out = xp.full(fdmt.plan.dmt_size + 7, -1.0, dtype=np.float32)
        used = xp.used_bytes()
        got = fdmt.execute(wf, out=out)
        assert xp.used_bytes() == used  # no copy of the input or output
        assert xp.ptr(got) == xp.ptr(out)
        assert tuple(got.shape) == (fdmt.plan.dmt_ndms, fdmt.plan.dmt_nsamps)
        assert float(out[-1]) == -1.0
        np.testing.assert_array_equal(xp.get(got), FDMT(*FDMT_ARGS).execute(xp.get(wf)))

    def test_input_is_not_copied(self, cuda: str, xp: Lib) -> None:
        # A view at an offset into a larger allocation is read in place.
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        big = np.random.default_rng(2).standard_normal((65, 256), dtype=np.float32)
        view = xp.asarray(big)[1:]
        np.testing.assert_array_equal(
            xp.get(fdmt.execute(view)),
            FDMT(*FDMT_ARGS, backend=cuda).execute(big[1:]),
        )

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_beams_and_packed(self, cls: type, cuda: str, xp: Lib) -> None:
        values = np.random.default_rng(3).integers(0, 4, (2, 64, 256))
        packed = pack2(values)
        host = cls(*FDMT_ARGS, nbeams=2, backend=cuda)
        dev = cls(*FDMT_ARGS, nbeams=2, backend=cuda)
        got = dev.execute(xp.asarray(packed), 2)
        assert tuple(got.shape) == (2, dev.plan.dmt_ndms, dev.plan.dmt_nsamps)
        np.testing.assert_array_equal(xp.get(got), host.execute(packed, 2))

    def test_streams(self, cuda: str, xp: Lib) -> None:
        # "roll" mode keeps no history, so every call gives the same result.
        fdmt = FDMT(*FDMT_ARGS, mode="roll", backend=cuda)
        wf = np.random.default_rng(4).standard_normal((64, 256), dtype=np.float32)
        want = FDMT(*FDMT_ARGS, mode="roll", backend=cuda).execute(wf)
        stream = xp.stream()
        with xp.use(stream):  # default: the library's current stream
            d_wf = xp.asarray(wf)
            got = fdmt.execute(d_wf)
        xp.synchronize()
        np.testing.assert_array_equal(xp.get(got), want)
        handle = xp.handle(stream)
        for s in (
            stream,
            handle,
            StreamProtocol(handle),
            xp.default_stream(),
            0,
        ):
            got = fdmt.execute(d_wf, stream=s)
            xp.synchronize()
            np.testing.assert_array_equal(xp.get(got), want)

    def test_errors(self, cuda: str, xp: Lib) -> None:
        fdmt = FDMT(*FDMT_ARGS, backend=cuda)
        wf = xp.asarray(np.zeros((64, 256), dtype=np.float32))
        with pytest.raises(TypeError, match="float32"):
            fdmt.execute(xp.astype(wf, np.float64))
        with pytest.raises(TypeError, match="C-contiguous"):
            fdmt.execute(xp.empty((64, 512), np.float32)[:, ::2])
        with pytest.raises(TypeError, match="nbits"):
            fdmt.execute(xp.astype(wf, np.uint8))
        with pytest.raises(ValueError, match="cpu backend"):
            FDMT(*FDMT_ARGS).execute(wf)
        with pytest.raises(ValueError, match="needs at least"):
            fdmt.execute(wf, out=xp.empty(10, np.float32))
        with pytest.raises(TypeError, match="out must be a float32"):
            fdmt.execute(wf, out=xp.empty(fdmt.plan.dmt_size, np.int32))
        with pytest.raises(TypeError, match="out must be a device array"):
            fdmt.execute(wf, out=np.empty(fdmt.plan.dmt_size, dtype=np.float32))
        with pytest.raises(TypeError):
            fdmt.execute(wf, stream="not a stream")

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_stepper(self, cls: type, cuda: str, xp: Lib) -> None:
        wf = np.random.default_rng(5).integers(0, 4, (64, 256)).astype(np.float32)
        want = cls(*FDMT_ARGS, backend=cuda).execute(wf)
        eng = cls(*FDMT_ARGS, backend=cuda)
        stream = xp.stream()
        eng.reset(xp.asarray(wf), stream=stream)
        eng.advance(stream=stream)
        eng.advance_until_remaining(0)  # stays on the reset's stream
        got = eng.finalize()
        xp.synchronize()
        assert xp.is_array(got)
        np.testing.assert_allclose(xp.get(got), want, rtol=1e-5, atol=1e-4)
        # A host reset() afterwards goes back to NumPy results.
        eng.reset(wf)
        eng.advance_until_remaining(0)
        assert isinstance(eng.finalize(), np.ndarray)

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_stepper_packed(self, cls: type, cuda: str, xp: Lib) -> None:
        packed = pack2(np.random.default_rng(18).integers(0, 4, (64, 256)))
        want = cls(*FDMT_ARGS, backend=cuda).execute(packed, 2)
        eng = cls(*FDMT_ARGS, backend=cuda)
        eng.reset(xp.asarray(packed), 2)
        eng.advance_until_remaining(0)
        got = eng.finalize()
        xp.synchronize()
        assert xp.is_array(got)
        np.testing.assert_allclose(xp.get(got), want, rtol=1e-5, atol=1e-4)

    def test_stepper_dmt_buffer(self, cuda: str, xp: Lib) -> None:
        fdmt = FDMT(*FDMT_ARGS, int_tree=False, backend=cuda)
        dmt = xp.empty(fdmt.plan.dmt_size, np.float32)
        fdmt.reset(xp.asarray(np.zeros((64, 256), dtype=np.float32)), dmt)
        fdmt.advance_until_remaining(0)
        assert xp.ptr(fdmt.finalize()) == xp.ptr(dmt)

    @pytest.mark.parametrize("cls", [FDMT, FDMTFFT])
    def test_history_round_trip(self, cls: type, cuda: str, xp: Lib) -> None:
        rng = np.random.default_rng(6)
        blocks = rng.standard_normal((2, 64, 256), dtype=np.float32)
        host = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        host.execute(blocks[0])
        want = host.execute(blocks[1])

        first = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        first.execute(xp.asarray(blocks[0]))
        hist = xp.empty(first.history_state_size(), np.float32)
        assert first.save_history(hist) is hist
        np.testing.assert_array_equal(xp.get(hist), host_history(cls, blocks[0], cuda))

        second = cls(*FDMT_ARGS, mode="valid", backend=cuda)
        second.load_history(hist)
        np.testing.assert_array_equal(
            xp.get(second.execute(xp.asarray(blocks[1]))), want
        )


def host_history(cls: type, block: np.ndarray, backend: str) -> np.ndarray:
    eng = cls(*FDMT_ARGS, mode="valid", backend=backend)
    eng.execute(block)
    return eng.save_history()


DDMT_ARGS = (1000.0, 1500.0, 64, 0.001, 50.0, 1.0)


class TestDDMT:
    @pytest.mark.parametrize("cls", [DDMT, SDMT])
    def test_execute_matches_host(self, cls: type, cuda: str, xp: Lib) -> None:
        rng = np.random.default_rng(7)
        host = cls(*DDMT_ARGS, backend=cuda)
        dev = cls(*DDMT_ARGS, backend=cuda)
        for _ in range(2):  # the second block uses the carried history
            wf = rng.standard_normal((64, 1024), dtype=np.float32)
            want = host.execute(wf)
            got = dev.execute(xp.asarray(wf))
            assert xp.is_array(got)
            assert tuple(got.shape) == want.shape
            np.testing.assert_allclose(xp.get(got), want, rtol=1e-5, atol=1e-4)

    def test_packed_and_time_major(self, cuda: str, xp: Lib) -> None:
        rng = np.random.default_rng(8)
        packed = rng.integers(0, 256, (64, 1024), dtype=np.uint8)
        host = DDMT(*DDMT_ARGS, nbits=8, backend=cuda)
        dev = DDMT(*DDMT_ARGS, nbits=8, backend=cuda)
        got = dev.execute(xp.asarray(packed), 1024)
        assert xp.get(got).dtype == np.int32
        np.testing.assert_array_equal(xp.get(got), host.execute(packed, 1024))
        tm = np.ascontiguousarray(packed.T)
        out = xp.empty(got.shape[0] * dev.get_output_nsamps(1024), np.int32)
        got = dev.execute_time_major(xp.asarray(tm), 1024, out=out)
        assert xp.ptr(got) == xp.ptr(out)
        np.testing.assert_array_equal(xp.get(got), host.execute_time_major(tm, 1024))

    @pytest.mark.parametrize(("nbits", "dtype"), [(32, np.float32), (8, np.uint8)])
    def test_history_round_trip(
        self, nbits: int, dtype: type, cuda: str, xp: Lib
    ) -> None:
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
        run(first, xp.asarray(blocks[0]))
        hist = xp.empty(first.history_state_size(), dtype)
        first.save_history(hist)
        second = DDMT(*DDMT_ARGS, nbits=nbits, backend=cuda)
        second.load_history(hist)
        np.testing.assert_allclose(
            xp.get(run(second, xp.asarray(blocks[1]))), want, rtol=1e-5, atol=1e-4
        )
        with pytest.raises(TypeError, match=np.dtype(dtype).name):
            first.save_history(xp.empty(first.history_state_size(), np.int16))

    def test_errors(self, cuda: str, xp: Lib) -> None:
        dev = DDMT(*DDMT_ARGS, backend=cuda)
        with pytest.raises(ValueError, match="channels"):
            dev.execute(xp.asarray(np.zeros((32, 128), dtype=np.float32)))
        with pytest.raises(ValueError, match="beam"):
            dev.execute(xp.asarray(np.zeros((2, 64, 128), dtype=np.float32)))

    @pytest.mark.parametrize("method", ["brute", "nufft"])
    def test_ddmt_fft(self, method: str, cuda: str, xp: Lib) -> None:
        rng = np.random.default_rng(10)
        args = (1100.0, 1500.0, 64, 0.001, 30.0, 0.25)
        host = DDMTFFT(*args, nbeams=2, method=method, backend=cuda)
        dev = DDMTFFT(*args, nbeams=2, method=method, backend=cuda)
        for _ in range(2):
            wf = rng.standard_normal((2, 64, 2000), dtype=np.float32)
            want = host.execute(wf)
            got = dev.execute(xp.asarray(wf))
            assert xp.is_array(got)
            np.testing.assert_allclose(
                xp.get(got), want, rtol=0, atol=1e-5 * np.abs(want).max()
            )
        hist = xp.empty(dev.history_state_size(), np.float32)
        dev.save_history(hist)
        np.testing.assert_allclose(
            xp.get(hist), host.save_history(), rtol=0, atol=1e-5 * np.abs(want).max()
        )
        fresh = DDMTFFT(*args, nbeams=2, method=method, backend=cuda)
        fresh.load_history(hist)
        wf = rng.standard_normal((2, 64, 2000), dtype=np.float32)
        want = host.execute(wf)
        np.testing.assert_allclose(
            xp.get(fresh.execute(xp.asarray(wf))),
            want,
            rtol=0,
            atol=1e-5 * np.abs(want).max(),
        )

    def test_ddmt_fft_packed(self, cuda: str, xp: Lib) -> None:
        rng = np.random.default_rng(11)
        args = (1100.0, 1500.0, 64, 0.001, 30.0, 0.25)
        packed = rng.integers(0, 256, (64, 2000), dtype=np.uint8)
        want = DDMTFFT(*args, nbits=8, backend=cuda).execute(packed, 2000)
        got = DDMTFFT(*args, nbits=8, backend=cuda).execute(xp.asarray(packed), 2000)
        assert xp.get(got).dtype == np.float32
        np.testing.assert_allclose(
            xp.get(got), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )


class TestCohFDMT:
    def test_execute_matches_host(self, cuda: str, xp: Lib) -> None:
        cfg = dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0)
        coh = CohFDMT(cfg, backend=cuda)
        rng = np.random.default_rng(12)
        data = rng.normal(0.0, 8.0, size=coh.input_size()).astype(np.int8)
        want = coh.execute(data)
        got = coh.execute(xp.asarray(data))
        assert xp.is_array(got)
        assert tuple(got.shape) == want.shape
        np.testing.assert_allclose(
            xp.get(got), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )
        out = xp.empty(coh.dmt_size, np.float32)
        again = coh.execute(xp.asarray(data.view(np.uint8)), out=out, stream=0)
        assert xp.ptr(again) == xp.ptr(out)
        xp.synchronize()
        np.testing.assert_array_equal(xp.get(again), xp.get(got))

    def test_subband_groups(self, cuda: str, xp: Lib) -> None:
        cfg = dmtlib.CohFDMTConfig(
            400.0, 1.0, 16, 4.0e-6, 10.0, 13.0, subband_groups=[6, 10]
        )
        coh = CohFDMT(cfg, backend=cuda)
        rng = np.random.default_rng(13)
        groups = [
            rng.integers(0, 256, coh.input_size(g), dtype=np.uint8) for g in (0, 1)
        ]
        want = coh.execute(groups)
        got = coh.execute([xp.asarray(g) for g in groups])
        np.testing.assert_allclose(
            xp.get(got), want, rtol=0, atol=1e-5 * np.abs(want).max()
        )

    def test_errors(self, cuda: str, xp: Lib) -> None:
        cfg = dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0)
        coh = CohFDMT(cfg, backend=cuda)
        data = np.zeros(coh.input_size(), dtype=np.uint8)
        with pytest.raises(TypeError, match="stream"):
            coh.execute(data, stream=0)
        with pytest.raises(TypeError, match="uint8 or int8"):
            coh.execute(xp.empty(coh.input_size() // 2, np.int16))
        with pytest.raises(TypeError, match="device array"):
            coh.execute([xp.asarray(data), data])
        with pytest.raises(ValueError, match="cpu backend"):
            CohFDMT(cfg).execute(xp.asarray(data))


# Stream semantics checked against PyTorch. Each "slow producer" holds a
# stream with torch.cuda._sleep before the work that writes the input, so a
# transform that is not ordered after that work reads the wrong data.

SLEEP_CYCLES = 200_000_000  # ~0.1 s on an L40


@pytest.fixture
def torch(cuda: str) -> Any:  # noqa: ANN401, ARG001
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    return torch


@contextlib.contextmanager
def slow_producer(torch: Any, stream: Any, src: Any) -> Iterator[Any]:  # noqa: ANN401
    """On ``stream``: sleep, then copy ``src`` into a new tensor (yielded)."""
    with torch.cuda.stream(stream):
        x = torch.zeros_like(src)
        torch.cuda._sleep(SLEEP_CYCLES)  # noqa: SLF001
        x.copy_(src)
        yield x


def roll_fdmt(torch: Any) -> tuple[FDMT, np.ndarray, np.ndarray]:  # noqa: ANN401
    """A warmed-up FDMT, a waterfall and its transform.

    With CUDA's lazy module loading (the default), the first launch of a
    kernel in the process loads its module and synchronizes the device,
    which would hide a missing stream dependency; the warm-up call does it.
    """
    fdmt = FDMT(*FDMT_ARGS, mode="roll", backend="cuda")
    wf = np.random.default_rng(15).standard_normal((64, 256), dtype=np.float32)
    fdmt.execute(torch.zeros((64, 256), device="cuda"))
    torch.cuda.synchronize()
    return fdmt, wf, FDMT(*FDMT_ARGS, mode="roll").execute(wf)


def stream_of_block(torch: Any, ptr: int) -> int | None:  # noqa: ANN401
    """The stream PyTorch's caching allocator assigned to the block at ``ptr``."""
    for seg in torch.cuda.memory_snapshot():
        if seg["address"] <= ptr < seg["address"] + seg["total_size"]:
            return seg["stream"]
    return None


class TestTorchStreams:
    def test_current_stream_is_used(self, torch: Any) -> None:  # noqa: ANN401
        # stream=None inside `with torch.cuda.stream(s)`: the output is
        # allocated on s and the transform queued on s after the producer's
        # work there; synchronizing s alone is enough.
        fdmt, wf, want = roll_fdmt(torch)
        src = torch.from_numpy(wf).cuda()
        s = torch.cuda.Stream()
        torch.cuda.synchronize()
        with slow_producer(torch, s, src) as x:
            got = fdmt.execute(x)
        assert stream_of_block(torch, got.data_ptr()) == s.cuda_stream
        s.synchronize()
        np.testing.assert_array_equal(got.cpu().numpy(), want)

    def test_producer_ordering(self, torch: Any) -> None:  # noqa: ANN401
        # The input is written on stream A, still busy, and the call runs on
        # B while A is current: __dlpack__(stream=B) makes B wait for A.
        fdmt, wf, want = roll_fdmt(torch)
        src = torch.from_numpy(wf).cuda()
        a, b = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with slow_producer(torch, a, src) as x:
            got = fdmt.execute(x, stream=b)
        assert stream_of_block(torch, got.data_ptr()) == b.cuda_stream
        b.synchronize()
        np.testing.assert_array_equal(got.cpu().numpy(), want)

    @pytest.mark.parametrize("default", ["zero", "object"])
    def test_default_stream(self, torch: Any, default: str) -> None:  # noqa: ANN401
        # The legacy default stream (handle 0) is passed to __dlpack__ as 1;
        # PyTorch rejects 0 there. A side stream does not synchronize with it
        # implicitly (PyTorch streams are non-blocking).
        fdmt, wf, want = roll_fdmt(torch)
        src = torch.from_numpy(wf).cuda()
        stream = 0 if default == "zero" else torch.cuda.default_stream()
        side = torch.cuda.Stream()
        torch.cuda.synchronize()
        with slow_producer(torch, side, src) as x:
            got = fdmt.execute(x, stream=stream)
        torch.cuda.default_stream().synchronize()
        np.testing.assert_array_equal(got.cpu().numpy(), want)

    def test_per_thread_default_stream(self, torch: Any) -> None:  # noqa: ANN401
        # Handle 2 is cudaStreamPerThread (DLPack's 2 as well); PyTorch does
        # not export to it and says so.
        fdmt, wf, _ = roll_fdmt(torch)
        with pytest.raises(BufferError, match="per-thread"):
            fdmt.execute(torch.from_numpy(wf).cuda(), stream=2)

    def test_input_dropped_after_call(self, torch: Any) -> None:  # noqa: ANN401
        # The input is allocated on A, the call runs on a busy stream B, and
        # the caller drops the input at once. record_stream(B) keeps the
        # caching allocator from reusing the block on A while B still has to
        # read it.
        fdmt, wf, want = roll_fdmt(torch)
        a, b = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(b):
            torch.cuda._sleep(SLEEP_CYCLES)  # noqa: SLF001
        with torch.cuda.stream(a):
            x = torch.from_numpy(wf).cuda()
            got = fdmt.execute(x, stream=b)
            del x
            junk = torch.full((64, 256), np.nan, device="cuda")
        b.synchronize()
        assert junk.isnan().all()
        np.testing.assert_array_equal(got.cpu().numpy(), want)

    def test_output_type(self, torch: Any) -> None:  # noqa: ANN401
        ddmt = DDMT(*DDMT_ARGS, nbits=8, backend="cuda")
        packed = torch.randint(0, 256, (64, 512), dtype=torch.uint8, device="cuda")
        shape = (len(ddmt.plan.dm_arr), ddmt.get_output_nsamps(512))
        got = ddmt.execute(packed, 512)
        assert type(got) is torch.Tensor
        assert got.dtype == torch.int32
        assert got.device == packed.device
        assert tuple(got.shape) == shape
        assert got.is_contiguous()

    def test_multi_gpu(self, torch: Any) -> None:  # noqa: ANN401
        if torch.cuda.device_count() < 2:
            pytest.skip("needs two visible CUDA devices")
        wf = np.random.default_rng(16).standard_normal((64, 256), dtype=np.float32)
        x1 = torch.from_numpy(wf).to("cuda:1")
        with pytest.raises(ValueError, match="cuda:1 but this instance runs on cuda:0"):
            FDMT(*FDMT_ARGS, backend="cuda", device=0).execute(x1)
        got = FDMT(*FDMT_ARGS, backend="cuda", device=1).execute(x1)
        assert got.device == x1.device
        torch.cuda.synchronize(1)
        np.testing.assert_array_equal(got.cpu().numpy(), FDMT(*FDMT_ARGS).execute(wf))


# The CUDA array interface (CAI) path, through objects that only define
# __cuda_array_interface__ over CuPy memory (the generic case) and Numba.


class CAIOnly:
    """A CAI-only view of a CuPy array, with overridable interface fields."""

    def __init__(self, base: Any, **fields: Any) -> None:  # noqa: ANN401
        self.base = base
        self.shape = base.shape
        self.fields = fields

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        desc = dict(self.base.__cuda_array_interface__)
        desc.pop("stream", None)
        desc.update(self.fields)
        return desc


SPIN_SOURCE = r"""
extern "C" __global__ void spin(long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {}
}
"""


@pytest.fixture
def cp(cuda: str) -> Any:  # noqa: ANN401, ARG001
    return pytest.importorskip("cupy")


def busy_producer(cp: Any, wf: np.ndarray) -> tuple[Any, Any]:  # noqa: ANN401
    """A waterfall written on a new stream after ~0.1 s of spinning on it."""
    spin = cp.RawKernel(SPIN_SOURCE, "spin")
    src = cp.asarray(wf)
    cp.cuda.Device().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        x = cp.zeros_like(src)
        spin((1,), (1,), (np.int64(SLEEP_CYCLES),))
        x[...] = src
    return x, stream


def warm_fdmt(cp: Any) -> tuple[FDMT, np.ndarray, np.ndarray]:  # noqa: ANN401
    """A warmed-up "roll" FDMT (see roll_fdmt), a waterfall and its transform."""
    fdmt = FDMT(*FDMT_ARGS, mode="roll", backend="cuda")
    fdmt.execute(cp.zeros((64, 256), dtype=cp.float32))
    cp.cuda.Device().synchronize()
    wf = np.random.default_rng(17).standard_normal((64, 256), dtype=np.float32)
    return fdmt, wf, FDMT(*FDMT_ARGS, mode="roll").execute(wf)


class TestCudaArrayInterface:
    def test_generic_object(self, cp: Any) -> None:  # noqa: ANN401
        fdmt, wf, want = warm_fdmt(cp)
        x = CAIOnly(cp.asarray(wf))
        out = cp.empty(fdmt.plan.dmt_size, dtype=cp.float32)
        got = fdmt.execute(x, out=CAIOnly(out))
        cp.cuda.Device().synchronize()
        assert isinstance(got, CAIOnly)  # `out` itself: it has no reshape()
        np.testing.assert_array_equal(out.get().reshape(want.shape), want)
        shaped = CAIOnly(cp.empty(want.shape, dtype=cp.float32))
        assert fdmt.execute(x, out=shaped) is shaped
        cp.cuda.Device().synchronize()
        np.testing.assert_array_equal(shaped.base.get(), want)
        # Without out=, there is no library to allocate the result with.
        with pytest.raises(TypeError, match="pass out="):
            fdmt.execute(x)

    @pytest.mark.parametrize("sync", ["default", "0"])
    def test_producer_stream(
        self,
        cp: Any,  # noqa: ANN401
        sync: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The producer's stream is still busy writing the input; the call runs
        # on another stream and must wait for it (an event, not a host sync)
        # unless NUMBA_CUDA_ARRAY_INTERFACE_SYNC=0 turns that off.
        if sync == "0":
            monkeypatch.setenv("NUMBA_CUDA_ARRAY_INTERFACE_SYNC", "0")
        else:
            monkeypatch.delenv("NUMBA_CUDA_ARRAY_INTERFACE_SYNC", raising=False)
        fdmt, wf, want = warm_fdmt(cp)
        x, producer = busy_producer(cp, wf)
        call = cp.cuda.Stream(non_blocking=True)
        out = cp.empty(fdmt.plan.dmt_size, dtype=cp.float32)
        fdmt.execute(CAIOnly(x, stream=producer.ptr), out=out, stream=call)
        assert not producer.done  # the host did not wait
        call.synchronize()
        got = out.get().reshape(want.shape)
        if sync == "0":
            assert not np.array_equal(got, want)  # read before it was written
        else:
            np.testing.assert_array_equal(got, want)

    def test_numba_producer_stream(self, cp: Any) -> None:  # noqa: ANN401
        numba = Numba()
        fdmt, wf, want = warm_fdmt(cp)
        x, producer = busy_producer(cp, wf)
        # A Numba array bound to the busy stream reports it in its CAI.
        nb = numba.cuda.as_cuda_array(x, sync=False).bind(
            numba.cuda.external_stream(producer.ptr)
        )
        assert nb.__cuda_array_interface__["stream"] == producer.ptr
        got = fdmt.execute(nb, stream=0)
        numba.cuda.default_stream().synchronize()
        np.testing.assert_array_equal(got.copy_to_host(), want)

    def test_errors(self, cp: Any) -> None:  # noqa: ANN401
        fdmt = FDMT(*FDMT_ARGS, backend="cuda")
        wf = cp.zeros((64, 256), dtype=cp.float32)
        out = cp.empty(fdmt.plan.dmt_size, dtype=cp.float32)
        with pytest.raises(TypeError, match="C-contiguous"):
            fdmt.execute(CAIOnly(cp.zeros((64, 512), cp.float32)[:, ::2]), out=out)
        with pytest.raises(TypeError, match="writeable"):
            ptr = out.data.ptr
            fdmt.execute(wf, out=CAIOnly(out, data=(ptr, True)))
        with pytest.raises(TypeError, match="masked"):
            fdmt.execute(CAIOnly(wf, mask=CAIOnly(wf)), out=out)
        with pytest.raises(ValueError, match="stream 0"):
            fdmt.execute(CAIOnly(wf, stream=0), out=out)
        with pytest.raises(TypeError, match="typestr"):
            fdmt.execute(CAIOnly(wf, typestr=">f4"), out=out)
        with pytest.raises(TypeError, match="float32"):
            fdmt.execute(CAIOnly(wf, typestr="<i4"), out=out)
        host = np.zeros((64, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="not cuda device memory"):
            fdmt.execute(CAIOnly(wf, data=(host.ctypes.data, False)), out=out)
        with pytest.raises(ValueError, match="cpu backend"):
            FDMT(*FDMT_ARGS).execute(CAIOnly(wf))

    def test_dlpack_preferred(self, cp: Any) -> None:  # noqa: ANN401
        # CuPy arrays have both protocols; DLPack is used, so the result is a
        # CuPy array allocated by CuPy (the CAI path could not allocate).
        fdmt, wf, want = warm_fdmt(cp)
        x = cp.asarray(wf)
        assert hasattr(x, "__cuda_array_interface__")
        got = fdmt.execute(x)
        assert isinstance(got, cp.ndarray)
        np.testing.assert_array_equal(got.get(), want)
