"""Device-array support for the device-memory paths of the engines (private).

The extension calls these to resolve the stream of a device-memory call, to
export DLPack arrays to it, and to allocate its output with the caller's own
array library, so that the result is a native PyTorch / CuPy / Numba array
from that library's memory pool.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any


def _library(like: Any) -> str:  # noqa: ANN401
    return type(like).__module__.partition(".")[0]


def stream_handle(stream: Any, like: Any) -> int:  # noqa: ANN401
    """Native stream handle (``cudaStream_t`` / ``hipStream_t`` as an int).

    ``stream`` is an int handle, an object with ``__cuda_stream__``, a
    ``torch.cuda.Stream`` (``cuda_stream``), a ``cupy.cuda.Stream``
    (``ptr``) or a Numba stream (``handle``). ``None`` selects the current
    stream of ``like``'s array library on ``like``'s device: PyTorch's and
    CuPy's current stream, Numba's default stream, and 0 (the default
    stream) for other libraries.
    """
    if stream is None:
        return _current_stream(like)
    if isinstance(stream, int):
        return stream
    protocol = getattr(stream, "__cuda_stream__", None)
    if protocol is not None:
        return int(protocol()[1])
    if _library(stream) == "numba":
        return _numba_handle(stream)
    for attr in ("cuda_stream", "ptr"):
        handle = getattr(stream, attr, None)
        if handle is not None:
            return int(handle)
    msg = (
        "stream must be None, an int handle, or a torch.cuda.Stream / "
        f"cupy.cuda.Stream / numba stream; got {type(stream).__name__}"
    )
    raise TypeError(msg)


def dlpack(
    obj: Any,  # noqa: ANN401
    stream: int,
    native: int,
    max_version: tuple[int, int],
) -> Any:  # noqa: ANN401
    """``obj.__dlpack__(stream=stream, ...)`` for a call queued on ``native``.

    ``stream`` is the DLPack stream value (CUDA's legacy default stream is
    1) and ``native`` the raw handle of the same stream. PyTorch tensors are
    exported with their device current (PyTorch refuses otherwise), and
    ``record_stream(native)`` marks their memory as in use on the call's
    stream: PyTorch's caching allocator then does not hand it out again
    before the call's work has run, even if the caller drops the tensor
    right away.
    """
    if _library(obj) == "torch":
        import torch  # noqa: PLC0415

        device = obj.device
        with torch.cuda.device(device):
            capsule = _export(obj, stream, max_version)
            obj.record_stream(_torch_stream(native, device))
        return capsule
    return _export(obj, stream, max_version)


def _export(obj: Any, stream: int, max_version: tuple[int, int]) -> Any:  # noqa: ANN401
    try:
        return obj.__dlpack__(stream=stream, max_version=max_version)
    except TypeError:
        # Producers predating DLPack 1.0 take no max_version.
        return obj.__dlpack__(stream=stream)


def _current_stream(like: Any) -> int:  # noqa: ANN401
    library = _library(like)
    if library == "torch":
        import torch  # noqa: PLC0415

        return torch.cuda.current_stream(like.device).cuda_stream
    if library == "cupy":
        import cupy  # noqa: PLC0415

        with like.device:
            return cupy.cuda.get_current_stream().ptr
    if library == "numba":
        from numba import cuda  # noqa: PLC0415

        # Numba has no current stream; its default stream is the legacy one
        # (0) or, with NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM, the per-thread one.
        return _numba_handle(cuda.default_stream())
    return 0


def _numba_handle(stream: Any) -> int:  # noqa: ANN401
    """Raw handle of a Numba stream (``ctypes.c_void_p`` or ``CUstream``)."""
    handle = stream.handle
    value = getattr(handle, "value", handle)  # c_void_p(None) is stream 0
    return int(value or 0)


def _on_stream(like: Any, stream: int) -> AbstractContextManager[Any]:  # noqa: ANN401
    """Make ``stream`` current in ``like``'s library, for stream-ordered pools."""
    if _library(like) == "torch":
        return _on_torch_stream(like, stream)
    import cupy  # noqa: PLC0415

    if stream == cupy.cuda.get_current_stream().ptr:
        return nullcontext()
    if stream == 0:
        return cupy.cuda.Stream.null
    if hasattr(cupy.cuda.Stream, "from_external"):
        return cupy.cuda.Stream.from_external(_StreamHandle(stream))
    return cupy.cuda.ExternalStream(stream)


def _on_torch_stream(like: Any, stream: int) -> AbstractContextManager[Any]:  # noqa: ANN401
    import torch  # noqa: PLC0415

    device = like.device
    if stream == torch.cuda.current_stream(device).cuda_stream:
        return nullcontext()
    return torch.cuda.stream(_torch_stream(stream, device))


def _torch_stream(stream: int, device: Any) -> Any:  # noqa: ANN401
    """``torch.cuda.Stream`` object of the raw handle ``stream`` on ``device``."""
    import torch  # noqa: PLC0415

    current = torch.cuda.current_stream(device)
    if stream == current.cuda_stream:
        return current
    if stream == 0:
        return torch.cuda.default_stream(device)
    return torch.cuda.ExternalStream(stream, device=device)


class _StreamHandle:
    """A raw stream handle as a CUDA stream protocol object."""

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


def empty(
    like: Any,  # noqa: ANN401
    shape: tuple[int, ...],
    dtype: str,
    stream: int,
    device: int,
) -> Any:  # noqa: ANN401
    """Uninitialised array of ``shape`` and ``dtype`` on ``like``'s device.

    Allocated by ``like``'s own library (PyTorch, CuPy or Numba) on
    ``stream``, so that its caching allocator orders reuse of the memory
    after the call. ``device`` is the device ordinal of the call (Numba
    arrays do not carry one).
    """
    library = _library(like)
    if library == "torch":
        import torch  # noqa: PLC0415

        with _on_stream(like, stream):
            return torch.empty(shape, dtype=getattr(torch, dtype), device=like.device)
    if library == "cupy":
        import cupy  # noqa: PLC0415

        with like.device, _on_stream(like, stream):
            return cupy.empty(shape, dtype=dtype)
    if library == "numba":
        from numba import cuda  # noqa: PLC0415

        with cuda.gpus[device]:
            # The array is bound to the call's stream, so the stream its
            # __cuda_array_interface__ reports orders later consumers after
            # the call.
            bound = cuda.external_stream(stream) if stream else 0
            return cuda.device_array(shape, dtype=dtype, stream=bound)
    msg = (
        f"cannot allocate the output for a {type(like).__module__}."
        f"{type(like).__name__} device array: pass out= (a writeable, "
        f"C-contiguous {dtype} device array)"
    )
    raise TypeError(msg)


def view(out: Any, count: int, shape: tuple[int, ...]) -> Any:  # noqa: ANN401
    """Return the first ``count`` elements of contiguous ``out`` as ``shape``.

    ``out`` itself when it already has that shape, or when it cannot be
    reshaped (a bare ``__cuda_array_interface__`` object, for instance).
    """
    if tuple(getattr(out, "shape", ())) == shape or not hasattr(out, "reshape"):
        return out
    return out.reshape(-1)[:count].reshape(shape)
