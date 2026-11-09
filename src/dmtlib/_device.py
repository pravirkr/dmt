"""Device-array support for the DLPack paths of the engines (private).

The extension calls these to resolve the stream of a device-memory call and
to allocate its output with the caller's own array library, so that the
result is a native PyTorch / CuPy array from that library's memory pool.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any


def _library(like: Any) -> str:  # noqa: ANN401
    return type(like).__module__.partition(".")[0]


def stream_handle(stream: Any, like: Any) -> int:  # noqa: ANN401
    """Native stream handle (``cudaStream_t`` / ``hipStream_t`` as an int).

    ``stream`` is an int handle, an object with ``__cuda_stream__``, a
    ``torch.cuda.Stream`` (``cuda_stream``) or a ``cupy.cuda.Stream``
    (``ptr``). ``None`` selects the current stream of ``like``'s array
    library on ``like``'s device (0, the default stream, for other
    libraries).
    """
    if stream is None:
        return _current_stream(like)
    if isinstance(stream, int):
        return stream
    protocol = getattr(stream, "__cuda_stream__", None)
    if protocol is not None:
        return int(protocol()[1])
    for attr in ("cuda_stream", "ptr"):
        handle = getattr(stream, attr, None)
        if handle is not None:
            return int(handle)
    msg = (
        "stream must be None, an int handle, or a torch.cuda.Stream / "
        f"cupy.cuda.Stream; got {type(stream).__name__}"
    )
    raise TypeError(msg)


def _current_stream(like: Any) -> int:  # noqa: ANN401
    library = _library(like)
    if library == "torch":
        import torch  # noqa: PLC0415

        return torch.cuda.current_stream(like.device).cuda_stream
    if library == "cupy":
        import cupy  # noqa: PLC0415

        with like.device:
            return cupy.cuda.get_current_stream().ptr
    return 0


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
    if stream == 0:
        return torch.cuda.stream(torch.cuda.default_stream(device))
    return torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device))


class _StreamHandle:
    """A raw stream handle as a CUDA stream protocol object."""

    def __init__(self, handle: int) -> None:
        self.handle = handle

    def __cuda_stream__(self) -> tuple[int, int]:
        return (0, self.handle)


def empty(like: Any, shape: tuple[int, ...], dtype: str, stream: int) -> Any:  # noqa: ANN401
    """Uninitialised array of ``shape`` and ``dtype`` on ``like``'s device.

    Allocated by ``like``'s own library (PyTorch or CuPy) on ``stream``, so
    that its caching allocator orders reuse of the memory after the call.
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
    msg = (
        f"cannot allocate the output for a {type(like).__module__}."
        f"{type(like).__name__} device array: pass out= (a writeable, "
        f"C-contiguous {dtype} device array)"
    )
    raise TypeError(msg)


def view(out: Any, count: int, shape: tuple[int, ...]) -> Any:  # noqa: ANN401
    """Return the first ``count`` elements of contiguous ``out`` as ``shape``."""
    if tuple(out.shape) == shape:
        return out
    return out.reshape(-1)[:count].reshape(shape)
