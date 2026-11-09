import functools
import os

import pytest


def _gpu_required() -> bool:
    """DMT_REQUIRE_GPU is set (to anything but "" or "0").

    GPU runners set it: a missing GPU backend or device then fails the run
    instead of skipping every GPU test.
    """
    return os.environ.get("DMT_REQUIRE_GPU", "") not in ("", "0")


@functools.cache
def _usable(backend: str) -> bool:
    """Whether this dmtlib build has `backend` and a device for it here."""
    import dmtlib  # noqa: PLC0415 - imported lazily at collection time

    if backend not in dmtlib.available_backends():
        return False
    try:
        dmtlib.FDMT(1000.0, 1500.0, 4, 8, 0.001, 2, backend=backend)
    except Exception:
        if _gpu_required():
            raise  # DMT_REQUIRE_GPU: show why the backend is unusable
        return False
    return True


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # `gpu_backend`: the GPU backend of this build (CUDA or HIP) when it has a
    # device, or one skipped case otherwise.
    if "gpu_backend" not in metafunc.fixturenames:
        return
    backends = [b for b in ("cuda", "hip") if _usable(b)]
    if backends:
        metafunc.parametrize("gpu_backend", backends)
        return
    if _gpu_required():
        msg = "DMT_REQUIRE_GPU is set but this dmtlib build has no GPU backend"
        raise RuntimeError(msg)
    skip = pytest.mark.skip(
        reason="this dmtlib build has no GPU backend, or no GPU device is available"
    )
    metafunc.parametrize("gpu_backend", [pytest.param(None, marks=skip)])
