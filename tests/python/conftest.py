import functools

import pytest


@functools.cache
def _usable(backend: str) -> bool:
    """Whether this dmtlib build has `backend` and a device for it here."""
    import dmtlib  # noqa: PLC0415 - imported lazily at collection time

    if backend not in dmtlib.available_backends():
        return False
    try:
        dmtlib.FDMT(1000.0, 1500.0, 4, 8, 0.001, 2, backend=backend)
    except Exception:  # noqa: BLE001 - any failure means "not usable here"
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
    skip = pytest.mark.skip(
        reason="this dmtlib build has no GPU backend, or no GPU device is available"
    )
    metafunc.parametrize("gpu_backend", [pytest.param(None, marks=skip)])
