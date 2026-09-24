import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "cuda: tests that require the CUDA Python extension and a GPU"
    )


def _cuda_available() -> bool:
    try:
        import dmtlib.libcudmt as libcudmt
    except Exception:
        return False
    try:
        libcudmt.FDMTCUDA(1000.0, 1500.0, 4, 8, 0.001, 2)
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _cuda_available():
        return
    skip_cuda = pytest.mark.skip(
        reason="dmtlib.libcudmt is not importable or no CUDA device is available"
    )
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip_cuda)
