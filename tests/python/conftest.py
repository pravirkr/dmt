import sys
from pathlib import Path

import pytest

# Ensure the local build/src and src directories take precedence over stale editable site-packages
repo_root = Path(__file__).resolve().parents[2]
build_src = repo_root / "build" / "src"
src_dir = repo_root / "src"

sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]

if str(build_src) not in sys.path:
    sys.path.insert(0, str(build_src))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


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
        libcudmt.FDMTGPU(1000.0, 1500.0, 4, 8, 0.001, 2)
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
