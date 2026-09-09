import contextlib
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from .libdmt import FDMTCPU

with contextlib.suppress(Exception):
    from .libcudmt import FDMTGPU

__all__ = ["FDMTCPU", "FDMTGPU"]
