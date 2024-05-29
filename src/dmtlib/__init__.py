import contextlib
from importlib import metadata

__version__ = metadata.version(__name__)

from .libdmt import FDMTCPU

with contextlib.suppress(Exception):
    from .libcudmt import FDMTGPU

__all__ = ["FDMTCPU", "FDMTGPU"]
