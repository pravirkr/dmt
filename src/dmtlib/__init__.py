import contextlib
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from .libdmt import (
    FDMTCPU,
    FDMTFFTCPU,
    CohFDMTPlan,
    CohFDMTCPU,
    compute_fdmt,
    compute_fdmt_fft,
)

with contextlib.suppress(Exception):
    from .libcudmt import FDMTFFTGPU, FDMTGPU, CohFDMTCUDA, CohFDMTGPU

__all__ = [
    "FDMTCPU",
    "FDMTFFTCPU",
    "FDMTFFTGPU",
    "FDMTGPU",
    "CohFDMTPlan",
    "CohFDMTCPU",
    "CohFDMTCUDA",
    "CohFDMTGPU",
    "compute_fdmt",
    "compute_fdmt_fft",
]
