import contextlib
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from .libdmt import (
    DDMTCPU,
    FDMTCPU,
    FDMTFFTCPU,
    CohFDMTCPU,
    CohFDMTPlan,
    DDMTPlan,
    LevinConfig,
    compute_fdmt,
    compute_fdmt_fft,
)

with contextlib.suppress(Exception):
    from .libcudmt import DDMTCUDA, FDMTFFTGPU, FDMTGPU, CohFDMTCUDA, CohFDMTGPU
    DDMTGPU = DDMTCUDA

__all__ = [
    "DDMTCPU",
    "DDMTCUDA",
    "DDMTGPU",
    "FDMTCPU",
    "FDMTFFTCPU",
    "FDMTFFTGPU",
    "FDMTGPU",
    "CohFDMTCPU",
    "CohFDMTCUDA",
    "CohFDMTGPU",
    "CohFDMTPlan",
    "DDMTPlan",
    "LevinConfig",
    "compute_fdmt",
    "compute_fdmt_fft",
]

