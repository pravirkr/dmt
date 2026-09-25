import contextlib
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from .grid import (
    calculate_snr_loss,
    generate_optimal_dm_grid,
    generate_optimal_dt_grid,
)
from .libdmt import (
    DDMTCPU,
    FDMTCPU,
    FDMTFFTCPU,
    CohFDMTCPU,
    CohFDMTPlan,
    DDMTPlan,
    FDMTComplexity,
    FDMTMemoryUsage,
    FDMTPlan,
    LevinConfig,
    add_frb_track,
    compute_fdmt,
    compute_fdmt_fft,
)

with contextlib.suppress(Exception):
    from .libcudmt import (
        CohFDMTCUDA,
        DDMTCUDA,
        FDMTCUDA,
        FDMTFFTCUDA,
    )

__all__ = [
    "CohFDMTCPU",
    "CohFDMTCUDA",
    "CohFDMTPlan",
    "DDMTCPU",
    "DDMTCUDA",
    "DDMTPlan",
    "FDMTCPU",
    "FDMTCUDA",
    "FDMTFFTCPU",
    "FDMTFFTCUDA",
    "CohFDMTPlan",
    "DDMTPlan",
    "FDMTComplexity",
    "FDMTMemoryUsage",
    "FDMTPlan",
    "LevinConfig",
    "add_frb_track",
    "calculate_snr_loss",
    "compute_fdmt",
    "compute_fdmt_fft",
    "generate_optimal_dm_grid",
    "generate_optimal_dt_grid",
]

