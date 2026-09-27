import contextlib
from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.3.0"

from . import libdmt as _libdmt
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

_log_modules = [_libdmt]

with contextlib.suppress(Exception):
    from . import libcudmt as _libcudmt
    from .libcudmt import (
        DDMTCUDA,
        FDMTCUDA,
        FDMTFFTCUDA,
        CohFDMTCUDA,
    )

    _log_modules.append(_libcudmt)

_LOG_LEVELS = {"off": 0, "debug": 1}


def set_log_level(level: str) -> None:
    """Set dmt's process-wide log level: ``"off"`` (default) or ``"debug"``.

    ``"debug"`` writes construction-time details (plan parameters, FFT plans,
    CUDA fusion geometry) to stderr. ``execute()`` never logs, and errors are
    raised as exceptions. For a description of a plan or engine, print its
    ``summary()``.
    """
    try:
        value = _LOG_LEVELS[level]
    except KeyError:
        msg = f"log level must be one of {sorted(_LOG_LEVELS)}, got {level!r}"
        raise ValueError(msg) from None
    for module in _log_modules:
        module._set_log_level(value)  # noqa: SLF001 - our own extensions


def get_log_level() -> str:
    """Return dmt's current log level (``"off"`` or ``"debug"``)."""
    value = _libdmt._get_log_level()  # noqa: SLF001
    return next(name for name, v in _LOG_LEVELS.items() if v == value)


__all__ = [
    "DDMTCPU",
    "DDMTCUDA",
    "FDMTCPU",
    "FDMTCUDA",
    "FDMTFFTCPU",
    "FDMTFFTCUDA",
    "CohFDMTCPU",
    "CohFDMTCUDA",
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
    "get_log_level",
    "set_log_level",
]
