from importlib import metadata

try:
    __version__ = metadata.version(__name__)
except metadata.PackageNotFoundError:
    __version__ = "0.7.0"

from . import libdmt as _libdmt
from .grid import (
    calculate_snr_loss,
    generate_optimal_dm_grid,
    generate_optimal_dt_grid,
)
from .libdmt import (
    DDMT,
    DDMTFFT,
    FDMT,
    FDMTFFT,
    SDMT,
    BasebandFormat,
    CohFDMT,
    CohFDMTConfig,
    CohFDMTPlan,
    DDMTPlan,
    FDMTComplexity,
    FDMTMemoryUsage,
    FDMTPlan,
    FFTPlanner,
    LevinConfig,
    add_frb_track,
    available_backends,
    compute_fdmt,
    compute_fdmt_fft,
    export_fft_wisdom,
    forget_fft_wisdom,
    get_fft_planner,
    import_fft_wisdom,
    pack_baseband,
    set_fft_planner,
    simulate_baseband,
)

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
    _libdmt._set_log_level(value)  # noqa: SLF001 - our own extension


def get_log_level() -> str:
    """Return dmt's current log level (``"off"`` or ``"debug"``)."""
    value = _libdmt._get_log_level()  # noqa: SLF001
    return next(name for name, v in _LOG_LEVELS.items() if v == value)


__all__ = [
    "DDMT",
    "DDMTFFT",
    "FDMT",
    "FDMTFFT",
    "SDMT",
    "BasebandFormat",
    "CohFDMT",
    "CohFDMTConfig",
    "CohFDMTPlan",
    "DDMTPlan",
    "FDMTComplexity",
    "FDMTMemoryUsage",
    "FDMTPlan",
    "FFTPlanner",
    "LevinConfig",
    "add_frb_track",
    "available_backends",
    "calculate_snr_loss",
    "compute_fdmt",
    "compute_fdmt_fft",
    "export_fft_wisdom",
    "forget_fft_wisdom",
    "generate_optimal_dm_grid",
    "generate_optimal_dt_grid",
    "get_fft_planner",
    "get_log_level",
    "import_fft_wisdom",
    "pack_baseband",
    "set_fft_planner",
    "set_log_level",
    "simulate_baseband",
]
