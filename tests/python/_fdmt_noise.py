"""White-noise validation helpers for FDMT/DDMT tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, libdmt

if TYPE_CHECKING:
    from collections.abc import Sequence

Metrics = dict[str, Any]
Report = dict[str, list[Metrics]]

# Engines whose output must be white for white input (fdmt_box is coloured).
WHITE_ENGINES = ("fdmt_raw", "fdmt_fft_raw", "ddmt", "ddmt_fft")


def collect_noise(
    *,
    nchans: int = 32,
    nsamps: int = 2048,
    realizations: int = 256,
    dt_grid: Sequence[int] = (0, 128, 512),
    f_min: float = 400.0,
    f_max: float = 800.0,
    tsamp: float = 0.001,
    seed: int = 20261103,
    backend: str = "cpu",
    blocks: int = 2,
) -> tuple[dict[str, np.ndarray], libdmt.FDMTPlan]:
    """Return complete-support streams from the same independent input beams."""
    grid = np.asarray(dt_grid, dtype=np.int64)
    common = {"nbeams": realizations, "backend": backend}
    plain = FDMT(
        f_min,
        f_max,
        nchans,
        nsamps,
        tsamp,
        dt_grid=grid,
        use_box_smearing=False,
        mode="valid",
        **common,
    )
    boxed = FDMT(
        f_min,
        f_max,
        nchans,
        nsamps,
        tsamp,
        dt_grid=grid,
        use_box_smearing=True,
        mode="valid",
        **common,
    )
    fft = FDMTFFT(
        f_min,
        f_max,
        nchans,
        nsamps,
        tsamp,
        dt_grid=grid,
        use_box_smearing=False,
        mode="valid",
        **common,
    )
    dms = np.asarray(plain.dm_grid_final, dtype=np.float32)
    ddmt = DDMT(f_min, f_max, nchans, tsamp, dms, **common)
    ddmt_fft = DDMTFFT(f_min, f_max, nchans, tsamp, dms, **common)
    engines = {
        "fdmt_raw": plain,
        "fdmt_box": boxed,
        "fdmt_fft_raw": fft,
        "ddmt": ddmt,
        "ddmt_fft": ddmt_fft,
    }
    rng = np.random.default_rng(seed)
    startup = max(plain.startup_samples, fft.startup_samples) + 128
    warmup_blocks = max(1, math.ceil(startup / nsamps))
    streams = {name: [] for name in engines}
    for block in range(warmup_blocks + blocks):
        x = rng.standard_normal((realizations, nchans, nsamps), dtype=np.float32)
        for name, engine in engines.items():
            y = engine.execute(x)
            if block >= warmup_blocks:
                y = np.asarray(y).reshape(realizations, grid.size, -1)
                if y.shape[-1] != nsamps:
                    message = f"{name}: incomplete warmup/output support"
                    raise AssertionError(message)
                streams[name].append(y.copy())
    return {k: np.concatenate(v, axis=-1) for k, v in streams.items()}, plain.plan


def _exponential_distance(samples: np.ndarray) -> float:
    ordered = np.sort(samples)
    cdf = -np.expm1(-ordered)
    ranks = np.arange(1, ordered.size + 1) / ordered.size
    return float(max(np.max(ranks - cdf), np.max(cdf - ranks + 1 / ordered.size)))


def white_noise_metrics(
    rows: np.ndarray,
    nchans: int,
    *,
    alpha: float = 1e-6,
    comparisons: int = 1,
) -> Metrics:
    """Known-variance white-noise diagnostics with a family-wise error budget."""
    rows = np.asarray(rows, dtype=np.float64) / math.sqrt(nchans)
    b, n = rows.shape
    interior = slice(1, -1 if n % 2 == 0 else None)
    bins = sorted({1, 2, 4, n // 8, n // 4, n // 2 - 2, n // 2 - 1})
    lags = [lag for lag in (1, 2, 3, 8, 16) if lag < n]
    checks = 2 + len(lags) + (n // 2 - int(n % 2 == 0)) + len(bins)
    tail = math.log(2 * comparisons * checks / alpha)
    count = b * n
    mean_bound = math.sqrt(2 * tail / count)
    energy_bound = 2 * math.sqrt(tail / count) + 2 * tail / count
    acf = {str(lag): float(np.mean(rows[:, :-lag] * rows[:, lag:])) for lag in lags}
    acf_bounds = {
        str(lag): (math.sqrt(2 * b * (n - lag) * tail) + 2 * tail) / (b * (n - lag))
        for lag in lags
    }
    powers = np.abs(np.fft.rfft(rows, axis=-1)) ** 2 / n
    spectrum = powers.mean(axis=0)
    spectral_bound = 2 * math.sqrt(tail / (2 * b)) + 2 * tail / (2 * b)
    distances = {str(k): _exponential_distance(powers[:, k]) for k in bins}
    cdf_bound = math.sqrt(tail / (2 * b))
    mean, energy = float(rows.mean()), float(np.mean(rows**2))
    passed = (
        abs(mean) <= mean_bound
        and abs(energy - 1) <= energy_bound
        and all(abs(acf[k]) <= acf_bounds[k] for k in acf)
        and np.max(np.abs(spectrum[interior] - 1)) <= spectral_bound
        and max(distances.values()) <= cdf_bound
    )
    return {
        "passed": bool(passed),
        "mean": mean,
        "energy": energy,
        "acf": acf,
        "spectral_mean": spectrum.tolist(),
    }


def analyze(
    streams: dict[str, np.ndarray],
    plan: libdmt.FDMTPlan,
    *,
    alpha: float = 1e-6,
) -> Report:
    """Check white engines; retain box filtering as an expected coloured control."""
    comparisons = len(WHITE_ENGINES) * len(plan.dt_grid_final)
    return {
        name: [
            white_noise_metrics(
                data[:, trial], plan.nchans, alpha=alpha, comparisons=comparisons
            )
            for trial in range(data.shape[1])
        ]
        for name, data in streams.items()
    }


def cold_start_noise(
    *,
    mean: float,
    nchans: int = 32,
    nsamps: int = 8192,
    realizations: int = 128,
    delay: int = 2048,
    f_min: float = 400.0,
    f_max: float = 800.0,
    tsamp: float = 0.001,
    seed: int = 20261103,
    backend: str = "cpu",
    alpha: float = 1e-6,
) -> Metrics:
    """One cold-start FDMT block versus the same data with its startup dropped."""
    common = {"nbeams": realizations, "backend": backend}
    fdmt = FDMT(
        f_min,
        f_max,
        nchans,
        nsamps,
        tsamp,
        dt_grid=np.array([delay]),
        use_box_smearing=False,
        mode="valid",
        **common,
    )
    ddmt = DDMT(
        f_min,
        f_max,
        nchans,
        tsamp,
        np.asarray(fdmt.dm_grid_final, dtype=np.float32),
        **common,
    )
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((realizations, nchans, nsamps), dtype=np.float32)
    x += np.float32(mean)
    cold = np.asarray(fdmt.execute(x)).reshape(realizations, -1)
    skip = fdmt.startup_samples
    series = {
        "fdmt_cold": cold,
        "fdmt_trimmed": cold[:, skip:],
        "ddmt": np.asarray(ddmt.execute(x)).reshape(realizations, -1),
    }
    result: Metrics = {"mean": mean, "startup_samples": skip}
    for name, y in series.items():
        y64 = np.asarray(y, dtype=np.float64)
        y64 -= y64.mean(axis=1, keepdims=True)
        result[name] = white_noise_metrics(y64, nchans, alpha=alpha, comparisons=3)
    return result
