"""Compare FDMT and DDMT Gaussian-noise statistics, without measuring sensitivity.

Run from a checkout with dmtlib installed::

    python bench/scripts/fdmt_noise.py --output /tmp/fdmt-noise
    python bench/scripts/fdmt_noise.py --cold-start-mean 1.0 --output /tmp/cold

The first form checks steady-state (complete-history) output of unfiltered
FDMT, FDMT-FFT, DDMT and DDMT-FFT against the white-noise hypothesis, with
box-filtered FDMT as a coloured control. The second additionally runs FDMT
from a cold start on nonzero-mean input, once keeping and once discarding its
``startup_samples`` zero-filled outputs, next to DDMT's first call.

Only NumPy and dmtlib are required; --plot additionally needs matplotlib.
Trials and Fourier bins share input and are correlated. Distribution tests
therefore use independent beams at each fixed trial/bin, never pooled bins.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, libdmt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

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
) -> tuple[dict[str, np.ndarray], libdmt.FDMTPlan, Metrics]:
    """Return complete-support streams from the same independent input beams.

    The returned time rows span consecutive execute calls. Each engine receives
    the same warmup and measured data; DDMT uses FDMT's effective physical DMs.
    Arrays are (independent realization, trial, time), including at nbeams=1.
    """
    grid = np.asarray(dt_grid, dtype=np.int64)
    if (
        nchans < 2
        or nsamps < 16
        or realizations < 2
        or blocks < 1
        or grid.ndim != 1
        or not grid.size
        or np.any(grid < 0)
        or np.any(np.diff(grid) <= 0)
    ):
        message = (
            "Need >=2 channels/realizations, >=16 samples, >=1 block, "
            "and strictly increasing nonnegative integer delays"
        )
        raise ValueError(message)
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
    # No assumption that a gulp is longer than the maximum dispersion delay;
    # the FFT engines add an interpolation guard on both sides.
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
    metadata = {
        "nchans": nchans,
        "nsamps": nsamps,
        "realizations": realizations,
        "dt_grid": grid.tolist(),
        "dm_grid": plain.dm_grid_final.tolist(),
        "f_min": f_min,
        "f_max": f_max,
        "tsamp": tsamp,
        "seed": seed,
        "backend": backend,
        "blocks": blocks,
        "warmup_blocks": warmup_blocks,
        "startup_samples": {
            "fdmt": plain.startup_samples,
            "fdmt_fft": fft.startup_samples,
        },
        "extension": str(libdmt.__file__),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    return (
        {k: np.concatenate(v, axis=-1) for k, v in streams.items()},
        plain.plan,
        metadata,
    )


def exponential_distance(samples: np.ndarray) -> float:
    """Two-sided empirical CDF distance to Exp(1), without fitting its scale."""
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
    """Known-variance white-noise diagnostics with a family-wise error budget.

    Mean, energy and spectral-mean bounds use Gaussian/chi-square tail bounds.
    Fixed-bin CDF bounds use DKW over independent realizations. ACF uses an
    exact Gaussian quadratic-form tail bound for the lag-product sum. Allocate
    alpha across all diagnostics and the caller's number of rows/configurations;
    no independence between diagnostics or between trials is assumed.
    """
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
    # For A=sum_i x_i*x_{i+lag}, ||Q||_F^2=M/2, ||Q||_op<=1,
    # and E[A]=0. Laurent-Massart applied to Q's eigenvalues gives this bound.
    acf = {str(lag): float(np.mean(rows[:, :-lag] * rows[:, lag:])) for lag in lags}
    acf_bounds = {
        str(lag): (math.sqrt(2 * b * (n - lag) * tail) + 2 * tail) / (b * (n - lag))
        for lag in lags
    }
    powers = np.abs(np.fft.rfft(rows, axis=-1)) ** 2 / n
    spectrum = powers.mean(axis=0)
    spectral_bound = 2 * math.sqrt(tail / (2 * b)) + 2 * tail / (2 * b)
    distances = {str(k): exponential_distance(powers[:, k]) for k in bins}
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
        "mean_bound": mean_bound,
        "energy": energy,
        "energy_bound": energy_bound,
        "acf": acf,
        "acf_bounds": acf_bounds,
        "spectral_mean": spectrum.tolist(),
        "spectral_mean_bound": spectral_bound,
        "cdf_distance": distances,
        "cdf_bound": cdf_bound,
        "samples": n,
        "realizations": b,
        # These endpoints intentionally do not enter chi-square_2 checks.
        "endpoint_mean": {
            "dc": float(spectrum[0]),
            **({"nyquist": float(spectrum[-1])} if n % 2 == 0 else {}),
        },
    }


def box_spectrum(plan: libdmt.FDMTPlan, trial: int, samples: int) -> np.ndarray:
    """Predict finite-record periodogram / nchans for stationary box output.

    Bartlett weighting accounts for the finite, non-circular record. Input
    channels have unit variance. This is a PSD model, not an exponential-law
    assertion about finite-record coloured Fourier coefficients.
    """
    lengths = plan.smearing_grid_final.reshape(-1, plan.nchans)[trial].astype(int) + 1
    weights = np.asarray(plan.box_weights, dtype=float) * 2.0**-plan.box_fraction_bits
    covariance = np.zeros(samples)
    for length in lengths:
        stop = min(int(length), samples)
        covariance[:stop] += weights[length - 1] ** 2 * (length - np.arange(stop))
    freq = np.fft.rfftfreq(samples)
    result = np.full(freq.size, covariance[0])
    for lag in np.flatnonzero(covariance[1:]) + 1:
        result += (
            2 * covariance[lag] * (1 - lag / samples) * np.cos(2 * np.pi * freq * lag)
        )
    return result / plan.nchans


def analyze(
    streams: dict[str, np.ndarray],
    plan: libdmt.FDMTPlan,
    *,
    alpha: float = 1e-6,
) -> Report:
    """Check white engines; retain box filtering as an expected coloured control."""
    comparisons = len(WHITE_ENGINES) * len(plan.dt_grid_final)
    report = {}
    for name, data in streams.items():
        report[name] = [
            white_noise_metrics(
                data[:, trial], plan.nchans, alpha=alpha, comparisons=comparisons
            )
            for trial in range(data.shape[1])
        ]
        if name == "fdmt_box":
            for trial, row in enumerate(report[name]):
                row["expected_spectrum"] = box_spectrum(
                    plan, trial, data.shape[-1]
                ).tolist()
    return report


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
    """One cold-start FDMT block versus the same data with its startup dropped.

    Input is Gaussian noise plus ``mean`` (real filterbank data is not zero
    mean). Each per-realization series has its own mean removed, as a power
    spectrum pipeline would. ``fdmt_cold`` keeps the first
    ``startup_samples`` outputs, partial sums over zero-filled history, so its
    low Fourier bins carry a deterministic ramp. ``fdmt_trimmed`` drops them
    and ``ddmt`` (which withholds unsupported samples itself) is the
    reference. Only the latter two must pass the white-noise checks.
    """
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


def plot_analysis(
    report: Report,
    plan: libdmt.FDMTPlan,
    destination: Path | None = None,
) -> Figure:
    """Plot the highest trial; matplotlib is an optional script dependency."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 - optional plotting dependency

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name, rows in report.items():
        row = rows[-1]
        freq = np.fft.rfftfreq(row["samples"])
        axes[0].semilogy(freq[1:-1], row["spectral_mean"][1:-1], label=name)
        axes[1].plot(
            [int(k) for k in row["acf"]], list(row["acf"].values()), "o-", label=name
        )
        axes[2].plot(
            [int(k) for k in row["cdf_distance"]],
            list(row["cdf_distance"].values()),
            "o-",
            label=name,
        )
    control = report["fdmt_box"][-1]
    axes[0].semilogy(
        freq[1:-1], control["expected_spectrum"][1:-1], "k--", label="box model"
    )
    axes[0].axhline(1, color="grey", linestyle=":")
    axes[1].axhline(0, color="grey", linestyle=":")
    axes[2].axhline(
        control["cdf_bound"], color="grey", linestyle=":", label="DKW bound"
    )
    axes[0].set(xlabel="Cycles / sample", ylabel="Mean power / (N * nchans)")
    axes[1].set(xlabel="Lag (samples)", ylabel="Autocovariance / nchans")
    axes[2].set(xlabel="Fourier bin", ylabel="CDF distance from Exp(1)", xscale="log")
    for ax in axes:
        ax.legend(fontsize="small")
    fig.suptitle(
        f"Independent Gaussian noise, effective DM {plan.dm_grid_final[-1]:.3f}"
    )
    if destination is not None:
        fig.savefig(destination, dpi=150)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fdmt-noise"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda", "hip"), default="cpu")
    parser.add_argument("--nchans", type=int, default=32)
    parser.add_argument(
        "--nsamps", type=int, default=2048, help="Samples per execute call"
    )
    parser.add_argument("--realizations", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 128, 512])
    parser.add_argument("--f-min", type=float, default=400.0)
    parser.add_argument("--f-max", type=float, default=800.0)
    parser.add_argument("--tsamp", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument(
        "--cold-start-mean",
        type=float,
        default=None,
        help="Also run the cold-start experiment with this input mean",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-6,
        help="Family-wise error budget for the white-engine checks",
    )
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must lie strictly between 0 and 1")
    streams, plan, metadata = collect_noise(
        nchans=args.nchans,
        nsamps=args.nsamps,
        realizations=args.realizations,
        dt_grid=args.delays,
        f_min=args.f_min,
        f_max=args.f_max,
        tsamp=args.tsamp,
        seed=args.seed,
        backend=args.backend,
        blocks=args.blocks,
    )
    report = analyze(streams, plan, alpha=args.alpha)
    payload = {"configuration": metadata, "alpha": args.alpha, "results": report}
    cold_ok = True
    if args.cold_start_mean is not None:
        cold = cold_start_noise(
            mean=args.cold_start_mean,
            f_min=args.f_min,
            f_max=args.f_max,
            tsamp=args.tsamp,
            seed=args.seed,
            backend=args.backend,
            alpha=args.alpha,
        )
        payload["cold_start"] = cold
        for name in ("fdmt_cold", "fdmt_trimmed", "ddmt"):
            low = np.round(cold[name]["spectral_mean"][1:5], 2).tolist()
            print(f"cold start {name}: white={cold[name]['passed']}, bins 1-4 {low}")
        cold_ok = cold["fdmt_trimmed"]["passed"] and cold["ddmt"]["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.with_suffix(".json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    for name, rows in report.items():
        print(f"{name}: white-noise checks {[r['passed'] for r in rows]}")
    print(f"Wrote {path}; fdmt_box is a coloured control, not a required white result.")
    if args.plot:
        plot_analysis(report, plan, args.output.with_suffix(".png"))
    white = all(row["passed"] for name in WHITE_ENGINES for row in report[name])
    if not (white and cold_ok):
        message = "White-noise validation failed; inspect the saved diagnostics."
        raise SystemExit(message)


if __name__ == "__main__":
    main()
