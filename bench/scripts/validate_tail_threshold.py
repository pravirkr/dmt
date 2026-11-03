"""Validate the up-crossing threshold fit on long synthetic search statistics.

The sensitivity study (``run_sensitivity.py``) extrapolates each engine's
false-alarm threshold from up-crossings of the best-boxcar S/N at 2-4.5 sigma
in 0.7-1.0e5 noise samples per DM (``dmtlib.sensitivity.tail_fit``). This
script checks that extrapolation against truth: it synthesizes a long
stationary Gaussian output series with the autocorrelation of

* ``white``: a direct engine (DDMT, SDMT, DDMT-FFT), white output noise;
* ``fdmt``: the FDMT with box means at ``--dm``, i.e. the power spectrum
  ``sum_c w_c^2 |B_{L_c}(nu)|^2`` with the plan's level-0 box lengths
  ``L_c`` and weights ``w_c`` for the frozen configuration
  (``dmtlib.sensitivity.CRYOPAF``);

searches it with the study's boxcar ladder, normalized by the analytic noise
of each width, and compares

* a fit of the long run over 3-6.5 sigma (the truth, extrapolated a little),
* fits over 2-4.5 sigma of 8.5e4-sample segments, as in the study,

at the study's rate of one false alarm per hour over all trials.

Usage::

    python bench/scripts/validate_tail_threshold.py                 # both
    python bench/scripts/validate_tail_threshold.py --kind fdmt --chunks 96
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]
sys.path.insert(0, str(REPO / "src"))

from dmtlib import libdmt  # noqa: E402
from dmtlib import sensitivity as sens  # noqa: E402

CFG = sens.CRYOPAF
SEGMENT = 85_000  # noise samples per DM in the study (n_phase 32)
CHUNK = 1 << 22


def fdmt_channels(dm: float) -> tuple[np.ndarray, np.ndarray]:
    """Level-0 box lengths and box-mean weights of the FDMT trial nearest ``dm``."""
    trials = CFG.dm_trials.astype(np.float64)
    i = int(np.argmin(np.abs(trials - dm)))
    pair = np.asarray(sorted({trials[i], trials[i - 1]}), dtype=np.float32)
    band = (CFG.f_min, CFG.f_max, CFG.nchans)
    plan = libdmt.FDMTPlan(*band, 4096, CFG.tsamp, dm_grid=pair)
    smear = np.asarray(plan.smearing_grid_final).reshape(len(pair), -1)
    length = np.abs(smear[list(pair).index(np.float32(trials[i]))]) + 1.0
    return length, sens.plan_box_weights(plan, length)


def spectrum(kind: str, dm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amplitude spectrum of unit-variance output noise, and its channel model."""
    nu = np.fft.rfftfreq(CHUNK)
    if kind == "white":
        return np.ones_like(nu), np.zeros(0), np.zeros(0)
    length, weight = fdmt_channels(dm)
    psd = np.zeros_like(nu)
    s = np.sin(np.pi * nu)
    for ell in np.unique(length):
        sel = length == ell
        with np.errstate(invalid="ignore", divide="ignore"):
            box = np.where(nu == 0, ell, np.abs(np.sin(np.pi * nu * ell) / s))
        psd += np.sum(weight[sel] ** 2) * box**2
    norm = np.sum(weight**2 * length)  # variance of one output sample
    return np.sqrt(psd / norm), length, weight


def run(kind: str, chunks: int, dm: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    widths = sens.boxcar_widths(
        2.0 * (max(sens.WIDTHS) + CFG.bottom_smear(CFG.dm_max)) + 4.0
    )
    amp, length, weight = spectrum(kind, dm)
    if kind == "white":
        sigma = np.sqrt(widths.astype(np.float64))
    else:
        var = sens.box_smearing_variance(length[None, :], widths, weight[None, :])
        sigma = np.sqrt(var[0] / np.sum(weight**2 * length))
    levels = np.arange(2.0, 6.51, 0.25)
    counts = np.zeros(levels.size)
    total = 0
    short = []
    t0 = time.perf_counter()
    for _ in range(chunks):
        z = rng.standard_normal(amp.size) + 1j * rng.standard_normal(amp.size)
        x = np.fft.irfft(z * amp, n=CHUNK) * math.sqrt(CHUNK / 2.0)
        c = np.concatenate([[0.0], np.cumsum(x)])
        n = CHUNK - int(widths[-1])
        best = np.full(n, -np.inf)
        for w, sw in zip(widths, sigma, strict=True):
            np.maximum(best, (c[w : w + n] - c[:n]) / sw, out=best)
        cnt, tot = sens.upcrossings([best], levels)
        counts += cnt
        total += tot
        short.extend(
            sens.upcrossings([best[a : a + SEGMENT]], sens.TAIL_LEVELS)
            for a in range(0, n - SEGMENT, 4 * SEGMENT)
        )
    rate = CFG.tsamp / (3600.0 * len(CFG.dm_trials))
    label = kind if kind == "white" else f"fdmt DM {dm:g}"
    print(f"{label}: {total:.3g} samples, {time.perf_counter() - t0:.0f} s")
    for u, cn in zip(levels, counts, strict=True):
        print(f"  u {u:4.2f}  up-crossings {int(cn):9d}  rate {cn / total:.3e}")
    hi = levels >= 3.0
    c_l, b_l = sens.tail_fit(levels[hi], counts[hi], total)
    print(
        f"  long run, fit 3-6.5 sigma: beta {b_l:+.2f}, "
        f"threshold {sens.tail_threshold(c_l, b_l, rate):.3f}"
    )
    fits = []
    for sc, st in short:
        try:
            c_s, b_s = sens.tail_fit(sens.TAIL_LEVELS, sc, st)
        except ValueError:
            continue
        fits.append((sens.tail_threshold(c_s, b_s, rate), b_s))
    th = np.asarray(fits)
    p16, p50, p84 = np.percentile(th[:, 0], [16, 50, 84])
    print(
        f"  {len(th)} segments of {SEGMENT}, fit 2-4.5 sigma: threshold "
        f"{p50:.3f} (16-84%: {p16:.3f}-{p84:.3f}), beta {np.median(th[:, 1]):+.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kind", choices=["white", "fdmt", "both"], default="both")
    ap.add_argument("--chunks", type=int, default=48, help="chunks of 2^22 samples")
    ap.add_argument("--dm", type=float, default=1966.3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    kinds = ["white", "fdmt"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        chunks = args.chunks if kind == "white" else 2 * args.chunks
        run(kind, chunks, args.dm, args.seed)


if __name__ == "__main__":
    main()
