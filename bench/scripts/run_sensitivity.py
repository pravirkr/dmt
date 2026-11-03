"""Measure the S/N recovery of the incoherent dedispersion engines.

For a set of trial DMs of the search grid, injects exactly smeared dispersed
pulses (flat spectrum, boxcar profiles of 1, 8 and 64 samples, random
sub-sample phase) at exactly that DM, runs FDMT, FDMT-FFT, DDMT, SDMT and
DDMT-FFT (plus FDMT without box smearing) with that single trial on identical
data, and writes

    bench/results/sensitivity/cryopaf.npz   (arrays)
    bench/results/sensitivity/cryopaf.json  (metadata)

for ``dmtlib.sensitivity.CRYOPAF`` (512 channels over 704-1216 MHz at
512 us, DM 0-2000, Levin tolerance 1.1). ``--config cryopaf_lband`` and
``--config parkes_uwl`` select the other presets and write their named
``{npz,json}`` outputs.

which plot_sensitivity.py turns into figures. A unit-variance noise
realisation, the same for every engine, calibrates each engine's threshold at
a fixed false-alarm rate and gives completeness curves. See
``dmtlib.sensitivity`` for the model. The results are deterministic (fixed
seeds), so they are not kept per machine.

    python bench/scripts/run_sensitivity.py
    python bench/scripts/run_sensitivity.py --n-dm 8 --out /tmp/check   # quick
    python bench/scripts/run_sensitivity.py --config parkes_uwl
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import platform
import re
import subprocess
import sys
import time
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# Prefer a compatible in-tree build. Otherwise retain the editable loader,
# which locates the extension built for this Python interpreter.
if any(
    (REPO / "src/dmtlib" / f"libdmt{suffix}").is_file() for suffix in EXTENSION_SUFFIXES
):
    sys.meta_path = [
        f
        for f in sys.meta_path
        if "ScikitBuildRedirectingFinder" not in type(f).__name__
    ]
sys.path.insert(0, str(REPO / "src"))

import dmtlib  # noqa: E402
from dmtlib import sensitivity as sens  # noqa: E402


def repo_version() -> str:
    """Return the version in CMakeLists.txt (an installed dmtlib may be stale)."""
    match = re.search(
        r"project\(\s*dmt\s+VERSION\s+([0-9.]+)",
        (REPO / "CMakeLists.txt").read_text(),
    )
    return match.group(1) if match else dmtlib.__version__


def git_sha() -> str:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-dm", type=int, default=48, help="DM strata")
    ap.add_argument("--n-phase", type=int, default=32)
    ap.add_argument("--widths", type=float, nargs="+", default=list(sens.WIDTHS))
    ap.add_argument("--far", type=float, default=1.0, help="false alarms per hour")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--engines",
        nargs="+",
        default=[*sens.ENGINES, *sens.VARIANTS],
        choices=[*sens.ENGINES, *sens.VARIANTS],
    )
    ap.add_argument(
        "--config",
        default="cryopaf",
        choices=sorted(sens.CONFIGS),
        help="search configuration (default cryopaf, the published one)",
    )
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPO / "bench/results/sensitivity")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dataclasses.replace(sens.CONFIGS[args.config], nthreads=args.nthreads)
    dms = sens.select_trials(cfg, args.n_dm, seed=args.seed)
    print(
        f"{cfg.name}: {cfg.nchans} ch, {cfg.f_min:g}-{cfg.f_max:g} MHz, "
        f"{cfg.tsamp * 1e6:g} us, {len(cfg.dm_trials)} piecewise Levin trials "
        f"(tol {cfg.levin_tol}) to DM {cfg.dm_max:g}; {len(dms)} DMs x "
        f"{len(args.widths)} widths x {args.n_phase} phases; bottom-channel smear "
        f"<= {cfg.bottom_smear(cfg.dm_max):.0f} samples",
        flush=True,
    )
    start = time.perf_counter()

    def progress(done: int, total: int, dm: float) -> None:
        el = time.perf_counter() - start
        print(f"  {done}/{total} DM {dm:8.2f}, {el:.0f} s", flush=True)

    res = sens.measure_recovery(
        cfg, dms, args.widths, args.engines, n_phase=args.n_phase,
        seed=args.seed, far_per_hour=args.far, progress=progress,
    )  # fmt: skip
    elapsed = time.perf_counter() - start
    res.meta.update(
        {
            "dmt_version": repo_version(),
            "git_sha": git_sha(),
            "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "machine": platform.platform(),
            "elapsed_s": round(elapsed, 1),
        }
    )
    path = res.save(args.out / cfg.name)
    print(f"wrote {path} ({elapsed:.0f} s)")
    for name in res.engines:
        cols = res.engines[name]
        print(
            f"  {name:15s} eta_mf median {np.median(res.eta_mf(name)):.3f} "
            f"min {np.min(res.eta_mf(name)):.3f} | sigma ratio "
            f"{np.min(cols['sigma_ratio']):.3f}-{np.max(cols['sigma_ratio']):.3f} | "
            f"threshold {np.min(cols['threshold']):.2f}-{np.max(cols['threshold']):.2f}"
            f" (median-n_eff {np.min(cols['threshold_median']):.2f}-"
            f"{np.max(cols['threshold_median']):.2f})"
        )


if __name__ == "__main__":
    main()
