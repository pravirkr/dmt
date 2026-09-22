"""FDMT sensitivity characterization: S/N-recovery fraction vs. true
dispersive delay and sub-sample pulse-arrival phase.

Not part of CI -- re-run this whenever `dt_step`, `use_box_smearing`, or the
DM grid law changes, to visually characterize the resulting trade-off.

Usage:
    python bench/fdmt_sensitivity.py [--plot] [--out fdmt_sensitivity.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the local build/src and src directories take precedence over stale editable site-packages
repo_root = Path(__file__).resolve().parents[1]
build_src = repo_root / "build" / "src"
src_dir = repo_root / "src"
tests_python = repo_root / "tests" / "python"

sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]

for p in (build_src, src_dir, tests_python):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from _dispersed_pulse import sweep_recovery  # noqa: E402

from dmtlib import libdmt


def run_sweep(
    nchans: int = 64,
    dt_max: int = 64,
    n_dt: int = 200,
    n_phase: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f_min, f_max, tsamp = 1000.0, 1500.0, 1.0
    nsamps = 4 * dt_max
    fdmt = libdmt.FDMTCPU(f_min, f_max, nchans, nsamps, tsamp, dt_max)

    dt_values = np.linspace(1.0, dt_max - 1.0, n_dt)
    phases = np.linspace(0.0, 0.9, n_phase)

    def sigma_fn(idm: int) -> float:
        return float(fdmt.get_effective_sigma_grid(1)[idm])

    fractions = sweep_recovery(
        fdmt.execute,
        dt_values,
        phases,
        nchans,
        nsamps,
        f_min,
        f_max,
        sigma_fn=sigma_fn,
    )
    return dt_values, phases, fractions


def plot_sweep(
    dt_values: np.ndarray, phases: np.ndarray, fractions: np.ndarray, out_path: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        fractions.T,
        origin="lower",
        aspect="auto",
        extent=(dt_values[0], dt_values[-1], phases[0], phases[-1]),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xlabel("True dispersive delay across band (samples)")
    ax.set_ylabel("Sub-sample arrival phase")
    ax.set_title("FDMT S/N-recovery fraction")
    fig.colorbar(im, ax=ax, label="Recovery fraction")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Render the 2-D heatmap")
    parser.add_argument("--out", default="fdmt_sensitivity.png")
    args = parser.parse_args()

    dt_values, phases, fractions = run_sweep()
    print(
        f"Recovery fraction: min={fractions.min():.3f} "
        f"median={np.median(fractions):.3f} max={fractions.max():.3f}"
    )

    if args.plot:
        plot_sweep(dt_values, phases, fractions, args.out)


if __name__ == "__main__":
    main()
