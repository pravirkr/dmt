"""FDMT operations cost analysis & scaling benchmark.

Compares:
1. Brute-force dedispersion: O(N_dt * N_chans)
2. Dense FDMT: O(N_chans * log2(N_chans) + N_dt)
3. S/N-Loss-Bounded Sparse FDMT (dmtlib.grid): optimal non-uniform spacing

Usage:
    python bench/fdmt_operations_costs.py [--plot] [--out fdmt_operations_costs.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure the local build/src and src directories take precedence
repo_root = Path(__file__).resolve().parents[1]
build_src = repo_root / "build" / "src"
src_dir = repo_root / "src"

sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]

for p in (build_src, src_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dmtlib import FDMTPlan, generate_optimal_dt_grid


def run_cost_analysis(
    nchans: int = 256,
    f_min: float = 800.0,
    f_max: float = 1056.0,
    tsamp: float = 0.001,
    dt_values: list[int] | None = None,
) -> dict:
    if dt_values is None:
        dt_values = [16, 32, 64, 128, 256, 512, 1024, 2048]

    results = {
        "dt_values": dt_values,
        "brute_force_ops": [],
        "dense_fdmt_ops": [],
        "sparse_fdmt_ops": [],
        "dense_speedup": [],
        "sparse_speedup": [],
        "iterations_breakdown": {},
    }

    print(f"\nFDMT Operational Cost Analysis (N_chans={nchans}, Band={f_min}-{f_max} MHz)")
    print("=" * 80)
    print(
        f"{'N_dt':>7} | {'Brute Force':>12} | {'Dense FDMT':>12} | "
        f"{'Sparse FDMT':>12} | {'Dense Spdup':>11} | {'Sparse Spdup':>12}"
    )
    print("-" * 80)

    for dt_max in dt_values:
        # 1. Dense FDMT Plan
        dense_plan = FDMTPlan(f_min, f_max, nchans, 1024, tsamp, dt_max=dt_max)
        bf_ops = dense_plan.complexity.brute_force_ops
        dense_ops = dense_plan.total_operations

        # 2. Sparse Grid FDMT Plan (bounded to 5% max S/N loss)
        sparse_dt = generate_optimal_dt_grid(
            f_min, f_max, nchans, tsamp, dt_max=dt_max, max_snr_loss=0.05
        )
        sparse_plan = FDMTPlan(f_min, f_max, nchans, 1024, tsamp, dt_arr=sparse_dt)
        sparse_ops = sparse_plan.total_operations

        dense_sp = bf_ops / dense_ops if dense_ops > 0 else 1.0
        sparse_sp = bf_ops / sparse_ops if sparse_ops > 0 else 1.0

        results["brute_force_ops"].append(bf_ops)
        results["dense_fdmt_ops"].append(dense_ops)
        results["sparse_fdmt_ops"].append(sparse_ops)
        results["dense_speedup"].append(dense_sp)
        results["sparse_speedup"].append(sparse_sp)

        # Store iteration breakdown for largest dt
        if dt_max == dt_values[-1]:
            results["iterations_breakdown"]["dense"] = dense_plan.operations_by_iteration
            results["iterations_breakdown"]["sparse"] = sparse_plan.operations_by_iteration

        print(
            f"{dt_max:7d} | {bf_ops:12d} | {dense_ops:12d} | "
            f"{sparse_ops:12d} | {dense_sp:10.2f}x | {sparse_sp:11.2f}x"
        )

    print("=" * 80)
    return results


def plot_costs(results: dict, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    dt_values = results["dt_values"]

    # Plot 1: Operations Scaling (log-log)
    ax1.plot(dt_values, results["brute_force_ops"], "r--o", label="Brute Force", linewidth=1.5)
    ax1.plot(dt_values, results["dense_fdmt_ops"], "b-s", label="Dense FDMT", linewidth=2)
    ax1.plot(
        dt_values,
        results["sparse_fdmt_ops"],
        "g-^",
        label="Sparse FDMT (5% S/N loss bound)",
        linewidth=2,
    )
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of DM Trials ($N_{dt}$)")
    ax1.set_ylabel("Operations per Time Sample")
    ax1.set_title("Computational Scaling: Brute Force vs. FDMT")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(frameon=True)

    # Plot 2: Operations by Iteration Level
    dense_iters = results["iterations_breakdown"]["dense"]
    sparse_iters = results["iterations_breakdown"]["sparse"]
    n_iters = len(dense_iters)
    x = np.arange(n_iters)
    width = 0.35

    ax2.bar(x - width / 2, dense_iters, width, label="Dense FDMT", color="royalblue")
    ax2.bar(
        x + width / 2,
        sparse_iters,
        width,
        label="Sparse FDMT (5% S/N loss)",
        color="forestgreen",
    )
    ax2.set_xlabel("Iteration Level ($0 = $ level 0 init, $M = $ root)")
    ax2.set_ylabel("Additions per Time Sample")
    ax2.set_title(f"Operations per Iteration Level ($N_{{dt}}={dt_values[-1]}$)")
    ax2.set_xticks(x)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Render operations scaling plot")
    parser.add_argument("--out", default="fdmt_operations_costs.png", help="Output plot filename")
    parser.add_argument("--nchans", type=int, default=256, help="Number of channels")
    args = parser.parse_args()

    results = run_cost_analysis(nchans=args.nchans)
    if args.plot:
        plot_costs(results, args.out)


if __name__ == "__main__":
    main()
