"""FDMT cache reuse & memory locality benchmark.

Analyzes:
1. Exact Head and Tail row reuse ratios across each iteration level.
2. Distribution of coordinates per unique head row.
3. Execution throughput (M elements/sec) as a function of time block size
   N_samps (testing L1/L2/L3 cache boundary effects).

Usage:
    python bench/fdmt_cache_benchmark.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Ensure local build/src takes precedence
repo_root = Path(__file__).resolve().parents[1]
build_src = repo_root / "build" / "src"
src_dir = repo_root / "src"

sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]

for p in (build_src, src_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dmtlib import FDMTCPU, FDMTPlan


def analyze_plan_reuse(plan: FDMTPlan) -> dict:
    """Analyze exact head and tail buffer reuse from the compiled C++ plan."""
    container = plan.container
    niters = plan.niters

    total_coords = 0
    head_reuses = 0
    tail_reuses = 0

    iter_stats = []

    for l in range(1, niters + 1):
        coords_sum = container.coordinates_sum[l]
        n_sum = len(coords_sum)
        if n_sum == 0:
            continue

        head_offsets = coords_sum["head_buf_offset"]
        tail_offsets = coords_sum["tail_buf_offset"]

        unique_heads = len(set(head_offsets))
        unique_tails = len(set(tail_offsets))

        head_counts = Counter(head_offsets)
        max_head_fanout = max(head_counts.values()) if head_counts else 1
        avg_head_fanout = n_sum / unique_heads if unique_heads else 1.0

        # Number of calls that reuse an already-loaded head row
        h_reuse = n_sum - unique_heads
        t_reuse = n_sum - unique_tails

        total_coords += n_sum
        head_reuses += h_reuse
        tail_reuses += t_reuse

        iter_stats.append({
            "level": l,
            "n_coords": n_sum,
            "unique_heads": unique_heads,
            "unique_tails": unique_tails,
            "head_reuse_pct": (h_reuse / n_sum) * 100.0,
            "tail_reuse_pct": (t_reuse / n_sum) * 100.0,
            "max_head_fanout": max_head_fanout,
            "avg_head_fanout": avg_head_fanout,
        })

    return {
        "total_coords": total_coords,
        "overall_head_reuse_pct": (head_reuses / total_coords) * 100.0 if total_coords else 0.0,
        "overall_tail_reuse_pct": (tail_reuses / total_coords) * 100.0 if total_coords else 0.0,
        "iter_stats": iter_stats,
    }


def benchmark_throughput_vs_blocksize(
    nchans: int = 256,
    dt_max: int = 256,
    block_sizes: list[int] | None = None,
) -> None:
    if block_sizes is None:
        block_sizes = [1024, 4096, 16384, 65536, 262144]

    f_min, f_max, tsamp = 1000.0, 1500.0, 0.001

    print("\nThroughput Benchmark vs. Time Block Size (Cache Boundary Analysis)")
    print("=" * 85)
    print(
        f"{'N_samps':>9} | {'Row Size':>10} | {'3 Rows (L1)':>12} | "
        f"{'Exec Time':>12} | {'Throughput':>16} | {'Effective BW':>14}"
    )
    print("-" * 85)

    for nsamps in block_sizes:
        fdmt = FDMTCPU(f_min, f_max, nchans, nsamps, tsamp, dt_max=dt_max)
        row_kb = (nsamps * 4) / 1024.0
        three_rows_kb = 3 * row_kb

        # Warmup
        wf = np.random.randn(nchans, nsamps).astype(np.float32)
        _ = fdmt.execute(wf)

        # Timed runs
        n_runs = max(3, int(200000 / nsamps))
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = fdmt.execute(wf)
        t1 = time.perf_counter()

        elapsed = (t1 - t0) / n_runs
        total_elems = nchans * nsamps
        throughput_m = (total_elems / elapsed) / 1e6

        # Approximate memory bandwidth (total plan additions * 3 memory ops * 4 bytes)
        total_ops = fdmt.plan.total_operations
        mem_bytes = total_ops * nsamps * 3 * 4
        bw_gbps = (mem_bytes / elapsed) / 1e9

        print(
            f"{nsamps:9d} | {row_kb:8.1f} KB | {three_rows_kb:10.1f} KB | "
            f"{elapsed*1000:9.2f} ms | {throughput_m:11.1f} MSamp/s | {bw_gbps:10.2f} GB/s"
        )
    print("=" * 85)


def main() -> None:
    nchans = 256
    dt_max = 256
    plan = FDMTPlan(1000.0, 1500.0, nchans, 1024, 0.001, dt_max=dt_max)

    reuse = analyze_plan_reuse(plan)
    print(f"\nFDMT Plan Buffer Re-Use Analysis (N_chans={nchans}, dt_max={dt_max})")
    print("=" * 80)
    print(
        f"{'Level':>6} | {'Coords':>8} | {'Uniq Heads':>11} | "
        f"{'Head Reuse':>11} | {'Tail Reuse':>11} | {'Avg Fanout':>11} | {'Max Fanout':>11}"
    )
    print("-" * 80)
    for s in reuse["iter_stats"]:
        print(
            f"{s['level']:6d} | {s['n_coords']:8d} | {s['unique_heads']:11d} | "
            f"{s['head_reuse_pct']:10.1f}% | {s['tail_reuse_pct']:10.1f}% | "
            f"{s['avg_head_fanout']:11.2f} | {s['max_head_fanout']:11d}"
        )
    print("-" * 80)
    print(
        f"OVERALL: Total Coords={reuse['total_coords']}, "
        f"Head Reuse={reuse['overall_head_reuse_pct']:.1f}%, "
        f"Tail Reuse={reuse['overall_tail_reuse_pct']:.1f}%"
    )
    print("=" * 80)

    benchmark_throughput_vs_blocksize(nchans=nchans, dt_max=dt_max)


if __name__ == "__main__":
    main()
