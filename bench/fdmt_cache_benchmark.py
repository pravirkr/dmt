"""FDMT cache reuse & memory locality benchmark.

Analyzes:
1. Exact Head and Tail row reuse ratios across each iteration level.
2. Distribution of coordinates per unique head row.
3. Execution throughput (M elements/sec) as a function of time block size
   N_samps (testing L1/L2/L3 cache boundary effects).
4. Input-row DRAM traffic model: rows read per output by the merge loop
   (which reuses the previous coordinate's operands from cache) vs an ideal
   intra-level blocking that reads each distinct operand row of a chunk of
   coordinates once. This bounds what any intra-level cache tiling can save;
   it is ~1%, which is why FDMT has no tiled schedule (see the performance
   page of the pipeline guide).
5. Level-fusion depth sweep (block size x threads x ``fuse_levels``), and
   packed low-bit input (nbits x ``int_tree``) vs float input.

Usage:
    python bench/fdmt_cache_benchmark.py                 # everything
    python bench/fdmt_cache_benchmark.py --section fusion \
        --nchans 4096 --dt-max 2048 --nsamps 16384 65536 --threads 1 8
    python bench/fdmt_cache_benchmark.py --section packed --nsamps 16384
"""

from __future__ import annotations

import argparse
import statistics
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


def analyze_merge_traffic(
    plan: FDMTPlan, chunk_sizes: tuple[int, ...] = (1, 2, 4, 8, 32)
) -> dict:
    """Input rows read from memory per output row, summed over all levels.

    The merge loop (``coord``) is assumed to keep only the previous
    coordinate's two operand rows cached (true whenever ~4 rows fit in the
    per-core cache); an ideal blocking of ``ndt`` consecutive coordinates
    reads each distinct operand row of the chunk once. Each output also costs
    one write (+ one read-for-ownership), so total row traffic is
    ``reads + 2``.
    """
    container = plan.container
    outputs = 0
    coord_reads = 0.0
    chunk_reads = dict.fromkeys(chunk_sizes, 0.0)
    for level in range(1, plan.niters + 1):
        cs = container.coordinates_sum[level]
        ncopy = len(container.coordinates_copy[level])
        outputs += len(cs) + ncopy
        coord_reads += ncopy
        for ndt in chunk_sizes:
            chunk_reads[ndt] += ncopy
        if len(cs) == 0:
            continue
        tails = cs["tail_buf_offset"]
        heads = cs["head_buf_offset"]
        subs = cs["i_sub"]
        prev_t, prev_h = tails[:-1], heads[:-1]
        cur_t, cur_h = tails[1:], heads[1:]
        hit_t = (cur_t == prev_t) | (cur_t == prev_h)
        hit_h = (cur_h == prev_t) | (cur_h == prev_h)
        coord_reads += 2 + np.sum(2 - hit_t.astype(int) - hit_h.astype(int))
        for ndt in chunk_sizes:
            begin = 0
            for k in range(1, len(cs) + 1):
                if k == len(cs) or subs[k] != subs[begin] or k - begin == ndt:
                    uniq = set(tails[begin:k].tolist()) | set(heads[begin:k].tolist())
                    chunk_reads[ndt] += len(uniq)
                    begin = k
    return {
        "outputs": outputs,
        "coord": coord_reads / outputs,
        "chunks": {ndt: r / outputs for ndt, r in chunk_reads.items()},
    }


def _median_time(fn, reps: int) -> float:
    fn()  # warm-up (first-touch page faults, thread pool start)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def benchmark_fusion(
    nchans: int,
    dt_max: int,
    block_sizes: list[int],
    threads: list[int],
    mode: str = "valid",
    smearing: bool = True,
    reps: int = 5,
) -> None:
    """Median wall time per ``fuse_levels`` depth; output is bit-identical."""
    depths = [0, 1, 2, 3, 4, 5, 6, None]  # None = automatic (default)
    print(
        f"\nFusion sweep (nchans={nchans}, dt_max={dt_max}, mode={mode}, "
        f"smearing={smearing}, median of {reps}; speedup vs unfused)"
    )
    names = [("auto" if d is None else f"F{d}") for d in depths]
    header = f"{'N_samps':>9} | {'thr':>3} | " + " | ".join(
        f"{name:>15}" for name in names
    )
    print(header)
    print("-" * len(header))
    rng = np.random.default_rng(0)
    for nsamps in block_sizes:
        wf = rng.random((nchans, nsamps), dtype=np.float32)
        for nthreads in threads:
            cells = []
            base = None
            for depth in depths:
                fdmt = FDMTCPU(
                    1000.0, 1500.0, nchans, nsamps, 0.001, dt_max,
                    use_box_smearing=smearing, mode=mode, nthreads=nthreads,
                    fuse_levels=depth,
                )
                t = _median_time(lambda: fdmt.execute(wf), reps)
                base = base or t
                tag = f"({fdmt.fuse_levels})" if depth is None else ""
                cells.append(f"{t * 1e3:7.1f}ms {base / t:4.2f}x{tag}")
            print(f"{nsamps:9d} | {nthreads:3d} | " + " | ".join(cells))


def benchmark_packed(
    nchans: int,
    dt_max: int,
    nsamps: int,
    threads: list[int],
    mode: str = "valid",
    smearing: bool = True,
    reps: int = 5,
) -> None:
    """Packed low-bit input (``int_tree`` off/on) vs float input."""
    print(
        f"\nPacked input (nchans={nchans}, dt_max={dt_max}, nsamps={nsamps}, "
        f"mode={mode}, smearing={smearing}, median of {reps}; speedup vs float)"
    )
    rng = np.random.default_rng(0)
    for nthreads in threads:
        engines = {
            int_tree: FDMTCPU(
                1000.0, 1500.0, nchans, nsamps, 0.001, dt_max,
                use_box_smearing=smearing, mode=mode, nthreads=nthreads,
                int_tree=int_tree,
            )
            for int_tree in (False, True)
        }
        fdmt = engines[True]
        wf = rng.random((nchans, nsamps), dtype=np.float32)
        t_float = _median_time(lambda: fdmt.execute(wf), reps)
        row = [f"float {t_float * 1e3:7.1f}ms"]
        for nbits in (1, 2, 4, 8, 16):
            packed = rng.integers(
                0, 256, size=(nchans, (nsamps * nbits + 7) // 8), dtype=np.uint8
            )
            for int_tree, engine in engines.items():
                t = _median_time(lambda: engine.execute(packed, nbits), reps)
                tag = f"{nbits}b{'+int' if int_tree else ''}"
                row.append(f"{tag} {t_float / t:4.2f}x")
        print(f"threads={nthreads:2d}: " + "  ".join(row))


def print_reuse_table(nchans: int, dt_max: int) -> None:
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

    traffic = analyze_merge_traffic(plan)
    chunks = "  ".join(
        f"chunk{ndt}={r:.3f}" for ndt, r in traffic["chunks"].items()
    )
    print(
        f"\nInput rows read per output row: merge loop={traffic['coord']:.3f}  "
        f"ideal blocking: {chunks}"
    )
    best = min(traffic["chunks"].values())
    gain = 1.0 - (best + 2.0) / (traffic["coord"] + 2.0)
    print(
        f"Max traffic saving of intra-level blocking while ~4 rows fit in "
        f"cache: {gain * 100:.1f}% (the merge loop loses its reuse, reading 2 "
        f"rows/output, only once 4 rows exceed the per-core cache)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--section",
        choices=["all", "reuse", "blocksize", "fusion", "packed"],
        default="all",
    )
    parser.add_argument("--nchans", type=int, default=None)
    parser.add_argument("--dt-max", type=int, default=None)
    parser.add_argument("--nsamps", type=int, nargs="+", default=None)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--mode", default=None)
    parser.add_argument("--smearing", type=int, choices=[0, 1], default=None)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    run = lambda name: args.section in ("all", name)  # noqa: E731

    if run("reuse"):
        print_reuse_table(args.nchans or 256, args.dt_max or 256)
    if run("blocksize"):
        benchmark_throughput_vs_blocksize(
            nchans=args.nchans or 256, dt_max=args.dt_max or 256,
            block_sizes=args.nsamps,
        )
    if run("fusion"):
        benchmark_fusion(
            nchans=args.nchans or 1024,
            dt_max=args.dt_max or 512,
            block_sizes=args.nsamps or [4096, 16384, 65536],
            threads=args.threads,
            mode=args.mode or "valid",
            smearing=bool(args.smearing) if args.smearing is not None else True,
            reps=args.reps,
        )
    if run("packed"):
        benchmark_packed(
            nchans=args.nchans or 1024,
            dt_max=args.dt_max or 512,
            nsamps=(args.nsamps or [16384])[0],
            threads=args.threads,
            mode=args.mode or "valid",
            smearing=bool(args.smearing) if args.smearing is not None else True,
            reps=args.reps,
        )


if __name__ == "__main__":
    main()
