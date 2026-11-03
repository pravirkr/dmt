"""Time and memory of chunked FDMT execution (``chunk_nsamps``).

Streams the same block through the FDMT for several chunk lengths and prints
the time per block, the time per input sample, the engine's state and
history memory and the process peak RSS. Every chunk length gives the same
output bit for bit (checked against the automatic choice); only memory and
time change.

The configuration is the current CryoPAF preset
(``dmtlib.sensitivity.CRYOPAF``: 512 channels over 704-1216 MHz at 512 us,
DM 0-2000 on the piecewise-uniform Levin grid of tolerance 1.1), with the
benchmark suite's 32768-sample blocks.

Usage::

    python bench/scripts/bench_chunking.py
    python bench/scripts/bench_chunking.py --chunks 2048 8192 --nthreads 1

Each chunk length runs in its own process (peak RSS is per process).
"""

from __future__ import annotations

import argparse
import json
import resource
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
from dmtlib.sensitivity import CRYOPAF as CFG  # noqa: E402

NSAMPS = 32768  # the benchmark suite's reference block (16.8 s)


def run_one(nsamps: int, chunk: int | None, nthreads: int, repeats: int) -> dict:
    kw = {"dm_grid": CFG.dm_trials, "mode": "valid", "nthreads": nthreads}
    if chunk is not None:
        kw["chunk_nsamps"] = chunk
    fdmt = dmtlib.FDMT(CFG.f_min, CFG.f_max, CFG.nchans, nsamps, CFG.tsamp, **kw)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((CFG.nchans, nsamps), dtype=np.float32)
    out = np.empty(fdmt.plan.dmt_size, np.float32)
    fdmt.execute(x, out=out)  # first call: page faults
    times = []
    for _ in range(repeats):
        fdmt.reset_history()
        t0 = time.perf_counter()
        fdmt.execute(x, out=out)
        times.append(time.perf_counter() - t0)
    mem = fdmt.memory_usage
    return {
        "chunk": int(fdmt.chunk_nsamps),
        "time_s": min(times),
        "ns_per_sample": min(times) * 1e9 / nsamps,
        "fuse_levels": int(fdmt.fuse_levels),
        "state_gib": mem.state / 2**30,
        "history_gib": mem.history / 2**30,
        "output_gib": mem.output / 2**30,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
        "checksum": float(np.sum(out, dtype=np.float64)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nsamps", type=int, default=NSAMPS)
    ap.add_argument("--chunks", type=int, nargs="+",
                    default=[0, 2048, 4096, 8192, 16384, 32768],
                    help="chunk lengths (0: automatic)")  # fmt: skip
    ap.add_argument("--nthreads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one is not None:
        chunk = None if args.one == 0 else args.one
        print(json.dumps(run_one(args.nsamps, chunk, args.nthreads, args.repeats)))
        return
    print(
        f"{CFG.nchans} ch, {CFG.f_min:g}-{CFG.f_max:g} MHz, {CFG.tsamp * 1e6:g} us, "
        f"{len(CFG.dm_trials)} trials to DM {CFG.dm_max:g}, {args.nsamps} samples, "
        f"{args.nthreads} threads"
    )
    checksum = None
    for chunk in args.chunks:
        res = subprocess.run(  # noqa: S603
            [sys.executable, __file__, "--nsamps", str(args.nsamps), "--one",
             str(chunk), "--nthreads", str(args.nthreads),
             "--repeats", str(args.repeats)],
            capture_output=True, text=True, check=True,
        )  # fmt: skip
        r = json.loads(res.stdout.strip().splitlines()[-1])
        if checksum is None:
            checksum = r["checksum"]
        same = "" if r["checksum"] == checksum else "  OUTPUT DIFFERS"
        print(
            f"  chunk {r['chunk']:7d}{' (auto)' if chunk == 0 else '       '} "
            f"{r['time_s'] * 1e3:9.1f} ms ({r['ns_per_sample']:8.1f} ns/sample, "
            f"F={r['fuse_levels']})  "
            f"state {r['state_gib']:6.2f} GiB  history {r['history_gib']:5.2f} GiB  "
            f"peak RSS {r['peak_rss_gib']:6.2f} GiB{same}",
            flush=True,
        )


if __name__ == "__main__":
    main()
