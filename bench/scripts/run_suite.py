"""Run the published dmt benchmark suite on this machine.

One command per machine; the JSON it writes is what plot_suite.py reads:

    python bench/scripts/run_suite.py --machine m1-pro
    python bench/scripts/run_suite.py --machine xeon-6348h --build-dir build
    python bench/scripts/run_suite.py --machine l40s --no-cpu      # GPU box

Writes ``bench/results/<machine>/suite_cpu.json`` and/or ``suite_cuda.json``
(Google Benchmark JSON with the machine description in its ``context``).
See bench/README.md for the full workflow.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BINARY = "dmt_bench_suite"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed local tools
            cmd, check=True, capture_output=True, text=True, cwd=REPO
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def find_binary(build_dir: str | None) -> Path:
    candidates = (
        [Path(build_dir)]
        if build_dir
        else [REPO / "build", REPO / "build2", *sorted(REPO.glob("build*"))]
    )
    for base in candidates:
        path = (base if base.is_absolute() else REPO / base) / "bench" / BINARY
        if path.is_file():
            return path
    sys.exit(
        f"{BINARY} not found (looked in {', '.join(str(c) for c in candidates)}). "
        "Build with -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_BENCHMARKS=ON, "
        "or pass --build-dir."
    )


def build_type(binary: Path) -> str:
    cache = binary.parent.parent / "CMakeCache.txt"
    if cache.is_file():
        m = re.search(r"^CMAKE_BUILD_TYPE:\w+=(\w*)", cache.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    return "unknown"


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if m:
            return m.group(1).strip()
    return platform.processor() or "unknown"


def ram_gb() -> float:
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out) / 2**30 if out else 16.0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        m = re.search(r"^MemTotal:\s*(\d+) kB", meminfo.read_text(), re.MULTILINE)
        if m:
            return int(m.group(1)) / 2**20
    return 16.0


def gpu_model() -> str:
    if shutil.which("nvidia-smi") is None:
        return ""
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return names.splitlines()[0] if names else ""


def dmt_version() -> str:
    text = (REPO / "CMakeLists.txt").read_text()
    m = re.search(r"project\(\s*dmt\s+VERSION\s+([\d.]+)", text)
    return m.group(1) if m else "unknown"


def context_arg(ctx: dict[str, str]) -> str:
    # Google Benchmark splits --benchmark_context on ',' and '='.
    clean = {k: re.sub(r"[,=]", " ", v).strip() or "-" for k, v in ctx.items()}
    return ",".join(f"{k}={v}" for k, v in clean.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--machine", required=True, help="results folder name, e.g. m1-pro")
    ap.add_argument("--label", help="display name in plots (default: CPU/GPU model)")
    ap.add_argument("--build-dir", help="CMake build directory (default: auto)")
    ap.add_argument("--threads", default="1,8", help="CPU thread counts (default 1,8)")
    ap.add_argument("--max-gb", type=float, help="memory budget (default 60%% of RAM)")
    ap.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run CUDA benchmarks (default: if the binary has them)",
    )
    ap.add_argument("--quick", action="store_true", help="1 repetition, short runs")
    ap.add_argument("--filter", default="", help="extra regex AND-ed with the suite")
    args = ap.parse_args()

    binary = find_binary(args.build_dir)
    btype = build_type(binary)
    if btype != "Release":
        print(f"warning: {binary} is a {btype!r} build; publish Release numbers")

    listed = _run([str(binary), "--benchmark_list_tests"]).splitlines()
    has_cuda = any("/cuda" in name for name in listed)
    run_cuda = has_cuda if args.cuda is None else args.cuda
    if run_cuda and not has_cuda:
        sys.exit("the suite binary has no CUDA benchmarks (no CUDA build or no GPU)")

    out_dir = REPO / "bench" / "results" / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_model() if run_cuda else ""
    ctx = {
        "machine": args.machine,
        "label": args.label or "",
        "cpu_model": cpu_model(),
        "gpu_model": gpu,
        "ram_gb": f"{ram_gb():.0f}",
        "dmt_version": dmt_version(),
        "build_type": btype,
        "threads": args.threads.replace(",", " "),
    }
    env = {
        **os.environ,
        "DMT_BENCH_THREADS": args.threads,
        "DMT_BENCH_MAX_GB": f"{args.max_gb or 0.6 * ram_gb():.1f}",
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "close"),
        "OMP_PLACES": os.environ.get("OMP_PLACES", "cores"),
    }

    runs = []
    if args.cpu:
        runs.append(("cpu", r"^suite/.*/cpu[0-9]+/"))
    if run_cuda:
        runs.append(("cuda", r"^suite/.*/cuda(_host)?/"))
    for kind, pattern in runs:
        regex = pattern + (f".*{args.filter}" if args.filter else "")
        out = out_dir / f"suite_{kind}.json"
        cmd = [
            str(binary),
            f"--benchmark_filter={regex}",
            f"--benchmark_repetitions={1 if args.quick else 3}",
            f"--benchmark_min_time={'0.05s' if args.quick else '0.5s'}",
            f"--benchmark_out={out}",
            "--benchmark_out_format=json",
            f"--benchmark_context={context_arg(ctx)}",
        ]
        print(f"[{kind}] {binary.name} -> {out.relative_to(REPO)}", flush=True)
        start = time.monotonic()
        subprocess.run(cmd, check=True, env=env, cwd=REPO)  # noqa: S603
        print(f"[{kind}] done in {(time.monotonic() - start) / 60:.1f} min")
    print("Next: python bench/scripts/plot_suite.py")


if __name__ == "__main__":
    main()
