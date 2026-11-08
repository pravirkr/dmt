"""Run the published dmt benchmark suite on this machine.

One command per preset and suite; the JSON it writes is what plot_suite.py,
plot_cfdmt.py and plot_uwl.py read:

    export OMP_PLACES='{0}:8:8'
    python bench/scripts/run_suite.py                         # meerkat_lband
    python bench/scripts/run_suite.py --config cryopaf_lband
    python bench/scripts/run_suite.py --config cryopaf_low
    python bench/scripts/run_suite.py --suite boxcar          # each preset
    python bench/scripts/run_suite.py --suite boxcar --config parkes_uwl
    python bench/scripts/run_suite.py --suite cfdmt

Every result lands in ``bench/results/<config>/<machine>/`` (``cfdmt`` for
``--suite cfdmt``): ``suite_{cpu,<gpu>}.json`` (main), ``boxcar_{cpu,<gpu>}.json``
(FDMTBoxcar at the preset's reference point; for ``parkes_uwl`` also the
real-time sweep at 64-1024 us) or ``suite_cfdmt_{cpu,<gpu>}.json``, where
``gpu`` is ``cuda`` or ``hip``, whichever the build has (Google Benchmark
JSON with the machine description in its ``context``). The machine defaults
to the reference node, ``l40s-epyc9654``. See bench/README.md.
"""

from __future__ import annotations

import argparse
import json
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


def compact_benchmark_json(path: Path) -> None:
    """Drop pretty-printing so suite JSON stays under pre-commit size limits."""
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")


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


def gpu_model(kind: str) -> str:
    if kind == "hip":
        if shutil.which("rocm-smi") is None:
            return ""
        out = _run(["rocm-smi", "--showproductname", "--csv"])
        rows = [r for r in out.splitlines()[1:] if r.strip()]
        return rows[0].split(",")[-1].strip() if rows else ""
    if shutil.which("nvidia-smi") is None:
        return ""
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return names.splitlines()[0] if names else ""


def load_average() -> str:
    """1/5/15-minute load average and online CPU count (Linux, macOS)."""
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return "-"
    return f"{one:.1f} {five:.1f} {fifteen:.1f} of {os.cpu_count()} CPUs"


def virtualized() -> str:
    """Hypervisor vendor when the CPU flags report one, else "no"."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file() or " hypervisor" not in cpuinfo.read_text():
        return "no"
    out = _run(["lscpu"])
    m = re.search(r"^Hypervisor vendor:\s*(.+)$", out, re.MULTILINE)
    return m.group(1).strip() if m else "yes"


def dmt_version() -> str:
    text = (REPO / "CMakeLists.txt").read_text()
    m = re.search(r"project\(\s*dmt\s+VERSION\s+([\d.]+)", text)
    return m.group(1) if m else "unknown"


def context_arg(ctx: dict[str, str]) -> str:
    # Google Benchmark splits --benchmark_context on ',' and '='.
    clean = {k: re.sub(r"[,=]", " ", v).strip() or "-" for k, v in ctx.items()}
    return ",".join(f"{k}={v}" for k, v in clean.items())


def results_root(suite: str, config: str) -> Path:
    """bench/results/<config> (bench/results/cfdmt for the CohFDMT suite)."""
    return REPO / "bench" / "results" / ("cfdmt" if suite == "cfdmt" else config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--machine",
        default="l40s-epyc9654",
        help="results folder name (default l40s-epyc9654, the reference node)",
    )
    ap.add_argument("--label", help="display name in plots (default: CPU/GPU model)")
    ap.add_argument("--build-dir", help="CMake build directory (default: auto)")
    ap.add_argument("--threads", default="1,8", help="CPU thread counts (default 1,8)")
    ap.add_argument(
        "--placement",
        default="close",
        choices=["close", "none"],
        help="CPU thread placement: close (OMP_PROC_BIND=close OMP_PLACES=cores, "
        "packed onto neighbouring cores) or none (unbound, scheduled by the OS "
        "as a shared batch node runs them); recorded in the results",
    )
    ap.add_argument(
        "--repetitions", type=int, default=3, help="repetitions per point (default 3)"
    )
    ap.add_argument("--max-gb", type=float, help="memory budget (default 60%% of RAM)")
    ap.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--gpu",
        "--cuda",
        dest="gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the GPU (CUDA or HIP) benchmarks (default: if the binary has them)",
    )
    ap.add_argument(
        "--suite",
        default="main",
        choices=["main", "boxcar", "cfdmt"],
        help="main: the filterbank algorithms (FDMT, DDMT, ...); boxcar: "
        "FDMTBoxcar at the preset's reference point (parkes_uwl: plus the "
        "real-time sweep); cfdmt: the CohFDMT baseband search",
    )
    ap.add_argument(
        "--config",
        default="meerkat_lband",
        choices=["meerkat_lband", "cryopaf_lband", "cryopaf_low", "parkes_uwl"],
        help="data configuration of --suite main / boxcar "
        "(bench/suite/suite_common.hpp): meerkat_lband (default; 4096 "
        "channels, 856-1712 MHz at 612.48 us), cryopaf_lband (1152 channels, "
        "1100-1714 MHz at 455.6 us, subband depth 1), cryopaf_low (512 "
        "channels, 704-1216 MHz at 512 us) or parkes_uwl (6656 channels, "
        "704-4032 MHz at 64 us, subband depth 4; boxcar suite only); all DM "
        "0-2000, Levin tolerance 1.1 (UWL: 1.6)",
    )
    ap.add_argument("--quick", action="store_true", help="1 repetition, short runs")
    ap.add_argument("--filter", default="", help="extra regex AND-ed with the suite")
    ap.add_argument(
        "--fftw-planner",
        default="measure",
        choices=["estimate", "measure", "patient"],
        help="FFTW planner effort of the Fourier engines (default measure: what "
        "a long-running pipeline uses with saved wisdom)",
    )
    ap.add_argument(
        "--fftw-wisdom",
        help="FFTW wisdom file, reused across runs (default: "
        "<build-dir>/fftw_<machine>.wisdom)",
    )
    args = ap.parse_args()

    binary = find_binary(args.build_dir)
    btype = build_type(binary)
    if btype != "Release":
        print(f"warning: {binary} is a {btype!r} build; publish Release numbers")

    listed = _run([str(binary), "--benchmark_list_tests"]).splitlines()
    gpu_kind = next(
        (k for k in ("cuda", "hip") if any(f"/{k}/" in name for name in listed)), ""
    )
    run_gpu = bool(gpu_kind) if args.gpu is None else args.gpu
    if run_gpu and not gpu_kind:
        sys.exit("the suite binary has no GPU benchmarks (no GPU build or no GPU)")

    out_dir = results_root(args.suite, args.config) / args.machine
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_model(gpu_kind) if run_gpu else ""
    ctx = {
        "machine": args.machine,
        "label": args.label or "",
        "cpu_model": cpu_model(),
        "gpu_model": gpu,
        "ram_gb": f"{ram_gb():.0f}",
        "dmt_version": dmt_version(),
        "build_type": btype,
        "threads": args.threads.replace(",", " "),
        "fftw_planner": args.fftw_planner,
        "config": args.config if args.suite != "cfdmt" else "-",
        "placement": args.placement,
        "virtualized": virtualized(),
    }
    wisdom = args.fftw_wisdom or str(
        binary.parent.parent / f"fftw_{args.machine}.wisdom"
    )
    env = {
        **os.environ,
        "DMT_BENCH_THREADS": args.threads,
        "DMT_BENCH_MAX_GB": f"{args.max_gb or 0.6 * ram_gb():.1f}",
        **(
            {
                "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "close"),
                "OMP_PLACES": os.environ.get("OMP_PLACES", "cores"),
            }
            if args.placement == "close"
            else {}
        ),
        # Planned once per size and machine (the first run pays for it, at
        # engine construction, outside the timed region).
        "DMT_FFTW_PLANNER": args.fftw_planner,
        "DMT_FFTW_WISDOM": wisdom,
        "DMT_BENCH_CONFIG": args.config,
    }

    if args.suite == "main" and args.config == "parkes_uwl":
        sys.exit(
            "parkes_uwl is benchmarked with --suite boxcar (FDMT-FFT needs "
            "~50 GB and CPU DDMT hours on it)"
        )
    runs = []
    if args.suite == "boxcar":
        stem = "boxcar"
        if args.cpu:
            runs.append(("cpu", r"^BM_fdmt_boxcar_(stream|rt)/.*/cpu[0-9]+"))
        if run_gpu:
            runs.append((gpu_kind, rf"^BM_fdmt_boxcar_(cuda/|rt/.*/{gpu_kind}/)"))
    else:
        prefix, stem = (
            ("cfdmt", "suite_cfdmt") if args.suite == "cfdmt" else ("suite", "suite")
        )
        if args.cpu:
            runs.append(("cpu", rf"^{prefix}/.*/cpu[0-9]+/"))
        if run_gpu:
            runs.append((gpu_kind, rf"^{prefix}/.*/{gpu_kind}/"))
    if args.placement == "none":
        for key in ("OMP_PROC_BIND", "OMP_PLACES"):
            env.pop(key, None)
    for kind, pattern in runs:
        regex = pattern + (f".*{args.filter}" if args.filter else "")
        out = out_dir / f"{stem}_{kind}.json"
        run_env = env
        if kind != "cpu":
            # Thread binding is for the CPU runs; on a GPU run it would pin
            # the host threads that build the plans to a single core.
            run_env = {
                k: v
                for k, v in env.items()
                if k not in ("OMP_PROC_BIND", "OMP_PLACES") or k in os.environ
            }
        # Google Benchmark >= 1.10 re-executes itself to disable ASLR. With
        # OMP_PROC_BIND set, libgomp has already bound the main thread to
        # its first place by then, and the re-executed process inherits that
        # one-CPU affinity: every OpenMP thread runs on one core. Starting
        # with ASLR off (setarch -R) skips the re-exec.
        setarch = shutil.which("setarch") if sys.platform == "linux" else None
        cmd = [
            *([setarch, "-R"] if setarch else []),
            str(binary),
            f"--benchmark_filter={regex}",
            f"--benchmark_repetitions={1 if args.quick else args.repetitions}",
            f"--benchmark_min_time={'0.05s' if args.quick else '0.2s'}",
            f"--benchmark_out={out}",
            "--benchmark_out_format=json",
            f"--benchmark_context={context_arg({**ctx, 'load_start': load_average()})}",
        ]
        print(f"[{kind}] {binary.name} -> {out.relative_to(REPO)}", flush=True)
        start = time.monotonic()
        subprocess.run(cmd, check=True, env=run_env, cwd=REPO)  # noqa: S603
        data = json.loads(out.read_text())
        data.setdefault("context", {})["load_end"] = load_average()
        out.write_text(json.dumps(data))
        compact_benchmark_json(out)
        print(f"[{kind}] done in {(time.monotonic() - start) / 60:.1f} min")
    if args.suite == "cfdmt":
        print("Next: python bench/scripts/plot_cfdmt.py")
    elif args.suite == "boxcar" and args.config == "parkes_uwl":
        print("Next: python bench/scripts/plot_uwl.py")
    elif args.suite == "main":
        print(f"Next: python bench/scripts/plot_suite.py --config {args.config}")


if __name__ == "__main__":
    main()
