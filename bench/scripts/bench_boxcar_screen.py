"""Side-by-side BoxcarScreenMode A/B benchmark and candidate parity harness.

Compares ``FDMTBoxcar`` across ``BoxcarScreenMode`` settings (``prefix``,
``kadane``, ``kadane_surrogate``, ``none``) against a bare ``FDMT`` transform
baseline on CPU or GPU (``cuda`` / ``hip``), reporting steady-state block wall
time, boxcar-only overhead above bare FDMT, engine memory usage, and candidate
count / peak S/N agreement on an injected dispersed pulse.

Usage::

    # Quick laptop A/B matrix across 6 key regimes on 1 and 2 threads:
    OMP_NUM_THREADS=1 python bench/scripts/bench_boxcar_screen.py --threads 1 2

    # Full preset reference block on an HPC CPU or GPU node:
    python bench/scripts/bench_boxcar_screen.py --preset meerkat_lband --threads 8
    python bench/scripts/bench_boxcar_screen.py --preset meerkat_lband --backend cuda
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _bootstrap_in_tree_dmtlib() -> Path | None:
    """Stage a matching in-tree build*/src/libdmt*.so when dmtlib is not staged."""
    src_pkg = REPO / "src" / "dmtlib"
    if any((src_pkg / f"libdmt{sfx}").is_file() for sfx in EXTENSION_SUFFIXES):
        sys.path.insert(0, str(REPO / "src"))
        return None
    for candidate in (
        REPO / "build-review",
        REPO / "build2",
        REPO / "build",
        *sorted(REPO.glob("build*")),
    ):
        for sfx in EXTENSION_SUFFIXES:
            so_path = candidate / "src" / f"libdmt{sfx}"
            if so_path.is_file():
                stage_root = Path(tempfile.mkdtemp(prefix="dmt_bench_pkg_"))
                pkg_dir = stage_root / "dmtlib"
                pkg_dir.mkdir(parents=True, exist_ok=True)
                for py_file in src_pkg.glob("*.py"):
                    (pkg_dir / py_file.name).symlink_to(py_file)
                (pkg_dir / so_path.name).symlink_to(so_path)
                sys.path.insert(0, str(stage_root))
                return stage_root
    sys.path.insert(0, str(REPO / "src"))
    return None


_STAGE_DIR = _bootstrap_in_tree_dmtlib()

import dmtlib  # noqa: E402
from dmtlib import (  # noqa: E402
    FDMT,
    BoxcarConfig,
    BoxcarScreenMode,
    FDMTBoxcar,
    FDMTPlan,
    add_frb_track,
)


@dataclass(frozen=True)
class Preset:
    f_min: float
    f_max: float
    nchans: int
    nsamps: int
    tsamp: float
    dm_max: float
    levin_tol: float


PRESETS: dict[str, Preset] = {
    "laptop": Preset(704.0, 1216.0, 256, 4096, 512e-6, 500.0, 1.15),
    "cryopaf_low": Preset(704.0, 1216.0, 512, 32768, 512e-6, 2000.0, 1.1),
    "cryopaf_lband": Preset(1100.0, 1714.0, 1152, 32768, 455.6e-6, 2000.0, 1.1),
    "meerkat_lband": Preset(856.0, 1712.0, 4096, 32768, 612.48e-6, 2000.0, 1.1),
}

MODE_MAP: dict[str, BoxcarScreenMode] = {
    "prefix": BoxcarScreenMode.PREFIX,
    "kadane": BoxcarScreenMode.KADANE,
    "kadane_surrogate": BoxcarScreenMode.KADANE_SURROGATE,
    "none": BoxcarScreenMode.NONE,
}


@dataclass(frozen=True)
class Regime:
    label: str
    subband_depth: int
    wtsp: float
    nbits: int
    trapezoid: bool


DEFAULT_REGIMES: tuple[Regime, ...] = (
    Regime("d_sub=0, wtsp=1.5 (f32)", 0, 1.5, 32, trapezoid=False),
    Regime("d_sub=0, wtsp=1.0 (f32)", 0, 1.0, 32, trapezoid=False),
    Regime("d_sub=0, wtsp=1.5 (uint8)", 0, 1.5, 8, trapezoid=False),
    Regime("d_sub=0, wtsp=1.5, trap", 0, 1.5, 32, trapezoid=True),
    Regime("d_sub=0, wtsp=1.0, trap", 0, 1.0, 32, trapezoid=True),
    Regime("d_sub=1, wtsp=1.5 (f32)", 1, 1.5, 32, trapezoid=False),
)


@dataclass
class ModeMeasurement:
    ms: float
    overhead_ms: float
    mem_mib: float
    ncands: int
    peak_snr: float
    exact_match: bool | None


def _sorted_cands(cands: np.ndarray) -> np.ndarray:
    fields = [
        "beam_idx",
        "subband_level",
        "subband_idx",
        "dm_idx",
        "filter_idx",
        "begin_sample",
        "end_sample",
        "peak_sample",
    ]
    return np.sort(cands, order=fields)


def _same_candidates(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) != len(b):
        return False
    for name in a.dtype.names or ():
        if name == "snr":
            if not np.allclose(a[name], b[name], rtol=2e-5, atol=1e-5):
                return False
        elif not np.array_equal(a[name], b[name]):
            return False
    return True


def _time_bare_fdmt(
    plan: FDMTPlan,
    dm_grid: list[float],
    wf_warm: np.ndarray,
    wf_timed: np.ndarray,
    nbits: int,
    nthreads: int,
    backend: str,
    device: int,
    warmup_blocks: int,
    repeats: int,
) -> float:
    fdmt = FDMT(
        plan.f_min,
        plan.f_max,
        plan.nchans,
        plan.nsamps,
        plan.tsamp,
        dm_grid=dm_grid,
        use_box_smearing=True,
        mode="valid",
        nthreads=nthreads,
        backend=backend,
        device=device,
    )
    out = np.empty(fdmt.plan.dmt_size, dtype=np.float32)
    for _ in range(warmup_blocks):
        if nbits == 32:
            fdmt.execute(wf_warm, out=out)
        else:
            fdmt.execute(wf_warm, nbits=nbits, out=out)
    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if nbits == 32:
            fdmt.execute(wf_timed, out=out)
        else:
            fdmt.execute(wf_timed, nbits=nbits, out=out)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times_ms)


def run_regime(
    regime: Regime,
    f_min: float,
    f_max: float,
    nchans: int,
    nsamps: int,
    tsamp: float,
    dm_grid: list[float],
    max_boxcar: int,
    modes: list[str],
    nthreads: int,
    backend: str,
    device: int,
    repeats: int,
) -> tuple[float, int, int, dict[str, ModeMeasurement]]:
    plan = FDMTPlan(
        f_min,
        f_max,
        nchans,
        nsamps,
        tsamp,
        dm_grid=dm_grid,
        mode="valid",
        subband_depth=regime.subband_depth,
    )
    rng = np.random.default_rng(1234)
    warm_f32 = rng.standard_normal((nchans, nsamps), dtype=np.float32)
    timed_f32 = rng.standard_normal((nchans, nsamps), dtype=np.float32)
    pulse_dm = min(len(plan.dm_grid_final) // 3, len(plan.dm_grid_final) - 1)
    pulse_w = min(8, max_boxcar)
    amp = 16.0 / float(np.sqrt(nchans * pulse_w))
    add_frb_track(timed_f32, plan, pulse_dm, amp, nsamps // 2, pulse_w)

    if regime.nbits == 8:
        wf_warm = np.clip(np.round(127.0 + 12.0 * warm_f32), 0, 255).astype(np.uint8)
        wf_timed = np.clip(np.round(127.0 + 12.0 * timed_f32), 0, 255).astype(np.uint8)
        sigma, mean = 12.0, 127.0
    else:
        wf_warm, wf_timed = warm_f32, timed_f32
        sigma, mean = 1.0, 0.0

    warmup_samples = max(int(plan.dt_max) + max_boxcar, nsamps)
    warmup_blocks = max(1, (warmup_samples + nsamps - 1) // nsamps)

    fdmt_ms = _time_bare_fdmt(
        plan,
        dm_grid,
        wf_warm,
        wf_timed,
        regime.nbits,
        nthreads,
        backend,
        device,
        warmup_blocks,
        repeats,
    )

    ref_cands: np.ndarray | None = None
    nwidths = 0
    results: dict[str, ModeMeasurement] = {}
    for mname in modes:
        cfg = BoxcarConfig(
            max_boxcar,
            wtsp=regime.wtsp,
            detect_thresh=6.0,
            noise_sigma=sigma,
            input_nbits=regime.nbits,
            input_mean=mean,
            use_trapezoid=regime.trapezoid,
            screen_mode=MODE_MAP[mname],
        )
        eng = FDMTBoxcar(
            plan,
            cfg,
            nthreads=nthreads,
            backend=backend,
            device=device,
        )
        nwidths = len(eng.search_plans[0].widths)
        for _ in range(warmup_blocks):
            eng.execute(wf_warm)
        first_cands: np.ndarray | None = None
        times_ms: list[float] = []
        for idx in range(repeats):
            t0 = time.perf_counter()
            res = eng.execute(wf_timed)
            times_ms.append((time.perf_counter() - t0) * 1e3)
            if idx == 0:
                first_cands = _sorted_cands(res.candidates)
        assert first_cands is not None  # noqa: S101
        med_ms = statistics.median(times_ms)
        peak_snr = float(first_cands["snr"].max()) if len(first_cands) else 0.0
        exact_ok: bool | None = None
        if mname == "prefix":
            ref_cands = first_cands
            exact_ok = True
        elif ref_cands is not None and mname in ("kadane", "none"):
            exact_ok = _same_candidates(first_cands, ref_cands)
        results[mname] = ModeMeasurement(
            ms=med_ms,
            overhead_ms=med_ms - fdmt_ms,
            mem_mib=eng.memory_usage.total / (1024.0 * 1024.0),
            ncands=len(first_cands),
            peak_snr=peak_snr,
            exact_match=exact_ok,
        )
    return fdmt_ms, len(plan.dm_grid_final), nwidths, results


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--preset",
        default="laptop",
        choices=sorted(PRESETS),
        help="filterbank geometry preset (default: laptop, 256 ch x 4096 samps)",
    )
    ap.add_argument("--nchans", type=int, help="override channel count")
    ap.add_argument("--nsamps", type=int, help="override samples per block")
    ap.add_argument("--dm-max", type=float, help="override maximum DM")
    ap.add_argument(
        "--max-boxcar",
        type=int,
        default=0,
        help="maximum boxcar width (default: floor(0.1 s / tsamp))",
    )
    ap.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "hip"])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1],
        help="CPU thread counts to compare (default: 1)",
    )
    ap.add_argument(
        "--screen-modes",
        nargs="+",
        default=["prefix", "kadane", "kadane_surrogate"],
        choices=list(MODE_MAP),
        help="BoxcarScreenMode values to run side by side",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="timed repetitions per mode after warmup (default: 3)",
    )
    ap.add_argument("--wtsp", type=float, help="single-regime override: wtsp")
    ap.add_argument("--subband-depth", type=int, default=0, help="single-regime depth")
    ap.add_argument("--nbits", type=int, default=32, choices=[8, 32])
    ap.add_argument("--trapezoid", action="store_true", help="single-regime trapezoid")
    return ap.parse_args()


def _format_parity(
    reg: Regime, modes: list[str], meas: dict[str, ModeMeasurement]
) -> str:
    parts: list[str] = []
    if "prefix" in meas:
        p = meas["prefix"]
        parts.append(f"pfx:{p.ncands}@{p.peak_snr:.1f}")
    for m in modes:
        if m == "prefix":
            continue
        cur = meas[m]
        if cur.exact_match is not None:
            parts.append(f"{m}:{'EXACT' if cur.exact_match else 'MISMATCH'}")
            if not cur.exact_match:
                msg = f"Exact candidate mismatch in {reg.label} for {m}"
                raise RuntimeError(msg)
        else:
            parts.append(f"sur:{cur.ncands}@{cur.peak_snr:.1f}")
    return ", ".join(parts)


def _resolve_geometry(
    args: argparse.Namespace,
) -> tuple[Preset, int, int, float, int, list[float]]:
    preset = PRESETS[args.preset]
    nchans = args.nchans if args.nchans is not None else preset.nchans
    nsamps = args.nsamps if args.nsamps is not None else preset.nsamps
    dm_max = args.dm_max if args.dm_max is not None else preset.dm_max
    max_boxcar = args.max_boxcar or max(1, int(0.1 / preset.tsamp))
    dm_grid = list(
        dmtlib.DDMTPlan.generate_levin_dm_grid_piecewise(
            0.0,
            dm_max,
            preset.tsamp,
            preset.tsamp,
            preset.f_min,
            preset.f_max,
            nchans,
            preset.levin_tol,
        )
    )
    return preset, nchans, nsamps, dm_max, max_boxcar, dm_grid


def main() -> None:
    try:
        args = parse_args()
        preset, nchans, nsamps, _dm_max, max_boxcar, dm_grid = _resolve_geometry(args)
        regimes = (
            [
                Regime(
                    f"d_sub={args.subband_depth}, wtsp={args.wtsp:g} "
                    f"({'uint8' if args.nbits == 8 else 'f32'}"
                    f"{', trap' if args.trapezoid else ''})",
                    args.subband_depth,
                    args.wtsp,
                    args.nbits,
                    trapezoid=args.trapezoid,
                )
            ]
            if args.wtsp is not None
            else list(DEFAULT_REGIMES)
        )
        thread_list = args.threads if args.backend == "cpu" else [1]
        modes: list[str] = list(dict.fromkeys(args.screen_modes))

        print(
            f"# BoxcarScreenMode A/B ({args.preset}: {nchans} ch, "
            f"{preset.f_min:g}-{preset.f_max:g} MHz, tsamp={preset.tsamp * 1e6:g} us, "
            f"nsamps={nsamps}, max_boxcar={max_boxcar}, backend={args.backend})"
        )
        for thr in thread_list:
            hdr_modes = " | ".join(f"{m} ms (ovh)" for m in modes)
            print(f"\n## Threads = {thr}" if args.backend == "cpu" else "\n## GPU")
            print(f"| regime | ndms | M | FDMT ms | {hdr_modes} | cands / S/N parity |")
            sep = " | ".join("---:" for _ in modes)
            print(f"| :--- | ---: | ---: | ---: | {sep} | :--- |")
            for reg in regimes:
                fdmt_ms, ndms, nwidths, meas = run_regime(
                    reg,
                    preset.f_min,
                    preset.f_max,
                    nchans,
                    nsamps,
                    preset.tsamp,
                    dm_grid,
                    max_boxcar,
                    modes,
                    thr,
                    args.backend,
                    args.device,
                    args.repeats,
                )
                cells = [
                    f"{meas[m].ms:.2f} ({meas[m].overhead_ms:+.2f})" for m in modes
                ]
                parity = _format_parity(reg, modes, meas)
                print(
                    f"| {reg.label} | {ndms} | {nwidths} | {fdmt_ms:.2f} | "
                    f"{' | '.join(cells)} | {parity} |"
                )
    finally:
        if _STAGE_DIR is not None:
            shutil.rmtree(_STAGE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
