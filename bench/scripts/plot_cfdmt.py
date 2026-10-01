"""Plot the CohFDMT section of the dmt benchmark suite.

Reads every ``bench/results/<machine>/suite_cfdmt_*.json`` written by
``run_suite.py --suite cfdmt`` and writes, for both themes,

    bench/results/plots/{light,dark}/cfdmt_rtf.png

plus ``bench/results/plots/cfdmt_summary.md`` (reference-point table), which
the docs page includes. CohFDMT searches baseband voltages (one GUPPI node,
64 x 2.93 MHz at 1312.5-1500 MHz, int8, t_p = 10 us, DM 50-60, see
bench/suite/suite_cfdmt_common.hpp), so it has its own figures and is not
compared with the filterbank algorithms.

    python bench/scripts/plot_cfdmt.py [--results bench/results]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_suite import MARKERS, REPO, THEMES, style  # noqa: E402

# Reference point (suite_cfdmt_common.hpp).
REF = {"tp": 10, "nbits": 8, "dmw": 10, "block": 0}
SWEEPS = [
    ("tp", "target resolution t_p (µs)", "tp"),
    ("nbits", "input bits per real sample", "nbits"),
    ("dmw", "DM range width (pc cm⁻³, from DM 50)", "dmw"),
    ("block", "block length (raw samples per subband; 0 = auto)", "block"),
]
# Series colours: the suite's categorical slots in fixed order.
SLOTS = ["FDMT", "FDMT-FFT", "DDMT", "SDMT", "DDMT-FFT"]


@dataclass
class Point:
    machine: str
    backend: str
    sweep: str
    params: dict[str, int]
    times: list[float] = field(default_factory=list)
    counters: dict[str, float] = field(default_factory=dict)
    skipped: str = ""

    @property
    def time(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")

    @property
    def rtf(self) -> float:
        return self.counters.get("rtf_data_s", float("nan")) / self.time


def load(results_dir: Path) -> tuple[list[Point], dict[str, dict[str, str]]]:
    points: dict[tuple, Point] = {}
    contexts: dict[str, dict[str, str]] = {}
    unit = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
    for path in sorted(results_dir.glob("*/suite_cfdmt_*.json")):
        data = json.loads(path.read_text())
        machine = path.parent.name
        contexts.setdefault(machine, data.get("context", {}))
        for b in data.get("benchmarks", []):
            parts = b["name"].split("/")
            if parts[0] != "cfdmt" or b.get("run_type") == "aggregate":
                continue
            params = {k: int(v) for k, v in (p.split(":", 1) for p in parts[3:7])}
            key = (machine, parts[2], parts[1], tuple(sorted(params.items())))
            pt = points.setdefault(key, Point(machine, parts[2], parts[1], params))
            if b.get("error_occurred"):
                pt.skipped = b.get("error_message", "skipped")
                continue
            t = b["real_time"] * unit[b.get("time_unit", "ns")]
            pt.times.append(t)
            # Google Benchmark stores rate counters already divided by time:
            # recover the data seconds per call.
            for k in ("ndm", "ndm_coh", "efficiency", "mem_mib", "nchans"):
                if k in b:
                    pt.counters[k] = b[k]
            if "rtf" in b:
                pt.counters["rtf_data_s"] = b["rtf"] * t
    return list(points.values()), contexts


def series_label(machine: str, backend: str, contexts: dict) -> str:
    ctx = {k: v for k, v in contexts.get(machine, {}).items() if v and v != "-"}
    if backend.startswith("cpu"):
        name = ctx.get("label") or ctx.get("cpu_model") or machine
        n = backend[3:]
        return f"{name}, {n} thread{'s' if n != '1' else ''}"
    name = ctx.get("gpu_model") or f"{machine} GPU"
    return f"{name}{' (host arrays)' if backend.endswith('_host') else ''}"


def rtf_figure(points: list[Point], contexts: dict, theme: dict) -> plt.Figure:
    series = sorted({(p.machine, p.backend) for p in points})
    colours = {s: theme["algo"][SLOTS[i % len(SLOTS)]] for i, s in enumerate(series)}
    markers = {s: MARKERS[i % len(MARKERS)] for i, s in enumerate(series)}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for ax, (sweep, xlabel, key) in zip(axes.flat, SWEEPS):
        for s in series:
            pts = sorted(
                (p for p in points if (p.machine, p.backend) == s and p.sweep == sweep
                 and p.times),
                key=lambda p: p.params[key],
            )
            if not pts:
                continue
            xs = list(range(len(pts))) if key == "block" else [p.params[key] for p in pts]
            ys = [p.rtf for p in pts]
            ax.plot(xs, ys, color=colours[s], marker=markers[s], lw=2, ms=8,
                    markeredgecolor=theme["surface"], markeredgewidth=2,
                    label=series_label(*s, contexts))
            ax.annotate(f"{ys[-1]:.2g}×", (xs[-1], ys[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8,
                        color=theme["ink2"])
            if key == "block":
                ax.set_xticks(xs, [str(p.params[key]) if p.params[key] else "auto"
                                   for p in pts])
        ax.axhline(1.0, color=theme["muted"], lw=1, ls="--")
        ax.set_yscale("log")
        plain = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.tick_params(axis="y", which="minor", labelsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("real-time factor (data s / compute s)")
        if key == "nbits":
            ticks = sorted({p.params[key] for p in points if p.sweep == sweep})
            ax.set_xticks(ticks, [str(t) for t in ticks])
        if key in ("tp", "dmw"):
            ax.set_xscale("log")
            ticks = sorted({p.params[key] for p in points if p.sweep == sweep})
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.minorticks_off()
    # Every series once, from whichever panel has it (the host-array series
    # only run in some sweeps).
    legend: dict[str, object] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            legend.setdefault(lab, h)
    handles, labels = list(legend.values()), list(legend.keys())
    fig.legend(handles, labels, loc="outside lower center", ncols=min(3, len(labels)),
               frameon=False, fontsize=8.5)
    fig.suptitle("CohFDMT: baseband searched per second of compute", fontweight="bold",
                 color=theme["ink"])
    fig.text(0.5, 0.955, "GUPPI node: 64 × 2.93 MHz at 1.31-1.50 GHz, int8, t_p 10 µs, "
             "DM 50-60 (reference); above the dashed line is faster than real time",
             ha="center", fontsize=8.5, color=theme["ink2"])
    return fig


def summary_md(points: list[Point], contexts: dict) -> str:
    ref = [p for p in points if p.sweep == "ref" and p.times]
    lines = [
        "| Platform | Time per block | Real-time factor | Coarse trials | DM rows |"
        " Useful fraction | Engine memory |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in sorted(ref, key=lambda p: -p.rtf):
        c = p.counters
        lines.append(
            f"| {series_label(p.machine, p.backend, contexts)} | {p.time * 1e3:.1f} ms "
            f"| {p.rtf:.2f}× | {c.get('ndm_coh', 0):.0f} | {c.get('ndm', 0):.0f} "
            f"| {100 * c.get('efficiency', 0):.0f}% | {c.get('mem_mib', 0):.0f} MiB |"
        )
    skipped = [p for p in points if p.skipped]
    if skipped:
        lines += ["", "Skipped points:"]
        lines += [f"- {p.machine} {p.backend} {p.sweep} {p.params}: {p.skipped}"
                  for p in skipped]
    versions = sorted({c.get("dmt_version", "?") for c in contexts.values()})
    lines += ["", f"dmt version(s): {', '.join(versions)}; FFTW planner: "
              f"{', '.join(sorted({c.get('fftw_planner', '?') for c in contexts.values()}))}."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the CohFDMT benchmark suite.")
    ap.add_argument("--results", default=str(REPO / "bench" / "results"))
    args = ap.parse_args()
    results_dir = Path(args.results)
    points, contexts = load(results_dir)
    if not points:
        sys.exit(f"no CohFDMT results under {results_dir}/*/suite_cfdmt_*.json")
    out_root = results_dir / "plots"
    for name, theme in THEMES.items():
        style(theme)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        fig = rtf_figure(points, contexts, theme)
        fig.savefig(out / "cfdmt_rtf.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    (out_root / "cfdmt_summary.md").write_text(summary_md(points, contexts))
    print(f"wrote cfdmt_rtf.png x {len(THEMES)} themes and cfdmt_summary.md to {out_root}")


if __name__ == "__main__":
    main()
