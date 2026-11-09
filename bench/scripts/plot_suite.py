"""Plot the published dmt benchmark suite.

Reads every ``bench/results/<machine>/suite_*.json`` written by run_suite.py
and writes, for both a light and a dark theme,

    bench/results/plots/{light,dark}/runtime_vs_nsamps.png
    bench/results/plots/{light,dark}/runtime_vs_ndms.png
    bench/results/plots/{light,dark}/throughput_nbits.png
    bench/results/plots/{light,dark}/ops_theory.png        (needs dmtlib)
    bench/results/plots/{light,dark}/readme_highlight.png

plus ``bench/results/plots/summary.md`` (machines, reference-point numbers,
real-time factors, skipped points), which the docs page includes verbatim.

    python bench/scripts/plot_suite.py [--results bench/results] [--no-ops]

Only matplotlib and numpy are required (``pip install -e ".[bench]"``).
Colour = algorithm and marker = platform, in every figure.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

REPO = Path(__file__).resolve().parents[2]
TIMES = "\N{MULTIPLICATION SIGN}"

# Suite configurations (bench/suite/suite_common.hpp). main() installs the
# selected geometry and trial-grid reference in the module constants below.
CONFIGS: dict[str, dict[str, Any]] = {
    "cryopaf_low": {
        "F_MIN": 704.0,
        "F_MAX": 1216.0,
        "TSAMP": 5.12e-4,
        "NCHANS": 512,
        "LEVIN_TOL": 1.1,
        "DM_MAX_REF": 2000.0,
        "NDMS_REF": 1545,
        "DM_MAX_SWEEP": [125.0, 250.0, 500.0, 1000.0, 2000.0],
    },
    "cryopaf_lband": {
        "F_MIN": 1100.0,
        "F_MAX": 1714.0,
        "TSAMP": 455.6e-6,
        "NCHANS": 1152,
        "LEVIN_TOL": 1.1,
        "DM_MAX_REF": 2000.0,
        "NDMS_REF": 1941,
        "DM_MAX_SWEEP": [125.0, 250.0, 500.0, 1000.0, 2000.0],
    },
    "parkes_uwl": {
        "F_MIN": 704.0,
        "F_MAX": 4032.0,
        "TSAMP": 64e-6,
        "NCHANS": 6656,
        "LEVIN_TOL": 1.6,
        "DM_MAX_REF": 2000.0,
        "NDMS_REF": 4749,
        "DM_MAX_SWEEP": [125.0, 250.0, 500.0, 1000.0, 2000.0],
    },
    "meerkat_lband": {
        "F_MIN": 856.0,
        "F_MAX": 1712.0,
        "TSAMP": 612.48e-6,
        "NCHANS": 4096,
        "LEVIN_TOL": 1.1,
        "DM_MAX_REF": 2000.0,
        "NDMS_REF": 4057,
        "DM_MAX_SWEEP": [125.0, 250.0, 500.0, 1000.0, 2000.0],
    },
}
F_MIN: float = CONFIGS["meerkat_lband"]["F_MIN"]
F_MAX: float = CONFIGS["meerkat_lband"]["F_MAX"]
TSAMP: float = CONFIGS["meerkat_lband"]["TSAMP"]
NSAMPS_REF = 32768
NSAMPS_TICKS = [8192, 16384, 32768, 65536]
NCHANS: int = CONFIGS["meerkat_lband"]["NCHANS"]
LEVIN_TOL: float = CONFIGS["meerkat_lband"]["LEVIN_TOL"]
DM_MAX_REF: float = CONFIGS["meerkat_lband"]["DM_MAX_REF"]
NDMS_REF: int = CONFIGS["meerkat_lband"]["NDMS_REF"]
DM_MAX_SWEEP: list[float] = CONFIGS["meerkat_lband"]["DM_MAX_SWEEP"]
NBITS = [1, 2, 4, 8, 16, 32]
NBITS_LABELS = [f"{b}-bit" if b < 32 else "float32" for b in NBITS]

Theme = dict[str, Any]
# The five engines compared. FDMT-FFT and DDMT-FFT are the Fourier-domain
# engines with fractional delays (their point). Older result files may hold
# the integer-delay FDMT-FFT and brute-force DDMT-FFT, which are not drawn
# (see SUITE_ALGOS); the suite no longer measures them.
ALGOS = ["FDMT", "FDMT-FFT", "DDMT", "SDMT", "DDMT-FFT"]
# Suite benchmark name -> plotted engine (None: not plotted).
SUITE_ALGOS: dict[str, str | None] = {
    "FDMT": "FDMT",
    "FDMT-FFT-frac": "FDMT-FFT",
    "FDMT-FFT": None,
    "DDMT": "DDMT",
    "SDMT": "SDMT",
    "DDMT-FFT": "DDMT-FFT",
    "DDMT-FFT-brute": None,
}
# Validated categorical slots 1-5 (blue, orange, aqua, yellow, magenta) for
# each theme, in fixed order.
# Slot 4 sits below 3:1 on the light surface, so every line is direct-labelled.
THEMES: dict[str, Theme] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "algo": {
            "FDMT": "#2a78d6",
            "FDMT-FFT": "#eb6834",
            "DDMT": "#1baf7a",
            "SDMT": "#eda100",
            "DDMT-FFT": "#e87ba4",
        },
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "algo": {
            "FDMT": "#3987e5",
            "FDMT-FFT": "#d95926",
            "DDMT": "#199e70",
            "SDMT": "#c98500",
            "DDMT-FFT": "#d55181",
        },
    },
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class Result:
    """All repetitions of one benchmark point on one machine."""

    machine: str
    sweep: str
    algo: str
    backend: str  # cpu<N>, cuda, cuda_host, hip, hip_host
    nsamps: int
    dm_max: float
    nbits: int
    times: list[float] = field(default_factory=list)  # seconds per block
    skipped: str = ""
    ndms: int = 0  # trials of the point's grid (benchmark counter)
    # Peak memory per repetition (bytes): process peak RSS, and device
    # memory above the point's starting level on the GPU.
    peak_rss: list[float] = field(default_factory=list)
    peak_device: list[float] = field(default_factory=list)

    @property
    def time(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")

    @property
    def rtf(self) -> float:
        """Real-time factor: seconds of data processed per second of compute."""
        return self.nsamps * TSAMP / self.time

    @property
    def peak_memory(self) -> float:
        """Peak bytes of the point: device memory on a GPU, else RSS."""
        vals = self.peak_device if self.backend in GPU_KINDS else self.peak_rss
        return max(vals) if vals else float("nan")


@dataclass
class Machine:
    """One results folder and its benchmark context."""

    name: str
    context: dict[str, str] = field(default_factory=dict)
    ran_cpu: bool = False  # has CPU results (threads are meaningful)

    def platform(self, kind: str) -> str:
        """Display name of this machine's CPU or GPU."""
        ctx = self.context
        if kind in GPU_KINDS:
            return ctx.get("gpu_model") or f"{self.name} GPU"
        return ctx.get("label") or ctx.get("cpu_model") or self.name


# GPU backends; "<kind>_host" is the same engine fed from host arrays.
GPU_KINDS = ("cuda", "hip")

_UNIT = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}
_CONTEXT_KEYS = (
    "label",
    "cpu_model",
    "gpu_model",
    "dmt_version",
    "ram_gb",
    "threads",
    "build_type",
    "date",
    "placement",
    "virtualized",
    "load_start",
    "load_end",
)


def validate_geometry(path: Path, benchmark: dict[str, Any]) -> None:
    """Reject mismatched or mixed geometries before labeling a result."""
    geometry = {
        "nchans": NCHANS,
        "f_min": F_MIN,
        "f_max": F_MAX,
        "tsamp": TSAMP,
        "levin_tol": LEVIN_TOL,
    }
    for field_name, expected in geometry.items():
        if field_name in benchmark and not np.isclose(
            benchmark[field_name], expected, rtol=1e-6, atol=0
        ):
            msg = (
                f"{path}: {field_name}={benchmark[field_name]} does not match "
                f"selected preset ({expected}); rerun the suite for this preset."
            )
            raise ValueError(msg)


def load(results_dir: Path) -> tuple[list[Result], dict[str, Machine]]:  # noqa: C901
    """Parse every suite JSON; names carry the point, errors mark skips."""
    results: dict[tuple[str, str, str, str, int, float, int], Result] = {}
    machines: dict[str, Machine] = {}
    for path in sorted(results_dir.glob("*/suite_*.json")):
        if path.name.startswith("suite_cfdmt_"):
            continue  # CohFDMT has its own plots (plot_cfdmt.py)
        data = json.loads(path.read_text())
        names = [b["name"] for b in data.get("benchmarks", [])]
        if names and not any("/dmmax:" in n for n in names):
            print(f"warning: skipping {path} (an older suite configuration; rerun it)")
            continue
        mname = path.parent.name
        mach = machines.setdefault(mname, Machine(mname))
        ctx = data.get("context", {})
        for key in _CONTEXT_KEYS:
            if ctx.get(key) and ctx[key] != "-":
                mach.context.setdefault(key, ctx[key])
        for b in data.get("benchmarks", []):
            parts = b["name"].split("/")
            if b.get("run_type") == "aggregate" or len(parts) < 7:
                continue
            if parts[0] != "suite":
                continue
            validate_geometry(path, b)
            kv = dict(p.split(":", 1) for p in parts[4:7])
            algo = SUITE_ALGOS.get(parts[2], parts[2])
            if algo is None:
                continue
            key = (
                mname,
                parts[1],
                algo,
                parts[3],
                int(kv["nsamps"]),
                float(kv["dmmax"]),
                int(kv["nbits"]),
            )
            res = results.setdefault(key, Result(*key))
            mach.ran_cpu |= parts[3].startswith("cpu")
            if b.get("error_occurred"):
                res.skipped = b.get("error_message", "skipped")
                continue
            res.times.append(b["real_time"] * _UNIT[b.get("time_unit", "ns")])
            res.ndms = int(b.get("ndms", res.ndms))
            if "peak_rss_bytes" in b:
                res.peak_rss.append(float(b["peak_rss_bytes"]))
            if "peak_device_bytes" in b:
                res.peak_device.append(float(b["peak_device_bytes"]))
    versions = {m.context.get("dmt_version") for m in machines.values()} - {None}
    if len(versions) > 1:
        print(f"warning: results come from different dmt versions: {sorted(versions)}")
    return list(results.values()), machines


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def style(theme: Theme) -> None:
    """Apply the theme's surfaces, ink and hairline chrome."""
    mpl.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "axes.edgecolor": theme["axis"],
            "axes.labelcolor": theme["ink2"],
            "axes.titlecolor": theme["ink"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.color": theme["grid"],
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": theme["ink2"],
            "text.color": theme["ink"],
            "font.family": "sans-serif",
            "lines.linewidth": 1.6,
            "lines.markersize": 5.5,
        }
    )


def pow2_axis(
    ax: Axes, values: Sequence[int], label: str, ticklabels: list[str] | None = None
) -> None:
    """Log2 x axis with ticks at `values`.

    Multiples of 1024 are labelled 8K, 16K, ...; other values as they are.
    """

    def short(v: int) -> str:
        if v >= 1024 and v % 1024 == 0:
            return f"{v // 1024}K"
        return str(v)

    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.set_xticklabels(ticklabels or [short(int(v)) for v in values])
    ax.minorticks_off()
    ax.set_xlabel(label)


def ndms_ticks(results: list[Result]) -> list[int]:
    """Trial counts of the DM-range sweep (from the measured points)."""
    return sorted({r.ndms for r in results if r.sweep == "ndms" and r.ndms})


def fmt_seconds(y: float, _pos: int | None = None) -> str:
    """Format a duration in s, ms or µs with 3 significant figures."""
    if y >= 1:
        return f"{y:.3g} s"
    if y >= 1e-3:
        return f"{y * 1e3:.3g} ms"
    return f"{y * 1e6:.3g} µs"


def fmt_factor(v: float, _pos: int | None = None) -> str:
    return f"{v:g}{TIMES}"


LABEL_GAP_PT = 9.0  # minimum vertical spacing of end labels (points)


def end_label(ax: Axes, x: float, y: float, text: str) -> None:
    """Queue a label for a line's last point (placed by place_end_labels)."""
    pending = ax.__dict__.setdefault("_dmt_end_labels", [])
    pending.append((x, y, text))


def place_end_labels(fig: Figure) -> None:
    """Draw the queued end labels, in ink rather than the series colour,
    spread apart vertically where lines end close together (a short leader
    joins a moved label to its point)."""
    fig.canvas.draw()  # final limits for the data-to-display transform
    pt_per_px = 72.0 / fig.dpi
    for i, ax in enumerate(fig.axes):
        # Earlier panels on top: a label overhanging into the next panel's
        # margin is not hidden under its background.
        ax.set_zorder(len(fig.axes) - i)
        pending = ax.__dict__.pop("_dmt_end_labels", [])
        if not pending:
            continue
        # Points from the bottom, then push each label up past its neighbour.
        items = sorted(
            (
                (ax.transData.transform((x, y))[1] * pt_per_px, x, y, t)
                for x, y, t in pending
            ),
        )
        placed: list[float] = []
        for want, *_ in items:
            placed.append(want if not placed else max(want, placed[-1] + LABEL_GAP_PT))
        # Centre the stack on the points it labels.
        shift = (sum(p for p, *_ in items) - sum(placed)) / len(placed)
        for (want, x, y, text), pos in zip(items, placed, strict=True):
            dy = pos + shift - want
            ax.annotate(
                text,
                (x, y),
                xytext=(6, dy),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=mpl.rcParams["axes.labelcolor"],
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "lw": 0.5,
                        "color": "0.6",
                        "shrinkA": 0,
                        "shrinkB": 2,
                    }
                    if abs(dy) > 1.0
                    else None
                ),
            )


def plot_line(
    ax: Axes,
    theme: Theme,
    x: Sequence[float],
    y: Sequence[float],
    *,
    algo: str,
    label: str,
    variant: bool = False,
    marker: str = "o",
) -> None:
    """One series: solid and filled, or dashed and hollow for a variant."""
    color = theme["algo"][algo]
    ax.plot(
        x,
        y,
        "--" if variant else "-",
        color=color,
        marker=marker,
        markerfacecolor=theme["surface"] if variant else color,
        markeredgecolor=color,
        label=label,
        gid=f"{algo}|{'variant' if variant else 'main'}",
    )


def legend_proxies(
    theme: Theme, gids: set[str | None], *, variant_label: str, realtime: bool
) -> tuple[list[Line2D], list[str]]:
    """Legend entries for the algorithms drawn, the variant style and real time."""
    handles, labels = [], []
    for algo in ALGOS:
        if f"{algo}|main" in gids or f"{algo}|variant" in gids:
            color = theme["algo"][algo]
            handles.append(
                Line2D([], [], color=color, marker="o", markerfacecolor=color)
            )
            labels.append(algo)
    if variant_label and any(g and g.endswith("|variant") for g in gids):
        handles.append(Line2D([], [], color=theme["muted"], ls="--", marker="o",
                              markerfacecolor=theme["surface"]))  # fmt: skip
        labels.append(variant_label)
    if realtime:
        handles.append(Line2D([], [], color=theme["muted"], lw=1.0))
        labels.append("real time")
    return handles, labels


def finish(
    fig: Figure,
    axes_row: Sequence[Axes],
    theme: Theme,
    title: str,
    subtitle: str,
    *,
    legend: bool = True,
    variant_label: str = "",
    realtime: bool = True,
) -> None:
    """Shared figure chrome: centred title and subtitle, one legend below.

    The legend explains the encoding (colour = algorithm, dashed and hollow =
    `variant_label`); the per-series detail is in the direct labels.
    """
    if legend:
        gids = {ln.get_gid() for ax in axes_row for ln in ax.get_lines()}
        handles, labels = legend_proxies(theme, gids, variant_label=variant_label,
                                         realtime=realtime)  # fmt: skip
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.0))  # fmt: skip
    fig.suptitle(title, color=theme["ink"], fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.935, subtitle, ha="center", color=theme["ink2"], fontsize=9)
    bottom = 0.0
    if legend:
        bottom = 0.16 if len(axes_row) == 1 else 0.1
    fig.tight_layout(rect=(0, bottom, 1, 0.92))


# --------------------------------------------------------------------------
# Data selection
# --------------------------------------------------------------------------
def select(
    results: list[Result], machine: str, sweep: str, algo: str, backend: str
) -> list[Result]:
    """Measured (not skipped) points of one series."""
    return [
        r
        for r in results
        if r.machine == machine
        and r.sweep == sweep
        and r.algo == algo
        and r.backend == backend
        and r.times
    ]


def platforms(results: list[Result]) -> list[tuple[str, str]]:
    """Ordered (machine, "cpu" | GPU kind) panels: CPU machines, then GPUs."""
    cpu = sorted({r.machine for r in results if r.backend.startswith("cpu")})
    panels = [(m, "cpu") for m in cpu]
    for kind in GPU_KINDS:
        gpu = sorted({r.machine for r in results if r.backend.startswith(kind)})
        panels += [(m, kind) for m in gpu]
    return panels


def cpu_backends(results: list[Result], machine: str) -> list[str]:
    """Return the machine's CPU backends, fewest threads first."""
    threads = {
        int(r.backend[3:])
        for r in results
        if r.machine == machine and r.backend.startswith("cpu")
    }
    return [f"cpu{t}" for t in sorted(threads)]


def kind_name(kind: str) -> str:
    return "GPU" if kind in GPU_KINDS else "CPU"


def backend_tag(backend: str) -> str:
    if backend in GPU_KINDS:
        return ""
    if backend.endswith("_host"):
        return " · incl. PCIe"
    return f" · {backend[3:]} thr"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def new_panels(n: int) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(
        1, max(n, 1), figsize=(max(4.4 * n, 6.8), 4.4), sharey=True, squeeze=False
    )
    return fig, list(axes[0])


def runtime_figure(
    results: list[Result],
    machines: dict[str, Machine],
    theme: Theme,
    *,
    sweep: str,
    xlabel: str,
    title: str,
    subtitle: str,
) -> Figure:
    """Time per block along one sweep; one panel per CPU/GPU platform."""
    xkey = "nsamps" if sweep == "nsamps" else "ndms"
    xticks = NSAMPS_TICKS if sweep == "nsamps" else ndms_ticks(results)
    panels = platforms(results)
    fig, axes = new_panels(len(panels))
    for ax, (mname, kind) in zip(axes, panels, strict=False):
        if kind == "cpu":
            threads = cpu_backends(results, mname)
            lines = [(a, threads[-1], False) for a in ALGOS]
            lines += [("FDMT", t, True) for t in threads[:-1]]
        else:
            lines = [(a, kind, False) for a in ALGOS]
        ax.set_title(f"{machines[mname].platform(kind)} · {kind_name(kind)}")
        for algo, backend, variant in lines:
            pts = sorted(
                (getattr(r, xkey), r.time)
                for r in select(results, mname, sweep, algo, backend)
            )
            if not pts:
                continue
            x, y = zip(*pts, strict=True)
            label = f"{algo}{backend_tag(backend)}"
            plot_line(ax, theme, x, y, algo=algo, label=label, variant=variant)
            end_label(ax, x[-1], y[-1], label)
        rt_label = "real time (block duration)"
        if xkey == "nsamps":
            xs = np.array(xticks, dtype=float)
            ax.plot(xs, xs * TSAMP, "-", color=theme["muted"], lw=1.0, label=rt_label)
        else:
            ax.axhline(NSAMPS_REF * TSAMP, color=theme["muted"], lw=1.0, label=rt_label)
        pow2_axis(ax, xticks, xlabel)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_seconds))
        ax.margins(x=0.35)
    axes[0].set_ylabel("time per block (median)")
    finish(fig, axes, theme, title, subtitle, variant_label="1 thread")
    return fig


def throughput_figure(
    results: list[Result], machines: dict[str, Machine], theme: Theme
) -> Figure:
    """Real-time factor vs input width; one panel per CPU/GPU platform."""
    panels = platforms(results)
    fig, axes = new_panels(len(panels))
    for ax, (mname, kind) in zip(axes, panels, strict=False):
        if kind == "cpu":
            threads = cpu_backends(results, mname)
            lines = [
                (a, threads[-1], False)
                for a in ("FDMT", "DDMT", "SDMT", "FDMT-FFT", "DDMT-FFT")
            ]
            lines += [("FDMT", t, True) for t in threads[:-1]]
        else:
            lines = [
                ("FDMT", kind, False),
                ("DDMT", kind, False),
                ("SDMT", kind, False),
                ("FDMT-FFT", kind, False),
                ("DDMT-FFT", kind, False),
            ]
        ax.set_title(f"{machines[mname].platform(kind)} · {kind_name(kind)}")
        for algo, backend, variant in lines:
            pts = sorted(
                (NBITS.index(r.nbits), r.rtf)
                for r in select(results, mname, "nbits", algo, backend)
            )
            if not pts:
                continue
            x, y = zip(*pts, strict=True)
            label = f"{algo}{backend_tag(backend)}"
            plot_line(ax, theme, x, y, algo=algo, label=label, variant=variant)
            end_label(ax, x[-1], y[-1], label)
        ax.axhline(1.0, color=theme["muted"], lw=1.0, label=f"real time (1{TIMES})")
        ax.set_xticks(range(len(NBITS)))
        ax.set_xticklabels(NBITS_LABELS)
        ax.set_xlabel("input sample width")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_factor))
        ax.set_xlim(-0.4, len(NBITS) + 0.6)
    axes[0].set_ylabel("real-time factor (data s / compute s)")
    finish(
        fig,
        axes,
        theme,
        "Real-time factor vs input sample width",
        f"{NCHANS} channels, {NSAMPS_REF} samples {TIMES} {NDMS_REF} DM trials;"
        f" above 1{TIMES} keeps up with the telescope",
        variant_label="1 thread (CPU)",
    )
    return fig


def ops_model(dm_max: float, nchans: int, nsamps: int = NSAMPS_REF) -> dict[str, float]:
    """Operations per output time sample (summed over DM trials)."""
    from dmtlib import DDMTPlan, FDMTPlan  # noqa: PLC0415 - optional dependency

    grid = DDMTPlan.generate_levin_dm_grid_piecewise(
        0.0, dm_max, TSAMP, TSAMP, F_MIN, F_MAX, nchans, LEVIN_TOL
    )
    plan = FDMTPlan(F_MIN, F_MAX, nchans, nsamps, TSAMP, dm_grid=np.asarray(grid))
    ndms = plan.dmt_ndms
    nfft = nsamps + 2 * plan.dt_max  # valid-mode FFT length (model)
    nodes = float(sum(plan.nodes_by_iteration))
    return {
        "ndms": float(ndms),
        "FDMT": float(plan.total_operations),
        "FDMT-FFT": 2.5 * np.log2(nfft) * (nchans + ndms) + 4.0 * nodes * nfft / nsamps,
        "DDMT": float(nchans * ndms),
    }


def ops_figure(theme: Theme) -> Figure:
    """Theoretical operation count vs DM trials and vs channels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    by_dms = [ops_model(dm, NCHANS) for dm in DM_MAX_SWEEP]
    chans = sorted({256, 512, 1024, 2048, 4096, NCHANS})
    by_chans = [ops_model(DM_MAX_REF, c) for c in chans]
    labels = {
        "FDMT": "FDMT",
        "FDMT-FFT": "FDMT-FFT (model)",
        "DDMT": "DDMT (brute force)",
    }
    for ax, xs, rows, xlabel in (
        (ax1, [int(r["ndms"]) for r in by_dms], by_dms, "number of DM trials"),
        (ax2, chans, by_chans, "number of channels"),
    ):
        for algo, label in labels.items():  # SDMT's count depends on the grid
            ys = [r[algo] for r in rows]
            plot_line(ax, theme, xs, ys, algo=algo, label=label)
            end_label(ax, xs[-1], ys[-1], label)
        pow2_axis(ax, xs, xlabel)
        ax.set_yscale("log")
        ax.margins(x=0.3)
    ax1.set_title(f"{NCHANS} channels, DM 0-{DM_MAX_SWEEP[0]:g} to 0-{DM_MAX_REF:g}")
    ax2.set_title(f"DM 0-{DM_MAX_REF:g}, vary channels")
    ax1.set_ylabel("operations per output time sample")
    finish(
        fig,
        [ax1],
        theme,
        "Operation count (theory)",
        "operations per output time sample, summed over all DM trials",
        legend=False,
    )
    ax1.legend(loc="upper left")
    return fig


def highlight_figure(
    results: list[Result], machines: dict[str, Machine], theme: Theme
) -> Figure:
    """README: runtime vs DM trials (left) and FDMT real-time factor (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    platform_handles: list[Line2D] = []
    platform_labels: list[str] = []
    for i, (mname, kind) in enumerate(platforms(results)):
        marker = MARKERS[i % len(MARKERS)]
        backend = cpu_backends(results, mname)[-1] if kind == "cpu" else kind
        plat = machines[mname].platform(kind)
        if kind == "cpu":
            plat += f" · {backend[3:]} thr"
        platform_handles.append(
            Line2D([], [], color=theme["muted"], marker=marker, ls="none")
        )
        platform_labels.append(plat)
        for algo in ALGOS:
            pts = sorted(
                (r.ndms, r.time) for r in select(results, mname, "ndms", algo, backend)
            )
            if pts:
                x, y = zip(*pts, strict=True)
                plot_line(
                    ax1, theme, x, y, algo=algo, label=f"{algo} · {plat}", marker=marker
                )
        pts = sorted(
            (NBITS.index(r.nbits), r.rtf)
            for r in select(results, mname, "nbits", "FDMT", backend)
        )
        if pts:
            x, y = zip(*pts, strict=True)
            plot_line(
                ax2, theme, x, y, algo="FDMT", label=f"FDMT · {plat}", marker=marker
            )
            end_label(ax2, x[-1], y[-1], plat)
    ax1.axhline(NSAMPS_REF * TSAMP, color=theme["muted"], lw=1.0)
    ticks = ndms_ticks(results)
    ax1.annotate(
        "real time",
        (ticks[-1] if ticks else NDMS_REF, NSAMPS_REF * TSAMP),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=theme["muted"],
        ha="right",
    )
    pow2_axis(ax1, ticks, "number of DM trials")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_seconds))
    ax1.set_ylabel(f"time per {NSAMPS_REF // 1024}K-sample block")
    ax1.set_title(f"{NCHANS} channels, {NSAMPS_REF * TSAMP:.1f} s blocks")
    ax2.axhline(1.0, color=theme["muted"], lw=1.0)
    ax2.set_xticks(range(len(NBITS)))
    ax2.set_xticklabels(NBITS_LABELS)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt_factor))
    ax2.set_xlim(-0.4, len(NBITS) + 1.2)
    ax2.set_ylabel("FDMT real-time factor")
    ax2.set_title(f"FDMT throughput, {NDMS_REF} DM trials")
    ax2.annotate(
        "real time",
        (len(NBITS) - 1, 1.0),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=theme["muted"],
        ha="right",
    )
    gids = {ln.get_gid() for ln in ax1.get_lines()}
    handles, labels = legend_proxies(theme, gids, variant_label="", realtime=False)
    fig.legend(
        handles + platform_handles,
        labels + platform_labels,
        loc="lower center",
        ncol=len(labels) + len(platform_labels),
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


# --------------------------------------------------------------------------
# Summary table (included by docs/benchmarks.md)
# --------------------------------------------------------------------------
def _machines_table(machines: dict[str, Machine]) -> list[str]:
    out = [
        "**Machines**",
        "",
        "| results | CPU | GPU | RAM | CPU threads | placement | load (1/5/15 min) "
        "| dmt |",
        "| :--- | :--- | :--- | ---: | :--- | :--- | :--- | :--- |",
    ]
    for m in sorted(machines.values(), key=lambda m: m.name):
        c = m.context
        threads = c.get("threads", "") if m.ran_cpu else "—"
        placement = c.get("placement", "close") if m.ran_cpu else "—"
        if m.ran_cpu and c.get("virtualized", "no") not in ("no", "-"):
            placement += f", {c['virtualized']} VM"
        load = c.get("load_start", "—")
        if c.get("load_end"):
            load += f" → {c['load_end']}"
        out.append(
            f"| `{m.name}` | {c.get('cpu_model', '')} "
            f"| {c.get('gpu_model', '') or '—'} | {c.get('ram_gb', '')} GB "
            f"| {threads} | {placement} | {load} | {c.get('dmt_version', '')} |"
        )
    return out


def _reference_table(results: list[Result], machines: dict[str, Machine]) -> list[str]:
    out = [
        "",
        (
            f"**Reference point**: {NSAMPS_REF} samples {TIMES} {NDMS_REF} DM "
            f"trials (DM 0-{DM_MAX_REF:g}, {NSAMPS_REF * TSAMP:.1f} s of data), "
            "float32 input, median "
            "time per block. FDMT-FFT and DDMT-FFT use fractional delays "
            "(DDMT-FFT: NUFFT over the DM axis)."
        ),
        "",
        (
            "| platform | "
            + " | ".join(ALGOS)
            + " | DDMT / FDMT | SDMT / FDMT | FDMT-FFT / FDMT "
            "| FDMT real-time factor |"
        ),
        "| :--- | " + " | ".join("---:" for _ in range(len(ALGOS) + 4)) + " |",
    ]
    ref = {
        (r.machine, r.backend, r.algo): r
        for r in results
        if r.sweep == "ndms" and r.dm_max == DM_MAX_REF and r.times
    }
    for mname, kind in platforms(results):
        backends = cpu_backends(results, mname) if kind == "cpu" else [kind]
        for b in backends:
            t = {a: ref.get((mname, b, a)) for a in ALGOS}
            fdmt = t["FDMT"]
            if fdmt is None:
                continue
            cells = [fmt_seconds(t[a].time) if t[a] else "—" for a in ALGOS]
            ratios = [
                fmt_ratio(t[a].time / fdmt.time) if t[a] else "—"
                for a in ("DDMT", "SDMT", "FDMT-FFT")
            ]
            plat = machines[mname].platform(kind) + backend_tag(b)
            out.append(
                f"| {plat} | {' | '.join(cells)} | {' | '.join(ratios)} "
                f"| {fdmt.rtf:.0f}{TIMES} |"
            )
    return out


GIB = 1024.0**3


def fmt_gib(b: float) -> str:
    if math.isnan(b):  # not measured
        return "—"
    return f"{b / GIB:.2f} GiB" if b < 10 * GIB else f"{b / GIB:.1f} GiB"


def _memory_table(results: list[Result], machines: dict[str, Machine]) -> list[str]:
    ref = {
        (r.machine, r.backend, r.algo): r
        for r in results
        if r.sweep == "ndms" and r.dm_max == DM_MAX_REF and r.times
    }
    if not any(r.peak_rss or r.peak_device for r in ref.values()):
        return []
    out = [
        "",
        (
            "**Peak memory at the reference point** (float32 input): process "
            "peak resident set size on the CPU (input, output, engine and the "
            "benchmark process itself), device memory in use on the GPU "
            "(input, output and engine; sampled every millisecond). Measured "
            "in the same runs as the times above."
        ),
        "",
        "| platform | " + " | ".join(ALGOS) + " |",
        "| :--- | " + " | ".join("---:" for _ in ALGOS) + " |",
    ]
    for mname, kind in platforms(results):
        backends = cpu_backends(results, mname) if kind == "cpu" else [kind]
        for b in backends:
            t = {a: ref.get((mname, b, a)) for a in ALGOS}
            if t["FDMT"] is None:
                continue
            cells = [fmt_gib(t[a].peak_memory) if t[a] else "—" for a in ALGOS]
            plat = machines[mname].platform(kind) + backend_tag(b)
            out.append(f"| {plat} | {' | '.join(cells)} |")
    return out


def load_boxcar(results_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Per-iteration rows of the boxcar suite JSON (run_suite.py --suite boxcar)."""
    rows = []
    for path in sorted(results_dir.glob("*/boxcar_*.json")):
        data = json.loads(path.read_text())
        rows += [
            (path.parent.name, b)
            for b in data.get("benchmarks", [])
            if b.get("run_type") == "iteration" and not b.get("error_occurred")
        ]
    return rows


def _boxcar_table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """FDMTBoxcar at the reference point, one line per benchmark."""
    if not rows:
        return []
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for mname, b in rows:
        name = b["name"].split("/real_time")[0].split("/min_warmup_time")[0]
        by[(mname, name)].append(b)

    def med(bs: list[dict[str, Any]], key: str) -> float:
        vals = [float(b[key]) for b in bs if key in b]
        return statistics.median(vals) if vals else float("nan")

    out = [
        "",
        (
            f"**FDMTBoxcar** (`run_suite.py --suite boxcar`): {NSAMPS_REF} "
            f"samples {TIMES} {NDMS_REF} DM trials, boxcars to 100 ms, "
            "threshold 6, median time per block. *allocated*: the library's "
            "own accounting plus the input block; *peak*: process peak RSS "
            "(CPU) or device memory in use (GPU), measured in the same run."
        ),
        "",
        "| benchmark | time per block | real-time factor | allocated | peak |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for (_mname, name), bs in sorted(by.items()):
        parts = name.split("/")
        label = "/".join(p for p in parts[2:] if p)
        backend = parts[0].removeprefix("BM_fdmt_boxcar_")
        label = f"{'cpu' if backend == 'stream' else backend}: {label}"
        t = med(bs, "real_time") * _UNIT[bs[0].get("time_unit", "ns")]
        gpu = "peak_device_bytes" in bs[0]
        peak = max(
            float(b.get("peak_device_bytes" if gpu else "peak_rss_bytes", "nan"))
            for b in bs
        )
        out.append(
            f"| `{label}` | {fmt_seconds(t)} | {med(bs, 'realtime_factor'):.0f}"
            f"{TIMES} | {fmt_gib(med(bs, 'working_bytes'))} | {fmt_gib(peak)} |"
        )
    return out


def fmt_ratio(r: float) -> str:
    return f"{r:.1f}{TIMES}" if r < 10 else f"{r:.0f}{TIMES}"


def _nbits_table(results: list[Result], machines: dict[str, Machine]) -> list[str]:
    out = [
        "",
        (
            "**Real-time factor by input width** (data seconds per compute "
            f"second, {NSAMPS_REF} samples {TIMES} {NDMS_REF} DM trials):"
        ),
        "",
        "| platform | engine | " + " | ".join(NBITS_LABELS) + " |",
        "| :--- | :--- | " + " | ".join("---:" for _ in NBITS) + " |",
    ]
    by: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for r in results:
        if r.sweep == "nbits" and r.times:
            by[(r.machine, r.backend, r.algo)][r.nbits] = r.rtf
    engines = {"cuda": "device-resident", "hip": "device-resident"}
    for mname, kind in platforms(results):
        backends = cpu_backends(results, mname) if kind == "cpu" else [kind]
        for b in backends:
            for a in ("FDMT", "DDMT", "SDMT", "FDMT-FFT", "DDMT-FFT"):
                row = by.get((mname, b, a))
                if not row:
                    continue
                n = b[3:]
                eng = engines.get(b, f"{n} thread{'' if n == '1' else 's'}")
                cells = [f"{row[x]:.1f}{TIMES}" if x in row else "—" for x in NBITS]
                out.append(
                    f"| {machines[mname].platform(kind)} | {a}, {eng} "
                    f"| {' | '.join(cells)} |"
                )
    return out


def _skipped_list(results: list[Result]) -> list[str]:
    skipped = sorted(
        (r for r in results if r.skipped and not r.times),
        key=lambda r: (r.machine, r.sweep, r.algo, r.nsamps, r.dm_max),
    )
    if not skipped:
        return []
    out = ["", "**Skipped points** (over the memory budget):", ""]
    out += [
        f"- `{r.machine}` {r.algo} {r.backend}, {r.nsamps} samples {TIMES} "
        f"DM 0-{r.dm_max:g}: {r.skipped}"
        for r in skipped
    ]
    return out


def summary_md(
    results: list[Result],
    machines: dict[str, Machine],
    boxcar: list[tuple[str, dict[str, Any]]] | None = None,
) -> str:
    out = ["<!-- generated by bench/scripts/plot_suite.py; do not edit -->", ""]
    out += _machines_table(machines)
    out += _reference_table(results, machines)
    out += _memory_table(results, machines)
    out += _nbits_table(results, machines)
    out += _boxcar_table(boxcar or [])
    out += _skipped_list(results)
    return "\n".join([*out, ""])


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the dmt benchmark suite.")
    ap.add_argument(
        "--config",
        default="meerkat_lband",
        choices=sorted(c for c in CONFIGS if c != "parkes_uwl"),
        help="suite configuration (default meerkat_lband); reads "
        "bench/results/<config>/*/suite_*.json and writes "
        "bench/results/<config>/plots/ unless --results is given",
    )
    ap.add_argument("--results", help="results root (default: bench/results/<config>)")
    ap.add_argument("--no-ops", action="store_true", help="skip the dmtlib ops plot")
    args = ap.parse_args()
    globals().update(CONFIGS[args.config])
    default = REPO / "bench" / "results" / args.config
    results_dir = Path(args.results) if args.results else default
    try:
        results, machines = load(results_dir)
    except ValueError as exc:
        ap.error(str(exc))
    # GPU timings are device-resident (no host<->device copies); the suite's
    # host-array series are not plotted.
    results = [r for r in results if not r.backend.endswith("_host")]
    if not results:
        sys.exit(f"no suite results under {results_dir}/*/suite_*.json")
    out_root = results_dir / "plots"
    rt_note = "below the grey line is faster than real time"
    for name, theme in THEMES.items():
        style(theme)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        figs = {
            "runtime_vs_nsamps": runtime_figure(
                results,
                machines,
                theme,
                sweep="nsamps",
                xlabel="samples per block",
                title="Time per block vs block length",
                subtitle=f"{NCHANS} channels, {NDMS_REF} DM trials "
                f"(DM 0-{DM_MAX_REF:g}), float32; {rt_note}",
            ),
            "runtime_vs_ndms": runtime_figure(
                results,
                machines,
                theme,
                sweep="ndms",
                xlabel="number of DM trials",
                title="Time per block vs number of DM trials",
                subtitle=f"{NCHANS} channels, {NSAMPS_REF} samples "
                f"({NSAMPS_REF * TSAMP:.1f} s), DM 0-{DM_MAX_SWEEP[0]:g} to "
                f"0-{DM_MAX_REF:g}, float32; {rt_note}",
            ),
            "throughput_nbits": throughput_figure(results, machines, theme),
            "readme_highlight": highlight_figure(results, machines, theme),
        }
        if not args.no_ops:
            try:
                figs["ops_theory"] = ops_figure(theme)
            except ImportError:
                print("dmtlib not importable; skipping ops_theory (use --no-ops)")
        for fname, fig in figs.items():
            place_end_labels(fig)
            fig.savefig(out / f"{fname}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    (out_root / "summary.md").write_text(
        summary_md(results, machines, load_boxcar(results_dir))
    )
    print(
        f"wrote {len(figs)} figures {TIMES} {len(THEMES)} themes and summary.md "
        f"to {out_root}"
    )


if __name__ == "__main__":
    main()
