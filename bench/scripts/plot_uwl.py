"""Plot the parkes_uwl search benchmarks (run_suite.py --suite boxcar --config parkes_uwl).

Reads ``bench/results/parkes_uwl/*/boxcar_*.json`` and writes

    bench/results/parkes_uwl/plots/{light,dark}/uwl_realtime.png
    bench/results/parkes_uwl/plots/summary.md

The figure is the real-time sweep (``BM_fdmt_boxcar_rt``): FDMTBoxcar at
subband depth 4 on the 6656-channel, 704-4032 MHz band, DM 0-2000, at the
native 64 us and downsampled to 256, 512 and 1024 us, on 2.097 s blocks,
as a real-time factor (block duration / time per block; above 1 keeps up).
The summary also lists the reference-point FDMTBoxcar runs at 64 us.

    python bench/scripts/plot_uwl.py [--results bench/results/parkes_uwl]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_suite import THEMES, style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RT = re.compile(
    r"^BM_fdmt_boxcar_rt/parkes_uwl/tsamp_us:(\d+)/(uint8|float32)/sb:(\d+)/([a-z]+\d*)"
)
STREAM = re.compile(
    r"^BM_fdmt_boxcar_(?:stream/parkes_uwl/(uint8/)?wtsp:([\d.]+)/(cpu\d+)"
    r"|cuda/parkes_uwl/(fused|fused_uint8)/wtsp:([\d.]+)/sb:(\d+))"
)
GIB = 1024.0**3


def _median(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return statistics.median(vals) if vals else float("nan")


def load(results_dir: Path) -> tuple[dict, dict, dict]:
    """Real-time and reference points by key, and the machine contexts."""
    rt: dict[tuple, list[dict]] = defaultdict(list)
    ref: dict[tuple, list[dict]] = defaultdict(list)
    contexts: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*/boxcar_*.json")):
        data = json.loads(path.read_text())
        contexts[path.parent.name] = data.get("context", {})
        for b in data.get("benchmarks", []):
            if b.get("run_type") != "iteration" or b.get("error_occurred"):
                continue
            name = b["name"]
            if m := RT.match(name):
                key = (int(m[1]), m[2], m[4])
                rt[key].append(b)
            elif m := STREAM.match(name):
                if m[3]:
                    key = (m[3], "uint8" if m[1] else "float32", float(m[2]), 4)
                else:
                    nbits = "uint8" if m[4] == "fused_uint8" else "float32"
                    key = ("gpu", nbits, float(m[5]), int(m[6]))
                ref[key].append(b)
    return rt, ref, contexts


def summarize(rows: list[dict]) -> dict[str, float]:
    return {
        "ms": _median(rows, "real_time"),
        "rtf": _median(rows, "realtime_factor"),
        "mem": _median(rows, "working_bytes") / GIB,
        "engine": _median(rows, "engine_bytes") / GIB,
        "chunk": _median(rows, "chunk_nsamps"),
        "ndms": _median(rows, "ndms"),
        "rows": _median(rows, "search_rows"),
        "widths": _median(rows, "nwidths"),
        "block": _median(rows, "block_s"),
    }


def figure(rt: dict, theme: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    series = sorted(
        {(k[2], k[1]) for k in rt},
        key=lambda s: (s[0].startswith("cpu"), s[1] != "uint8"),
    )
    # Backend picks the hue (GPU blue, CPU green); bit depth the line style.
    hue = {"gpu": theme["algo"]["FDMT"], "cpu": theme["algo"]["DDMT"]}
    tsamps = sorted({k[0] for k in rt})
    for backend, nbits in series:
        pts = sorted(
            (t, summarize(rt[(t, nbits, backend)])["rtf"])
            for t in tsamps
            if (t, nbits, backend) in rt
        )
        if not pts:
            continue
        kind = "cpu" if backend.startswith("cpu") else "gpu"
        label = (
            f"{'GPU' if kind == 'gpu' else 'CPU, ' + backend[3:] + ' threads'}, {nbits}"
        )
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=hue[kind],
            linestyle="-" if nbits == "uint8" else "--",
            marker="o" if nbits == "uint8" else "s",
            markersize=7,
            linewidth=2,
            label=label,
        )
    ax.axhline(1.0, color=theme["muted"], linewidth=1.2, zorder=0)
    ax.text(
        tsamps[-1],
        1.0,
        "real time ",
        color=theme["ink2"],
        va="bottom",
        ha="right",
        fontsize=9,
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(tsamps, [f"{t} µs" for t in tsamps])
    ax.minorticks_off()
    ax.set_xlabel("search time resolution")
    ax.set_ylabel("real-time factor (block duration / time)")
    fig.suptitle(
        "parkes_uwl search, subband depth 4",
        x=0.02,
        ha="left",
        fontsize=13,
        color=theme["ink"],
    )
    ax.set_title(
        "6656 channels, 704-4032 MHz, DM 0-2000, boxcars to 100 ms; "
        "above the line keeps up",
        loc="left",
        fontsize=9,
        color=theme["ink2"],
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return fig


def summary_md(rt: dict, ref: dict, contexts: dict) -> str:
    out = ["<!-- generated by bench/scripts/plot_uwl.py; do not edit -->", ""]
    for machine, ctx in contexts.items():
        out.append(
            f"Machine `{machine}`: {ctx.get('cpu_model', '-')}, "
            f"{ctx.get('gpu_model', '-')}; CPU threads one per CCD "
            f"(OMP_PLACES `{{0}}:8:8`). Default memory budget (10 GiB)."
        )
    out += [
        "",
        "### Real-time sweep (FDMTBoxcar, subband depth 4, 2.097 s blocks)",
        "",
        "| time resolution | backend | input | DM trials | searched rows | "
        "widths | chunk | time per block | real-time factor | working memory |",
        "|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(rt, key=lambda k: (k[0], k[2] != "cuda", k[1] != "uint8")):
        t, nbits, backend = key
        s = summarize(rt[key])
        be = "GPU" if not backend.startswith("cpu") else f"CPU ({backend[3:]} thr.)"
        out.append(
            f"| {t} µs | {be} | {nbits} | {s['ndms']:.0f} | {s['rows']:.0f} | "
            f"{s['widths']:.0f} | {s['chunk']:.0f} | {s['ms'] / 1e3:.3f} s | "
            f"{s['rtf']:.2f}{chr(215)} | {s['mem']:.2f} GiB |"
        )
    out += [
        "",
        "### Reference point (64 µs, 32768-sample blocks, DM 0-2000)",
        "",
        "| backend | input | wtsp | subband depth | chunk | time per block | "
        "real-time factor | working memory |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(ref, key=lambda k: (k[0] != "gpu", k[0], k[3], k[1], k[2])):
        be, nbits, wtsp, depth = key
        s = summarize(ref[key])
        be_s = "GPU" if be == "gpu" else f"CPU ({be[3:]} thr.)"
        out.append(
            f"| {be_s} | {nbits} | {wtsp:g} | {depth} | {s['chunk']:.0f} | "
            f"{s['ms'] / 1e3:.3f} s | {s['rtf']:.2f}{chr(215)} | {s['mem']:.2f} GiB |"
        )
    return "\n".join([*out, ""])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=str(REPO / "bench" / "results" / "parkes_uwl"))
    args = ap.parse_args()
    results_dir = Path(args.results)
    rt, ref, contexts = load(results_dir)
    if not rt and not ref:
        sys.exit(f"no parkes_uwl results under {results_dir}/*/boxcar_*.json")
    out_root = results_dir / "plots"
    for name, theme in THEMES.items():
        style(theme)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        if rt:
            fig = figure(rt, theme)
            fig.savefig(out / "uwl_realtime.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    (out_root / "summary.md").write_text(summary_md(rt, ref, contexts))
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
