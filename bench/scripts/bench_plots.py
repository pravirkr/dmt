"""Plot dmt_bench / dmt_bench_mem JSON output, comparing FDMT / FDMT-FFT / DDMT.

Usage:
    python bench_plots.py <timing.json> [memory.json]

`timing.json` is the --benchmark_out of the `dmt_bench` executable (FDMT,
FDMT-FFT, DDMT timing benchmarks, CPU and/or CUDA). The optional
`memory.json` is the --benchmark_out of `dmt_bench_mem` (PeakHeap_MB /
ProcessPeakRSS_MB / TotalAlloc_MB_per_iter counters).

Benchmark names are parsed structurally rather than by a fixed positional
schema, since the different fixtures (FDMT/FDMT-FFT/DDMT, CPU/CUDA,
float/packed, threaded/multi-beam sweeps, ...) each register a different
number and meaning of ArgsProduct() columns:
  - the "algorithm family" (FDMT / FDMT-FFT / DDMT) is read off the fixture
    name prefix;
  - the swept sample count is taken as the largest numeric token in the
    name, since nsamps (>= 1024 everywhere in these benchmarks) is always
    larger than any other swept parameter (nbits <= 16, nthreads <= 16,
    nbeams <= 4);
  - any other numeric tokens are kept as a compact "[a, b, ...]" suffix on
    the legend label so distinct configs of the same benchmark (e.g.
    different thread counts) still appear as separate, identifiable lines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FAMILY_PALETTE = {
    "FDMT": "#4C72B0",
    "FDMT-FFT": "#DD8452",
    "DDMT": "#55A868",
}

# smallest nsamps swept anywhere; larger than nbits/threads/beams
MIN_NSAMPS_TOKEN = 256


def classify_family(fixture: str) -> str:
    if fixture.startswith("DDMT"):
        return "DDMT"
    if fixture.startswith("FDMTFFT"):
        return "FDMT-FFT"
    if fixture.startswith("FDMT"):
        return "FDMT"
    return fixture


def parse_benchmark_name(name: str) -> dict | None:
    parts = name.split("/")
    if len(parts) < 2:
        return None
    fixture = parts[0]
    bm_name = parts[1].removeprefix("BM_")

    numeric_tokens: list[int] = []
    for token in parts[2:]:
        if token.startswith(("iterations:", "repeats:", "threads:")):
            continue
        if token in ("process_time", "real_time", "cpu_time", "manual_time"):
            continue
        try:
            numeric_tokens.append(int(token))
        except ValueError:
            continue

    if not numeric_tokens:
        return None
    nsamps_candidates = [t for t in numeric_tokens if t >= MIN_NSAMPS_TOKEN]
    if not nsamps_candidates:
        return None
    nsamps = max(nsamps_candidates)
    other = sorted(t for t in numeric_tokens if t != nsamps)

    family = classify_family(fixture)
    label = f"{family}: {bm_name}"
    if other:
        label += f" {other}"
    return {"family": family, "label": label, "nsamps": nsamps}


def load_benchmarks(path: str) -> pd.DataFrame:
    with Path(path).open("r") as jfile:
        jdata = json.load(jfile)
    rows = []
    for entry in jdata.get("benchmarks", []):
        if entry.get("run_type", "iteration") != "iteration":
            continue
        parsed = parse_benchmark_name(entry["name"])
        if parsed is None:
            continue
        rows.append({**parsed, **entry})
    return pd.DataFrame(rows)


def family_palette_for(labels: list[str]) -> dict[str, tuple[float, float, float]]:
    families = sorted({lbl.split(":")[0] for lbl in labels})
    base = {f: FAMILY_PALETTE.get(f, "#888888") for f in families}
    # Give distinct variants of the same family related but distinguishable
    # shades by blending toward white in registration order.
    palette = {}
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for lbl in labels:
        fam = lbl.split(":")[0]
        totals[fam] = totals.get(fam, 0) + 1
    for lbl in sorted(labels):
        fam = lbl.split(":")[0]
        counts[fam] = counts.get(fam, 0) + 1
        n = totals[fam]
        frac = 0.0 if n <= 1 else (counts[fam] - 1) / (n - 1)
        r, g, b = sns.color_palette([base[fam]])[0]
        blend = 0.6 * frac
        palette[lbl] = (r + (1 - r) * blend, g + (1 - g) * blend, b + (1 - b) * blend)
    return palette


def plot_time_comparison(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        print(f"No timing rows to plot for {out_path}, skipping.")
        return
    df = df.sort_values(["label", "nsamps"])
    palette = family_palette_for(df["label"].unique())

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=df,
        x="nsamps",
        y="real_time",
        hue="label",
        palette=palette,
        marker="o",
        markersize=6,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input samples (nsamps)")
    ax.set_ylabel("Wall-clock time (ns)")
    ax.set_title("DDMT vs FDMT vs FDMT-FFT: execution time")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_throughput_comparison(df: pd.DataFrame, out_path: str) -> None:
    df = (
        df[df["bytes_per_second"].notna()]
        if "bytes_per_second" in df.columns
        else df.iloc[0:0]
    )
    if df.empty:
        print(
            f"No throughput data (bytes_per_second) to plot for {out_path}, skipping."
        )
        return
    df = df.copy()
    df["GB_per_s"] = df["bytes_per_second"] / 1e9
    df = df.sort_values(["label", "nsamps"])
    palette = family_palette_for(df["label"].unique())

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.lineplot(
        data=df,
        x="nsamps",
        y="GB_per_s",
        hue="label",
        palette=palette,
        marker="o",
        markersize=6,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Input samples (nsamps)")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title("DDMT vs FDMT vs FDMT-FFT: throughput")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_memory_comparison(df: pd.DataFrame, out_path: str) -> None:
    counters = [
        c
        for c in ("PeakHeap_MB", "ProcessPeakRSS_MB", "TotalAlloc_MB_per_iter")
        if c in df.columns
    ]
    if df.empty or not counters:
        print(f"No memory counters to plot for {out_path}, skipping.")
        return
    df = df.sort_values(["label", "nsamps"])
    palette = family_palette_for(df["label"].unique())

    fig, axes = plt.subplots(
        1, len(counters), figsize=(6 * len(counters), 5.5), squeeze=False
    )
    for ax, counter in zip(axes[0], counters, strict=False):
        sns.lineplot(
            data=df,
            x="nsamps",
            y=counter,
            hue="label",
            palette=palette,
            marker="o",
            markersize=6,
            ax=ax,
            legend=(counter == counters[0]),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Input samples (nsamps)")
        ax.set_ylabel(counter.replace("_", " "))
    axes[0][0].legend(
        fontsize=8, loc="upper left", bbox_to_anchor=(-0.05, -0.25), ncol=2
    )
    fig.suptitle("DDMT vs FDMT vs FDMT-FFT: memory footprint")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    timing_path = sys.argv[1]
    basename = Path(timing_path).stem

    timing_df = load_benchmarks(timing_path)
    plot_time_comparison(timing_df, f"{basename}_time.png")
    plot_throughput_comparison(timing_df, f"{basename}_throughput.png")

    if len(sys.argv) >= 3:
        mem_path = sys.argv[2]
        mem_basename = Path(mem_path).stem
        mem_df = load_benchmarks(mem_path)
        plot_memory_comparison(mem_df, f"{mem_basename}_memory.png")


if __name__ == "__main__":
    main()
