import json
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

filename = sys.argv[1]
basename = filename.split(".")[0]

# load benchmarks data
with Path(filename).open("r") as jfile:
    jdata = json.load(jfile)
bench_df = pd.DataFrame(jdata["benchmarks"])

# only keep names with /, i.e. drop avg/rms/etc entries
bench_df = bench_df.loc[bench_df.name.str.contains("/")]

# create a column with complexity n
bench_df[["benchmark_name", "benchmark_type", "n"]] = bench_df.name.str.split(
    "/BM_|_|/",
    expand=True,
).apply(
    lambda x: ["_".join([x[1], x[2]]), x[3], x[4]],
    axis=1,
    result_type="expand",
)
bench_df["n"] = bench_df["n"].astype("uint32")
bench_df = bench_df[["benchmark_name", "n", "cpu_time", "benchmark_type"]]
benchmarks = bench_df.benchmark_name.unique()

palette = sns.color_palette("husl", len(benchmarks))
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for i, benchmark in enumerate(benchmarks):
    ax = axes[0, i] if i < 2 else axes[1, i - 2]
    data = bench_df[bench_df["benchmark_name"] == benchmark]
    sns.lineplot(
        x="n",
        y="cpu_time",
        hue="benchmark_type",
        data=data,
        ax=ax,
        color=palette[i],
        marker="o",
        markersize=8,
    )
    ax.set_title(benchmark)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nsamples")
    ax.set_ylabel("Time (ms)")
    ax.legend()

fig.tight_layout()
fig.savefig(f"{basename}.png", bbox_inches="tight", dpi=120)
