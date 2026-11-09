"""Plot the S/N-recovery results written by run_sensitivity.py.

Reads ``bench/results/sensitivity/<config>.npz`` and writes, for a light and a
dark theme,

    bench/results/plots/{light,dark}/sensitivity_eta_vs_dm.png
    bench/results/plots/{light,dark}/sensitivity_eta_mf_vs_dm.png
    bench/results/plots/{light,dark}/sensitivity_eta_vs_width.png
    bench/results/plots/{light,dark}/sensitivity_threshold_vs_dm.png
    bench/results/plots/{light,dark}/sensitivity_completeness.png
    bench/results/plots/{light,dark}/sensitivity_snr50_vs_dm.png

plus ``bench/results/plots/sensitivity_summary.md``, which the docs include.
With ``--paper`` it also writes light-theme PDFs sized for MNRAS (two
columns, 6.97 in) to ``paper/figures/`` and a LaTeX table to
``paper/tables/sensitivity.tex`` (``paper/`` is git-ignored).

    python bench/scripts/plot_sensitivity.py [--paper]

Every curve is the median over the injected sub-sample phases at one DM, for
the intrinsic widths W = 1, 8 and 64 samples. SDMT computes exactly the DDMT
sums (checked), so the two share one curve.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path = [
    f for f in sys.meta_path if "ScikitBuildRedirectingFinder" not in type(f).__name__
]
sys.path.insert(0, str(REPO / "src"))

from plot_suite import THEMES, Theme, style  # noqa: E402

from dmtlib.sensitivity import RecoveryResult, channel_sweep  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

PLOTTED = ["FDMT", "FDMT-FFT", "DDMT", "DDMT-FFT"]
# Variants drawn in their parent engine's colour with their own line style.
VARIANTS = {"FDMT-nobox": ("FDMT", "-.")}
LABELS = {
    "DDMT": "DDMT / SDMT",
    "FDMT": "FDMT (box smearing)",
    "FDMT-nobox": "FDMT, no box smearing",
}
TABULATED = ["FDMT", "FDMT-FFT", "DDMT", "DDMT-FFT", "FDMT-nobox"]
WIDTHS = (1.0, 8.0, 64.0)
MNRAS_PAGE = 6.97
SHOW_TITLES = True  # off for paper figures (the caption carries the title)


def width_label(w: float) -> str:
    if w == 0:
        return "W = 0 (delta)"
    return f"W = {w:g} sample{'s' if w != 1 else ''}"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def by_dm(
    res: RecoveryResult, values: np.ndarray, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """(DM, median over phases) at each injected DM, for one width."""
    sel = res.width == width
    dms = np.unique(res.dm[sel])
    return dms, np.array([np.median(values[sel & (res.dm == d)]) for d in dms])


def textbook(res: RecoveryResult, dm: np.ndarray, width: float) -> np.ndarray:
    """sqrt(W / W_eff), W_eff = sqrt(W^2 + t_smear^2) at the centre channel."""
    c = res.config
    mid = c["nchans"] // 2
    smear = np.array(
        [
            channel_sweep(c["f_min"], c["f_max"], c["nchans"], c["tsamp"], d)[1][mid]
            for d in dm
        ]
    )
    w = max(width, 1.0)
    return np.sqrt(w / np.sqrt(w**2 + smear**2))


def snr50(res: RecoveryResult, name: str, select: np.ndarray) -> float:
    """Injected S/N at 50% completeness (log-interpolated)."""
    curve = res.completeness(name, select)
    grid = res.snr_grid
    if curve[-1] < 0.5:
        return np.nan
    k = int(np.argmax(curve >= 0.5))
    if k == 0:
        return float(grid[0])
    x0, x1 = np.log(grid[k - 1]), np.log(grid[k])
    y0, y1 = curve[k - 1], curve[k]
    return float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / max(y1 - y0, 1e-9)))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def draw(ax: Axes, theme: Theme, x: np.ndarray, y: np.ndarray, name: str) -> None:
    if name in VARIANTS:
        parent, ls = VARIANTS[name]
        ax.plot(x, y, color=theme["algo"][parent], ls=ls, lw=1.2)
    else:
        ax.plot(x, y, color=theme["algo"][name], lw=1.5)


def engine_handles(
    theme: Theme, *, extras: Sequence[tuple[str, str, str]] = ()
) -> tuple[list[Line2D], list[str]]:
    handles = [Line2D([], [], color=theme["algo"][a], lw=1.5) for a in PLOTTED]
    labels = [LABELS.get(a, a) for a in PLOTTED]
    for name, (parent, ls) in VARIANTS.items():
        handles.append(Line2D([], [], color=theme["algo"][parent], ls=ls, lw=1.2))
        labels.append(LABELS[name])
    for color, ls, label in extras:
        handles.append(Line2D([], [], color=theme[color], ls=ls, lw=1.2))
        labels.append(label)
    return handles, labels


def finish(
    fig: Figure,
    theme: Theme,
    title: str,
    handles: list,
    labels: list,
    *,
    ncol: int = 4,
    bottom: float = 0.1,
) -> None:
    if not SHOW_TITLES:
        title = ""
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=8,
    )
    if title:
        fig.suptitle(title, color=theme["ink"], fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, bottom, 1, 0.96 if title else 1))


def log_y(ax: Axes) -> None:
    """Log y axis labelled at 1-2-5 decades and their minor ticks."""
    ax.set_yscale("log")
    fmt = mpl.ticker.FormatStrFormatter("%g")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(subs=(0.2, 0.3, 0.5, 0.7)))
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="minor", labelsize=7)


def smear_axis(ax: Axes, res: RecoveryResult) -> None:
    """Top axis: bottom-channel smear in samples (linear in DM)."""
    c = res.config
    per_dm = channel_sweep(c["f_min"], c["f_max"], c["nchans"], c["tsamp"], 1.0)[1][0]
    top = ax.secondary_xaxis(
        "top", functions=(lambda d: d * per_dm, lambda s: s / per_dm)
    )
    top.set_xlabel("bottom-channel smear (samples)", fontsize=7.5)
    top.tick_params(labelsize=7)


def eta_vs_dm(
    res: RecoveryResult, theme: Theme, *, mf: bool, size: tuple, title: bool = True
) -> Figure:
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=size, sharex=True)
    for ax, width in zip(axes.flat, WIDTHS, strict=True):
        for name in PLOTTED + list(VARIANTS):
            if name in res.engines:
                vals = res.eta_mf(name) if mf else res.eta(name)
                draw(ax, theme, *by_dm(res, vals, width), name)
        if mf:
            ax.axhline(1.0, color=theme["ink2"], ls="--", lw=1.0)
        else:
            dm, lim = by_dm(res, res.snr_matched_filter / res.snr_injected, width)
            ax.plot(dm, lim, color=theme["ink2"], ls="--", lw=1.1)
            ax.plot(dm, textbook(res, dm, width), color=theme["muted"], ls=":", lw=1.1)
            log_y(ax)
        ax.set_title(width_label(width), fontsize=9)
        smear_axis(ax, res)
    for ax in axes:
        ax.set_xlabel("DM (pc cm⁻³)")
    axes[0].set_ylabel(
        "η_MF = S/N / matched filter" if mf else "η = S/N / injected S/N"
    )
    extras = (
        []
        if mf
        else [
            ("ink2", "--", "incoherent limit (matched filter)"),
            ("muted", ":", "textbook √(W/W_eff)"),
        ]
    )
    handles, labels = engine_handles(theme, extras=extras)
    c = res.config
    what = "Fraction of the incoherent limit" if mf else "S/N recovered at the exact DM"
    head = (
        (
            f"{what}: {c['nchans']} channels, {c['f_min']:g}-{c['f_max']:g} MHz, "
            f"{c['tsamp'] * 1e6:g} µs"
        )
        if title
        else ""
    )
    finish(fig, theme, head, handles, labels, ncol=4, bottom=0.2)
    return fig


def eta_vs_width(res: RecoveryResult, theme: Theme, size: tuple) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    dm_max = res.config["dm_max"]
    ranges = [
        ("DM ≤ 300", res.dm <= 300),
        (f"DM ≥ {0.65 * dm_max:.0f}", res.dm >= 0.65 * dm_max),
    ]
    widths = np.asarray([w for w in WIDTHS if np.any(res.width == w)])
    x = widths
    for ax, (label, sel) in zip(axes, ranges, strict=True):
        for name in PLOTTED + list(VARIANTS):
            if name in res.engines:
                y = [np.median(res.eta(name)[sel & (res.width == w)]) for w in widths]
                draw(ax, theme, x, np.asarray(y), name)
        lim = [
            np.median(
                (res.snr_matched_filter / res.snr_injected)[sel & (res.width == w)]
            )
            for w in widths
        ]
        ax.plot(x, lim, color=theme["ink2"], ls="--", lw=1.1)
        ax.set_xscale("log", base=2)
        log_y(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w:g}" for w in widths], fontsize=7)
        ax.minorticks_off()
        ax.set_xlabel("intrinsic width W (samples)")
        ax.set_title(label, fontsize=9)
    axes[0].set_ylabel("η = S/N / injected S/N")
    handles, labels = engine_handles(theme, extras=[("ink2", "--", "incoherent limit")])
    finish(
        fig,
        theme,
        "S/N recovered against pulse width",
        handles,
        labels,
        ncol=4,
        bottom=0.18,
    )
    return fig


def threshold_vs_dm(
    res: RecoveryResult, theme: Theme, size: tuple, *, title: bool = True
) -> Figure:
    fig, ax = plt.subplots(figsize=size)
    for name in PLOTTED + list(VARIANTS):
        if name in res.engines:
            draw(ax, theme, res.dms, res.engines[name]["threshold"], name)
    ax.set_xlabel("DM (pc cm⁻³)")
    ax.set_ylabel("threshold S/N")
    if title:
        far = res.meta.get("far_per_hour", 1.0)
        ax.set_title(
            f"S/N threshold for {far:g} false candidate per hour over "
            f"{res.config['n_trials']} trials (fitted up-crossing rate)",
            fontsize=9,
        )
    handles, labels = engine_handles(theme)
    ncol = 3 if title else 2
    finish(fig, theme, "", handles, labels, ncol=ncol, bottom=0.2 if title else 0.3)
    return fig


def completeness_figure(res: RecoveryResult, theme: Theme, size: tuple) -> Figure:
    dm_max = res.config["dm_max"]
    regions = [
        ("DM ≤ 300", res.dm <= 300),
        (f"DM ≥ {0.65 * dm_max:.0f}", res.dm >= 0.65 * dm_max),
    ]
    widths = WIDTHS
    fig, axes = plt.subplots(
        len(widths), len(regions), figsize=size, sharex=True, sharey=True
    )
    for i, width in enumerate(widths):
        for j, (label, sel) in enumerate(regions):
            ax = axes[i, j]
            pick = sel & (res.width == width)
            for name in PLOTTED + list(VARIANTS):
                if name in res.engines:
                    draw(ax, theme, res.snr_grid, res.completeness(name, pick), name)
            ax.set_xscale("log")
            ax.set_title(f"{width_label(width)}, {label}", fontsize=9)
            if i == len(widths) - 1:
                ax.set_xlabel("injected S/N")
            if j == 0:
                ax.set_ylabel("completeness")
    handles, labels = engine_handles(theme)
    finish(
        fig,
        theme,
        "Completeness at a fixed false-alarm rate",
        handles,
        labels,
        ncol=3,
        bottom=0.1,
    )
    return fig


def snr50_figure(res: RecoveryResult, theme: Theme, size: tuple) -> Figure:
    widths = WIDTHS
    fig, axes = plt.subplots(1, len(widths), figsize=size, sharex=True)
    for ax, width in zip(axes, widths, strict=True):
        edges = np.quantile(res.dms, np.linspace(0, 1, 9))
        for name in PLOTTED + list(VARIANTS):
            if name not in res.engines:
                continue
            # Line: threshold over the expected S/N fraction (the injected
            # S/N at which the expected S/N reaches the threshold).
            dm, eta = by_dm(res, res.eta(name), width)
            thr = np.interp(dm, res.dms, res.engines[name]["threshold"])
            draw(ax, theme, dm, thr / eta, name)
            # Points: Monte-Carlo 50% completeness, pooled over DM bins.
            color = theme["algo"][VARIANTS.get(name, (name,))[0]]
            for lo, hi in itertools.pairwise(edges):
                pick = (res.width == width) & (res.dm >= lo) & (res.dm <= hi)
                if pick.any():
                    ax.plot(
                        np.median(res.dm[pick]),
                        snr50(res, name, pick),
                        "o",
                        ms=3,
                        color=color,
                        mfc="none" if name in VARIANTS else color,
                    )
        log_y(ax)
        ax.set_title(width_label(width), fontsize=9)
        ax.set_xlabel("DM (pc cm⁻³)")
    axes[0].set_ylabel("injected S/N for 50% completeness")
    handles, labels = engine_handles(theme)
    finish(
        fig,
        theme,
        "Injected S/N needed for detection (line: expected S/N at threshold; "
        "points: 50% completeness)",
        handles,
        labels,
        ncol=3,
        bottom=0.2,
    )
    return fig


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def summary_md(res: RecoveryResult) -> str:
    c = res.config
    hi = res.dm >= 0.65 * c["dm_max"]
    lines = [
        "<!-- generated by bench/scripts/plot_sensitivity.py; do not edit -->",
        "",
        (
            f"{c['nchans']} channels, {c['f_min']:g}-{c['f_max']:g} MHz, "
            f"{c['tsamp'] * 1e6:g} µs, piecewise Levin grid of {c['n_trials']} trials "
            f"(tol {c['levin_tol']}) to DM {c['dm_max']:g}; {len(res.dms)} DMs, "
            f"{len(res.dm)} pulses, dmt {res.meta.get('dmt_version', '?')}."
        ),
        "",
        (
            "**Fraction of the incoherent limit**, η_MF = S/N / matched-filter "
            f"S/N, median over all DMs (and over DM ≥ {0.65 * c['dm_max']:.0f} "
            "in brackets):"
        ),
        "",
        "| engine | " + " | ".join(f"W = {w:g}" for w in WIDTHS) + " |",
        "| :--- |" + " ---: |" * len(WIDTHS),
    ]
    for name in TABULATED:
        if name not in res.engines:
            continue
        cells = []
        for w in WIDTHS:
            sel = res.width == w
            cells.append(
                f"{np.median(res.eta_mf(name)[sel]):.3f} "
                f"({np.median(res.eta_mf(name)[sel & hi]):.3f})"
            )
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    mf = res.snr_matched_filter / res.snr_injected
    lines += [
        "",
        "**Incoherent limit itself**, matched-filter S/N / injected S/N (the loss to "
        "intra-channel smearing and sampling that only coherent dedispersion "
        "recovers): "
        + ", ".join(
            f"W = {w:g}: {np.median(mf[(res.width == w) & hi]):.3f}" for w in WIDTHS
        )
        + f" at DM ≥ {0.65 * c['dm_max']:.0f}.",
        "",
        (
            f"**Detection** at {res.meta.get('far_per_hour', 1.0):g} false alarm "
            "per hour of the whole search: S/N threshold (min-max over DM), and "
            f"injected S/N for 50% completeness at DM ≥ {0.65 * c['dm_max']:.0f}:"
        ),
        "",
        "| engine | threshold | "
        + " | ".join(f"S/N₅₀ (W = {w:g})" for w in WIDTHS)
        + " | noise ratio |",
        "| :--- | ---: |" + " ---: |" * len(WIDTHS) + " ---: |",
    ]
    for name in TABULATED:
        if name not in res.engines:
            continue
        cols = res.engines[name]
        s50 = " | ".join(
            f"{snr50(res, name, hi & (res.width == w)):.1f}" for w in WIDTHS
        )
        lines.append(
            f"| {name} | {cols['threshold'].min():.2f}-{cols['threshold'].max():.2f} | "
            f"{s50} | {np.median(cols['sigma_ratio']):.3f} |"
        )
    sd, dd = res.engines.get("SDMT"), res.engines.get("DDMT")
    if sd is not None and dd is not None:
        dev = np.max(np.abs(sd["snr"] / dd["snr"] - 1))
        lines += ["", f"SDMT vs DDMT: largest relative S/N difference {dev:.1e}."]
    lines.append("")
    return "\n".join(lines)


def latex_table(res: RecoveryResult) -> str:
    hi = res.dm >= 0.65 * res.config["dm_max"]
    rows = []
    for name in TABULATED:
        if name not in res.engines:
            continue
        cells = " & ".join(
            f"{np.median(res.eta_mf(name)[(res.width == w) & hi]):.3f}" for w in WIDTHS
        )
        thr = res.engines[name]["threshold"]
        s50 = " & ".join(
            f"{snr50(res, name, hi & (res.width == w)):.1f}" for w in WIDTHS
        )
        rows.append(f"{name} & {cells} & {thr.min():.2f}--{thr.max():.2f} & {s50} \\\\")
    mf = res.snr_matched_filter / res.snr_injected
    lim = " & ".join(f"{np.median(mf[(res.width == w) & hi]):.3f}" for w in WIDTHS)
    head = " & ".join(f"$W{{=}}{w:g}$" for w in WIDTHS)
    n = len(WIDTHS)
    return (
        "% generated by bench/scripts/plot_sensitivity.py --paper\n"
        f"\\begin{{tabular}}{{l{'r' * n}r{'r' * n}}}\n\\toprule\n"
        f"& \\multicolumn{{{n}}}{{c}}{{$\\eta_\\mathrm{{MF}}$}} & & "
        f"\\multicolumn{{{n}}}{{c}}{{$\\mathrm{{S/N}}_{{50}}$}} \\\\\n"
        f"Engine & {head} & Threshold & {head} \\\\\n\\midrule\n"
        + "\n".join(rows)
        + f"\n\\midrule\nlimit / injected & {lim} &{' &' * n} \\\\\n"
        + "\\bottomrule\n\\end{tabular}\n"
    )


# --------------------------------------------------------------------------
def figures(res: RecoveryResult, theme: Theme, *, paper: bool) -> dict[str, Figure]:
    big = (MNRAS_PAGE, 3.0) if paper else (12, 4.6)
    small = (MNRAS_PAGE, 2.9) if paper else (11, 4.6)
    return {
        "sensitivity_eta_vs_dm": eta_vs_dm(
            res, theme, mf=False, size=big, title=not paper
        ),
        "sensitivity_eta_mf_vs_dm": eta_vs_dm(
            res, theme, mf=True, size=big, title=not paper
        ),
        "sensitivity_eta_vs_width": eta_vs_width(res, theme, small),
        "sensitivity_threshold_vs_dm": threshold_vs_dm(
            res, theme, (MNRAS_PAGE / 2, 3.4) if paper else (7, 4.6), title=not paper
        ),
        "sensitivity_completeness": completeness_figure(
            res, theme, (MNRAS_PAGE, 6.0) if paper else (10, 8.6)
        ),
        "sensitivity_snr50_vs_dm": snr50_figure(res, theme, small),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=REPO / "bench" / "results")
    ap.add_argument(
        "--config",
        default="cryopaf_low",
        help="result name under <results>/sensitivity (default cryopaf_low); other "
        "configs plot to <results>/plots/<config>/ and suffix paper files",
    )
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--paper-dir", type=Path, default=REPO / "paper")
    args = ap.parse_args()
    path = args.results / "sensitivity" / f"{args.config}.npz"
    if not path.exists():
        sys.exit(f"no results at {path}")
    res = RecoveryResult.load(path)
    out_root = args.results / "plots"
    suffix = ""
    if args.config != "cryopaf_low":
        out_root /= args.config
        suffix = f"_{args.config}"
    for name, theme in THEMES.items():
        style(theme)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        for fname, fig in figures(res, theme, paper=False).items():
            fig.savefig(out / f"{fname}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    (out_root / "sensitivity_summary.md").write_text(summary_md(res))
    print(f"wrote sensitivity figures and summary to {out_root}")
    if args.paper:
        global SHOW_TITLES  # noqa: PLW0603 - one switch for every figure
        SHOW_TITLES = False
        style(THEMES["light"])
        mpl.rcParams.update(
            {
                "font.size": 8,
                "axes.titlesize": 8.5,
                "axes.labelsize": 8,
                "legend.fontsize": 7.5,
            }
        )
        fig_dir = args.paper_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for fname, fig in figures(res, THEMES["light"], paper=True).items():
            fig.savefig(fig_dir / f"{fname}{suffix}.pdf", bbox_inches="tight")
            plt.close(fig)
        tab_dir = args.paper_dir / "tables"
        tab_dir.mkdir(parents=True, exist_ok=True)
        (tab_dir / f"sensitivity{suffix}.tex").write_text(latex_table(res))
        print(f"wrote paper figures to {fig_dir} and table to {tab_dir}")


if __name__ == "__main__":
    main()
