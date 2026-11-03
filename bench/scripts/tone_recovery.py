"""Recovered power of smeared sinusoids, per engine, relative to DDMT.

Run from a checkout with dmtlib installed::

    python bench/scripts/tone_recovery.py --dm 200
    python bench/scripts/tone_recovery.py --plot docs/_static/periodicity

Each test tone is a sinusoid at an exact Fourier bin, integrated over each
channel's bandwidth (``--nsub`` sub-frequencies) and over the sample bin, the
way a filterbank records it. Every engine sees the same three contiguous
blocks; the last block is measured once the history is complete. The figure of
merit is the power at the tone's bin divided by that engine's own noise power
at that bin (white ``nchans`` for the unfiltered engines, the box-filter PSD
for ``fdmt_box``), so the numbers are signal-to-noise in the power spectrum.
The ``ideal`` column is a perfectly phase-aligned channel sum of the same
smeared signal.

This measures signal recovery, not noise statistics; for those see
``fdmt_noise.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, dispersion_delay

ENGINES = ("fdmt", "fdmt_box", "fdmt_fft", "ddmt_fft")
PLOT_FREQS = (0.03, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.45)
# (engine, legend label, colour key in plot_suite.THEMES, dashed)
PLOT_LINES = (
    ("ddmt", "DDMT", "DDMT", False),
    ("fdmt", "FDMT", "FDMT", False),
    ("fdmt_box", "FDMT, box smearing", "FDMT", True),
    ("fdmt_fft", "FDMT-FFT", "FDMT-FFT", False),
    ("ddmt_fft", "DDMT-FFT", "DDMT-FFT", False),
)


def recovery(
    *,
    dm: float,
    f_min: float = 400.0,
    f_max: float = 800.0,
    nchans: int = 1024,
    tsamp: float = 0.001,
    nsamps: int = 8192,
    nsub: int = 64,
    freqs: tuple[float, ...] = (0.03, 0.0625, 0.125, 0.25, 0.35, 0.45),
    spread: int = 16,
    spread_step: int = 5,
) -> list[dict[str, float]]:
    """Return one row per requested Fourier frequency (cycles/sample).

    Each row averages ``spread`` tones ``spread_step`` bins apart around that
    frequency: mean S/N per engine, divided by DDMT's mean S/N.
    """
    grid = np.array([dm])
    engines = {
        "fdmt": FDMT(
            f_min, f_max, nchans, nsamps, tsamp, dm_grid=grid, use_box_smearing=False
        ),
        "fdmt_box": FDMT(
            f_min, f_max, nchans, nsamps, tsamp, dm_grid=grid, use_box_smearing=True
        ),
        "fdmt_fft": FDMTFFT(
            f_min, f_max, nchans, nsamps, tsamp, dm_grid=grid, use_box_smearing=False
        ),
    }
    dm_eff = float(engines["fdmt"].dm_grid_final[0])
    dms = np.array([dm_eff], dtype=np.float32)
    engines["ddmt"] = DDMT(f_min, f_max, nchans, tsamp, dms)
    engines["ddmt_fft"] = DDMTFFT(f_min, f_max, nchans, tsamp, dms)

    df = (f_max - f_min) / nchans
    sub = f_min + df * (np.arange(nchans)[:, None] + (np.arange(nsub) + 0.5) / nsub)
    tau = dispersion_delay(dm_eff, sub, f_max) / tsamp  # (nchans, nsub) samples
    plan = engines["fdmt_box"].plan
    lengths = plan.smearing_grid_final.reshape(-1, nchans)[0].astype(int) + 1
    weights = np.asarray(plan.box_weights, float) * 2.0**-plan.box_fraction_bits
    w_c = weights[lengths - 1]

    # The engines are linear, so all tones go into one signal. Each tone sits
    # on an exact Fourier bin of the measured block (no leakage); around each
    # requested frequency, `spread` tones are averaged, because one tone's
    # ratio jumps with the exact phase scatter at that frequency.
    groups = [
        round(f_req * nsamps) + spread_step * (np.arange(spread) - spread // 2)
        for f_req in freqs
    ]
    bins = np.concatenate(groups)
    f = bins / nsamps
    chan = np.exp(-2j * np.pi * f[:, None, None] * tau[None]).mean(axis=2)
    amp = chan * np.sinc(f)[:, None]  # (ntones, nchans), with the bin integral
    t = np.arange(3 * nsamps)
    x = np.zeros((nchans, t.size))
    for i in range(bins.size):
        x += np.real(amp[i][:, None] * np.exp(2j * np.pi * f[i] * (t + 0.5)))
    x = x.astype(np.float32)
    power = {}
    for name, eng in engines.items():
        for block in range(3):
            y = eng.execute(np.ascontiguousarray(x[:, block * nsamps :][:, :nsamps]))
        y = np.asarray(y).reshape(-1)[-nsamps:]
        power[name] = np.abs(np.fft.rfft(y)[bins]) ** 2
    noise = dict.fromkeys(engines, nsamps * nchans * np.ones(bins.size))
    box_h = w_c * np.sin(np.pi * f[:, None] * lengths) / np.sin(np.pi * f[:, None])
    noise["fdmt_box"] = nsamps * np.sum(box_h**2, axis=1)
    snr = {name: power[name] / noise[name] for name in engines}
    ideal = (nsamps / 2 * np.abs(amp).sum(axis=1)) ** 2 / (nsamps * nchans)
    rows = []
    start = 0
    for f_req, group in zip(freqs, groups, strict=True):
        sel = slice(start, start + group.size)
        start += group.size
        ddmt = snr["ddmt"][sel].mean()
        rows.append(
            {
                "freq": f_req,
                "ddmt_snr": ddmt,
                "ideal_snr": ideal[sel].mean(),
                **{name: snr[name][sel].mean() / ddmt for name in ENGINES},
            }
        )
    return rows


def plot(outdir: Path, dms: list[float], **kwargs: float) -> None:
    """Write tone_recovery.png in light and dark themes to ``outdir/<theme>/``.

    Each panel is one DM; each line is an engine's S/N as a fraction of the
    ideal, perfectly aligned sum.
    """
    import matplotlib as mpl  # noqa: PLC0415 - plotting is optional

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sys.path.insert(0, str(Path(__file__).parent))
    from plot_suite import THEMES, style  # noqa: PLC0415

    data = {dm: recovery(dm=dm, freqs=PLOT_FREQS, **kwargs) for dm in dms}
    for name, theme in THEMES.items():
        style(theme)
        fig, axes = plt.subplots(
            1, len(dms), figsize=(4.6 * len(dms), 3.9), sharey=True, squeeze=False
        )
        for ax, dm in zip(axes[0], dms, strict=True):
            rows = data[dm]
            f = [r["freq"] for r in rows]
            ddmt = np.array([r["ddmt_snr"] / r["ideal_snr"] for r in rows])
            for key, label, colour, dashed in PLOT_LINES:
                y = ddmt if key == "ddmt" else ddmt * np.array([r[key] for r in rows])
                ax.plot(
                    f,
                    y,
                    "--" if dashed else "-",
                    marker="o",
                    color=theme["algo"][colour],
                    label=label,
                    markerfacecolor=theme["surface"] if dashed else None,
                )
            ax.set_title(f"DM {dm:g} pc cm$^{{-3}}$")
            ax.set_xlabel("Fourier frequency (cycles per sample)")
            ax.set_yscale("log")
            ax.set_xlim(0, 0.5)
        axes[0][0].set_ylabel("S/N / ideal aligned sum")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncols=len(labels),
            bbox_to_anchor=(0.5, -0.04),
        )
        fig.suptitle("Recovered power of a smeared sinusoid", fontweight="bold")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        out = outdir / name
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out / "tone_recovery.png",
            dpi=150,
            facecolor=theme["surface"],
            bbox_inches="tight",
        )
        plt.close(fig)
    print(f"wrote tone_recovery.png x {len(THEMES)} themes to {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dm", type=float, default=200.0)
    parser.add_argument("--nchans", type=int, default=1024)
    parser.add_argument("--f-min", type=float, default=400.0)
    parser.add_argument("--f-max", type=float, default=800.0)
    parser.add_argument("--tsamp", type=float, default=0.001)
    parser.add_argument(
        "--plot", type=Path, metavar="DIR", help="write the figure to DIR instead"
    )
    parser.add_argument(
        "--plot-dms", type=float, nargs="+", default=[50.0, 200.0], metavar="DM"
    )
    args = parser.parse_args()
    if args.plot:
        plot(
            args.plot,
            args.plot_dms,
            nchans=args.nchans,
            f_min=args.f_min,
            f_max=args.f_max,
            tsamp=args.tsamp,
        )
        return
    rows = recovery(
        dm=args.dm,
        nchans=args.nchans,
        f_min=args.f_min,
        f_max=args.f_max,
        tsamp=args.tsamp,
    )
    head = f"{'cyc/samp':>9} {'DDMT/ideal':>10} " + " ".join(
        f"{n + '/DDMT':>13}" for n in ENGINES
    )
    print(f"S/N in the power spectrum at DM {args.dm}, relative to DDMT")
    print(head)
    for r in rows:
        cells = " ".join(f"{r[n]:13.3f}" for n in ENGINES)
        print(f"{r['freq']:9.4f} {r['ddmt_snr'] / r['ideal_snr']:10.3f} {cells}")


if __name__ == "__main__":
    main()
