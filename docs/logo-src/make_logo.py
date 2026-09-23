# /// script
# requires-python = ">=3.10"
# dependencies = ["fonttools", "resvg-py", "pillow"]
# ///
# ruff: noqa: RUF002
"""Generate the dmt logo.

The mark is three strokes: two dispersion sweeps (Δt ∝ DM·ν⁻², frequency
running down the page) around the straight, dedispersed pulse. The lockup
sets "dmt" in IBM Plex Mono beside it. Text is converted to outlines, so
the SVGs render identically without the font installed.

Usage::

    uv run docs/logo-src/make_logo.py            # SVGs used by README + docs
    uv run docs/logo-src/make_logo.py --exports  # also PNG/ICO/mono/social exports

The first form rewrites ``docs/_static/logo/``. Exports go to
``docs/logo-src/build/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import resvg_py
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

HERE = Path(__file__).resolve().parent
STATIC = HERE.parent / "_static" / "logo"
BUILD = HERE / "build"
CACHE = HERE / ".cache"

# IBM Plex Mono, SIL Open Font License 1.1. Pinned by version and hash.
FONT_URL = (
    "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@1.1.0"
    "/fonts/complete/woff/IBMPlexMono-{}.woff"
)
FONT_SHA256 = {
    "Medium": "15d0fcad70465530211f78116d13df0fc1a930361b613f4418238d212bafebf1",
    "Regular": "1d5732f53287cbe58936deabda52a60d810b84ac4a3fbdefa8e7632a2c38c39d",
}

# Palette. Accent is the sweep colour; the centre stroke and text use ink.
INK, ACCENT, MUTED = "#12151B", "#2456E8", "#5C6573"
INK_D, ACCENT_D, MUTED_D = "#EDF0F4", "#7196FF", "#8A94A3"
TILE_L, TILE_D = "#FFFFFF", "#0E1117"

# Mark geometry, in a box 100 units tall with the centre stroke at x=0.
REACH = 32  # sideways swing of each sweep, top to bottom
STROKE = 13.5
NU_LO = 0.25  # lowest frequency as a fraction of the highest; sets the kick


def fmt(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


# ---------------------------------------------------------------- mark
def pulse_geom(reach: float = REACH) -> tuple[list[tuple[float, float]], float]:
    """Left sweep centreline points and the mark's half-width."""
    h, n = STROKE / 2, 96
    gap = 35 - (reach - 25) * 0.35  # distance from centre axis to the sweep tip

    def sweep(t: float) -> float:
        nu = 1 - t * (1 - NU_LO)
        return (nu**-2 - 1) / (NU_LO**-2 - 1)

    # End the sweep so its angled butt cap bottoms out on the baseline.
    y_end = 100 - h * 0.9
    for _ in range(4):
        pts = [
            (-(gap + reach) + reach * sweep(i / n), y_end * i / n) for i in range(n + 1)
        ]
        (ax, ay), (bx, by) = pts[-2], pts[-1]
        y_end = 100 - h * (bx - ax) / math.hypot(bx - ax, by - ay)
    return pts, gap + reach + h


def mark_body(ink: str, accent: str) -> str:
    pts, _ = pulse_geom()
    left = "M" + "L".join(f"{fmt(x)},{fmt(y)}" for x, y in pts)
    right = "M" + "L".join(f"{fmt(-x)},{fmt(y)}" for x, y in pts)
    common = (
        f'fill="none" stroke-width="{fmt(STROKE)}" '
        'stroke-linecap="butt" stroke-linejoin="miter"'
    )
    return (
        f'<path d="M0,0V100" stroke="{ink}" {common}/>'
        f'<path d="{left}" stroke="{accent}" {common}/>'
        f'<path d="{right}" stroke="{accent}" {common}/>'
    )


# ---------------------------------------------------------------- type
def font_path(weight: str) -> Path:
    path = CACHE / f"IBMPlexMono-{weight}.woff"
    if (
        not path.exists()
        or hashlib.sha256(path.read_bytes()).hexdigest() != FONT_SHA256[weight]
    ):
        CACHE.mkdir(parents=True, exist_ok=True)
        # FONT_URL is a fixed https constant, and the payload is hash-checked.
        url = FONT_URL.format(weight)
        data = urllib.request.urlopen(url, timeout=30).read()  # noqa: S310
        digest = hashlib.sha256(data).hexdigest()
        if digest != FONT_SHA256[weight]:
            msg = f"IBMPlexMono-{weight}: unexpected sha256 {digest}"
            raise SystemExit(msg)
        path.write_bytes(data)
    return path


class Face:
    def __init__(self, weight: str) -> None:
        self.font = TTFont(font_path(weight))
        self.glyphs = self.font.getGlyphSet()
        self.cmap = self.font.getBestCmap()
        self.upm = self.font["head"].unitsPerEm
        self.hmtx = self.font["hmtx"]

    def bounds(self, ch: str) -> tuple[float, float, float, float]:
        pen = BoundsPen(self.glyphs)
        self.glyphs[self.cmap[ord(ch)]].draw(pen)
        return pen.bounds

    def outline(
        self, text: str, size: float, x0: float, baseline: float, tracking: float = 0.0
    ) -> tuple[str, float]:
        """Path data for ``text`` and its advance width. ``tracking`` is in em."""
        scale = size / self.upm
        x, parts = x0, []
        for i, ch in enumerate(text):
            name = self.cmap[ord(ch)]
            pen = SVGPathPen(self.glyphs, ntos=fmt)
            self.glyphs[name].draw(
                TransformPen(pen, (scale, 0, 0, -scale, x, baseline))
            )
            parts.append(pen.getCommands())
            x += self.hmtx[name][0] * scale
            if i < len(text) - 1:
                x += tracking * size
        return "".join(parts), x - x0


# ---------------------------------------------------------------- compositions
def svg(viewbox: Sequence[float], body: str, label: str) -> str:
    vb = " ".join(fmt(v) for v in viewbox)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" role="img" '
        f'aria-label="{label}"><title>{label}</title>{body}</svg>\n'
    )


def mark_svg(ink: str, accent: str, pad: float = 4.0) -> str:
    _, half = pulse_geom()
    return svg(
        (-half - pad, -pad, 2 * half + 2 * pad, 100 + 2 * pad),
        mark_body(ink, accent),
        "dmt",
    )


def tile_svg(
    ink: str, accent: str, bg: str, inner: float = 0.6, radius: float = 0.22
) -> str:
    _, half = pulse_geom()
    s = 512
    k = s * inner / max(2 * half, 100)
    body = (
        f'<rect width="{s}" height="{s}" rx="{fmt(s * radius)}" fill="{bg}"/>'
        f'<g transform="translate({fmt(s / 2)} {fmt(s / 2 - 50 * k)}) scale({k:.5f})">'
        f"{mark_body(ink, accent)}</g>"
    )
    return svg((0, 0, s, s), body, "dmt")


def lockup_svg(
    med: Face, reg: Face, ink: str, accent: str, muted: str, *, tagline: bool
) -> str:
    _, half = pulse_geom()
    asc = 78  # height of "d" as a share of the mark
    name_size = asc / (med.bounds("d")[3] / med.upm)
    x_text = half + 36
    if tagline:
        sub_size, sub_gap = 15.5, 16
        cap = reg.bounds("D")[3] / reg.upm * sub_size
        baseline = 50 - (asc + sub_gap + cap) / 2 + asc
        sub_base = baseline + sub_gap + cap
    else:
        baseline = 50 + asc / 2
    name_d, name_w = med.outline("dmt", name_size, x_text, baseline)
    # Drop the side bearing of "d" so the gap to the mark is optical, not metric.
    lsb = med.bounds("d")[0] / med.upm * name_size
    body = f'<g transform="translate({fmt(-lsb)} 0)"><path d="{name_d}" fill="{ink}"/>'
    right = x_text + name_w - lsb
    if tagline:
        sub_d, sub_w = reg.outline(
            "DISPERSION MEASURE TRANSFORM", sub_size, x_text, sub_base, tracking=0.14
        )
        sub_lsb = reg.bounds("D")[0] / reg.upm * sub_size
        shift = fmt(lsb - sub_lsb)
        body += f'<path d="{sub_d}" transform="translate({shift} 0)" fill="{muted}"/>'
        right = max(right, x_text + sub_w - sub_lsb)
    body += "</g>" + mark_body(ink, accent)
    pad = 4
    label = "dmt: Dispersion Measure Transform" if tagline else "dmt"
    return svg((-half - pad, -pad, right + half + 2 * pad, 100 + 2 * pad), body, label)


def favicon_svg() -> str:
    """Bare mark that follows the browser's light/dark preference."""
    _, half = pulse_geom()
    k = 0.78 * 100 / max(2 * half, 100)
    mark = (
        mark_body("INK", "ACC")
        .replace('stroke="INK"', 'class="i"')
        .replace('stroke="ACC"', 'class="a"')
    )
    body = (
        f"<style>.i{{stroke:{INK}}}.a{{stroke:{ACCENT}}}"
        "@media (prefers-color-scheme:dark)"
        f"{{.i{{stroke:{INK_D}}}.a{{stroke:{ACCENT_D}}}}}</style>"
        f'<g transform="translate(50 {fmt(50 - 50 * k)}) scale({k:.5f})">{mark}</g>'
    )
    return svg((0, 0, 100, 100), body, "dmt")


def social_svg(lockup_dark: str) -> str:
    """1280x640 GitHub social preview: tagline lockup centred on dark."""
    width = 820
    height = width / aspect(lockup_dark)
    nested = lockup_dark.replace(
        "<svg ",
        f'<svg x="{fmt((1280 - width) / 2)}" y="{fmt((640 - height) / 2)}" '
        f'width="{width}" height="{fmt(height)}" ',
        1,
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" '
        'viewBox="0 0 1280 640">'
        f'<rect width="1280" height="640" fill="{TILE_D}"/>{nested}</svg>\n'
    )


# ---------------------------------------------------------------- output
def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        path.write_bytes(content)
    print(f"  {path.relative_to(HERE.parent.parent)}")


def png(svg_text: str, width: int, height: int) -> bytes:
    return bytes(resvg_py.svg_to_bytes(svg_string=svg_text, width=width, height=height))


def aspect(svg_text: str) -> float:
    vb = [float(v) for v in svg_text.split('viewBox="', 1)[1].split('"', 1)[0].split()]
    return vb[2] / vb[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--exports",
        action="store_true",
        help="also write PNG/ICO/mono/social exports to build/",
    )
    args = parser.parse_args()

    med, reg = Face("Medium"), Face("Regular")

    # Files referenced by README.md, docs/conf.py and docs/index.md.
    print("docs/_static/logo:")
    write(
        STATIC / "dmt-logo-tagline.svg",
        lockup_svg(med, reg, INK, ACCENT, MUTED, tagline=True),
    )
    write(
        STATIC / "dmt-logo-tagline-dark.svg",
        lockup_svg(med, reg, INK_D, ACCENT_D, MUTED_D, tagline=True),
    )
    write(
        STATIC / "dmt-logo.svg", lockup_svg(med, reg, INK, ACCENT, MUTED, tagline=False)
    )
    write(
        STATIC / "dmt-logo-dark.svg",
        lockup_svg(med, reg, INK_D, ACCENT_D, MUTED_D, tagline=False),
    )
    write(STATIC / "dmt-mark.svg", mark_svg(INK, ACCENT))
    write(STATIC / "dmt-mark-dark.svg", mark_svg(INK_D, ACCENT_D))
    write(STATIC / "favicon.svg", favicon_svg())

    if not args.exports:
        return

    print("docs/logo-src/build:")
    tones = {"": (ACCENT, ACCENT_D), "-mono": (INK, INK_D)}
    for tone, (acc_l, acc_d) in tones.items():
        for dark, (ink, acc, muted, tile) in {
            "": (INK, acc_l, MUTED, TILE_L),
            "-dark": (INK_D, acc_d, MUTED_D, TILE_D),
        }.items():
            assets = {
                f"dmt-mark{tone}{dark}": (mark_svg(ink, acc), "h", 1024),
                f"dmt-logo{tone}{dark}": (
                    lockup_svg(med, reg, ink, acc, muted, tagline=False),
                    "w",
                    2400,
                ),
                f"dmt-logo-tagline{tone}{dark}": (
                    lockup_svg(med, reg, ink, acc, muted, tagline=True),
                    "w",
                    2400,
                ),
                f"dmt-icon{tone}{dark}": (tile_svg(ink, acc, tile), "w", 1024),
            }
            for name, (text, axis, size) in assets.items():
                a = aspect(text)
                w, h = (
                    (size, round(size / a)) if axis == "w" else (round(size * a), size)
                )
                write(BUILD / f"{name}.svg", text)
                write(BUILD / f"{name}.png", png(text, w, h))

    # Favicon/touch icon use a tile with a larger mark so it survives 16 px.
    fav = tile_svg(INK, ACCENT, TILE_L, inner=0.74, radius=0.2)
    write(BUILD / "apple-touch-icon.png", png(fav, 180, 180))
    frames = [
        Image.open(io.BytesIO(png(fav, s, s))).convert("RGBA") for s in (64, 48, 32, 16)
    ]
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="ICO",
        sizes=[(s, s) for s in (16, 32, 48, 64)],
        append_images=frames[1:],
    )
    write(BUILD / "favicon.ico", buf.getvalue())

    social = social_svg(lockup_svg(med, reg, INK_D, ACCENT_D, MUTED_D, tagline=True))
    write(BUILD / "social-preview.png", png(social, 1280, 640))


if __name__ == "__main__":
    main()
