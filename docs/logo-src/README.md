# dmt logo source

The mark is three strokes: two dispersion sweeps that follow the cold-plasma
delay curve, `Δt ∝ DM · ν⁻²`, with frequency running down the page, and the
straight pulse between them after dedispersion. The lockup sets "dmt" in
IBM Plex Mono (Medium) beside it.

## Regenerate

```bash
uv run docs/logo-src/make_logo.py            # rewrites docs/_static/logo/*.svg
uv run docs/logo-src/make_logo.py --exports  # also writes build/ (PNG, ICO, mono, social)
```

The script downloads IBM Plex Mono v1.1.0 once into `.cache/` and verifies its
sha256. Text is converted to outlines, so the SVGs need no fonts to render.

## Files in the repo

Only the files the README and docs reference are committed:

| File | Used by |
| --- | --- |
| `dmt-logo-tagline.svg`, `dmt-logo-tagline-dark.svg` | `README.md` header |
| `dmt-logo.svg`, `dmt-logo-dark.svg` | Sphinx sidebar (`docs/conf.py`) |
| `dmt-mark.svg`, `dmt-mark-dark.svg` | Docs landing page hero (`docs/index.md`) |
| `favicon.svg` | Sphinx favicon (follows the browser's light/dark preference) |

Everything else comes from `--exports`: high-res PNGs, mono variants,
rounded-tile icons (GitHub avatar, conda-forge), `favicon.ico`,
`apple-touch-icon.png`, and `social-preview.png` (1280×640, for GitHub
Settings → Social preview).

## Parameters

| Constant | Value | Meaning |
| --- | --- | --- |
| `REACH` | 32 | Sideways swing of each sweep (mark is 100 units tall) |
| `STROKE` | 13.5 | Stroke width |
| `NU_LO` | 0.25 | Lowest/highest frequency ratio; sets how sharply the sweep kicks |
| `ACCENT` / `ACCENT_D` | `#2456E8` / `#7196FF` | Sweep colour on light / dark |
| `INK` / `INK_D` | `#12151B` / `#EDF0F4` | Centre stroke and text on light / dark |

Reach above ~38 closes the tip-to-centre gap below 1.5 px at 16 px favicon size.
