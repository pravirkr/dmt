---
jupytext:
  notebook_metadata_filter: kernelspec,-jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial 3: Sparse DM grids and negative DMs

This tutorial covers two plan options: sparse DM grids, which drop trials made
redundant by intra-channel dispersion smearing, and negative delay ranges.

+++

## Intra-channel smearing

Each channel of a filterbank has a finite width $\Delta \nu = \frac{B}{N_{\text{chans}}}$.
Within a channel of width $\Delta \nu$ centred at $\nu$, dispersion delays the
bottom of the channel relative to the top by the **intra-channel smearing
time**

$$\Delta t_{\text{smear}} \approx 2 \mathcal{D} \cdot \text{DM} \cdot \nu^{-3} \Delta \nu$$

### Why a dense linear grid is wasteful at high DM

$\Delta t_{\text{smear}}$ grows linearly with DM and as $\nu^{-3}$.

- **Low DM.** $\Delta t_{\text{smear}} < t_{\text{samp}}$, so you need DM
  steps of one sample.
- **High DM.** $\Delta t_{\text{smear}} \gg t_{\text{samp}}$: the pulse is
  already smeared over 10, 20 or 50 samples inside each channel.

Fine uniform steps $\delta(\text{DM})$ at those DMs give heavily overlapping
trials and waste memory and compute.

### Sparse grids in DMT

You can pass your own non-uniform grid:

`dt_grid` / `dt_arr`
: Explicit sample-delay trials $[d_0, d_1, d_2, \dots]$.

`dm_grid` / `dm_arr`
: Explicit DM trials in $\text{pc}\,\text{cm}^{-3}$.

```{note}
The planner sorts the grid, removes duplicates, and maps it onto the tree.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from dmtlib import FDMTPlan, FDMT

f_min = 800.0    # Low-frequency band (high dispersion smearing)
f_max = 1200.0
nchans = 256
nsamps = 1024
tsamp = 1e-3

# 1. Standard dense linear grid: dt in [0, 500] with step 1
plan_dense = FDMTPlan(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=tsamp,
    dt_max=500,
    dt_min=0,
    dt_step=1,
)

# 2. Optimal sparse grid: geometric / quadratic spacing matching smearing width
# Fine steps at low delay, geometrically expanding steps at high delay
dt_sparse = np.unique(np.round(np.geomspace(1, 501, 120) - 1).astype(int))

plan_sparse = FDMTPlan(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=tsamp,
    dt_grid=dt_sparse,
)

print("Grid Comparison:")
print(f"  Dense linear grid trials:  {len(plan_dense.dt_grid_final)}")
print(f"  Sparse adaptive grid trials: {len(plan_sparse.dt_grid_final)}")
print(f"  Reduction in output trials:  {len(plan_dense.dt_grid_final) / len(plan_sparse.dt_grid_final):.2f}x")
```

```{code-cell} ipython3
# Compare tree memory consumption
mem_dense_kb = plan_dense.container.memory_usage / 1024.0
mem_sparse_kb = plan_sparse.container.memory_usage / 1024.0

print(f"Plan buffer memory:")
print(f"  Dense plan:  {mem_dense_kb:.1f} KB")
print(f"  Sparse plan: {mem_sparse_kb:.1f} KB ({(1.0 - mem_sparse_kb / mem_dense_kb) * 100:.1f}% reduction)")

# Plot trial spacing comparison
plt.figure(figsize=(10, 4))
plt.plot(plan_dense.dt_grid_final, label="Dense linear trials", color="#94a3b8", lw=1.5)
plt.plot(plan_sparse.dt_grid_final, label="Sparse adaptive trials", color="#2563eb", lw=2)
plt.title("Delay Trial Distribution Comparison", fontsize=13)
plt.xlabel("Trial Index", fontsize=11)
plt.ylabel("Delay (samples)", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

## Negative DMs

Set `dt_min < 0` to search negative delays, or pass negative values in
`dm_grid`. A **negative DM** means the low frequencies arrive *first*. No
astrophysical signal does that, but swept RFI can, so negative-DM trials are a
useful control.

```{tip}
A candidate that is just as strong at negative DM is probably not
astrophysical.
```

```{code-cell} ipython3
# Setup a bipolar plan searching symmetric negative and positive delays
dt_max = 64
dt_min = -64

plan_bipolar = FDMTPlan(
    f_min=1200.0,
    f_max=1600.0,
    nchans=128,
    nsamps=512,
    tsamp=1e-3,
    dt_max=dt_max,
    dt_min=dt_min,
    dt_step=1,
    mode="valid",
)

print(f"Bipolar plan created:")
print(f"  dt range: [{plan_bipolar.dt_min}, {plan_bipolar.dt_max}]")
print(f"  Total delay trials: {len(plan_bipolar.dt_grid_final)}")
print(f"  First 5 trials: {plan_bipolar.dt_grid_final[:5]}")
print(f"  Last 5 trials:  {plan_bipolar.dt_grid_final[-5:]}")
```

```{code-cell} ipython3
# An inverted chirp: the same nu^-2 sweep as dispersion, but the low
# frequencies arrive first. Channel c is centred at f_min + (c + 1/2) df.
rng = np.random.default_rng(1)
waterfall_rfi = rng.normal(size=(128, 512)).astype(np.float32)

target_delay = -30
df = (1600.0 - 1200.0) / 128
f_chan = 1200.0 + df * (np.arange(128) + 0.5)
frac = (f_chan**-2 - 1600.0**-2) / (1200.0**-2 - 1600.0**-2)  # 1 at the bottom, 0 at the top
for c in range(128):
    waterfall_rfi[c, int(round(250 + target_delay * frac[c]))] += 3.0

# Dedisperse using FDMT
fdmt_bipolar = FDMT(
    f_min=1200.0,
    f_max=1600.0,
    nchans=128,
    nsamps=512,
    tsamp=1e-3,
    dt_max=dt_max,
    dt_min=dt_min,
    dt_step=1,
    mode="valid",
)

dmt_bipolar = fdmt_bipolar.execute(waterfall_rfi)

# Locate peak in negative DM territory
peak_dt_idx, peak_t = np.unravel_index(np.argmax(dmt_bipolar), dmt_bipolar.shape)
recovered_dt = plan_bipolar.dt_grid_final[peak_dt_idx]

print(f"Recovered chirp peak at trial dt = {recovered_dt} (Injected: {target_delay})")

# Plot the bipolar DMT plane
plt.figure(figsize=(10, 5))
plt.imshow(
    dmt_bipolar,
    aspect="auto",
    origin="lower",
    extent=[0, dmt_bipolar.shape[1], plan_bipolar.dt_grid_final[0], plan_bipolar.dt_grid_final[-1]],
    cmap="magma",
)
plt.axhline(0, color="white", linestyle=":", alpha=0.6, label="DM = 0 (Un-dispersed)")
plt.axhline(recovered_dt, color="cyan", linestyle="--", label=f"inverted chirp ($d_t={recovered_dt}$)")
plt.title("Negative and positive delays in one plane", fontsize=13)
plt.xlabel("Time Sample", fontsize=11)
plt.ylabel("Delay Trial $d_t$ (samples)", fontsize=11)
plt.colorbar(label="Power")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
```

:::{admonition} Takeaways
:class: dmt-summary

- Intra-channel smearing grows linearly with DM, so at high DM fine uniform
  steps give heavily overlapping trials.
- Pass `dt_grid` or `dm_grid` for a non-uniform grid; the planner sorts it,
  removes duplicates and maps it onto the tree.
- `dt_min < 0` or negative `dm_grid` values search negative delays, a control
  against swept RFI.
:::

{doc}`Tutorial 4 <04_fdmt_vs_ddmt_comparison>` compares FDMT with direct
dedispersion (DDMT).
