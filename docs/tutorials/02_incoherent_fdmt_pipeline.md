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

# Tutorial 2: Inside the FDMT tree

This tutorial looks at how the Fast Dispersion Measure Transform (FDMT) of
Zackay & Ofek (2017) builds its output: the sub-band tree, the stepper API
(`reset`, `advance`, `view_subband_data`, `finalize`) that lets you watch it
level by level, and the analytic noise calibration
(`get_effective_variance`, `get_effective_sigma`).

+++

## How the tree merges sub-bands

Start with $N_{\text{chans}}$ frequency channels. At the input level (level 0)
each channel has no internal dispersion (ignoring intra-channel smearing for
now).

1. **Level 1.** Pairs of adjacent channels $(0, 1), (2, 3), \dots$ are
   combined. To form the sum over both channels for a total delay $\Delta t$,
   the top channel is shifted relative to the bottom one by the right fraction
   of $\Delta t$.
2. **Level $l$.** Sub-bands from level $l-1$ are merged in pairs, so the
   number of sub-bands halves at every level:
   $$N_{\text{subbands}}(l) = \frac{N_{\text{chans}}}{2^l}$$
3. **Level $\log_2 N_{\text{chans}}$.** A single band from $f_{\min}$ to
   $f_{\max}$ remains. It holds the finished transform for every trial delay.

```
Level 0:  [C0] [C1] [C2] [C3] ... [C254] [C255]   (256 subbands)
             \  /     \  /              \    /
Level 1:     [S0]     [S1]    ...       [S127]     (128 subbands)
               \       /
Level 2:          [B0]        ...                  (64 subbands)
                   ...
Level 8:                       [DMT]               (1 band = full DM-time plane)
```

The **stepper API** lets you pause the transform and inspect the intermediate
sub-bands at any level.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from dmtlib import FDMT
from dmtlib.simulate import DM_CONSTANT, generate_frb

# Observation parameters
f_min = 1200.0   # MHz
f_max = 1600.0   # MHz
nchans = 64      # 64 channels for clean visual tree demonstration
nsamps = 512
tsamp = 1e-3
dt_max = 64
target_delay = 32  # samples across the band

# generate_frb takes a physical DM, not a sample delay.
dm = target_delay * tsamp / (DM_CONSTANT * (f_min**-2 - f_max**-2))
waterfall = generate_frb(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=tsamp,
    dm=dm,
    amp=3.0,
    offset=100,
    noise_rms=1.0,
)

# Instantiate FDMT engine
fdmt = FDMT(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=tsamp,
    dt_max=dt_max,
    mode="valid",
)

plan = fdmt.plan
print(f"FDMT plan: {plan.niters} merge levels above the {nchans} input channels.")
print(f"Injected DM = {dm:.2f} pc cm^-3 ({target_delay} samples across the band).")
```

## The stepper API

`fdmt.reset(waterfall)`
: Starts the stepper and loads the level-0 input.

`fdmt.advance(levels=1)`
: Runs one or more tree levels.

`fdmt.current_level`
: The current level, from $0$ to `total_levels`.

`fdmt.view_subband_data(subband_idx)`
: A zero-copy NumPy view of one sub-band's intermediate state, shape
  `(n_delays, n_samps)`.

`fdmt.finalize()`
: Runs the remaining levels and returns the DM-time plane.

```{code-cell} ipython3
# Reset stepper with input waterfall
fdmt.reset(waterfall)

print(f"Level after reset: {fdmt.current_level} / {fdmt.total_levels}")
print(f"Number of active subbands at Level 0: {fdmt.num_subbands}")

# Step through levels and record subband statistics
level_snapshots = []

for step in range(fdmt.total_levels):
    curr_lvl = fdmt.current_level
    n_subs = fdmt.num_subbands

    # Inspect subband 0 at the current level
    sub0_data = fdmt.view_subband_data(0)
    level_snapshots.append((curr_lvl, n_subs, sub0_data.shape, np.max(sub0_data)))

    print(f"Level {curr_lvl}: {n_subs} subbands, Subband 0 shape (n_dt, n_t) = {sub0_data.shape}")

    # Advance to next level
    fdmt.advance(1)

# Finish the transform. finalize() returns a view of the engine's buffer,
# which the next reset() overwrites, so keep a copy.
dmt_result = fdmt.finalize().copy()
print(f"\nFinal DM-time plane: {dmt_result.shape}")
```

```{code-cell} ipython3
# Visualize the evolution of Subband 0 across tree levels
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Reset and capture subband 0 views at levels 0, 1, 2, 3, 4, 5
fdmt.reset(waterfall)

for lvl in range(min(6, fdmt.total_levels + 1)):
    ax = axes[lvl]
    sub0 = fdmt.view_subband_data(0)
    im = ax.imshow(sub0, aspect="auto", origin="lower", cmap="magma")
    ax.set_title(f"Level {fdmt.current_level}: Subband 0 ({fdmt.num_subbands} subbands total)", fontsize=11)
    ax.set_xlabel("Time sample")
    ax.set_ylabel("Trial delay $d_t$")
    fig.colorbar(im, ax=ax)

    if not fdmt.is_finished:
        fdmt.advance(1)

plt.tight_layout()
plt.show()
```

## Noise calibration

Take independent Gaussian noise of variance $\sigma_0^2$ in every channel.
To turn the output into S/N you divide each DM trial by its noise standard
deviation $\sigma_\text{eff}(\text{DM})$.

With the default box smearing, **that sigma differs between trials**. Each
channel contributes the mean of the samples its dispersion smears across, and
that window grows with DM, so the mean of more samples has less noise. Without
box smearing (`use_box_smearing=False`) every trial has variance
$N_\text{chans}\sigma_0^2$. The tree shares partial sums between neighbouring
DM trials, which correlates *different* trials, but it does not change any
single trial's variance.

`dmt` computes $\sigma_\text{eff}$ exactly from the plan:

- `fdmt.get_effective_variance(dm_idx, boxcar_width=1)`
- `fdmt.get_effective_sigma(dm_idx, boxcar_width=1)`
- `fdmt.get_effective_sigma_grid(boxcar_width=1)`

```{seealso}
{doc}`Noise calibration <../pipeline_guide/noise_calibration>` for the
formula.
```

```{code-cell} ipython3
# Compute theoretical noise sigma across all DM trials
dm_trials = plan.dm_grid_final
sigma_grid_w1 = fdmt.get_effective_sigma_grid(boxcar_width=1)
sigma_grid_w4 = fdmt.get_effective_sigma_grid(boxcar_width=4)

plt.figure(figsize=(10, 4))
plt.plot(dm_trials, sigma_grid_w1, label="Boxcar width = 1", lw=2, color="#2563eb")
plt.plot(dm_trials, sigma_grid_w4, label="Boxcar width = 4", lw=2, color="#7c3aed", linestyle="--")
plt.title(r"Theoretical Noise $\sigma_{\text{eff}}$ vs. Dispersion Measure", fontsize=13)
plt.xlabel("Dispersion Measure (pc cm$^{-3}$)", fontsize=11)
plt.ylabel("Output Noise $\\sigma / \\sigma_0$", fontsize=11)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Calibrate DMT plane to true SNR
snr_map = dmt_result / sigma_grid_w1[:, np.newaxis]
print(f"Maximum candidate SNR: {np.max(snr_map):.2f} sigma")
```

:::{admonition} Takeaways
:class: dmt-summary

- Each tree level merges sub-bands in pairs, halving their number; after
  $\log_2 N_{\text{chans}}$ levels a single band holds every trial delay.
- `reset`, `advance` and `finalize` run the tree level by level, and
  `view_subband_data` gives a zero-copy view of any sub-band in between.
- With the default box smearing the noise sigma differs between DM trials;
  `get_effective_sigma_grid` gives it exactly from the plan.
:::

{doc}`Tutorial 3 <03_sparse_dm_grids_and_negative_dms>` covers sparse DM
grids and negative DMs.
