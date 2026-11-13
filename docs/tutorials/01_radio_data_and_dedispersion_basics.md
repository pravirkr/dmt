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

# Tutorial 1: Radio data and dedispersion

This first tutorial covers the physics of dispersion, the filterbank data
format, and a first FDMT run on simulated data.

## Interstellar dispersion

Radio signals from pulsars and fast radio bursts (FRBs) cross hundreds to
billions of light-years of the **interstellar medium (ISM)** and
**intergalactic medium (IGM)** before they reach a telescope.

The ISM is a tenuous, cold, unmagnetised plasma of free electrons and ions. In
such a plasma, the phase and group velocities of an electromagnetic wave depend
on frequency through the **plasma frequency** $\nu_p$:

$$\nu_p = \sqrt{\frac{n_e e^2}{\pi m_e}} \approx 8.98 \sqrt{n_e\,[\text{cm}^{-3}]}\;\text{kHz}$$

Observing frequencies $\nu$ (typically $100\,\text{MHz}$ to $8\,\text{GHz}$)
are far above $\nu_p$, so the refractive index is

$$\mu(\nu) = \sqrt{1 - \frac{\nu_p^2}{\nu^2}} \approx 1 - \frac{1}{2} \frac{\nu_p^2}{\nu^2}$$

Wave packets travel at the **group velocity** $v_g = c \,\mu(\nu) < c$, so the
low-frequency part of a broadband pulse arrives later than the high-frequency
part. The travel time over a distance $L$ is

$$t(\nu) = \int_0^L \frac{dl}{v_g} \approx \frac{L}{c} + \frac{e^2}{2\pi m_e c} \frac{1}{\nu^2} \int_0^L n_e(l)\,dl$$

The **dispersion measure (DM)** is the column density of free electrons along
the line of sight:

$$\text{DM} \equiv \int_0^L n_e(l)\,dl \quad [\text{pc}\,\text{cm}^{-3}]$$

The delay between a high frequency $\nu_{\text{high}}$ and a low frequency
$\nu_{\text{low}}$ is then

$$\Delta t = \mathcal{D} \cdot \text{DM} \left( \nu_{\text{low}}^{-2} - \nu_{\text{high}}^{-2} \right)$$

where $\mathcal{D}$ is the **dispersion constant**:

$$\mathcal{D} = \frac{e^2}{2\pi m_e c} \approx 4148.808\,\text{MHz}^2\,\text{pc}^{-1}\,\text{cm}^3\,\text{s}$$

```{note}
This is the physical value. Pulsar software traditionally uses the rounded
$\mathcal{D} = 1/(2.41\times10^{-4}) \approx 4149.38$ (Manchester & Taylor),
and so does `dmt` (`dmtlib.grid.DM_CONSTANT`); the two differ by 0.01%.
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Dispersion constant in MHz^2 pc^-1 cm^3 s, the value dmt uses
D_CONST = 1 / 2.41e-4

def dispersion_delay(f_mhz, dm):
    """Compute arrival time delay in seconds relative to infinite frequency."""
    return D_CONST * dm * (f_mhz ** -2.0)

# Example: L-band observation (1200 - 1600 MHz) with DM = 100 pc cm^-3
freqs = np.array([1200.0, 1600.0])
dm_test = 100.0

delays = dispersion_delay(freqs, dm_test)
total_delay_ms = (delays[0] - delays[-1]) * 1e3

print(f"Total dispersion delay across 1200-1600 MHz band for DM={dm_test} pc cm^-3:")
print(f"  Delta t = {total_delay_ms:.2f} ms")
```

## Filterbank data

A telescope backend digitises the voltages from the antennas, splits them into
frequency channels with a polyphase filterbank (PFB) or an FFT, detects total
power (Stokes I), and integrates over a sampling time $t_{\text{samp}}$
(typically $10\,\mu\text{s}$ to $1\,\text{ms}$).

The result is a **dynamic spectrum**, stored as a 2D **waterfall**:

$$I[c, s] \quad \text{where } c \in [0, N_{\text{chans}}-1] \text{ (frequency channels)}, \; s \in [0, N_{\text{samps}}-1] \text{ (time samples)}$$

The next cell simulates a waterfall with Gaussian radiometer noise and one
dispersed FRB.

```{code-cell} ipython3
# Simulation parameters
nchans = 256
f_min = 1200.0   # MHz
f_max = 1600.0   # MHz
tsamp = 1e-3     # 1 millisecond
nsamps = 1024

# Channel centres (channel 0 is the lowest; f_min and f_max are band edges)
df = (f_max - f_min) / nchans
f_chans = f_min + df * (np.arange(nchans) + 0.5)

# Compute per-channel sample delay for a pulse arriving at time t0
dm_sim = 150.0  # pc cm^-3
t0_samp = 200   # arrival time at f_max, the top of the band

delays_sec = dispersion_delay(f_chans, dm_sim) - dispersion_delay(f_max, dm_sim)
delays_samp = np.round(delays_sec / tsamp).astype(int)

# Create simulated waterfall with Gaussian radiometer noise
np.random.seed(42)
waterfall = np.random.normal(loc=0.0, scale=1.0, size=(nchans, nsamps)).astype(np.float32)

# Inject pulse track with amplitude 2.0 per channel
pulse_amp = 2.0
for c in range(nchans):
    arrival_samp = t0_samp + delays_samp[c]
    if 0 <= arrival_samp < nsamps:
        waterfall[c, arrival_samp] += pulse_amp

print(f"Injected pulse with DM = {dm_sim} pc cm^-3 at sample {t0_samp}.")
print(f"Maximum channel delay: {delays_samp[0]} samples ({delays_samp[0] * tsamp * 1e3:.1f} ms)")
```

```{code-cell} ipython3
# Visualize the time-frequency waterfall
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    waterfall,
    aspect="auto",
    origin="lower",
    extent=[0, nsamps, f_min, f_max],
    cmap="viridis",
)
ax.set_title(r"Simulated Dispersed FRB Waterfall $I(\nu, t)$", fontsize=14)
ax.set_xlabel("Time sample (1 ms)", fontsize=12)
ax.set_ylabel("Frequency (MHz)", fontsize=12)
fig.colorbar(im, ax=ax, label="Normalized Intensity")
plt.tight_layout()
plt.show()
```

## Why brute-force dedispersion is slow

The pulse is spread over hundreds of time samples. In any single channel or
time bin its signal-to-noise ratio (S/N) is below the noise.

To recover it, shift each channel back by its dispersion delay and sum over
frequency:

$$S(t; \text{DM}) = \sum_{c=0}^{N_{\text{chans}}-1} I\left[c, \, t + \Delta s(c, \text{DM})\right]$$

### Cost

In a blind survey the DM is unknown, so you test a grid of $N_{\text{DM}}$
trial values, often $N_{\text{DM}} \sim 1000 - 10{,}000$.

- **Brute force** takes
  $N_{\text{DM}} \times N_{\text{chans}} \times N_{\text{samps}}$
  operations. At SKA or CHIME scale ($N_{\text{chans}} = 16{,}384$,
  $N_{\text{samps}} = 10^5$) that is tens of teraflops, which easily exceeds
  real-time compute budgets.
- **FDMT.** The Fast Dispersion Measure Transform (FDMT) uses a recursive,
  divide-and-conquer tree, like the FFT. It combines pairs of adjacent
  sub-bands level by level, which brings the cost down to

$$\mathcal{O}\left(N_{\text{samps}} \cdot N_{\text{chans}} \cdot \log_2 N_{\text{chans}}\right)$$

That is a speed-up of roughly $\frac{N_{\text{DM}}}{\log_2 N_{\text{chans}}} \sim 100\times - 1000\times$.

## Dedispersion with DMT

Run `compute_fdmt` from `dmtlib` on the simulated waterfall:

```{code-cell} ipython3
from dmtlib import compute_fdmt

# Maximum delay trial to search (in samples)
dt_max = int(delays_samp[0] * 1.5)

# Run the Fast Dispersion Measure Transform
dmt_matrix, plan = compute_fdmt(
    waterfall,
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=tsamp,
    dt_max=dt_max,
    dt_min=0,
    mode="valid",
)

# Reshape DMT output into (N_dm_trials, N_time_samples)
ndms = plan.dmt_ndms
nsamps_out = plan.dmt_nsamps
dmt_grid = dmt_matrix.reshape(ndms, nsamps_out)

print(f"FDMT Plan summary:")
print(f"  Iterations: {plan.niters}")
print(f"  DM trials:  {ndms}")
print(f"  Output time samples: {nsamps_out}")
```

```{code-cell} ipython3
# Map delay trials back to physical DM (pc cm^-3)
dm_trials = plan.dm_grid_final

# Find candidate peak in the DM-time plane
peak_idx = np.unravel_index(np.argmax(dmt_grid), dmt_grid.shape)
best_dm_trial = dm_trials[peak_idx[0]]
best_time_samp = peak_idx[1]

print(f"Recovered peak:")
print(f"  True DM:      {dm_sim:.2f} pc cm^-3")
print(f"  Recovered DM: {best_dm_trial:.2f} pc cm^-3")
print(f"  Peak coordinate (DM trial index, time sample): {peak_idx}")

# Plot the DM-Time plane
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.imshow(
    dmt_grid,
    aspect="auto",
    origin="lower",
    extent=[0, nsamps_out, dm_trials[0], dm_trials[-1]],
    cmap="inferno",
)
ax1.set_title(r"FDMT Transform Plane $DMT(\text{DM}, t)$", fontsize=13)
ax1.set_xlabel("Output Time Sample", fontsize=11)
ax1.set_ylabel("Dispersion Measure (pc cm$^{-3}$)", fontsize=11)
ax1.axhline(best_dm_trial, color="cyan", linestyle="--", alpha=0.7, label=f"Best DM={best_dm_trial:.1f}")
ax1.legend(loc="upper right")
fig.colorbar(im1, ax=ax1, label="Integrated Power")

# Dedispersed time series profile at best DM
ax2.plot(dmt_grid[peak_idx[0], :], color="#2563eb", lw=1.5)
ax2.set_title(f"Dedispersed Pulse Profile at DM = {best_dm_trial:.1f} pc cm$^{{-3}}$", fontsize=13)
ax2.set_xlabel("Time Sample", fontsize=11)
ax2.set_ylabel("Integrated Flux (Arbitrary Units)", fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

The pulse was injected at sample 200 in the top channel, but the peak is at
sample 390.

```{note}
FDMT's time axis is the **arrival time at `f_min`**, the bottom of the band,
so the peak comes later by about the 189-sample delay across the band.
```

:::{admonition} Takeaways
:class: dmt-summary

- The dispersion delay is
  $\Delta t = \mathcal{D} \cdot \text{DM} \left( \nu_{\text{low}}^{-2} - \nu_{\text{high}}^{-2} \right)$,
  so low frequencies arrive later.
- Filterbank data is a waterfall $I[c, s]$ of frequency channels by time
  samples.
- Brute force costs $N_{\text{DM}} \times N_{\text{chans}} \times N_{\text{samps}}$;
  FDMT costs
  $\mathcal{O}\left(N_{\text{samps}} \cdot N_{\text{chans}} \cdot \log_2 N_{\text{chans}}\right)$.
- FDMT's time axis is the arrival time at `f_min`.
:::

{doc}`Tutorial 2 <02_incoherent_fdmt_pipeline>` looks inside the FDMT tree.
