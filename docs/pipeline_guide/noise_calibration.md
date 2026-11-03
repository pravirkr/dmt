# Noise calibration

To turn an FDMT output into a signal-to-noise ratio you divide by the noise
standard deviation of that DM trial. With the default box smearing that
sigma is **different for every trial**. A single global sigma over-reports
candidates at low DM and loses sensitivity at high DM. `dmt` computes the
exact sigma of every trial from the plan, so you never have to estimate it
from the data.

## Where the variation comes from

Take white Gaussian input with unit variance per channel.

- **Without box smearing** (`use_box_smearing=False`), each output sample is
  the sum of one sample per channel, so every trial has variance
  $N_\text{chans}$, whatever the DM. The tree shares partial sums between
  neighbouring DM trials. That correlates *different* trials with each other,
  but it does not change the variance of any one trial.
- **With box smearing** (the default), each channel contributes the mean of a
  box of $L_c$ samples, where $L_c$ grows with DM and toward low frequencies
  (see {ref}`box means <box-mean>` below). A mean of $L_c$ samples has
  variance of about $1/L_c$, so the total variance falls with DM.
- **A boxcar** of $W$ samples, used to search for wider pulses, adds the
  overlap between the boxcar and each channel's box.

## The exact sigma

```python
fdmt.get_effective_variance(dm_idx, boxcar_width=1)   # one trial
fdmt.get_effective_sigma(dm_idx, boxcar_width=1)      # its square root
fdmt.get_effective_sigma_grid(boxcar_width=1)         # all trials, shape (n_dm,)
```

The same methods are on `FDMTPlan`. Pass `use_box_smearing` if you query the
plan directly. For unit-variance white input the variance of trial $k$ after a
boxcar of $W$ samples is

$$\text{Var}(\text{DM}_k, W) = \sum_c w_c^2\, V(L_c, W), \qquad
V(L, W) = (h - l + 1)\, l^2 + \frac{(2l - 1)\, l\, (l - 1)}{3},$$

where $L_c$ is the level-0 box length of channel $c$ on that trial's path,
$w_c$ its box-mean weight, $h = \max(L_c, W)$ and $l = \min(L_c, W)$. $V$ is
the sum of the squared weights of a box of $L_c$ samples convolved with a
boxcar of $W$. For $L_c = 1$ it is $W$, so without box smearing the variance
is $N_\text{chans} W$.

Normalising a block then looks like this:

```python
import numpy as np
from dmtlib import FDMT

fdmt = FDMT(f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
            tsamp=1e-3, dt_max=150)

dmt_plane = fdmt.execute(waterfall)                  # (n_dm, n_times)
sigma = fdmt.get_effective_sigma_grid(boxcar_width=1)
snr = dmt_plane / sigma[:, None]

for dm_idx, t_idx in np.argwhere(snr > 8.0):
    print(f"DM {fdmt.dm_grid_final[dm_idx]:.2f}, sample {t_idx}, "
          f"S/N {snr[dm_idx, t_idx]:.1f}")
```

```{warning}
The sigma assumes independent channels with **zero mean and equal, unit
variance**. Subtract each channel's mean and divide by its standard deviation
first, or work out the scale for your own channel variances. On raw data the
S/N values are meaningless.
```

(box-mean)=
## Box means

A level-0 row with delay $|dt| = s$ covers $L = s + 1$ samples of a channel,
the samples between the arrival times at its upper and lower edges. For a
pulse smeared across those $L$ samples, the matched filter is the **box
mean**. With `use_box_smearing=True` FDMT therefore sums the $L$ samples and
multiplies by a weight $w(L) \approx 1/L$.

- **Output units.** A pulse that fills each channel's box at amplitude $a_c$
  gives about $\sum_c a_c$. A constant input of 1 gives
  $\sum_c w_c L_c \approx N_\text{chans}$ on every trial.
- **Other engines.** `FDMTFFT` uses the same float weights. The fine FDMT
  inside `CohFDMT` uses box means too.
- **Noise colour.** Box means correlate neighbouring output samples, so the
  noise power spectrum falls toward Nyquist, with ripples that depend on DM.
  The sigma above is still correct for time-domain S/N, but no single number
  makes the spectrum flat.

```{tip}
For a periodicity or power-spectrum search, use `use_box_smearing=False`. The
output is then white and has the same distribution as DDMT's. See
[periodicity searches](periodicity.md).
```

:::{dropdown} How the weights are rounded
:icon: info

- **Float input.** The weight is dyadic,
  $w(L) = \mathrm{round}(2^b / L)\, 2^{-b}$ with $b = \lceil \log_2 L \rceil + 2$.
  It is within 1/8 of $1/L$, which costs under 0.15% of S/N against the exact
  mean. `plan.box_weights` lists the integer weights on the common scale
  $2^{-B}$, with $B$ = `plan.box_fraction_bits`.
- **Packed low-bit input.** The mean is rounded half up to $f(L)$ fractional
  bits, the smallest $f$ with $3 \cdot 4^f \ge 50 L$, on the scale $2^{-F}$
  with $F$ = `plan.box_mean_bits`. That keeps the integer tree narrow. The
  rounding noise is at most 2% of the box-mean noise for input with
  $\sigma \ge 0.5$ LSB (at most 0.5% S/N loss); the variance formula above
  leaves it out.
- **Agreement.** Packed results are bit-identical with `int_tree` on or off
  and across backends. They differ from float input with the same values by
  at most $\sum_c 2^{-(f(L_c)+1)}$. Without box smearing, packed and float
  input agree exactly.
:::
