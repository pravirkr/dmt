# Noise Calibration & SNR Scaling

Evaluating candidate statistical significance requires dividing integrated flux by the true background noise standard deviation to obtain standard Gaussian **Signal-to-Noise Ratio ($\text{SNR}$)** units.

However, in FDMT, the output noise variance is **not constant across DM trials**. This guide explains why and shows how to calibrate the transform plane using DMT's analytical variance engine.

---

## 1. Why Output Noise Varies with DM

Assume the input waterfall has zero-mean Gaussian radiometer noise with variance $\sigma_0^2 = 1.0$ per channel:

1. **Trial $\text{DM} = 0$**: Integrates samples along a straight vertical column ($N_{\text{chans}}$ independent channels). The theoretical variance is simply:
   $$\text{Var}(\text{DM}=0) = N_{\text{chans}} \cdot \sigma_0^2$$
2. **High DM Trials**: Sum along tilted trajectories. Because the tree combines overlapping subbands, intermediate subband sums share noise samples across adjacent delay branches, introducing noise correlation.
3. **Box smearing**: With `use_box_smearing=True` (the default) each channel contributes the weighted mean of its level-0 box of $L_c$ samples (see {ref}`box means <box-mean>`), so channels with more intra-channel smearing average more samples and carry less noise.
4. **Boxcar Convolutions**: Searching for pulses wider than 1 sample (e.g. boxcar filters of width $W \in \{2, 4, 8, 16\}$ samples) adds further time-domain correlation.

Evaluating candidates using a single global noise estimate $\sigma$ results in severe false alarm rates at low DMs or systematic sensitivity loss at high DMs.

---

## 2. Analytical Noise Tracking in DMT

Rather than wasting compute power estimating empirical variances across noisy output buffers, `dmt` computes the **exact theoretical noise propagation** directly from the plan geometry:

- `fdmt.get_effective_variance(dm_idx, boxcar_width=1)`: Returns theoretical output variance $\text{Var}(\text{DM}_k, W)$.
- `fdmt.get_effective_sigma(dm_idx, boxcar_width=1)`: Returns $\sigma_{\text{eff}} = \sqrt{\text{Var}}$.
- `fdmt.get_effective_sigma_grid(boxcar_width=1)`: Returns a 1D NumPy array of shape $(N_{\text{DM}},)$ containing $\sigma_{\text{eff}}$ for all DM trials.

For unit-variance white input, the variance of trial $k$ after a boxcar of
$W$ samples is

$$\text{Var}(\text{DM}_k, W) = \sum_c w_c^2\, V(L_c, W), \qquad
V(L, W) = (h - l + 1)\, l^2 + \frac{(2l - 1)\, l\, (l - 1)}{3},$$

with $L_c$ the level-0 box length of channel $c$ on that trial's path,
$w_c$ its box-mean weight, $h = \max(L_c, W)$ and $l = \min(L_c, W)$. $V$ is
the sum of the squared weights of a box of $L_c$ samples convolved with a
boxcar of $W$; for $L_c = 1$ it is $W$, and without box smearing the
variance is $N_{\text{chans}} W$. The `get_effective_variance*` methods (also
on `FDMTPlan`) include the weights.

(box-mean)=
## 3. Box means (`use_box_smearing`)

A level-0 row with delay $|dt| = s$ covers a box of $L = s + 1$ samples of a
channel. The matched filter for a channel smeared over $L$ samples is the box
**mean**, which weights every channel's contribution in proportion to its
S/N. With `use_box_smearing=True` the FDMT therefore averages each level-0
box: the sum of its $L$ samples times a weight $w(L) \approx 1/L$:

- **Float input:** the dyadic weight $w(L) = \mathrm{round}(2^b / L)\, 2^{-b}$
  with $b = \lceil \log_2 L \rceil + 2$, within 1/8 of $1/L$ (under 0.15% of
  S/N against the exact mean). It depends on $L$ alone. `plan.box_weights`
  lists the integer weights on the common scale $2^{-B}$, $B$ =
  `plan.box_fraction_bits` (C++ `get_box_weights()`,
  `get_box_fraction_bits()`).
- **Packed low-bit input:** the same mean rounded half up to $f(L)$
  fractional bits, the smallest $f$ with $3 \cdot 4^f \ge 50 L$, on the common
  fixed-point scale $2^{-F}$, $F$ = `plan.box_mean_bits`. The rounding noise is
  at most 2% of the box-mean noise for input with $\sigma \ge 0.5$ LSB (S/N
  loss ≤ 0.5%), and it keeps the integer tree (`int_tree`) narrow. Packed
  results are bit-identical with `int_tree` on and off and across backends,
  and differ from float input with the same values by at most
  $\sum_c 2^{-(f(L_c)+1)}$. The variance formula above does not include this
  small rounding noise. Without box smearing packed and float input agree
  exactly.

Consequences:

- **Output units.** The output is in box-mean units: a pulse that fills each
  channel's box at amplitude $a_c$ per sample gives about $\sum_c a_c$
  (roughly $N_{\text{chans}}$ times its mean amplitude). A constant input of 1
  gives $\sum_c w_c L_c \approx
  N_{\text{chans}}$ on every trial.
- **S/N.** Divide by `get_effective_sigma_grid()` as below; the weights are
  already in it.
- `FDMTFFT` uses the same dyadic float weights (no rounding: its level 0 is
  formed in the Fourier domain), and the fine FDMT of `CohFDMT` uses box means
  too (see [Coherent Hybrid Search](coherent_hybrid.md)).

---

## 4. Calibrating to True SNR Units

```python
import numpy as np
from dmtlib import FDMT

fdmt = FDMT(
    f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dt_max=150, mode="valid"
)

# Dedisperse block
dmt_matrix = fdmt.execute(waterfall) # shape: (n_dm, n_times)

# Query exact theoretical sigma grid
sigma_grid = fdmt.get_effective_sigma_grid(boxcar_width=1) # shape: (n_dm,)

# Normalize DMT plane to true Gaussian SNR units:
# (sigma_grid is broadcast along the time axis)
snr_map = dmt_matrix / sigma_grid[:, np.newaxis]

# Detect significant candidates (e.g. SNR > 8 sigma)
candidates = np.argwhere(snr_map > 8.0)
for dm_idx, t_idx in candidates:
    peak_snr = snr_map[dm_idx, t_idx]
    dm_val = fdmt.plan.dm_grid_final[dm_idx]
    print(f"Candidate detected at DM = {dm_val:.2f} pc cm^-3, time = {t_idx}: SNR = {peak_snr:.2f} sigma")
```
