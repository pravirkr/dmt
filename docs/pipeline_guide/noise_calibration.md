# Noise Calibration & SNR Scaling

Evaluating candidate statistical significance requires dividing integrated flux by the true background noise standard deviation to obtain standard Gaussian **Signal-to-Noise Ratio ($\text{SNR}$)** units.

However, in FDMT, the output noise variance is **not constant across DM trials**. This guide explains why and shows how to calibrate the transform plane using DMT's analytical variance engine.

---

## 1. Why Output Noise Varies with DM

Assume the input waterfall has zero-mean Gaussian radiometer noise with variance $\sigma_0^2 = 1.0$ per channel:

1. **Trial $\text{DM} = 0$**: Integrates samples along a straight vertical column ($N_{\text{chans}}$ independent channels). The theoretical variance is simply:
   $$\text{Var}(\text{DM}=0) = N_{\text{chans}} \cdot \sigma_0^2$$
2. **High DM Trials**: Sum along tilted trajectories. Because the tree combines overlapping subbands, intermediate subband sums share noise samples across adjacent delay branches, introducing noise correlation.
3. **Boxcar Convolutions**: Searching for pulses wider than 1 sample (e.g. boxcar filters of width $W \in \{2, 4, 8, 16\}$ samples) adds further time-domain correlation.

Evaluating candidates using a single global noise estimate $\sigma$ results in severe false alarm rates at low DMs or systematic sensitivity loss at high DMs.

---

## 2. Analytical Noise Tracking in DMT

Rather than wasting compute power estimating empirical variances across noisy output buffers, `dmt` computes the **exact theoretical noise propagation** directly from the plan geometry:

- `fdmt.get_effective_variance(dm_idx, boxcar_width=1)`: Returns theoretical output variance $\text{Var}(\text{DM}_k, W)$.
- `fdmt.get_effective_sigma(dm_idx, boxcar_width=1)`: Returns $\sigma_{\text{eff}} = \sqrt{\text{Var}}$.
- `fdmt.get_effective_sigma_grid(boxcar_width=1)`: Returns a 1D NumPy array of shape $(N_{\text{DM}},)$ containing $\sigma_{\text{eff}}$ for all DM trials.

---

## 3. Calibrating to True SNR Units

```python
import numpy as np
from dmtlib import FDMTCPU

fdmt = FDMTCPU(
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
