# Noise calibration

With the default box smearing, the FDMT noise sigma is **different for every
DM trial**. Dividing by one global sigma over-reports candidates at low DM and
loses sensitivity at high DM. `dmt` computes each trial's exact sigma from the
plan; you don't estimate it from the data.

## Getting the sigma

```python
fdmt.get_effective_sigma_grid(boxcar_width=1)        # all trials, shape (n_dm,)
fdmt.get_effective_sigma(dm_idx, boxcar_width=1)     # one trial
fdmt.get_effective_variance(dm_idx, boxcar_width=1)  # its square
```

```python
dmt_plane = fdmt.execute(waterfall)                  # (n_dm, n_times)
snr = dmt_plane / fdmt.get_effective_sigma_grid()[:, None]
```

```{warning}
The sigma assumes independent channels with **zero mean and unit variance**.
Normalise each channel first; on raw data the S/N is meaningless.
It also assumes white noise: slow drifts (red noise) make wide boxes
noisier than their sigma, so detrend each channel upstream on timescales
several times the widest box, and keep boxcars to about 100 ms or less.
```

- **`FDMTPlan`** has the same methods. Pass `use_box_smearing` and
  `kill_mask=` when you query the plan directly.
- **Kill masks.** Killed channels read as zero and drop out of the sum.
  `FDMTBoxcar`, `FDMTFFTBoxcar` and `FDMTFFT` apply their own `kill_mask=`.
- **Stepper subbands.** `FDMTFFT.get_subband_sigma_grid(level, subband,
  boxcar_width)` (or `FDMTPlan.get_subband_sigma_grid` with the mask) gives
  the sigma of each subband view.
- **Per-block masks.** `FDMTBoxcar.execute(block, kill_mask=...)` uses the
  sigma of that block's mask (ANDed with the constructor's). For the first
  `S + W - 1` endpoints after a mask change it uses the sigma of the OR of
  both masks, an upper bound, so S/N there is slightly low, never high.
- **Don't rescale by hand.** Zeroing channels and scaling the S/N by one
  factor such as `sqrt(N / G)` is wrong: the right factor differs between
  rows and DM trials.

## Why it varies

For unit-variance white input:

- **Without box smearing**, every trial sums one sample per channel, so its
  variance is $N_\text{chans}$ at every DM.
- **With box smearing**, each channel contributes the mean of $L_c$ samples
  (variance about $1/L_c$), and $L_c$ grows with DM and toward low
  frequencies, so the variance falls with DM.
- **A boxcar** of $W$ samples adds its overlap with each channel's box.

:::{dropdown} The formula
:icon: info

$$\text{Var}(\text{DM}_k, W) = \sum_c w_c^2\, V(L_c, W), \qquad
V(L, W) = (h - l + 1)\, l^2 + \frac{(2l - 1)\, l\, (l - 1)}{3},$$

where $L_c$ is the level-0 box length of channel $c$ on trial $k$'s path,
$w_c$ its box-mean weight, $h = \max(L_c, W)$ and $l = \min(L_c, W)$. $V$ is
the sum of the squared weights of a box of $L_c$ convolved with a boxcar of
$W$. For $L_c = 1$ it is $W$, so without box smearing the variance is
$N_\text{chans} W$.
:::

(box-mean)=
## Box means

A level-0 row with delay $|dt| = s$ covers $L = s + 1$ samples of a channel,
between the arrival times at its upper and lower edges. With
`use_box_smearing=True`, FDMT sums those samples and multiplies by
$w(L) \approx 1/L$: the matched filter for a pulse smeared across the channel.

- **Units.** A pulse filling each channel's box at amplitude $a_c$ gives about
  $\sum_c a_c$; a constant input of 1 gives about $N_\text{chans}$ on every
  trial.
- **Other engines.** `FDMTFFT` and the fine FDMT inside `CohFDMT` use the same
  weights.
- **Noise colour.** Box means correlate neighbouring samples, so the noise
  spectrum falls toward Nyquist. The sigma is still right for time-domain
  S/N; for power spectra use `use_box_smearing=False` (see
  [periodicity searches](periodicity.md)).

:::{dropdown} How the weights are rounded
:icon: info

- **Float input.** The weight is dyadic,
  $w(L) = \mathrm{round}(2^b / L)\, 2^{-b}$ with $b = \lceil \log_2 L \rceil + 2$,
  within 1/8 of $1/L$ (under 0.15% S/N loss). `plan.box_weights` lists the
  integer weights on the scale $2^{-B}$, $B$ = `plan.box_fraction_bits`.
- **Packed low-bit input.** The mean is rounded half up to $f(L)$ fractional
  bits, the smallest $f$ with $3 \cdot 4^f \ge 50 L$, on the scale $2^{-F}$,
  $F$ = `plan.box_mean_bits`. For input with $\sigma \ge 0.5$ LSB the rounding
  noise is at most 2% of the box-mean noise (at most 0.5% S/N loss); the
  variance formula leaves it out.
- **Agreement.** Packed results are bit-identical with `int_tree` on or off
  and across backends, and differ from float input with the same values by at
  most $\sum_c 2^{-(f(L_c)+1)}$. Without box smearing they agree exactly.
:::
