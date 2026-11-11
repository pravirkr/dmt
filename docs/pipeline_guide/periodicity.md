# Periodicity searches and plain time series

The engine defaults suit streamed single-pulse searches. For a time series to
Fourier transform, change three things:

1. **`use_box_smearing=False`**, so the noise is white.
2. **Drop the first `startup_samples` outputs** once per observation.
3. **Know each engine's time zero** when you compare or fold.

Offline, `dmtlib.dedisperse` does all three.

## `dmtlib.dedisperse`

```python
import dmtlib

r = dmtlib.dedisperse(waterfall, f_min, f_max, tsamp, dms,
                      engine="fdmt")   # or "fdmt_fft", "ddmt", "sdmt", "ddmt_fft"
r.data             # (n_dm, n_out) white time series, complete samples only
r.dms              # effective DMs (FDMT snaps to its delay grid)
r.f_ref, r.start   # r.data[:, i] arrives at f_ref at input sample r.start + i
r.arrival_times()  # (n_dm, n_out) in seconds, referred to infinite frequency
```

`use_box_smearing` defaults to `False` here, unlike in the engine classes.

## White noise: turn off box smearing

Box smearing replaces each channel's sample with the mean of its
$L_c = |dt_c| + 1$ sample box ({ref}`box means <box-mean>`). For noise that is
a low-pass filter:

$$S_y(f) = \sum_c \sigma_c^2\, w_c^2 \left[\frac{\sin(\pi f L_c)}{\sin(\pi f)}\right]^2,
\qquad w_c \approx 1/L_c ,$$

falling toward Nyquist, more steeply at high DM, with ripples at each box's
nulls. No single number flattens it.

With `use_box_smearing=False` each channel contributes one sample, at its
upper-edge arrival time. White input then gives white output with variance
`nchans`, distributed exactly like DDMT's. FDMT-FFT and DDMT-FFT also keep the
noise white.

```python
fdmt = dmtlib.FDMT(f_min, f_max, nchans, nsamps, tsamp,
                   dm_grid=dms, use_box_smearing=False)
```

## Drop the startup samples

After a cold start the first `fdmt.startup_samples` outputs are partial sums
(see [streaming](streaming_and_history.md)). On real data, which has a nonzero
mean, they form a ramp that grows with DM. In tutorial 7 a mean of 1 per
sample put several thousand times the white level into bins 1–4. Dropping
them once per observation removes it. DDMT withholds these samples itself.

## Time zero of each engine

Every engine refers output times to `plan.f_ref`, by default `f_min`: a pulse
is complete when its lowest frequency arrives, which suits real-time triggers.

| Engine | Output sample `t` is the `f_ref` arrival at input sample | Sample taken from each channel |
| --- | --- | --- |
| FDMT, FDMT-FFT, FDMTBoxcar | `t` (FDMT-FFT valid mode: `t - output_latency`) | the box from upper to lower edge; without box smearing, the upper-edge sample |
| DDMT, SDMT, DDMT-FFT | `t + plan.time_offset` | the channel centre |
| CohFDMT | `t0 + t * tsamp` (seconds, `t0` per block) | the coherently dedispersed channel, shifted from its centre |

- **`plan.time_offset`** follows the plan's largest DM. Align two DDMT plans
  with different DM grids by their offsets, not by index.
- **SIGPROC/PRESTO-style series** start at the top of the band. Pass
  `f_ref=f_max` to DDMT, SDMT, DDMT-FFT or `dedisperse()`: `time_offset` is
  then 0, the file's `tstart` is every row's epoch, and the incomplete samples
  are at the end. Speed and history are unchanged.
- **Comparing engines.** Refer times to one frequency with
  `dmtlib.dispersion_delay(dm, f_lo, f_hi)`, or use `arrival_times()`.
- **Leading-edge timing.** Without box smearing, FDMT times a strongly smeared
  pulse by its leading edge, up to about half a channel's smearing time late
  (2.5 ms at DM 60 with 64 channels; under one sample with 256).

```{note}
`f_min` and `f_max` are band edges, not channel centres, and row 0 is always
the lowest channel. See {ref}`channel order <channel-order>` for SIGPROC
headers.
```

## Which engine keeps the most signal

Channel timing errors become phase errors at high Fourier frequency, so
engines keep different amounts of signal there. `bench/scripts/tone_recovery.py`
injects smeared sinusoids (1024 channels, 400–800 MHz, 1 ms sampling) and
plots each engine's S/N as a fraction of a perfectly aligned sum:

```{image} ../_static/periodicity/light/tone_recovery.png
:class: only-light
:alt: Recovered S/N of smeared sinusoids against Fourier frequency at DM 50 and DM 200, for DDMT, FDMT, FDMT with box smearing, FDMT-FFT and DDMT-FFT
```
```{image} ../_static/periodicity/dark/tone_recovery.png
:class: only-dark
:alt: Recovered S/N of smeared sinusoids against Fourier frequency at DM 50 and DM 200, for DDMT, FDMT, FDMT with box smearing, FDMT-FFT and DDMT-FFT
```

- **Below ~0.1 cycles per sample** all engines agree within a few per cent.
- **Integer FDMT** falls behind DDMT toward Nyquist: its per-merge rounding
  scatters channel phases more than DDMT's single rounding.
- **DDMT-FFT and FDMT-FFT keep the most** at high frequency. Use one of them
  for millisecond pulsars or high DM; FDMT-FFT is much faster with many DMs.
- **Box-smeared FDMT**, with its noise spectrum divided out, is competitive up
  to about a quarter of the sampling rate and falls off fastest above it.

No engine recovers what the channel smearing itself has removed.

```{seealso}
{doc}`Tutorial 7 <../tutorials/07_fdmt_noise_statistics>` and
`bench/scripts/fdmt_noise.py` for the noise measurements.
```
