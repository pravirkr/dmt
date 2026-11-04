# Periodicity searches and plain time series

The engine defaults suit streamed single-pulse searches. If you want a
dedispersed time series to Fourier transform, for a periodicity search or just
to look at, you need three different choices. This page explains them. The
measurements behind it come from two scripts in the repository,
`bench/scripts/fdmt_noise.py` and `bench/scripts/tone_recovery.py`, and from
{doc}`tutorial 7 <../tutorials/07_fdmt_noise_statistics>`.

For offline work, one call does all of it:

```python
import dmtlib

r = dmtlib.dedisperse(waterfall, f_min, f_max, tsamp, dms, engine="fdmt")
# r.data: (n_dm, n_out) white-noise time series, startup already removed
```

## Turn off box smearing

By default (`use_box_smearing=True`) FDMT replaces each channel's sample with
the mean of its level-0 box. That box is the $L_c = |dt_c| + 1$ samples
between the arrival times at the channel's upper and lower edges. For a single
pulse this is a matched filter for smearing within the channel. For noise it
is a low-pass filter:

$$S_y(f) = \sum_c \sigma_c^2\, w_c^2 \left[\frac{\sin(\pi f L_c)}{\sin(\pi f)}\right]^2,
\qquad w_c \approx 1/L_c .$$

The noise power then falls toward Nyquist, more steeply at higher DM, with
ripples where individual channels' boxes have nulls. A single sigma per trial
still normalises time-domain S/N correctly (see
[noise calibration](noise_calibration.md)), but no single number flattens this
spectrum. That is why `use_box_smearing` is an option.

With `use_box_smearing=False` each channel contributes a single sample, the
one at its upper-edge arrival time. Every output time then uses different
samples of every channel. White input gives white output with variance
`nchans`, distributed exactly like DDMT's output. Rounding delays to whole
samples changes how well a signal lines up, but it cannot correlate the noise.
FDMT-FFT and DDMT-FFT apply fractional delays as phase ramps, which keep the
noise white too.

```python
fdmt = dmtlib.FDMT(f_min, f_max, nchans, nsamps, tsamp,
                   dm_grid=dms, use_box_smearing=False)
```

## Drop the startup samples

In `mode="valid"`, FDMT returns `nsamps` outputs per call, so blocks join
exactly. After construction or `reset_history()` the past is zero-filled, and
the first `fdmt.startup_samples` outputs are partial sums over the channels
that have data so far. They are written like any other sample.

With zero-mean input the only effect is lower variance at the start. With real
data, which has a nonzero mean, the partial sums form a ramp. The ramp grows
with DM and puts a lot of power into the lowest Fourier bins. In tutorial 7, a
mean of 1 per sample produces several thousand times the white level in bins
1–4. **Drop the first `startup_samples` outputs once per observation** and the
excess is gone. DDMT withholds its incomplete samples itself.

In a streaming transient search, mask candidates in that interval instead.

```{warning}
Never call `reset_history()` between consecutive blocks of one observation.
```

## Know where each engine's time zero is

By default every engine refers output times to `f_min`, the bottom edge of
the band (`plan.f_ref`). The time of the band's last-arriving edge is when a
pulse is complete, which is what a real-time trigger wants. The engines
differ in which sample they take from each channel and in where output
sample 0 sits:

| Engine | Output sample `t` is the `f_ref` arrival at input sample | Sample taken from each channel |
| --- | --- | --- |
| FDMT, FDMT-FFT, FDMTBoxcar (`f_ref = f_min`) | `t` (FDMT-FFT valid mode: `t - output_latency`) | the box from upper to lower edge, or with `use_box_smearing=False` the upper-edge sample |
| DDMT, SDMT, DDMT-FFT (`f_ref = f_min` or `f_max`) | `t + plan.time_offset` (from the start of the stream) | the sample at the channel centre's delay |
| CohFDMT (`f_ref = f_min`) | `t0 + t * tsamp` (seconds, `t0` per block) | the coherently dedispersed channel, shifted from its centre |

`plan.time_offset` of the DDMT family is how far the earliest channel centre
arrives ahead of `f_ref` over the plan's DM trials, rounded up. With the
default `f_min` it follows the plan's largest DM, so align the outputs of two
DDMT plans with different DM grids by their offsets, not by index.

**Offline pulsar searches.** SIGPROC and PRESTO time series start at the
file's first sample at the top of the band. Pass `f_ref=f_max` to DDMT,
SDMT, DDMT-FFT or `dedisperse()`: then `plan.time_offset == 0`, output sample
`t` is input sample `t` at `f_max` for every DM, the file's `tstart`
is the epoch of every row, and the incomplete samples are at the end. The
engines only read a different delay table; speed and history are the same.

```{note}
`f_min` and `f_max` are always band edges, never channel centres, and rows
are always ascending in frequency (row 0 is the lowest channel). See
{ref}`channel order <channel-order>` for
SIGPROC headers.
```

To compare times between engines, refer them to one frequency with
`dmtlib.dispersion_delay(dm, f_lo, f_hi)`, or use
`Dedispersed.arrival_times()` below. One consequence of the upper-edge sample:
with box smearing off, FDMT times a strongly smeared pulse by its leading edge,
so its arrival times come out late by up to about half a channel's smearing
time. In a test at DM 60 with 64 channels that was 2.5 ms; with 256 channels
it was under one sample.

## `dmtlib.dedisperse`

For offline work, this gives a time series per DM without any history to
manage:

```python
r = dmtlib.dedisperse(waterfall, f_min, f_max, tsamp, dms,
                      engine="fdmt")   # or "fdmt_fft", "ddmt", "sdmt", "ddmt_fft"
# Direct engines only: r.start == 0, sample 0 = the first input sample at f_max
r = dmtlib.dedisperse(waterfall, f_min, f_max, tsamp, dms, engine="ddmt",
                      f_ref=f_max)
r.data             # (n_dm, n_out), complete samples only
r.dms              # effective DMs (FDMT snaps to its delay grid)
r.f_ref, r.start   # r.data[:, i] arrives at f_ref at input sample r.start + i
r.arrival_times()  # (n_dm, n_out) in seconds, referred to infinite frequency
```

```{tip}
`use_box_smearing` defaults to `False` here, unlike in the engine classes.
```

## Which engine keeps the most signal

Everything above concerns noise. How much *signal* survives at high Fourier
frequencies differs between engines, because channels are added with small
timing errors and at high frequency those errors become phase errors.
`bench/scripts/tone_recovery.py` measures it. It injects sinusoids smeared
across each channel's bandwidth and sample bin (1024 channels, 400–800 MHz,
1 ms sampling), divides each engine's power by its own noise power at that
frequency, and averages 16 tones around each frequency. The figure shows that
S/N as a fraction of a perfectly aligned sum of the same smeared signal:

```{image} ../_static/periodicity/light/tone_recovery.png
:class: only-light
:alt: Recovered S/N of smeared sinusoids against Fourier frequency at DM 50 and DM 200, for DDMT, FDMT, FDMT with box smearing, FDMT-FFT and DDMT-FFT
```
```{image} ../_static/periodicity/dark/tone_recovery.png
:class: only-dark
:alt: Recovered S/N of smeared sinusoids against Fourier frequency at DM 50 and DM 200, for DDMT, FDMT, FDMT with box smearing, FDMT-FFT and DDMT-FFT
```

At high DM even the ideal sum has little left near Nyquist, because the
channels themselves have smeared the signal out; the curves show how much of
what is left each engine keeps.

What to take from it:

- **Low frequencies.** Below about 0.1 cycles per sample all engines are
  within a few per cent of each other.
- **Integer FDMT** falls behind DDMT toward Nyquist. Its recursive delay
  rounding scatters the channel phases more than DDMT's single rounding does.
- **Fractional delays keep the most.** DDMT-FFT and FDMT-FFT stay above
  DDMT at high frequencies; FDMT-FFT is much faster when there are many DMs.
  For millisecond pulsars or high-DM searches, use one of them.
- **Box smearing.** With its known noise spectrum divided out, box-smeared
  FDMT is competitive up to about a quarter of the sampling rate and falls
  off fastest above it. A running-median normaliser does not recover even
  that.

No engine recovers power that the channel smearing has already removed.

```{seealso}
- [Noise calibration](noise_calibration.md): the exact sigma of every DM trial.
- [Streaming and history](streaming_and_history.md): the cold start and `reset_history()`.
- [Fourier-domain variants](fourier_variants.md): FDMT-FFT and DDMT-FFT.
- {doc}`Tutorial 7 <../tutorials/07_fdmt_noise_statistics>`: the noise measurements.
```
