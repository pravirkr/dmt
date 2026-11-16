# DM grids

Every engine searches whatever DM trials you give it (`FDMTPlan` rounds
them to integer delays across the band). `dmt` also generates grids from the
observation and a sensitivity target, in C++ (`dmt/common/dm_grids.hpp`) and
Python:

| Call | Returns |
|---|---|
| `dmt::plans::generate_dm_grid(config, dm_min, dm_max)` | trial DMs by dmt's rule |
| `dmt::plans::generate_levin_dm_grid(config, dm_min, dm_max)` | trial DMs by the Levin rule |
| `dmt::plans::calculate_snr_loss(config, trials, dms)` | the S/N a grid loses at any DM |
| `dmtlib.generate_dm_grid`, `dmtlib.generate_levin_dm_grid`, `dmtlib.calculate_snr_loss` | the same, from Python |

All take frequencies in MHz and times (`tsamp`, `pulse_width`) in seconds,
and return DMs in pc cm⁻³ (float64).

## Two rules

**`generate_dm_grid` (dmt's rule).** Trials are spaced so a pulse halfway
between two keeps `1 - max_snr_loss` (0.95 by default) of the S/N it has on a
trial, after the boxcar search. The S/N is modelled over every channel, in
up to 64 groups: the misalignment a DM error causes in each, its smearing
inside the channel and in FDMT's level-0 box, the sampling time, the
pulse's own width, and the best boxcar width with the noise it adds. So:

- it holds on very wide bands, where one frequency can't describe the
  band;
- it supports negative DMs (mirroring the positive trials) and never stops
  short at high DM;
- it takes `chan_dm` for coherently dedispersed data (below) and
  `kill_mask` for dead channels;
- `tol` (> 1) sets `max_snr_loss = 1 - 1/tol` instead.

**`generate_levin_dm_grid`.** The classic Levin (2012) rule, as in Heimdall,
kept exactly as it is: smearing and sweep are both taken at the band centre,
with the sweep as a straight line (its slope there), and pulses may broaden
by at most `tol` (default 1.25). The benchmarks use it, so their grids stay
fixed.

Both take `piecewise=True`: uniform segments, split where the step halves or
doubles, each at the smallest step inside it and at least `min_run`
trials long. That is the grid `DDMTFFT` needs for its NUFFT path
([Fourier-domain variants](fourier_variants.md)). It never spaces trials
wider than the plain grid.

`calculate_snr_loss` evaluates the model of `generate_dm_grid` for any grid,
so you can check a hand-made or Levin grid against the same target.

## Coherently dedispersed data

If the channels were coherently dedispersed to some DM before the
filterbank was formed, a pulse at that DM has no smearing inside a channel,
and the smearing elsewhere grows with `|DM - chan_dm|`. Pass that DM as
`chan_dm` to `generate_dm_grid`: the trials then get dense around it, where
pulses are sharpest. The Levin rule doesn't take it.

FDMT's level-0 box smearing assumes ordinary data unless told otherwise: it
averages each channel over the whole smearing span at the trial DM, which
dilutes sharp pulses near `chan_dm`. Build the FDMT plan with the same DM,
`FDMTPlan(..., chan_dm=413.0)`, and pass that plan to `FDMT`, `FDMTBoxcar`,
`FDMTFFT` or `FDMTFFTBoxcar`. Each channel's box then covers only the
residual sweep, about `|dt(DM) - dt(chan_dm)|` samples, centred on the
channel's arrival, and the noise model follows. Outputs then lag the
bottom-edge arrival by `plan.level0_causal_delay` samples (39 on Parkes UWL
at 64 µs with `chan_dm` 413). It needs `mode="valid"`.

Don't switch box smearing off instead: on simulated UWL bursts at
`chan_dm` 413 that keeps less S/N than the default at every DM and width
tried. Recovered S/N as a fraction of the ideal `sqrt(nchans * width)`, at
DM 413:

| Pulse width | `use_box_smearing=False` | default | `chan_dm` plan |
|---:|---:|---:|---:|
| 1 sample | 0.37 | 0.52 | 0.58 |
| 4 samples | 0.64 | 0.80 | 0.89 |
| 16 samples | 0.83 | 0.94 | 0.97 |

The gain is largest near `chan_dm` and on wide bands. Far from it, or on
narrower bands, the `chan_dm` plan can keep slightly less than the default
(UWL at DM 100 and one sample: 0.50 against 0.51; CRYO-PAF L-band at DM 413
and four samples: 0.87 against 0.90). Much of the remaining loss at one
sample is the FDMT tree's integer delay rounding: on CRYO-PAF L-band,
`FDMTFFT` with fractional delays raises it from 0.62 to 0.71.

## Very wide bands

The Levin rule describes the whole band by its centre, which only works when
the band is narrow compared with its centre frequency. On Parkes UWL
(704–4032 MHz) it takes the sweep's slope at the centre: 2080 µs per unit
DM, against a true sweep of 8116, so its trials are up to about four times
too far apart. `generate_dm_grid` sums the pulse over every channel instead,
so its target holds across such a band.

## How close the target holds

The model describes each channel's smeared, box-averaged pulse by a
Gaussian of the same width. Injected pulses on a GPU `FDMTBoxcar` (one-sample
pulses halfway between trials, averaged over arrival phase, DM 100-2000)
show how that plays out:

- On bands with many channels the grid is conservative: Parkes UWL, MeerKAT
  L-band and CRYO-PAF L-band lose 0.1-0.65 of `max_snr_loss`.
- Where intra-channel smearing dominates (CRYO-PAF at 512 µs, Parkes HTRU at
  64 µs, DM 900-2000) the true response is closer to a triangle, which
  loses S/N faster for small offsets: about 6.5-7% at a 5% target and 11% at
  10%, but under target at 20%.

FDMT's integer delays add scatter of their own at low DM, where a step is
only a few samples. For a tighter guarantee in the smear-dominated case,
ask for a smaller `max_snr_loss`.

## Example

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
import dmtlib

# Parkes UWL, 64 us, unresolved pulses (pulse_width 0), 5% loss.
dms = dmtlib.generate_dm_grid(704.0, 4032.0, 6656, 64e-6, 0.0, 2000.0)
# Coherently dedispersed at DM 413: denser trials around it.
dms_coh = dmtlib.generate_dm_grid(704.0, 4032.0, 6656, 64e-6, 0.0, 2000.0,
                                  chan_dm=413.0)
plan = dmtlib.FDMTPlan(704.0, 4032.0, 6656, 65536, 64e-6, dm_grid=dms_coh,
                       chan_dm=413.0)
# What a Levin grid loses at its midpoints, on the same model.
levin = dmtlib.generate_levin_dm_grid(704.0, 4032.0, 6656, 64e-6, 0.0, 2000.0)
loss = dmtlib.calculate_snr_loss(0.5 * (levin[1:] + levin[:-1]), levin,
                                 704.0, 4032.0, 6656, 64e-6)
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
#include <dmt/dmt.hpp>

dmt::plans::DMGridConfig cfg;
cfg.f_min       = 704.0;   // MHz
cfg.f_max       = 4032.0;
cfg.nchans      = 6656;
cfg.tsamp       = 64e-6;   // s
cfg.pulse_width = 64e-6;   // s
cfg.chan_dm     = 413.0;   // coherently dedispersed at DM 413
cfg.max_snr_loss = 0.05;
const auto dms  = dmt::plans::generate_dm_grid(cfg, 0.0, 2000.0);
cfg.chan_dm     = 0.0;     // the Levin rule does not take chan_dm
cfg.piecewise   = true;    // piecewise uniform, for DDMTFFT
const auto pw   = dmt::plans::generate_levin_dm_grid(cfg, 0.0, 2000.0);
```
:::
::::
