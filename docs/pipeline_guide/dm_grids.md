# DM grids

Every engine searches whatever DM or delay trials you give it. `dmt` also
generates grids from the observation and a sensitivity target, in C++
(`dmt/common/dm_grids.hpp`) and Python:

| Call | Returns |
|---|---|
| `dmt::plans::generate_dm_grid(config, dm_min, dm_max)` | trial DMs (pc cm⁻³) |
| `dmt::plans::generate_dt_grid(config, dt_min, dt_max)` | integer delays across the band, for `FDMTPlan` |
| `dmtlib.generate_dm_grid(...)`, `dmtlib.generate_dt_grid(...)` | the same, from Python |
| `dmtlib.generate_optimal_dm_grid(...)`, `dmtlib.generate_optimal_dt_grid(...)` | the same, with the pulse width in samples |

The C++ calls and `dmtlib.generate_dm_grid` take frequencies in MHz and
times (`tsamp`, `pulse_width`) in seconds.

## Two rules

Both rules ask the same question: how far apart can two trials be before a
pulse halfway between them broadens too much? The pulse's effective width
combines its intrinsic width, the sampling time and the dispersion smearing
inside one channel. Searching it half a step off misaligns it across the
band, and the rule allows the width to grow by at most a factor `tol`.

`tol` is the knob of both rules. The S/N-loss rule also takes it as
`max_snr_loss = 1 - 1/tol` (5% is `tol` = 1.053).

**`snr_loss` (default).** dmt's rule:

- the misalignment uses the exact dispersion sweep across the band;
- the smearing is that of the bottom channel, the widest one;
- supports negative DMs, and never stops short at high DM;
- takes `chan_dm` for coherently dedispersed data (below).

**`levin` and `levin_piecewise`.** The classic Levin (2012) rule, as in
Heimdall, kept exactly as it is:

- smearing and sweep are both taken at the band centre, with the sweep as a
  straight line (its slope there);
- `levin_piecewise` makes it piecewise uniform, the grid `DDMTFFT` needs for
  its NUFFT path ([Fourier-domain variants](fourier_variants.md)).

## Coherently dedispersed data

If the channels were coherently dedispersed to some DM before the
filterbank was formed, a pulse at that DM has no smearing inside a channel,
and the smearing elsewhere grows with `|DM - chan_dm|`. Pass that DM as
`chan_dm` (S/N-loss rule): the trials then get dense around it, where
pulses are sharpest. The Levin rules don't take it.

FDMT's `use_box_smearing=True` (the default) also assumes ordinary data: it
averages each channel over the smearing span at the trial DM. On coherently
dedispersed data that dilutes sharp pulses near `chan_dm`, so use
`use_box_smearing=False` there.

## Very wide bands

Both rules describe a whole band by one frequency, which only works when
the band is narrow compared with its centre frequency. On a band like Parkes
UWL (704–4032 MHz) neither keeps its promise everywhere:

- **Levin** takes the sweep's slope at the centre: on UWL that's 2080 µs per
  unit DM, against a true sweep of 8116. Its trials are up to about four
  times too far apart.
- **`snr_loss`** has the exact sweep, but its bottom-channel smearing is 38×
  that at the centre, so at high DM it treats pulses as broader than most of
  the band sees them. In a model of the UWL band, it loses up to about 20%
  at high DM instead of 5%.

On such bands, check the grid with injected pulses, or pass a tighter `tol`.
A rule that sums the pulse over every channel is planned.

## Example

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
import dmtlib

# Parkes UWL, 64 us; pulse width in samples.
dms = dmtlib.generate_optimal_dm_grid(704.0, 4032.0, 6656, 64e-6, 2000.0,
                                      pulse_width=1.0)
# Coherently dedispersed at DM 413: denser trials around it.
dms_coh = dmtlib.generate_optimal_dm_grid(704.0, 4032.0, 6656, 64e-6, 2000.0,
                                          chan_dm=413.0)
# Integer delays for FDMT.
dts = dmtlib.generate_optimal_dt_grid(704.0, 4032.0, 6656, 64e-6, 16000)
plan = dmtlib.FDMTPlan(704.0, 4032.0, 6656, 65536, 64e-6, dt_grid=dts)
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
const auto dms  = dmt::plans::generate_dm_grid(cfg, 0.0, 2000.0);
const auto dts  = dmt::plans::generate_dt_grid(cfg, 0.0, 16000.0);
```
:::
::::
