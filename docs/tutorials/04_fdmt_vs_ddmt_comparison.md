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

# Tutorial 4: FDMT and direct dedispersion

`dmt` has two families of incoherent dedispersion:

- **FDMT** builds the DM–time plane with a tree. It merges neighbouring
  sub-bands level by level and reuses partial sums across DMs, so its cost
  grows as $N_t N_f \log_2 N_f$, almost independent of the number of DMs.
- **DDMT** (direct dedispersion) shifts every channel by its own delay for
  every DM and adds them up. Its cost grows as $N_t N_f N_\text{DM}$. `SDMT`
  computes exactly the same sums faster by sharing sub-band partial sums
  between DMs.

This notebook runs both on the same data and shows where their outputs agree
and where they differ: the time reference, the start of the stream, and how a
very short pulse lands in the output.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from dmtlib import FDMT, DDMT
from dmtlib.simulate import DM_CONSTANT, generate_frb

f_min, f_max = 1200.0, 1600.0      # band edges, MHz
nchans, nsamps, tsamp = 128, 512, 1e-3
target_delay = 24                  # samples across the band

dm = target_delay * tsamp / (DM_CONSTANT * (f_min**-2 - f_max**-2))
waterfall = generate_frb(f_min=f_min, f_max=f_max, nchans=nchans, nsamps=nsamps,
                         tsamp=tsamp, dm=dm, amp=2.5, offset=100, noise_rms=1.0)
print(f"DM {dm:.2f} pc cm^-3, waterfall {waterfall.shape}")
```

## Running both

DDMT runs at FDMT's effective DMs, so trial `k` means the same DM in both.

```{code-cell} ipython3
fdmt = FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max=48)
ddmt = DDMT(f_min, f_max, nchans, tsamp, np.asarray(fdmt.dm_grid_final, dtype=np.float32))

out_fdmt = fdmt.execute(waterfall)
out_ddmt = ddmt.execute(waterfall)
print("FDMT", out_fdmt.shape, " DDMT", out_ddmt.shape)
print("FDMT startup_samples:", fdmt.startup_samples)
print(f"time reference: FDMT {fdmt.plan.f_ref:.3f} MHz, DDMT {ddmt.plan.f_ref:.3f} MHz")
print("DDMT time_offset:", ddmt.plan.time_offset)
```

The outputs differ in two ways before any arithmetic happens.

- **Length.** On its first call FDMT returns all 512 samples, but the first
  `startup_samples` (48) are **partial sums over a zero-filled past**. DDMT
  withholds what it cannot compute yet and returns 512 − 48 = 464. Both are
  complete from the next block on.
- **Time zero.** Both refer times to `f_min`, the bottom of the band
  (`plan.f_ref`). FDMT's output sample `t` is the arrival at input sample `t`;
  DDMT's output sample `t` is the arrival at input sample
  `t + plan.time_offset`, the largest lag of any channel centre behind `f_min`
  over its DM trials (48 here).

Offset FDMT by `time_offset` and the two line up:

```{code-cell} ipython3
k = target_delay
off = ddmt.plan.time_offset
d_row = out_ddmt[k]
f_row = out_fdmt[k, off:off + d_row.size]
start = max(0, fdmt.startup_samples - off)

t = np.arange(f_row.size) + off   # input sample of the f_min arrival
plt.figure(figsize=(10, 4))
plt.plot(t[start:], f_row[start:], label="FDMT")
plt.plot(t[start:], d_row[start:], "--", label="DDMT")
plt.xlim(60, 160)
plt.xlabel("input sample of the arrival at f_min")
plt.ylabel("sum over channels")
plt.legend()
plt.tight_layout()
plt.show()
print("FDMT peak at", t[start + f_row[start:].argmax()], " DDMT peak at", t[start + d_row[start:].argmax()])
```

The pulse lands at the same time, to within a sample. The noise around it is only partly
correlated between the two. With the default box smearing, FDMT averages each
channel over its smearing window; DDMT takes one sample per channel; and the
two round delays differently. So don't expect identical time series, only
**the same pulse at the same time and DM**.

## How a one-sample pulse lands

`generate_frb` puts a pulse exactly one sample wide in each channel. Without
noise, it is easy to see where its flux ends up:

```{code-cell} ipython3
clean = generate_frb(f_min=f_min, f_max=f_max, nchans=nchans, nsamps=nsamps,
                     tsamp=tsamp, dm=dm, amp=2.5, offset=100, noise_rms=0.0)
raw = FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max=48, use_box_smearing=False)
for name, engine in (("FDMT", raw), ("DDMT", ddmt)):
    engine.reset_history()
    row = np.asarray(engine.execute(clean))[k]
    j = row.argmax()
    print(f"{name}: samples around the peak {np.round(row[j - 2:j + 3], 1)}, "
          f"total {row.sum():.1f}")
print("flux injected:", 2.5 * nchans)
```

Both keep all of the flux. DDMT rounds each channel's delay once, so most
channels land in the same output sample. FDMT rounds at every level of the
tree, so the pulse spreads over two or three samples. For a pulse this short
that lowers FDMT's single-sample peak. A boxcar search over 2–3 samples gets
most of it back. For real pulses, which are wider and arrive at arbitrary
sub-sample times, the difference is smaller.

```{note}
The same rounding is why integer FDMT keeps less signal near Nyquist in a
periodicity search; FDMT-FFT avoids it with fractional delays (see the
{doc}`periodicity guide <../pipeline_guide/periodicity>`).
```

The noise, on the other hand, behaves the same: with
`use_box_smearing=False`, FDMT output is white with variance `nchans`, like
DDMT's ({doc}`tutorial 7 <07_fdmt_noise_statistics>`).

## Packed low-bit input

Both engines read packed unsigned integers directly (`nbits` = 1, 2, 4, 8 or
16), so a 1-bit stream needs no unpacking to float first. DDMT accumulates
packed input in 32-bit integers:

```{code-cell} ipython3
waterfall_u8 = np.clip((waterfall + 4.0) * 32.0, 0, 255).astype(np.uint8)
ddmt_u8 = DDMT(f_min, f_max, nchans, tsamp,
               np.asarray(fdmt.dm_grid_final, dtype=np.float32), nbits=8)
out_u8 = ddmt_u8.execute(waterfall_u8, nsamps=nsamps)
print(f"input {waterfall_u8.dtype} {waterfall_u8.shape} -> output {out_u8.dtype} {out_u8.shape}")
```

FDMT takes the same layout through `execute(packed, nbits)`. Its tree stays in
narrow integers where the value range allows, which makes 1-bit FDMT about
2–3× faster than float input.

## Which one to use

:::{admonition} Takeaways
:class: dmt-summary

- **Many DMs** (hundreds or more): FDMT. It is much faster, and the gap grows
  with the number of DMs.
- **A handful of DMs**, for example re-checking candidates: DDMT or SDMT.
- **Signal near Nyquist** in a periodicity search: FDMT-FFT or DDMT-FFT.
:::
