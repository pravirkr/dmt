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

# Tutorial 5: Hybrid coherent FDMT

This tutorial searches raw baseband voltages for a short dispersed pulse with
the hybrid coherent + incoherent algorithm of Zackay & Ofek, as implemented by
`dmtlib.CohFDMT`: coarse coherent dedispersion, channelisation and detection,
then a fine FDMT around each coarse DM.

+++

## Why coherent dedispersion

Incoherent dedispersion sums detected power across channels, so dispersion
*within* a channel of width $\Delta f$ smears a pulse by

$$\Delta t_{\text{smear}} \approx 2\, k_{\mathrm{DM}}\, \mathrm{DM}\, \nu^{-3} \Delta f .$$

**Coherent dedispersion** removes it by applying the inverse cold-plasma phase
to the voltage spectrum, but a blind search would need one full coherent pass
per DM trial. The **hybrid algorithm** dedisperses coherently only on a coarse
DM grid and leaves the small residual to an FDMT. The coarse step is the
largest for which the residual smearing inside the bottom channel stays below
one output sample. With `nsub` subbands of `n_p` channels each, that counts
all `nsub * n_p` channels.

```{code-cell} ipython3
import numpy as np
import dmtlib

K_DISP = 1.0 / 2.41e-4  # dmt's dispersion constant

# A GUPPI-like band: 16 coarse channels of 2.93 MHz at 1.4 GHz, int8 FTPRI.
cfg = dmtlib.CohFDMTConfig(
    f_center=1406.25, bw_sub=1500 / 512, nsub=16,
    t_p=10e-6, dm_min=50.0, dm_max=60.0,
    format=dmtlib.BasebandFormat("FTPRI", nbits=8),
)
plan = dmtlib.CohFDMTPlan(cfg)
print(plan.summary())
```

The plan rounds `t_p` to `n_p` subband samples (a 2,3,5,7-smooth count), picks
the FFT lengths and the block geometry, and reports the coarse grid. For
multi-subband data the exact criterion needs far fewer coherent trials than
the single-band $N_p^2$ rule applied with the subband sampling time:

```{code-cell} ipython3
wide = dmtlib.CohFDMTPlan(dmtlib.CohFDMTConfig(1406.25, 1500 / 512, 64, 10e-6, 0.0, 500.0))
delay = K_DISP * 500 * (wide.f_min**-2 - wide.f_max**-2)
old = int(np.ceil(delay * wide.tbin / (2 * 1e-5**2)))
print(f"GUPPI node, DM 0-500: {wide.ndm_coh} coarse trials (single-band rule: {old}), "
      f"residual intra-channel smear {wide.intra_channel_smear:.2f} samples")
```

## Simulate baseband and search it

`simulate_baseband` builds dual-polarisation voltages with the exact
dispersion phase at every RF frequency; `pack_baseband` quantises them into
the configured byte layout. The next cell injects one pulse at DM 55.3 in
Gaussian noise and searches a single block.

```{code-cell} ipython3
search = dmtlib.CohFDMT(cfg, nthreads=4)
dm_true, j_true = 55.3, 3000                   # output sample of the pulse
t_ref = plan.output_time_offset + j_true * plan.tsamp       # arrival at f_ref
pulse = (dm_true, t_ref - K_DISP * dm_true / plan.f_ref**2, 3.0e5, 0.0)
volts = dmtlib.simulate_baseband(cfg.f_center, cfg.bw_sub, cfg.nsub,
                                 plan.block_nsamps, [pulse], noise_sigma=16.0,
                                 seed=1, nthreads=4)
block = dmtlib.pack_baseband(volts, cfg.format)             # int8 FTPRI bytes
out = search.execute(block)                                 # (ndm, output_nsamps)

sigma = np.sqrt(plan.get_effective_variance_grid())         # exact noise per row
snr = out / sigma[:, None]
row, j = np.unravel_index(np.argmax(snr), snr.shape)
print(f"{plan.ndm} DM rows x {plan.output_nsamps} samples; peak S/N {snr[row, j]:.1f} "
      f"at DM {plan.dm_grid_final[row]:.3f} (true {dm_true}), sample {j} (true {j_true})")
quiet = snr[:, :2500]                                       # well before the pulse
print(f"noise away from the pulse: mean {quiet.mean():+.3f}, std {quiet.std():.3f} (theory 0, 1)")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
t_ms = (np.arange(j - 150, j + 150) - j) * plan.tsamp * 1e3
ax0.imshow(snr[:, j - 150:j + 150], aspect="auto", origin="lower", cmap="Blues",
           extent=[t_ms[0], t_ms[-1], plan.dm_grid_final[0], plan.dm_grid_final[-1]])
ax0.set(xlabel="time from peak (ms)", ylabel="DM (pc cm$^{-3}$)", title="S/N, DM-time plane")
ax1.plot(t_ms, snr[row, j - 150:j + 150], color="#2a78d6", lw=1.5)
ax1.set(xlabel="time from peak (ms)", ylabel="S/N", title=f"Best row, DM {plan.dm_grid_final[row]:.3f}")
plt.show()
```

With `normalize=True` (the default) each channel is scaled to zero mean and
unit variance from the block's own spectrum, so `get_effective_variance_grid(w)`
is the exact noise variance of every row after a boxcar of `w` samples. It
includes the FDMT's box smearing and the channel filter's small lag
correlation, so **S/N needs no calibration run**.

```{note}
The statistics are per block: a pulse as bright as this one slightly raises
its block's channel means, which can push the noise mean a little below zero.
```

## 3. Processing a file: stateless blocks

The engine keeps no state between calls. Read blocks of `plan.block_nsamps`
samples and advance by `plan.stride_nsamps`; the valid outputs of consecutive
blocks tile the time axis. Here a longer stream stands in for a file, with a
second pulse near a block boundary:

```{code-cell} ipython3
nblocks = 3
total = plan.block_nsamps + (nblocks - 1) * plan.stride_nsamps
t_b = plan.output_time_offset + (plan.output_nsamps + 20) * plan.tsamp   # in block 1
pulses = [(52.0, t_b - K_DISP * 52.0 / plan.f_ref**2, 3.0e5, 0.0)]
stream = dmtlib.simulate_baseband(cfg.f_center, cfg.bw_sub, cfg.nsub, total, pulses,
                                  noise_sigma=16.0, seed=2, nthreads=4)
buf = None
for b in range(nblocks):
    s0 = b * plan.stride_nsamps                          # file read position
    raw = dmtlib.pack_baseband(stream, cfg.format, t_begin=s0, t_count=plan.block_nsamps)
    buf = search.execute(raw, out=buf)
    snr = buf / sigma[:, None]
    r, jj = np.unravel_index(np.argmax(snr), snr.shape)
    t = s0 * plan.tbin + plan.output_time_offset + jj * plan.tsamp
    print(f"block {b}: best S/N {snr[r, jj]:5.1f} at DM {plan.dm_grid_final[r]:.2f}, "
          f"t = {t * 1e3:7.3f} ms (arrival at {plan.f_ref:.1f} MHz)")
print(f"injected: DM 52.0 at t = {t_b * 1e3:.3f} ms")
```

The pulse is found in block 1 at the right DM and time. Block 0 ends 20 samples
before its arrival and only sees its leading edge at a mismatched DM (the
DM-time "bowtie"); a candidate sifter keeps the brightest detection.

+++

## 4. Input formats

`BasebandFormat` describes the bytes: any order of the axes `P`, `RI`, `T`, `F`
(`"FTPRI"` GUPPI, `"PRITF"` LOFAR, `"TFPRI"` PSRDADA), 8/4/2-bit samples,
signed or offset binary. Several arrays (one per GUPPI node file) can form one
band with `subband_groups`. Every layout gives the same result:

```{code-cell} ipython3
ref = search.execute(block)
for fmt in (dmtlib.BasebandFormat("PRITF", is_signed=False),
            dmtlib.BasebandFormat("TFPRI")):
    alt = dmtlib.CohFDMTConfig(cfg.f_center, cfg.bw_sub, cfg.nsub, cfg.t_p,
                               cfg.dm_min, cfg.dm_max, format=fmt)
    same = np.array_equal(dmtlib.CohFDMT(alt, 4).execute(dmtlib.pack_baseband(volts, fmt)), ref)
    print(f"{fmt}: identical output = {same}")

groups = dmtlib.CohFDMTConfig(cfg.f_center, cfg.bw_sub, cfg.nsub, cfg.t_p,
                              cfg.dm_min, cfg.dm_max, subband_groups=[8, 8])
parts = [dmtlib.pack_baseband(volts, cfg.format, sub_begin=0, sub_count=8),
         dmtlib.pack_baseband(volts, cfg.format, sub_begin=8, sub_count=8)]
print("two node files:", np.array_equal(dmtlib.CohFDMT(groups, 4).execute(parts), ref))
```

## Summary

:::{admonition} Takeaways
:class: dmt-summary

- The coarse coherent grid bounds the residual smearing inside the bottom
  channel, counting all `nsub * n_p` channels: far fewer coherent trials for
  subband-channelised data such as GUPPI.
- `CohFDMT` is stateless: advance the read position by `plan.stride_nsamps`
  per block of `plan.block_nsamps` samples.
- Output sample `j` of a block starting at raw sample `s0` is the arrival time
  at `plan.f_ref` of `s0 * tbin + output_time_offset + j * tsamp`.
- With normalisation, `plan.get_effective_variance_grid()` gives the exact noise
  of every row, so S/N follows directly.
:::
