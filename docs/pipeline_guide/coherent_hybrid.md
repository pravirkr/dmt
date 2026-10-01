# Coherent Hybrid Search (CohFDMT)

`CohFDMT` searches **baseband voltages** for short dispersed pulses with the
hybrid algorithm of Zackay & Ofek (2014):

1. Each subband block is Fourier transformed once.
2. For every *coarse* DM trial, each subband is split into `n_p` fine channels,
   each channel is coherently dedispersed to the trial DM **within the
   channel**, then inverse transformed and detected to Stokes I
   (`|X|² + |Y|²`) at `tsamp = n_p / bw_sub ≈ t_p`.
3. Channels are aligned by their inter-channel delay at the trial DM, and a
   fine FDMT searches the residual DM in `[-Δ, +Δ]` around it.

This gives nearly the sensitivity of full coherent dedispersion at every DM, at
FDMT cost. It is aimed at **offline processing of recorded baseband** (GUPPI
raw, LOFAR, PSRDADA files), read block by block.

## Configuration

```python
import dmtlib

cfg = dmtlib.CohFDMTConfig(
    f_center=1406.25,        # MHz, centre of the whole band
    bw_sub=1500 / 512,       # MHz per subband (GUPPI: 2.9296875)
    nsub=64,
    t_p=10e-6,               # target resolution, s
    dm_min=50.0, dm_max=60.0,
    dt_step=16,              # optional: coarser DM rows for wider pulses
    format=dmtlib.BasebandFormat("FTPRI", nbits=8),
)
search = dmtlib.CohFDMT(cfg, nthreads=8)          # or backend="cuda"
print(search.plan.summary())
```

The same in C++:

```cpp
dmt::CohFDMTConfig cfg{.f_center = 1406.25F, .bw_sub = 1500.0F / 512.0F,
                       .nsub = 64, .t_p = 10.0E-6F,
                       .dm_min = 50.0F, .dm_max = 60.0F, .dt_step = 16,
                       .format = {.order = "FTPRI", .nbits = 8}};
dmt::algorithms::CohFDMT search(cfg, dmt::Exec::cpu(8));
```

| Field | Meaning |
| :--- | :--- |
| `f_center`, `bw_sub`, `nsub` | Band layout. Each subband is complex baseband, **critically sampled** (interval `1 / bw_sub`), upper sideband, DC at its centre. |
| `t_p` | Target output resolution. `n_p = t_p · bw_sub` is rounded to the nearest 2,3,5,7-smooth integer (`plan.n_p`, `plan.tsamp`). A power of two is fastest. |
| `dm_min`, `dm_max` | DM range. |
| `block_nsamps` | Raw samples per subband you read per block (0 = automatic, see below). |
| `nbin` | Forward FFT length (0 = automatic). |
| `smear_tol` | Largest residual intra-channel smearing, in output samples (default 1). |
| `filter_leakage` | Share of the channel filter's impulse-response energy allowed past each FFT block's overlap (default `1e-4`). It sets the filter's part of `plan.noverlap`. |
| `dt_step` | Stride between fine delay trials (default 1). |
| `normalize` | Normalise every channel to zero mean and unit variance (default on). |
| `format` | Input layout and encoding (below). |
| `subband_groups` | Subbands per input array, e.g. `[64] * 8` for eight GUPPI node files. |

## Input formats

`BasebandFormat.order` names the four axes outermost first; any permutation of
`P` (polarisation), `RI` (real/imaginary), `T` (time) and `F` (subband) is
accepted.

| Data | `order` | `nbits` | Notes |
| :--- | :--- | :--- | :--- |
| GUPPI raw (GBT, Parkes, BL) | `FTPRI` | 8 or 4 | signed; 4-bit has the real part in the high nibble (`msb_first=True`) |
| LOFAR (separate Xr, Xi, Yr, Yi) | `PRITF` | 8 | concatenate the four streams |
| PSRDADA time-major | `TFPRI` | 8 | |

Interleaved T-inner layouts (such as GUPPI `FTPRI`) take a vectorised fast
path. Other orders are decoded with strided loads.

**Multiple files per block.** A GUPPI band spread over several nodes (for
example blc00–blc07) can be passed as one array per node, in ascending
frequency. Set `subband_groups=[64] * 8` and call `search.execute([node0,
node1, ...])`. All subbands then go through **one** full-band FDMT; splitting
the FDMT per node would lose a factor `√8` in S/N.

## Processing a file: stateless blocks

The engine keeps **no state between calls**. Each `execute()` searches one
self-contained block of `plan.block_nsamps` raw samples per subband. Advance
the read position by `plan.stride_nsamps` between blocks. Consecutive blocks
then overlap by `plan.overlap_nsamps` (the full dispersion sweep at `dm_max`
plus the coherent filter margins), and the valid outputs of consecutive blocks
tile the time axis exactly.

```python
plan = search.plan
out = None
for s0 in range(0, nsamps_file - plan.block_nsamps + 1, plan.stride_nsamps):
    block = read_block(s0, plan.block_nsamps)       # your reader, uint8/int8
    out = search.execute(block, out=out)            # (plan.ndm, plan.output_nsamps)
    t0 = s0 * plan.tbin + plan.output_time_offset   # time of out[:, 0]
    find_candidates(out, t0, plan.tsamp, plan.dm_grid_final)
```

- `out[i, j]` is DM `plan.dm_grid_final[i]` at arrival time
  `t0 + j · plan.tsamp` **at frequency `plan.f_ref`** (the centre of the lowest
  channel), for every row.
- The byte size of a block is `search.input_size()` (per group:
  `input_size(g)`). The array may have any shape, as long as it is
  C-contiguous and its bytes are in the configured `order`.
- `execute(out=...)` reuses a result buffer.
- One instance owns one set of working buffers. Calls from several threads
  take turns. On the GPU, device-memory calls on different streams are
  ordered, each after the previous call's device work. To process blocks
  concurrently, use one instance per thread or stream.

**Choosing the block length.** Each block pays once for the forward transform
of the whole block, sweep included. After that, every coarse trial only
transforms the part of each channel that reaches the valid output. The useful
fraction is `stride / block` (`plan.summary()` prints it). Longer blocks help
most when there are few coarse trials, and memory grows with the block:

| Buffer | Size |
| :--- | :--- |
| spectrum | 16 bytes × `nsub` × `nfft` × `nbin` (≈ 16 bytes per dual-pol raw sample) |
| waterfall | `nchans` × `plan.fdmt_nsamps` floats (one coarse trial) |
| result | `plan.ndm` × `plan.output_nsamps` floats (your buffer) |

`plan.memory_estimate()` and `search.memory_usage()` give the actual numbers.
The automatic choice aims for four times as many valid samples as the sweep costs,
within 1 GiB of spectrum.

**Overlap.** Each FFT block keeps `plan.noverlap` raw samples per side for the
coherent filter. That margin is the channel's chirp response at `dm_max` plus
the ringing of the channel filter. The ringing part is sized so that at most
`filter_leakage` of the filter's impulse-response energy falls outside it.
The default of `1e-4` leaves an amplitude tail of about 1%, below 8-bit
quantisation noise. Against `1e-6` it changes the output by about `2e-4` of the
peak. In the GUPPI reference search it cuts the margin from 1470 to 540 samples
per side, and the automatic `nbin` then picks 256-bin channel transforms
instead of 1024-bin ones.

## The DM grid

**Coarse trials.** The coarse step is the largest for which the residual
dispersion smearing inside the **bottom channel**, at the edge of a coarse
window, stays within `smear_tol · tsamp`. Two consequences:

- Channel width is `bw_sub / n_p`, so there are `nsub · n_p` channels across
  the band. For subband-channelised data this makes the grid about `nsub / 2`
  times coarser than applying the single-band `N_p²` rule to the subband
  sampling time. That is 29× to 380× fewer coherent trials for GUPPI- or
  MeerKAT-like data.
- For wideband single-subband data it is correctly *denser* than that rule,
  which leaves up to about 2 samples of smearing at the bottom of the band.

`plan.intra_channel_smear` reports the achieved worst case.

**Fine rows.** Within a coarse window, rows are one output sample of full-band
delay apart (`dt_step` samples with `dt_step > 1`). The total number of rows is
therefore about `sweep / (tsamp · dt_step)`. That number is set by the time
resolution, not by `dm_max`: coherent dedispersion keeps a pulse `t_p` wide at
every DM, so the DM step has to stay at one sample of band delay. Use
`dt_step` (or a narrow DM range) when you only need to detect wider pulses.

## Noise statistics and S/N

With `normalize=True`, every channel is scaled by its own noise statistics,
measured once per block from the forward spectrum (Parseval). The chirp only
changes phase, so these statistics are the same for all coarse trials.

For Gaussian noise the output then has zero mean, and row `i` has variance
`plan.get_effective_variance_grid(w)[i]` after a boxcar of `w` samples. That
value is exact, not an estimate:

- it includes the FDMT's box smearing;
- it includes the small correlation between neighbouring samples of a channel
  that the channel filter introduces (`plan.get_lag_correlation`: about 0.4%
  per lag, which adds a few percent for long boxcars).

So `S/N = boxcar_sum / sqrt(variance_grid[i])`. There is no need for a
calibration run on noise or on an array of ones. The statistics are
per-block, so a very bright pulse slightly raises its own block's mean power;
use blocks much longer than the pulses you search for.

With `normalize=False` the output is raw detected power. Row `i` then sums
`plan.get_cumulative_count_grid()[i]` channel samples.

## Performance notes

- FFTW planning matters for the non-power-of-two forward transforms. Use
  `dmtlib.set_fft_planner("measure")` together with a wisdom file (see
  [Fourier-Domain Variants](fourier_variants.md)); it gave about 30% on an M1.
- On the CPU, a single-coarse-trial search is bound by the forward FFT. With
  several coarse trials, the per-trial inverse FFTs and the fine FDMT dominate.
  Both run at the speed of FFTW and the FDMT; the chirp, detection and
  alignment are fused into the per-channel inverse-FFT pass.
- On the GPU, the block's front end (unpack, forward cuFFT, channel powers)
  runs in chunks of subbands sized to a quarter of the L2 cache. Each chunk's
  passes then stay on chip, which made long multi-pass transforms (30720
  points) 1.7× faster on an L40S. The chunks also pace the host-array upload.
- The GPU coherent stage is one fused kernel per coarse trial: gather times
  chirp, an in-shared-memory inverse FFT, detection, normalisation and the
  delay-aligned write. It reads the spectrum once and writes the waterfall once.
  This covers power-of-two channel transforms of 16–4096 bins (`plan.mbin`),
  which is what the automatic `nbin` gives. It runs at about the speed of a
  device-to-device copy of the same bytes (0.86 ms per trial in the GUPPI
  reference search on an L40S). Other lengths fall back to batched cuFFT
  through a work buffer, about 4.7× slower for this stage. The output matches
  the CPU engine to 1e-4 relative.
- With many coarse trials, the fine FDMT becomes the largest GPU cost per
  trial: 4.7 ms per trial for DM 50–250, against 1.2 ms for the coherent
  stage.
- The host-array `execute()` on the GPU uploads the block chunk by chunk on a
  copy stream. The front end of each chunk starts as soon as its bytes have
  landed, and the result comes back through pinned staging. It is bound by the
  host-to-device copy: about 130 MB per block in the GUPPI reference search.
  Pass device arrays (`DeviceSpan`, C++) to keep the input on the GPU.

See [Benchmarks](../benchmarks.md#coherent-hybrid-search-cohfdmt) for
measured throughput.
