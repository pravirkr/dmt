# Fourier-Domain Variants

`dmt` has two dedispersion engines that work in the Fourier domain, where a
time shift is a phase ramp, $x(t-\tau) \leftrightarrow
\tilde x[k]\,e^{-2\pi i k\tau/N}$:

| | `FDMTFFT` | `DDMTFFT` |
| :--- | :--- | :--- |
| Algorithm | the FDMT tree, merges as phase ramps | direct sum over channels, per frequency bin (FDD) |
| Delays | fractional merge delays (default); `fractional_delays=False` rounds like FDMT, for equivalence tests only | exact fractional delays, always |
| Grid | FDMT delay grids (`dt_max`, custom `dt`/DM grids) | any DDMT grid: linear, explicit, Levin, piecewise-uniform Levin |
| Streaming | `mode="valid"` overlap-save, `"full"`, `"roll"` | DDMT's model: history across calls |
| Input | float32, packed 1/2/4/8/16-bit (channel- or time-major) | float32, packed 1/2/4/8/16-bit (channel- or time-major) |
| Kill mask, history save/load | yes | yes |
| Stepper / level views | yes | no |
| Backends | CPU, CUDA/HIP (same results to float rounding) | CPU, CUDA/HIP (same results to float rounding) |

Both use the same conventions as the time-domain engines: channel `c` at
`f_min + c * df`, delays referenced to the highest channel, the same DM
constant, and time at native resolution.

## When to use them

| Use | Engine |
| :--- | :--- |
| Fastest blind search, any bit width | `FDMT` (time domain) |
| Low-bit (1–8 bit) raw data, throughput first | `FDMT`, `SDMT` (integer adds on packed data) |
| Exact per-channel fractional delays, uniform or piecewise-uniform DM grid | `DDMTFFT` (NUFFT) |
| Exact per-channel delays on an arbitrary DM list | `DDMTFFT` (`method="brute"`), or `DDMT` if rounding is acceptable |
| FDMT tree with sub-sample merges (less rounding smear) | `FDMTFFT` (fractional delays by default) |
| FDMT-identical output from the Fourier engine (equivalence tests only) | `FDMTFFT(fractional_delays=False)` |

The Fourier engines transform to complex float spectra. Low-bit input does
not make them any faster algorithmically; it only reduces the input bytes read
(both engines unpack packed rows straight into the transform rows). They pay
off for float waterfalls, and when fractional delays matter.

Packed input, the kill mask and the history work as for the time-domain
engines:

```python
fdmt = dmtlib.FDMTFFT(1200.0, 1600.0, 1024, 16384, 6.4e-5, 1024,
                      kill_mask=mask, backend="cuda")
dmt = fdmt.execute(packed_rows, 2)              # (nchans, row_bytes) uint8
dmt = fdmt.execute_time_major(filterbank, 2)    # (nsamps, sample_bytes) uint8
state = fdmt.save_history()                     # same layout on every backend
```

## Why Fourier-domain dedispersion

Time-domain DDMT rounds every channel delay to a whole sample, and FDMT rounds
the delay of every tree merge. The rounding smears a pulse by up to half a
sample per channel, so narrow pulses lose S/N. A phase ramp shifts by any
real amount, at no extra cost:

- `DDMTFFT` delays every channel by its exact $\tau_{d,c} = \mathrm{DM}_d\,
  r_c$. On a band-limited pulse it recovers the ideal peak to 0.1%, where
  DDMT loses about 4% (test `DDMTFFT recovers fractional-delay pulses better
  than DDMT`).
- `FDMTFFT` (fractional delays are its default and its nature) replaces each
  merge's rounded shift with
  the least-squares shift that aligns the mean timing of the head sub-band's
  channels with the tail's at the node's DM. The children stay on integer
  delay grids, so this is not simply the unrounded `dt * phi`: in the integer
  tree a child's trial and the merge shift use the same rounded value, and
  their errors partly cancel. Rounding away only the merge shift breaks that
  cancellation and *loses* S/N. The least-squares shift roughly halves the
  rms per-channel misalignment of the integer tree (0.53 to 0.31 samples at
  4096 channels and 2049 trials), for about 1.5x the integer mode's time.
- `FDMTFFT(fractional_delays=False)` rounds every merge shift exactly as the
  time-domain FDMT does and reproduces its output through the Fourier domain.
  It exists only for equivalence tests: rounding the shifts gives up what the
  Fourier domain is for.

A fractional shift is band-limited (periodic sinc) interpolation, so every
output needs context on both sides. Both engines keep `guard` samples (64)
of look-behind *and* look-ahead, so a stream of blocks gives the same result
as one long call (to the sinc truncation, roughly $1/(\pi\sqrt{\text{guard}})$
of the per-sample noise):

- `DDMTFFT`: `get_max_delay()` is `ceil(max tau) + guard`; the look-ahead is
  part of its streaming model (`get_output_nsamps()`).
- `FDMTFFT` in valid mode (fractional delays, the default): the output lags the input
  by the guard, `get_output_latency()` (Python `output_latency`) = 64. Output
  sample `o` of a block is stream time `block_start - 64 + o`. Full and roll
  mode, and the integer equivalence mode, have no latency.

## DDMTFFT: NUFFT, piecewise NUFFT or brute force

Per frequency bin $k$ the channel sum over a uniform run of trials
$\mathrm{DM}_d = \mathrm{DM}_0 + d\,\Delta\mathrm{DM}$ is
$Y_k(d) = \sum_c a_c\,e^{2\pi i\,d\,x_c}$ with $x_c = k\,\Delta\mathrm{DM}\,r_c/N$:
a type-1 non-uniform FFT over the DM axis.

- The default `method="auto"` splits the DM grid into maximal uniformly
  spaced runs. Every run of at least 32 trials is summed by the NUFFT (one
  per run, all sharing the channel spectra), the other trials by brute force.
  `method_used` reports `"nufft"` (a linear grid, one run),
  `"piecewise_nufft"` (several runs, or runs plus brute-force trials) or
  `"brute"`.
- The NUFFT spreads the channels onto an oversampled grid with the
  exponential-of-semicircle kernel and takes one small FFT per bin: $O(n_\text{chans}
  w + n_\text{DM}\log n_\text{DM})$ per bin and run instead of $O(n_\text{chans}
  n_\text{DM})$, accurate to `tolerance` (default $10^{-6}$ relative, at
  float precision).
- `method="brute"` rotates every channel by its exact phase for every trial
  and bin. It is exact for any grid and is the reference the NUFFT is tested
  against. On the CPU it is FMA-bound (explicit AVX-512 kernel on x86, `omp
  simd` elsewhere). One NUFFT run costs about as much as 20–100 brute-force
  trials (NEON vs AVX-512), which is why runs shorter than 32 trials are not
  worth a NUFFT.

### Non-linear DM grids: the piecewise-uniform Levin grid

A Levin grid (`LevinConfig`) has a continuously growing step, so it has no
uniform runs and `DDMTFFT` would fall back to brute force. For fast
Fourier-domain dedispersion over a wide DM range, use the DDplan-style
**piecewise-uniform Levin grid**:

- the DM range is split where the Levin step doubles;
- each segment is uniformly spaced at the smallest Levin step inside it, so it
  is never coarser than the Levin grid (at least its sensitivity);
- each segment holds at least 32 trials (short segments are refined).

It has typically 1.2–1.4x the Levin trials, and `DDMTFFT` runs one NUFFT per
segment. With 4–6 segments the cost is still far below brute force, because the
NUFFT cost per run hardly depends on its trial count.

```python
import dmtlib

# Either ask the plan for it ...
cfg = dmtlib.LevinConfig(0.0, 2000.0, pulse_width=1e-4, tol=1.2,
                         piecewise_uniform=True)
eng = dmtlib.DDMTFFT(1200.0, 1600.0, 1024, 6.4e-5, cfg)
assert eng.method_used == "piecewise_nufft"

# ... or build the grid and pass it explicitly (same trials).
dms = dmtlib.DDMTPlan.generate_levin_dm_grid_piecewise(
    0.0, 2000.0, 6.4e-5, 1e-4, 1200.0, 1600.0, 1024, tol=1.2)
eng = dmtlib.DDMTFFT(1200.0, 1600.0, 1024, 6.4e-5, dms)
# grid.generate_optimal_dm_grid(..., method="levin_piecewise") is the same.
```

In C++: `plans::LevinConfig{..., .piecewise_uniform = true}` or
`plans::DDMTPlan::generate_levin_dm_grid_piecewise(...)`. Any DDplan-style
grid you build yourself (uniform segments of at least 32 trials) is detected
the same way.

## Block length

Every Fourier-domain call transforms the new block **plus** its context: the
overlap history of `FDMTFFT` in valid mode, `get_max_delay() + guard` for
`DDMTFFT`. A block much shorter than the context spends most of every FFT on
it, which the time-domain engines never pay:

| block / context | output fraction of each transform |
| ---: | ---: |
| 0.5 | 33% |
| 1 | 50% |
| 4 | 80% |
| 8 | 89% |

`get_suggested_nsamps()` (Python `suggested_nsamps`) returns a block length
that keeps at least 80%, rounded so that the transform length stays
FFT-friendly. With debug logging on, the engines log when a block keeps less
than 60%.

## FFT planning

CPU transforms use FFTW. The engines are FFT-bound (on the M1 about half of
`FDMTFFT`'s time is FFTW), so the plan quality matters:

- The default `FFTW_ESTIMATE` plans are instant.
- `FFTW_MEASURE` plans are 1.3–1.5x faster on the M1 for the non-power-of-two
  lengths the engines use (for example 18432 = 2^11 · 9), but take about 2 s
  per transform length to make.

For long-running pipelines, measure once per machine and keep the wisdom:

```python
import dmtlib
dmtlib.set_fft_planner(dmtlib.FFTPlanner.MEASURE)   # plans created from now on
eng = dmtlib.DDMTFFT(...)                           # slower to construct
dmtlib.export_fft_wisdom("dmt.wisdom")              # later runs: import_fft_wisdom
```

Or, without code changes, from the environment:

```bash
export DMT_FFTW_PLANNER=measure          # estimate | measure | patient | exhaustive
export DMT_FFTW_WISDOM=$HOME/.dmt.wisdom # imported before the first plan,
                                         # rewritten after every measured plan
```

In C++ the controls are `dmt::fft::set_planner`, `import_wisdom` and
`export_wisdom` (`dmt/common/fft_config.hpp`). The benchmark suite runs with
`DMT_FFTW_PLANNER=measure` and a per-machine wisdom file
(`run_suite.py --fftw-planner`).

```{note}
In a Python environment where NumPy is linked to MKL, importing NumPy before
`dmtlib` can let MKL's FFTW-compatible wrappers stand in for FFTW. The
transforms still work, but the wisdom functions are no-ops (they return
`False`). Import `dmtlib` first to bind to FFTW.
```

Transform lengths are always FFTW-friendly (even, with only the prime factors
2, 3, 5 and 7). `FDMTFFT` in valid and full mode transforms long blocks in
overlap-save segments whose length is chosen from FFTW's own cost estimate.
The result is the same linear convolution as one long transform. Memory stays
bounded, and the transform length no longer grows with the block.

## GPU engines

Both engines run on CUDA and HIP with the CPU engines' geometry: the same
transform lengths, overlap-save segments (one shared rule; a GPU shortens
them only when device memory is short), delays, NUFFT parameters and
conventions. Their results agree with the CPU to float rounding (the tests
use 2e-5 of the output peak).

- **`FDMTFFT`** runs the tree as *programs* over tiles of 32 frequency bins,
  one bin per lane. A warp applies one merge to 32 bins at a time, so the
  merge list is read once per 32 bins. The tree is cut into stages. Each
  program keeps its intermediate levels in shared memory, so only the stage
  boundaries go through device memory (two stages at the reference point:
  ~90 KB of traffic per bin, against ~770 KB for a level-by-level tree).
- **Phases** in both engines come from an exact 0.64 fixed-point turn count
  per shift, multiplied by the bin index modulo 2^64: no phasor tables and
  no FP64, which the L40S runs at 1/64 rate.
- **`DDMTFFT` brute force** holds 8 bins × 4 trials per thread. Each
  (trial, channel) phasor is set up once and rotated along the bins, and it
  runs near the FP32 issue limit.
- **`DDMTFFT` NUFFT** spreads with integer shared-memory atomics. Float
  shared-memory atomics are compare-and-swap loops on GPUs before sm_90; the
  integers are scaled per bin, so the rounding stays far below the
  tolerance. A uniform run whose fine grid would not fit shared memory
  (above ~4K trials) is transformed as a few equal sub-runs, still by NUFFT.
- Device-memory calls run asynchronously on the given stream, ordered after
  the engine's previous work. Host-array calls stage through the device and
  block. `set_gulp_size()` is stored but not used, as on the CPU, because
  chunking a call would move the transform boundaries.

## Performance

At the benchmark reference point (4096 channels, 16K samples, 2049 trials;
CPUs with 8 threads and FFTW MEASURE plans, the GPU with device-resident data;
see the [benchmarks](../benchmarks.md) for the sweeps):

| Engine | Apple M1 Pro | Xeon Gold 6348H | NVIDIA L40S | Delays |
| :--- | ---: | ---: | ---: | :--- |
| FDMT | 25.5 ms | 53 ms | 4.8 ms | rounded tree |
| FDMT-FFT (fractional delays, the default) | 83 ms | 77 ms | 5.5 ms | least-squares tree |
| FDMT-FFT, `fractional_delays=False` (equivalence tests) | 57 ms | 58 ms | 5.0 ms | rounded tree (FDMT-identical) |
| SDMT | 206 ms | 170 ms | 8.5 ms | rounded per channel |
| DDMT | 851 ms | 573 ms | 21 ms | rounded per channel |
| DDMT-FFT, NUFFT | 111 ms | 127 ms | 10.7 ms | exact per channel |
| DDMT-FFT, brute force | 3.33 s | 1.35 s | 39 ms | exact per channel |

In 0.5.0 FDMT-FFT took 415 ms on the M1 Pro, 701 ms on the Xeon and 22 ms
on the L40S.

Where the time goes (reference point; M1 Pro unless noted):

- **`FDMTFFT` is FFT-bound.** 4096 forward and 2049 inverse real FFTs of
  18432 points take about half the time, and the scatter into bin tiles about
  a sixth. The whole tree runs out of per-thread cache one 16-bin tile at a
  time: about 30%, near the cache-bandwidth limit of its indexed operand
  reads. The FFT alone costs about as much as the whole time-domain FDMT, so
  on this machine `FDMTFFT` stays about 2x behind `FDMT`. On the Xeon, where
  FDMT is memory-bound, it is 1.05–1.45x FDMT from 2K trials or 8K samples
  on.
- **`DDMTFFT` (NUFFT)** spends roughly a third on the channel FFTs, a third on
  preparing and spreading the points (a vectorised, register-blocked Horner
  kernel, the ES kernel at width 7 for $10^{-6}$), and the rest on the fine-grid
  FFTs.
- **`DDMTFFT` brute force** is bound by its complex FMA rate: 8 vector
  operations per 16 complex terms, half of them for the exact phase rotation.
  That is inherent, since every (trial, channel, bin) needs its own phasor.
- **On the L40S** the Fourier engines are cuFFT-bound: the forward and
  inverse transforms take about half of `FDMTFFT` (whose tree stages run
  near the shared-memory limit) and of `DDMTFFT`'s NUFFT path (spreading is
  about a fifth). Brute force runs near the card's FP32 issue rate, ~10
  instructions per complex term.
