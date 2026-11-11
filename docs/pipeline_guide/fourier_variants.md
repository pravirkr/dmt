# Fourier-domain variants

Two engines apply delays as phase ramps,
$x(t-\tau) \leftrightarrow \tilde x[k]\,e^{-2\pi i k\tau/N}$, so a shift can
be any real number. Time-domain engines round delays to whole samples, which
smears narrow pulses by up to half a sample per channel.

| | `FDMTFFT` | `DDMTFFT` |
| :--- | :--- | :--- |
| Algorithm | FDMT tree, merges as phase ramps | direct channel sum per frequency bin |
| Delays | fractional per merge (least-squares fit of the head and tail subbands' mean timing); `fractional_delays=False` rounds like FDMT, for equivalence tests only | exact per channel, always |
| Grid | FDMT delay grids | any DDMT grid: linear, explicit, Levin, piecewise-uniform Levin |
| Streaming | `"valid"` (overlap-save), `"full"`, `"roll"` | DDMT's history model |
| Stepper / level views | yes | no |

Both take float32 and packed 1–16-bit input (channel- or time-major), a kill
mask, and `save_history` / `load_history`, on the CPU and CUDA/HIP (same
results to float rounding). Each follows its time-domain counterpart's
conventions: DM constant, native time resolution, `plan.f_ref` (`f_min` by
default; `DDMTFFT` also takes `f_max`), and for `DDMTFFT` channel-centre
delays and `plan.time_offset` (see [periodicity](periodicity.md)).

```python
fdmt = dmtlib.FDMTFFT(1200.0, 1600.0, 1024, 16384, 6.4e-5, 1024,
                      kill_mask=mask, backend="cuda")
dmt = fdmt.execute(packed_rows, 2)              # (nchans, row_bytes) uint8
dmt = fdmt.execute_time_major(filterbank, 2)    # (nsamps, sample_bytes) uint8
state = fdmt.save_history()                     # same layout on every backend
```

**When to use them:** float waterfalls, and searches where fractional delays
matter (narrow pulses, high Fourier frequencies). Packed input only saves
reading bytes; the transforms run on complex floats either way.

## Look-ahead, latency and startup

A fractional shift is band-limited interpolation, so each output needs
`guard` = 64 samples of context on both sides. Streams then match one long
call to the sinc truncation, about $1/(\pi\sqrt{\text{guard}})$ of the
per-sample noise.

- **`DDMTFFT`.** `get_max_delay()` is `ceil(max tau) + guard`; the look-ahead
  is part of its streaming model (`get_output_nsamps()`).
- **`FDMTFFT`, valid mode.** The output lags the input by
  `get_output_latency()` (Python `output_latency`) = 64: output `o` is stream
  time `block_start - 64 + o`. Full, roll and integer mode have no latency.
- **Box smearing.** `FDMTFFT` uses the FDMT's dyadic box-mean weights
  ({ref}`box means <box-mean>`) but never rounds the means, for packed input
  too.

```{warning}
After a cold start, drop or mask the first `startup_samples` outputs of
`FDMTFFT` in valid mode (support, guard and latency together).
```

## Block length

Every call transforms the block plus its context (the overlap history for
`FDMTFFT`, `get_max_delay() + guard` for `DDMTFFT`). Short blocks waste most
of each FFT on context:

| block / context | 0.5 | 1 | 4 | 8 |
| :--- | ---: | ---: | ---: | ---: |
| useful output | 33% | 50% | 80% | 89% |

`get_suggested_nsamps()` (Python `suggested_nsamps`) gives an FFT-friendly
block length that keeps at least 80%. Debug logging warns below 60%.

## `DDMTFFT` methods

Per frequency bin, a uniform run of DM trials is a type-1 non-uniform FFT
over the DM axis.

- **`method="auto"`** (default) splits the grid into maximal uniform runs.
  Runs of ≥ 32 trials use the NUFFT (accurate to `tolerance`, default
  $10^{-6}$), the rest brute force. `method_used` reports `"nufft"`,
  `"piecewise_nufft"` or `"brute"`.
- **`method="brute"`** rotates every channel for every trial and bin. Exact
  for any grid; the reference the NUFFT is tested against. One NUFFT run costs
  about 20–100 brute-force trials, hence the 32-trial threshold.

### Piecewise-uniform Levin grid

A plain Levin grid has no uniform runs, so `DDMTFFT` would fall back to brute
force. The DDplan-style **piecewise-uniform Levin grid** fixes that:

- **Segments** break where the Levin step doubles.
- **Spacing** in each segment is the smallest Levin step inside it, so it is
  never coarser than the Levin grid.
- **Length** is at least 32 trials per segment.

It has 1.2–1.4× the Levin trials but runs one NUFFT per segment, far cheaper
than brute force.

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
cfg = dmtlib.LevinConfig(0.0, 2000.0, pulse_width=1e-4, tol=1.2,
                         piecewise_uniform=True)
eng = dmtlib.DDMTFFT(1200.0, 1600.0, 1024, 6.4e-5, cfg)
assert eng.method_used == "piecewise_nufft"

# Or build the same grid explicitly:
dms = dmtlib.DDMTPlan.generate_levin_dm_grid_piecewise(
    0.0, 2000.0, 6.4e-5, 1e-4, 1200.0, 1600.0, 1024, tol=1.2)
```
:::
:::{tab-item} C++
:sync: cpp

`plans::LevinConfig{..., .piecewise_uniform = true}` or
`plans::DDMTPlan::generate_levin_dm_grid_piecewise(...)`.
:::
::::

Any grid of uniform segments of ≥ 32 trials is detected the same way.

## FFT planning

CPU transforms use FFTW, and the engines are FFT-bound. `FFTW_ESTIMATE` (the
default) plans instantly; `FFTW_MEASURE` plans are 1.3–1.5× faster for the
non-power-of-two lengths used, but take about 2 s per length. For long-running
pipelines, measure once per machine and keep the wisdom.

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
dmtlib.set_fft_planner(dmtlib.FFTPlanner.MEASURE)   # plans created from now on
eng = dmtlib.DDMTFFT(...)
dmtlib.export_fft_wisdom("dmt.wisdom")              # later: import_fft_wisdom
```
:::
:::{tab-item} C++
:sync: cpp

`dmt::fft::set_planner`, `import_wisdom` and `export_wisdom`
(`dmt/common/fft_config.hpp`).
:::
:::{tab-item} Environment

```bash
export DMT_FFTW_PLANNER=measure          # estimate | measure | patient | exhaustive
export DMT_FFTW_WISDOM=$HOME/.dmt.wisdom # read before the first plan, updated after
```
:::
::::

```{note}
If NumPy is linked to MKL, importing it before `dmtlib` can make MKL's FFTW
wrappers stand in for FFTW: transforms work, but the wisdom functions return
`False`. Import `dmtlib` first.
```

:::{dropdown} Transform lengths
:icon: info

Lengths are even with prime factors 2, 3, 5 and 7 only. `FDMTFFT` in valid and
full mode transforms long blocks in overlap-save segments sized by FFTW's own
cost estimate, so memory stays bounded and the result is the same linear
convolution.
:::

## Performance

On the CPU, `DDMTFFT` with the NUFFT is the fastest engine at high DM, and
`FDMTFFT` costs up to about twice FDMT. On the GPU both stay within a small
factor of FDMT. See the [benchmarks](../benchmarks.md).
