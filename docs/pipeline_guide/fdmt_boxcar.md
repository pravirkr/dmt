# Fused FDMT and boxcar search

`FDMTBoxcar` and `FDMTFFTBoxcar` search normalized waterfalls and return raw
detections, without materializing the DM–time output. Use them when that
output would only be consumed by a boxcar search: the dedispersed rows are
scored while they are still on chip or in cache, instead of being written to
memory and read back by a separate search pass. `FDMT`, `FDMTFFT` and the
other algorithms are unchanged.

| | `FDMTBoxcar` | `FDMTFFTBoxcar` |
| :--- | :--- | :--- |
| Transform | time-domain FDMT tree (integer delays) | Fourier-domain FDMT tree (fractional delays by default) |
| Fused into the search | the root merge (root rows are never stored) | each overlap-save segment's inverse transform |
| Input | float32, or packed 1/2/4/8/16-bit | float32 |
| Subband search | plan `subband_depth` (any depth below the root), with staggered subbands | full band only |
| Latency | none | `output_latency` samples (64 with fractional delays) |

## Input and statistical contract

Input is finite, contiguous float32 in `(nbeams, nchans, nsamps)` order, with
channels ordered from low to high frequency. Python also accepts
`(nchans, nsamps)` for one beam. Supply non-overlapping consecutive blocks.
Only valid-mode plans are supported. Delay grids may be regular, signed, or
custom. A physical DM grid is snapped to integer band delays; the DM values
the plan reports are reconstructed from those delays.

```{warning}
Prepare the waterfall upstream: remove the baseline and normalize to zero
mean and a known common per-channel noise standard deviation (`noise_sigma`,
default 1). Analytical S/N assumes independent channel/time input noise before
the FDMT's modeled filtering. Normalization alone does not establish this
independence. Channel masking, common-mode subtraction, temporal filtering and
residual correlations can change the variance and require upstream calibration.
**The search does not estimate a baseline or RMS** from dedispersed rows.
```

Sigma accounts for the actual FDMT leaf box means and temporal covariance,
including each subband's tree lineage.
`FDMTPlan.get_effective_sigma_grid(width)` already gives sigma for the raw
full-band sum, so **do not multiply it by the number of channels**.

## Packed integer input

Set `config.input_nbits` to 1, 2, 4, 8, or 16 before construction and supply
`input_mean` and `noise_sigma` in the unsigned input's code units. For example,
8-bit data with mean 127 and standard deviation 12 uses:

```python
config = BoxcarConfig(input_nbits=8, input_mean=127.0, noise_sigma=12.0)
search = FDMTBoxcar(plan, config)
result = search.execute(uint8_block)
```

The mean may be fractional. These must be common calibrated statistics across
channels; the engine does not estimate them. Float32 remains the default
(`input_nbits=32`, `input_mean=0`). A constructed engine has a fixed input type.

- **Layout.** Packed arrays use `uint8` bytes, including for 16-bit samples.
  Each channel row occupies `ceil(nsamps * input_nbits / 8)` bytes,
  independently padded; sub-byte samples are LSB-first and 16-bit samples are
  little-endian. Native little-endian contiguous `uint16` arrays can be passed
  as `.view(np.uint8)`.
- **C++.** The C++ overloads accept `std::span<const uint8_t>` or
  `DeviceSpan<const uint8_t>` with the same layout on CPU and GPU backends.

:::{dropdown} Integer arithmetic and thresholds
:icon: info

The packed search keeps exact dyadic FDMT weights, sums, merge histories and
boxcar prefixes in uint64. It does not add the level-0 rounding used by the
packed `FDMT` API. This preserves a linear expected baseline and
avoids a distribution-dependent rounding bias when only input mean and sigma
are known. Its numerical reference is float-input FDMT on centered codes,
subject to float precision. The packed `FDMT` itself is unchanged.
Thresholds include the expected baseline and are rounded up to inclusive
integer cutoffs. Per-width peak comparisons remain integer until emission;
across-width reduction evaluates normalized S/N at threshold crossings because
different widths have different baselines and noise scales.
:::

Accumulator bounds are checked at construction, including prefix workspace.
The root merge is fused into scoring. The tree state is uint32 when the
plan bounds every tree value below 2^32, else uint64. On the GPU the packed
search computes levels 0–3 directly from integer running sums and, where the
plan allows, sums the upper levels' rows straight from a lower level (the
direct top of the float transform, see the performance guide), all in exact
integers. 8-bit input then takes about the float32 search time (0.94–1.05×
on the three presets, L40S). The CPU packed search does not fuse the lower
levels and takes 1.3× the float32 time on 8 threads (1.1–1.2× on one).

## Running a search

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
from dmtlib import FDMTPlan, FDMTBoxcar, BoxcarConfig, PeakFindingMode

plan = FDMTPlan(700.0, 1200.0, 256, 4096, 0.001, 512, subband_depth=2)
config = BoxcarConfig(
    max_boxcar=256,
    wtsp=1.5,
    detect_thresh=6.0,
    peak_mode=PeakFindingMode.CONTIGUOUS_PEAK_PER_WIDTH,
)
search = FDMTBoxcar(plan, config, nthreads=4)

# Each block is a normalized float32 array of shape (256, 4096).
for block in blocks:
    result = search.execute(block)
    consume(result.candidates)  # Owned structured NumPy array.
    if result.overflowed:
        report_drops(result.dropped_global, result.dropped_per_dm)
consume(search.flush().candidates)
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
#include <dmt/dmt.hpp>

dmt::plans::FDMTPlan plan(700.0F, 1200.0F, 256, 4096, 0.001F, 512);
dmt::BoxcarConfig config(256);
dmt::algorithms::FDMTBoxcar search(plan, config, true, dmt::Exec::cpu(4));
auto result = search.execute(waterfall); // std::span<const float>
// Consume result.candidates before the next result-changing operation.
auto last = search.flush();
```
:::
::::

The C++ and Python `BoxcarConfig` constructors accept the search settings
directly, and the fields stay editable until you construct the search. They
are validated when `FDMTBoxcar` combines them with its plan.

Choose `backend="cuda", device=0` in a CUDA build. Python GPU calls stage
input and block until candidates are available on the host. No full root
plane is copied to the host. `config` and `search_plans` properties return
copies; modifying them does not reconfigure the constructed engine.

## C++ and device pipelines

On GPU backends, `execute(DeviceSpan<const float>, Stream)` and
`flush_device(Stream)` return asynchronous device views. The candidate span
has capacity `max_candidates`, not the emitted length. Its one-element
`counts` span contains `CandidateCounts`; consume the first `emitted_count`
candidates after stream completion. `copy_result(stream)` waits and returns a
host view of the latest result.

```{warning}
Device and host views are engine-owned and invalidated by subsequent
result-changing operations. Python arrays own copies.
```

Order operations on one stream. Synchronize before changing streams on an
instance; instances are not safe for concurrent calls. CPU device calls reject
the operation. Host execute/flush calls block; GPU `reset_history()` is ordered
asynchronously on its supplied stream. GPU input staging is allocated lazily and
reused. Warm CPU execution and flush allocate no working memory.

## Widths, excursions and boundaries

Widths start at 1; each next width is
`max(previous + 1, ceil(previous * wtsp))`, stopping at `max_boxcar`.
`wtsp=2` gives powers of two, `1.5` a denser geometric grid, and `1` every
integer. Widths 1–4096 are supported on both backends, without a 32-filter cap.
Every trial is searched at the original time resolution.

**Trapezoid filters (`use_trapezoid=True`).** Each width `W >= 3` is searched
with a trapezoid *instead of* the boxcar: an edge of `w_edge = (W + 1) // 4`
samples at weight 1/2 on each side of a flat core at weight 1, still `W`
samples in all. Widths 1 and 2 stay boxcars. The analytical noise model
includes the taper exactly, so the S/N scale and the thresholds are
unchanged. The taper matches pulses that are not aligned with the sample
grid, or that have soft edges. A narrow pulse that straddles sample
boundaries gains about 6% S/N (the off-phase `[0.5, 1, 0.5]` profile at
`W = 3`). A flat pulse aligned with the boxcar loses about 5% (`W = 8`). The
reported `width` is the full support `W`, edges included. The trapezoids
replace the boxcars; they are not searched alongside them, so the number of
filters and the cost per width stay the same. Packed input supports them too:
the integer thresholds use the filter weight `2 (W - w_edge)`.

**Width screen.** With dense width grids, consecutive widths
`[w_lo, w_hi]` with `w_hi < 2 w_lo` form a group. Before any width in a group
is scored, one bound is checked per endpoint: the prefix sum at `t` minus the
smallest prefix sum over `[t - w_hi, t - w_lo]` (a sparse-table range
minimum). This bound is compared with the group's smallest threshold, and
only the groups that pass are scored width by width. The screen is exact, so
the candidates are the same with or without it. It is used only when it
merges widths, i.e. when there are fewer than half as many groups plus
range-minimum levels as widths. In practice that means `wtsp=1` grids
(e.g. 195 widths in 8 groups). `wtsp>=1.5` grids are unaffected. On the
CPU, the screen made `meerkat_lband` float32 at `wtsp=1` 4.4× faster. With
`use_trapezoid`, the GPU turns the screen off, and the CPU finds no group
worth screening (trapezoid widths stay single), so trapezoid searches pay
the full per-width cost.

`CONTIGUOUS_PEAK_PER_WIDTH`
: One peak per contiguous threshold excursion of each DM/width response.

`CONTIGUOUS_PEAK_ACROSS_WIDTHS`
: Take the highest normalized S/N across widths at each endpoint, then reduce
  contiguous excursions.

`ALL_ABOVE_THRESHOLD`
: Emit every above-threshold DM/width/endpoint.

Threshold comparisons use `>=`. Equal peaks select the earliest endpoint,
then narrower width. The code performs no gap bridging, across-DM clustering,
subband deduplication or rate limiting.

`peak_sample`, `begin_sample` and `end_sample` are absolute uint64 stream
indices of **trailing-window endpoints**. End is inclusive. For width `W`,
the filter at endpoint `t` covers `[t-W+1, t]`; its center is
`t - (W-1)/2.0`. Excursion bounds describe response endpoints, not the union
of all input windows in the excursion. Across-width excursions retain the
winning peak's width.

- **Cold windows.** They are skipped by default until their complete FDMT
  dependency support and boxcar lookback are available. This can span
  multiple blocks. `skip_cold_prefix=False` searches the zero-padded cold
  response using the stationary sigma table.
- **Block boundaries.** Raw sample carry and pending peaks survive internal
  chunks and input-block boundaries. A returned peak may therefore refer to an
  earlier block.
- **Flush.** Call `flush()` once at end of stream to emit remaining peaks;
  repeated flushes return no further candidates. Flush adds no padded samples
  and permits subsequent input to continue, with excursions split at that
  flush.
- **Reset.** `reset_history()` discards all histories and pending peaks and
  restarts sample 0.

## Subbands, overflow and memory

The search depth is the plan's `subband_depth` (default 0: full band only).
Depth `k` searches the `2**k` equal subbands the FDMT tree already computes,
plus the `2**k - 1` **staggered** subbands of the same width centred on their
boundaries:

| `subband_depth` | Subbands searched (1152 channels) |
| :--- | :--- |
| 0 | full band |
| 1 | + 2 halves (576 channels) + the central half [288, 864) |
| 2 | + 4 quarters (288) + 3 staggered quarters [144, 432), [432, 720), [720, 1008) |
| 3 | + 8 eighths (144) + 7 staggered eighths |

A staggered subband is one extra merge of two adjacent subbands from the next
level down that have different parents, so it costs about one merge level's
work, not a separate dedispersion. Without it, a narrowband burst sitting on
a boundary is split between two aligned subbands and is caught at best by the
level above. For example, a 200-channel burst anywhere in a 1152-channel band
recovers at worst 0.42 of its matched S/N with aligned subbands to depth 2,
and 0.72 with the staggered ones added; to depth 3 with staggered subbands
the worst case is 0.83.

The tree halves the band top-down, so the subbands at any depth differ by at
most one channel (exactly equal when the depth divides the channel count:
1152 channels stay equal down to 9-channel subbands). `subband_depth` must be
below `plan.niters`. Each row's sigma follows its own channels and tree
lineage. Candidates carry `subband_level` (the depth), `subband_idx` (tree
subbands first, then staggered ones), `staggered`, half-open input channel
extents and physical frequency edges. All planes share the candidate buffer.
Overlapping subbands detect the same burst more than once; group candidates
downstream (for example by time and DM, keeping the highest S/N).

On the CPU and GPU the lower tree is still fused into channel groups, but
only below the lowest level that holds staggered subbands
(`plan.max_fuse_levels`), since a staggered merge crosses group boundaries.

Counters reset per execute/flush call. `total_detected` counts closed
excursions (or points in all mode), including dropped candidates. A
`max_cands_per_dm` cap applies independently per plane/beam/DM/call; zero
removes that cap. Global capacity is `max_candidates`. Counters satisfy
`total_detected = emitted_count + dropped_per_dm + dropped_global`.
Scanning and history updates continue after saturation. Parallel candidate
order and the retained subset during overflow are unspecified.

The root merge is fused into scoring. Selected intermediate subbands are
searched in their existing FDMT buffers before the next merge. FDMT state,
sample histories, pending peaks, candidate buffers and bounded scoring
workspace remain allocated; fusion does not mean zero scratch memory.
`memory_usage` reports these categories and includes GPU host candidate
storage and lazily allocated input staging. It excludes caller input and
Python's owned output copies.

:::{dropdown} How the scorer works on CPU and GPU
:icon: info

Both backends score a row tile by tile from running prefix sums, and run the
peak bookkeeping only for tiles that need it. A cheap pass first checks every
width of the tile against its threshold. When no window crosses and no
excursion is open, every peak mode leaves its state unchanged, so the tile is
done; on noise that is nearly every tile, and the cost per window is one
prefix difference and one comparison.

- CPU: double prefixes (uint64 for packed input), slid from tile to tile;
  the carried samples are read once per call and written once at its end.
- GPU: one thread block per (beam, DM) row. The prefix of each 1024-sample
  tile is a warp-shuffle scan into a float32 ring in shared memory (a power of
  two of at least `max_boxcar + 1024` entries), rebased after every tile so
  values stay near the magnitude of one tile. Width and threshold tables are
  staged in shared memory. Tiles with a crossing or an open excursion run the
  exact ordered run summaries.
:::

```{note}
Candidate semantics match across backends; S/N is compared with tolerances,
not promised bit-identical. Arithmetic near an exact threshold or near-equal
peak can affect selection.
```

## Validation and performance

CPU tests compare against materialized FDMT (or FDMT-FFT) and independent
direct float64 window sums, impulse-response energy, Gaussian noise, streaming
boundaries, overflow and allocation checks. GPU tests are tagged
`[gpu][boxcar]`.

:::{dropdown} Building and running the GPU tests
:icon: info

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDMT_GPU=CUDA -DDMT_BUILD_TESTING=ON -DDMT_BUILD_PYTHON=OFF
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
compute-sanitizer --tool memcheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
compute-sanitizer --tool racecheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
compute-sanitizer --tool synccheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
```
:::

Build benchmarks with `DMT_BUILD_BENCHMARKS=ON`. The streaming benchmarks run
at the reference point of a suite configuration (`DMT_BENCH_CONFIG`:
`cryopaf`, `cryopaf_lband`, `parkes_uwl` or `meerkat_lband`) with widths up to 100 ms
(`max_boxcar = floor(0.1 s / tsamp)`):

```bash
# GPU: transform alone, transform + separate search, FDMTBoxcar (float32,
# uint8, host arrays) and FDMT-FFT / FDMTFFTBoxcar
DMT_BENCH_CONFIG=cryopaf build/bench/dmt_bench \
  --benchmark_filter='^BM_fdmt_boxcar_cuda/'
# CPU
DMT_BENCH_CONFIG=cryopaf_lband build/bench/dmt_bench \
  --benchmark_filter='^BM_fdmt_boxcar_stream/'
```

On noise, with 32768-sample blocks and float32 input, no subband search
(NVIDIA L40S, device-resident, median of 3 runs; AMD EPYC 9654, 8 threads
pinned one per CCD with `OMP_PLACES='{0}:8:8'`, median of 3 runs):

| preset, backend | `FDMT` (writes the plane) | `FDMTBoxcar`, wtsp 1.5 | `FDMTBoxcar`, wtsp 1.0 |
| :--- | ---: | ---: | ---: |
| `cryopaf` (1545 trials), GPU | 21.8 ms | 22.4 ms (12 widths) | 23.3 ms (195 widths) |
| `cryopaf_lband` (1941 trials), GPU | 11.7 ms | 12.4 ms (13 widths) | 13.4 ms (219 widths) |
| `meerkat_lband` (4057 trials), GPU | 25.9 ms | 27.9 ms (12 widths) | 29.7 ms (163 widths) |
| `parkes_uwl` (4749 trials), GPU | 353 ms | 355 ms (19 widths) | – |
| `cryopaf`, CPU | 258 ms | 311 ms | 334 ms |
| `cryopaf_lband`, CPU | 141 ms | 472 ms | 564 ms |
| `meerkat_lband`, CPU | 368 ms | 510 ms | 553 ms |

On the GPU the root merge the search saves costs about as much as scoring a
dozen widths, so the search adds 1–15% (and on `parkes_uwl`, `fused_uint8`
runs in 342 ms and 2.41 GB working memory, 6.1× real time). On the CPU the
scorer is a separate pass over each chunk's root rows (1.2–4.0× the
transform), still well faster than real time on 8 threads; on `parkes_uwl`
with 8-bit input (`chunk = 1024`), `FDMTBoxcar` automatically uses the
skewed-window engine in 2.72 GB working memory (9.22 s per block on 8
threads). Thanks to the width screen, scoring every integer width up to 100 ms
costs about the same as a dozen widths on both backends. `working_bytes` is
the algorithm allocation estimate, not process RSS.

## FDMTFFTBoxcar

```python
from dmtlib import FDMTPlan, FDMTFFTBoxcar, BoxcarConfig

plan = FDMTPlan(704.0, 1216.0, 512, 32768, 512e-6, dm_arr=dm_trials)
search = FDMTFFTBoxcar(plan, BoxcarConfig(max_boxcar=195, wtsp=1.5),
                       backend="cuda")
for block in blocks:
    consume(search.execute(block).candidates)
consume(search.flush().candidates)
```

`FDMTFFTBoxcar` takes the same `BoxcarConfig`, returns the same candidates and
result views, and has the same streaming, flush and overflow rules. It runs
`FDMTFFT` in valid mode and hands the DM rows of each overlap-save segment to
the scorer straight from the inverse transform: on the GPU the scorer reads
the cuFFT output buffer in place of the copy into a DM–time plane, and on the
CPU it scores one beam's segment from a buffer of `ndm × segment` samples.

- **Noise model.** A merge's shift, integer or fractional, is an all-pass
  operation: it leaves each channel's noise spectrum unchanged. A row's boxcar
  variance therefore depends only on its channels' level-0 boxes, as for
  `FDMTBoxcar`, and the same analytical sigma applies. Channels with
  `kill_mask[c] = 0` read as zero and are left out of it.
- **Latency.** With fractional delays the transform needs
  `fdmt_fft::kFractionalGuard` (64) samples of look-ahead, so a block searches
  input samples `[block_start - 64, block_end - 64)`. Candidate sample indices
  are input-stream indices, comparable with `FDMTBoxcar`'s; `output_latency`
  reports the lag (0 with `fractional_delays=False`). Cold windows also wait
  for the interpolation guard behind each row's support.
- **Streaming.** With fractional delays, overlap-save blocks differ from one
  long call by the truncation of the interpolation kernel (FDMTFFT: below
  0.5% of the row peak), so a near-threshold point can move between blocked
  and unblocked runs; with integer delays they agree to float rounding.
- **Limits.** Float32 input only and plan `subband_depth=0`: subband rows exist
  only in the Fourier domain and would each need an extra inverse transform.

```{seealso}
- [Noise calibration](noise_calibration.md): the exact per-trial sigma.
- [Benchmarks: candidate search](../benchmarks.md#candidate-search).
- [Streaming and history](streaming_and_history.md).
```
