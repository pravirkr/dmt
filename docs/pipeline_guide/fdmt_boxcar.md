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
| Kill mask | yes, in the noise model of every subband row; also per block (float32) | yes |
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
independence. Channel masking (other than through `kill_mask`, below),
common-mode subtraction, temporal filtering and residual correlations can
change the variance and require upstream calibration.
**The search does not estimate a baseline or RMS** from dedispersed rows.
```

Sigma accounts for the actual FDMT leaf box means and temporal covariance,
including each subband's tree lineage.
`FDMTPlan.get_effective_sigma_grid(width)` already gives sigma for the raw
full-band sum, so **do not multiply it by the number of channels**.

### Kill mask

Pass `kill_mask=` (Python, keyword-only; C++: the trailing
`std::span<const uint8_t>` constructor argument) to drop channels flagged
as RFI: one bool or integer per channel, low frequency first, 0 for a
killed channel. Killed channels read as zero (code 0 for packed input),
whatever the block holds there, and are left out of every row's noise
model and packed mean, so S/N stays calibrated on the channels that remain:
the sigma is `FDMTPlan.get_subband_sigma_grid(..., kill_mask=mask)`. A pulse
across all `N` channels with `k` of them killed scores `sqrt((N - k) / N)` of
its unmasked S/N. Zeroing the channels upstream without a mask is not the
same: the noise model still counts them, so sigma is too high by
`sqrt(N / (N - k))` (5% S/N for a 10% mask). Rows whose channels are all
killed (a killed subband) have no noise and never detect.

Each block is copied once into a staging buffer to zero the killed rows; it
appears in `memory_usage.staging` and counts against `max_memory_bytes`.
Without a mask there is no copy.

#### A mask per block

A real-time pipeline flags RFI per gulp, so `execute()` also takes a mask
for its block (float32 input): `search.execute(block, kill_mask=mask)` in
Python, `execute(waterfall, kill_mask)` in C++ (host or device waterfall;
the mask is a host span, read before the call returns). The format and
convention are the constructor's: one value per channel, nonzero active.
The block's mask applies to every beam and sample of the block and is
ANDed with the constructor mask; `None` (an empty span, or the overloads
without a mask) means the constructor mask alone, not the previous block's.
Static zaps can stay in the constructor and the dynamic flags go per call.
The stream keeps its FDMT history, boxcar carry and open excursions across
mask changes; samples keep the mask they were read with.

While the mask stays the same, S/N is calibrated exactly as with that mask
given to the constructor. When it changes between blocks, the endpoints
whose input windows reach back into the previous block, the first
`S + W - 1` of each row (`S` the row's FDMT support with every channel
active, `W` the boxcar width), mix samples read under both masks. Those
endpoints use the variance of the **OR** of the two masks, an upper bound
of their true variance: a change adds no false alarms, and S/N there is low
by at most `sqrt(V_or / V_true)` (about `1 - k / 2G` when `k` of `G` similar
channels flip). A wholly killed block still closes the excursions of the
block before it. At most `(dt_max + max_boxcar) / nsamps` of the stream is
affected per change, and none while the mask is unchanged.

The first per-call mask checks two things and raises (`ValueError`;
C++ `std::invalid_argument`) without changing the stream if either fails:

- the block is long enough for every window to span at most two blocks:
  `nsamps >= S_max + max_boxcar - 1` (`parkes_uwl` at depth 4 with 100 ms
  boxcars needs 255,452 samples);
- the noise tables fit in `max_memory_bytes`: the noise model, a few table
  sets of `rows x widths x 8` bytes, the staging block if the constructor
  had no mask and, on GPUs, device and pinned copies of the tables.

The tables of a new mask are computed on the host when it first appears,
with the instance's `nthreads` (also on GPUs, where it has no other use):
a few milliseconds on the bench presets, 0.2 s per set on `parkes_uwl` at
depth 4 with 8 threads (1.3 s with one). A change needs a set for the new
mask and, unless it only kills or only revives channels, one for the OR;
the last three masks are cached. `skip_cold_prefix` keeps the constructor mask's
warm-up. `reset_history()` clears the mask state, and `drain()` pads with
the last block's mask. `search_plans` (`get_search_plans()`) shows the
constructor mask's tables only. Packed input takes no per-call mask.

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
consume(search.drain().candidates)  # end of stream
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
auto last = search.drain(); // end of stream
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
`max(previous + 1, ceil(previous * wtsp))` while it does not exceed
`max_boxcar`, and `max_boxcar` itself ends the bank when the last step falls
short of it (`max_boxcar=9, wtsp=2` searches 1, 2, 4, 8, 9), so the widest
box searched is always `max_boxcar`. `wtsp=2` gives powers of two, `1.5` a
denser geometric grid, and `1` every integer, each ending on `max_boxcar`.
`dmtlib.generate_box_width_trials(max_boxcar, wtsp)` returns the bank.
Widths 1–4096 are supported on both backends, without a 32-filter cap.
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
the integer thresholds use the filter weight `2 (W - w_edge)`. Boxcar-only is
the default; a mode that searches trapezoids alongside the boxcars would be
added as a separate option.

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
  response using the stationary sigma table, which does not describe it;
  hence the default `True`.
- **Block boundaries.** Raw sample carry and pending peaks survive internal
  chunks and input-block boundaries. A returned peak may therefore refer to an
  earlier block.
- **End of stream.** Call `drain()` once at the end of the stream. It
  searches the samples still behind the search latency (the last
  `get_search_latency()` samples of the skewed subband search, the last
  `get_output_latency()` of `FDMTFFTBoxcar` with fractional delays; none
  otherwise), then emits the open excursions. Endpoints after the last
  sample are never searched, so no candidate lies past the end. The stream
  ends there: `execute()` raises until `reset_history()`.
  `get_searched_nsamps()` (`searched_nsamps` in Python) gives the stream
  samples every row has searched so far.
- **Flush.** `flush()` only emits the open excursions: repeated flushes
  return no further candidates, it adds no samples and lets the input
  continue, with excursions split at that flush.
- **Reset.** `reset_history()` discards all histories and pending peaks and
  restarts sample 0.
- **Candidate frontier.** In the contiguous modes an excursion is emitted
  only when it ends, and nothing limits how long it stays open, so a later
  call can return a candidate far older than `dt_max + max_boxcar` samples
  (persistent RFI, a long bright event). `get_candidate_frontier()` gives
  the oldest `begin_sample` any later `execute()`, `flush()` or `drain()`
  can still emit: the smallest begin of the excursions still open (any
  beam, row or width), or `get_searched_nsamps()` when none is open. It
  never decreases between resets. Use it as the watermark for releasing
  grouped candidates instead of a fixed lag. With `ALL_ABOVE_THRESHOLD`
  nothing stays open, so it is the searched position. After `flush()` or
  `drain()` it equals `get_searched_nsamps()`, and `reset_history()` sets it
  to 0. After device calls, synchronise their stream before reading it, as
  for a device result. `FDMTFFTBoxcar` has the same method.

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

### Memory budget and wide bands

`FDMTBoxcar(..., max_memory_bytes=...)` (default 10 GiB, `FDMT` takes the
same argument) bounds everything the instance allocates
(`get_memory_usage().total()`, not the caller's input or output). The
boxcar's own buffers are fixed by the plan and config: each searched row
keeps `max_boxcar` carried samples (float32, or uint32 for packed input
when every tree value fits 32 bits) and its pending peaks. The FDMT takes
the fastest path that fits in the rest:

- **The tree path** when its streaming history fits: every preset except
  `parkes_uwl`, which is unchanged.
- **The skewed-window path** otherwise (a wide band at fine time
  resolution with a long DM sweep: 6656 channels at 64 µs to DM 2000 keeps
  10–14 GB of tree history). It keeps only the raw input of each channel's
  own dispersion reach and recomputes the upper levels over short per-row
  windows ([](skewed_window_fdmt.md)), at the longest automatic chunk
  (4096 down to 256 samples) that fits; a longer chunk is faster.
- Construction raises `ValueError` (`std::invalid_argument`) when nothing
  fits.

On the skewed path a subband search reads every searched row (tree,
staggered, and delays the full-band grid does not use) from the full
band's windows, each at a fixed **per-row lag**: at the call whose
full-band samples are `[T, T + n)` a subband row is searched over
`[T - lag, T - lag + n)`. The lag is about the dispersion delay from the
subband down to the bottom of the band (zero for the lowest subbands, up to
the full sweep for the top ones), i.e. a subband candidate is reported at
about the time the full-band search would report the same burst.
Candidates and S/N are the tree path's, bit for bit; only the call that
returns a candidate moves. `flush()` adds no samples, so at the end of a
stream call `drain()`: it searches the last `lag` samples of each subband
row, and the candidates then match the tree path's over the whole stream.

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
  Tiles grow with the widest boxcar (1024 to 4096 samples, about four times
  the lookback) so rebuilding the lookback per tile stays cheap, and every
  row of a level's planes shares one dynamically scheduled loop.
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

Build benchmarks with `DMT_BUILD_BENCHMARKS=ON`. The FDMTBoxcar benchmarks
are the `boxcar` suite of `dmt_bench_suite`, at the reference point of each
preset (`bench/README.md`) with widths up to 100 ms
(`max_boxcar = floor(0.1 s / tsamp)`) and the default memory budget:

```bash
OMP_PLACES='{0}:8:8' python bench/scripts/run_suite.py --suite boxcar \
    --config meerkat_lband          # or cryopaf_lband, cryopaf_low, parkes_uwl
```

On noise, with 32768-sample blocks, float32 input, warmed histories (NVIDIA
L40S, device-resident; AMD EPYC 9654, 8 threads one per CCD; median of 3):

| preset, backend | `FDMT` (writes the plane) | `FDMTBoxcar`, wtsp 1.5 | `FDMTBoxcar`, wtsp 1.0 |
| :--- | ---: | ---: | ---: |
| `meerkat_lband` (4057 trials), GPU | 25.9 ms | 27.7 ms (12 widths) | 29.6 ms (163 widths) |
| `cryopaf_lband` (1941 trials), GPU | 11.6 ms | 12.8 ms (13 widths) | 14.0 ms (219 widths) |
| `cryopaf_lband`, subband depth 1, GPU | – | 15.1 ms | 19.0 ms |
| `cryopaf_low` (1545 trials), GPU | 21.8 ms | 22.4 ms (12 widths) | 23.2 ms (195 widths) |
| `meerkat_lband`, CPU | 359 ms | 502 ms | 552 ms |
| `cryopaf_lband`, subband depth 1, CPU | 141 ms | 468 ms | 544 ms |
| `cryopaf_low`, CPU | 260 ms | 312 ms | 339 ms |

Measured with the 0.7.0 width banks, before a bank ended on `max_boxcar`
(0.8.0): the wtsp 1.5 banks of these presets now hold one more width (13, 14
and 13), which adds a few percent of search work.

On the GPU the root merge the search saves costs about as much as scoring a
dozen widths, so the full-band search adds 3–10%; a depth-1 subband search
on `cryopaf_lband` adds a fifth (wtsp 1.5) to a third (wtsp 1.0) more. On the CPU the scorer is a separate
pass over each chunk's searched rows (1.2–3.3× the transform), still well
faster than real time on 8 threads. Thanks to the width screen, scoring
every integer width up to 100 ms costs about the same as a dozen widths on
both backends. `working_bytes` in the results is the allocation plus the
input block, not process RSS.

The `parkes_uwl` search (6656 channels at 64 µs, 4749 trials, 2.1 s
blocks, subband depth 4: 218k searched rows, wtsp 1.5) runs within the
10 GiB budget on the skewed-window path:

| backend | input | chunk | time per block | real-time factor | working memory | peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| GPU | uint8 | 4096 | 1.16 s | 1.81× | 9.4 GiB | 9.2 GiB |
| GPU | float32 | 2048 | 1.59 s | 1.32× | 10.8 GiB (9.96 GiB engine) | 10.7 GiB |
| CPU, 8 threads | uint8 | 4096 | 14.3 s | 0.15× | 6.2 GiB | 6.3 GiB |
| CPU, 8 threads | float32 | 2048 | 20.1 s | 0.10× | 9.2 GiB | 9.4 GiB |

*working memory* is the library's accounting plus the input block; *peak* is
the measured device memory (GPU) or process resident set (CPU) in the same
run. The GPU peak can be below the working memory because the accounting
also counts host-side tables and the host candidate buffer.

The full band alone takes 0.36 s on the GPU (5.8× real time, 2.1–2.7 GiB).
Searching at a coarser resolution is much cheaper: at 256 µs the GPU runs
depth 4 at 36× real time and 8 CPU threads keep up (1.15–1.35×), and at
512 µs the CPU is 6× faster than real time (the real-time sweep on the
[Benchmarks](../benchmarks.md#parkes-uwl-search) page).

## FDMTFFTBoxcar

```python
from dmtlib import FDMTPlan, FDMTFFTBoxcar, BoxcarConfig

plan = FDMTPlan(704.0, 1216.0, 512, 32768, 512e-6, dm_arr=dm_trials)
search = FDMTFFTBoxcar(plan, BoxcarConfig(max_boxcar=195, wtsp=1.5),
                       backend="cuda")
for block in blocks:
    consume(search.execute(block).candidates)
consume(search.drain().candidates)  # searches the last output_latency samples
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
- **Memory.** `max_memory_bytes` (keyword-only, default 10 GiB) bounds
  everything the instance allocates (`memory_usage`), as on `FDMTBoxcar`. Over
  it the overlap-save segments are shortened (each try halves the transform
  length; `chunk_nsamps` reports the segment) until the instance fits, and
  construction raises `ValueError` with the memory at the shortest segment
  when nothing does. Shorter segments change the rows as blocking does (see
  Streaming).
- **Limits.** Float32 input only and plan `subband_depth=0`: subband rows exist
  only in the Fourier domain and would each need an extra inverse transform.

```{seealso}
- [Noise calibration](noise_calibration.md): the exact per-trial sigma.
- [Benchmarks: Parkes UWL search](../benchmarks.md#parkes-uwl-search).
- [Streaming and history](streaming_and_history.md).
```
