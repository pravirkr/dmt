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

### Red noise and wide boxcars

The sigma of a width-`W` box assumes white input noise. Slow variation
that survives upstream (gain drift, baseline wander, zero-DM power, red
noise from the receiver or RFI) is correlated over many samples, so a box's
real variance grows faster than its sigma. The excess grows with `W`. At
the widest widths, stretches of plain noise then cross the threshold at a
high S/N in every DM row, and a pipeline sees a flood of wide candidates.

The search cannot correct for this. It does not estimate a baseline or RMS
from the dedispersed rows, and its running state per row is only the last
`max_boxcar` samples, so it cannot see trends longer than the widest box.
Correct for it upstream:

- Detrend each channel on a timescale several times longer than the widest
  box: subtract a running median or mean, or high-pass filter, before
  normalizing. Zero-DM subtraction alone doesn't remove drifts that differ
  between channels.
- Keep `max_boxcar` at about 100 ms or less (in samples, `0.1 / tsamp`)
  unless the input is known to be white on longer timescales. Bursts wider
  than that are rare, and every extra width adds red-noise candidates.
- To check, run the search on data without a burst and look at the S/N
  distribution per width: on white input it follows the model at every
  width; red noise shows up as a tail that grows with width.

`OverflowPolicy.KEEP_BRIGHTEST` (below) doesn't help here. It keeps the
highest S/N, and red-noise candidates at wide widths can have a higher S/N
than a real burst, so they still take the quota.

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

On the CPU each block is copied once into a staging buffer to zero the
killed rows; it appears in `memory_usage.staging` and counts against
`max_memory_bytes`. On a GPU the FDMT reads killed rows as zero itself, from
one byte per beam and channel, so there is no copy of the block (device
input under a mask needs no extra block-sized memory). Without a mask there
is no copy.

#### A mask per block

A real-time pipeline flags RFI per gulp, so `execute()` also takes a mask
for its block: `search.execute(block, kill_mask=mask)` in
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
given to the constructor. When it changes, the endpoints whose input
windows reach back before their block, the first `S + W - 1` of each row
(`S` the row's FDMT support with every channel active, `W` the boxcar
width), mix samples read under several masks. Those endpoints use the
variance of the **history mask**: the block's channels and every channel
live within `R = S_max + max_boxcar - 1` samples before it (for blocks of at
least `R` samples, simply the OR of the previous and the current mask). It
is an upper bound of their true variance: a change adds no false alarms,
and S/N there is low by at most `sqrt(V_history / V_true)` (about
`1 - k / 2G` when `k` of `G` similar channels flip). A wholly killed block
still closes the excursions of the block before it.

Any block length works, down to one sample, and the memory of the masks
does not grow with the number of blocks a window spans: one timestamp per
channel records when it was last live. Paths with a search latency (the
skewed subband search) keep the blocks a call still searches; when the
blocks are shorter than that needs, the oldest block kept stands for the
older ones with a single **catch-all** table (every channel live within
`R` samples before the oldest endpoint still searched), again an upper
bound. Short blocks with masks that change often therefore score most
high-DM endpoints with the history or catch-all variance, and computing
the tables of new masks (below) costs time per call.

The first per-call mask checks that its noise tables fit in
`max_memory_bytes` and raises (`ValueError`; C++ `std::invalid_argument`)
without changing the stream if not: on the CPU the noise model, a few table
sets of `rows x widths x 8` bytes (16 for packed input) and the staging
block if the constructor had no mask; on GPUs the device copies of the tables (the host-side model
and tables are reported in `memory_usage.host`).

The tables of a new mask are computed on the host when it first appears,
with the instance's `nthreads` (also on GPUs, where it has no other use):
a few milliseconds on the bench presets, 0.2 s per set on `parkes_uwl` at
depth 4 with 8 threads (1.3 s with one). A change needs a set for the new
mask and, unless it only kills or only revives channels, one for the OR;
the last three masks are cached. `skip_cold_prefix` keeps the constructor mask's
warm-up. `reset_history()` clears the mask state, and `drain()` pads with
the last block's mask. `search_plans` (`get_search_plans()`) shows the
constructor mask's tables only.

**Packed input** (Python `search.execute(codes, kill_mask=mask)` with
uint8 bytes; C++ `execute(waterfall, kill_mask)`, host or device input; CPU
and GPU, every screen): channels the block's mask kills, but the
constructor's keeps, read the code `input_mean` instead of 0, so
`input_mean` must be an integer. Every window's expected sum is then the
constructor plan's, exactly, and only the cutoffs and S/N scales change with
the mask, by the same rules as on float32 input. Without box smearing the
candidates equal those of the float32 search on `code - input_mean`.
Per-width peaks whose samples use different tables are ranked by S/N, then
by the raw sum as before.

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
asynchronously on its supplied stream. GPU host-input staging is reserved at
construction. Warm CPU execution and flush allocate no working memory.

## Widths, excursions and boundaries

Widths start at 1; each next width is
`max(previous + 1, ceil(previous * wtsp))` while it does not exceed
`max_boxcar`, and `max_boxcar` itself ends the bank when the last step falls
short of it (`max_boxcar=9, wtsp=2` searches 1, 2, 4, 8, 9), so the widest
box searched is always `max_boxcar`. `wtsp=2` gives powers of two, `1.5` a
denser geometric grid, and `1` every integer, each ending on `max_boxcar`.
`dmtlib.generate_box_width_trials(max_boxcar, wtsp)` returns the bank.
Every trial is searched at the original time resolution, without a
32-filter cap.

**How wide.** The search is tuned for `max_boxcar` up to 4096 samples
(262 ms at 64 µs). Wider boxes, up to 2^20 samples, are searched exactly
(the same candidates), with no speed promise, in banks of at most 4096
widths (so `wtsp` must exceed 1 beyond 4096): on a GPU their prefix rings
move from shared to device memory. The same happens below 4096 on GPUs
with little shared memory per block, after the search first drops the
width screen and then uses smaller blocks; the search path then ends in
`(boxcar: ...)`. A wide box also costs memory per DM row (its carry holds
`max_boxcar` samples, counted in the budget).
For pulses much wider than about 100 ms, prefer downsampling the data in
time and searching narrower boxes: it is much faster, and red noise (see
[Red noise and wide boxcars](#red-noise-and-wide-boxcars)) grows with the
width.

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

**Width screen (`screen_mode`).** `BoxcarConfig.screen_mode`
(`dmtlib.BoxcarScreenMode`, C++ `dmt::BoxcarScreenMode`) selects how quiet
tiles are rejected before per-width scoring:

- `AUTO` (default): picks the fastest lossless screen for the backend —
  `KADANE` on the CPU and `PREFIX` on a GPU.
- `KADANE` (lossless, `max_boxcar` up to $2^{20}$): partitions the width bank
  into up to three convex-hull buckets and runs a biased Kadane scan per
  bucket (plus a bounded `LagWindow` for `{1, 2, 1}` trapezoid edge lags). A
  bucket that never reaches its cut cannot cross any filter threshold in that
  bucket; quiet tiles skip Stage-2 prefix sums altogether. Unlike `PREFIX`,
  `KADANE` screens sparse geometric banks (`wtsp >= 1.5`) and `use_trapezoid`
  equally well, with identical candidates.
- `PREFIX` (lossless, `max_boxcar` up to $2^{20}$): groups consecutive boxcar
  widths $[w_{\mathrm{lo}}, w_{\mathrm{hi}}]$ with $w_{\mathrm{hi}} < 2 w_{\mathrm{lo}}$
  and bounds all widths in a group by the running prefix sum at `t` minus a
  sparse-table range minimum over $[t - w_{\mathrm{hi}}, t - w_{\mathrm{lo}}]$.
  Used when groups plus range-minimum levels are fewer than half the widths
  (dense `wtsp=1` grids); `wtsp >= 1.5` and `use_trapezoid` widths stay
  ungrouped.
- `KADANE_SURROGATE` (approximate, `max_boxcar` up to 4096): standalone
  multi-bias Kadane excursion detector with local lookback replay, skipping
  the full-stream Stage-2 prefix scan. Candidate counts and S/N are
  approximate rather than exact.
- `NONE` (lossless, `max_boxcar` up to $2^{20}$): disables Stage-1 screening
  and scores every width on every tile directly (`AUTO`, `KADANE`, `PREFIX`
  and `NONE` emit identical candidates).

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
  nothing stays open, so it equals `get_searched_nsamps()` (minus
  `FDMTPlan::get_level0_causal_delay()` clamped at 0 when `chan_dm != 0`).
  After `flush()` or `drain()` it reaches the full searched frontier, and
  `reset_history()` sets it to 0. After device calls, synchronise their stream
  before reading it, as for a device result. `FDMTFFTBoxcar` has the same
  method.

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

### Subband DM grids

By default every subband searches the trials the tree computes for it, which
are as dense as the full band's grid needs. A narrow subband sweeps less, so
it needs fewer trials. `FDMTPlan(..., subband_max_snr_loss=0.1)` (off at 0)
gives each staggered subband its own `generate_dm_grid` trials at that loss
(one-sample pulses), and has the search score only the tree-subband rows
nearest such a grid; the tree still computes every row its parents need.
The full band is never pruned. Each plane lists the subband trials it scores
in `coord_indices` (the plan grid's `search_rows`), and `dm_idx` indexes the
plane's own `dm_grid`.

At depth 4, with a full-band grid at 10% loss and `subband_max_snr_loss` 0.1,
on a GPU with 8-bit input:

| Preset | Scored rows | Memory (budgeted) | Time per 32768-sample block |
|---|---:|---:|---:|
| Parkes UWL | 303,442 -> 99,433 | 7.77 -> 6.45 GiB | 1.64 -> 1.51 s |
| MeerKAT L-band | 82,861 -> 36,914 | 0.63 -> 0.51 GiB | 94 -> 61 ms |

Building the plan takes 0.2-0.5 s longer at depth 4 (one grid per subband).

The subband grids inherit the limits of `generate_dm_grid`'s S/N model: narrow
subbands are dominated by intra-channel smearing, where it is optimistic.
Pulses confined to one subband lose on average 1-4% more S/N than with every row
searched, but the worst ones lose up to about 1.5 times `subband_max_snr_loss` (0.10-0.15 at 0.1, on CRYO-PAF,
CRYO-PAF L-band and MeerKAT L-band). For a firm bound, ask for about 0.65
times the loss you can accept.

Counters reset per execute/flush call. `total_detected` counts closed
excursions (or points in all mode), including dropped candidates. A
`max_cands_per_dm` cap applies independently per plane/beam/DM/call; zero
removes that cap. Global capacity is `max_candidates`. Counters satisfy
`total_detected = emitted_count + dropped_per_dm + dropped_global`.
Scanning and history updates continue after saturation. Parallel candidate
order is unspecified.

What survives an overflow depends on `BoxcarConfig.overflow_policy`:

- `OverflowPolicy.DROP_NEW` (default): first come. The retained subset is
  unspecified, and in practice it is the earliest candidates of the call: an
  RFI storm at the start of a block can fill the quotas and hide a bright
  burst later in the same block.
- `OverflowPolicy.KEEP_BRIGHTEST`: exactly the brightest. Each row keeps its
  `max_cands_per_dm` highest-S/N candidates of the call, and the call keeps
  the `max_candidates` highest of those (ties go to the earlier peak
  sample, then beam, subband, DM row and width), so the retained set is
  deterministic. Each row has a store of `max_cands_per_dm` candidates,
  counted in `memory_usage.candidates` and the memory budget, so the cap
  must be in 1..1024 and is best small (e.g. 16): the `DROP_NEW` default of
  4096 per row would be about 20 GB on a UWL-sized search (on GPUs an entry takes 32 bytes, on the CPU a full
  candidate). On GPUs the brightest are kept per row inside the scoring
  kernel; when the stores can exceed `max_candidates`, a radix select on
  the device picks the global brightest at the end of the call.

Independently of the policy, `PeakFindingMode.CONTIGUOUS_PEAK_ACROSS_WIDTHS`
emits one candidate per excursion instead of one per width, which cuts the
count during a storm by about the number of widths.

### Planning memory: block length, budget and chunk

Three constructor arguments touch memory, each with one job:

- **`nsamps`** (the plan's block length) is your pipeline's I/O unit: it
  sets your input memory, your latency, and the per-call units
  (`max_candidates`, per-call kill masks). It is not a memory setting for
  an `FDMTBoxcar` search: the FDMT's memory does not depend on it (Parkes
  UWL depth 4 holds the same 3.86 GiB at 65536 and at 262144 samples), and
  any length works, since histories carry across calls. Two exceptions
  scale with the block: a kill mask on the CPU copies the block into a
  budgeted staging buffer, and the FDMT-FFT engines transform whole
  blocks (`FDMTFFTBoxcar` also reserves a device copy of one).
- **`max_memory_bytes`** is the only memory setting. The default is a
  fixed 10 GiB; set it to what you give dmt on your machine or card. A
  larger budget can buy a faster configuration.
- **`chunk_nsamps`** is an expert override for benchmarks and unusual
  plans. Leave it at 0 (automatic). Forcing a chunk can raise where the
  automatic choice fits.

What dmt promises for `FDMTBoxcar`, on CPU and GPU (`FDMT` and the FDMT-FFT
engines follow the same rules, except where noted):

1. **dmt counts what it allocates, and only that.** `memory_usage` and the
   estimates report the instance's own buffers, including (for
   `FDMTBoxcar` and `FDMTFFTBoxcar`) the device copy it makes of host
   (NumPy) input. Your input block and anything else your pipeline holds
   are yours to plan.
2. **It fits or raises at construction.** Nothing allocates during the
   stream. The error states what the configuration needs. (Exception:
   plain `FDMT` and `FDMTFFT` on a GPU still copy a host block and output
   plane on their first host call, outside the budget; their estimates
   report it as `host_staging`.)
3. **A fast configuration that fits.** dmt picks the path and chunk from a
   short list in measured speed order, by a simple rule that is close to
   the fastest that fits (not an exhaustive search: planning stays cheap).
   There is no timing trial at construction. A larger budget is not
   always faster: in narrow windows a budget first fits a short chunk of
   the faster path (see the table below).
4. **Every budget from the minimum up builds.**
   `estimate_minimum_memory()` reports the minimum.
5. **The same candidates on every configuration**, to float precision: a
   budget that moves the search to another path or chunk changes float
   rounding only. (`FDMTFFTBoxcar` with fractional delays: shorter
   segments also move rows by a few hundredths of sigma; see below.)
6. **Memory follows the arguments only.** On a GPU, construction (never an
   estimate) also checks the chosen configuration against the device's
   free memory and raises `MemoryError` (`std::bad_alloc`) before
   allocating anything when it does not fit; free memory never changes
   the choice.

To plan, estimate before you build (nothing is allocated):

```python
est = dmtlib.FDMTBoxcar.estimate_memory(
    plan, config, backend="cuda", max_memory_bytes=budget)
est.usage.budgeted   # what dmt holds on the device
est.chunk_nsamps, est.search_path, est.search_latency
low = dmtlib.FDMTBoxcar.estimate_minimum_memory(plan, config, backend="cuda")
low.max_memory_bytes # the smallest budget that builds
```

Your pipeline's total is `est.usage.budgeted` plus its own input blocks
(and, on a GPU, `est.usage.host` of host memory). Parkes UWL depth 4 (6656
channels at 64 µs, DM 100–1100, `max_boxcar` 4096, 8-bit input,
65536-sample blocks; L40):

| `max_memory_bytes` | path, chunk | budgeted | `search_latency` |
| :--- | :--- | :--- | :--- |
| minimum | skewed, 256 | 3.86 GiB | 138833 samples (8.9 s) |
| 6 GiB | skewed, 2048 | 5.69 GiB | 138833 |
| 8–10 GiB | skewed, 4096 (2.8x real time) | 7.79 GiB | 138833 |
| 10.2–10.6 GiB | direct, 512 (1.9x) | 10.14 GiB | 0 |
| 11–11.5 GiB | direct, 1024 (2.7x) | 10.66 GiB | 0 |
| 12 GiB | direct, 2048 (3.2x) | 11.71 GiB | 0 |

`max_boxcar` and `subband_max_snr_loss` are the other levers: with
`max_boxcar` 1536 the minimum is 2.26 GiB, and with
`subband_max_snr_loss` 0.2 as well 1.63 GiB. Both change what is searched.
Float input needs more (minimum 6.04 GiB here).

A **search latency** above 0 means the skewed path: subband rows return
their candidates up to that many samples later, and the end of a stream
needs `drain()` (not `flush()`) to search them. GPU subband searches take
the skewed path when the budget gives it, and CPU subband searches on packed
input with blocks over 2048 samples take it first (it is faster there), so
plan for `drain()` whenever `est.search_latency > 0`.

### Memory budget and wide bands

`FDMTBoxcar(..., max_memory_bytes=...)` (default 10 GiB, `FDMT` takes the
same argument) bounds what the instance allocates where it runs
(`memory_usage.budgeted`, not the caller's input or output): everything on
the CPU, device memory on a GPU. On a GPU the host plan copies, host result
storage and the per-call mask noise model live in host memory; they are
reported in `memory_usage.host` (included in `total`) but not budgeted. The
boxcar's own buffers are fixed by the plan and config: each searched row
keeps `max_boxcar` carried samples (float32, or uint32 for packed input
when every tree value fits 32 bits) and its pending peaks. The FDMT takes
a fast path that fits in the rest (rule 3 above):

- **The tree path** when its streaming history fits: every preset except
  `parkes_uwl`. Exception: on the CPU, packed input with blocks longer
  than 2048 samples takes the skewed-window path first where it applies
  (2.7-4x faster on 8-bit UWL subband searches, as fast or faster on small
  plans; see
  [memory budget](skewed_window_fdmt.md#memory-budget-and-chunk-length)).
- **The skewed-window path** otherwise (a wide band at fine time
  resolution with a long DM sweep: 6656 channels at 64 µs to DM 2000 keeps
  10–14 GB of tree history). It keeps only the raw input of each channel's
  own dispersion reach and recomputes the upper levels over short per-row
  windows ([](skewed_window_fdmt.md)), at the longest automatic chunk
  (4096 down to 256 samples) that fits; a longer chunk is faster.
- **GPU subband searches** (`subband_depth > 0`) take the first that fits
  of a short list: the plan's L2-cache chunk on the direct path, the direct
  path at chunks 4096 down to `max_boxcar / 8`, the skewed path at 4096
  down to 256, then shorter direct chunks
  ([details](skewed_window_fdmt.md#memory-budget-and-chunk-length)).
- Construction raises `ValueError` (`std::invalid_argument`) when nothing
  fits.

On a GPU, host-memory input is copied to the device one chunk at a time,
into a `beams x nchans x chunk` buffer reserved at construction and
counted in `max_memory_bytes` (Parkes UWL, 8-bit, chunk 4096: 27 MiB;
float32: 109 MiB). Host calls allocate nothing, and the buffer is held
whether or not host input is used.

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
storage and the host-input staging buffer. It excludes caller input and
Python's owned output copies.

:::{dropdown} How the scorer works on CPU and GPU
:icon: info

Both backends score rows tile by tile and run full peak bookkeeping only on
tiles with a threshold crossing or an open excursion; on noise, almost every
tile stops after Stage 1.

- CPU (`AUTO` $\to$ `KADANE`): batches of rows are transposed in SIMD
  registers (`float32`, or exact clamped `int32` for packed input when
  bounds fit) across 256-sample tiles and scanned by the multi-bucket Kadane
  gate (with `LagWindow` tail tracking for `{1, 2, 1}` trapezoids). Quiet
  tiles never build a Stage-2 prefix array; only rows whose gate fires or
  that carry an open excursion build lazy `double` (`uint64` for packed
  input) running prefixes and run exact `score_tile` bookkeeping. Explicit
  `PREFIX` / `NONE` instead slide `double` / `uint64` prefixes across 1024–4096
  sample tiles.
- GPU (`AUTO` $\to$ `PREFIX`): one thread block per (beam, DM) row. The prefix
  of each 1024-sample tile is a warp-shuffle scan into a float32 ring in
  shared memory (a power of two of at least `max_boxcar + 1024` entries),
  rebased after every tile so values stay near the magnitude of one tile,
  with sparse range-minimum screening on dense banks. Width and threshold
  tables are staged in shared memory, and tiles with a crossing or an open
  excursion run the exact ordered run summaries. Explicit `KADANE` runs a
  warp/block tropical-monoid Kadane gate before Stage-2 prefix scoring, and
  `KADANE_SURROGATE` runs multi-strip Kadane excursion detection with local
  replay.
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
| `meerkat_lband` (4057 trials), GPU | 25.9 ms | 27.8 ms (13 widths) | 29.4 ms (163 widths) |
| `cryopaf_lband` (1941 trials), GPU | 11.6 ms | 12.8 ms (14 widths) | 13.6 ms (219 widths) |
| `cryopaf_lband`, subband depth 1, GPU | – | 15.1 ms | 18.0 ms |
| `cryopaf_low` (1545 trials), GPU | 21.8 ms | 22.5 ms (13 widths) | 23.2 ms (195 widths) |
| `meerkat_lband`, CPU | 362 ms | 515 ms | 559 ms |
| `cryopaf_lband`, subband depth 1, CPU | 139 ms | 486 ms | 545 ms |
| `cryopaf_low`, CPU | 252 ms | 315 ms | 339 ms |

On the GPU the root merge the search saves costs about as much as scoring a
dozen widths, so the full-band search adds 3–10%; a depth-1 subband search
on `cryopaf_lband` adds a fifth (wtsp 1.5) to a third (wtsp 1.0) more. On the CPU the scorer is a separate
pass over each chunk's searched rows (1.3–3.5× the transform), still well
faster than real time on 8 threads. Thanks to the width screen, scoring
every integer width up to 100 ms costs about the same as a dozen widths on
both backends. `working_bytes` in the results is the allocation plus the
input block, not process RSS.

The `parkes_uwl` search (6656 channels at 64 µs, 4749 trials, 2.1 s
blocks, subband depth 4: 218k searched rows, wtsp 1.5) runs within the
10 GiB budget on the skewed-window path:

| backend | input | chunk | time per block | real-time factor | working memory | peak |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| GPU | uint8 | 4096 | 1.17 s | 1.80× | 9.8 GiB | 9.6 GiB |
| GPU | float32 | 1024 | 2.34 s | 0.90× | 9.6 GiB | 9.5 GiB |
| CPU, 8 threads | uint8 | 4096 | 15.5 s | 0.14× | 6.6 GiB | 6.7 GiB |
| CPU, 8 threads | float32 | 2048 | 22.0 s | 0.10× | 9.7 GiB | 9.8 GiB |

*working memory* is the library's accounting plus the input block; *peak* is
the measured device memory (GPU) or process resident set (CPU) in the same
run. The GPU peak can be below the working memory because the accounting
also counts host-side tables and the host candidate buffer. Float32 input on
the GPU fits the 10 GiB budget only at 1024-sample chunks, which keeps it
just below real time (0.90×); the 8-bit tree is narrower and fits 4096.

The full band alone takes 0.36 s on the GPU (5.8× real time, 2.1–2.7 GiB).
Searching at a coarser resolution is much cheaper: at 256 µs the GPU runs
depth 4 at 39–41× real time and 8 CPU threads keep up (1.11–1.31×), and at
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
  `memory_usage.budgeted`, as on `FDMTBoxcar` (device memory on a GPU). Over
  it the overlap-save segments are shortened (each try halves the transform
  length; `chunk_nsamps` reports the segment) until the instance fits, and
  construction raises `ValueError` with the memory at the shortest segment
  when nothing does. Shorter segments change the rows as blocking does (see
  Streaming). On a GPU the device copy of a host-memory block is reserved
  inside this budget at construction, so host calls allocate nothing. On `parkes_uwl` the CPU instance needs 19.45
  GiB even at the shortest segment, so the default budget raises; pass a
  larger `max_memory_bytes`.
- **Limits.** Float32 input only and plan `subband_depth=0`: subband rows exist
  only in the Fourier domain and would each need an extra inverse transform.

```{seealso}
- [Noise calibration](noise_calibration.md): the exact per-trial sigma.
- [Benchmarks: Parkes UWL search](../benchmarks.md#parkes-uwl-search).
- [Streaming and history](streaming_and_history.md).
```
