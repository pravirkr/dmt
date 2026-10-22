# Fused FDMT and boxcar search

`FDMTBoxcar` searches normalized waterfalls and returns raw detections, without
materializing the full-band DM–time output. Use it when that output would only
be consumed by boxcar search. Existing `FDMT` and other algorithms retain their
interfaces and behavior.

## Input and statistical contract

Input is finite, contiguous float32 in `(nbeams, nchans, nsamps)` order, with
channels ordered from low to high frequency. Python also accepts
`(nchans, nsamps)` for one beam. Supply non-overlapping consecutive blocks.
Only valid-mode plans are supported. Delay grids may be regular, signed, or
custom; physical DM grids are preserved at the root.

Prepare the waterfall upstream: remove the baseline and normalize to zero
mean and a known common per-channel noise standard deviation (`noise_sigma`,
default 1). Analytical S/N assumes independent channel/time input noise before
the FDMT's modeled filtering. Normalization alone does not establish this
independence. Channel masking, common-mode subtraction, temporal filtering and
residual correlations can change the variance and require upstream calibration.
The search does not estimate a baseline or RMS from dedispersed rows.

Sigma accounts for the actual FDMT leaf box means and temporal covariance,
including each subband's tree lineage. Existing
`FDMTPlan.get_effective_sigma_grid(width)` already gives sigma for the raw
full-band sum: **do not multiply it by the number of channels**.

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

Packed arrays use `uint8` bytes, including for 16-bit samples. Each channel
row occupies `ceil(nsamps * input_nbits / 8)` bytes, independently padded;
sub-byte samples are LSB-first and 16-bit samples are little-endian. Native
little-endian contiguous `uint16` arrays can be passed as `.view(np.uint8)`.
The C++ overloads accept `std::span<const uint8_t>` or
`DeviceSpan<const uint8_t>` with the same layout on CPU and GPU backends.

The packed search keeps exact dyadic FDMT weights, sums, merge histories and
boxcar prefixes in uint64. It does not add the level-0 rounding used by the
existing packed `FDMT` API. This preserves a linear expected baseline and
avoids a distribution-dependent rounding bias when only input mean and sigma
are known. Its numerical reference is float-input FDMT on centered codes,
subject to float precision. Existing packed FDMT behavior is unchanged.
Thresholds include the expected baseline and are rounded up to inclusive
integer cutoffs. Per-width peak comparisons remain integer until emission;
across-width reduction evaluates normalized S/N at threshold crossings because
different widths have different baselines and noise scales.

Accumulator bounds are checked at construction, including prefix workspace.
The root merge is fused into scoring. Lower-level fusion is currently disabled
for packed searches, and wide state/history can cost more memory than float32;
this version makes no packed-throughput improvement claim. GPU integer scoring
uses less than 48 KiB shared memory per block and needs no opt-in shared-memory
limit. GPU compilation and runtime parity still require validation on a GPU
machine.

## Python

```python
from dmtlib import FDMTPlan, FDMTBoxcar, BoxcarConfig, PeakFindingMode

plan = FDMTPlan(700.0, 1200.0, 256, 4096, 0.001, 512)
config = BoxcarConfig(
    max_boxcar=256,
    wtsp=1.5,
    detect_thresh=6.0,
    subband_levels=2,
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

The C++ and Python constructors accept the search settings directly; the
fields remain editable before constructing the search. Validation occurs
when `FDMTBoxcar` combines the settings with its plan.

Choose `backend="cuda", device=0` in a CUDA build. Python GPU calls stage
input and block until candidates are available on the host. No full root
plane is copied to the host. `config` and `search_plans` properties return
copies; modifying them does not reconfigure the constructed engine.

## C++ and device pipelines

```cpp
#include <dmt/dmt.hpp>

dmt::plans::FDMTPlan plan(700.0F, 1200.0F, 256, 4096, 0.001F, 512);
dmt::BoxcarConfig config(256);
dmt::algorithms::FDMTBoxcar search(plan, config, true, dmt::Exec::cpu(4));
auto result = search.execute(waterfall); // std::span<const float>
// Consume result.candidates before the next result-changing operation.
auto last = search.flush();
```

On GPU backends, `execute(DeviceSpan<const float>, Stream)` and
`flush_device(Stream)` return asynchronous device views. The candidate span
has capacity `max_candidates`, not the emitted length. Its one-element
`counts` span contains `CandidateCounts`; consume the first `emitted_count`
candidates after stream completion. `copy_result(stream)` waits and returns a
host view of the latest result. Device and host views are engine-owned and
invalidated by subsequent result-changing operations. Python arrays own copies.

Order operations on one stream. Synchronize before changing streams on an
instance; instances are not safe for concurrent calls. CPU device calls reject
the operation. Host execute/flush calls block; GPU reset_history() is ordered
asynchronously on its supplied stream. GPU input staging is allocated lazily and
reused. Warm CPU execution and flush allocate no working memory.

## Widths, excursions and boundaries

Widths start at 1; each next width is
`max(previous + 1, ceil(previous * wtsp))`, stopping at `max_boxcar`.
`wtsp=2` gives powers of two, `1.5` a denser geometric grid, and `1` every
integer. Widths 1–4096 are supported on both backends, without a 32-filter cap.
Every trial is searched at the original time resolution.

- `CONTIGUOUS_PEAK_PER_WIDTH`: one peak per contiguous threshold excursion
  of each DM/width response.
- `CONTIGUOUS_PEAK_ACROSS_WIDTHS`: take the highest normalized S/N across
  widths at each endpoint, then reduce contiguous excursions.
- `ALL_ABOVE_THRESHOLD`: emit every above-threshold DM/width/endpoint.

Threshold comparisons use `>=`. Equal peaks select the earliest endpoint,
then narrower width. The code performs no gap bridging, across-DM clustering,
subband deduplication or rate limiting.

`peak_sample`, `begin_sample` and `end_sample` are absolute uint64 stream
indices of **trailing-window endpoints**. End is inclusive. For width `W`,
the filter at endpoint `t` covers `[t-W+1, t]`; its center is
`t - (W-1)/2.0`. Excursion bounds describe response endpoints, not the union
of all input windows in the excursion. Across-width excursions retain the
winning peak's width.

Cold windows are skipped by default until their complete FDMT dependency
support and boxcar lookback are available. This can span multiple blocks.
`skip_cold_prefix=False` searches the zero-padded cold response using the
stationary sigma table. Raw sample carry and pending peaks survive internal
chunks and input-block boundaries. A returned peak may therefore refer to an
earlier block. Call `flush()` once at end of stream to emit remaining peaks;
repeated flushes return no further candidates. Flush adds no padded samples
and permits subsequent input to continue, with excursions split at that flush.
`reset_history()` discards all histories and pending peaks and restarts sample 0.

## Subbands, overflow and memory

`subband_levels=0` searches the full band; 1 also searches the preceding level;
2 also searches the level before that. On a balanced tree these are halves
and quarters; odd channel counts follow the actual tree, including copy
nodes. Requests deeper than the tree are rejected. Candidate channel extents
are half-open input indices, and frequency extents are physical observed
edges. All selected planes share the candidate buffer.

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

CPU scoring uses tile-local double prefixes. GPU scoring uses a rebased
float prefix ring with double scan intermediates and ordered parallel run
summaries. Candidate semantics match across backends; S/N is compared with
tolerances, not promised bit-identical. Arithmetic near an exact threshold
or near-equal peak can affect selection.

## Validation and performance

CPU tests compare against materialized FDMT and independent direct float64
window sums, impulse-response energy, Gaussian noise, streaming boundaries,
overflow and allocation checks. GPU tests are tagged `[gpu][boxcar]`.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDMT_GPU=CUDA -DDMT_BUILD_TESTING=ON -DDMT_BUILD_PYTHON=OFF
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
compute-sanitizer --tool memcheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
compute-sanitizer --tool racecheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
compute-sanitizer --tool synccheck build-cuda/tests/cpp/dmt_tests '[gpu][boxcar]'
```

The new CUDA implementation has not been compiled or run on the development
Mac; GPU execution and performance require validation on a GPU machine.

Build benchmarks with `DMT_BUILD_BENCHMARKS=ON`, then run:

```bash
build/bench/dmt_bench --benchmark_filter='^BM_fdmt_boxcar_(fused|separate)/' \
  --benchmark_min_time=0.2s
```

These noise-only CPU benchmarks use the same scorer in both paths and disable
lower-tree fusion to isolate root fusion. They cover 32/4096 maximum widths,
full-band/subband searching, and 1/4 threads. `working_bytes` is the algorithm
allocation estimate, not process RSS. Root allocation and traffic savings
are reliable; speedup depends on the plan, width count, backend and candidate
rate. Further fusion of subband searching with subsequent merges is deferred.

A local Apple M1 Pro Release comparison (128 channels, 4096 samples, delays 0–256,
maximum width 32, one beam/thread, full band only, lower fusion disabled;
three repetitions of at least 0.5 seconds) measured median times of 13.06 ms
fused and 13.02 ms separated. Reported working memory was 21.96 MiB and
25.97 MiB respectively. This sample shows the expected 4.02 MiB root-plane
saving, with no measurable CPU speedup. It does not predict GPU performance.
