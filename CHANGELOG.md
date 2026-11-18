# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`FDMTBoxcar::estimate_minimum_memory(...)`** (C++ and Python): the
  estimate at the smallest `max_memory_bytes` the search can be built with
  (same arguments as `estimate_memory()`, no budget), and
  `FDMTBoxcarMemoryEstimate::max_memory_bytes`, the budget an estimate was
  made with (for the minimum: the smallest that builds; one byte less
  raises).
- **`dmt::dispersion_delay(dm, f_lo_mhz, f_hi_mhz)`** (C++,
  `dmt/common/types.hpp`): the dispersion delay in seconds, as Python's
  `dmtlib.dispersion_delay` (TEMPO2 constant 1 / 2.41e-4; `f_hi_mhz` =
  infinity for the delay from infinite frequency).
- Documented time conventions of `RawCandidate`: `peak_sample` is the
  trailing end of the window at the arrival at the lower edge of the
  candidate's own plane (`f_low_mhz`; plan frequencies are channel edges);
  a `chan_dm` plan's level-0 causal delay is subtracted on every plane. And
  `FDMTBoxcar`'s input contract: no preprocessing; each channel zero mean,
  common sigma, white in time over at least `max_boxcar` samples.
- **`BoxcarConfig::per_call_masks`** (Python `per_call_masks=`, default
  false): declares that an `FDMTBoxcar` stream will pass per-call kill
  masks. Their requirements (float32 input) are checked and their memory
  reserved at construction and in `estimate_memory()`, before the
  transform's path is chosen, so a per-call mask no longer fails mid-run
  for those reasons, and `get_memory_usage()` stays at the estimate.
  Candidates are unchanged. `FDMTFFTBoxcar` rejects it.
- **Per-plane candidate counts:** `CandidateResultView::plane_counts`
  (Python `plane_counts`, a structured array) and
  `CandidateDeviceResultView::plane_counts` give each searched plane's
  `emitted_count`, `total_detected`, `dropped_global` and `dropped_per_dm`
  for the call, in `get_search_plans()` order (`FDMTBoxcar`,
  `FDMTFFTBoxcar`); they sum to the call's counters, so a pipeline can tell
  which subband lost candidates.
- **`BoxcarConfig::max_candidates_per_plane`** (Python
  `max_candidates_per_plane=`, default 0: off): each searched plane keeps at
  most this many candidates per call, its first ones under `kDropNew` and
  exactly its brightest under `kKeepBrightest`, so a plane flooded with
  detections (RFI in one subband) cannot crowd the others out of
  `max_candidates`. The planes' shares must fit `max_candidates`.
- **`FDMTBoxcarMemoryEstimate::search_path` and `search_latency`** (C++ and
  Python): the transform's path and the search latency of the instance the
  estimate describes.

- **`FDMT::estimate_memory(...)`** (C++ and Python `FDMT.estimate_memory`),
  one overload per constructor with the same arguments: the memory the
  instance would allocate (`usage`, equal to its `get_memory_usage()`), its
  `chunk_nsamps`, and on a GPU the `host_staging` bytes the first host-memory
  call adds, all computed without allocating. A GPU backend reads the
  device's limits but allocates no device memory.
- **`FDMTFFT::estimate_memory(...)`** (one overload per constructor) and
  **`FDMTFFTBoxcar::estimate_memory(...)`**, C++ and Python: the same
  estimate for the Fourier-domain engines (`chunk_nsamps` is the
  overlap-save segment). Python `FDMTFFT` gains `memory_usage`.
- **`CohFDMT::estimate_memory(config, exec)`** (C++ and Python
  `CohFDMT.estimate_memory`): the memory a `CohFDMT` built with these
  arguments would allocate, equal to its `get_memory_usage()`, computed
  without allocating.
- **`DDMTFFT::estimate_memory(plan, nsamps, exec, nbeams, options)`** (C++
  and Python `DDMTFFT.estimate_memory`): the memory a `DDMTFFT` holds once a
  stream of calls of `nsamps` samples is warm (`usage`), plus the staging
  that host-memory, time-major and packed device-memory calls add, computed
  without allocating. **`DDMTFFT::get_memory_usage()`** (Python
  `memory_usage`) reports what an instance holds now.
- **`DDMT::estimate_memory(plan, nsamps, exec, nbeams, gulp_size)`** and
  **`SDMT::estimate_memory(...)`** (C++ and Python): the memory an instance
  making calls of `nsamps` samples holds (`usage`), plus on a GPU the device
  and pinned staging that host-memory and time-major calls add, computed
  without allocating. **`DDMT::get_memory_usage()`** (Python
  `memory_usage`) reports what an instance holds now.
- **`FDMTBoxcar::estimate_memory(...)`** (C++ and Python
  `FDMTBoxcar.estimate_memory`), with the constructor's arguments: the same
  `usage`, `chunk_nsamps` and `host_staging` (the input staging block of the
  first host-memory call on a GPU), computed without allocating.
- **`FDMTPlan` `chan_dm`** (C++ and Python, default 0) for coherently
  dedispersed channels: level-0 boxes cover the residual intra-channel sweep
  at the trial DM, centred on each channel's arrival, in `FDMT`,
  `FDMTBoxcar`, `FDMTFFT` and `FDMTFFTBoxcar` (CPU and GPU, float and packed
  input), with the noise model to match. FDMT output rows lag by the new
  `get_level0_causal_delay()`, which `FDMTBoxcar` and `FDMTFFTBoxcar`
  subtract from candidate times and `get_candidate_frontier()` (clamped at 0,
  while `get_searched_nsamps()` stays in FDMT output samples);
  `get_level0_boxes()` lists the boxes. Valid mode only. With `chan_dm` 0
  every output is unchanged.
- **Subband DM grids**: `FDMTPlan(..., subband_max_snr_loss=...)` (off at 0)
  gives staggered subbands their own `generate_dm_grid` trials and has
  `FDMTBoxcar` score only the tree-subband rows nearest such a grid, so deep
  subband searches score 2-4x fewer rows (the full band is unchanged).
  `BoxcarPlanePlan.coord_indices`, `FDMTCoordGrid::search_rows`, and
  `FDMTSubDTGrid.search_rows` list the scored rows. The worst pulses can lose
  up to about 1.5x the requested loss.
- **DM-grid generators** (C++ `dmt/common/dm_grids.hpp`, Python `dmtlib`):
  `generate_dm_grid` spaces trials so a pulse halfway between two keeps
  `1 - max_snr_loss` (default 0.05) of its S/N after the boxcar search,
  modelled over every channel, and takes `chan_dm`, `kill_mask`, `tol` and
  `piecewise` (uniform segments for `DDMTFFT`). `generate_levin_dm_grid`
  returns the Levin grids of `DDMTPlan` unchanged.
- **`OverflowPolicy::kKeepBrightest`** for `FDMTBoxcar` and `FDMTFFTBoxcar`
  (CPU and GPU): on overflow each DM row keeps its `max_cands_per_dm`
  brightest candidates, and the call keeps the `max_candidates` brightest of
  those, deterministically. Set a small positive `max_cands_per_dm`; the
  stores count in the memory budget. `kDropNew` stays the default.
- **`BoxcarConfig::screen_mode` / `dmtlib.BoxcarScreenMode`** for `FDMTBoxcar`
  and `FDMTFFTBoxcar` (CPU and GPU): `kAuto` (`AUTO`, default), `kKadane`
  (`KADANE`, lossless multi-bucket Kadane tile gate + exact Stage-2 scoring),
  `kPrefix` (`PREFIX`, prefix-sum / sparse range-minimum screen), `kNone`
  (`NONE`, unscreened exact scoring), and `kKadaneSurrogate`
  (`KADANE_SURROGATE`, approximate multi-bias Kadane excursion detector with
  local replay, `max_boxcar` up to 4096).

### Changed

- **`FDMTBoxcar` per-call kill masks work with any block length** (they
  needed blocks of at least the largest row support plus `max_boxcar - 1`
  samples, and on a GPU a search latency of at most one block). Windows
  that reach back before their block use the variance of the history mask:
  the block's channels and every channel live within that reach (one
  timestamp per channel). When blocks are shorter than the search latency
  needs, the oldest kept block stands for the older ones with one
  conservative catch-all table, so the masks' memory stays fixed whatever
  the block length. Results with blocks at least that long are unchanged.
- `FDMT`, `FDMTBoxcar`: a budget no longer fails where a smaller path
  fits. The tree path (CPU) or unskewed direct levels (GPU) were chosen
  when the budget looked large enough and then turned out over it, so some
  budgets above a working one failed (Parkes UWL subband depth 4: 6.5-7.2
  GiB on the GPU while 3.6 GiB worked). The smaller path is now taken
  instead; budgets that worked choose as before.
- An `FDMTBoxcar` over its budget now reports what the whole search needs
  (it reported the FDMT's share). Python: still `ValueError`.
- `FDMTBoxcar::drain()` (Python `drain`) no longer builds a zero block of
  the whole input in host memory (and, on a GPU, a device staging block)
  to search the samples behind the search latency: it searches them without
  any input block, so memory stays at `get_memory_usage()` and its estimate
  through the end of the stream. Candidates are unchanged. On a GPU, a
  subband search (plan `subband_depth` > 0) now reserves one byte per beam
  and channel for this in `get_memory_usage().staging` (and in
  `estimate_memory()`). `FDMTFFTBoxcar::drain()` is unchanged.
- `FDMTBoxcar` and `FDMTFFTBoxcar` construction and `estimate_memory()` are
  much faster on wide bands: the per-row noise tables and warm-up are built
  once per FDMT tree node instead of tracing every row's subtree. Parkes
  UWL (6656 channels, 4039 trials, GPU): 1.6 s to 0.55 s at subband depth
  0, 3.9 s to 0.35 s at depth 2. The tables are bitwise unchanged.
- `FDMT` construction no longer allocates while it chooses its path and
  chunk: on a GPU the budget search sizes candidate engines without
  allocating them (it could briefly hold more than `max_memory_bytes`), and
  on the CPU the budget is checked before allocating.
- `FDMTBoxcar` construction likewise sizes the engines it tries for a chunk
  and allocates only the one it keeps; on the CPU the budget is checked
  before allocating.
- GPU `FDMTFFT` no longer shortens its overlap-save segments to fit the
  device's free memory: memory follows the arguments alone, so the same
  configuration behaves the same on any device. One that does not fit fails
  at construction with a `std::bad_alloc` (Python `MemoryError`) saying
  what it needs and what was free before it allocated (use shorter blocks
  or fewer beams; `estimate_memory` gives the requirement); GPU `FDMT`,
  `FDMTBoxcar`, `FDMTFFTBoxcar` and `CohFDMT` allocation failures carry the
  same message (still `std::bad_alloc`).
- A GPU device or pinned-host allocation that runs out of memory now
  raises `std::bad_alloc` (Python `MemoryError`, was `RuntimeError`) with
  the size that failed, for every engine (`DDMTFFT` included).
- `CohFDMT::get_memory_usage()` now counts every buffer the engine owns:
  the CPU per-channel power partials, and on a GPU the remaining tables and
  the device copy of a host-memory call's result (in `workspace`, which
  was reported as `output`; `output` is now the caller's result size on
  both backends).
- GPU `DDMTFFT` and `CohFDMT` likewise no longer size themselves from the
  device's free memory: `DDMTFFT` keeps the CPU engine's segment length
  (same segments on both backends), and the `CohFDMT` cuFFT path (channel
  FFT lengths the fused kernel can't take) uses a fixed 1 GiB work buffer.
  `FDMTFFT` and `FDMTFFTBoxcar` construction sizes the engines it tries and
  allocates only the one it keeps.
- The memory usage and estimate structs (`FDMTMemoryUsage`,
  `DDMTMemoryUsage`, `DDMTFFTMemoryUsage`, `CohFDMTMemoryUsage` and their
  estimates) start at zero when default-constructed.
- CPU `FDMTFFT` and `FDMTFFTBoxcar` count the twiddle table they hold (8
  bytes per transform sample) in `get_memory_usage().plan`.
- GPU `FDMTBoxcar` with a kill mask allocates the transform's per-row flags
  (one byte per beam and channel) at construction instead of on the first
  call, so `get_memory_usage()` includes them from the start.
- **CPU boxcar default Stage-1 screen (`BoxcarScreenMode::kAuto`)**: CPU
  `FDMTBoxcar` and `FDMTFFTBoxcar` now default to `kKadane`, a lossless
  multi-bucket SIMD Kadane tile gate with exact `kPrefix` candidate parity
  across all `wtsp` spacings and `{1, 2, 1}` trapezoids. GPU `kAuto` keeps
  `kPrefix`.
- `max_boxcar` may exceed 4096 (up to 2^20 samples, still at most 4096
  widths, except `kKadaneSurrogate` which is capped at 4096). Boxes wider than
  4096 are exact but not tuned for speed.
- On a GPU, `max_memory_bytes` of `FDMTBoxcar` and `FDMTFFTBoxcar` bounds
  device memory only. Host allocations are reported in the new
  `memory_usage.host`, and `memory_usage.budgeted` is the part checked.
- Tutorials 1-7 are MyST notebooks, executed by the docs build. The `docs`
  extra adds `ipykernel` and `jupyter-cache`.
- `dmtlib.calculate_snr_loss` uses the model of `generate_dm_grid` (now in
  C++ as `plans::calculate_snr_loss`). `pulse_width` is in seconds (default
  0), and it, `chan_dm` and the new `kill_mask` are keyword-only.

### Removed

- `dmtlib.generate_optimal_dm_grid` and `generate_optimal_dt_grid`. Use
  `generate_dm_grid` or `generate_levin_dm_grid`, and pass the DMs to
  `FDMTPlan(..., dm_grid=...)` for delay grids.

### Fixed

- `FDMTFFTBoxcar` on a GPU no longer counts its host result storage
  against `max_memory_bytes` when checking whether the boxcar buffers alone
  fit, so budgets that fit the device memory build as documented.
- `FDMTBoxcar` on the GPU:
  - Wide boxes that don't fit shared memory no longer throw. The search
    falls back to smaller blocks or a device-memory ring, with the same
    candidates.
  - Kill masks are applied inside the FDMT, without a block-sized staging
    copy. Candidates are bit-identical.
  - Plans whose root plane exceeds 2^31 elements now construct (e.g. Parkes
    UWL with fine DM grids).
- A per-call kill mask whose call throws no longer leaves its noise tables
  in the stream history.
- The `FDMT` docs no longer say that `execute()` never allocates: on a GPU
  the first host-memory call allocates device copies of the input block and
  output plane outside `max_memory_bytes`, as `FDMTBoxcar` and
  `FDMTFFTBoxcar` already document for the input block.

## [0.8.0] - 2026-11-11

- **Staggered subband search.** `FDMTPlan(..., subband_depth=d)` adds, at each
  depth `k = 1..d`, the `2**k - 1` subbands of the same width centred on the
  boundaries between that depth's `2**k` tree subbands. Each costs one extra
  merge of two existing nodes. `FDMTBoxcar` searches every tree and staggered
  subband to depth `d`; candidates gain a `staggered` field. A narrowband
  burst on a subband boundary is no longer split between aligned subbands.
- Every `FDMTPlan` variance and sigma method takes an optional kill mask
  (C++: a trailing `std::span<const uint8_t>`; Python: keyword-only
  `kill_mask=`) and leaves killed channels out.
- **`f_ref` for DDMT, SDMT and DDMT-FFT** (plans and engines, C++ and
  Python, and `dmtlib.dedisperse`): the reference frequency of the output
  times, either `f_min` (default) or `f_max`. Passing `f_max` gives
  PRESTO/SIGPROC-style series: output sample `t` is input sample `t` at `f_max`
  for every DM, and the incomplete samples are at the end. Only the delay
  table changes. Any other value raises `std::invalid_argument`
  (`ValueError` in Python). The FDMT engines stay referenced to `f_min`.
- **Memory budget.** `FDMT` and `FDMTBoxcar` take `max_memory_bytes`
  (default `kFDMTDefaultMaxMemory`, 10 GiB; Python keyword-only), a bound on
  everything the instance allocates. Within it the fastest streaming path
  is taken; a plan whose tree history does not fit runs the skewed-window
  path at the longest automatic chunk that fits; construction raises
  `std::invalid_argument` (`ValueError`) when nothing fits.
- **Skewed-window streaming FDMT** (`lib/dmt/fdmt_skew.hpp`, CUDA skewed
  direct levels): wide bands at fine time resolution (Parkes UWL, 6656
  channels at 64 µs to DM 2000) stream without their 10–14 GB tree history.
  FDMT output and FDMTBoxcar candidates are bit-identical to the tree path
  (float32 within rounding), and `save_history()`/`load_history()` save the
  skewed engines' own, much smaller state.
- **Subband search on the skewed path** (`subband_depth > 0`, CPU and GPU):
  every searched row is computed from the full band's windows at a fixed
  per-row lag, with the tree path's candidates and S/N. On `parkes_uwl` at
  depth 4 FDMTBoxcar drops from 17.4 GB to 6.4 GB on the CPU and from 15 GB
  to 9.9 GB on the GPU, within the default budget at every depth to 5 on
  8-bit input.
- Benchmark preset `cryopaf` is renamed `cryopaf_low`; `meerkat_lband` is
  the default preset; `parkes_uwl` searches at subband depth 4. Results
  live in `bench/results/<config>/<machine>/` for every preset
  (`bench/results/cfdmt/` for CohFDMT), and the suite no longer measures
  unplotted points (integer-delay FDMT-FFT, brute-force DDMT-FFT, host-array
  GPU rows). `dmt_bench` and `dmt_bench_mem` keep only developer
  microbenchmarks.
- **Kill mask per block on `FDMTBoxcar`** (float32 input): `execute(waterfall,
  kill_mask)` for host and device waterfalls (Python: `execute(waterfall,
  kill_mask=None)`), ANDed with the constructor mask, so a pipeline that
  flags RFI per block keeps one engine. Within a block the S/N is calibrated
  on that block's active channels. After a mask change, the first `S + W - 1`
  endpoints of each row (`S` the row's dispersion support) read samples
  under both masks and use the noise of the OR of the two: never more false
  alarms, S/N there slightly low.
- **Device arrays in Python (DLPack and CUDA array interface).** On a GPU
  backend, `FDMT`, `FDMTFFT`, `DDMT`, `SDMT`, `DDMTFFT` and `CohFDMT` accept
  PyTorch, CuPy, Numba or any other `__dlpack__` or
  `__cuda_array_interface__` array on the engine's device in `execute()` (and
  `execute_time_major()` for DDMT/SDMT), the FDMT steppers' `reset()`, and
  `save_history()`/`load_history()`.

## [0.7.0] - 2026-11-03

The first release since 0.2.0. It rebuilds `dmt` as a family of
dedispersion engines that share plans, DM grids and conventions, each running
on the CPU and on NVIDIA and AMD GPUs. Versions 0.3 to 0.6 were never released;
their changes are included here.

### Added

- **Engines.** `FDMT`, `FDMTFFT` (FDMT tree in the Fourier domain, fractional
  delays), `DDMT` (direct dedispersion), `SDMT` (direct dedispersion with
  shared partial sums), `DDMTFFT` (Fourier-domain direct dedispersion with a
  NUFFT over DM) and `CohFDMT` (coherent search of baseband voltages).
- **Backends.** One class per engine with a runtime backend: CPU (OpenMP,
  AVX2, AVX-512, NEON), CUDA and HIP.
- **Streaming.** Valid-mode overlap-save history, so consecutive blocks join
  exactly; `save_history` / `load_history`; multi-beam execution (`nbeams`).
- **Input.** Packed 1/2/4/8/16-bit samples, channel-major or time-major, read
  directly; channel kill masks.
- **DM grids.** Custom, sparse and negative delay or DM grids;
  smearing-aware and Levin grids, including a piecewise-uniform Levin grid for
  `DDMTFFT`.
- **Noise.** Exact noise variance of every DM trial from each plan, and for
  FDMT after a boxcar of any width (`get_effective_sigma_grid`).
- **Candidate search.** `FDMTBoxcar` and `FDMTFFTBoxcar` fuse the transform
  with a boxcar search and return candidates without writing the DM–time
  plane.
- **Periodicity use.** `dmtlib.dedisperse` returns complete dedispersed time
  series for any engine, with box smearing off and startup samples removed;
  `dmtlib.dispersion_delay`.
- **Time references.** `plan.f_ref` gives the frequency each engine's output
  times refer to; `startup_samples` on FDMT and FDMT-FFT gives the number of
  incomplete outputs after a cold start.
- **Memory.** Chunked valid-mode FDMT execution (`chunk_nsamps`), so the
  state no longer grows with the block length.
- Benchmark suite (`bench/`) with published results for an Intel Xeon and an
  NVIDIA L40S; noise (`fdmt_noise.py`) and signal-recovery
  (`tone_recovery.py`) analysis scripts.
- Documentation site with getting-started guides, a pipeline guide,
  tutorials and the C++/Python API.

### Changed

- **Breaking:** `FDMTCPU` and `FDMTGPU` are replaced by `FDMT(...,
  backend="cpu" | "cuda" | "hip")`.
- **Breaking:** with box smearing (the default), FDMT averages each channel's
  level-0 box instead of summing it, so the output is in box-mean units.
  Divide by `get_effective_sigma_grid()` for S/N.
- **Breaking:** `DDMT`, `SDMT` and `DDMTFFT` delay each channel from its
  centre and refer output times to the centre of the top channel. Output times
  shift by up to one sample compared with earlier development versions.
- **Breaking:** output buffers hold only the transform,
  `nbeams * plan.dmt_size` floats; working state is owned by the engine.
- Invalid input now raises instead of being used silently: non-finite or
  negative DMs, non-positive frequencies or sampling times, zero beams, and
  buffers whose size does not match the plan.
- Non-binary kill masks are normalised to 0/1, and a plan can no longer be
  modified once an engine has been built from it.

## [0.2.0] - 2024-05-29

- FDMT on the CPU (`FDMTCPU`) and GPU (`FDMTGPU`) with Python bindings.

[Unreleased]: https://github.com/pravirkr/dmt/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/pravirkr/dmt/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/pravirkr/dmt/compare/v0.2.0...v0.7.0
[0.2.0]: https://github.com/pravirkr/dmt/releases/tag/v0.2.0
