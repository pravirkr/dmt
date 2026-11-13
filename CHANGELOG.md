# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **DM-grid generators in C++** (`dmt/common/dm_grids.hpp`:
  `plans::generate_dm_grid`, `plans::generate_dt_grid`; Python
  `dmtlib.generate_dm_grid`, `dmtlib.generate_dt_grid`, `DMGridMethod`), so
  C++ pipelines get the same grids as `dmtlib.generate_optimal_dm_grid` /
  `generate_optimal_dt_grid`, which now wrap them (same trials). The default
  S/N-loss rule gains `chan_dm`, for data coherently dedispersed to a DM
  (trials get dense around it), and takes `tol` as its knob like the Levin
  rules (`max_snr_loss = 1 - 1/tol`). The Levin rules are unchanged. A new
  pipeline-guide page explains the rules and their limits on very wide
  bands.
- **`OverflowPolicy::kKeepBrightest`** (`BoxcarConfig::overflow_policy`;
  Python `OverflowPolicy.KEEP_BRIGHTEST`, `BoxcarConfig(overflow_policy=...)`)
  for `FDMTBoxcar` and `FDMTFFTBoxcar`, CPU and GPU: when more candidates
  cross the threshold than fit, each DM row keeps exactly its
  `max_cands_per_dm` brightest candidates of the call and the call keeps the
  `max_candidates` brightest of those, with a deterministic tie-break. An
  early RFI storm no longer hides a bright burst later in the block. The
  per-row stores count in the memory budget, so `max_cands_per_dm` must be
  positive and small. The default `kDropNew` (first come) is unchanged.
- Pipeline guide section on red noise and wide boxcars: slow drifts inflate
  the S/N of wide boxes and the search cannot correct for them, so detrend
  each channel upstream and keep `max_boxcar` at about 100 ms or less.
  `kKeepBrightest` does not help against red-noise candidates.

### Changed

- **`max_boxcar` above 4096.** Boxes up to 2^20 samples are accepted
  (before: at most 4096). Up to 4096 the search is tuned as before. Wider
  boxes are searched exactly but with no speed promise: on a GPU their
  prefix rings live in device memory, which is also the fallback when a
  narrower box doesn't fit a small GPU's shared memory.

- **GPU memory budget.** On a GPU, `max_memory_bytes` of `FDMTBoxcar` and
  `FDMTFFTBoxcar` now bounds device memory only. Host-side allocations (host
  plan copies, host result storage, the per-call kill-mask noise model) are
  reported in the new `FDMTBoxcarMemoryUsage::host` field (Python
  `memory_usage.host`) but not budgeted; `budgeted()` (`memory_usage.budgeted`)
  is the part checked against the budget. `total()` is unchanged. On the CPU
  nothing changes (`host` is 0).

- Tutorials 1-7 are MyST notebooks (`docs/tutorials/*.md`) without stored
  outputs, executed by every docs build against the current `dmtlib`. Their
  committed outputs had gone stale (the CohFDMT tutorial's peak S/N moved
  with the 0.8.0 time reference, since a pulse's sub-sample phase changes
  its peak S/N by up to 25%). The GPU tutorial (8) and the landing-page
  example stay `.ipynb` with stored outputs. The `docs` extra adds
  `ipykernel` and `jupyter-cache`; `NB_EXECUTION_MODE=off` builds without
  running them.

### Fixed

- `dmtlib.generate_optimal_dm_grid(method="levin" or "levin_piecewise")`
  and `generate_optimal_dt_grid` with those methods passed `pulse_width`
  (in samples) on as seconds: the default of one sample became a 1 s pulse,
  and the grid nearly empty (3 trials for DM 0-2000 on Parkes UWL).

- `FDMTBoxcar` on the GPU no longer rejects wide boxes that don't fit a
  block's shared memory. It now drops the width screen and then uses
  smaller blocks, with the same candidates. Packed input with
  `max_boxcar` 4096 and dense widths (`wtsp` near 1) failed this way even
  on 99 KB GPUs (L40); GPUs with 64 KB per block fit wider boxes than
  before. Configurations that fitted are unchanged.

- `FDMTBoxcar` on the GPU applies kill masks (constructor and per-call)
  inside the FDMT, which reads killed channels as zero from one byte per beam
  and channel. It no longer copies each block to a staging buffer to zero
  them: device-memory input under a per-call mask needed a block-sized
  device buffer (6.5 GiB for 6656 channels × 262144 samples), which could
  push a configuration over the budget. Candidates are bit-identical.

- Per-call kill masks: a call that throws after the engine took its mask
  (an allocation or launch failure during the search) no longer leaves that
  mask's noise tables in the stream's history. On paths with a search
  latency (skewed path, GPU windowed top) they replaced the previous
  block's tables for lagged rows on the next call.
- Per-call kill masks: an unchanged mask reuses its kill runs instead of
  rebuilding them every call.

- GCC 12 Release builds with AVX-512 no longer print hundreds of false
  `-Wmaybe-uninitialized` warnings from its own `avx512fintrin.h`
  (`_mm512_undefined_epi32`, GCC PR 105593). The warning is silenced in
  that header only.

- `FDMTBoxcar` on the GPU no longer rejects plans whose full-block root
  plane (`ndt × nsamps`) exceeds 2^31 elements. The search consumes the
  root chunk by chunk and never allocates that plane, so the 32-bit output
  check only applies to the root-writing `FDMT` calls, which still enforce
  it. Fine DM grids on long blocks (e.g. Parkes UWL, 9466 trials ×
  262144 samples) now construct.

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
