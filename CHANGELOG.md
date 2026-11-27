# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.0] - 2026-11-27

### Added

- **Memory estimation:** static `estimate_memory(...)` across all engines (C++
  and Python, no allocation), `FDMTBoxcar::estimate_minimum_memory()`,
  `DDMT` / `DDMTFFT` `get_memory_usage()`, and `pinned` / `host_pinned` fields
  for GPU pinned host staging.
- **DM grids and `chan_dm`:** `generate_dm_grid` and `generate_levin_dm_grid`
  (C++ `dmt/common/dm_grids.hpp`, Python `dmtlib`),
  `FDMTPlan(..., subband_max_snr_loss=...)` for pruned subband grids, and
  `FDMTPlan(..., chan_dm=...)` for coherently dedispersed channels.
- **Boxcar search (`FDMTBoxcar`, `FDMTFFTBoxcar`):** `BoxcarScreenMode`
  (`kAuto`, `kKadane`, `kPrefix`, `kNone`; lossless `kKadane` is now the CPU
  default), `OverflowPolicy::kKeepBrightest`, `max_candidates_per_plane` and
  `plane_counts`, per-call kill masks on packed input, and
  `BoxcarConfig::per_call_masks`.
- **GPU staging threads:** `Exec::cuda(device, nthreads)` /
  `Exec::hip(device, nthreads)` (Python `nthreads=`, default 1, up to 4) caps
  host-output staging workers; pass `nthreads=4` for 0.8.0 host-output speed on
  large planes.
- Keyword-only `out=` on host `FDMTFFT.execute` / `execute_time_major`
  (Python) and `dmt::dispersion_delay()` in `dmt/common/types.hpp`.
- **Short final blocks:** `FDMT::execute_tail()` (valid mode) and
  `FDMTBoxcar::drain(tail, n_tail, ...)` (with optional per-call `kill_mask`)
  take a whole block of which only the first `n_tail` samples are data (host
  or device, float or packed; Python `FDMT.execute_tail(block, n_tail)` and
  `FDMTBoxcar.drain(block, n_tail=..., kill_mask=...)`). Neither the first
  `n_tail` outputs nor any returned candidate depends on the padding; the
  stream then ends.
- **Mirrored baseband:** `BasebandFormat::descending` (subbands stored from
  the top down) and `BasebandFormat::lower_sideband` (inverted subband
  spectra), e.g. GUPPI `CHAN_BW < 0`, read directly by `CohFDMT` on the CPU
  and GPU (Python keyword-only `descending=`, `lower_sideband=`);
  `pack_baseband` writes them.
- **Python device arrays on `FDMTBoxcar` and `FDMTFFTBoxcar`:** `execute()`
  (and `FDMTBoxcar.drain(tail, ...)`) read PyTorch, CuPy, Numba and other
  DLPack / `__cuda_array_interface__` arrays in place, with `kill_mask=` and
  keyword-only `stream=`, and return host candidates.
- **Python parity:** `CohFDMTPlan.fdmt_nsamps`, `fdmt_window_start`,
  `row_offsets` and `channel_taper`; `DDMTPlan(..., kill_mask=)`;
  keyword-only `out=` on host `DDMT` / `SDMT` / `DDMTFFT` `execute` and
  `execute_time_major`; `dmtlib.dedisperse(..., device=)`.
- `CohFDMT::execute<char>` (alongside `int8_t` and `uint8_t`); the
  `dmt::ByteDataType` concept rejects other types at compile time.
- **Docs:** `FDMTBoxcar`, `FDMTFFTBoxcar`, the candidate types, build
  information and the simulation utilities in the C++ API reference;
  `FDMTBoxcar` in both quickstarts; known limits (subband candidate times,
  `CohFDMTPlan.dm_grid_final` layout, packed cold start, 32-bit GPU indices)
  and cross-compiling for login nodes and containers.

### Changed

- **Breaking — strict Python host inputs:** waterfalls, packed arrays,
  in-place buffers and host `load_history` require exact dtype (`float32`,
  `uint8`, `complex64`) and C-contiguous layout, raising `ValueError` instead
  of copying or casting.
- **Breaking — keyword-only Python arguments:** `nbits=` on packed `FDMT` /
  `FDMTFFT` calls, `nsamps=` on packed `DDMT` / `SDMT` / `DDMTFFT` calls, and
  `dmt=` on all `FDMT.reset` / `FDMTFFT.reset` overloads.
- **Breaking — `dmtlib.calculate_snr_loss`:** uses `generate_dm_grid`'s model
  with keyword-only `pulse_width` (seconds), `chan_dm` and `kill_mask`.
- **Deterministic GPU sizing and memory budgets:** engines size from arguments
  alone (fixed 96 MiB GPU `FDMT` cache target and 24 MiB `CohFDMT` chunks),
  check free device memory before allocating, and raise `std::bad_alloc`
  (`MemoryError`) on OOM. Plain float GPU `FDMT` allocates its full
  tree-history ring upfront inside `max_memory_bytes`; GPU `FDMTBoxcar` /
  `FDMTFFTBoxcar` budget device memory only (including host-input staging).
- **Boxcar path selection and masks:** `max_boxcar` supports up to 2^20
  samples; per-call kill masks work on any block length; GPU subband and CPU
  packed searches favour faster skewed chunks where applicable; `drain()` no
  longer allocates a zero input block.
- **Python bindings:** GIL released around host transforms and history
  calls, engine construction, `estimate_memory` / `estimate_minimum_memory`
  and `pack_baseband`; `dmtlib.simulate.generate_pure_frb` wraps C++
  `libdmt.generate_pure_frb` (float `dt`; Numba version removed) and returns
  the dispersed-sample count as an `int`.
- **Breaking — `BoxcarConfig` is an aggregate:** its positional constructors
  are removed; set fields by name (`BoxcarConfig{.max_boxcar = 64,
  .detect_thresh = 7.0F}`). Python is unchanged.
- **Breaking — `FDMTFFTBoxcar` argument order:** the constructor and
  `estimate_memory()` end with `(..., max_memory_bytes, kill_mask)` as on
  `FDMTBoxcar`; in Python `kill_mask` is keyword-only after
  `max_memory_bytes`.
- **Breaking — input shapes in Python:** `FDMT`, `FDMTFFT`, `DDMT`, `SDMT`
  and `DDMTFFT` (host and device, `execute` and `reset`) check every axis:
  a transposed, time-major `(nsamps, nchans)` array with the right element
  count raises `ValueError` instead of being read as channel-major. Wrong
  `ndim`, non-contiguous `CohFDMT` blocks and bad host `out=` buffers raise
  `ValueError` (was `RuntimeError` / `TypeError`).
- **Dispersion constants are `double`:** `kDispConst` (`1 / 2.41e-4`),
  `kDispConstMT`, `kDispConstLK` and `kDispConstSI`; `kDispConstD` is
  removed. Candidate DMs and `CohFDMT` output move at the float rounding
  level.
- **`drain()` keeps the call's quotas:** `max_candidates`,
  `max_candidates_per_plane` and `kKeepBrightest` now hold over the whole
  drained tail, not each internal step.
- **Logging without spdlog:** debug messages go to `stderr` under a mutex
  (`[dmt] [debug] ...`); spdlog is no longer a dependency and fmt is not
  installed.
- **CMake:** `DMT_BUILD_PYTHON` defaults to `ON` only under `pip install`
  (scikit-build-core), so a plain C++ configure needs no Python; a CMake
  install no longer copies the Python module; FFTW is linked privately.
- **Wheels:** Intel macOS wheels target `x86-64-v3` (AVX2/FMA) like the Linux
  x86-64 ones; `matplotlib` joins the `tests` extra.
- **`CohFDMT` memory:** the fine FDMT's output buffer holds only its root
  output (it held the largest tree level), and the channel-power partials are
  allocated only with `normalize=True`, on the CPU and GPU.
  `CohFDMTPlan.summary()` names the `block_nsamps` that gives 80% valid output
  when under half of a block is valid, and debug logging says when the
  automatic block is held to the 1 GiB spectrum target.
- **CPU SIMD:** AVX2 and ARM64 NEON prefix scans and AVX2 streaming stores on
  the skewed-window path (previously AVX-512 only; ~18% faster packed
  `FDMTBoxcar` on an x86-64-v3 build of `cryopaf_low`); integer `|` SIMD
  reductions in the prefix screen, which Clang vectorizes.

### Removed

- `BoxcarScreenMode::kKadaneSurrogate` (`KADANE_SURROGATE`); use `kKadane`.
- `dmt::AlignedAllocator` and `dmt::AlignedVector` from `dmt/common/types.hpp`.
- `dmt::kDispConstD` (`kDispConst` is now `double`) and the `BoxcarConfig`
  positional constructors.
- No-op `DDMTFFT::set_gulp_size()` / `get_gulp_size()` (C++ and Python).
- `dmtlib.generate_optimal_dm_grid` and `generate_optimal_dt_grid`.
- Internal `DMT_HD`, `DMT_D` and `DMT_H` macros from `dmt/common/types.hpp`.

### Fixed

- GPU `FDMTBoxcar::reset_history()` on the skewed path now clears carried
  top-level samples; failed per-call mask calls leave stream state unchanged.
- GPU kernel shared-memory opt-in limits are only ever raised across engines
  in the same process.
- GPU `FDMTBoxcar` falls back cleanly for wide boxes, applies kill masks
  in-transform without a block staging copy, and handles plans with > 2^31
  root elements; GPU `FDMTFFTBoxcar` excludes host results from
  `max_memory_bytes`.
- GPU `FDMT` stepper on the skewed path raises `std::invalid_argument`
  (`ValueError`) instead of failing with a CUDA error; `CohFDMTPlan`
  rejects automatic `nbin` > 2^20; NaN/inf saved history headers are rejected
  under `-ffast-math`.
- `CohFDMT` (CPU) no longer forms an output pointer before the start of a
  waterfall row, and keeps the forward-FFT rows aligned for a manual `nbin`
  that is not a multiple of 8 (e.g. `n_p=1, nbin=1026`). `CohFDMTPlan` rejects
  a single channel (`nsub * n_p < 2`) with a clear message instead of a fine
  FDMT error, and rejects sizes that overflow the index type.
- `generate_pure_frb` computes in double and no longer writes negative or
  spurious values into sample 0 when the sweep starts before the block.
- CPU `FDMTFFT` stepper: `reset()` no longer keeps a pointer to the caller's
  block (read again by `finalize()`); on CPU and GPU, `execute()` and
  `reset()` in mode `"valid"` raise while a stepper block is unfinished, as
  on `FDMT`. Python `reset()` keeps the running block's buffers when it is
  rejected.
- GPU `DDMTFFT` splits NUFFT runs until their grids fit the spread kernel's
  dynamic shared memory (it failed on 64 KiB GPUs, e.g. `meerkat_lband`).
  GPU engines take the per-block shared-memory limit where the opt-in limit
  is 0 (ROCm), subtract each kernel's static shared memory, and set kernel
  attributes only when they allocate, never in `estimate_memory()`.
- GPU `FDMTBoxcar` sizes the packed-input width screen in floats (it reserved
  twice that and dropped the screen for wide boxes on 64 KiB GPUs and for
  3073–4096 on 99 KiB ones).
- A NaN or infinity in one block no longer blinds the Kadane-screened search
  for the rest of the stream: non-finite carries restart at 0 between calls.
- HIP builds: `cudaMemcpy2D`, `hipHostFree` in the tests, `cuda::std` only
  from a libhipcxx CMake found, and 64-bit warp shuffles as two 32-bit ones.
- `find_package(FFTW)` and `find_package(dmt)` in one project no longer
  fail on duplicate `FFTW::*` targets.
- `DDMT` / `SDMT` reject plans whose delays or DM trial counts exceed `int`;
  `BasebandFormat::levels_2bit` must be finite;
  `FDMTBoxcar::estimate_minimum_memory()` steps the budget up when an
  estimate falls short instead of by one byte.

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

[Unreleased]: https://github.com/pravirkr/dmt/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/pravirkr/dmt/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/pravirkr/dmt/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/pravirkr/dmt/compare/v0.2.0...v0.7.0
[0.2.0]: https://github.com/pravirkr/dmt/releases/tag/v0.2.0
