# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- **Staggered subband search.** `FDMTPlan(..., subband_depth=d)` adds, at each
  depth `k = 1..d`, the `2**k - 1` subbands of the same width centred on the
  boundaries between that depth's `2**k` tree subbands. Each costs one extra
  merge of two existing nodes. `FDMTBoxcar` searches every tree and staggered
  subband to depth `d`; candidates gain a `staggered` field. A narrowband
  burst on a subband boundary is no longer split between aligned subbands.
- `FDMT(plan, ...)`: build an engine from a prepared plan. The stepper's
  `num_subbands()`/`view_subband()` include the staggered subbands, and
  subband views gain `chan_begin`, `chan_end` and `staggered`.
- `FDMTFFT(plan, ...)`, likewise: its stepper exposes the staggered subbands.
- `FDMTPlan.get_subband_variance_grid(level, subband, boxcar_width,
  use_box_smearing)` and `get_subband_sigma_grid(...)`: the noise of every
  row of any tree or staggered subband, for S/N on stepper views.
  `FDMTFFT.get_subband_variance_grid(level, subband, boxcar_width)` and
  `get_subband_sigma_grid(...)` give the same with the engine's box
  smearing and kill mask.
- Every `FDMTPlan` variance and sigma method takes an optional kill mask
  (C++: a trailing `std::span<const uint8_t>`; Python: keyword-only
  `kill_mask=`) and leaves killed channels out.
- `FDMTCoordGrid` (Python `FDMTSubDTGrid`) gains `chan_begin`, `chan_end`,
  `i_sub_tail`, `i_sub_head` and `staggered`; `FDMTShape` gains `nstaggered`;
  `FDMTPlan.get_subband_depth()` and `get_max_fuse_levels()`.
- **`f_ref` for DDMT, SDMT and DDMT-FFT** (plans and engines, C++ and
  Python, and `dmtlib.dedisperse`): the reference frequency of the output
  times, either `f_min` (default) or `f_max`. Passing `f_max` gives
  PRESTO/SIGPROC-style series: output sample `t` is input sample `t` at `f_max`
  for every DM, and the incomplete samples are at the end. Only the delay
  table changes. Any other value raises `std::invalid_argument`
  (`ValueError` in Python). The FDMT engines stay referenced to `f_min`.
- `DDMTPlan.get_time_offset()` (Python `plan.time_offset`): output sample
  `k` of DDMT, SDMT and DDMT-FFT is the `f_ref` arrival at input sample
  `k + time_offset` (0 when `f_ref == f_max`).
- `dmtlib.sigproc_band(fch1, foff, nchans)` and `dmtlib.to_ascending()`:
  band edges from a SIGPROC header and the flip to ascending channel order.
  The getting-started guides and README document the channel order.
- `dmtlib.dispersion_delay` broadcasts over array frequencies.
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
- **Faster float32 on the CPU skewed path.** The engine now picks its leaf
  level (the level read from the raw ring) together with the chunk, in
  measured speed order within the budget: leaf level 1 for float input, 2
  for 8-bit (when the width is known at construction, as in FDMTBoxcar),
  instead of always 4. On `parkes_uwl` (8 threads) the float32
  FDMT transform takes 4.9 s per 2.1 s block instead of 13.2 s (the old
  tree path took 5.75 s in 16 GB), and the depth-4 float32 FDMTBoxcar search
  20 s instead of 27.8 s. Output stays bit-identical to the tree path.
- Benchmark preset `cryopaf` is renamed `cryopaf_low`; `meerkat_lband` is
  the default preset; `parkes_uwl` searches at subband depth 4. Results
  live in `bench/results/<config>/<machine>/` for every preset
  (`bench/results/cfdmt/` for CohFDMT), and the suite no longer measures
  unplotted points (integer-delay FDMT-FFT, brute-force DDMT-FFT, host-array
  GPU rows). `dmt_bench` and `dmt_bench_mem` keep only developer
  microbenchmarks.
- **End of stream: `drain()`** on `FDMTBoxcar` and `FDMTFFTBoxcar` (C++
  and Python). It searches the samples still behind the search latency, then
  emits the open excursions: the last `get_output_latency()` samples of
  `FDMTFFTBoxcar` with fractional delays, and the last subband row lags of a
  subband search on the skewed paths, which `flush()` never searched.
  Endpoints after the last sample are not searched, so no candidate lies
  past the end, and a drained skewed-path subband search now matches the
  tree path's candidates over the whole stream. The stream ends there:
  `execute()` throws `std::logic_error` (`RuntimeError`) until
  `reset_history()`. New: `get_searched_nsamps()` (`searched_nsamps`), the
  stream samples every row has searched, and
  `FDMTBoxcar::get_search_latency()` (`search_latency`).

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

[0.7.0]: https://github.com/pravirkr/dmt/compare/v0.2.0...v0.7.0
[0.2.0]: https://github.com/pravirkr/dmt/releases/tag/v0.2.0
