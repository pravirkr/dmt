# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

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
