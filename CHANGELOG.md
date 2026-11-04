# Changelog

All notable changes to `dmt` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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

### Changed

- **Breaking:** the FDMT tree halves the band top-down, the lower half taking
  the extra channel of an odd count, so subbands at any depth differ in width
  by at most one channel (1152 channels: 576, 288, ..., 9 at depths 1 to 7).
  Previously neighbours were paired bottom-up and an odd subband was carried
  up, leaving the root split 1024 + 128. Power-of-two channel counts give the
  same tree and bit-identical outputs; other counts give different (slightly
  more accurate) outputs and subband indices.
- **Breaking:** `BoxcarConfig.subband_levels` is removed; the search depth is
  the plan's `subband_depth`, now any depth below the root (was 0 to 2). The
  positional `BoxcarConfig` constructor loses that argument.
- **Breaking:** `FDMTPlanContainer.df_top`/`df_bot` are removed; each
  subband's channel extent is in its grid.
- Level fusion (CPU, GPU tiles, FDMT-FFT on the CPU) forms channel groups
  from the tree; fusion and the GPU direct levels stay below the lowest level
  holding staggered subbands.
- **Breaking:** every engine refers output times to `f_min`, the bottom band
  edge, by default (`plan.f_ref == f_min`), as FDMT always has: the
  last-arriving edge, when a pulse is complete, is the natural trigger time.
  Pass `f_ref=f_max` to DDMT, SDMT or DDMT-FFT for top-of-band reference times.
  The 0.7.0 reference of those engines, the top channel's centre
  (`f_max - df/2`), is no longer available: with `f_ref=f_max` a pulse is
  timed earlier than in 0.7.0 by `K DM ((f_max - df/2)^-2 - f_max^-2)`, about
  half the top channel's dispersion sweep.
  - DDMT, SDMT and DDMT-FFT (were the top channel's centre, `f_max - df/2`)
    still sample each channel at its centre. Their delay table is now
    `time_offset + delay`, with `delay` each channel centre's delay behind
    `f_ref` (negative above it); at the default `f_min` entries shrink with
    DM and `fractional_delay_table` (`d(delay)/d(DM)`) is `<= 0`. Output sample `k` is the `f_min` arrival at input
    sample `k + plan.time_offset`, an offset that follows the plan's largest
    DM: align plans with different DM grids by it. `dmtlib.dedisperse`
    reports it as `start`. History, output counts and performance are
    essentially unchanged (the history grows by about half a channel's
    sweep).
  - CohFDMT (was the bottom channel's centre) shifts channels by their
    centre's lag behind `f_min`, matching its fine FDMT stage.
    `output_time_offset` keeps its meaning at the new reference.
  - `utils::generate_delay_table`/`generate_fractional_delay_table` take
    `(f_min, df, tsamp, f_ref)` of an ascending band instead of
    `(fch1, foff, tsamp)`; `generate_delay_offset` is new.
- Rows are always ascending in frequency (row 0 lowest); descending data
  must be flipped before it reaches dmt (see `dmtlib.sigproc_band`).
- **Breaking (Python):** every `kill_mask` argument, and the
  `DDMTPlan.kill_mask` setter, takes a 1-D bool or integer array or sequence
  of any integer dtype, and any nonzero value keeps the channel. Float,
  complex, object and string masks raise `TypeError`; they used to be cast to
  uint8, so a 0.5 killed its channel. Integer values are tested before any
  narrowing, so 256 no longer wraps to 0 and kills the channel.

### Fixed

- `FDMTFFT.get_effective_variance`/`get_effective_sigma` and their grids
  ignored the kill mask. Killed channels are now left out, as in
  `FDMTFFTBoxcar`; the engine passes its mask to the plan's methods.
- `FDMTFFTBoxcar` held back detections until killed channels had warmed up.
  The warmup now counts only active channels, so killing the top of the band
  shortens it.
- `FDMTPlan` rejects NaN or infinite `dm_grid` values and DMs whose delay
  does not fit in 32 bits; `generate_levin_dm_grid` rejects non-finite
  parameters. The NaN and infinity checks read the bit pattern through a
  `volatile`, so `-ffast-math` Release builds cannot optimise them away.

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
