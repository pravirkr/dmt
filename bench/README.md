# dmt benchmarks

Three benchmark executables are built with `-DDMT_BUILD_BENCHMARKS=ON`:

| binary | sources | description |
| :--- | :--- | :--- |
| `dmt_bench_suite` | `bench/suite/` | The published benchmarks (every preset, every engine, FDMTBoxcar, CohFDMT) |
| `dmt_bench` | `bench/*_b.cpp`, `bench/*_b.cu` | Developer microbenchmarks: kernel/fusion calibration, the skewed-window stage breakdown, CohFDMT stages |
| `dmt_bench_mem` | `bench/*_mem_b.cpp` | Peak memory of the tree and skewed-window FDMT engines on the presets |

`bench/scripts/` holds the Python drivers: `run_suite.py` (runs one suite on
one preset), `plot_suite.py`, `plot_uwl.py` and `plot_cfdmt.py` (plots and
tables), and developer analysis scripts (`fdmt_cache_benchmark.py`,
`fdmt_operations_costs.py`, sensitivity studies).

## Presets

Four filterbank presets share DM 0–2000, a piecewise-uniform Levin grid,
valid mode (streaming) and box smearing. They are frozen: the published
numbers are only comparable while they stay as they are.

| preset | band (MHz) | channels | sample interval | Levin tol. | DM trials | subband depth | reference block |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `meerkat_lband` (default) | 856–1712 | 4096 | 612.48 µs | 1.1 | 4057 | 0 | 32768 samples (20.07 s) |
| `cryopaf_lband` | 1100–1714 | 1152 | 455.6 µs | 1.1 | 1941 | 1 | 32768 samples (14.93 s) |
| `cryopaf_low` | 704–1216 | 512 | 512 µs | 1.1 | 1545 | 0 | 32768 samples (16.78 s) |
| `parkes_uwl` | 704–4032 | 6656 | 64 µs | 1.6 | 4749 | 4 | 32768 samples (2.097 s) |

- **`meerkat_lband`** is the headline preset (README and the first tab of
  the docs): the 4096-channel MeerTRAP / TRAPUM filterbank mode downsampled
  by two in time, 12 tree levels on a dense grid, 101× fewer operations
  than brute force. Its DM 2000 sweep is 8.49 s (13869 samples).
- **`cryopaf_lband`** is the CryoPAF production configuration, including the
  FDMTBoxcar search at subband depth 1.
- **`cryopaf_low`** is the sparse-grid, few-channel, wide-fractional-band
  regime where FDMT, DDMT, SDMT and DDMT-FFT trade places (5.7× operation
  ratio).
- **`parkes_uwl`** is the Parkes UWL search: FDMTBoxcar (and the FDMT
  transform) only, at subband depth 4 (16 subbands, their 15 staggered
  neighbours, then 8 + 7, 4 + 3, 2 + 1 and the full band). Its DM 2000
  sweep is 16.23 s (253891 samples), eight reference blocks, so histories
  are warmed before timing. It is not in the five-engine plots: FDMT-FFT
  needs ~50 GB (510k-sample segments) and brute-force CPU DDMT takes hours.
  Tolerance 1.6 keeps the grid under 5000 trials (1.1 would need 12941).

The CohFDMT baseband search has its own configuration (`cfdmt`, below).

## Suites

`run_suite.py --suite <main|boxcar|cfdmt> --config <preset>` runs one suite
and writes Google Benchmark JSON (machine description in its `context`) to
one directory per preset:

```text
bench/results/
├── meerkat_lband/  l40s-epyc9654/{suite_cpu,suite_cuda,boxcar_cpu,boxcar_cuda}.json  plots/
├── cryopaf_lband/  l40s-epyc9654/{suite_cpu,suite_cuda,boxcar_cpu,boxcar_cuda}.json  plots/
├── cryopaf_low/    l40s-epyc9654/{suite_cpu,suite_cuda,boxcar_cpu,boxcar_cuda}.json  plots/
├── parkes_uwl/     l40s-epyc9654/{boxcar_cpu,boxcar_cuda}.json                       plots/
└── cfdmt/          l40s-epyc9654/{suite_cfdmt_cpu,suite_cfdmt_cuda}.json             plots/
```

**`main`** (`suite/...`): FDMT, FDMT-FFT (fractional delays), DDMT, SDMT and
DDMT-FFT (NUFFT) on the same DM grid, in three sweeps through the reference
point:

- **nsamps**: 8K, 16K, 32K, 64K samples per block, at DM 0–2000.
- **ndms**: DM 0–125, 0–250, 0–500, 0–1000 and 0–2000, at 32K samples.
- **nbits**: 1, 2, 4, 8, 16-bit packed and float32 input.

CPU at 1 and 8 threads (FDMT-FFT, DDMT and SDMT at 8 only); CUDA with
device-resident data. DDMT and SDMT points of more than 10¹⁰ additions run
one repetition (they are compute-bound and stable). Points whose estimated
memory exceeds `--max-gb` are skipped and listed in the summary.

**`boxcar`** (`BM_fdmt_boxcar_*`): FDMTBoxcar at the preset's reference
point and subband depth, boxcars to 100 ms (`max_boxcar = floor(0.1 s /
tsamp)`) at two width spacings (1.0 and 1.5; UWL 1.5 and 2.0), float32 and
packed 8-bit, with the default 10 GiB memory budget:

- CPU (`BM_fdmt_boxcar_stream`): 1 and 8 threads (UWL: 8).
- GPU (`BM_fdmt_boxcar_cuda`): the FDMT transform alone (the floor), FDMT
  plus a separate search of the plane, FDMTBoxcar at depth 0 and the
  preset's depth, and FDMT-FFT with FDMTFFTBoxcar.
- **`parkes_uwl` real-time sweep** (`BM_fdmt_boxcar_rt`, CPU at 8 threads
  and GPU): FDMTBoxcar at subband depth 4 on the UWL band at the native
  64 µs and downsampled to 256, 512 and 1024 µs, on blocks of the same
  2.097 s, each resolution with its own Levin grid and 100 ms widths: the
  real-time factor and memory of searching at a coarser resolution.

They warm the search histories first and report the real-time factor,
working memory, chunk length and candidates per block. `DMT_BENCH_CHUNK`
overrides the automatic chunk length (tuning).

**Peak memory** is measured next to every `main` and `boxcar` timing, in the
same run (`bench/suite/suite_memory.hpp`, `suite_memory_gpu.cuh`):

- `peak_rss_bytes`: the process's peak resident set size from the point's
  construction through warm-up and its timed blocks. On Linux the kernel's
  high-water mark is reset when the point starts (`/proc/self/clear_refs`,
  after returning freed heap with `malloc_trim`); `base_rss_bytes` is the
  resident size at that reset (the benchmark process itself). Elsewhere it
  is the monotonic `getrusage` peak (`rss_peak_resettable = 0`).
- `peak_device_bytes` (GPU): device memory in use above the point's
  starting level, sampled every millisecond (`cudaMemGetInfo`) through
  construction and warm-up, and once after the timed blocks; the sampler is
  paused while they run.
- `working_bytes` (boxcar suite): the library's own accounting,
  `get_memory_usage().total()` plus the input block.

`plot_suite.py` and `plot_uwl.py` print them in the summaries (*allocated*
against *peak*). The CohFDMT suite does not record them.

**`cfdmt`** (`cfdmt/...`): CohFDMT on one GUPPI node (64 × 2.93 MHz at
1.31–1.50 GHz, int8 FTPRI, t_p = 10 µs, DM 50–60;
`bench/suite/suite_cfdmt_common.hpp`), swept over t_p, input bit width, DM
range width and block length.

The Fourier-domain engines are FFT-bound, so the suite plans their FFTs
with `FFTW_MEASURE` (`--fftw-planner`), keeping the wisdom in
`<build-dir>/fftw_<machine>.wisdom` so only the first run pays the planning
time (at construction, outside the timed region).

## Reproducing the published numbers

They come from one node: an AMD EPYC 9654 (up to 8 threads, one per CCD)
and an NVIDIA L40S, results folder `l40s-epyc9654` (the runner's default
`--machine`).

```bash
# 1. Release build with benchmarks (CUDA is detected automatically)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_BENCHMARKS=ON
cmake --build build -j --target dmt_bench_suite

# 2. Run the suites (CPU, then GPU, for each)
export OMP_PLACES='{0}:8:8'
for config in meerkat_lband cryopaf_lband cryopaf_low; do
  python bench/scripts/run_suite.py --config $config
  python bench/scripts/run_suite.py --suite boxcar --config $config
done
python bench/scripts/run_suite.py --suite boxcar --config parkes_uwl
python bench/scripts/run_suite.py --suite cfdmt

# 3. Plots and tables (dmtlib importable for the operation-count plot)
pip install -e ".[bench]"
for config in meerkat_lband cryopaf_lband cryopaf_low; do
  python bench/scripts/plot_suite.py --config $config
done
python bench/scripts/plot_uwl.py
python bench/scripts/plot_cfdmt.py
```

Each plot script writes light and dark PNGs to
`bench/results/<config>/plots/{light,dark}/` and a numbers table to
`plots/summary.md`. The README's highlight figure comes from
`meerkat_lband`; the docs Benchmarks page has a tab per preset plus the UWL
and CohFDMT sections; both reference those files directly. Brute-force DDMT
takes most of a `main` CPU run; `--quick` (1 repetition, short runs) is for
smoke tests only.

### Getting comparable numbers

- Release build only; `run_suite.py` warns otherwise. The JSON records the
  dmt version, and `plot_suite.py` warns when result files disagree.
- `--placement close` (the default) pins threads with
  `OMP_PROC_BIND=close` and `OMP_PLACES` (default `cores`; the published
  numbers use `'{0}:8:8'`). `--placement none` leaves them unbound. The
  JSON records the placement, virtualisation and the load average before
  and after each run.
- The FDMT is L3-bound. On CPUs whose L3 is split per core complex (AMD
  EPYC: 32 MB per 8-core CCD), `OMP_PLACES=cores` packs 8 threads into one
  CCD: the `cryopaf_low` transform takes 619 ms there against 258 ms with
  one thread per CCD (`OMP_PLACES='{0}:8:8'`).
- The runner starts the binaries under `setarch -R`. Google Benchmark 1.10
  re-executes itself to turn off ASLR, and the re-executed process inherits
  the one-core affinity libgomp gave the main thread. Running a binary by
  hand with `OMP_PROC_BIND` set needs `setarch -R` too.
- On a shared host, run only the GPU half (`--no-cpu`): device timings do not
  depend on host load.

### Runner options

```text
--suite main       main | boxcar | cfdmt
--config NAME      meerkat_lband (default) | cryopaf_lband | cryopaf_low | parkes_uwl
--machine NAME     results folder (default l40s-epyc9654)
--label TEXT       display name in plots (default: the CPU / GPU model)
--build-dir DIR    CMake build directory (default: build, build2, build*)
--threads 1,8      CPU thread counts
--placement close  close: OMP_PROC_BIND=close; none: unbound
--repetitions 3    repetitions per point (plots use the median)
--max-gb GB        suite memory budget for skipping points (default 60% of RAM)
--no-cpu / --no-gpu (alias --no-cuda)
--quick            1 repetition, short minimum time
--filter REGEX     extra filter AND-ed with the suite (e.g. 'FDMT/')
```

The raw binary can also be run directly, e.g.
`DMT_BENCH_CONFIG=parkes_uwl build/bench/dmt_bench_suite
--benchmark_filter='^BM_fdmt_boxcar_rt/'`.

## Developer microbenchmarks

`dmt_bench` keeps the focused measurements behind the library's tuning
constants: FDMT CPU fusion depth and tile (`fdmt_cpu_b.cpp`), the CUDA
kernels and fusion (`fdmt_cuda_b.cu`), the DDMT kernel roofline
(`ddmt_cpu_b.cpp`, `ddmt_cuda_b.cu`), the skewed-window CPU engine's
chunk / level / stage breakdown on the presets (`fdmt_skew_cpu_b.cpp`), and
the CohFDMT stages (`cfdmt_cpu_b.cpp`: unpack, forward FFT, fine FDMT, the
inverse-FFT roofline; `cfdmt_cuda_b.cu`: fused and cuFFT coherent stages,
chunked forward FFT, the fused kernel against a device copy). `bm_skew_env` in
`fdmt_skew_cpu_b.cpp` builds one skewed engine from `DMT_SKEW_NBITS`,
`DMT_SKEW_CHUNK`, `DMT_SKEW_LEVEL` (leaf level), `DMT_SKEW_FUSE` and
`DMT_SKEW_THREADS` and prints its window summary and stage times (the sweep
behind the CPU engine's (chunk, leaf level) order). `dmt_bench_mem`
reports the tree and skewed-window FDMT transform's peak memory on the
preset given by `DMT_BENCH_CONFIG` (run one benchmark per process and
compare `ProcessPeakRSS_MB`); the published memory figures come from the
suites.
