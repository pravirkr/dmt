# dmt benchmarks

Three benchmark executables are built with `-DDMT_BUILD_BENCHMARKS=ON`:

| binary | sources | description |
| :--- | :--- | :--- |
| `dmt_bench_suite` | `bench/suite/` | The published benchmarks suite |
| `dmt_bench` | `bench/*_b.cpp`, `bench/*_b.cu` | Developer microbenchmarks |
| `dmt_bench_mem` | `bench/*_mem_b.cpp` | Peak heap / RSS per engine |

`bench/scripts/` holds the Python drivers: `run_suite.py` and `plot_suite.py`
for the published suite, and developer analysis scripts
(`fdmt_cache_benchmark.py`, `fdmt_operations_costs.py`).

## The published suite

The presets share DM 0–2000, a piecewise-uniform Levin grid, valid mode and
box smearing:

| preset | band (MHz) | channels | sample interval | Levin tolerance | DM trials | reference block |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `cryopaf` (default) | 704–1216 | 512 | 512 µs | 1.1 | 1545 | 32768 samples (16.777216 s) |
| `cryopaf_lband` | 1100–1714 | 1152 | 455.6 µs | 1.1 | 1941 | 32768 samples (14.9291008 s) |
| `parkes_uwl` | 704–4032 | 6656 | 64 µs | 1.6 | 4749 | 32768 samples (2.097152 s) |

UWL uses tolerance 1.6 to keep its grid under 5000 trials (1.1 would need
12941 and 23.7 GiB of FDMT tree history, against 9.8 GiB now).

CryoPAF emphasizes the sparse-grid, lower-channel regime; L-band is the
pipeline target, including one subband search level; UWL stresses wide
fractional bandwidth, non-power-of-two channel counts, fine time sampling,
and long streaming histories. At DM 2000 the UWL sweep is **16.233853 s**
(about **253654 samples**), much longer than its reference block. Valid
streaming permits this; histories must be warmed before steady-state timing.
Full-band search is the default for CryoPAF and UWL.

Each preset runs three sweeps through its reference point:

- **nsamps**: 8K, 16K, 32K, 64K samples per block, at DM 0–2000.
- **ndms**: DM 0–125, 0–250, 0–500, 0–1000 and 0–2000, at 32K samples.
- **nbits**: 1, 2, 4, 8, 16-bit packed and float32 input.

Every engine is given the same DM grid (the FDMT engines round it to integer
delays and keep every trial; DDMT-FFT runs one NUFFT per uniform segment), so
all algorithms compute the same trials. The backends are:

- CPU with 1 and 8 threads. FDMT-FFT, DDMT and SDMT run at 8 threads only.
- CUDA with device-resident data (`cuda`).
- CUDA with host arrays including PCIe (`cuda_host`), for FDMT, FDMT-FFT
  and DDMT-FFT throughput in the nbits sweep. These rows include host-side
  staging (pageable copies, pinned chunking), so they depend on the host's
  load as well as the GPU.

Points whose estimated memory exceeds the budget are skipped and listed in
the summary.

Non-default results and plots go under `bench/results/<config>/`; the
default uses `bench/results/`. FDMT memory preflight includes streaming
histories and host plan tables, with doubled history storage on GPUs (an
upper bound: the GPU now keeps the tree history above the fused levels in one
in-place ring).

**UWL is deferred.** On the L40S the float32 search needs 15–16 GB (uint8:
26 GB) and FDMT-FFT cannot run at all (its shortest overlap-save segment is
510k samples). Real-time use must fit in under 10 GB, so no UWL results are
published until a lower-memory design (recomputing levels instead of storing
history) exists; the preset stays for development.

The fused search benchmarks (`FDMTBoxcar`, `FDMTFFTBoxcar`) are in `dmt_bench`,
at the reference point of the preset selected with `DMT_BENCH_CONFIG`, with
boxcar widths up to 100 ms (`max_boxcar = floor(0.1 s / tsamp)`) at two width
spacings per preset (1.0 and 1.5 for CryoPAF, 1.5 and 2.0 for UWL):

- GPU, `--benchmark_filter='^BM_fdmt_boxcar_cuda/'`: the transforms alone
  (FDMT and FDMT-FFT writing the DM–time plane), FDMT followed by a separate
  search of that plane, `FDMTBoxcar` (float32, packed uint8 and host arrays)
  and `FDMTFFTBoxcar`.
- CPU, `--benchmark_filter='^BM_fdmt_boxcar_stream/'`: `FDMTBoxcar` on float32
  and packed uint8 input at the same width sets, on 1 and 8 threads.

They warm the search histories first and report the real-time factor,
working memory, candidates per block and the chunk length.
`DMT_BENCH_CHUNK` overrides the automatic FDMT chunk length. Results are kept
as `bench/results/[<config>/]<machine>/boxcar_{cuda,cpu}.json` (Google
Benchmark JSON, written with `--benchmark_out`), for example:

```bash
DMT_BENCH_CONFIG=cryopaf_lband build/bench/dmt_bench \
  --benchmark_filter='^BM_fdmt_boxcar_cuda/' --benchmark_repetitions=3 \
  --benchmark_out=bench/results/cryopaf_lband/l40s/boxcar_cuda.json \
  --benchmark_out_format=json
```

The Fourier-domain engines (FDMT-FFT, DDMT-FFT) are FFT-bound, so the suite
plans their FFTs with `FFTW_MEASURE` (`run_suite.py --fftw-planner`, default
`measure`), as a long-running pipeline would. The wisdom is kept in
`<build-dir>/fftw_<machine>.wisdom`, so only the first run pays the planning
time (a few seconds per transform length, at engine construction, outside
the timed region).

### Running it on a machine

On **each** machine (the Intel Xeon for the CPU results, the L40S box for the
GPU results), for each preset:

```bash
# 1. Release build with benchmarks (on the GPU box, CUDA is detected automatically)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_BENCHMARKS=ON
cmake --build build -j --target dmt_bench_suite

# 2. Run the suite: bench/results/<machine>/suite_{cpu,cuda}.json for
#    cryopaf, bench/results/<config>/<machine>/ for the other presets
for config in cryopaf cryopaf_lband; do      # parkes_uwl: deferred, see above
  python bench/scripts/run_suite.py --machine xeon-6348h \
      --label "Xeon Gold 6348H" --no-gpu --config $config     # CPU box
  python bench/scripts/run_suite.py --machine l40s --label "NVIDIA L40S" \
      --no-cpu --config $config                               # GPU box
done

# 3. Commit the JSON
git add bench/results/<machine>/ && git commit -m "bench: <machine> results"
```

Then, **once all machines have committed** (on any machine with dmtlib
installed, for the theory plot):

```bash
pip install -e ".[bench]"      # matplotlib + numpy
python bench/scripts/plot_suite.py
python bench/scripts/plot_suite.py --config cryopaf_lband
git add bench/results/plots && git commit -m "bench: update plots"
```

`plot_suite.py` writes light and dark PNGs to `bench/results/plots/{light,dark}/`
and a numbers table to `bench/results/plots/summary.md`. The README and the
docs Benchmarks page reference those files directly, so nothing else needs
editing. Brute-force DDMT takes most of a CPU run. `--quick` (1 repetition,
short runs) is for smoke tests only.

### Getting comparable numbers

- Run every machine from **the same dmt version** (the `project(... VERSION)`
  in `CMakeLists.txt`). The JSON records `dmt_version`, and `plot_suite.py`
  warns when the machines disagree.
- Release build only; `run_suite.py` warns otherwise.
- Threads are pinned with `OMP_PROC_BIND=close OMP_PLACES=cores`, which you can
  override in the environment. `--threads 1,8` is the default.
- `--max-gb` sets the memory budget (default 60% of RAM). Points above it
  are skipped rather than swapped.

### Runner options

```text
--machine NAME     results folder under bench/results/ (required)
--label TEXT       display name in plots (default: the CPU / GPU model)
--build-dir DIR    CMake build directory (default: build, build2, build*)
--threads 1,8      CPU thread counts
--no-cpu / --no-gpu (alias --no-cuda)
--quick            1 repetition, short minimum time
--filter REGEX     extra filter AND-ed with the suite (e.g. 'FDMT/')
--suite cfdmt      the CohFDMT baseband search instead (own config and plots)
```

### CohFDMT section

CohFDMT searches baseband voltages, so it has its own fixed configuration
(one GUPPI node: 64 x 2.93 MHz at 1.31-1.50 GHz, int8 FTPRI, t_p = 10 us,
DM 50-60, `dt_step` 16; `bench/suite/suite_cfdmt_common.hpp`) swept over t_p,
input bit width, DM range width and block length. It is not compared with the
filterbank algorithms:

```bash
python bench/scripts/run_suite.py --machine <name> --suite cfdmt  # suite_cfdmt_<kind>.json
python bench/scripts/plot_cfdmt.py   # plots/{light,dark}/cfdmt_rtf.png, plots/cfdmt_summary.md
```

The stage microbenchmarks are in `dmt_bench`:

- CPU (`--benchmark_filter=cfdmt_(execute|stage)`): unpack, forward FFT, fine
  FDMT per trial, and the FFTW roofline of the per-trial inverse transforms.
- GPU (`--benchmark_filter=cfdmt_cuda`): end to end with the fused or the
  cuFFT coherent stage, unpack, the forward FFT as one batch or in L2-sized
  chunks, and the fused coherent kernel. For the fused kernel, `roofline` is
  its speed against a device-to-device copy of the same bytes. The fine FDMT
  per trial is also timed.

On a shared host, run only the GPU half (`run_suite.py --no-cpu`): the device
timings do not depend on host load, except for the `*_host` points, which
include the PCIe copies.

The raw binary can also be run directly, e.g.
`build/bench/dmt_bench_suite --benchmark_filter='suite/ndms/FDMT/'`.
