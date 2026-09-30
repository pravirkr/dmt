# dmt benchmarks

Three benchmark executables are built with `-DDMT_BUILD_BENCHMARKS=ON`:

| binary | sources | description |
| :--- | :--- | :--- |
| `dmt_bench_suite` | `bench/suite/` | The published benchmarks suite |
| `dmt_bench` | `bench/*_b.cpp`, `bench/*_b.cu` | Developer microbenchmarks |
| `dmt_bench_mem` | `bench/*_mem_b.cpp` | Peak heap / RSS per engine |

`bench/scripts/` holds the Python drivers: `run_suite.py` and `plot_suite.py`
for the published suite, and analysis scripts (`fdmt_cache_benchmark.py`,
`fdmt_operations_costs.py`, `fdmt_sensitivity.py`).

## The published suite

Fixed data: **4096 channels, 704–1216 MHz, 81.92 µs sampling, `valid` mode,
box smearing on**. Three sweeps, each through the reference point
(16,384 samples × 2,049 DM trials):

- **nsamps**: 4K, 8K, 16K, 32K, 64K samples per block, at 2049 DM trials.
- **ndms**: 256, 512, 1K, 2K, 4K DM trials (via `dt_max`), at 16K samples.
- **nbits**: 1, 2, 4, 8, 16-bit packed and float32 input, at the reference point.

DDMT and SDMT are given the FDMT plan's DM grid, so all
algorithms compute the same trials. The backends are:

- CPU with 1 and 8 threads. FDMT-FFT, DDMT and SDMT run at 8 threads only; DDMT at
  1 thread takes ~25 s per call.
- CUDA with device-resident data (`cuda`).
- CUDA with host arrays including PCIe (`cuda_host`), for FDMT, FDMT-FFT
  and DDMT-FFT throughput in the nbits sweep. These rows include host-side
  staging (pageable copies, pinned chunking), so they depend on the host's
  load as well as the GPU.

Points whose estimated memory exceeds the budget are skipped and listed in
the summary.

The Fourier-domain engines (FDMT-FFT, DDMT-FFT) are FFT-bound, so the suite
plans their FFTs with `FFTW_MEASURE` (`run_suite.py --fftw-planner`, default
`measure`), as a long-running pipeline would. The wisdom is kept in
`<build-dir>/fftw_<machine>.wisdom`, so only the first run pays the planning
time (a few seconds per transform length, at engine construction, outside
the timed region).

### Running it on a machine

On **each** machine (the Apple M1 Pro, the Intel Xeon, the GPU box):

```bash
# 1. Release build with benchmarks (on the GPU box, CUDA is detected automatically)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_BENCHMARKS=ON
cmake --build build -j --target dmt_bench_suite

# 2. Run the suite (writes bench/results/<machine>/suite_{cpu,cuda}.json)
python bench/scripts/run_suite.py --machine m1-pro --label "Apple M1 Pro"
python bench/scripts/run_suite.py --machine xeon-6348h --label "Xeon Gold 6348H"
python bench/scripts/run_suite.py --machine l40s --no-cpu   # GPU box: CUDA only

# 3. Commit the JSON
git add bench/results/<machine>/ && git commit -m "bench: <machine> results"
```

Then, **once all machines have committed** (on any machine with dmtlib
installed, for the theory plot):

```bash
pip install -e ".[bench]"      # matplotlib + numpy
python bench/scripts/plot_suite.py
git add bench/results/plots && git commit -m "bench: update plots"
```

`plot_suite.py` writes light and dark PNGs to `bench/results/plots/{light,dark}/`
and a numbers table to `bench/results/plots/summary.md`. The README and the
docs Benchmarks page reference those files directly, so nothing else needs
editing. A full run takes about 8 minutes on the M1 Pro; brute-force DDMT is most of that. `--quick` (1 repetition, short runs) is for smoke tests only.

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

The stage microbenchmarks (unpack, forward FFT, fine FDMT per trial and the
FFTW roofline of the per-trial inverse transforms) are in `dmt_bench`
(`--benchmark_filter=cfdmt`).

The raw binary can also be run directly, e.g.
`build/bench/dmt_bench_suite --benchmark_filter='suite/ndms/FDMT/'`.
