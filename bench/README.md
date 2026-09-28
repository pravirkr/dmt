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

DDMT and SDMT (CPU only) are given the FDMT plan's DM grid, so all
algorithms compute the same trials. The backends are:

- CPU with 1 and 8 threads. FDMT-FFT, DDMT and SDMT run at 8 threads only; DDMT at
  1 thread takes ~25 s per call.
- CUDA with device-resident data (`cuda`).
- CUDA with host arrays including PCIe (`cuda_host`), for FDMT throughput.

Points whose estimated memory exceeds the budget are skipped and listed in
the summary.

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
```

The raw binary can also be run directly, e.g.
`build/bench/dmt_bench_suite --benchmark_filter='suite/ndms/FDMT/'`.
