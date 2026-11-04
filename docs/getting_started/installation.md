# Installation

You can use `dmt` as a Python library (`dmtlib`), as a C++20 library found
with CMake `find_package(dmt)`, or both.

## System requirements

Hardware:

- CPU: x86_64 with AVX2/AVX-512, or Apple Silicon (ARM64 with NEON).
- GPU (optional): an NVIDIA GPU with compute capability 7.0+ (7.5+ with
  CUDA 13), or an AMD GPU supported by ROCm (HIP).

Compilers and dependencies:

- A C++20 compiler: GCC 11+ (the system compiler of Ubuntu 22.04, Debian 12
  and RHEL 9) or Clang 17+.
- CMake 3.18+ (3.21+ for the presets and HIP) and Ninja or Make.
- Python 3.12 or newer.
- System libraries:
  - `libfftw3-dev` (FFTW 3.3+)
  - OpenMP runtime (`libomp` on macOS, included with GCC on Linux)
  - Optional GPU toolchain (one per build; every algorithm runs on it):
    - CUDA Toolkit 12.2+ (NVIDIA); a 12.x build runs on driver R525 or newer
    - ROCm 6.2+ with hipFFT and rocThrust, and CMake 3.21+ (AMD, via HIP);
      libhipcxx is used when installed

::::{tab-set}
:::{tab-item} Ubuntu/Debian
:sync: linux

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libfftw3-dev libomp-dev
```
:::
:::{tab-item} macOS (Homebrew)
:sync: macos

```bash
brew install cmake ninja fftw libomp
```
:::
:::{tab-item} conda (no root)
:sync: conda

For clusters and other machines without root access, or with a system
compiler older than GCC 11, `environment.yml` provides the compiler, FFTW,
CMake and Python from conda-forge:

```bash
conda env create -f environment.yml   # or mamba / micromamba
conda activate dmt
```

Then install from the clone as below (`pip install .`).
:::
::::

## Python package

### From source with `uv` or `pip`

Clone the repository and install into your active Python environment:

```bash
git clone https://github.com/pravirkr/dmt.git
cd dmt
```

::::{tab-set}
:::{tab-item} Install
:sync: install

```bash
# Build with scikit-build-core and install
uv pip install .
```
:::
:::{tab-item} Editable (development)
:sync: editable

```bash
# Or an editable install with the development extras
uv pip install -e ".[develop,docs,tests]"
```
:::
::::

```{tip}
For a Python pipeline, `uv pip install .` with the default `DMT_GPU=AUTO` is
all you need: the build picks up a CUDA or ROCm toolchain if it finds one.
```

### GPU builds

By default the build uses whichever GPU toolchain it finds (`DMT_GPU=AUTO`:
CUDA first, then ROCm). To require one:

::::{tab-set}
:::{tab-item} CUDA
:sync: cuda

```bash
CMAKE_ARGS="-DDMT_GPU=CUDA" uv pip install .   # NVIDIA
```
:::
:::{tab-item} HIP (AMD ROCm)
:sync: hip

```bash
CMAKE_ARGS="-DDMT_GPU=HIP" uv pip install .    # AMD (ROCm)
```
:::
::::

To check the install:

```python
import dmtlib

print(f"dmtlib version: {dmtlib.__version__}")
```

## C++ library

### Build and install

To build and install the C++20 static or shared library:

```bash
mkdir build && cd build
```

Configure for the backend you want:

::::{tab-set}
:::{tab-item} CPU
:sync: cpu

```bash
# Configure a CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_TESTING=ON
```
:::
:::{tab-item} CUDA
:sync: cuda

```bash
# Configure a CUDA build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDMT_GPU=CUDA -DDMT_CUDA_ARCHITECTURES=native
```
:::
:::{tab-item} HIP (AMD ROCm)
:sync: hip

```bash
# Configure a HIP (ROCm) build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDMT_GPU=HIP -DDMT_HIP_ARCHITECTURES="gfx90a;gfx942"
```
:::
::::

Then build, test and install:

```bash
# Build and run the unit tests
cmake --build . -j
ctest --output-on-failure

# Install to the system or a prefix
cmake --install .
```

### With CMake presets

`CMakePresets.json` (CMake 3.21+) names the common configurations. Each one
configures, builds and tests in its own directory:

```bash
cmake --preset release            # or cuda, hip, portable, asan, ...
cmake --build --preset release
ctest --preset release
```

### Using `find_package(dmt)`

Once installed, link `dmt::dmt` in your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_telescope_pipeline CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dmt REQUIRED)

add_executable(search_engine main.cpp)
target_link_libraries(search_engine PRIVATE dmt::dmt)
```

The installed headers are the same for every build and include no GPU
runtime header; `dmt::available_backends()` reports the backends at run time.
Static and shared installs both work: where a static `dmt` needs them, the
package config finds FFTW, OpenMP and the GPU libraries (CUDA Toolkit, or
HIP, hipFFT and rocThrust) for you.

### Using dmt as a dependency

To build `dmt` inside your own project instead of installing it, use
FetchContent, CPM or `add_subdirectory`. As a subproject, `dmt` defaults
`DMT_BUILD_PYTHON` and `DMT_BUILD_TESTING` to `OFF`, so only the library is
built. FFTW 3 (float) and OpenMP must be installed on the system.

```cmake
include(FetchContent)
FetchContent_Declare(
  dmt
  GIT_REPOSITORY https://github.com/pravirkr/dmt.git
  GIT_TAG main            # pin a tag or commit in production
)
set(DMT_GPU OFF CACHE STRING "" FORCE)   # or AUTO / CUDA / HIP
FetchContent_MakeAvailable(dmt)

target_link_libraries(my_pipeline PRIVATE dmt::dmt)
```

With CPM: `CPMAddPackage("gh:pravirkr/dmt#main")`. With a vendored copy:
`add_subdirectory(external/dmt)`. In all cases include `<dmt/dmt.hpp>` (or
`<dmt/algorithms/fdmt.hpp>`), then continue with the
[C++ quickstart](quickstart_cpp.md).

```{note}
For a Python pipeline, install `dmtlib` from source as above (there is no
PyPI wheel yet) and continue with the [Python quickstart](quickstart_python.md).
```

## CMake options

| CMake option | Default | Description |
| :--- | :--- | :--- |
| `DMT_BUILD_PYTHON` | `ON` (top level), `OFF` (subproject) | Build Python bindings via pybind11 |
| `DMT_BUILD_TESTING` | `ON` (top level), `OFF` (subproject) | Build the Catch2 C++ unit tests |
| `DMT_BUILD_BENCHMARKS` | `OFF` | Build the Google Benchmark suites (`dmt_bench`) |
| `DMT_BUILD_DOCS` | `OFF` | Configure Doxygen and Sphinx documentation targets |
| `DMT_GPU` | `AUTO` | GPU backend: `AUTO` (CUDA if nvcc is found, else HIP if ROCm is found), `CUDA`, `HIP` (require one), `OFF`. One per build |
| `DMT_GPU_CHECK_DEVICE` | `ON` | With `DMT_GPU=CUDA`/`HIP`, require a visible GPU at configure time (`OFF` for build-only machines) |
| `DMT_CUDA_ARCHITECTURES` | `native` | CUDA architectures, e.g. `80` or `86;90` (passed to `CMAKE_CUDA_ARCHITECTURES`) |
| `DMT_HIP_ARCHITECTURES` | `gfx90a;gfx942` | AMD architectures (passed to `CMAKE_HIP_ARCHITECTURES`), e.g. `gfx1100` for RDNA3 |
| `DMT_ENABLE_NATIVE_ARCH`| `ON` | `-march=native` in Release builds. Turn `OFF` for binaries or wheels that must run on other CPUs |
| `DMT_CPU_ARCH` | (empty) | Fixed Release `-march` value that overrides native, e.g. `x86-64-v3` (AVX2), `x86-64-v4` (AVX-512) |
| `DMT_ENABLE_IPO` | `OFF` | Link-time optimization in Release builds |
| `DMT_USE_SYSTEM_DEPS` | `OFF` | Prefer compatible system packages over the pinned bundled dependencies (fmt, spdlog) |
| `DMT_ENABLE_COVERAGE` | `OFF` | Coverage instrumentation |
| `BUILD_SHARED_LIBS` | `OFF` | Build shared library (`ON`) or static library (`OFF`) |

```{warning}
`DMT_ENABLE_NATIVE_ARCH` is `ON` by default, so Release builds use
`-march=native`. For binaries that must run on other CPUs (containers, shared
cluster installs, a login node that differs from the compute nodes), set
`DMT_CPU_ARCH` instead, or use the `portable` preset. The SIMD kernels are
chosen at compile time, so `DMT_ENABLE_NATIVE_ARCH=OFF` without `DMT_CPU_ARCH`
builds for the baseline CPU (SSE2 on x86-64) and leaves out the AVX2 and
AVX-512 paths. On build-only machines without a GPU, also set
`DMT_GPU_CHECK_DEVICE=OFF` when you require `DMT_GPU=CUDA` or `HIP`.
```
