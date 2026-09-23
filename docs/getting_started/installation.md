# Installation Guide

`dmt` can be used as a Python library (`dmtlib`), a C++20 header/library dependency via CMake `find_package(dmt)`, or both.

---

## 1. System Requirements

### Hardware

- **CPU**: x86_64 with AVX2/AVX-512 support, or Apple Silicon (ARM64 with NEON).
- **GPU (Optional)**: NVIDIA GPU with Compute Capability 7.0+.

### Compilers & Dependencies

- **C++ Compiler**: Modern C++20 compliant compiler:
  - GCC 13+
  - Clang 15+
- **Build System**: CMake 3.18+ and Ninja or Make.
- **Python**: Python 3.12 or newer.
- **System Libraries**:
  - `libfftw3-dev` (FFTW 3.3+)
  - OpenMP runtime (`libomp` on macOS, standard in GCC on Linux)
  - CUDA Toolkit 12.6+ (Optional, for GPU acceleration)

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libfftw3-dev libomp-dev
```

On macOS (using Homebrew):

```bash
brew install cmake ninja fftw libomp
```

---

## 2. Python Package Installation

### Installing from Source with `uv` or `pip`

Clone the repository and install into your active Python environment:

```bash
git clone https://github.com/pravirkr/dmt.git
cd dmt

# Using pip with scikit-build-core
uv pip install .

# Or editable development mode with development extras
uv pip install -e ".[develop,docs,tests]"
```

### CUDA-Enabled Python Wheels

To force building Python bindings with CUDA GPU support enabled:

```bash
CMAKE_ARGS="-DDMT_CUDA=ON" uv pip install .
```

To verify Python installation:

```python
import dmtlib

print(f"dmtlib version: {dmtlib.__version__}")
```

---

## 3. C++ Library Integration (CMake)

### Building and Installing C++ Library

To build and install the C++20 static/shared library:

```bash
mkdir build && cd build

# Configure CPU-only build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDMT_BUILD_TESTING=ON

# Or configure with CUDA GPU acceleration
cmake .. -DCMAKE_BUILD_TYPE=Release -DDMT_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native

# Build and run unit tests
cmake --build . -j
ctest --output-on-failure

# Install to system or prefix
cmake --install .
```

### Using `find_package(dmt)` in Your Project

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

If `dmt` was built with CUDA support, the target automatically propagates necessary CUDA flags and include directories.

---

## 4. CMake Build Options Reference

| CMake Option | Default | Description |
| :--- | :--- | :--- |
| `DMT_BUILD_PYTHON` | `ON` | Build Python bindings via pybind11 |
| `DMT_BUILD_TESTING` | `ON` | Build GoogleTest C++ unit test suite |
| `DMT_BUILD_BENCHMARKS` | `OFF` | Build Google Benchmark performance suites |
| `DMT_BUILD_DOCS` | `OFF` | Configure Doxygen and Sphinx documentation targets |
| `DMT_CUDA` | `AUTO` | CUDA support mode: `AUTO` (detect GPU), `ON` (require GPU), `OFF` (CPU-only) |
| `DMT_ENABLE_NATIVE_ARCH`| `ON` | Enable `-march=native` compiler optimizations |
| `BUILD_SHARED_LIBS` | `OFF` | Build shared library (`ON`) or static library (`OFF`) |
