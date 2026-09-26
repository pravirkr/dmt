# C++ Quickstart

`dmt` is implemented in modern **C++20** (`std::span`, concepts, `<format>`). This guide shows how to construct an `FDMTCPU`, run it block by block, feed packed low-bit data, use the stepper, and link `dmt` into your own CMake project.

The public API is in `<dmt/dmt.hpp>` (or just `<dmt/algorithms/fdmt.hpp>`); everything lives in `dmt::algorithms` and `dmt::plans`.

---

## 1. Minimal Example

```cpp
#include <iostream>
#include <span>
#include <vector>

#include <dmt/dmt.hpp>

int main() {
    const float f_min    = 1200.0F;  // MHz, bottom edge of the band
    const float f_max    = 1600.0F;  // MHz, top edge of the band
    const size_t nchans  = 256;
    const size_t nsamps  = 1024;     // samples per block
    const float tsamp    = 1e-3F;    // s
    const int32_t dt_max = 128;      // largest delay trial, in samples

    // Construct once: plans the tree and allocates all working memory.
    dmt::algorithms::FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                  /*dt_min=*/0, /*dt_step=*/1,
                                  /*use_box_smearing=*/true, /*mode=*/"valid",
                                  /*verbose=*/0, /*nthreads=*/4);

    const auto& plan = fdmt.get_plan();
    std::cout << plan.get_dmt_ndms() << " DM trials x "
              << plan.get_dmt_nsamps() << " samples per block\n";

    // Input (nchans, nsamps) row-major. The output buffer is
    // get_buffer_size() floats; the result is its first get_dmt_size().
    std::vector<float> waterfall(nchans * nsamps, 0.0F);
    std::vector<float> dmt_buf(plan.get_buffer_size());

    // Per block, contiguous in time ("valid" mode keeps the history):
    fdmt.execute(waterfall, dmt_buf);
    std::span<const float> result(dmt_buf.data(), plan.get_dmt_size());
    // result[i_dm * plan.get_dmt_nsamps() + t]; DM values: plan.get_dm_grid_final()
    return 0;
}
```

---

## 2. Inputs and Outputs

| input | overload | notes |
| :--- | :--- | :--- |
| float32 | `execute(std::span<const float>, std::span<float>)` | `(nbeams, nchans, nsamps)` row-major |
| packed unsigned 1/2/4/8/16-bit | `execute(std::span<const uint8_t>, nbits, std::span<float>)` | each channel row is `ceil(nsamps * nbits / 8)` bytes, samples LSB-first within a byte; the output is identical to the float path on the same values |

- The **output is always float32**. Allocate `nbeams * plan.get_buffer_size()`
  floats. Each beam's first `plan.get_dmt_size()` floats are the
  `(ndms, dmt_nsamps)` result, and beam `b` starts at `b * get_buffer_size()`.
  The rest is scratch. See {ref}`Output Buffers <output-buffers>`
  for why.
- Packed input is the fastest path for ≤ 8-bit data (up to ~2.7× on 8 CPU
  threads; see [Performance](../pipeline_guide/performance.md)). Never convert
  it to float first.
- `mode`: `"valid"` (streaming, `nsamps` out per block), `"full"`
  (`nsamps + dt_max` out, independent blocks) or `"roll"` (circular).
- Custom trial grids: pass a `std::vector<int32_t>` of delays (samples) or a
  `std::vector<float>` of DMs (pc cm⁻³; non-uniform and negative DMs are
  fine) instead of `dt_max, dt_min, dt_step`.
- Invalid sizes, `nbits` values or configurations throw
  `std::invalid_argument`; misuse of the stepper throws `std::logic_error`.
- `execute()` allocates nothing. `get_memory_usage()` reports what the
  constructor allocated.

```cpp
// 2-bit samples, 4 per byte, LSB-first
const auto row_bytes = (nsamps * 2 + 7) / 8;
std::vector<uint8_t> packed(nchans * row_bytes);
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt_buf);
```

---

## 3. Stepper Interface

To inspect intermediate sub-bands, for example to stop 2 levels before the root for a sub-band search:

```cpp
fdmt.reset(waterfall, dmt_buf);          // consumes one block; level 0
fdmt.advance_until_remaining(2);         // 4 sub-bands left
for (size_t s = 0; s < fdmt.num_subbands(); ++s) {
    const auto sub = fdmt.view_subband(s);   // zero-copy: data, ndt, nsamps,
                                             // f_start, f_end, dt_grid
}
fdmt.finalize();                         // root; result in dmt_buf as above
```

In `"valid"` mode every block must be finalized before the next
`reset()`/`execute()` (else `std::logic_error`). The stepper never fuses
levels. See {ref}`Stepper Rules <stepper-rules>`.

---

## 4. Streaming, History and GPUs

- In `"valid"` mode, consecutive calls on contiguous, non-overlapping blocks
  give contiguous output. `reset_history()` starts a new stream. To
  multiplex several streams on one engine, use
  `history_state_size()`/`save_history()`/`load_history()`. See
  [Streaming](../pipeline_guide/streaming_and_history.md).
- `FDMTCUDA` has the same constructor (with `device_id` in place of
  `nthreads`) and API. Its `execute()` takes host spans (staged internally)
  or device `cuda::std::span`s plus a `cudaStream_t`. The device path is
  asynchronous, and its output buffer is also `nbeams * get_buffer_size()`
  floats of device memory.

```cpp
dmt::algorithms::FDMTCUDA gpu(f_min, f_max, nchans, nsamps, tsamp, dt_max);
// d_wf: nchans * nsamps floats, d_dmt: get_buffer_size() floats, on the device
gpu.execute(cuda::std::span<const float>(d_wf, nchans * nsamps),
            cuda::std::span<float>(d_dmt, gpu.get_plan().get_buffer_size()),
            stream);
cudaStreamSynchronize(stream);
```

---

## 5. CMake Setup

After `cmake --install`, point `CMAKE_PREFIX_PATH` at the install prefix:

```cmake
cmake_minimum_required(VERSION 3.20)
project(dmt_example CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(dmt REQUIRED)
add_executable(dmt_example main.cpp)
target_link_libraries(dmt_example PRIVATE dmt::dmt)
```

To build `dmt` as part of your project instead (FetchContent, CPM or
`add_subdirectory`), see
[Using dmt as a dependency](installation.md#using-dmt-as-a-dependency).
