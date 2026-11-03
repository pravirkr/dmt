# C++ quickstart

`dmt` is written in C++20 (`std::span`, concepts, `<format>`). This page shows
how to construct an `FDMT`, run it block by block, feed it packed low-bit
data, use the stepper, and link `dmt` into your own CMake project.

The public API is in `<dmt/dmt.hpp>` (or just `<dmt/algorithms/fdmt.hpp>`).
The engines live in `dmt::algorithms`, the plans in `dmt::plans`, and backend
selection (`dmt::Exec`) in `dmt`.

## Minimal example

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
    dmt::algorithms::FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                               /*dt_min=*/0, /*dt_step=*/1,
                               /*use_box_smearing=*/true, /*mode=*/"valid",
                               dmt::Exec::cpu(/*nthreads=*/4));

    const auto& plan = fdmt.get_plan();
    std::cout << plan.get_dmt_ndms() << " DM trials x "
              << plan.get_dmt_nsamps() << " samples per block\n";

    // Input (nchans, nsamps) row-major; output (ndms, dmt_nsamps) row-major,
    // exactly get_dmt_size() floats.
    std::vector<float> waterfall(nchans * nsamps, 0.0F);
    std::vector<float> dmt_buf(plan.get_dmt_size());

    // Per block, contiguous in time ("valid" mode keeps the history):
    fdmt.execute(waterfall, dmt_buf);
    // dmt_buf[i_dm * plan.get_dmt_nsamps() + t]; DM values: plan.get_dm_grid_final()
    return 0;
}
```

## Inputs and outputs

| input | overload | notes |
| :--- | :--- | :--- |
| float32 | `execute(std::span<const float>, std::span<float>)` | `(nbeams, nchans, nsamps)` row-major |
| packed unsigned 1/2/4/8/16-bit | `execute(std::span<const uint8_t>, nbits, std::span<float>)` | each channel row is `ceil(nsamps * nbits / 8)` bytes, samples LSB-first within a byte; the output equals the float path on the same values, up to the rounding of the level-0 box means with box smearing (see {ref}`box means <box-mean>`) |

- **Output layout.** The output is always float32, laid out `(nbeams, ndms, dmt_nsamps)`:
  allocate `nbeams * plan.get_dmt_size()` floats; beam `b` starts at
  `b * get_dmt_size()`. The engine keeps all its working state internally and
  only writes the output. See {ref}`output buffers <output-buffers>`.
- **Units.** With `use_box_smearing=true` (the default) each channel contributes the
  *mean* of its level-0 box, so the output is in box-mean units. See
  {ref}`Box means <box-mean>`.
- **Packed input.** Packed input is the fastest path for ≤ 8-bit data (up to ~2.7× on 8 CPU
  threads). Never convert
  it to float first.
- **Modes.** `mode`: `"valid"` (streaming, `nsamps` out per block), `"full"`
  (`nsamps` plus the largest delay magnitude out, independent blocks) or `"roll"` (circular).
- **Custom trial grids.** Pass a `std::vector<int32_t>` of delays (samples) or a
  `std::vector<float>` of DMs (pc cm⁻³; non-uniform and negative DMs are
  fine) instead of `dt_max, dt_min, dt_step`.
- **Errors.** Invalid sizes, `nbits` values or configurations throw
  `std::invalid_argument`; misuse of the stepper throws `std::logic_error`.
- **Memory.** `execute()` allocates nothing. `get_memory_usage()` reports what the
  constructor allocated. In `"valid"` mode the tree runs in chunks of
  `get_chunk_nsamps()` samples (the trailing `chunk_nsamps` constructor
  argument, `0` = automatic), which bounds the state at high DM without
  changing the output.

```cpp
// 2-bit samples, 4 per byte, LSB-first
const auto row_bytes = (nsamps * 2 + 7) / 8;
std::vector<uint8_t> packed(nchans * row_bytes);
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt_buf);
```

## Stepper interface

The stepper lets you inspect intermediate sub-bands, for example to stop 2
levels before the root for a sub-band search:

```cpp
fdmt.reset(waterfall, dmt_buf);          // consumes one block; level 0
fdmt.advance_until_remaining(2);         // 4 sub-bands left
for (size_t s = 0; s < fdmt.num_subbands(); ++s) {
    const auto sub = fdmt.view_subband(s);   // zero-copy: data, ndt, nsamps,
                                             // f_start, f_end, dt_grid
}
fdmt.finalize();                         // root; result in dmt_buf as above
```

```{note}
In `"valid"` mode every block must be finalized before the next
`reset()`/`execute()` (else `std::logic_error`). The stepper never fuses
levels, and runs whole blocks: it needs `get_chunk_nsamps() == nsamps`
(construct with `chunk_nsamps = nsamps`), else `std::logic_error`. See
{ref}`stepper rules <stepper-rules>`.
```

## Streaming, history and GPUs

- **Streaming.** In `"valid"` mode, consecutive calls on contiguous, non-overlapping blocks
  give contiguous output. `reset_history()` starts a new stream. To
  multiplex several streams on one engine, use
  `history_state_size()`/`save_history()`/`load_history()`. See
  [Streaming](../pipeline_guide/streaming_and_history.md).
- **GPUs.** The same class runs on a GPU: pass `dmt::Exec::cuda(device)` (or `dmt::Exec::hip(device)` on AMD) in place of
  `Exec::cpu(nthreads)` (check `dmt::available_backends()` first; a backend
  missing from the build throws `std::invalid_argument`). Host `std::span`s
  work on every backend (a GPU stages them and blocks). Device memory goes
  through `dmt::DeviceSpan` plus an optional `dmt::Stream` (a `cudaStream_t`
  converts to it); the device path is asynchronous, and its output buffer is
  also `nbeams * get_dmt_size()` floats of device memory. The public
  headers include no CUDA header, so a CPU-only and a CUDA build of `dmt`
  expose the same declarations.

```cpp
dmt::algorithms::FDMT gpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1,
                          true, "valid", dmt::Exec::cuda(/*device=*/0));
// d_wf: nchans * nsamps floats, d_dmt: get_dmt_size() floats, on the device
gpu.execute(dmt::DeviceSpan<const float>(d_wf, nchans * nsamps),
            dmt::DeviceSpan<float>(d_dmt, gpu.get_plan().get_dmt_size()),
            stream);
cudaStreamSynchronize(stream);
```

## CMake setup

After `cmake --install`, point `CMAKE_PREFIX_PATH` at the install prefix:

```cmake
cmake_minimum_required(VERSION 3.20)
project(dmt_example CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(dmt REQUIRED)
add_executable(dmt_example main.cpp)
target_link_libraries(dmt_example PRIVATE dmt::dmt)
```

```{seealso}
To build `dmt` as part of your project instead (FetchContent, CPM or
`add_subdirectory`), see
[Using dmt as a dependency](installation.md#using-dmt-as-a-dependency).
```
