# C++ Quickstart

`dmt` is implemented in modern **C++20** (`std::span`, concepts, `<format>`). This guide demonstrates how to instantiate an `FDMTPlan`, run `FDMTCPU` on contiguous memory blocks, and link `dmt` into a real-time C++ telescope backend.

---

## 1. Minimal C++ Example

```cpp
#include <iostream>
#include <vector>
#include <span>

#include <dmt/dmt.hpp>

int main() {
    const float f_min    = 1200.0f;  // MHz
    const float f_max    = 1600.0f;  // MHz
    const size_t nchans  = 256;
    const size_t nsamps  = 1024;
    const float tsamp    = 1e-3f;    // 1 ms
    const int32_t dt_max = 128;
    const int32_t dt_min = 0;

    // 1. Initialize the FDMT engine
    dmt::algorithms::FDMTCPU fdmt(
        f_min, f_max, nchans, nsamps, tsamp,
        dt_max, dt_min, /*dt_step=*/1,
        /*use_box_smearing=*/true,
        /*mode=*/"valid",
        /*verbose=*/true,
        /*nthreads=*/4,
        /*nbeams=*/1
    );

    const auto& plan = fdmt.get_plan();
    std::cout << "FDMT initialized: " << plan.get_niters() << " tree levels, "
              << plan.get_dmt_ndms() << " DM trials, "
              << plan.get_dmt_nsamps() << " output time samples.\n";

    // 2. Prepare input waterfall and output buffer
    std::vector<float> waterfall(nchans * nsamps, 0.0f);
    std::vector<float> dmt_out(plan.get_buffer_size(), 0.0f);

    // (Fill waterfall with channelized telescope samples here...)

    // 3. Execute transform
    fdmt.execute(
        std::span<const float>(waterfall.data(), waterfall.size()),
        std::span<float>(dmt_out.data(), dmt_out.size())
    );

    std::cout << "Dedispersion transform complete!\n";
    return 0;
}
```

---

## 2. Using the Stepper Interface in C++

When intermediate subband states need to be inspected (e.g. for subband RFI detection or hardware telemetry):

```cpp
// Reset stepper with input and output storage
fdmt.reset(
    std::span<const float>(waterfall.data(), waterfall.size()),
    std::span<float>(dmt_out.data(), dmt_out.size())
);

// Advance level by level
while (!fdmt.is_finished()) {
    std::cout << "Current level: " << fdmt.current_level()
              << " (" << fdmt.num_subbands() << " active subbands)\n";

    // Inspect subband 0 at this level
    dmt::algorithms::FDMTSubbandView sub0 = fdmt.view_subband(0);
    std::cout << "  Subband 0 delay trials: " << sub0.ndt
              << ", samples: " << sub0.nsamps << "\n";

    fdmt.advance(1);
}

// Finalize transform
fdmt.finalize();
```

---

## 3. Streaming and History Buffers

In `"valid"` mode, `FDMTCPU` retains intermediate shift history across successive blocks. For time-multiplexing independent beams:

```cpp
// Query size needed for history state
const size_t hist_size = fdmt.history_state_size();
std::vector<float> saved_history(hist_size);

// Save state
fdmt.save_history(std::span<float>(saved_history));

// Later, restore state for this stream
fdmt.load_history(std::span<const float>(saved_history));
```

---

## 4. CMake Setup

Create a `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(dmt_example CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(dmt REQUIRED)

add_executable(dmt_example main.cpp)
target_link_libraries(dmt_example PRIVATE dmt::dmt)
```
