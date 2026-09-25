# Multi-Beam Batching

Modern radio interferometers (such as MeerKAT, CHIME, ASKAP, and SKA) generate dozens to thousands of simultaneous tied-array or synthesized beams to survey wide fields of view.

`dmt` supports native **Multi-Beam Batching** across CPU SIMD and CUDA GPU kernels.

---

## 1. Why Batch Across Beams?

Instead of instantiating $N_{\text{beams}}$ separate engine objects or looping over single-beam calls, batching provides major advantages:

1. **Amortized Plan Overhead**: The plan (coordinate DAG, subband offsets, tree indices) is built once and shared by all beams; only the state and history buffers scale with `nbeams`.
2. **GPU Occupancy**: On CUDA, every beam of a level runs in the same kernel launch (beams are a grid dimension), so even small blocks keep the GPU busy.
3. **One Engine, One Thread Pool**: On the CPU, beams are processed one after another, each spread over all `nthreads` threads with the same (fused, vectorized) kernels as a single beam.

---

## 2. Using Multi-Beam Batching in Python

Set `nbeams > 1` in the engine constructor. The engine expects 3D inputs and produces 3D outputs:

```python
import numpy as np
from dmtlib import FDMTCPU

nbeams = 16
nchans = 256
nsamps = 1024

# Instantiate multi-beam engine
fdmt = FDMTCPU(
    f_min=1200.0,
    f_max=1600.0,
    nchans=nchans,
    nsamps=nsamps,
    tsamp=1e-3,
    dt_max=100,
    nbeams=nbeams,
    nthreads=8,  # OpenMP worker threads
)

# Input: 3D array of shape (nbeams, nchans, nsamps)
batch_waterfall = np.random.normal(0, 1, size=(nbeams, nchans, nsamps)).astype(np.float32)

# Output: 3D array of shape (nbeams, n_delays, n_times)
batch_dmt = fdmt.execute(batch_waterfall)

print(f"Batch waterfall input: {batch_waterfall.shape}")
print(f"Batch DMT output:      {batch_dmt.shape}")
```

---

## 3. Using Multi-Beam Batching in C++

```cpp
#include <dmt/dmt.hpp>

const size_t nbeams = 16;
const size_t nchans = 256;
const size_t nsamps = 1024;

dmt::algorithms::FDMTCPU fdmt(
    1200.0f, 1600.0f, nchans, nsamps, 1e-3f,
    /*dt_max=*/100, /*dt_min=*/0, /*dt_step=*/1,
    /*use_box_smearing=*/true, "valid", /*verbose=*/false,
    /*nthreads=*/8, /*nbeams=*/nbeams
);

const auto& plan = fdmt.get_plan();
std::vector<float> batch_input(nbeams * nchans * nsamps);
std::vector<float> batch_output(nbeams * plan.get_buffer_size());

fdmt.execute(
    std::span<const float>(batch_input.data(), batch_input.size()),
    std::span<float>(batch_output.data(), batch_output.size())
);
```
