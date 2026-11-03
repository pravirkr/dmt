# Multi-beam batching

Interferometers such as MeerKAT, CHIME, ASKAP and SKA form dozens to
thousands of tied-array or synthesized beams at once. `dmt` engines can take
all of them in one call, on the CPU and the GPU: one engine, one plan,
`nbeams` beams per call.

## Why batch across beams

Compared with one engine per beam, or a loop of single-beam calls:

- **One plan.** The plan (coordinate DAG, subband offsets, tree indices) is built once and
  shared by all beams. Only the state and history buffers scale with
  `nbeams`.
- **CUDA.** On CUDA, every beam of a level runs in the same kernel launch (beams are a
  grid dimension), so even small blocks keep the GPU busy.
- **CPU.** On the CPU, one thread pool serves all beams. Beams are processed one after
  another, each spread over all `nthreads` threads with the same fused,
  vectorized kernels as a single beam.

## Python

Set `nbeams > 1` in the engine constructor. The engine then takes 3D input
and returns 3D output:

```python
import numpy as np
from dmtlib import FDMT

nbeams = 16
nchans = 256
nsamps = 1024

# Multi-beam engine
fdmt = FDMT(
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

## C++

```cpp
#include <dmt/dmt.hpp>

const size_t nbeams = 16;
const size_t nchans = 256;
const size_t nsamps = 1024;

dmt::algorithms::FDMT fdmt(
    1200.0f, 1600.0f, nchans, nsamps, 1e-3f,
    /*dt_max=*/100, /*dt_min=*/0, /*dt_step=*/1,
    /*use_box_smearing=*/true, "valid",
    dmt::Exec::cpu(/*nthreads=*/8), /*nbeams=*/nbeams
);

const auto& plan = fdmt.get_plan();
std::vector<float> batch_input(nbeams * nchans * nsamps);
std::vector<float> batch_output(nbeams * plan.get_dmt_size());

fdmt.execute(
    std::span<const float>(batch_input.data(), batch_input.size()),
    std::span<float>(batch_output.data(), batch_output.size())
);

// Output (nbeams, ndms, dmt_nsamps): beam b's result is the get_dmt_size()
// floats at offset b * get_dmt_size().
std::span<const float> beam3(batch_output.data() + 3 * plan.get_dmt_size(),
                             plan.get_dmt_size());
```

The output buffer is exactly the transform; the engine keeps its tree state
and history internally (see {ref}`output buffers <output-buffers>`).

```{note}
The tree state scales with `nbeams`, so the automatic `chunk_nsamps` (valid
mode) is chosen for all beams together and may be shorter than for one beam
(the chunk is set by the plan's memory budget).
```

```{seealso}
- {ref}`Output buffers <output-buffers>`: layout and reuse of the output buffer.
- [Streaming and history](streaming_and_history.md): history kept per beam, and
  `save_history` / `load_history` for beams served in turn.
```
