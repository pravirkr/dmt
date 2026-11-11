# Multi-beam batching

Set `nbeams` in the constructor and one engine processes every beam in each
call, on the CPU and the GPU.

- **One plan.** It is built once and shared; only state and history scale with
  `nbeams`.
- **GPU.** All beams of a level run in one kernel launch, so even small blocks
  keep the GPU busy.
- **CPU.** Beams run one after another, each across all `nthreads` threads.

## Python

```python
import numpy as np
from dmtlib import FDMT

fdmt = FDMT(f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
            tsamp=1e-3, dt_max=100, nbeams=16, nthreads=8)

waterfall = np.random.normal(size=(16, 256, 1024)).astype(np.float32)
dmt = fdmt.execute(waterfall)   # (nbeams, n_dm, n_times)
```

## C++

```cpp
#include <dmt/dmt.hpp>

dmt::algorithms::FDMT fdmt(
    1200.0f, 1600.0f, /*nchans=*/256, /*nsamps=*/1024, 1e-3f,
    /*dt_max=*/100, /*dt_min=*/0, /*dt_step=*/1,
    /*use_box_smearing=*/true, "valid",
    dmt::Exec::cpu(/*nthreads=*/8), /*nbeams=*/16);

const size_t beam_size = fdmt.get_plan().get_dmt_size();
std::vector<float> in(16 * 256 * 1024);
std::vector<float> out(16 * beam_size);
fdmt.execute(std::span<const float>(in), std::span<float>(out));

std::span<const float> beam3(out.data() + 3 * beam_size, beam_size);
```

The output is laid out (nbeams, n_dm, n_times); see
{ref}`output buffers <output-buffers>`.

```{note}
The tree state scales with `nbeams`, so the automatic `chunk_nsamps` may be
shorter than for one beam.
```

To serve beams in turn rather than together, see `save_history` /
`load_history` in [streaming and history](streaming_and_history.md).
