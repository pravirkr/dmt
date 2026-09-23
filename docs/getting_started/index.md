# Getting Started

Welcome to **dmt**! This section guides you through installing the library (both the Python package and C++ headers/library) and running your first dedispersion transforms.

```{toctree}
:maxdepth: 1

installation
quickstart_python
quickstart_cpp
example
```

---

## 30-Second Quick Example (Python)

```python
import numpy as np
from dmtlib import compute_fdmt
from dmtlib.simulate import generate_frb

# 1. Generate a synthetic dispersed Fast Radio Burst (FRB)
waterfall = generate_frb(
    f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dm=150.0, amp=3.0, offset=200, noise_rms=1.0,
)

# 2. Dedisperse using the Fast Dispersion Measure Transform
dmt, plan = compute_fdmt(
    waterfall, f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dt_max=180, mode="valid"
)

# 3. Locate the candidate in the DM-time plane
dmt_plane = dmt.reshape(plan.dmt_ndms, plan.dmt_nsamps)
peak_dm_idx, peak_time = np.unravel_index(np.argmax(dmt_plane), dmt_plane.shape)
print(f"Detected FRB at DM trial: {plan.dm_grid_final[peak_dm_idx]:.2f} pc cm^-3, time: {peak_time}")
```

---

## Next Steps

- Follow the [Installation Guide](installation.md) for prerequisite toolchains, FFTW3, OpenMP, and CUDA build instructions.
- Check the [Python Quickstart](quickstart_python.md) for object-oriented stepper usage and batch execution.
- Check the [C++ Quickstart](quickstart_cpp.md) for integrating `dmt::algorithms::FDMTCPU` directly into real-time C++ telescope pipelines.
- The {doc}`worked example <example>` draws the waterfall and DM-time figures on the README.
