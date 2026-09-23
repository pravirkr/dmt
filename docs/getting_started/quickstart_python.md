# Python Quickstart

The `dmtlib` package provides high-level convenience functions as well as object-oriented engine classes for fine-grained pipeline control.

---

## 1. One-Shot Dedispersion: `compute_fdmt`

The quickest way to dedisperse a 2D waterfall matrix `(nchans, nsamps)` is using `compute_fdmt`:

```python
import numpy as np
from dmtlib import compute_fdmt

# Create dummy waterfall (256 channels, 1024 time samples)
waterfall = np.random.normal(0, 1, size=(256, 1024)).astype(np.float32)

# Run FDMT
dmt_matrix, plan = compute_fdmt(
    waterfall,
    f_min=1200.0,       # Bottom of frequency band (MHz)
    f_max=1600.0,       # Top of frequency band (MHz)
    nchans=256,         # Number of frequency channels
    nsamps=1024,        # Number of time samples in block
    tsamp=1e-3,         # Sampling interval in seconds (1 ms)
    dt_max=150,         # Maximum delay to search (in samples)
    dt_min=0,           # Minimum delay (can be negative!)
    mode="valid",       # 'valid', 'full', or 'roll'
)

# Reshape into 2D DM-time plane: (N_dm_trials, N_time_samples)
dmt_plane = dmt_matrix.reshape(plan.dmt_ndms, plan.dmt_nsamps)

print("Transform complete!")
print("DMT shape:", dmt_plane.shape)
print("Physical DM trials (pc cm^-3):", plan.dm_grid_final)
```

---

## 2. Object-Oriented Engine: `FDMTCPU`

For production loops and streaming pipelines, pre-planning avoids recomputing tree indices on every block. Use the `FDMTCPU` class:

```python
from dmtlib import FDMTCPU

# Initialize execution engine once
fdmt = FDMTCPU(
    f_min=1200.0,
    f_max=1600.0,
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,
    dt_max=150,
    mode="valid",
    nthreads=4,         # Number of OpenMP threads
)

# Process successive incoming data blocks
for block_id in range(10):
    waterfall = get_next_block() # shape: (256, 1024), float32
    dmt_plane = fdmt.execute(waterfall)
    # Output shape: (n_dm, n_times)
    process_candidates(dmt_plane)
```

---

## 3. Custom & Sparse Trial Grids

Instead of searching a dense linear integer range `[0, dt_max]`, specify custom delay or DM grids to save compute and match intra-channel dispersion smearing:

```python
import numpy as np
from dmtlib import FDMTPlan, FDMTCPU

# Non-linear geometrically spaced delay trials
custom_dt = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=int)

fdmt_sparse = FDMTCPU(
    f_min=1200.0,
    f_max=1600.0,
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,
    dt_grid=custom_dt,
)

print("Active delay trials:", fdmt_sparse.plan.dt_grid_final)
```

---

## 4. Multi-Beam Batching

Process $N_{\text{beams}}$ simultaneously using vectorized SIMD loops:

```python
nbeams = 8

fdmt_beams = FDMTCPU(
    f_min=1200.0,
    f_max=1600.0,
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,
    dt_max=100,
    nbeams=nbeams,
)

# 3D input: (nbeams, nchans, nsamps)
batch_input = np.random.normal(0, 1, size=(nbeams, 256, 1024)).astype(np.float32)

# 3D output: (nbeams, n_delays, n_times)
batch_output = fdmt_beams.execute(batch_input)
print("Batch output shape:", batch_output.shape)
```
