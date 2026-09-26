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

# Reuse one output buffer for every block (no per-call allocation)
out = np.empty(fdmt.nbeams * fdmt.plan.buffer_size, dtype=np.float32)

# Successive blocks, contiguous and non-overlapping in time
for block_id in range(10):
    waterfall = get_next_block()            # (256, 1024), float32
    dmt_plane = fdmt.execute(waterfall, out=out)
    # (n_dm, n_times) float32 view into `out`, overwritten by the next call
    process_candidates(dmt_plane)
```

In `mode="valid"` the engine keeps a per-node history, so the output blocks
join seamlessly into one continuous DM-time stream. Call
`fdmt.reset_history()` when the input stream breaks. See
[Streaming](../pipeline_guide/streaming_and_history.md).

### Inputs, outputs and errors

| input | call | notes |
| :--- | :--- | :--- |
| float32 `(nchans, nsamps)` or `(nbeams, nchans, nsamps)` | `execute(waterfall)` | other float dtypes are cast (a copy); arrays must be C-contiguous |
| packed `uint8` `(nchans, row_bytes)` or `(nbeams, nchans, row_bytes)` | `execute(packed, nbits)` | `nbits` ∈ {1, 2, 4, 8, 16}; `row_bytes = ceil(nsamps * nbits / 8)`; samples LSB-first within a byte |

- The result is always float32 with shape `(n_dm, n_times)` or
  `(nbeams, n_dm, n_times)`. `n_dm = plan.dmt_ndms` (DM values:
  `fdmt.dm_grid_final`) and `n_times = plan.dmt_nsamps`.
- The returned array is a zero-copy view whose base is the engine-sized
  output buffer (`plan.buffer_size` floats per beam). Pass `out=` to reuse
  one buffer, or `.copy()` results you keep. See
  {ref}`Output Buffers <output-buffers>`.
- Misuse fails loudly: wrong sizes or `nbits` raise `ValueError`, a
  `uint8` array without `nbits` raises `TypeError`, and starting a new
  `valid`-mode block before finishing a stepped one raises `RuntimeError`.
- To inspect intermediate sub-bands, use the stepper (`reset`, `advance`,
  `advance_until_remaining`, `view_subband`, `finalize`); see
  {ref}`Stepper Rules <stepper-rules>`.
- On a GPU, `FDMTCUDA` (`from dmtlib import FDMTCUDA`, present when dmt is
  built with CUDA) takes the
  same arguments, with `device_id` in place of `nthreads`, and the same
  `execute`/stepper calls on host arrays.

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

Process $N_{\text{beams}}$ beams that share one plan in a single call:

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

---

## 5. Packed Low-Bit Input

Low-bit digitiser output can be passed packed (LSB-first within each byte,
rows padded to a whole byte); the result is identical to passing the same
values as float, and 1- to 4-bit input runs ~1.8-2.7x faster on 8 CPU threads:

```python
nbits = 2
packed = np.random.randint(0, 256, size=(256, 1024 * nbits // 8), dtype=np.uint8)
dmt_plane = fdmt.execute(packed, nbits)  # float32 (n_dm, n_times)
```

The execution defaults (automatic level fusion, narrow-integer tree for
packed input) are already the fastest settings; see
[Performance Tuning](../pipeline_guide/performance.md).
