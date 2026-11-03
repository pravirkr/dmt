# Python quickstart

The `dmtlib` package provides high-level convenience functions as well as object-oriented engine classes for fine-grained pipeline control.

---

## One-shot dedispersion with `compute_fdmt`

To dedisperse a single 2D waterfall of shape `(nchans, nsamps)`, call
`compute_fdmt`:

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
    dt_min=0,           # Minimum delay (may be negative)
    mode="valid",       # 'valid', 'full', or 'roll'
)

# Reshape into the 2D DM-time plane: (n_dm, n_times)
dmt_plane = dmt_matrix.reshape(plan.dmt_ndms, plan.dmt_nsamps)

print("DMT shape:", dmt_plane.shape)
print("Physical DM trials (pc cm^-3):", plan.dm_grid_final)
```

---

## The `FDMT` engine

In a loop over blocks, construct an `FDMT` once so the tree is planned once
rather than on every block:

```python
from dmtlib import FDMT

# Construct once
fdmt = FDMT(
    f_min=1200.0,
    f_max=1600.0,
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,
    dt_max=150,
    mode="valid",
    nthreads=4,         # Number of OpenMP threads
    # backend="cuda",   # run the same engine on a GPU (CUDA builds)
)

# Reuse one output buffer for every block (no per-call allocation)
out = np.empty(fdmt.nbeams * fdmt.plan.dmt_size, dtype=np.float32)

# Successive blocks, contiguous and non-overlapping in time
for block_id in range(10):
    waterfall = get_next_block()            # (256, 1024), float32
    dmt_plane = fdmt.execute(waterfall, out=out)
    # (n_dm, n_times) float32 view into `out`, overwritten by the next call
    process_candidates(dmt_plane)
```

```{warning}
`dmt_plane` is a view into `out`, overwritten by the next call. `.copy()` the
results you keep.
```

In `mode="valid"` the engine keeps a per-node history, so the output blocks
join into one continuous DM-time stream. Call
`fdmt.reset_history()` when the input stream breaks. See
[Streaming](../pipeline_guide/streaming_and_history.md).

### Inputs, outputs and errors

| input | call | notes |
| :--- | :--- | :--- |
| float32 `(nchans, nsamps)` or `(nbeams, nchans, nsamps)` | `execute(waterfall)` | other float dtypes are cast (a copy); arrays must be C-contiguous |
| packed `uint8` `(nchans, row_bytes)` or `(nbeams, nchans, row_bytes)` | `execute(packed, nbits)` | `nbits` ∈ {1, 2, 4, 8, 16}; `row_bytes = ceil(nsamps * nbits / 8)`; samples LSB-first within a byte |

- **Output shape.** The result is always float32 with shape `(n_dm, n_times)` or
  `(nbeams, n_dm, n_times)`. `n_dm = plan.dmt_ndms` (DM values:
  `fdmt.dm_grid_final`) and `n_times = plan.dmt_nsamps`.
- **Output buffer.** The returned array is a zero-copy view of an output buffer of exactly
  `nbeams * plan.dmt_size` floats (or of `out=`, which needs at least that
  many). Pass `out=` to reuse one buffer, and `.copy()` the results you keep
  from it. See {ref}`output buffers <output-buffers>`.
- **Units.** With `use_box_smearing=True` (the default) each channel contributes the
  *mean* of its level-0 box (the matched filter for intra-channel smearing),
  so the output is in box-mean units. Divide by
  `fdmt.get_effective_sigma_grid()` for S/N; see
  [noise calibration](../pipeline_guide/noise_calibration.md).
- **Errors.** Misuse raises an exception: wrong sizes or `nbits` raise `ValueError`, a
  `uint8` array without `nbits` raises `TypeError`, and starting a new
  `valid`-mode block before finishing a stepped one raises `RuntimeError`.
- **Stepper.** To inspect intermediate sub-bands, use the stepper (`reset`, `advance`,
  `advance_until_remaining`, `view_subband`, `finalize`) on an engine built
  with `chunk_nsamps=nsamps`; see {ref}`stepper rules <stepper-rules>`.
- **Chunked execution.** In `mode="valid"`, `execute()` runs the tree in chunks of
  `fdmt.chunk_nsamps` samples (keyword `chunk_nsamps=None`, automatic), which
  bounds its memory at high DM without changing the output.
- **GPU.** On a GPU, pass `backend="cuda"` (and `device=` for another GPU); the
  arguments and the `execute`/stepper calls on NumPy arrays are the same.
  `dmtlib.available_backends()` lists the backends in your build, e.g.
  `['cpu', 'cuda']`.

---

## Custom and sparse trial grids

Instead of the dense integer range `[0, dt_max]`, you can pass your own delay
or DM grid. A sparse grid saves compute and can be matched to the
intra-channel dispersion smearing:

```python
import numpy as np
from dmtlib import FDMTPlan, FDMT

# Geometrically spaced delay trials
custom_dt = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=int)

fdmt_sparse = FDMT(
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

## Multi-beam batching

Beams that share one plan can be processed in a single call:

```python
nbeams = 8

fdmt_beams = FDMT(
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

## Packed low-bit input

You can pass low-bit digitiser output packed (LSB-first within each byte,
rows padded to a whole byte), and 1- to 4-bit input runs ~1.8-2.7x faster on
8 CPU threads. The result equals passing the same values as float, except
that with box smearing each level-0 box mean is rounded to a few fractional
bits (`plan.box_mean_bits`; S/N loss ≤ 0.5%):

```python
nbits = 2
packed = np.random.randint(0, 256, size=(256, 1024 * nbits // 8), dtype=np.uint8)
dmt_plane = fdmt.execute(packed, nbits)  # float32 (n_dm, n_times)
```

```{tip}
The execution defaults (automatic level fusion, narrow-integer tree for
packed input) are already the fastest settings.
```
