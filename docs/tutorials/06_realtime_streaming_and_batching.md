---
jupytext:
  notebook_metadata_filter: kernelspec,-jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial 6: Streaming and multi-beam batching

Telescopes such as CHIME, FAST, MeerKAT and ASKAP produce a continuous stream.
A transient search cannot wait for the whole observation before it
dedisperses. This tutorial covers overlap-save streaming with `mode="valid"`,
moving one engine between beams with `save_history()` / `load_history()`, and
processing several beams per call with `nbeams > 1`.

+++

## Pulses on a block boundary

A streaming pipeline gets data in blocks, for example $N_{\text{samps}} = 1024$
samples at a time. If a pulse arrives near the end of block $k$, its
low-frequency part lands in block $k+1$.

If each block is dedispersed on its own, the pulse is cut in two and a
candidate on the boundary loses S/N or is missed.

### Overlap-save history

In `mode="valid"`, the FDMT engine keeps a short **history** between `execute()`
calls, inside the tree rather than on the raw input. Each block is joined to
the history left by the previous one, so concatenating the outputs of
consecutive blocks gives the same result as one transform of the whole stream,
and a pulse on a boundary is recovered in full.

```{warning}
The first block after construction or `reset_history()` starts from a zero
history: its first `startup_samples` outputs are partial sums. See
[streaming and history](../pipeline_guide/streaming_and_history.md).
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from dmtlib import FDMT
from dmtlib.simulate import DM_CONSTANT, generate_frb

nchans = 128
nsamps_block = 512
f_min = 1200.0
f_max = 1600.0
tsamp = 1e-3
dt_max = 60

fdmt = FDMT(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps_block,
    tsamp=tsamp,
    dt_max=dt_max,
    mode="valid",
)

plan = fdmt.plan
print(f"DM trials: {plan.dmt_ndms}")
print(f"Output samples per block: {plan.dmt_nsamps}")
```

```{code-cell} ipython3
# Simulate a continuous stream across two consecutive blocks.
# Place the pulse near the boundary between block 0 and block 1.
stream_nsamps = nsamps_block * 2
target_delay = 45
dm = target_delay * tsamp / (DM_CONSTANT * (f_min**-2 - f_max**-2))
stream_waterfall = generate_frb(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=stream_nsamps,
    tsamp=tsamp,
    dm=dm,
    amp=3.0,
    offset=nsamps_block - 15,
    noise_rms=1.0,
)

# Split into Block 0 and Block 1
block0 = stream_waterfall[:, :nsamps_block]
block1 = stream_waterfall[:, nsamps_block:]

# Stream through FDMT
fdmt.reset_history()  # Ensure a clean starting state
dmt_out0 = fdmt.execute(block0)
dmt_out1 = fdmt.execute(block1)

# Concatenate outputs
dmt_stream = np.concatenate([dmt_out0, dmt_out1], axis=1)

print(f"Processed Block 0 -> Output shape: {dmt_out0.shape}")
print(f"Processed Block 1 -> Output shape: {dmt_out1.shape}")
print(f"Total reconstructed time samples: {dmt_stream.shape[1]}")
```

```{code-cell} ipython3
# Visualize the continuous reconstructed DMT stream
peak_dm_idx = np.unravel_index(np.argmax(dmt_stream), dmt_stream.shape)[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# DMT Plane
im = ax1.imshow(dmt_stream, aspect="auto", origin="lower", cmap="viridis")
ax1.axvline(dmt_out0.shape[1], color="red", linestyle="--", lw=1.5, label="Block Boundary")
ax1.set_title("Real-Time Streaming Dedispersion Across Block Boundaries", fontsize=13)
ax1.set_ylabel("DM Trial Index", fontsize=11)
ax1.legend(loc="upper right")
fig.colorbar(im, ax=ax1, label="Power")

# 1D Pulse profile across boundary
ax2.plot(dmt_stream[peak_dm_idx, :], color="#2563eb", lw=2)
ax2.axvline(dmt_out0.shape[1], color="red", linestyle="--", lw=1.5, label="Block Boundary")
ax2.set_title("Dedispersed Pulse Profile (Seamless Recovery Across Boundary)", fontsize=12)
ax2.set_xlabel("Continuous Output Time Sample", fontsize=11)
ax2.set_ylabel("Integrated Flux", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()
```

## Parking a stream

`mode="valid"` keeps overlap-save state inside the engine, so one `FDMT` can
time-multiplex several beams:

- **`save_history()`** copies that state out.
- **`load_history()`** puts it back.
- **`reset_history()`** drops the state when the next block is not a
  continuation.

```{code-cell} ipython3
# Replay block 0, park its history, then restore it before block 1.
fdmt.reset_history()
fdmt.execute(block0)
saved = fdmt.save_history()
print(f"Saved history: {saved.size} floats")

fdmt.reset_history()
fdmt.load_history(saved)
dmt_restored = fdmt.execute(block1)
print(f"Original block-1 peak: {dmt_out1.max():.2f}")
print(f"Restored block-1 peak: {dmt_restored.max():.2f}")
```

## Multi-beam batching (`nbeams > 1`)

Interferometers and phased-array feeds form many beams at once: tens for a
phased-array feed, up to about a thousand tied-array beams for MeerKAT.

Instead of creating one FDMT object per beam, pass `nbeams` to the
constructor:

```python
# Configure FDMT instance for 16 simultaneous beams
nbeams = 16
fdmt_multibeam = FDMT(
    f_min=1200.0,
    f_max=1600.0,
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,
    dt_max=100,
    nbeams=nbeams,
    nthreads=8, # Multi-threaded OpenMP execution
)

# Input waterfall must be a 3D NumPy array of shape (nbeams, nchans, nsamps)
batch_input = np.zeros((nbeams, 256, 1024), dtype=np.float32)

# Output shape will be (nbeams, n_delays, n_times)
batch_output = fdmt_multibeam.execute(batch_input)
print("Batch output shape:", batch_output.shape)
```

The plan is built once and shared by all beams; only the state and history
buffers grow with `nbeams`. On CUDA all beams of a level run in one kernel
launch. See [multi-beam batching](../pipeline_guide/multibeam.md).

```{code-cell} ipython3
nbeams = 4
fdmt_batch = FDMT(
    f_min=f_min,
    f_max=f_max,
    nchans=nchans,
    nsamps=nsamps_block,
    tsamp=tsamp,
    dt_max=dt_max,
    nbeams=nbeams,
    mode="valid",
)
batch = np.zeros((nbeams, nchans, nsamps_block), dtype=np.float32)
batch_out = fdmt_batch.execute(batch)
print(f"Batch input {batch.shape} -> output {batch_out.shape}")
```

:::{admonition} Takeaways
:class: dmt-summary

- In `mode="valid"`, FDMT keeps history between `execute()` calls, so the
  outputs of consecutive blocks equal one transform of the whole stream.
- After construction or `reset_history()`, the first `startup_samples`
  outputs are partial sums.
- `save_history()` / `load_history()` let one engine time-multiplex several
  beams.
- With `nbeams > 1` the plan is built once and shared; only the state and
  history buffers grow with `nbeams`.
:::

{doc}`Tutorial 7 <07_fdmt_noise_statistics>` checks the noise statistics of
FDMT and DDMT output. For API details, see the
[Python API reference](../api/python_api.md) and the
[C++ API reference](../api/cpp_api.md).
