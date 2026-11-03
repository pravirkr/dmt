# Quirks and gotchas

Small things that are easy to trip over when you wire `dmt` into a pipeline.

## Output length depends on the mode

The output time axis has `plan.dmt_nsamps` samples (C++ `get_dmt_nsamps()`):

- **`mode="valid"`** gives `nsamps` per block. After a cold start (a new engine or
  `reset_history()`) the first `startup_samples` outputs of the first block
  are partial sums, because the history before time 0 is zero. They are
  written, not withheld; DDMT instead withholds its incomplete samples on the
  first call. From the second block on every sample is complete (see
  [streaming](streaming_and_history.md)).
- **`mode="full"`** gives `nsamps` plus the largest delay magnitude,
  $\max(|\Delta t_{\min}|, |\Delta t_{\max}|)$, per block, capturing the
  pulse tail by zero-padding. There is no history between blocks, so the first
  `startup_samples` outputs of every block are partial sums.
- **`mode="roll"`** gives `nsamps` per block, with samples delayed past the end
  wrapped to the start. Fine for strictly periodic test signals; avoid it for
  single-pulse searches.

## Custom grids are keyword-only in Python

This stops a float DM array from silently landing in the integer `dt_max`
argument:

```python
FDMT(1200.0, 1600.0, 256, 1024, 1e-3, my_dm_array)           # TypeError
FDMT(1200.0, 1600.0, 256, 1024, 1e-3, dm_grid=my_dm_array)   # or dt_grid=
```

The planner sorts custom grids and drops duplicates.

## Pass contiguous arrays

The bindings read C-contiguous memory. A strided slice such as
`waterfall[::2, :]` is not contiguous; wrap it in `np.ascontiguousarray` before
calling `execute()`.

(output-buffers)=
## Output buffers: `dmt_size`

The output buffer holds exactly the transform. The engine owns all its
working state (the tree levels and the streaming history), so your buffer is
only written, never used as scratch:

| C++ | Python | meaning |
| :--- | :--- | :--- |
| `plan.get_dmt_size()` | `plan.dmt_size` | floats per beam, $N_{\text{DM}} \times N_{\text{times}}$ |

- **Layout.** The buffer is `nbeams * get_dmt_size()` floats, laid out
  $(N_{\text{beams}}, N_{\text{DM}}, N_{\text{times}})$ row-major. Beam $b$
  starts at offset $b \cdot$ `get_dmt_size()`.
- **Size.** A larger buffer is accepted; only the leading `nbeams * get_dmt_size()`
  floats are written.
- **Reuse.** Allocate one buffer per stream and pass it to every
  `execute()`. The engine allocates nothing per call. Copy out (or consume)
  the result before the next call overwrites it.

```cpp
const auto& plan = fdmt.get_plan();
std::vector<float> buf(fdmt.get_nbeams() * plan.get_dmt_size());
fdmt.execute(waterfall, buf);                          // every block
std::span<const float> beam0(buf.data(), plan.get_dmt_size()); // (ndms, nsamps)
```

In Python, `execute()` returns an `(n_dm, n_times)` (or `(nbeams, n_dm,
n_times)`) view of the output buffer, without a copy:

- Pass `out=` (at least `nbeams * plan.dmt_size` float32 values) to reuse one
  buffer and avoid a fresh allocation and page-fault pass per call. This is
  ~10% faster for large blocks:

  ```python
  out = np.empty(fdmt.nbeams * fdmt.plan.dmt_size, dtype=np.float32)
  for block in stream:
      dmt = fdmt.execute(block, out=out)   # overwritten by the next call
  ```

- Without `out=`, every call allocates a new array of exactly the result's
  size.

```{warning}
With `out=`, the returned view is **overwritten by the next call**.
`.copy()` the views you keep.
```

:::{dropdown} Code written for the old `buffer_size` layout
:icon: info

If your code predates this layout and allocates `plan.buffer_size` per beam,
it still runs, since a larger buffer is accepted, but beams are now
`dmt_size` apart. `buffer_size` is now only the largest tree level of one
block, a rough size estimate. What the engine really allocates is in
`memory_usage` (C++ `get_memory_usage()`).
:::

(stepper-rules)=
## Stepper rules (`reset` / `advance` / `finalize`)

The stepper exposes intermediate sub-band levels, e.g. to stop 1–2 levels
before the root for a sub-band search:

```python
fdmt.reset(block)                  # consumes one input block (level 0)
fdmt.advance_until_remaining(2)    # 4 sub-bands left
for s in range(fdmt.num_subbands):
    sub = fdmt.view_subband(s)     # data, ndt, nsamps, f_start, f_end, dt_grid
dmt = fdmt.finalize()              # finish to the root
```

- **Finish each block.** In `mode="valid"`, finalize every block before the next
  `reset()`/`execute()`. Levels a block never computed would skip their
  cross-block history update, so starting the next block early raises
  `RuntimeError` (C++: `std::logic_error`). `reset_history()` abandons the
  unfinished block and starts a cold stream. `full`/`roll` have no history,
  so they allow restarting.
- **No fusion.** The stepper always runs level by level (it never fuses), so every
  level can be inspected. Only `execute()` uses `fuse_levels`.
- **Whole blocks.** The stepper runs whole blocks: it needs `fdmt.chunk_nsamps == nsamps`
  (C++: `get_chunk_nsamps()`) and raises otherwise. The automatic chunk is
  shorter than the block for large plans in `mode="valid"`, so construct a
  stepping engine with `chunk_nsamps=nsamps`.
- **Views are overwritten.** On the CPU, views share memory with the engine and are overwritten as later levels are
  computed; copy what you keep. On CUDA from Python, views are host copies.
- **Packed input.** With packed input and `int_tree=True` (the default), integer-stored levels
  cannot be viewed (the view raises). Construct with `int_tree=False` to
  inspect every level.
- **Multi-beam.** With `nbeams > 1`, the views show beam 0 only. `advance()`/`finalize()`
  still process every beam.

## Logging and summaries

dmt is silent by default.

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
import dmtlib
dmtlib.set_log_level("debug")   # "off" (default) or "debug"
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
#include <dmt/dmt.hpp>
dmt::set_log_level(dmt::LogLevel::kDebug);   // kOff (default) or kDebug
```
:::
::::

To describe a plan or an engine, ask for its summary instead:

```python
print(fdmt.summary())        # plan + engine: mode, threads, fusion, memory
print(fdmt.plan.summary())   # plan only (also CohFDMTPlan, DDMTPlan)
```
