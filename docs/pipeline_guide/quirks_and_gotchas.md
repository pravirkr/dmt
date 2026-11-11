# Quirks and gotchas

Small things that are easy to trip over.

## Output length depends on the mode

The time axis has `plan.dmt_nsamps` samples (C++ `get_dmt_nsamps()`):

| mode | samples per block | history | first `startup_samples` outputs |
| :--- | :--- | :--- | :--- |
| `"valid"` | `nsamps` | yes | partial sums on the first block after a cold start |
| `"full"` | `nsamps` + largest delay magnitude | no | partial sums on every block |
| `"roll"` | `nsamps`, late samples wrapped to the start | no | wrapped; avoid for single-pulse searches |

Partial sums are written, not withheld (DDMT withholds them). See
[streaming](streaming_and_history.md).

## Custom grids are keyword-only in Python

So a float DM array can't land in the integer `dt_max`:

```python
FDMT(1200.0, 1600.0, 256, 1024, 1e-3, my_dm_array)           # TypeError
FDMT(1200.0, 1600.0, 256, 1024, 1e-3, dm_grid=my_dm_array)   # or dt_grid=
```

The planner sorts custom grids and drops duplicates.

## Pass contiguous arrays

The bindings read C-contiguous memory. Wrap strided slices such as
`waterfall[::2, :]` in `np.ascontiguousarray` before `execute()`.

(output-buffers)=
## Output buffers

The output buffer holds only the transform; the engine owns its working
state, so your buffer is written, never used as scratch.

- **Size.** `nbeams * plan.get_dmt_size()` floats (Python `plan.dmt_size`),
  where `dmt_size` is $N_{\text{DM}} \times N_{\text{times}}$. Larger buffers
  are accepted; only the leading part is written.
- **Layout.** $(N_{\text{beams}}, N_{\text{DM}}, N_{\text{times}})$ row-major;
  beam $b$ starts at $b \cdot$ `dmt_size`.
- **Reuse.** Allocate once and pass it to every `execute()`. The engine
  allocates nothing per call.

```cpp
std::vector<float> buf(fdmt.get_nbeams() * fdmt.get_plan().get_dmt_size());
fdmt.execute(waterfall, buf);   // every block
```

In Python, `execute()` returns a view of the output buffer. Pass `out=` to
reuse one buffer, about 10% faster for large blocks; without it every call
allocates a new array.

```python
out = np.empty(fdmt.nbeams * fdmt.plan.dmt_size, dtype=np.float32)
for block in stream:
    dmt = fdmt.execute(block, out=out)
```

```{warning}
With `out=`, the returned view is **overwritten by the next call**. `.copy()`
what you keep.
```

:::{dropdown} Code written for the old `buffer_size` layout
:icon: info

Code that allocates `plan.buffer_size` per beam still runs (a larger buffer
is accepted), but beams are now `dmt_size` apart. `buffer_size` is only the
largest tree level of one block; real allocations are in `memory_usage` (C++
`get_memory_usage()`).
:::

(stepper-rules)=
## Stepper rules

The stepper exposes the intermediate subband levels, for example to stop
before the root for a subband search:

```python
plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_max, subband_depth=2)
fdmt = FDMT(plan, chunk_nsamps=nsamps)
fdmt.reset(block)                  # consume one block (level 0)
fdmt.advance_until_remaining(2)    # 4 subbands + 3 staggered
for s in range(fdmt.num_subbands):
    sub = fdmt.view_subband(s)     # data, ndt, nsamps, f_start, f_end, dt_grid,
                                   # chan_begin, chan_end, staggered
dmt = fdmt.finalize()              # finish to the root
```

With `k` levels remaining there are `2**k` equal subbands in ascending
frequency, then `2**k - 1` staggered ones if `subband_depth >= k`.

- **Finish each block.** In `mode="valid"`, `finalize()` before the next
  `reset()`/`execute()`, or it raises `RuntimeError` (C++
  `std::logic_error`). `reset_history()` abandons the block. `full`/`roll`
  allow restarting.
- **Whole blocks.** It needs `chunk_nsamps == nsamps` and raises otherwise.
- **No fusion.** It runs level by level; only `execute()` uses `fuse_levels`.
- **Views are overwritten.** On the CPU they share memory with the engine;
  copy what you keep. On CUDA from Python they are host copies.
- **Packed input.** With `int_tree=True` (the default), integer levels can't
  be viewed; construct with `int_tree=False`.
- **Multi-beam.** Views show beam 0 only; every beam is still processed.

## Logging and summaries

dmt is silent by default.

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
import dmtlib
dmtlib.set_log_level("debug")   # "off" (default) or "debug"
print(fdmt.summary())           # plan + engine: mode, threads, fusion, memory
print(fdmt.plan.summary())      # plan only
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
dmt::set_log_level(dmt::LogLevel::kDebug);   // kOff (default) or kDebug
```
:::
::::
