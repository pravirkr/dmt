# Quirks & Gotchas

Every high-performance digital signal processing library has architectural edge cases. This page documents practical "gotchas" and quirks to ensure smooth pipeline integration.

---

## 1. Output Alignment: `valid` vs. `full` vs. `roll`

The output time dimension is `plan.dmt_nsamps` (C++: `get_dmt_nsamps()`):

- **`mode="valid"`**: always $N_{\text{samps}}$ per block. After a cold start
  (a new engine or `reset_history()`) the first $\Delta t_{\max}$ output
  samples of the first block are partial sums, because the history before
  time 0 is zero. From the second block on, every sample is complete and the
  blocks join seamlessly (see [Streaming](streaming_and_history.md)).
- **`mode="full"`**: $N_{\text{samps}} + \Delta t_{\max}$ per block. The pulse
  tail is captured by zero-padding. There is no cross-block history.
- **`mode="roll"`**: $N_{\text{samps}}$ per block. Samples delayed past the
  end wrap circularly to the start. That is useful for periodic signals;
  avoid it for single-pulse searches.

---

## 2. Custom Grids are Keyword-Only in Python

To prevent silent bugs where a floating-point DM array is accidentally passed into the integer `dt_max` positional argument:

```python
# WRONG (Raises TypeError):
# fdmt = FDMTCPU(1200.0, 1600.0, 256, 1024, 1e-3, my_dm_array)

# CORRECT (Keyword-only):
fdmt = FDMTCPU(
    1200.0, 1600.0, 256, 1024, 1e-3,
    dm_grid=my_dm_array  # or dt_grid=my_dt_array
)
```

Unsorted or duplicate trials in custom grids are automatically sorted and uniqued by the C++ planner.

---

## 3. Array Contiguity (C-Contiguous Required)

All C++ bindings expect contiguous memory layouts (`std::span` over C-style pointers).

If you slice or sub-select channels in NumPy:

```python
# Slicing creates a non-contiguous strided array:
sub_waterfall = full_waterfall[::2, :]  # Not contiguous!

# FIX: Wrap with np.ascontiguousarray before passing to execute():
dmt = fdmt.execute(np.ascontiguousarray(sub_waterfall))
```

---

(output-buffers)=
## 4. Output Buffers: `buffer_size` vs `dmt_size`

An FDMT engine ping-pongs its tree levels between one internal buffer and
**your output buffer**. The output buffer must therefore hold the largest
tree level, not just the result:

| symbol | C++ | Python | meaning |
| :--- | :--- | :--- | :--- |
| $B$ | `plan.get_buffer_size()` | `plan.buffer_size` | floats per beam you allocate |
| $D$ | `plan.get_dmt_size()` | `plan.dmt_size` | floats per beam that are the result, $N_{\text{DM}} \times N_{\text{times}}$ |

- Each beam's first $D$ floats hold the $(N_{\text{DM}}, N_{\text{times}})$
  transform, row-major. The remaining $B - D$ are scratch with unspecified
  contents, which can differ between fusion depths or backends.
- Beam $b$ starts at offset $b \cdot B$, so the buffer is
  `nbeams * get_buffer_size()` floats.
- **Why:** the engine owns only one ping-pong half, which keeps total memory
  at about $2B$ per beam (important on a GPU) with no final copy. Typically
  $B \approx 3D$; for example, with 4096 channels and `dt_max=2048`, $B$ is
  384 MiB and $D$ is 128 MiB at 16K samples.
- **Reuse:** allocate one $B$-sized buffer per stream and pass it to every
  `execute()`. The engine allocates nothing per call. Copy out (or consume)
  the first $D$ values before the next call overwrites them.

```cpp
const auto& plan = fdmt.get_plan();
std::vector<float> buf(fdmt.get_nbeams() * plan.get_buffer_size());
fdmt.execute(waterfall, buf);                          // every block
std::span<const float> beam0(buf.data(), plan.get_dmt_size()); // (ndms, nsamps)
```

In Python, `execute()` returns an `(n_dm, n_times)` (or `(nbeams, n_dm,
n_times)`) **zero-copy view** of the first $D$ floats per beam. The view's
base is the whole $B$-sized buffer, so:

- Pass `out=` to reuse one buffer and avoid a fresh allocation and page-fault
  pass per call. This is ~10% faster for large blocks:

  ```python
  out = np.empty(fdmt.nbeams * fdmt.plan.buffer_size, dtype=np.float32)
  for block in stream:
      dmt = fdmt.execute(block, out=out)   # overwritten by the next call
  ```

- Without `out=`, every returned array keeps its whole $B$-sized buffer
  alive. Use `dmt.copy()` if you keep many blocks.

**CohFDMT arena.** `CohFDMTCPU` runs one fine FDMT per coarse-DM trial in
place in your output buffer. Trial $i$ writes its result at offset $i \cdot D$
and uses the following $B$ floats as scratch, which later trials then
overwrite. The buffer is therefore `get_buffer_size()` $= (N-1) \cdot D + B$
floats for $N$ coarse trials. Only the first `get_dmt_size()` $= N \cdot D$
are the result, shaped $(N \cdot N_{\text{DM,fine}}, N_{\text{times}})$.
`CohFDMTCUDA`'s host `execute()` owns its device arena and needs only
`get_dmt_size()`. Python handles all of this for you.

---

(stepper-rules)=
## 5. Stepper Rules (`reset` / `advance` / `finalize`)

The stepper exposes intermediate sub-band levels, e.g. to stop 1–2 levels
before the root for a sub-band search:

```python
fdmt.reset(block)                  # consumes one input block (level 0)
fdmt.advance_until_remaining(2)    # 4 sub-bands left
for s in range(fdmt.num_subbands):
    sub = fdmt.view_subband(s)     # data, ndt, nsamps, f_start, f_end, dt_grid
dmt = fdmt.finalize()              # finish to the root
```

- **In `mode="valid"`, every block must be finalized** before the next
  `reset()`/`execute()`. Levels a block never computed would skip their
  cross-block history update, so starting the next block early raises
  `RuntimeError` (C++: `std::logic_error`). `reset_history()` abandons the
  unfinished block and starts a cold stream. `full`/`roll` have no history,
  so they allow restarting.
- The stepper always runs **level by level** (it never fuses), so every
  level can be inspected. Only `execute()` uses `fuse_levels`.
- Views are **zero-copy** on the CPU and are overwritten as later levels are
  computed; copy what you keep. On CUDA from Python, views are host copies.
- With packed input and `int_tree=True` (the default), integer-stored levels
  cannot be viewed (the view raises). Construct with `int_tree=False` to
  inspect every level.
- With `nbeams > 1`, the views show beam 0 only. `advance()`/`finalize()`
  still process every beam.

---

## 6. Logging (`verbose`)

`verbose` sets the **process-wide** spdlog level: 0 = warnings only (e.g. a
clamped `fuse_levels`), 1 = info (plan and memory summary), 2 = debug. The
most recently constructed object's value wins.
