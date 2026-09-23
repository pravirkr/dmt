# Quirks & Gotchas

Every high-performance digital signal processing library has architectural edge cases. This page documents practical "gotchas" and quirks to ensure smooth pipeline integration.

---

## 1. Output Alignment: `valid` vs. `full` vs. `roll`

A common source of confusion is the output time dimension size:

- **In `mode="valid"`**:
  - **Cold Start (Block 0)**: The output length is $N_{\text{samps}} - \Delta t_{\max}$ because samples before time 0 are unknown.
  - **Streaming Steady State (Block 1+)**: The output length is **$N_{\text{samps}}$** because delayed tail samples are restored from the previous block.
  - *Gotcha:* If your pipeline assumes a fixed static output array shape, initialize your buffer size using `plan.dmt_nsamps` or check `dmt_out.shape[1]`.
- **In `mode="full"`**:
  - The output length is $N_{\text{samps}} + \Delta t_{\max}$. The tail of the pulse is completely captured by zero-padding.
- **In `mode="roll"`**:
  - The output length is $N_{\text{samps}}$. Samples delayed past the end wrap circularly to the start. (Ideal for periodic pulsar folding; avoid for single-pulse blind searches).

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

## 5. Buffer Allocation in C++

In C++, `FDMTCPU::execute(waterfall, dmt_out)` requires output buffer storage sized to `plan.get_buffer_size()`, not just `plan.get_dmt_size()`:

- `plan.get_dmt_size()`: Number of elements in the final $(N_{\text{DM}}, N_{\text{times}})$ matrix.
- `plan.get_buffer_size()`: Includes intermediate tree ping-pong buffers required during execution.

Always allocate using:

```cpp
std::vector<float> dmt_out(plan.get_buffer_size());
fdmt.execute(waterfall_span, std::span<float>(dmt_out));
```

In Python, this is managed automatically.
