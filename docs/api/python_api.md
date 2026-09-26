# Python API Reference

Documentation for the Python bindings and utilities provided by `dmtlib`.

---

## Algorithms & Compute Engines

```{eval-rst}
.. autoclass:: dmtlib.FDMTCPU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.DDMTCPU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.CohFDMTCPU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTFFTCPU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTMemoryUsage
   :members:
   :undoc-members:
```

The CUDA engines (`FDMTCUDA`, `DDMTCUDA`, `CohFDMTCUDA`, `FDMTFFTCUDA`) are
importable from `dmtlib` when the CUDA extension is built. They take host
NumPy arrays; the device-memory overloads are C++ only (see the C++ API
reference). They take the same constructor arguments as the CPU engines, with
`device_id` in place of `nthreads`.

`FDMTCUDA` in Python provides:

- `execute(waterfall, *, out=None)` and `execute(waterfall_packed, nbits, *,
  out=None)`, with the same layout and return value as `FDMTCPU.execute`.
- The stepper: `reset(waterfall)` / `reset(waterfall_packed, nbits)`,
  `advance`, `advance_until_remaining`, `view_level_data`,
  `view_subband_data`, `view_subband` and `finalize`. The block is staged on
  the device; views and `finalize()` return **host copies**, and
  `view_subband` returns a dict.
- `current_level`, `total_levels`, `remaining_levels`, `num_subbands`,
  `is_finished`, `reset_history`.
- `get_effective_variance`/`get_effective_sigma` (and their `_grid`
  variants), plus `plan`, `nbeams`, `device_id`, `fuse_levels`, `int_tree`,
  `memory_usage`, `dt_grid_final` and `dm_grid_final`.
- Not in Python: `save_history`/`load_history` (C++ only, used by
  `CohFDMTCUDA`).

## Execution Plans & Geometry

```{eval-rst}
.. autoclass:: dmtlib.FDMTPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.DDMTPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.CohFDMTPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTComplexity
   :members:
   :undoc-members:
```

## Convenience Functions

```{eval-rst}
.. autofunction:: dmtlib.compute_fdmt

.. autofunction:: dmtlib.compute_fdmt_fft

.. autofunction:: dmtlib.add_frb_track
```

## Simulation Utilities (`dmtlib.simulate`)

```{eval-rst}
.. automodule:: dmtlib.simulate
   :members:
   :undoc-members:
```

## Grid Utilities (`dmtlib.grid`)

```{eval-rst}
.. automodule:: dmtlib.grid
   :members:
   :undoc-members:
```
