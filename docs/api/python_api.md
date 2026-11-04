# Python API reference

## Engines

```{eval-rst}
.. autoclass:: dmtlib.FDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.DDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.SDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.CohFDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.CohFDMTConfig
   :members:
   :undoc-members:

.. autoclass:: dmtlib.BasebandFormat
   :members:
   :undoc-members:

.. autofunction:: dmtlib.simulate_baseband

.. autofunction:: dmtlib.pack_baseband

.. autoclass:: dmtlib.FDMTFFT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.DDMTFFT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTMemoryUsage
   :members:
   :undoc-members:
```

Every engine runs on the backend given by its keyword-only `backend=`
argument (`"cpu"` by default; `"cuda"` or `"hip"` when that backend is in the
build), with `nthreads` for the CPU and `device` for a GPU.
`available_backends()` lists the backends in the installed build; asking for any other raises `ValueError`. Inputs and
outputs are NumPy arrays on every backend: a GPU backend copies the input to
the device and the result back, and blocks until it is on the host. The
stepper's `view_*` methods return host views of the current level (a host
snapshot on a GPU). Device-memory overloads (`DeviceSpan`) are C++ only for
now (see the C++ API reference).

```{eval-rst}
.. autofunction:: dmtlib.available_backends
```

## Plans and geometry

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

.. autoclass:: dmtlib.LevinConfig
   :members:
   :undoc-members:
```

## Plain time series

```{eval-rst}
.. autofunction:: dmtlib.dedisperse

.. autoclass:: dmtlib.Dedispersed
   :members:

.. autofunction:: dmtlib.dispersion_delay
```

## Channel order

```{eval-rst}
.. autofunction:: dmtlib.sigproc_band

.. autofunction:: dmtlib.to_ascending

.. autoclass:: dmtlib.Band
   :members:
```

## Candidate search

```{eval-rst}
.. autoclass:: dmtlib.FDMTBoxcar
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTFFTBoxcar
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.BoxcarConfig
   :members:
   :undoc-members:

.. autoclass:: dmtlib.BoxcarPlanePlan
   :members:
   :undoc-members:

.. autoclass:: dmtlib.BoxcarResult
   :members:
   :undoc-members:

.. autoclass:: dmtlib.PeakFindingMode
   :members:
   :undoc-members:

.. autoclass:: dmtlib.FDMTBoxcarMemoryUsage
   :members:
   :undoc-members:

.. autofunction:: dmtlib.generate_box_width_trials
```

## Convenience functions

```{eval-rst}
.. autofunction:: dmtlib.compute_fdmt

.. autofunction:: dmtlib.compute_fdmt_fft

.. autofunction:: dmtlib.add_frb_track
```

## FFT planner (CPU)

```{eval-rst}
.. autoclass:: dmtlib.FFTPlanner

.. autofunction:: dmtlib.set_fft_planner

.. autofunction:: dmtlib.get_fft_planner

.. autofunction:: dmtlib.import_fft_wisdom

.. autofunction:: dmtlib.export_fft_wisdom

.. autofunction:: dmtlib.forget_fft_wisdom
```

## Logging

```{eval-rst}
.. autofunction:: dmtlib.set_log_level

.. autofunction:: dmtlib.get_log_level
```

## Simulation utilities (`dmtlib.simulate`)

```{eval-rst}
.. automodule:: dmtlib.simulate
   :members:
   :undoc-members:
```

## Grid utilities (`dmtlib.grid`)

```{eval-rst}
.. automodule:: dmtlib.grid
   :members:
   :undoc-members:
```
