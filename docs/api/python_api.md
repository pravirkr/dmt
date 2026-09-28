# Python API Reference

Documentation for the Python bindings and utilities provided by `dmtlib`.

---

## Algorithms & Compute Engines

```{eval-rst}
.. autoclass:: dmtlib.FDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.DDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.CohFDMT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTFFT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dmtlib.FDMTMemoryUsage
   :members:
   :undoc-members:
```

Every engine runs on the backend given by its keyword-only `backend=`
argument (`"cpu"` by default; `"cuda"` or `"hip"` when that backend is in the
build), with `nthreads` for the CPU and `device` for a GPU. `available_backends()` lists the backends in
the installed build; asking for any other raises `ValueError`. Inputs and
outputs are NumPy arrays on every backend: a GPU backend copies the input to
the device and the result back, and blocks until it is on the host. The
stepper's `view_*` methods return host views of the current level (a host
snapshot on a GPU). Device-memory overloads (`DeviceSpan`) are C++ only for
now (see the C++ API reference).

```{eval-rst}
.. autofunction:: dmtlib.available_backends
```

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

## Logging

```{eval-rst}
.. autofunction:: dmtlib.set_log_level

.. autofunction:: dmtlib.get_log_level
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
