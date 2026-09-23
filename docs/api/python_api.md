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
