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
`available_backends()` lists the backends in the installed build; asking for any other raises `ValueError`. NumPy
inputs and outputs work on every backend: a GPU backend copies the input to
the device and the result back, and blocks until it is on the host. The
stepper's `view_*` methods return host views of the current level (a host
snapshot on a GPU).

On a GPU backend, `execute()` (and DDMT's `execute_time_major()`), the FDMT
steppers' `reset()`, and `save_history()`/`load_history()` of `FDMT`,
`FDMTFFT`, `DDMT`, `SDMT`, `DDMTFFT` and `CohFDMT` also take device arrays:
PyTorch tensors, CuPy arrays, Numba device arrays, or any other `__dlpack__`
or `__cuda_array_interface__` (CAI) array on the engine's device. They are
read and written in place, without copies (the
[zero-copy GPU arrays tutorial](../tutorials/08_gpu_zero_copy_arrays.ipynb)
walks through a full GPU pipeline):

```python
import cupy as cp

fdmt = dmtlib.FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, backend="cuda")
waterfall = cp.asarray(block)                  # or a torch.cuda tensor
dmt = fdmt.execute(waterfall)                  # a cupy.ndarray on the GPU
out = cp.empty(fdmt.plan.dmt_size, cp.float32)
dmt = fdmt.execute(waterfall, out=out, stream=cp.cuda.Stream())
```

- **Dtypes.** Device arrays must be C-contiguous and have the exact dtype:
  float32, packed uint8 with `nbits`, or uint8/int8 baseband for `CohFDMT`.
  Anything else raises `TypeError`; nothing is converted.
- **Output.** Without `out=`, the result is a new array from the input's own
  library (PyTorch, CuPy or Numba), allocated on the call's stream from that
  library's memory pool (a Numba result is bound to the call's stream). With
  `out=` (a writeable, C-contiguous device array of at least the output
  size), the result is a view of `out`, or `out` itself if its library
  cannot reshape. Other libraries (JAX, a bare CAI object) need `out=`.
- **Protocols.** An array with `__dlpack__` is consumed through DLPack, even
  if it also has `__cuda_array_interface__`; CAI is the fallback (CUDA only;
  the device is looked up from the pointer). CAI arrays with a `mask` are
  rejected; a read-only CAI array cannot be `out=`.
- **Streams.** `stream=` (keyword-only) takes a `cudaStream_t`/`hipStream_t`
  handle as an int, a `torch.cuda.Stream`, a `cupy.cuda.Stream`, a Numba
  stream, or any object with `__cuda_stream__`. The default is the input
  library's current stream on its device; Numba has no current stream, so
  its default stream is used. The call queues its work on that stream and
  returns at once. Synchronize the stream, or keep using it, before reading
  the result.
- **Ordering after the producer.** DLPack producers order their pending work
  before the call's stream inside `__dlpack__(stream=...)`. PyTorch orders
  only the work on its *current* stream at the time of the call: if `x` was
  written on another stream, call inside `with torch.cuda.stream(...)` of
  that stream or synchronize it first. A CAI array that states a `stream`
  is ordered with an event the call's stream waits for (no host wait);
  `NUMBA_CUDA_ARRAY_INTERFACE_SYNC=0` turns that off, as it does in Numba.
- **Lifetime.** Keep the inputs alive until the call's stream has consumed
  them. PyTorch inputs are exempt: the call marks them in use on its stream
  (`Tensor.record_stream`), so PyTorch's allocator does not reuse their
  memory early even if they are dropped at once. CuPy and Numba have no such
  hook; dropping an input allocated on another stream before the call's
  stream has run is unsafe.
- **Stepper.** After a device `reset()`, `advance()`,
  `advance_until_remaining()` and `finalize()` keep using the reset's stream
  unless they get `stream=`, and `finalize()` returns a device view. The
  `view_*` methods still return host snapshots.

```{eval-rst}
.. autofunction:: dmtlib.available_backends
```

`build_info()` identifies the compiled extension (version, git commit and
dirty flag, compiler and flags, GPU toolkit, FFTW and OpenMP versions). Store
it with any result you publish.

```{eval-rst}
.. autofunction:: dmtlib.build_info
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
