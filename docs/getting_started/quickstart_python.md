# Python quickstart

From one waterfall to a streaming engine in `dmtlib`. The
[pipeline guide](../pipeline_guide/index.md) covers the details.

## One-shot dedispersion

```python
import numpy as np
from dmtlib import compute_fdmt

waterfall = np.random.normal(size=(256, 1024)).astype(np.float32)  # (nchans, nsamps)

dmt, plan = compute_fdmt(
    waterfall,
    f_min=1200.0,   # bottom edge of the band (MHz)
    f_max=1600.0,   # top edge of the band (MHz)
    nchans=256,
    nsamps=1024,
    tsamp=1e-3,     # sampling interval (s)
    dt_max=150,     # largest delay trial (samples)
)
dmt_plane = dmt.reshape(plan.dmt_ndms, plan.dmt_nsamps)   # (n_dm, n_times)
dms = plan.dm_grid_final                                  # DM of each row (pc cm^-3)
```

## Streaming with the `FDMT` engine

For a stream of blocks, construct the engine once and call it per block:

```python
from dmtlib import FDMT

fdmt = FDMT(f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
            tsamp=1e-3, dt_max=150, nthreads=4)   # backend="cuda" for a GPU

out = np.empty(fdmt.nbeams * fdmt.plan.dmt_size, dtype=np.float32)
for block in stream:                        # contiguous (256, 1024) float32 blocks
    dmt_plane = fdmt.execute(block, out=out)
    process(dmt_plane)
```

- **Blocks join exactly.** The engine keeps the history between calls
  (`mode="valid"`, the default). Call `fdmt.reset_history()` only when the
  stream breaks. See [streaming](../pipeline_guide/streaming_and_history.md).
- **`out=` is reused.** The result is a view of `out`, overwritten by the next
  call; `.copy()` what you keep.
- **S/N.** With box smearing (the default) the output is in box-mean units and
  the noise differs per DM trial. Divide by `fdmt.get_effective_sigma_grid()`;
  see [noise calibration](../pipeline_guide/noise_calibration.md).

## Inputs and outputs

| input | call |
| :--- | :--- |
| float32 `(nchans, nsamps)` or `(nbeams, nchans, nsamps)` | `execute(waterfall)` |
| packed `uint8` `(nchans, row_bytes)` or `(nbeams, nchans, row_bytes)` | `execute(packed, nbits=...)` |

- **Output.** float32, `(n_dm, n_times)` or `(nbeams, n_dm, n_times)`, with
  `n_dm = plan.dmt_ndms` and `n_times = plan.dmt_nsamps`.
- **Arrays** must be C-contiguous with exact dtype (`float32`, or `uint8` for
  packed input); other dtypes or non-contiguous views raise `ValueError` rather
  than silently copying or casting.
- **Packed input.** `nbits` ∈ {1, 2, 4, 8, 16}, samples LSB-first within a
  byte, `row_bytes = ceil(nsamps * nbits / 8)`. Pass digitiser output as is:
  it is faster than converting to float, with the same result up to the
  rounding of the box means (S/N loss ≤ 0.5%).
- **Errors.** Wrong sizes or `nbits` raise `ValueError`, as does a transposed
  (time-major `(nsamps, nchans)`) array; a `uint8` array without `nbits`
  raises `TypeError`.
- **GPU.** `backend="cuda"` (and `device=`) takes the same arguments.
  `dmtlib.available_backends()` lists what your build has. Arrays already on
  the GPU (PyTorch, CuPy, Numba, DLPack) go straight in; see the
  {doc}`zero-copy GPU tutorial <../tutorials/08_gpu_zero_copy_arrays>`.

```python
nbits = 2
packed = np.random.randint(0, 256, size=(256, 1024 * nbits // 8), dtype=np.uint8)
dmt_plane = fdmt.execute(packed, nbits=nbits)
```

(channel-order)=
## Channel order and band edges

- **Ascending rows.** Row 0 is the lowest channel. Flip descending data before
  it reaches dmt; the wrong order dedisperses the sweep backwards and finds
  nothing at positive DM.
- **Band edges.** `f_min` and `f_max` are edges, not channel centres: channel
  `c` covers `[f_min + c * df, f_min + (c + 1) * df)`.
- **Time reference.** Output times refer to `f_min` (`plan.f_ref`); see
  [periodicity searches](../pipeline_guide/periodicity.md) for `f_ref=f_max`.

SIGPROC headers give `fch1` (the first channel's centre) and a usually
negative `foff`. Convert them and flip the data with:

```python
band = dmtlib.sigproc_band(hdr["fch1"], hdr["foff"], hdr["nchans"])
wf = np.ascontiguousarray(dmtlib.to_ascending(data, band))  # (nchans, nsamps)
fdmt = dmtlib.FDMT(band.f_min, band.f_max, band.nchans, nsamps, tsamp, dt_max)
```

## Custom trial grids

Pass your own delays (`dt_grid=`, samples) or DMs (`dm_grid=`, pc cm⁻³)
instead of `dt_max`, for example a sparse grid matched to the intra-channel
smearing:

```python
fdmt = FDMT(1200.0, 1600.0, 256, 1024, 1e-3,
            dt_grid=np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256]))
fdmt.plan.dt_grid_final
```

## Candidate search with `FDMTBoxcar`

`FDMTBoxcar` runs the FDMT and a boxcar (matched-filter) search together and
returns candidates instead of the DM–time plane:

```python
from dmtlib import BoxcarConfig, FDMTBoxcar, FDMTPlan

plan = FDMTPlan(1200.0, 1600.0, 256, 1024, 1e-3, 150)  # f_min, f_max, nchans, nsamps, tsamp, dt_max
config = BoxcarConfig(64, detect_thresh=7.0)         # boxcars up to 64 samples, S/N >= 7
search = FDMTBoxcar(plan, config, nthreads=4)        # backend="cuda" for a GPU

for block in stream:                       # contiguous (256, 1024) float32 blocks
    res = search.execute(block)
    keep(res.candidates)                   # structured array: peak_sample, snr, dm, width, ...
keep(search.drain().candidates)            # end of stream: search the last samples
```

- **Input.** Each channel zero mean with a common sigma (`config.noise_sigma`,
  default 1); detrend upstream. Packed input: `BoxcarConfig(...,
  input_nbits=8, input_mean=..., noise_sigma=...)` and `uint8` blocks.
- **Times.** `peak_sample` counts from the start of the stream, at the
  arrival at the lower edge `f_low_mhz` of the candidate's own subband (the
  full band's `f_min` unless the plan has `subband_depth > 0`).
- **Last block.** A final block shorter than `nsamps` goes in as a whole
  block whose first `n` samples are data: `search.drain(block, n_tail=n)`.
- **Device arrays** (CuPy, PyTorch, Numba) are searched in place on a GPU
  backend; the candidates come back as NumPy arrays.

## Next

- [Multi-beam batching](../pipeline_guide/multibeam.md): `nbeams=` for several
  beams per call.
- [Quirks and gotchas](../pipeline_guide/quirks_and_gotchas.md): output
  modes, the stepper, logging.
- [Python API reference](../api/python_api.md).
