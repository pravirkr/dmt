# Getting started

Install the Python package or the C++ library, then dedisperse a first
waterfall. The example below is the whole idea in a few lines.

```{toctree}
:maxdepth: 1
:hidden:

installation
quickstart_python
quickstart_cpp
example
```

## A first example

```python
import numpy as np
from dmtlib import compute_fdmt
from dmtlib.simulate import generate_frb

# A simulated FRB in a 256-channel filterbank
waterfall = generate_frb(
    f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dm=150.0, amp=3.0, offset=200, noise_rms=1.0,
)

# Dedisperse with FDMT
dmt, plan = compute_fdmt(
    waterfall, f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dt_max=256, mode="valid"
)

# Find the peak in the DM-time plane
dmt_plane = dmt.reshape(plan.dmt_ndms, plan.dmt_nsamps)
peak_dm_idx, peak_time = np.unravel_index(np.argmax(dmt_plane), dmt_plane.shape)
print(f"Detected FRB at DM trial: {plan.dm_grid_final[peak_dm_idx]:.2f} pc cm^-3, time: {peak_time}")
```

## Next

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Compilers, FFTW, OpenMP and the GPU builds.
:::

:::{grid-item-card} Python quickstart
:link: quickstart_python
:link-type: doc

The engine classes, streaming, and the stepper.
:::

:::{grid-item-card} C++ quickstart
:link: quickstart_cpp
:link-type: doc

`dmt::algorithms::FDMT` in a C++ pipeline.
:::

:::{grid-item-card} Worked example
:link: example
:link-type: doc

Draws the waterfall and DM–time figures shown in the README.
:::
::::
