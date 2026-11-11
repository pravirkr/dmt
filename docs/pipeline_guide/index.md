# Pipeline Guide

What you need once `dmt` runs inside a pipeline. Building a streaming search:
read the first three. Want a plain dedispersed time series: read
*Periodicity searches*.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Streaming and history
:link: streaming_and_history
:link-type: doc

How consecutive blocks join without losing a dispersed pulse, what happens at the start of a stream, and block size versus chunk size.
:::

:::{grid-item-card} Noise calibration
:link: noise_calibration
:link-type: doc

Why FDMT noise varies with DM, how to get the exact sigma of every trial, and what `use_box_smearing` does to the noise.
:::

:::{grid-item-card} Quirks and gotchas
:link: quirks_and_gotchas
:link-type: doc

Output lengths per mode, output buffers, stepper rules, array layout, logging.
:::

:::{grid-item-card} Periodicity searches
:link: periodicity
:link-type: doc

White output, the cold start, each engine's time reference, `dmtlib.dedisperse`, and which engine keeps signal at high spin frequencies.
:::

:::{grid-item-card} Multi-beam batching
:link: multibeam
:link-type: doc

Several beams through one plan, on CPU and GPU.
:::

:::{grid-item-card} Fourier-domain variants
:link: fourier_variants
:link-type: doc

FDMT-FFT and DDMT-FFT, fractional delays, NUFFT options and FFT settings.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

streaming_and_history
noise_calibration
quirks_and_gotchas
periodicity
multibeam
fourier_variants
```
