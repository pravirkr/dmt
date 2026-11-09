# Tutorials

Notebooks that run on a laptop in seconds (the last one needs a CUDA GPU).
The first two build up from dispersion to how FDMT works; the rest each take
one topic.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 1. Radio data and dedispersion
:link: 01_radio_data_and_dedispersion_basics
:link-type: doc

Dispersion, filterbank data, and a first FDMT run.
:::

:::{grid-item-card} 2. Inside the FDMT tree
:link: 02_incoherent_fdmt_pipeline
:link-type: doc

Inside the FDMT tree, with the stepper API.
:::

:::{grid-item-card} 3. Sparse DM grids and negative DMs
:link: 03_sparse_dm_grids_and_negative_dms
:link-type: doc

Choosing DM trials, and searching negative DMs.
:::

:::{grid-item-card} 4. FDMT and direct dedispersion
:link: 04_fdmt_vs_ddmt_comparison
:link-type: doc

FDMT against direct dedispersion.
:::

:::{grid-item-card} 5. Hybrid coherent FDMT
:link: 05_coherent_fdmt_cfdmt_hybrid_search
:link-type: doc

Coherent dedispersion of baseband with CohFDMT.
:::

:::{grid-item-card} 6. Streaming and multi-beam batching
:link: 06_realtime_streaming_and_batching
:link-type: doc

Streaming blocks and batching beams.
:::

:::{grid-item-card} 7. Noise statistics of FDMT and DDMT
:link: 07_fdmt_noise_statistics
:link-type: doc

Is the output noise white? The cold start, and periodicity searches.
:::

:::{grid-item-card} 8. Zero-copy GPU arrays
:link: 08_gpu_zero_copy_arrays
:link-type: doc

PyTorch, CuPy and Numba arrays in place: a search that stays on the GPU.
:::
::::

```{toctree}
:maxdepth: 1
:hidden:

01_radio_data_and_dedispersion_basics
02_incoherent_fdmt_pipeline
03_sparse_dm_grids_and_negative_dms
04_fdmt_vs_ddmt_comparison
05_coherent_fdmt_cfdmt_hybrid_search
06_realtime_streaming_and_batching
07_fdmt_noise_statistics
08_gpu_zero_copy_arrays
```
