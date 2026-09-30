# dmt: Dispersion Measure Transform

```{raw} html
<div class="dmt-hero-card">
  <img class="dmt-hero-mark only-light" src="_static/logo/dmt-mark.svg" alt="">
  <img class="dmt-hero-mark only-dark" src="_static/logo/dmt-mark-dark.svg" alt="">
  <div class="dmt-hero-body">
  <p class="dmt-hero-lede">
    High-performance radio astronomy dedispersion algorithms.
    Written in modern <strong>C++20</strong> with <strong>OpenMP</strong> and <strong>CUDA</strong> acceleration,
    exposed with seamless, zero-copy <strong>Python</strong> bindings.
  </p>
  <div>
    <span class="dmt-badge dmt-badge-blue">C++20</span>
    <span class="dmt-badge dmt-badge-purple">Python 3.12+</span>
    <span class="dmt-badge dmt-badge-green">CUDA Accelerated</span>
    <span class="dmt-badge dmt-badge-blue">Zero-Copy NumPy</span>
    <span class="dmt-badge dmt-badge-purple">Real-Time Streaming</span>
  </div>
  </div>
</div>
```

---

## Key Features

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🚀 Comprehensive Algorithm Suite
:class-card: sd-shadow-sm

- **FDMT**: Fast Dispersion Measure Transform ($O(N_t N_f \log_2 N_f)$), float or packed 1/2/4/8/16-bit input
- **DDMT**: Direct (brute-force) dedispersion with DM-tiled CPU/GPU kernels, float or packed 1/2/4/8/16-bit input
- **SDMT**: Exact DDMT sums with partial sums shared between DM trials within subbands (CPU and GPU; bit-identical integer output)
- **CohFDMT**: Hybrid coherent + FDMT search of recorded baseband (GUPPI, LOFAR, PSRDADA; 8/4/2-bit) for microsecond pulses, with the exact multi-subband coarse DM grid
- **FDMT-FFT**: Frequency-domain phase-shift dedispersion
:::

:::{grid-item-card} ⚡ Hardware Acceleration
:class-card: sd-shadow-sm

- **CPU**: Highly vectorized C++20 code with OpenMP multi-threading and cache-conscious memory layouts.
- **GPU**: Native CUDA kernel support.
:::

:::{grid-item-card} 📡 Real-Time Pipeline Integration
:class-card: sd-shadow-sm

- **Sparse DM Grids**: Non-linear trial spacing.
- **Stepper API**: Interactive inspection of subband trees.
- **Overlap-Save Streaming**: Continuous data processing across block boundaries.
- **Multi-Beam Batching**: Batching across dozens of tied-array telescope beams.
:::

:::{grid-item-card} 🐍 Python & C++ Parity / Documentation
:class-card: sd-shadow-sm

- Production pipeline engines in low-level C++20.
- Exposed bindings for Python with zero-copy NumPy arrays.
- Rich documentation for both C++ and Python, including tutorials.
:::

::::

---

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🏁 Getting Started
:link: getting_started/index
:link-type: doc

Install `dmt` from source or wheels, configure CMake, and run your first dedispersion pipeline in 5 lines of Python or C++.
:::

:::{grid-item-card} 🛠️ Pipeline Guide & Quirks
:link: pipeline_guide/index
:link-type: doc

Learn production telescope integration recipes: overlap-save history, multi-beam batching, noise normalization, and gotchas to avoid.
:::

:::{grid-item-card} 📓 Interactive Tutorials
:link: tutorials/index
:link-type: doc

Hands-on Jupyter notebooks covering filterbanks, tree stepping, sparse grids, low-bit quantization, and baseband coherent search.
:::

:::{grid-item-card} 📊 Benchmarks
:link: benchmarks
:link-type: doc

FDMT, FDMT-FFT and brute-force DDMT on CPUs and a GPU, on one fixed 4096-channel configuration: runtime, real-time throughput by input bit width, and operation counts.
:::

::::

---

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

getting_started/index
tutorials/index
pipeline_guide/index
benchmarks
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

---

## Algorithm References

`dmt` implements and optimizes several dedispersion algorithms based on the radio astronomy literature:

| Algorithm | Method & Description | Literature References |
| :--- | :--- | :--- |
| **FDMT** | Fast Dispersion Measure Transform ($O(N_t N_f \log_2 N_f)$) | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |
| **DDMT** | Direct Dedispersion (brute-force delay-and-sum, DM-tiled GPU & CPU) | Barsdell, B. R. et al. (2012), *MNRAS*, 422(1), 379–392. [doi:10.1111/j.1365-2966.2012.20622.x](https://doi.org/10.1111/j.1365-2966.2012.20622.x); AstroAccelerate (Dimoudi et al. 2018) |
| **SDMT** | Subband-shared Dedispersion (exact prefix tree / trie partial-sum sharing) | Naidu et al. (2024), AT-RASC 2024 ([doi:10.46620/ursiatrasc24/hbrq1825](http://dx.doi.org/10.46620/ursiatrasc24/hbrq1825)) |
| **CFDMT** | Coherent Baseband Dedispersion + Incoherent FDMT | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |
| **FDMT-FFT** | Frequency-Domain Phase-Shift Dedispersion (chirp phase rotation) | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |

### Citation

If you use `dmt` in your astronomical research or telescope processing pipelines, please cite:

1. **Zackay, B. & Ofek, E. O. (2017)**. *An Accurate and Efficient Algorithm for Detection of Radio Bursts with an Unknown Dispersion Measure, for Single-Dish Telescopes and Interferometers*. [The Astrophysical Journal](https://doi.org/10.3847/1538-4357/835/1/11), 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373)
2. **Kumar, P. & Zackay, B. (2026)**. *dmt: Fast Dispersion Measure Transform Library*. GitHub: [https://github.com/pravirkr/dmt](https://github.com/pravirkr/dmt)
