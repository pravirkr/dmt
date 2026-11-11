# dmt: Dispersion Measure Transform

```{raw} html
<div class="dmt-hero-card">
  <img class="dmt-hero-mark only-light" src="_static/logo/dmt-mark.svg" alt="">
  <img class="dmt-hero-mark only-dark" src="_static/logo/dmt-mark-dark.svg" alt="">
  <div class="dmt-hero-body">
  <p class="dmt-hero-lede">
    Fast dedispersion for radio astronomy: FDMT, direct and Fourier-domain
    transforms in one library. A <strong>C++20</strong> core that runs on CPUs
    (<strong>OpenMP</strong>, SIMD) and GPUs (<strong>CUDA</strong>, <strong>HIP</strong>),
    with <strong>Python</strong> bindings that work on NumPy arrays without copies.
  </p>
  <div>
    <span class="dmt-badge dmt-badge-blue">C++20</span>
    <span class="dmt-badge dmt-badge-purple">Python 3.12+</span>
    <span class="dmt-badge dmt-badge-green">CUDA &amp; HIP</span>
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

:::{grid-item-card} 🚀 Algorithm Suite
:class-card: sd-shadow-sm

- **FDMT & FDMT-FFT**: the fast tree, in time or Fourier domain.
- **DDMT, SDMT & DDMT-FFT**: direct delay-and-sum, exact or fractional.
- **CohFDMT**: coherent baseband dedispersion, then FDMT.
- **FDMTBoxcar**: FDMT fused with a boxcar candidate search.
:::

:::{grid-item-card} ⚡ Hardware Acceleration
:class-card: sd-shadow-sm

- **CPU**: OpenMP threads, AVX2, AVX-512 and ARM NEON.
- **GPU**: CUDA and HIP kernels for every engine.
- **Low-bit input**: packed 1–16 bit samples, read directly.
- **Transfers**: pinned memory, copies overlapped with compute.
:::

:::{grid-item-card} 📡 Pipeline Integration
:class-card: sd-shadow-sm

- **Streaming**: blocks join bit-exactly, cut anywhere.
- **Multi-beam**: many beams through one plan per call.
- **DM grids**: custom, sparse and negative DM trials.
- **Noise**: exact sigma of every DM trial, from the plan.
:::

:::{grid-item-card} 🐍 Python & C++ Parity
:class-card: sd-shadow-sm

- **C++20 core**: RAII plans and engines, memory allocated up front.
- **`dmtlib`**: the same engines on NumPy arrays, without copies.
- **Tested**: Catch2 and pytest, bit-exact CPU/GPU checks.
- **Documented**: tutorials, pipeline guides, benchmarks.
:::

::::

---

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 🏁 Getting Started
:link: getting_started/index
:link-type: doc

Install the Python package or the C++ library, and dedisperse your first waterfall in a few lines.
:::

:::{grid-item-card} 🛠️ Pipeline Guide
:link: pipeline_guide/index
:link-type: doc

Putting `dmt` in a pipeline: streaming history, noise normalisation, periodicity searches, multi-beam batching, and the quirks to know about.
:::

:::{grid-item-card} 📓 Interactive Tutorials
:link: tutorials/index
:link-type: doc

Notebooks from dispersion basics to the FDMT tree, DM grids, FDMT versus DDMT, baseband search, streaming and noise statistics.
:::

:::{grid-item-card} 📊 Benchmarks
:link: benchmarks
:link-type: doc

Every engine on one shared cluster node (AMD EPYC 9654 CPU, NVIDIA L40S GPU), for three survey presets.
:::

::::

---

## Table of Contents

```{toctree}
:maxdepth: 1
:caption: User Guide

getting_started/index
tutorials/index
pipeline_guide/index
benchmarks
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/index
```

---

## Algorithm References

The algorithms in `dmt` come from the following papers:

| Algorithm | Method & Description | Literature References |
| :--- | :--- | :--- |
| **FDMT** | Fast Dispersion Measure Transform ($O(N_t N_f \log_2 N_f)$) | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |
| **DDMT** | Direct Dedispersion (brute-force delay-and-sum, DM-tiled GPU & CPU) | Barsdell, B. R. et al. (2012), *MNRAS*, 422(1), 379–392. [doi:10.1111/j.1365-2966.2012.20622.x](https://doi.org/10.1111/j.1365-2966.2012.20622.x); AstroAccelerate (Dimoudi et al. 2018) |
| **SDMT** | Subband-shared direct dedispersion (exact partial sums shared between DMs) | Naidu et al. (2024), AT-RASC 2024 ([doi:10.46620/ursiatrasc24/hbrq1825](http://dx.doi.org/10.46620/ursiatrasc24/hbrq1825)) |
| **CohFDMT** | Coherent dedispersion within channels, then incoherent FDMT | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |
| **FDMT-FFT** | The FDMT tree in the Fourier domain, with fractional (phase-ramp) delays | Zackay, B. & Ofek, E. O. (2017), *The Astrophysical Journal*, 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373) |

### Citation

If `dmt` is useful in your research or pipeline, please cite:

1. **Zackay, B. & Ofek, E. O. (2017)**. *An Accurate and Efficient Algorithm for Detection of Radio Bursts with an Unknown Dispersion Measure, for Single-Dish Telescopes and Interferometers*. [The Astrophysical Journal](https://doi.org/10.3847/1538-4357/835/1/11), 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373)
2. **Kumar, P. & Zackay, B. (2026)**. *dmt: Fast Dispersion Measure Transform Library*. GitHub: [https://github.com/pravirkr/dmt](https://github.com/pravirkr/dmt)
