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
- **DDMT**: Direct Dedispersion with 1/2/4/8/32-bit SIMD kernels
- **CFDMT**: Hybrid Coherent baseband dedispersion for microsecond pulses
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

::::

---

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

getting_started/index
tutorials/index
pipeline_guide/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

---

## Citation & References

If you use `dmt` in your astronomical research or telescope data processing, please cite:

1. **Zackay, B., & Ofek, E. O. (2017)**. *An Accurate and Efficient Algorithm for Detection of Radio Bursts with an Unknown Dispersion Measure, for Single-Dish Telescopes and Interferometers*. [The Astrophysical Journal](https://doi.org/10.3847/1538-4357/835/1/11), 835(1), 11. [arXiv:1411.5373](https://arxiv.org/abs/1411.5373)
2. **Kumar, P. & Zackay, B. (2026)**. *dmt: Fast Dispersion Measure Transform Library*. GitHub: [https://github.com/pravirkr/dmt](https://github.com/pravirkr/dmt)
