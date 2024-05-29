# dmt

[![GitHub CI](https://github.com/pravirkr/dmt/actions/workflows/ci.yml/badge.svg)](https://github.com/pravirkr/dmt/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pravirkr/dmt/graph/badge.svg?token=17BGN5IIM9)](https://codecov.io/gh/pravirkr/dmt)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpravirkr%2Fdmt%2Fmain%2Fpyproject.toml)
![C++ Version](https://img.shields.io/badge/C%2B%2B-17-blue)
[![License](https://img.shields.io/github/license/pravirkr/dmt)](https://github.com/pravirkr/dmt/blob/main/LICENSE)

## Dispersion Measure Transforms
|           |           |
| --------- | --------- |
| ![](docs/waterfall.png) | ![](docs/dmt.png) |


## Installation

Using [pip](https://pip.pypa.io):

```bash
pip install -U git+https://github.com/pravirkr/dmt
```

## Usage

```python
from dmtlib import FDMTCPU

frb = np.ones((nchans, nsamps), dtype=np.float32)
thefdmt = FDMTCPU(f_min, f_max, nchans, nsamps, tsamp, dt_max=dt_max, dt_min=0, dt_step=1)
dmt_transform = thefdmt.execute(frb.astype(np.float32))
```

## Benchmarks

```python
f_min = 704.0, f_max = 1216.0, nchans = 4096, tsamp = 0.00008192, dt_max = 2048, nsamps = n;
nthreads = 1, 8;
```
![](bench/results/bench.png)


