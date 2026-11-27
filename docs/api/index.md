# API reference

The public Python and C++ APIs of `dmt`.

```{toctree}
:maxdepth: 2
:hidden:

python_api
cpp_api
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Python API
:link: python_api
:link-type: doc

The `dmtlib` package: engines, plans, candidate search and convenience
functions.
:::

:::{grid-item-card} C++ API
:link: cpp_api
:link-type: doc

The `dmt` namespaces: backend selection, engines, plans and utilities.
:::
::::

## Python modules

- `dmtlib`: engines, plans, candidate search and convenience functions (`FDMT`, `DDMT`, `SDMT`, `CohFDMT`, `FDMTFFT`, `DDMTFFT`, `FDMTBoxcar`, `FDMTFFTBoxcar`, `compute_fdmt`, `FDMTPlan`, `available_backends`), and the CPU FFT planner controls (`set_fft_planner`, `import_fft_wisdom`, `export_fft_wisdom`).
- `dmtlib.dedisperse`: one-call dedispersion of a whole waterfall into complete time series (`dedisperse`, `Dedispersed`, `dispersion_delay`, also exported from `dmtlib`).
- `dmtlib.simulate`: pulse injection and dispersion modelling (`generate_frb`, `generate_pure_frb`, `generate_dispersed_periodic_signal`).
- `dmtlib.grid`: DM grid generators (`generate_dm_grid` by S/N loss over every channel, `generate_levin_dm_grid` by the Levin rule, both optionally piecewise uniform for `DDMTFFT`) and `calculate_snr_loss`, from C++ `dmt/common/dm_grids.hpp`.

## C++ namespaces

- `dmt`: backend selection (`Backend`, `Exec`, `DeviceSpan`, `Stream`, `available_backends`).
- `dmt::algorithms`: engines (`FDMT`, `DDMT`, `SDMT`, `CohFDMT`, `FDMTFFT`, `DDMTFFT`); each runs on the backend chosen by its `Exec` argument.
- `dmt::fft`: process-wide CPU FFT (FFTW) planner effort and wisdom (`set_planner`, `import_wisdom`, `export_wisdom`; `dmt/common/fft_config.hpp`).
- `dmt::plans`: execution plans, coordinate containers and DM-grid generators (`FDMTPlan`, `DDMTPlan`, `CohFDMTPlan`, `DMGridConfig`, `generate_dm_grid`, `generate_levin_dm_grid`, `calculate_snr_loss`).
- `dmt`: dispersion constants in the top-level namespace (`kDispConstLK`, `kDispConstMT`, `kDispConstSI`, `kDispConst`; all `double`) and `dispersion_delay(dm, f_lo_mhz, f_hi_mhz)` (seconds, as `dmtlib.dispersion_delay`).
- `dmt::utils`: simulation utilities.
