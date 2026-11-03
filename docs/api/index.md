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
- `dmtlib.grid`: smearing-aware sparse DM grid generators (`generate_optimal_dm_grid` with `method="snr_loss"`, `"levin"`, or the piecewise-uniform `"levin_piecewise"` for `DDMTFFT`).

## C++ namespaces

- `dmt`: backend selection (`Backend`, `Exec`, `DeviceSpan`, `Stream`, `available_backends`).
- `dmt::algorithms`: engines (`FDMT`, `DDMT`, `SDMT`, `CohFDMT`, `FDMTFFT`, `DDMTFFT`); each runs on the backend chosen by its `Exec` argument.
- `dmt::fft`: process-wide CPU FFT (FFTW) planner effort and wisdom (`set_planner`, `import_wisdom`, `export_wisdom`; `dmt/common/fft_config.hpp`).
- `dmt::plans`: execution plans and coordinate containers (`FDMTPlan`, `DDMTPlan`, `CohFDMTPlan`).
- `dmt`: dispersion constants in the top-level namespace (`kDispConstLK`, `kDispConstMT`, `kDispConst`).
- `dmt::utils`: simulation utilities.
