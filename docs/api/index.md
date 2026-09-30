# API Reference

Complete public API documentation for both Python and C++ interfaces of `dmt`.

```{toctree}
:maxdepth: 2

python_api
cpp_api
```

---

## Python Modules

- `dmtlib`: Core engines, plans, and convenience functions (`FDMT`, `DDMT`, `SDMT`, `CohFDMT`, `FDMTFFT`, `DDMTFFT`, `compute_fdmt`, `FDMTPlan`, `available_backends`), and the CPU FFT planner controls (`set_fft_planner`, `import_fft_wisdom`, `export_fft_wisdom`).
- `dmtlib.simulate`: Astronomical pulse injection and dispersion modeling (`generate_frb`, `generate_pure_frb`, `generate_dispersed_periodic_signal`).
- `dmtlib.grid`: Smearing-aware sparse DM grid generators (`snr_loss`, `levin`, and the piecewise-uniform `levin_piecewise` for `DDMTFFT`).

## C++ Namespaces

- `dmt`: Backend selection (`Backend`, `Exec`, `DeviceSpan`, `Stream`, `available_backends`).
- `dmt::algorithms`: Compute engines (`FDMT`, `DDMT`, `SDMT`, `CohFDMT`, `FDMTFFT`, `DDMTFFT`); each runs on the backend chosen by its `Exec` argument.
- `dmt::fft`: Process-wide CPU FFT (FFTW) planner effort and wisdom (`set_planner`, `import_wisdom`, `export_wisdom`; `dmt/common/fft_config.hpp`).
- `dmt::plans`: Execution plans and coordinate containers (`FDMTPlan`, `DDMTPlan`, `CohFDMTPlan`).
- `dmt`: Dispersion constants in the top-level namespace (`kDispConstLK`, `kDispConstMT`, `kDispConst`).
- `dmt::utils`: Simulation utilities.
