# API Reference

Complete public API documentation for both Python and C++ interfaces of `dmt`.

```{toctree}
:maxdepth: 2

python_api
cpp_api
```

---

## Python Modules

- `dmtlib`: Core engines, plans, and convenience functions (`FDMTCPU`, `DDMTCPU`, `CohFDMTCPU`, `compute_fdmt`, `FDMTPlan`).
- `dmtlib.simulate`: Astronomical pulse injection and dispersion modeling (`generate_frb`, `generate_pure_frb`, `generate_dispersed_periodic_signal`).
- `dmtlib.grid`: Smearing-aware sparse DM grid generators.

## C++ Namespaces

- `dmt::algorithms`: Compute engines (`FDMTCPU`, `DDMTCPU`, `CohFDMTCPU`, `FDMTFFTCPU`, and CUDA variants).
- `dmt::plans`: Execution plans and coordinate containers (`FDMTPlan`, `DDMTPlan`, `CohFDMTPlan`).
- `dmt`: Dispersion constants in the top-level namespace (`kDispConstLK`, `kDispConstMT`, `kDispConst`).
- `dmt::utils`: Simulation utilities, FFT wrappers, and baseband unpackers.
