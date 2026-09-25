#pragma once

/**
 * @file dmt.hpp
 * @brief Main entry point for the Dispersion Measure Transform (DMT) C++
 * library.
 *
 * This header aggregates the core algorithms, execution plans, common types,
 * and simulation utilities provided by DMT:
 * - Incoherent Fast Dispersion Measure Transform (FDMT): @ref
 * dmt::algorithms::FDMTCPU, @ref dmt::algorithms::FDMTCUDA
 * - Direct Dedispersion Transform (DDMT): @ref dmt::algorithms::DDMTCPU, @ref
 * dmt::algorithms::DDMTCUDA
 * - Coherent Fast Dispersion Measure Transform (CFDMT): @ref
 * dmt::algorithms::CohFDMTCPU, @ref dmt::algorithms::CohFDMTCUDA
 * - Fourier-Shift FDMT (FDMT-FFT): @ref dmt::algorithms::FDMTFFTCPU
 * - Execution Plans and Grid Generators: @ref dmt::plans::FDMTPlan, @ref
 * dmt::plans::DDMTPlan, @ref dmt::plans::CohFDMTPlan
 * - Utilities & Data Unpackers: @ref dmt::utils::DataUnpackerCPU, @ref
 * dmt::utils::generate_pure_frb
 */

// Import common types and constants
#include "common/plans.hpp"   // IWYU pragma: export
#include "common/types.hpp"   // IWYU pragma: export
#include "utils/simulate.hpp" // IWYU pragma: export

// Include headers for each algorithm
#include "algorithms/cfdmt.hpp"    // IWYU pragma: export
#include "algorithms/ddmt.hpp"     // IWYU pragma: export
#include "algorithms/fdmt.hpp"     // IWYU pragma: export
#include "algorithms/fdmt_fft.hpp" // IWYU pragma: export
