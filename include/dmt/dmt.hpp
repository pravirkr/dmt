#pragma once

/**
 * @file dmt.hpp
 * @brief Main entry point for the Dispersion Measure Transform (DMT) C++
 * library.
 *
 * This header aggregates the core algorithms, execution plans, common types,
 * and simulation utilities provided by DMT. Every algorithm is one class whose
 * backend (CPU, CUDA, ...) is chosen at construction with dmt::Exec; see
 * dmt::available_backends().
 * - Incoherent Fast Dispersion Measure Transform (FDMT): @ref
 * dmt::algorithms::FDMT
 * - Direct Dedispersion Transform (DDMT): @ref dmt::algorithms::DDMT
 * - Coherent Fast Dispersion Measure Transform (CFDMT): @ref
 * dmt::algorithms::CohFDMT
 * - Fourier-Shift FDMT with exact fractional delays (FDMT-FFT): @ref dmt::algorithms::FDMTFFT
 * - Fourier-Shift direct dedispersion with exact fractional delays
 * (DDMT-FFT): @ref dmt::algorithms::DDMTFFT
 * - Execution Plans and Grid Generators: @ref dmt::plans::FDMTPlan, @ref
 * dmt::plans::DDMTPlan, @ref dmt::plans::CohFDMTPlan
 * - Simulation utilities: @ref dmt::utils::generate_pure_frb
 */

// Import common types and constants
#include "common/backend.hpp" // IWYU pragma: export
#include "common/fft_config.hpp" // IWYU pragma: export
#include "common/logging.hpp" // IWYU pragma: export
#include "common/plans.hpp"   // IWYU pragma: export
#include "common/types.hpp"   // IWYU pragma: export
#include "utils/simulate.hpp" // IWYU pragma: export

// Include headers for each algorithm
#include "algorithms/cfdmt.hpp"    // IWYU pragma: export
#include "algorithms/ddmt.hpp"     // IWYU pragma: export
#include "algorithms/ddmt_fft.hpp" // IWYU pragma: export
#include "algorithms/fdmt.hpp"     // IWYU pragma: export
#include "algorithms/fdmt_fft.hpp" // IWYU pragma: export
#include "algorithms/sdmt.hpp"     // IWYU pragma: export
