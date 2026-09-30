#pragma once

/**
 * @file fft_config.hpp
 * @brief Process-wide settings of the CPU FFT backend (FFTW).
 *
 * The Fourier engines (FDMTFFT, DDMTFFT, CohFDMT) plan their CPU transforms
 * at construction. These settings apply to every plan created after the call;
 * existing engines keep their plans. The GPU backends (cuFFT/hipFFT) ignore
 * them.
 */

#include <cstdint>
#include <string>

namespace dmt::fft {

/// @brief FFTW planner effort. Higher effort times candidate algorithms at
/// plan time: slower engine construction, often faster transforms.
enum class Planner : std::uint8_t {
    kEstimate   = 0, ///< Heuristic plans, no timing (default).
    kMeasure    = 1, ///< Time a few candidates (typically seconds).
    kPatient    = 2, ///< Time many candidates (can take minutes).
    kExhaustive = 3, ///< Time every candidate.
};

/// @brief Sets the planner effort used for CPU FFT plans created from now on.
void set_planner(Planner planner) noexcept;
/// @brief Planner effort currently in effect.
[[nodiscard]] Planner get_planner() noexcept;

/// @brief Loads FFTW wisdom (earlier measured plans) from a file, so later
/// kMeasure/kPatient plans of the same sizes are instant.
/// @return True if the file was read and accepted.
bool import_wisdom(const std::string& path);
/// @brief Saves the FFTW wisdom accumulated by this process to a file.
/// @return True on success.
bool export_wisdom(const std::string& path);
/// @brief Forgets all accumulated FFTW wisdom.
void forget_wisdom() noexcept;

} // namespace dmt::fft
