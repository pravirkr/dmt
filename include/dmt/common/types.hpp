#pragma once

/**
 * @file types.hpp
 * @brief Common type definitions, astrophysical dispersion constants, and
 * memory utilities.
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dmt {

// Basic Type Definitions
using SizeType    = std::size_t;    // Common size type
using IndexType   = std::ptrdiff_t; // Common index type (for signed indexing)
using ComplexType = std::complex<float>;

// Constants for dispersion calculations
inline constexpr float kDispCoeff = -2.0F;
inline constexpr float kDispConstLK =
    4.1488080e3; // L&K Handbook of Pulsar Astronomy
inline constexpr float kDispConstMT =
    1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
inline constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
inline constexpr float kDispConst   = kDispConstMT;

/**
 * @brief Dispersion delay in seconds of `f_lo_mhz` behind `f_hi_mhz`.
 *
 * Uses the library's constant (1 / 2.41e-4 s MHz^2 cm^3 / pc, the TEMPO2
 * convention of kDispConst), in double precision, as Python's
 * `dmtlib.dispersion_delay`. `f_hi_mhz` = +infinity gives the delay from
 * infinite frequency. FDMTPlan frequencies are channel edges: a candidate's
 * f_low_mhz / f_high_mhz are its subband's lower and upper edges.
 */
[[nodiscard]] constexpr double
dispersion_delay(double dm, double f_lo_mhz, double f_hi_mhz) noexcept {
    constexpr double kConst = 1.0 / 2.41e-4;
    // 1 / inf^2 is 0 in IEEE arithmetic (also under -ffast-math: the
    // division is not folded for a runtime argument).
    return kConst * dm *
           ((1.0 / (f_lo_mhz * f_lo_mhz)) - (1.0 / (f_hi_mhz * f_hi_mhz)));
}

/**
 * @brief POSIX aligned memory allocator for SIMD vectorization and cache
 * alignment.
 * @tparam T Value type to allocate.
 */
template <typename T> struct AlignedAllocator {
    static constexpr std::size_t kAlignment = alignof(std::max_align_t);

    /**
     * @brief Allocates aligned memory.
     * @param n Number of elements to allocate.
     * @return Pointer to aligned memory buffer.
     * @throws std::bad_alloc if allocation fails.
     */
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, kAlignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(ptr);
    }

    /**
     * @brief Deallocates aligned memory.
     * @param ptr Pointer previously allocated with allocate().
     */
    void deallocate(T* ptr, std::size_t /*unused*/) noexcept { free(ptr); }
};

template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

template <typename T>
concept IntegralDataType = std::is_integral_v<T>;

/**
 * @brief FDMT mode for the FDMT tree.
 */
enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

// Callable from both host and device when compiled with nvcc or hip-clang
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define DMT_HD __host__ __device__
#define DMT_D __device__
#define DMT_H __host__
#else
#define DMT_HD
#define DMT_D
#define DMT_H
#endif

} // namespace dmt
