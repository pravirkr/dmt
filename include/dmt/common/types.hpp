#pragma once

/**
 * @file types.hpp
 * @brief Common type definitions, astrophysical dispersion constants, and
 * memory utilities.
 */

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif

namespace dmt {

// Basic Type Definitions
using SizeType    = std::size_t;    // Common size type
using IndexType   = std::ptrdiff_t; // Common index type (for signed indexing)
using ComplexType = std::complex<float>;

#ifdef DMT_ENABLE_CUDA
using ComplexTypeCUDA = cuda::std::complex<float>;
#endif

#ifdef DMT_ENABLE_CUDA
template <typename T> using DeviceVector = thrust::device_vector<T>;
#endif

// Constants for dispersion calculations
inline constexpr float kDispCoeff = -2.0F;
inline constexpr float kDispConstLK =
    4.1488080e3; // L&K Handbook of Pulsar Astronomy
inline constexpr float kDispConstMT =
    1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
inline constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
inline constexpr float kDispConst   = kDispConstMT;

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
 * @enum BasebandDataOrder
 * @brief Memory layouts for raw telescope complex baseband voltage streams.
 */
enum class BasebandDataOrder : uint8_t {
    kPRITF, /**< Polarization -> Real/Imag -> Time -> Frequency (LOFAR default)
             */
    kFTPRI, /**< Frequency -> Time -> Polarization -> Real/Imag */
    kRITFP, /**< Real/Imag -> Time -> Frequency -> Polarization */
};

/**
 * @brief FDMT mode for the FDMT tree.
 */
enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

// Callable from both host and device when compiled with nvcc.
#if defined(DMT_ENABLE_CUDA) && defined(__CUDACC__)
#define DMT_HD __host__ __device__
#define DMT_D __device__
#define DMT_H __host__
#else
#define DMT_HD
#define DMT_D
#define DMT_H
#endif

} // namespace dmt
