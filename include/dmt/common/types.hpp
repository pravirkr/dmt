#pragma once

#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

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
using DtGridType  = std::vector<SizeType>;

#ifdef DMT_ENABLE_CUDA
using ComplexTypeCUDA = cuda::std::complex<float>;
#endif

#ifdef DMT_ENABLE_CUDA
template <typename T>
using DeviceVector = thrust::device_vector<T>;
#endif

// Constants for dispersion calculations
inline constexpr float kDispCoeff = -2.0F;
inline constexpr float kDispConstLK =
    4.1488080e3; // L&K Handbook of Pulsar Astronomy
inline constexpr float kDispConstMT =
    1 / 2.41e-4; // TEMPO2, Manchester&Taylor (1972)
inline constexpr float kDispConstSI = 4.1488064e3; // SI value, Kulkarni (2020)
inline constexpr float kDispConst   = kDispConstMT;

// Utilities for aligning or enforcing hardware-specific requirements (e.g.,
// SIMD alignment)
template <typename T>
struct AlignedAllocator {
    static constexpr std::size_t kAlignment = alignof(std::max_align_t);

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, kAlignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t /*unused*/) noexcept { free(ptr); }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

template <typename T>
concept IntegralDataType = std::is_integral_v<T>;

/**
 * @enum BasebandDataOrder
 * @brief Describes the data order for different unpacking methods.
 *
 */
enum class BasebandDataOrder : uint8_t {
    kPRITF, /**< Polarisation-Real/Imag-time-frequency */
    kFTPRI, /**< Frequency-Time-Polarisation-Real/Imag */
    kRITFP  /**< Real/Imag-time-frequency-Polarisation */
};

static const std::unordered_map<std::string_view, BasebandDataOrder>
    kBasebandDataOrderMap = {{"FTPRI", BasebandDataOrder::kFTPRI},
                             {"PRITF", BasebandDataOrder::kPRITF},
                             {"RITFP", BasebandDataOrder::kRITFP}};

/**
 * @brief Set the number of OpenMP threads to use.
 *
 * @param nthreads The number of threads to use.
 * @return The number of threads actually used.
 */
inline int set_dmt_openmp_threads(int nthreads) {
#ifdef DMT_ENABLE_OPENMP
    if (nthreads <= 0) {
        nthreads = omp_get_max_threads();
    }
    omp_set_num_threads(nthreads);
    spdlog::debug("set_openmp_threads: Using {} OpenMP threads", nthreads);
#else
    // Warn if nthreads > 1 but OpenMP is not enabled
    if (nthreads > 1) {
        spdlog::warn(
            "set_openmp_threads: Warning - nthreads > 1 specified, but "
            "OpenMP is not enabled (DMT_ENABLE_OPENMP not defined).");
    }
    nthreads = 1;
#endif
    return nthreads;
}

} // namespace dmt

namespace dmt::backend {

/**
 * @brief Tag struct representing the CPU backend.
 */
struct CPU {};

/**
 * @brief Tag struct representing the CUDA backend.
 */
struct CUDA {};

/**
 * @brief Concept to constrain template parameters to valid execution backends.
 */
template <typename T>
concept ExecutionBackend = std::same_as<T, CPU> || std::same_as<T, CUDA>;

/**
 * @brief Helper struct to define backend-specific types.
 */
template <ExecutionBackend Backend>
struct BackendTypes; // Primary template (intentionally undefined)

/*
// Specialization for CPU backend
template <>
struct BackendTypes<CPU> {
    using ComplexType = ComplexType;
};

#ifdef DMT_ENABLE_CUDA
// Specialization for CUDA backend
template <>
struct BackendTypes<CUDA> {
    using ComplexType = ComplexTypeCUDA;
};
#endif // DMT_ENABLE_CUDA
*/
} // namespace dmt::backend
