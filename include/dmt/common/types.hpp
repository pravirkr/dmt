#pragma once

#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/complex>
#include <cuda/std/span>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif

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

// Specialization for CPU backend
template <>
struct BackendTypes<CPU> {
    using ComplexType = ComplexType;
    template <typename T>
    using SpanType = std::span<T>;
};

#ifdef DMT_ENABLE_CUDA
// Specialization for CUDA backend
template <>
struct BackendTypes<CUDA> {
    using ComplexType = ComplexTypeCUDA;
    template <typename T>
    using SpanType = cuda::std::span<T>;
};
#endif // DMT_ENABLE_CUDA

} // namespace dmt::backend
