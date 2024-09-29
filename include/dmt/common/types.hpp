#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#ifdef USE_CUDA
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif

// Basic Type Definitions
using SizeType    = std::size_t;    // Common size type
using IndexType   = std::ptrdiff_t; // Common index type (for signed indexing)
using ComplexType = std::complex<float>;
using DtGridType  = std::vector<SizeType>;

#ifdef USE_CUDA
using ComplexTypeCUDA = thrust::complex<float>;
#endif

#ifdef USE_CUDA
template <typename T> using DeviceVector = thrust::device_vector<T>;
#endif

// Utilities for aligning or enforcing hardware-specific requirements (e.g.,
// SIMD alignment)
template <typename T> struct AlignedAllocator {
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

template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;
