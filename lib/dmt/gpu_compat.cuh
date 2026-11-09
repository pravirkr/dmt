#pragma once

// One GPU source for CUDA and HIP. Every CUDA-dialect file (lib/cuda/*.cu,
// lib/dmt/*.cuh, the GPU tests and benchmarks) includes this header instead of
// <cuda_runtime.h>, <cufft.h> and <cuda/std/...>, and keeps using the CUDA
// names. In a HIP build (DMT_ENABLE_HIP, compiled by hip-clang) the names
// below are mapped onto HIP, hipFFT and rocThrust; no CUDA header is ever
// included there, so the macros cannot collide with the real CUDA API.
//
// Only the names this library uses are mapped: add a line when a new one is
// needed. Kernel-side syntax (__global__, <<<>>>, threadIdx, __syncthreads,
// __restrict__) is the same in both languages.

#if defined(DMT_ENABLE_HIP)

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <thrust/system/hip/execution_policy.h>

// libcu++ (cuda::std) is provided by libhipcxx when it is installed; the
// library only needs span and complex from it, so fall back to std::span and
// thrust::complex (rocThrust) otherwise. hip-clang compiles constexpr
// functions (all of std::span) for the device as well.
#if __has_include(<cuda/std/span>) && __has_include(<cuda/std/complex>)
#include <cuda/std/complex>
#include <cuda/std/span>
#else
#include <span>
#include <type_traits>

#include <thrust/complex.h>
namespace cuda::std {
using ::std::false_type;
using ::std::span;
using ::std::true_type;
template <typename T> using complex = ::thrust::complex<T>;
} // namespace cuda::std
#endif

// rocThrust spells the CUDA system `thrust::hip`.
namespace thrust {
namespace cuda = ::thrust::hip;
} // namespace thrust

#define DMT_GPU_NAME "hip"

// Runtime: types and constants
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorInvalidDevice hipErrorInvalidDevice
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaDevAttrMaxSharedMemoryPerBlockOptin                                \
    hipDeviceAttributeSharedMemPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor                            \
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaFuncAttributeMaxDynamicSharedMemorySize                            \
    hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaDevAttrL2CacheSize hipDeviceAttributeL2CacheSize
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount

// Runtime: functions
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemGetInfo hipMemGetInfo
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
// HIP's C++ overload takes the kernel itself, as CUDA's does.
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor                          \
    hipOccupancyMaxActiveBlocksPerMultiprocessor
// hipFuncSetAttribute takes the kernel as `const void*`. Variadic: the
// kernel argument is often a template-id whose commas would split it.
namespace dmt::gpu_compat {
template <typename Kernel>
inline hipError_t
func_set_attribute(Kernel kernel, hipFuncAttribute attr, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr,
                               value);
}
} // namespace dmt::gpu_compat
#define cudaFuncSetAttribute(...)                                              \
    ::dmt::gpu_compat::func_set_attribute(__VA_ARGS__)

// cuFFT -> hipFFT (same API, hipfft/HIPFFT_ prefix)
#define cufftResult hipfftResult
#define cufftHandle hipfftHandle
#define cufftType hipfftType
#define cufftComplex hipfftComplex
#define cufftCreate hipfftCreate
#define cufftDestroy hipfftDestroy
#define cufftMakePlanMany hipfftMakePlanMany
#define cufftMakePlanMany64 hipfftMakePlanMany64
#define cufftSetAutoAllocation hipfftSetAutoAllocation
#define cufftSetWorkArea hipfftSetWorkArea
#define cufftSetStream hipfftSetStream
#define cufftExecC2C hipfftExecC2C
#define cufftExecR2C hipfftExecR2C
#define cufftExecC2R hipfftExecC2R
#define CUFFT_SUCCESS HIPFFT_SUCCESS
#define CUFFT_INVALID_PLAN HIPFFT_INVALID_PLAN
#define CUFFT_ALLOC_FAILED HIPFFT_ALLOC_FAILED
#define CUFFT_INVALID_TYPE HIPFFT_INVALID_TYPE
#define CUFFT_INVALID_VALUE HIPFFT_INVALID_VALUE
#define CUFFT_INTERNAL_ERROR HIPFFT_INTERNAL_ERROR
#define CUFFT_EXEC_FAILED HIPFFT_EXEC_FAILED
#define CUFFT_SETUP_FAILED HIPFFT_SETUP_FAILED
#define CUFFT_INVALID_SIZE HIPFFT_INVALID_SIZE
#define CUFFT_UNALIGNED_DATA HIPFFT_UNALIGNED_DATA
#define CUFFT_C2C HIPFFT_C2C
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2R HIPFFT_C2R
#define CUFFT_FORWARD HIPFFT_FORWARD
#define CUFFT_INVERSE HIPFFT_BACKWARD

#else // CUDA

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>

#define DMT_GPU_NAME "cuda"

#endif

// True in the device compilation pass of either language.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define DMT_GPU_DEVICE_PASS 1
#else
#define DMT_GPU_DEVICE_PASS 0
#endif

// Warp shuffles over a logical warp of kLogicalWarp = 32 lanes, the warp the
// kernels' lane/warp arithmetic assumes. On CUDA they are the full-mask
// _sync intrinsics (width warpSize = 32: the same instructions as calling
// them directly). On HIP a 64-lane wavefront holds two logical warps, and the
// width argument keeps every shuffle inside its own 32-lane half; the
// _sync forms are not used (older ROCm lacks them). Callers keep all lanes of
// a logical warp converged at the shuffle.
namespace dmt::gpu_compat {
inline constexpr int kLogicalWarp = 32;

// Device code only: the header also reaches host translation units.
#if defined(__CUDACC__) || defined(__HIPCC__)
/// Value of lane (lane - delta) of this logical warp (own value below delta).
template <typename T>
__device__ __forceinline__ T shfl_up(T value, unsigned delta) {
#if defined(DMT_ENABLE_HIP)
    return __shfl_up(value, delta, kLogicalWarp);
#else
    return __shfl_up_sync(0xFFFFFFFFU, value, delta);
#endif
}

/// Value of lane `src` of this logical warp.
template <typename T> __device__ __forceinline__ T shfl(T value, int src) {
#if defined(DMT_ENABLE_HIP)
    return __shfl(value, src, kLogicalWarp);
#else
    return __shfl_sync(0xFFFFFFFFU, value, src);
#endif
}
#endif // __CUDACC__ || __HIPCC__
} // namespace dmt::gpu_compat
