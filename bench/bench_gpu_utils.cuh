#pragma once

#include <cstdio>
#include <format>
#include <stdexcept>

#include <benchmark/benchmark.h>

#include "dmt/common/backend.hpp"
#include "dmt/gpu_compat.cuh"

// Shared by the GPU benchmarks (dmt_bench and dmt_bench_suite), built for
// CUDA or HIP.

/// The GPU backend of this build, on device `device`.
inline dmt::Exec bench_gpu_exec(int device = 0) {
#ifdef DMT_ENABLE_HIP
    return dmt::Exec::hip(device);
#else
    return dmt::Exec::cuda(device);
#endif
}

#define BENCH_GPU_TRY(call)                                                    \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            throw std::runtime_error(std::format("{} runtime error: {}",       \
                                                 DMT_GPU_NAME,                 \
                                                 cudaGetErrorString(status))); \
        }                                                                      \
    } while (0)

#define BENCH_GPU_CHECK_NOTHROW(call)                                          \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            std::fprintf(stderr, "%s runtime error in destructor: %s\n",       \
                         DMT_GPU_NAME, cudaGetErrorString(status));            \
        }                                                                      \
    } while (0)

/**
 * @brief RAII manual timing range for one benchmark iteration, measured with
 * GPU events on `stream`.
 */
class GPUEventTimer {
public:
    explicit GPUEventTimer(benchmark::State& state,
                           cudaStream_t stream = nullptr)
        : m_stream(stream),
          m_state(&state) {
        BENCH_GPU_TRY(cudaEventCreate(&m_start));
        BENCH_GPU_TRY(cudaEventCreate(&m_stop));
        BENCH_GPU_TRY(cudaEventRecord(m_start, m_stream));
    }

    GPUEventTimer()                                = delete;
    GPUEventTimer(const GPUEventTimer&)            = delete;
    GPUEventTimer& operator=(const GPUEventTimer&) = delete;
    GPUEventTimer(GPUEventTimer&&)                 = delete;
    GPUEventTimer& operator=(GPUEventTimer&&)      = delete;

    ~GPUEventTimer() {
        BENCH_GPU_CHECK_NOTHROW(cudaEventRecord(m_stop, m_stream));
        BENCH_GPU_CHECK_NOTHROW(cudaEventSynchronize(m_stop));
        float milliseconds = 0.0F;
        BENCH_GPU_CHECK_NOTHROW(
            cudaEventElapsedTime(&milliseconds, m_start, m_stop));
        m_state->SetIterationTime(milliseconds / 1000.0F);
        BENCH_GPU_CHECK_NOTHROW(cudaEventDestroy(m_start));
        BENCH_GPU_CHECK_NOTHROW(cudaEventDestroy(m_stop));
    }

private:
    cudaEvent_t m_start{};
    cudaEvent_t m_stop{};
    cudaStream_t m_stream;
    benchmark::State* m_state;
};
