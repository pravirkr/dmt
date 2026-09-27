#pragma once

#include <cstdio>
#include <stdexcept>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

// Shared by the CUDA benchmarks (dmt_bench and dmt_bench_suite).

#define BENCH_CUDA_TRY(call)                                                   \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            throw std::runtime_error(cudaGetErrorString(status));              \
        }                                                                      \
    } while (0)

#define BENCH_CUDA_CHECK_NOTHROW(call)                                         \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            std::fprintf(stderr, "CUDA error in destructor: %s\n",             \
                         cudaGetErrorString(status));                          \
        }                                                                      \
    } while (0)

/**
 * @brief RAII manual timing range for one benchmark iteration, measured with
 * CUDA events on `stream` (after https://github.com/jrhemstad/
 * example_cuda_benchmark). Register the benchmark with UseManualTime().
 */
class CudaEventTimer {
public:
    explicit CudaEventTimer(benchmark::State& state,
                            cudaStream_t stream = nullptr)
        : m_stream(stream),
          m_state(&state) {
        BENCH_CUDA_TRY(cudaEventCreate(&m_start));
        BENCH_CUDA_TRY(cudaEventCreate(&m_stop));
        BENCH_CUDA_TRY(cudaEventRecord(m_start, m_stream));
    }

    CudaEventTimer()                                 = delete;
    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&&)                 = delete;
    CudaEventTimer& operator=(CudaEventTimer&&)      = delete;

    ~CudaEventTimer() {
        BENCH_CUDA_CHECK_NOTHROW(cudaEventRecord(m_stop, m_stream));
        BENCH_CUDA_CHECK_NOTHROW(cudaEventSynchronize(m_stop));
        float milliseconds = 0.0F;
        BENCH_CUDA_CHECK_NOTHROW(
            cudaEventElapsedTime(&milliseconds, m_start, m_stop));
        m_state->SetIterationTime(milliseconds / 1000.0F);
        BENCH_CUDA_CHECK_NOTHROW(cudaEventDestroy(m_start));
        BENCH_CUDA_CHECK_NOTHROW(cudaEventDestroy(m_stop));
    }

private:
    cudaEvent_t m_start{};
    cudaEvent_t m_stop{};
    cudaStream_t m_stream;
    benchmark::State* m_state;
};
