#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt.hpp"

// https://github.com/jrhemstad/example_cuda_benchmark
#define BENCH_CUDA_TRY(call)                                                   \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            throw std::runtime_error("CUDA error detected.");                  \
        }                                                                      \
    } while (0);

#define BENCH_CUDA_CHECK_NOTHROW(call)                                         \
    do {                                                                       \
        auto const status = (call);                                            \
        if (cudaSuccess != status) {                                           \
            std::fprintf(stderr, "CUDA error in destructor: %s\n",             \
                         cudaGetErrorString(status));                          \
        }                                                                      \
    } while (0)

class CudaEventTimer {
public:
    /**
     * @brief Constructs a `cuda_event_timer` beginning a manual timing range.
     *
     * Optionally flushes L2 cache.
     *
     * @param[in,out] state  This is the benchmark::State whose timer we are
     * going to update.
     * @param[in] flush_l2_cache_ whether or not to flush the L2 cache before
     *                            every iteration.
     * @param[in] m_stream The CUDA stream we are measuring time on.
     */
    explicit CudaEventTimer(benchmark::State& state,
                            bool flush_l2_cache = false,
                            cudaStream_t stream = 0)
        : m_stream(stream),
          m_state(&state) {
        // flush all of L2 cache
        if (flush_l2_cache) {
            int current_device = 0;
            BENCH_CUDA_TRY(cudaGetDevice(&current_device));

            int l2_cache_bytes = 0;
            BENCH_CUDA_TRY(cudaDeviceGetAttribute(
                &l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

            if (l2_cache_bytes > 0) {
                const int memset_value = 0;
                int* l2_cache_buffer   = nullptr;
                BENCH_CUDA_TRY(cudaMalloc(&l2_cache_buffer, l2_cache_bytes));
                BENCH_CUDA_TRY(cudaMemsetAsync(l2_cache_buffer, memset_value,
                                               l2_cache_bytes, m_stream));
                BENCH_CUDA_TRY(cudaFree(l2_cache_buffer));
            }
        }

        BENCH_CUDA_TRY(cudaEventCreate(&m_start));
        BENCH_CUDA_TRY(cudaEventCreate(&m_stop));
        BENCH_CUDA_TRY(cudaEventRecord(m_start, m_stream));
    }

    CudaEventTimer() = delete;

    /**
     * @brief Destroy the `cuda_event_timer` and ending the manual time range.
     *
     */
    ~CudaEventTimer() {
        BENCH_CUDA_CHECK_NOTHROW(cudaEventRecord(m_stop, m_stream));
        BENCH_CUDA_CHECK_NOTHROW(cudaEventSynchronize(m_stop));
        float milliseconds = 0.0F;
        BENCH_CUDA_CHECK_NOTHROW(
            cudaEventElapsedTime(&milliseconds, m_start, m_stop));
        m_state->SetIterationTime(milliseconds / (1000.0F));
        BENCH_CUDA_CHECK_NOTHROW(cudaEventDestroy(m_start));
        BENCH_CUDA_CHECK_NOTHROW(cudaEventDestroy(m_stop));
    }

private:
    cudaEvent_t m_start{};
    cudaEvent_t m_stop{};
    cudaStream_t m_stream;
    benchmark::State* m_state;
};

namespace {
template <typename T>
thrust::device_vector<T> generate_vector_device(size_t size) {
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);

    thrust::device_vector<T> vec(size);
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(size), vec.begin(),
        [=] __device__(size_t /*idx*/) mutable { return dist(rng); });

    return vec;
}
} // namespace

namespace dmt {
using algorithms::FDMTCUDA;
using plans::FDMTPlan;

class FDMTCUDAFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min       = 704.0F;
        f_max       = 1216.0F;
        nchans      = 4096;
        tsamp       = 0.00008192F;
        dt_max      = 2048;
        nsamps      = state.range(0);
        waterfall_d = generate_vector_device<float>(nchans * nsamps);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    void SetUp(benchmark::State& state) override {
        SetUp(static_cast<const benchmark::State&>(state));
    }

    void TearDown(benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{};
    size_t nchans{}, dt_max{}, nsamps{};
    thrust::device_vector<float> waterfall_d;
};

BENCHMARK_DEFINE_F(FDMTCUDAFixture, BM_fdmt_planBuffer_cuda)
(benchmark::State& state) {
    for (auto _ : state) {
        CudaEventTimer raii{state};
        FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTCUDAFixture, BM_fdmt_execute_cuda)
(benchmark::State& state) {
    FDMTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    thrust::device_vector<float> dmt_d(fdmt_cuda.get_plan().get_dmt_size(),
                                       0.0F);
    for (auto _ : state) {
        CudaEventTimer raii{state};
        fdmt_cuda.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()),
                waterfall_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size()));
    }
}

BENCHMARK_DEFINE_F(FDMTCUDAFixture, BM_fdmt_overall_cuda)
(benchmark::State& state) {
    FDMTPlan tmp_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    thrust::device_vector<float> dmt_d(tmp_plan.get_dmt_size(), 0.0F);
    for (auto _ : state) {
        CudaEventTimer raii{state};
        FDMTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        fdmt_cuda.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()),
                waterfall_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size()));
    }
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 15;

BENCHMARK_REGISTER_F(FDMTCUDAFixture, BM_fdmt_planBuffer_cuda) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2)})
    ->UseManualTime();

BENCHMARK_REGISTER_F(FDMTCUDAFixture, BM_fdmt_execute_cuda) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2)})
    ->UseManualTime();

BENCHMARK_REGISTER_F(FDMTCUDAFixture, BM_fdmt_overall_cuda) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2)})
    ->UseManualTime();

BENCHMARK_MAIN();
} // namespace dmt