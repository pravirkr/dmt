#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"

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

namespace {

class CudaEventTimer {
public:
    explicit CudaEventTimer(benchmark::State& state,
                            cudaStream_t stream = 0)
        : m_stream(stream),
          m_state(&state) {
        BENCH_CUDA_TRY(cudaEventCreate(&m_start));
        BENCH_CUDA_TRY(cudaEventCreate(&m_stop));
        BENCH_CUDA_TRY(cudaEventRecord(m_start, m_stream));
    }

    CudaEventTimer() = delete;

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
using algorithms::DDMTCUDA;
using plans::DDMTPlan;

// ============================================================================
// DDMTCUDA Float Execution Benchmark
// ============================================================================

class DDMTCUDAFloatFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min   = 1000.0F;
        f_max   = 1500.0F;
        nchans  = 1024;
        tsamp   = 0.00008192F;
        dm_max  = 50.0F;
        dm_step = 1.0F;
        nsamps  = static_cast<SizeType>(state.range(0));
        nbeams  = state.range(1) > 0 ? static_cast<SizeType>(state.range(1)) : 1;

        waterfall_d = generate_vector_device<float>(nbeams * nchans * nsamps);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {
        waterfall_d.clear();
        waterfall_d.shrink_to_fit();
    }

    void SetUp(benchmark::State& state) override {
        SetUp(static_cast<const benchmark::State&>(state));
    }

    void TearDown(benchmark::State& state) override {
        TearDown(static_cast<const benchmark::State&>(state));
    }

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{}, nbeams{1};
    thrust::device_vector<float> waterfall_d;
};

BENCHMARK_DEFINE_F(DDMTCUDAFloatFixture, BM_ddmt_cuda_float_execute)
(benchmark::State& state) {
    DDMTCUDA ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
                  /*device_id=*/0, /*nbits=*/32, {}, nbeams);
    const auto max_delay      = *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    thrust::device_vector<float> dmt_d(nbeams * dm_count * nsamps_reduced, 0.0F);

    for (auto _ : state) {
        CudaEventTimer timer{state};
        ddmt.reset_history();
        ddmt.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()), waterfall_d.size()),
            cuda::std::span<float>(
                thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size()));
    }
}

// ============================================================================
// DDMTCUDA Packed Execution Benchmark
// ============================================================================

class DDMTCUDAPackedFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min   = 1000.0F;
        f_max   = 1500.0F;
        nchans  = 1024;
        tsamp   = 0.00008192F;
        dm_max  = 50.0F;
        dm_step = 1.0F;
        nsamps  = static_cast<SizeType>(state.range(0));
        nbits   = static_cast<SizeType>(state.range(1));
        nbeams  = state.range(2) > 0 ? static_cast<SizeType>(state.range(2)) : 1;

        const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
        waterfall_packed_d.resize(nbeams * nchans * row_bytes, 0);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {
        waterfall_packed_d.clear();
        waterfall_packed_d.shrink_to_fit();
    }

    void SetUp(benchmark::State& state) override {
        SetUp(static_cast<const benchmark::State&>(state));
    }

    void TearDown(benchmark::State& state) override {
        TearDown(static_cast<const benchmark::State&>(state));
    }

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{}, nbits{8}, nbeams{1};
    thrust::device_vector<uint8_t> waterfall_packed_d;
};

BENCHMARK_DEFINE_F(DDMTCUDAPackedFixture, BM_ddmt_cuda_packed_execute)
(benchmark::State& state) {
    DDMTCUDA ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
                  /*device_id=*/0, nbits, {}, nbeams);
    const auto max_delay      = *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    thrust::device_vector<int32_t> dmt_d(nbeams * dm_count * nsamps_reduced, 0);

    for (auto _ : state) {
        CudaEventTimer timer{state};
        ddmt.reset_history();
        ddmt.execute(
            cuda::std::span<const uint8_t>(
                thrust::raw_pointer_cast(waterfall_packed_d.data()),
                waterfall_packed_d.size()),
            nsamps,
            cuda::std::span<int32_t>(
                thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size()));
    }
}

// Float benchmarks
BENCHMARK_REGISTER_F(DDMTCUDAFloatFixture, BM_ddmt_cuda_float_execute)
    ->ArgsProduct({{4096, 16384}, {1}})
    ->UseManualTime();

BENCHMARK_REGISTER_F(DDMTCUDAFloatFixture, BM_ddmt_cuda_float_execute)
    ->ArgsProduct({{4096}, {1, 4}})
    ->UseManualTime();

// Packed benchmarks across precisions
BENCHMARK_REGISTER_F(DDMTCUDAPackedFixture, BM_ddmt_cuda_packed_execute)
    ->ArgsProduct({{4096}, {1, 2, 4, 8, 16}, {1}})
    ->UseManualTime();

BENCHMARK_REGISTER_F(DDMTCUDAPackedFixture, BM_ddmt_cuda_packed_execute)
    ->ArgsProduct({{4096}, {8}, {1, 4}})
    ->UseManualTime();

} // namespace dmt
