#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt_fft.hpp"

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
    explicit CudaEventTimer(benchmark::State& state, cudaStream_t stream = 0)
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
using algorithms::FDMTFFTCUDA;

// Smaller than the ASKAP-like FDMTCUDA bench: FFT state is complex and
// full/valid pad N_fft = nsamps + L + max_shift, matching the CPU
// FDMT-FFT fixture's configuration for apples-to-apples comparison.
class FDMTFFTCUDAFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min       = 704.0F;
        f_max       = 1216.0F;
        nchans      = 256;
        tsamp       = 0.00008192F;
        dt_max      = 256;
        nsamps      = state.range(0);
        waterfall_d = generate_vector_device<float>(nchans * nsamps);
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

    float f_min{}, f_max{}, tsamp{};
    size_t nchans{}, dt_max{}, nsamps{};
    thrust::device_vector<float> waterfall_d;
};

BENCHMARK_DEFINE_F(FDMTFFTCUDAFixture, BM_fdmt_fft_execute_roll_cuda)
(benchmark::State& state) {
    FDMTFFTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                     "roll");
    thrust::device_vector<float> dmt_d(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        CudaEventTimer raii{state};
        fdmt.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()),
                waterfall_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size()));
    }
}

BENCHMARK_DEFINE_F(FDMTFFTCUDAFixture, BM_fdmt_fft_execute_valid_cuda)
(benchmark::State& state) {
    FDMTFFTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                     "valid");
    thrust::device_vector<float> dmt_d(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        CudaEventTimer raii{state};
        fdmt.execute(
            cuda::std::span<const float>(
                thrust::raw_pointer_cast(waterfall_d.data()),
                waterfall_d.size()),
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size()));
    }
}

constexpr size_t kMinNsamps = 1 << 10;
constexpr size_t kMaxNsamps = 1 << 13;

BENCHMARK_REGISTER_F(FDMTFFTCUDAFixture, BM_fdmt_fft_execute_roll_cuda) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2)})
    ->UseManualTime();

BENCHMARK_REGISTER_F(FDMTFFTCUDAFixture, BM_fdmt_fft_execute_valid_cuda) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2)})
    ->UseManualTime();

} // namespace dmt
