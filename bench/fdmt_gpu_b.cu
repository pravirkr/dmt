#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <dmt/fdmt_gpu.hpp>

#define BENCH_CUDA_TRY(call)                                                         \
  do {                                                                               \
    auto const status = (call);                                                      \
    if (cudaSuccess != status) { throw std::runtime_error("CUDA error detected."); } \
  } while (0);


class cuda_event_timer {
 public:
  /**
   * @brief Constructs a `cuda_event_timer` beginning a manual timing range.
   *
   * Optionally flushes L2 cache.
   *
   * @param[in,out] state  This is the benchmark::State whose timer we are going
   * to update.
   * @param[in] flush_l2_cache_ whether or not to flush the L2 cache before
   *                            every iteration.
   * @param[in] stream_ The CUDA stream we are measuring time on.
   */
  cuda_event_timer(benchmark::State &state, bool flush_l2_cache = false, cudaStream_t stream = 0)
    : p_state(&state), stream_(stream)
  {
    // flush all of L2$
    if (flush_l2_cache) {
      int current_device = 0;
      BENCH_CUDA_TRY(cudaGetDevice(&current_device));

      int l2_cache_bytes = 0;
      BENCH_CUDA_TRY(
        cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, current_device));

      if (l2_cache_bytes > 0) {
        const int memset_value = 0;
        int *l2_cache_buffer   = nullptr;
        BENCH_CUDA_TRY(cudaMalloc(&l2_cache_buffer, l2_cache_bytes));
        BENCH_CUDA_TRY(cudaMemsetAsync(l2_cache_buffer, memset_value, l2_cache_bytes, stream_));
        BENCH_CUDA_TRY(cudaFree(l2_cache_buffer));
      }
    }

    BENCH_CUDA_TRY(cudaEventCreate(&start_));
    BENCH_CUDA_TRY(cudaEventCreate(&stop_));
    BENCH_CUDA_TRY(cudaEventRecord(start_, stream_));
  }

  cuda_event_timer() = delete;

  /**
   * @brief Destroy the `cuda_event_timer` and ending the manual time range.
   *
   */
  ~cuda_event_timer()
  {
    BENCH_CUDA_TRY(cudaEventRecord(stop_, stream_));
    BENCH_CUDA_TRY(cudaEventSynchronize(stop_));
    float milliseconds = 0.0f;
    BENCH_CUDA_TRY(cudaEventElapsedTime(&milliseconds, start_, stop_));
    p_state->SetIterationTime(milliseconds / (1000.0f));
    BENCH_CUDA_TRY(cudaEventDestroy(start_));
    BENCH_CUDA_TRY(cudaEventDestroy(stop_));
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;
  cudaStream_t stream_;
  benchmark::State *p_state;
};

class FDMTGPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min  = 704.0F;
        f_max  = 1216.0F;
        nchans = 4096;
        tsamp  = 0.00008192F;
        dt_max = 2048;
        nsamps = state.range(0);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    template <typename T>
    thrust::device_vector<T> generate_vector_gpu(size_t size) {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> dist(0.0, 1.0);

        thrust::device_vector<T> vec(size);
        thrust::transform(thrust::counting_iterator<size_t>(0),
                          thrust::counting_iterator<size_t>(size), vec.begin(),
                          [=] __device__(size_t idx) mutable { return dist(rng); });

        return vec;
    }

    float f_min{};
    float f_max{};
    size_t nchans{};
    float tsamp{};
    size_t dt_max{};
    size_t nsamps{};
};

BENCHMARK_DEFINE_F(FDMTGPUFixture, BM_fdmt_plan_gpu)(benchmark::State& state) {
    for (auto _ : state) {
        cuda_event_timer raii{state};
        FDMTGPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTGPUFixture, BM_fdmt_initialise_gpu)
(benchmark::State& state) {
    FDMTGPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    auto waterfall_d       = generate_vector_gpu<float>(nchans * nsamps);
    float* waterfall_d_ptr = thrust::raw_pointer_cast(waterfall_d.data());
    const auto& plan       = fdmt.get_plan();
    const auto state_size  = plan.state_shape[0][3] * plan.state_shape[0][4];
    thrust::device_vector<float> state_init_d(state_size, 0.0F);
    float* state_init_d_ptr = thrust::raw_pointer_cast(state_init_d.data());
    for (auto _ : state) {
        cuda_event_timer raii{state};
        fdmt.initialise(waterfall_d_ptr, state_init_d_ptr);
    }
}

BENCHMARK_DEFINE_F(FDMTGPUFixture, BM_fdmt_execute_gpu)
(benchmark::State& state) {
    FDMTGPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    auto waterfall_d       = generate_vector_gpu<float>(nchans * nsamps);
    float* waterfall_d_ptr = thrust::raw_pointer_cast(waterfall_d.data());
    thrust::device_vector<float> dmt_d(fdmt.get_dt_grid_final().size() * nsamps,
                                       0.0F);
    float* dmt_d_ptr = thrust::raw_pointer_cast(dmt_d.data());
    for (auto _ : state) {
        cuda_event_timer raii{state};
        fdmt.execute(waterfall_d_ptr, waterfall_d.size(), dmt_d_ptr,
                     dmt_d.size());
    }
}

BENCHMARK_DEFINE_F(FDMTGPUFixture, BM_fdmt_overall_gpu)
(benchmark::State& state) {
    auto waterfall_d       = generate_vector_gpu<float>(nchans * nsamps);
    float* waterfall_d_ptr = thrust::raw_pointer_cast(waterfall_d.data());

    for (auto _ : state) {
        cuda_event_timer raii{state};
        FDMTGPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        state.PauseTiming();
        thrust::device_vector<float> dmt_d(
            fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
        float* dmt_d_ptr = thrust::raw_pointer_cast(dmt_d.data());
        state.ResumeTiming();
        fdmt.execute(waterfall_d_ptr, waterfall_d.size(), dmt_d_ptr,
                     dmt_d.size());
    }
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 15;

BENCHMARK_REGISTER_F(FDMTGPUFixture, BM_fdmt_plan_gpu)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->UseManualTime();
BENCHMARK_REGISTER_F(FDMTGPUFixture, BM_fdmt_initialise_gpu)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->UseManualTime();
BENCHMARK_REGISTER_F(FDMTGPUFixture, BM_fdmt_execute_gpu)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->UseManualTime();
BENCHMARK_REGISTER_F(FDMTGPUFixture, BM_fdmt_overall_gpu)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->UseManualTime();

//BENCHMARK_MAIN();
