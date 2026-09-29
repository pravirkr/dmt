#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include "dmt/gpu_compat.cuh"

#include <benchmark/benchmark.h>

#include "bench_gpu_utils.cuh"

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"

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
using algorithms::DDMT;
using plans::DDMTPlan;

// ============================================================================
// DDMT Float Execution Benchmark
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
        nbeams = state.range(1) > 0 ? static_cast<SizeType>(state.range(1)) : 1;

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
    DDMT ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
              bench_gpu_exec(),
              /*nbits=*/32, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    thrust::device_vector<float> dmt_d(nbeams * dm_count * nsamps_reduced,
                                       0.0F);

    for (auto _ : state) {
        GPUEventTimer timer{state};
        ddmt.reset_history();
        ddmt.execute(DeviceSpan<const float>(
                         thrust::raw_pointer_cast(waterfall_d.data()),
                         waterfall_d.size()),
                     DeviceSpan<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                       dmt_d.size()));
    }
}

// ============================================================================
// DDMT Packed Execution Benchmark
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
        nbeams = state.range(2) > 0 ? static_cast<SizeType>(state.range(2)) : 1;

        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
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
    DDMT ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
              bench_gpu_exec(), nbits, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    thrust::device_vector<int32_t> dmt_d(nbeams * dm_count * nsamps_reduced, 0);

    for (auto _ : state) {
        GPUEventTimer timer{state};
        ddmt.reset_history();
        ddmt.execute(DeviceSpan<const uint8_t>(
                         thrust::raw_pointer_cast(waterfall_packed_d.data()),
                         waterfall_packed_d.size()),
                     nsamps,
                     DeviceSpan<int32_t>(thrust::raw_pointer_cast(dmt_d.data()),
                                         dmt_d.size()));
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

// ============================================================================
// DDMT host-memory streaming benchmark (suite reference point: 4096 chans,
// 704-1216 MHz, ~2049 DM trials up to a 2048-sample delay, warm stream).
// Wall time of one host-span execute(), PCIe copies included.
// ============================================================================

void BM_ddmt_cuda_float_host_stream(benchmark::State& state) {
    constexpr float kFMin = 704.0F, kFMax = 1216.0F, kTsamp = 8.192e-5F;
    constexpr SizeType kNchans = 4096, kNdm = 2049;
    const auto df   = (kFMax - kFMin) / static_cast<float>(kNchans);
    const auto a    = 1.0F / kFMin;
    const auto b    = 1.0F / (kFMin + (static_cast<float>(kNchans - 1) * df));
    const auto dmax = 2048.0F / (4148.808F / kTsamp * ((a * a) - (b * b)));
    std::vector<float> dms(kNdm);
    for (SizeType i = 0; i < kNdm; ++i) {
        dms[i] = dmax * static_cast<float>(i) / static_cast<float>(kNdm - 1);
    }
    DDMT ddmt(kFMin, kFMax, kNchans, kTsamp, dms, bench_gpu_exec());
    ddmt.set_gulp_size(static_cast<SizeType>(state.range(1)));
    const auto n = static_cast<SizeType>(state.range(0));
    std::vector<float> wf(kNchans * n, 1.0F);
    std::vector<float> warm(kNdm * ddmt.get_output_nsamps(n));
    ddmt.execute(std::span<const float>(wf), std::span<float>(warm));
    std::vector<float> out(kNdm * ddmt.get_output_nsamps(n));
    for (auto _ : state) {
        ddmt.execute(std::span<const float>(wf), std::span<float>(out));
        benchmark::DoNotOptimize(out.data());
    }
}

BENCHMARK(BM_ddmt_cuda_float_host_stream)
    ->ArgsProduct({{16384, 65536}, {0, 4096, 8192, 16384}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace dmt
