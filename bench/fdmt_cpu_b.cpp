#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/common/plans.hpp"
#include "dmt/fdmt.hpp"

// Helper function to generate random data
template <typename T>
static std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

// Fixture for the FDMT CPU benchmarks
class FDMTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min  = 704.0F;
        f_max  = 1216.0F;
        nchans = 4096;
        tsamp  = 0.00008192F;
        dt_max = 2048;
        // If range is 0, fix nsamps for thread benchmarks
        nsamps = (state.range(0) == 0) ? 1 << 16 : state.range(0);
        // If second range is 0, use first range as nthreads
        nthreads  = (state.range(1) == 0) ? static_cast<int>(state.range(0))
                                          : static_cast<int>(state.range(1));
        gen       = std::mt19937(std::random_device()());
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{};
    size_t nchans{}, dt_max{}, nsamps{};
    int nthreads{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer)
(benchmark::State& state) {
    for (auto _ : state) {
        dmt::FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 1, 0,
                          false, false, nthreads);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute)
(benchmark::State& state) {
    dmt::FDMT<dmt::backend::CPU> fdmt(f_min, f_max, nchans, nsamps, tsamp,
                                      dt_max, 1, 0, false, false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(std::span(waterfall), std::span(dmt));
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall)
(benchmark::State& state) {
    FDMTPlan tmp_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(tmp_plan.get_dmt_size(), 0.0F);
    for (auto _ : state) {
        dmt::FDMT<dmt::backend::CPU> fdmt(f_min, f_max, nchans, nsamps, tsamp,
                                          dt_max, 1, 0, false, false, nthreads);
        fdmt.execute(std::span(waterfall), std::span(dmt));
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_threads)
(benchmark::State& state) {
    dmt::FDMT<dmt::backend::CPU> fdmt(f_min, f_max, nchans, nsamps, tsamp,
                                      dt_max, 1, 0, false, false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(std::span(waterfall), std::span(dmt));
    }
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 16;

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_planBuffer) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_threads) // NOLINT
    ->ArgsProduct({{0}, {1, 2, 4, 8, 10, 12, 16}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// BENCHMARK_MAIN();
