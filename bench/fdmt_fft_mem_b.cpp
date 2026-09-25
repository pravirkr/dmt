#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_mem_utils.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

namespace {

template <typename T>
std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

} // namespace

namespace dmt {
using algorithms::FDMTFFTCPU;

class FDMTFFTMemFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min     = 704.0F;
        f_max     = 1216.0F;
        nchans    = 256;
        tsamp     = 0.00008192F;
        dt_max    = 256;
        nsamps    = static_cast<size_t>(state.range(0));
        gen       = std::mt19937(42);
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{};
    size_t nchans{}, dt_max{}, nsamps{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(FDMTFFTMemFixture, BM_fdmt_fft_memory_execute)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTFFTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1,
                        false, "valid", false, /*nthreads=*/1);
        state.PauseTiming();
        std::vector<float> dmt(fdmt.get_plan().get_dmt_size());
        state.ResumeTiming();
        fdmt.execute(waterfall, dmt);
    }
    bench::report_memory_counters(state);
}

BENCHMARK_REGISTER_F(FDMTFFTMemFixture, BM_fdmt_fft_memory_execute) // NOLINT
    ->Arg(1 << 10)
    ->Arg(1 << 12)
    ->Arg(1 << 13)
    ->Iterations(4)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

} // namespace dmt
