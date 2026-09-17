#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

namespace dmt {
using algorithms::FDMTCPU;
using algorithms::FDMTFFTCPU;

template <typename T>
static std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

// Smaller than the ASKAP-like FDMTCPU bench: FFT state is complex and
// full/valid pad N_fft = nsamps + L + max_shift.
class FDMTFFTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min     = 704.0F;
        f_max     = 1216.0F;
        nchans    = 256;
        tsamp     = 0.00008192F;
        dt_max    = 256;
        nsamps    = (state.range(0) == 0) ? 1 << 12 : state.range(0);
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

BENCHMARK_DEFINE_F(FDMTFFTCPUFixture, BM_fdmt_fft_execute_roll)
(benchmark::State& state) {
    FDMTFFTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                    "roll", false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
}

BENCHMARK_DEFINE_F(FDMTFFTCPUFixture, BM_fdmt_fft_execute_valid)
(benchmark::State& state) {
    FDMTFFTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                    "valid", false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
}

BENCHMARK_DEFINE_F(FDMTFFTCPUFixture, BM_fdmt_direct_execute_valid)
(benchmark::State& state) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                 "valid", false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
}

constexpr size_t kMinNsamps = 1 << 10;
constexpr size_t kMaxNsamps = 1 << 13;

BENCHMARK_REGISTER_F(FDMTFFTCPUFixture, BM_fdmt_fft_execute_roll) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTFFTCPUFixture, BM_fdmt_fft_execute_valid) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTFFTCPUFixture, BM_fdmt_direct_execute_valid) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

} // namespace dmt
