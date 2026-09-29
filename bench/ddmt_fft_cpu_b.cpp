#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/ddmt_fft.hpp"

namespace dmt {
using algorithms::DDMT;
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

// DDMT-FFT (exact fractional delays) against DDMT on the same grid:
// 1024 channels over 1000-1500 MHz, DM 0-50 in steps of 0.5.
class DDMTFFTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        nsamps   = static_cast<SizeType>(state.range(0));
        nthreads = static_cast<int>(state.range(1));
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0F, 1.0F);
        waterfall.resize(kNchans * nsamps);
        for (auto& v : waterfall) {
            v = dist(gen);
        }
    }

    static constexpr SizeType kNchans = 1024;
    static constexpr float kFMin      = 1000.0F;
    static constexpr float kFMax      = 1500.0F;
    static constexpr float kTsamp     = 6.4e-5F;
    SizeType nsamps{};
    int nthreads{};
    std::vector<float> waterfall;

    template <typename Engine> void run(benchmark::State& state, Engine& e) {
        const auto ndm = e.get_plan().get_dm_arr().size();
        std::vector<float> out(ndm * nsamps);
        e.execute(waterfall,
                  std::span<float>(out).first(ndm * e.get_output_nsamps(nsamps)));
        for (auto _ : state) {
            e.execute(waterfall, out);
            benchmark::DoNotOptimize(out.data());
        }
        state.counters["cmacs_per_s"] = benchmark::Counter(
            static_cast<double>(kNchans * ndm * nsamps),
            benchmark::Counter::kIsIterationInvariantRate);
    }
};

BENCHMARK_DEFINE_F(DDMTFFTCPUFixture, BM_ddmt_fft_nufft)
(benchmark::State& state) {
    DDMTFFT e(kFMin, kFMax, kNchans, kTsamp, 50.0F, 0.5F, 0.0F,
              Exec::cpu(nthreads), 32, {}, 1,
              DDMTFFTOptions{.method = DDMTFFTMethod::kNUFFT});
    run(state, e);
}

BENCHMARK_DEFINE_F(DDMTFFTCPUFixture, BM_ddmt_fft_brute)
(benchmark::State& state) {
    DDMTFFT e(kFMin, kFMax, kNchans, kTsamp, 50.0F, 0.5F, 0.0F,
              Exec::cpu(nthreads), 32, {}, 1,
              DDMTFFTOptions{.method = DDMTFFTMethod::kBrute});
    run(state, e);
}

BENCHMARK_DEFINE_F(DDMTFFTCPUFixture, BM_ddmt_same_grid)
(benchmark::State& state) {
    DDMT e(kFMin, kFMax, kNchans, kTsamp, 50.0F, 0.5F, 0.0F,
           Exec::cpu(nthreads));
    run(state, e);
}

BENCHMARK_REGISTER_F(DDMTFFTCPUFixture, BM_ddmt_fft_nufft) // NOLINT
    ->ArgsProduct({{8192, 32768}, {1, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(DDMTFFTCPUFixture, BM_ddmt_fft_brute) // NOLINT
    ->ArgsProduct({{8192, 32768}, {1, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_REGISTER_F(DDMTFFTCPUFixture, BM_ddmt_same_grid) // NOLINT
    ->ArgsProduct({{8192, 32768}, {1, 8}})
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

} // namespace dmt
