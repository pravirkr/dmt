#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_mem_utils.hpp"
#include "dmt/algorithms/ddmt_fft.hpp"

namespace dmt {
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

// Peak memory of one DDMT-FFT construction + execute (range(1): 0 = NUFFT,
// 1 = brute force).
void BM_ddmt_fft_memory_execute(benchmark::State& state) {
    const auto nsamps = static_cast<SizeType>(state.range(0));
    const auto method =
        state.range(1) == 0 ? DDMTFFTMethod::kNUFFT : DDMTFFTMethod::kBrute;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> wf(256 * nsamps);
    for (auto& v : wf) {
        v = dist(gen);
    }
    for (auto _ : state) {
        DDMTFFT e(704.0F, 1216.0F, 256, 8.192e-5F, 100.0F, 1.0F, 0.0F,
                  Exec::cpu(1), 32, {}, 1, DDMTFFTOptions{.method = method});
        state.PauseTiming();
        std::vector<float> out(e.get_plan().get_dm_arr().size() *
                               e.get_output_nsamps(nsamps));
        state.ResumeTiming();
        e.execute(wf, out);
    }
    bench::report_memory_counters(state);
}

BENCHMARK(BM_ddmt_fft_memory_execute) // NOLINT
    ->ArgsProduct({{1 << 12, 1 << 14}, {0, 1}})
    ->Iterations(2)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

} // namespace dmt
