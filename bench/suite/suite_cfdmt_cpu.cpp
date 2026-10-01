// CohFDMT CPU benchmarks of the published suite (see suite_cfdmt_common.hpp).

#include <cstdint>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/cfdmt.hpp"
#include "suite_cfdmt_common.hpp"

namespace dmt::bench_suite::cfdmt {
namespace {

void bench_cpu(benchmark::State& state, const Point& p, int nthreads) {
    const plans::CohFDMTPlan plan(make_config(p));
    if (estimate_bytes(plan) > memory_budget_bytes()) {
        state.SkipWithMessage("over the DMT_BENCH_MAX_GB memory budget");
        return;
    }
    const algorithms::CohFDMT search(make_config(p), Exec::cpu(nthreads));
    const auto in = make_packed_input(search.get_input_size());
    std::vector<float> dmt(search.get_dmt_size());
    search.execute<uint8_t>(std::span<const uint8_t>(in), dmt); // warm-up
    for (auto _ : state) {
        search.execute<uint8_t>(std::span<const uint8_t>(in), dmt);
        benchmark::DoNotOptimize(dmt.data());
    }
    set_counters(state, plan, nthreads);
}

void register_cpu() {
    for (const auto& p : points()) {
        for (const int t : cpu_threads()) {
            benchmark::RegisterBenchmark(
                name(std::format("cpu{}", t), p),
                [p, t](benchmark::State& s) { bench_cpu(s, p, t); })
                ->UseRealTime()
                ->Unit(benchmark::kMillisecond);
        }
    }
}

[[maybe_unused]] const bool kRegistered = [] {
    register_cpu();
    return true;
}();

} // namespace
} // namespace dmt::bench_suite::cfdmt
