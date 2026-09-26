// CPU half of dmt_bench_suite; see suite_common.hpp for the configuration.
//
// Every benchmark times steady-state streaming: the engine is constructed and
// warmed up (one execute(), which also fills DDMT / valid-mode history)
// outside the timed loop, and each iteration is one execute() of one block
// into a preallocated output.

#include <format>
#include <new>
#include <span>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

#include "suite_common.hpp"

namespace dmt::bench_suite {
namespace {

using algorithms::DDMTCPU;
using algorithms::FDMTCPU;
using algorithms::FDMTFFTCPU;

// Skips (without running) a point whose estimated memory exceeds the budget.
bool skip_if_over_budget(benchmark::State& state,
                         Algo algo,
                         const plans::FDMTPlan& plan,
                         const Point& p) {
    const double need   = estimate_bytes(algo, plan, p);
    const double budget = memory_budget_bytes();
    if (need > budget) {
        state.SkipWithError(
            std::format("skipped: needs ~{:.1f} GB, over the {:.1f} GB budget "
                        "(DMT_BENCH_MAX_GB)",
                        need / 1.0737e9, budget / 1.0737e9));
        return true;
    }
    return false;
}

void bench_fdmt(benchmark::State& state, Point p, int nthreads) {
    const auto plan = make_plan(p);
    if (skip_if_over_budget(state, Algo::kFDMT, plan, p)) {
        return;
    }
    FDMTCPU fdmt(kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1, true,
                 "valid", nthreads);
    std::vector<float> dmt(plan.get_buffer_size());
    if (p.nbits == 32) {
        const auto wf = make_float_input(kNchans * p.nsamps);
        fdmt.execute(wf, dmt); // warm-up
        for (auto _ : state) {
            fdmt.execute(wf, dmt);
            benchmark::DoNotOptimize(dmt.data());
        }
    } else {
        const auto wf = make_packed_input(packed_bytes(p.nsamps, p.nbits));
        fdmt.execute(std::span<const uint8_t>(wf), p.nbits, dmt);
        for (auto _ : state) {
            fdmt.execute(std::span<const uint8_t>(wf), p.nbits, dmt);
            benchmark::DoNotOptimize(dmt.data());
        }
    }
    set_counters(state, plan, p, nthreads);
}

void bench_fdmt_fft(benchmark::State& state, Point p, int nthreads) {
    const auto plan = make_plan(p);
    if (skip_if_over_budget(state, Algo::kFDMTFFT, plan, p)) {
        return;
    }
    try {
        FDMTFFTCPU fdmt(kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1,
                        true, "valid", nthreads);
        const auto wf = make_float_input(kNchans * p.nsamps);
        std::vector<float> dmt(plan.get_dmt_size());
        fdmt.execute(wf, dmt);
        for (auto _ : state) {
            fdmt.execute(wf, dmt);
            benchmark::DoNotOptimize(dmt.data());
        }
    } catch (const std::bad_alloc&) {
        state.SkipWithError("skipped: out of memory");
        return;
    }
    set_counters(state, plan, p, nthreads);
}

void bench_ddmt(benchmark::State& state, Point p, int nthreads) {
    const auto plan = make_plan(p);
    if (skip_if_over_budget(state, Algo::kDDMT, plan, p)) {
        return;
    }
    const auto dms = plan.get_dm_grid_final();
    DDMTCPU ddmt(kFMin, kFMax, kNchans, kTsamp, std::span<const float>(dms),
                 nthreads, p.nbits);
    const auto ndms = dms.size();
    // The first (cold) call returns nsamps - max_delay samples; warm calls
    // return nsamps.
    const auto n_cold = ddmt.get_output_nsamps(p.nsamps);
    if (p.nbits == 32) {
        const auto wf = make_float_input(kNchans * p.nsamps);
        std::vector<float> dmt(ndms * p.nsamps);
        ddmt.execute(wf, std::span<float>(dmt).first(ndms * n_cold));
        for (auto _ : state) {
            ddmt.execute(wf, dmt);
            benchmark::DoNotOptimize(dmt.data());
        }
    } else {
        const auto wf = make_packed_input(packed_bytes(p.nsamps, p.nbits));
        std::vector<int32_t> dmt(ndms * p.nsamps);
        ddmt.execute(wf, p.nsamps,
                     std::span<int32_t>(dmt).first(ndms * n_cold));
        for (auto _ : state) {
            ddmt.execute(wf, p.nsamps, dmt);
            benchmark::DoNotOptimize(dmt.data());
        }
    }
    set_counters(state, plan, p, nthreads);
}

// DDMT points above this many additions per call run one repetition.
constexpr double kSingleRepetitionAdds = 5.0e10;

void register_cpu() {
    const auto threads   = cpu_threads();
    const int multi      = cpu_threads_max();
    const auto cpu_label = [](int t) { return std::format("cpu{}", t); };
    const auto add = [](const std::string& name, auto fn, Point p, int t) {
        return benchmark::RegisterBenchmark(
                   name, [fn, p, t](benchmark::State& s) { fn(s, p, t); })
            ->UseRealTime()
            ->Unit(benchmark::kMillisecond);
    };
    // Brute-force DDMT calls run for seconds; one repetition of those is
    // already stable and keeps the whole suite to minutes.
    const auto add_ddmt = [&](const Point& p, int t) {
        auto* bm =
            add(bench_name(Algo::kDDMT, cpu_label(t), p), bench_ddmt, p, t);
        const double adds = static_cast<double>(kNchans) *
                            static_cast<double>(p.dt_max + 1) *
                            static_cast<double>(p.nsamps);
        if (adds > kSingleRepetitionAdds) {
            bm->Repetitions(1);
        }
    };

    for (const auto sweep : {Sweep::kNsamps, Sweep::kNdms}) {
        for (const auto& p : sweep_points(sweep)) {
            for (const int t : threads) {
                add(bench_name(Algo::kFDMT, cpu_label(t), p), bench_fdmt, p, t);
            }
            add(bench_name(Algo::kFDMTFFT, cpu_label(multi), p), bench_fdmt_fft,
                p, multi);
            add_ddmt(p, multi);
        }
    }
    for (const auto& p : sweep_points(Sweep::kNbits)) {
        for (const int t : threads) {
            add(bench_name(Algo::kFDMT, cpu_label(t), p), bench_fdmt, p, t);
        }
        add_ddmt(p, multi);
    }
}

[[maybe_unused]] const bool kRegistered = [] {
    register_cpu();
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
