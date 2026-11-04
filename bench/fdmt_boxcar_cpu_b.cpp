#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/boxcar.hpp"
#include "suite/suite_common.hpp"

namespace dmt {
namespace {

// Both paths use the exact same CPU scorer, width trials and peak mode.
// The separated path runs the existing whole-block stepper, then searches
// each materialized selected level. Fusion is disabled on both paths here
// so the comparison isolates root fusion rather than lower-tree fusion.
void fdmt_boxcar_cpu(benchmark::State& state, bool fused) {
    using algorithms::FDMT;
    using algorithms::FDMTBoxcar;
    const auto depth = static_cast<SizeType>(state.range(1));
    plans::FDMTPlan plan(700, 1200, 128, 4096, 0.001F, 256, 0, 1, "valid",
                         depth);
    BoxcarConfig config(static_cast<SizeType>(state.range(0)), 2.0F, 8.0F, 1.0F,
                        PeakFindingMode::kContiguousPeakPerWidth);
    config.max_cands_per_dm = 0;
    const int threads       = static_cast<int>(state.range(2));
    const auto exec         = Exec::cpu(threads);
    FDMTBoxcar search(plan, config, true, exec, 1, 0, plan.get_nsamps());
    FDMT fdmt(plan, true, exec, 1, 0, false, plan.get_nsamps());
    const auto planes =
        algorithms::detail::make_boxcar_plans(plan, config, true);
    algorithms::detail::FDMTEngineConfig cfg;
    cfg.exec            = exec;
    cfg.fuse_levels     = 0;
    cfg.chunk_nsamps    = plan.get_nsamps();
    cfg.max_fuse_levels = 0;
    auto scorer         = algorithms::detail::make_fdmt_boxcar_cpu(
        plan, config, planes, cfg,
        algorithms::detail::make_boxcar_driver(
            algorithms::detail::make_fdmt_cpu(plan, cfg)));
    std::vector<float> waterfall(plan.get_nchans() * plan.get_nsamps());
    std::mt19937 random(42);
    std::normal_distribution<float> noise;
    for (auto& x : waterfall) {
        x = noise(random);
    }
    std::vector<float> output(plan.get_dmt_size());
    uint64_t position = 0;
    for (auto _ : state) {
        if (fused) {
            auto result = search.execute(waterfall);
            benchmark::DoNotOptimize(result.total_detected);
            if (result.total_detected != 0) {
                state.SkipWithError("noise-only benchmark crossed threshold");
                break;
            }
        } else {
            fdmt.reset(waterfall, output);
            for (SizeType level = plan.get_niters() - depth;
                 level <= plan.get_niters(); ++level) {
                fdmt.advance_until_remaining(plan.get_niters() - level);
                const auto data = fdmt.view_level_data();
                algorithms::detail::FDMTSearchInput input{
                    .state             = data.data(),
                    .row_stride        = plan.get_nsamps(),
                    .beam_stride       = data.size(),
                    .nsamps            = plan.get_nsamps(),
                    .sample_begin      = position,
                    .materialized_root = true,
                };
                scorer->consume(level, input);
            }
            fdmt.finalize();
            auto result = scorer->copy_result({});
            benchmark::DoNotOptimize(result.total_detected);
            if (result.total_detected != 0) {
                state.SkipWithError("noise-only benchmark crossed threshold");
                break;
            }
        }
        position += plan.get_nsamps();
    }
    state.counters["nwidths"] =
        static_cast<double>(planes.front().widths.size());
    state.counters["subband_depth"] = static_cast<double>(depth);
    const auto memory               = search.get_memory_usage();
    state.counters["working_bytes"] = static_cast<double>(
        memory.total() + (fused ? 0 : output.size() * sizeof(float)));
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(waterfall.size()));
}
void BM_fdmt_boxcar_fused(benchmark::State& state) {
    fdmt_boxcar_cpu(state, true);
}
void BM_fdmt_boxcar_separate(benchmark::State& state) {
    fdmt_boxcar_cpu(state, false);
}
BENCHMARK(BM_fdmt_boxcar_fused)
    ->ArgsProduct({{32, 4096}, {0, 2}, {1, 4}})
    ->UseRealTime();
BENCHMARK(BM_fdmt_boxcar_separate)
    ->ArgsProduct({{32, 4096}, {0, 2}, {1, 4}})
    ->UseRealTime();

} // namespace
} // namespace dmt

// Production-sized streaming target, selected through the same configuration
// as dmt_bench_suite. This includes both last-level subbands and automatic
// chunking; float uses lower-tree fusion, packed search currently does not.
namespace dmt::bench_suite {
namespace {

void bench_fdmt_boxcar_stream(benchmark::State& state,
                              int threads,
                              float wtsp,
                              bool packed = false) {
    const auto& cfg = suite_config();
    const Point point{Sweep::kNdms, cfg.nsamps_ref, cfg.dm_max_ref,
                      packed ? SizeType{8} : SizeType{32}};
    auto plan = make_plan(point, cfg.subband_depth);
    // The GPU benchmark's width sets (boxcars up to 100 ms), threshold 6,
    // per-width peaks. Input preparation and construction are not timed.
    const BoxcarConfig boxcar(
        max_boxcar_100ms(), wtsp, 6.0F, packed ? 12.0F : 1.0F,
        PeakFindingMode::kContiguousPeakPerWidth, 131072, 4096, true,
        packed ? 8 : 32, packed ? 127.0F : 0.0F);
    const auto input_size = cfg.nchans * cfg.nsamps_ref;
    double rows           = 0;
    for (SizeType level = plan.get_niters() - cfg.subband_depth;
         level <= plan.get_niters(); ++level) {
        rows += static_cast<double>(
            plan.get_container().state_shape[level].ncoords);
    }
    const auto widths =
        plans::generate_box_width_trials(boxcar.max_boxcar, boxcar.wtsp);
    const double extra =
        rows * (boxcar.max_boxcar * sizeof(float) +
                widths.size() * sizeof(algorithms::detail::BoxcarPeak));
    if ((packed ? 2 : 1) * (estimate_bytes(Algo::kFDMT, plan, point) + extra) >
        memory_budget_bytes()) {
        state.SkipWithError(
            "FDMTBoxcar target exceeds DMT_BENCH_MAX_GB budget");
        return;
    }
    algorithms::FDMTBoxcar search(std::move(plan), boxcar, true,
                                  Exec::cpu(threads));
    std::vector<float> input(packed ? 0 : input_size);
    std::vector<uint8_t> packed_input(packed ? input_size : 0);
    std::mt19937 random(42);
    std::normal_distribution<float> noise;
    for (auto& x : input) {
        x = noise(random);
    }
    for (auto& x : packed_input) {
        x = static_cast<uint8_t>(std::clamp(
            std::round(127.0F + 12.0F * noise(random)), 0.0F, 255.0F));
    }
    uint64_t warmup = 0;
    for (const auto& plane : search.get_search_plans()) {
        for (const auto support : plane.warmup_samples) {
            warmup = std::max(warmup, support + plane.widths.back() - 1);
        }
    }
    // UWL requires several blocks before all selected DM/width windows
    // are valid. Warm histories and workers outside the timed region.
    for (uint64_t filled = 0; filled < std::max<uint64_t>(warmup, 1);
         filled += cfg.nsamps_ref) {
        if (packed) {
            (void)search.execute(packed_input);
        } else {
            (void)search.execute(input);
        }
    }
    CandidateResultView result;
    for (auto _ : state) {
        result = packed ? search.execute(packed_input) : search.execute(input);
        benchmark::DoNotOptimize(result.candidates.data());
        benchmark::DoNotOptimize(result.total_detected);
        if (result.overflowed()) {
            state.SkipWithError("FDMTBoxcar candidate capacity exhausted");
            break;
        }
    }
    set_counters(state, search.get_plan(), point, threads);
    const auto memory = search.get_memory_usage();
    const double block_seconds =
        static_cast<double>(cfg.nsamps_ref) * cfg.tsamp;
    state.counters["realtime_factor"] = benchmark::Counter(
        block_seconds, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["working_bytes"] = static_cast<double>(
        memory.total() +
        input_size * (packed ? sizeof(uint8_t) : sizeof(float)));
    state.counters["input_nbits"] = packed ? 8 : 32;
    state.counters["root_output_bytes_avoided"] =
        static_cast<double>(search.get_plan().get_dmt_size() * sizeof(float));
    state.counters["max_boxcar"] = static_cast<double>(boxcar.max_boxcar);
    state.counters["nwidths"] =
        static_cast<double>(search.get_search_plans().front().widths.size());
    state.counters["subband_depth"] = static_cast<double>(cfg.subband_depth);
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    state.counters["fuse_levels"] =
        static_cast<double>(search.get_fuse_levels());
    state.counters["candidates_per_block"] =
        static_cast<double>(result.emitted_count);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(input_size));
}

[[maybe_unused]] const bool kStreamRegistered = [] {
    for (const float wtsp : wtsp_sweep()) {
        for (const auto threads : cpu_threads()) {
            for (const bool packed : {false, true}) {
                benchmark::RegisterBenchmark(
                    std::format("BM_fdmt_boxcar_stream/{}/{}wtsp:{:g}/cpu{}",
                                suite_config().name, packed ? "uint8/" : "",
                                wtsp, threads),
                    [threads, wtsp, packed](benchmark::State& state) {
                        bench_fdmt_boxcar_stream(state, threads, wtsp, packed);
                    })
                    ->UseRealTime()
                    ->Unit(benchmark::kMillisecond);
            }
        }
    }
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
