// CPU half of the boxcar suite (run_suite.py --suite boxcar): streaming
// FDMTBoxcar at the reference point of the selected configuration
// (DMT_BENCH_CONFIG), float32 and packed 8-bit, at the preset's subband
// depth, boxcars to 100 ms at the preset's two width spacings. parkes_uwl
// adds the real-time sweep (BM_fdmt_boxcar_rt, suite_common.hpp): depth 4
// at 64, 256, 512 and 1024 us on 2.097 s blocks.

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/format.h>

#include "dmt/algorithms/fdmt_boxcar.hpp"

#include "dmt/boxcar.hpp"

#include "suite_common.hpp"
#include "suite_memory.hpp"

namespace dmt::bench_suite {
namespace {

/// A warmed FDMTBoxcar and its noise input, built once per benchmark point
/// (cached_context): construction, input generation and the history
/// warm-up (several blocks on parkes_uwl) are serial or untimed work that
/// would otherwise repeat for every repetition.
struct CpuContext {
    std::unique_ptr<algorithms::FDMTBoxcar> search;
    std::vector<float> input;
    std::vector<uint8_t> packed_input;
    std::string error;
    /// Peak RSS from construction through warm-up and the timed blocks.
    PeakRss mem;
    CandidateResultView run() {
        return packed_input.empty()
                   ? search->execute(std::span<const float>(input))
                   : search->execute(std::span<const uint8_t>(packed_input));
    }
};

std::shared_ptr<CpuContext> cpu_context(const std::string& key,
                                        plans::FDMTPlan plan,
                                        const BoxcarConfig& config,
                                        int threads) {
    return cached_context<CpuContext>(key, [&] {
        auto ctx = std::make_shared<CpuContext>();
        ctx->mem.begin();
        const bool packed    = config.input_nbits != 32;
        const SizeType n     = plan.get_nchans() * plan.get_nsamps();
        const SizeType block = plan.get_nsamps();
        // FDMTBoxcar holds itself to its memory budget (10 GiB by default):
        // a configuration it rejects is reported as skipped.
        try {
            ctx->search = std::make_unique<algorithms::FDMTBoxcar>(
                std::move(plan), config, true, Exec::cpu(threads));
        } catch (const std::invalid_argument& e) {
            ctx->error = e.what();
            return ctx;
        }
        std::mt19937 random(42);
        std::normal_distribution<float> noise;
        if (packed) {
            ctx->packed_input.resize(n);
            for (auto& x : ctx->packed_input) {
                x = static_cast<uint8_t>(std::clamp(
                    std::round(127.0F + 12.0F * noise(random)), 0.0F, 255.0F));
            }
        } else {
            ctx->input.resize(n);
            for (auto& x : ctx->input) {
                x = noise(random);
            }
        }
        uint64_t warmup = 1;
        for (const auto& plane : ctx->search->get_search_plans()) {
            for (const auto support : plane.warmup_samples) {
                warmup = std::max(warmup, support + plane.widths.back() - 1);
            }
        }
        // Every selected DM/width window valid (parkes_uwl: ~8 blocks).
        for (uint64_t filled = 0; filled < warmup; filled += block) {
            (void)ctx->run();
        }
        return ctx;
    });
}

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
    auto ctx =
        cpu_context(fmt::format("stream/{}/{}/{}", wtsp, threads, packed),
                    std::move(plan), boxcar, threads);
    if (!ctx->error.empty()) {
        state.SkipWithError(ctx->error);
        return;
    }
    auto& search = *ctx->search;
    CandidateResultView result;
    for (auto _ : state) {
        result = ctx->run();
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
    ctx->mem.report(state);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(input_size));
}

/// One real-time sweep point at the largest thread count, default memory
/// budget; warm histories outside the timed region.
void bench_rt_cpu(benchmark::State& state,
                  RtPoint p,
                  bool packed,
                  int threads) {
    auto plan = make_rt_plan(p);
    const BoxcarConfig config(
        rt_max_boxcar(p), kRtWtsp, 6.0F, packed ? 12.0F : 1.0F,
        PeakFindingMode::kContiguousPeakPerWidth, 131072, 4096, true,
        packed ? 8 : 32, packed ? 127.0F : 0.0F);
    const SizeType n = kNchans * p.nsamps;
    auto ctx = cpu_context(rt_name("cpu", p, packed), std::move(plan), config,
                           threads);
    if (!ctx->error.empty()) {
        state.SkipWithError(ctx->error);
        return;
    }
    auto& search = *ctx->search;
    for (auto _ : state) {
        auto r = ctx->run();
        benchmark::DoNotOptimize(r.total_detected);
    }
    const auto mem    = search.get_memory_usage();
    const auto& rplan = search.get_plan();
    state.counters["realtime_factor"] =
        benchmark::Counter(static_cast<double>(p.nsamps) * p.tsamp,
                           benchmark::Counter::kIsIterationInvariantRate);
    state.counters["working_bytes"] = static_cast<double>(
        mem.total() + n * (packed ? sizeof(uint8_t) : sizeof(float)));
    state.counters["engine_bytes"]  = static_cast<double>(mem.total());
    state.counters["tsamp"]         = p.tsamp;
    state.counters["nsamps"]        = static_cast<double>(p.nsamps);
    state.counters["block_s"]       = static_cast<double>(p.nsamps) * p.tsamp;
    state.counters["ndms"]          = static_cast<double>(rplan.get_dmt_ndms());
    state.counters["dt_max"]        = static_cast<double>(rplan.get_dt_max());
    state.counters["subband_depth"] = static_cast<double>(kRtSubbandDepth);
    state.counters["max_boxcar"]    = static_cast<double>(rt_max_boxcar(p));
    state.counters["nwidths"] =
        static_cast<double>(search.get_search_plans().front().widths.size());
    state.counters["input_nbits"] = packed ? 8 : 32;
    state.counters["threads"]     = threads;
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    SizeType rows = 0;
    for (const auto& plane : search.get_search_plans()) {
        rows += plane.dm_grid.size();
    }
    state.counters["search_rows"] = static_cast<double>(rows);
    ctx->mem.report(state);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}

[[maybe_unused]] const bool kStreamRegistered = [] {
    // parkes_uwl: the real-time sweep's 64 us point is this reference point
    // (depth 4, wtsp 1.5); its blocks take 5-20 s on 8 threads after ~8
    // warm-up blocks, so it is measured once, there.
    if (rt_sweep_enabled()) {
        return false;
    }
    const std::vector<int> thread_counts = cpu_threads();
    for (const float wtsp : wtsp_sweep()) {
        for (const auto threads : thread_counts) {
            for (const bool packed : {false, true}) {
                benchmark::RegisterBenchmark(
                    fmt::format("BM_fdmt_boxcar_stream/{}/{}wtsp:{:g}/cpu{}",
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

[[maybe_unused]] const bool kRtRegistered = [] {
    if (!rt_sweep_enabled()) {
        return false;
    }
    const int threads = cpu_threads_max();
    for (const auto& p : rt_points()) {
        for (const bool packed : {true, false}) {
            benchmark::RegisterBenchmark(
                rt_name(fmt::format("cpu{}", threads), p, packed),
                [p, packed, threads](benchmark::State& s) {
                    bench_rt_cpu(s, p, packed, threads);
                })
                ->UseRealTime()
                ->Unit(benchmark::kMillisecond)
                // Multi-second blocks are stable; the warm-up dominates.
                ->Repetitions(p.tsamp < 1e-4F ? 1 : 3);
        }
    }
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
