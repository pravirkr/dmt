#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/algorithms/fdmt_fft_boxcar.hpp"

#include "dmt/boxcar.hpp"
#include "dmt/gpu_compat.cuh"

#include "bench_gpu_utils.cuh"
#include "suite/suite_common.hpp"
#include "suite/suite_memory_gpu.cuh"

// GPU half of the boxcar suite (run_suite.py --suite boxcar): streaming
// FDMTBoxcar at the reference point of the selected configuration
// (DMT_BENCH_CONFIG), device-resident:
// - transform: FDMT alone, writing the DM-time plane (the floor to beat);
// - separate: FDMT to the plane, then the boxcar scorer over the
//   materialised root (full band only);
// - fused: FDMTBoxcar, float32 or uint8 input, at subband depth 0 and the
//   preset's depth;
// - fft_transform / fft_fused: FDMT-FFT (fractional delays) to the plane,
//   and FDMTFFTBoxcar searching its segments.
// Widths follow the 100 ms rule, max_boxcar = floor(0.1 s / tsamp), at two
// spacings per configuration. Input is Gaussian noise at threshold 6.
// parkes_uwl adds the real-time sweep (BM_fdmt_boxcar_rt, suite_common.hpp):
// depth 4 at 64, 256, 512 and 1024 us on 2.097 s blocks.
namespace dmt::bench_suite {
namespace {

using algorithms::FDMT;
using algorithms::FDMTBoxcar;

// The separate benchmark scores a plane the FDMT has already written. The
// scorer's driver would be a second, unused transform engine.
class MaterializedRootDriver final : public algorithms::detail::BoxcarDriver {
public:
    using algorithms::detail::BoxcarDriver::search; // the packed overloads

    void search(std::span<const float>,
                algorithms::detail::FDMTSearchSink&) override {
        throw std::logic_error(
            "separate bench scores a materialized DM-time plane");
    }
    void search(DeviceSpan<const float>,
                algorithms::detail::FDMTSearchSink&,
                Stream) override {
        throw std::logic_error(
            "separate bench scores a materialized DM-time plane");
    }
    void reset(Stream) override {}
    [[nodiscard]] algorithms::FDMTMemoryUsage memory() const noexcept override {
        return {};
    }
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override { return 0; }
    [[nodiscard]] SizeType fuse_levels() const noexcept override { return 0; }
};

template <typename T> DeviceSpan<T> dspan(thrust::device_vector<T>& v) {
    return {thrust::raw_pointer_cast(v.data()), v.size()};
}
template <typename T>
DeviceSpan<const T> dspan_c(const thrust::device_vector<T>& v) {
    return {thrust::raw_pointer_cast(v.data()), v.size()};
}

thrust::device_vector<float> make_noise(SizeType n) {
    thrust::device_vector<float> v(n);
    thrust::transform(thrust::counting_iterator<SizeType>(0),
                      thrust::counting_iterator<SizeType>(n), v.begin(),
                      [] __device__(SizeType i) {
                          thrust::default_random_engine rng(1234);
                          rng.discard(2 * i);
                          thrust::normal_distribution<float> dist;
                          return dist(rng);
                      });
    return v;
}

/// make_noise() quantised to 8 bits (127 + 12 x, clamped), generated
/// directly so no float copy of the block is ever resident (it would show
/// in the point's peak device memory).
thrust::device_vector<uint8_t> make_packed_noise(SizeType n) {
    thrust::device_vector<uint8_t> v(n);
    thrust::transform(thrust::counting_iterator<SizeType>(0),
                      thrust::counting_iterator<SizeType>(n), v.begin(),
                      [] __device__(SizeType i) {
                          thrust::default_random_engine rng(1234);
                          rng.discard(2 * i);
                          thrust::normal_distribution<float> dist;
                          const float f = dist(rng);
                          return static_cast<uint8_t>(fminf(
                              fmaxf(rintf(127.0F + 12.0F * f), 0.0F), 255.0F));
                      });
    return v;
}

BoxcarConfig
make_config(float wtsp,
            bool packed,
            BoxcarScreenMode screen_mode = BoxcarScreenMode::kPrefix,
            bool trapezoid               = false) {
    return BoxcarConfig(max_boxcar_100ms(), wtsp, 6.0F, packed ? 12.0F : 1.0F,
                        PeakFindingMode::kContiguousPeakPerWidth, 131072, 0,
                        true, packed ? 8 : 32, packed ? 127.0F : 0.0F,
                        trapezoid, OverflowPolicy::kDropNew, screen_mode);
}

/// DMT_BENCH_CHUNK overrides the automatic FDMT chunk length (tuning).
SizeType bench_chunk() {
    const char* env = std::getenv("DMT_BENCH_CHUNK"); // NOLINT
    return env != nullptr ? static_cast<SizeType>(std::stoull(env)) : 0;
}

const Point& ref_point() {
    static const Point p{Sweep::kNdms, kNsampsRef, kDmMaxRef, 32};
    return p;
}

uint64_t warmup_samples(std::span<const plans::BoxcarPlanePlan> planes) {
    uint64_t warmup = 1;
    for (const auto& plane : planes) {
        for (const auto support : plane.warmup_samples) {
            warmup = std::max(warmup, support + plane.widths.back() - 1);
        }
    }
    return warmup;
}

void common_counters(benchmark::State& state,
                     const plans::FDMTPlan& plan,
                     SizeType bytes,
                     SizeType nwidths) {
    set_counters(state, plan, ref_point(), 0);
    state.counters["realtime_factor"] =
        benchmark::Counter(static_cast<double>(kNsampsRef) * kTsamp,
                           benchmark::Counter::kIsIterationInvariantRate);
    state.counters["working_bytes"] = static_cast<double>(bytes);
    state.counters["max_boxcar"]    = static_cast<double>(max_boxcar_100ms());
    state.counters["nwidths"]       = static_cast<double>(nwidths);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(kNchans * kNsampsRef));
}

void bench_transform(benchmark::State& state) {
    GpuPeakMemory mem;
    mem.begin();
    const auto plan = make_plan(ref_point());
    FDMT fdmt(kFMin, kFMax, kNchans, kNsampsRef, kTsamp, dm_grid(ref_point()),
              true, "valid", bench_gpu_exec(), 1, algorithms::kFDMTAutoFuse,
              true, bench_chunk());
    const auto wf = make_noise(kNchans * kNsampsRef);
    thrust::device_vector<float> dmt(plan.get_dmt_size());
    fdmt.execute(dspan_c(wf), dspan(dmt));
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    mem.pause();
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        fdmt.execute(dspan_c(wf), dspan(dmt));
    }
    mem.report(state);
    common_counters(state, plan,
                    fdmt.get_memory_usage().total() +
                        dmt.size() * sizeof(float) + wf.size() * sizeof(float),
                    0);
    state.counters["chunk_nsamps"] =
        static_cast<double>(fdmt.get_chunk_nsamps());
}

// FDMT into the plane, then the same GPU scorer over the materialised root.
void bench_separate(benchmark::State& state,
                    float wtsp,
                    BoxcarScreenMode screen_mode = BoxcarScreenMode::kPrefix,
                    bool trapezoid               = false) {
    GpuPeakMemory mem;
    mem.begin();
    const auto plan   = make_plan(ref_point());
    const auto config = make_config(wtsp, false, screen_mode, trapezoid);
    FDMT fdmt(kFMin, kFMax, kNchans, kNsampsRef, kTsamp, dm_grid(ref_point()),
              true, "valid", bench_gpu_exec());
    const auto planes =
        algorithms::detail::make_boxcar_plans(plan, config, true);
    algorithms::detail::FDMTEngineConfig cfg;
    cfg.exec            = bench_gpu_exec();
    cfg.fuse_levels     = 0;
    cfg.max_fuse_levels = 0;
    auto scorer         = algorithms::detail::make_fdmt_boxcar_gpu(
        plan, config, planes, cfg, std::make_unique<MaterializedRootDriver>());
    const auto wf = make_noise(kNchans * kNsampsRef);
    thrust::device_vector<float> dmt(plan.get_dmt_size());
    uint64_t position = 0;
    const auto run    = [&] {
        fdmt.execute(dspan_c(wf), dspan(dmt));
        const algorithms::detail::FDMTSearchInput input{
            .state             = thrust::raw_pointer_cast(dmt.data()),
            .row_stride        = kNsampsRef,
            .beam_stride       = dmt.size(),
            .nsamps            = kNsampsRef,
            .sample_begin      = position,
            .materialized_root = true,
        };
        scorer->consume(plan.get_niters(), input);
        position += kNsampsRef;
    };
    for (uint64_t filled = 0; filled < warmup_samples(planes);
         filled += kNsampsRef) {
        run();
    }
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    mem.pause();
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        run();
    }
    mem.report(state);
    const auto transform = fdmt.get_memory_usage().total();
    const auto score     = scorer->memory_usage().total();
    common_counters(state, plan,
                    transform + score + dmt.size() * sizeof(float) +
                        wf.size() * sizeof(float),
                    planes.front().widths.size());
    state.counters["screen_mode"]   = static_cast<double>(screen_mode);
    state.counters["use_trapezoid"] = trapezoid ? 1.0 : 0.0;
}

void bench_fft_transform(benchmark::State& state) {
    GpuPeakMemory mem;
    mem.begin();
    const auto plan = make_plan(ref_point());
    algorithms::FDMTFFT fft(kFMin, kFMax, kNchans, kNsampsRef, kTsamp,
                            dm_grid(ref_point()), true, "valid",
                            bench_gpu_exec());
    const auto wf = make_noise(kNchans * kNsampsRef);
    thrust::device_vector<float> dmt(plan.get_dmt_size());
    fft.execute(dspan_c(wf), dspan(dmt));
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    mem.pause();
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        fft.execute(dspan_c(wf), dspan(dmt));
    }
    mem.report(state);
    common_counters(state, plan,
                    fft.get_memory_usage().total() +
                        dmt.size() * sizeof(float) + wf.size() * sizeof(float),
                    0);
}

void bench_fft_fused(benchmark::State& state,
                     float wtsp,
                     BoxcarScreenMode screen_mode = BoxcarScreenMode::kPrefix,
                     bool trapezoid               = false) {
    GpuPeakMemory mem;
    mem.begin();
    auto plan         = make_plan(ref_point());
    const auto config = make_config(wtsp, false, screen_mode, trapezoid);
    algorithms::FDMTFFTBoxcar search(std::move(plan), config, true,
                                     bench_gpu_exec());
    const auto wf = make_noise(kNchans * kNsampsRef);
    cudaStream_t stream{};
    BENCH_GPU_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto run = [&] { (void)search.execute(dspan_c(wf), Stream{stream}); };
    for (uint64_t filled = 0;
         filled < warmup_samples(search.get_search_plans()) + kNsampsRef;
         filled += kNsampsRef) {
        run();
    }
    BENCH_GPU_TRY(cudaStreamSynchronize(stream));
    mem.pause();
    for (auto _ : state) {
        const GPUEventTimer timer{state, stream};
        run();
    }
    mem.report(state);
    const auto emitted = search.copy_result(Stream{stream}).emitted_count;
    BENCH_GPU_TRY(cudaStreamDestroy(stream));
    common_counters(state, search.get_plan(),
                    search.get_memory_usage().total() +
                        wf.size() * sizeof(float),
                    search.get_search_plans().front().widths.size());
    state.counters["screen_mode"]   = static_cast<double>(screen_mode);
    state.counters["use_trapezoid"] = trapezoid ? 1.0 : 0.0;
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    state.counters["candidates_per_block"] = static_cast<double>(emitted);
}

/// A warmed FDMTBoxcar, its device noise input and a stream, built once per
/// benchmark point (cached_context) instead of once per repetition.
struct GpuContext {
    std::unique_ptr<FDMTBoxcar> search;
    thrust::device_vector<float> wf;
    thrust::device_vector<uint8_t> wf8;
    cudaStream_t stream{};
    std::string error;
    /// Host and device peaks from construction through warm-up (sampled)
    /// and the timed blocks (host high-water, last device sample).
    GpuPeakMemory mem;
    GpuContext()                             = default;
    GpuContext(const GpuContext&)            = delete;
    GpuContext& operator=(const GpuContext&) = delete;
    ~GpuContext() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
    void run() {
        if (!wf8.empty()) {
            (void)search->execute(dspan_c(wf8), Stream{stream});
        } else {
            (void)search->execute(dspan_c(wf), Stream{stream});
        }
    }
};

std::shared_ptr<GpuContext> gpu_context(const std::string& key,
                                        plans::FDMTPlan plan,
                                        const BoxcarConfig& config) {
    return cached_context<GpuContext>(key, [&] {
        auto ctx = std::make_shared<GpuContext>();
        ctx->mem.begin();
        const SizeType n     = plan.get_nchans() * plan.get_nsamps();
        const SizeType block = plan.get_nsamps();
        try {
            ctx->search = std::make_unique<FDMTBoxcar>(
                std::move(plan), config, true, bench_gpu_exec(), 1,
                algorithms::kFDMTAutoFuse, bench_chunk());
        } catch (const std::exception& e) {
            ctx->error = e.what();
            return ctx;
        }
        if (config.input_nbits != 32) {
            ctx->wf8 = make_packed_noise(n);
        } else {
            ctx->wf = make_noise(n);
        }
        BENCH_GPU_TRY(
            cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking));
        for (uint64_t filled = 0;
             filled < warmup_samples(ctx->search->get_search_plans());
             filled += block) {
            ctx->run();
        }
        BENCH_GPU_TRY(cudaStreamSynchronize(ctx->stream));
        ctx->mem.pause();
        return ctx;
    });
}

void bench_fused(benchmark::State& state,
                 float wtsp,
                 SizeType subband_depth,
                 bool packed,
                 BoxcarScreenMode screen_mode = BoxcarScreenMode::kPrefix,
                 bool trapezoid               = false) {
    auto plan         = make_plan(ref_point(), subband_depth);
    const auto config = make_config(wtsp, packed, screen_mode, trapezoid);
    auto ctx          = gpu_context(
        fmt::format("fused/{}/{}/{}/{}/{}", wtsp, subband_depth, packed,
                    screen_mode_name(screen_mode), trapezoid),
        std::move(plan), config);
    if (!ctx->error.empty()) {
        state.SkipWithError(ctx->error);
        return;
    }
    auto& search = *ctx->search;
    for (auto _ : state) {
        const GPUEventTimer timer{state, ctx->stream};
        ctx->run();
    }
    const SizeType emitted =
        search.copy_result(Stream{ctx->stream}).emitted_count;
    const auto mem = search.get_memory_usage();
    const SizeType input_bytes =
        packed ? ctx->wf8.size() : ctx->wf.size() * sizeof(float);
    common_counters(state, search.get_plan(), mem.total() + input_bytes,
                    search.get_search_plans().front().widths.size());
    state.counters["input_nbits"]   = packed ? 8 : 32;
    state.counters["subband_depth"] = static_cast<double>(subband_depth);
    state.counters["screen_mode"]   = static_cast<double>(screen_mode);
    state.counters["use_trapezoid"] = trapezoid ? 1.0 : 0.0;
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    state.counters["fuse_levels"] =
        static_cast<double>(search.get_fuse_levels());
    state.counters["candidates_per_block"] = static_cast<double>(emitted);
    ctx->mem.report(state);
}

/// One real-time sweep point (suite_common.hpp rt_points()): FDMTBoxcar
/// at depth 4, device-resident, default memory budget.
void bench_rt(benchmark::State& state,
              RtPoint p,
              bool packed,
              BoxcarScreenMode screen_mode = BoxcarScreenMode::kPrefix,
              bool trapezoid               = false) {
    auto plan = make_rt_plan(p);
    const BoxcarConfig config(rt_max_boxcar(p), kRtWtsp, 6.0F,
                              packed ? 12.0F : 1.0F,
                              PeakFindingMode::kContiguousPeakPerWidth, 131072,
                              0, true, packed ? 8 : 32, packed ? 127.0F : 0.0F,
                              trapezoid, OverflowPolicy::kDropNew, screen_mode);
    auto ctx =
        gpu_context(fmt::format("{}/{}/{}", rt_name(DMT_GPU_NAME, p, packed),
                                screen_mode_name(screen_mode), trapezoid),
                    std::move(plan), config);
    if (!ctx->error.empty()) {
        state.SkipWithError(ctx->error);
        return;
    }
    auto& search    = *ctx->search;
    const auto& wf  = ctx->wf;
    const auto& wf8 = ctx->wf8;
    for (auto _ : state) {
        const GPUEventTimer timer{state, ctx->stream};
        ctx->run();
    }
    const auto mem    = search.get_memory_usage();
    const auto& rplan = search.get_plan();
    state.counters["realtime_factor"] =
        benchmark::Counter(static_cast<double>(p.nsamps) * p.tsamp,
                           benchmark::Counter::kIsIterationInvariantRate);
    state.counters["working_bytes"] = static_cast<double>(
        mem.total() + (packed ? wf8.size() : wf.size() * sizeof(float)));
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
    state.counters["input_nbits"]   = packed ? 8 : 32;
    state.counters["screen_mode"]   = static_cast<double>(screen_mode);
    state.counters["use_trapezoid"] = trapezoid ? 1.0 : 0.0;
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    SizeType rows = 0;
    for (const auto& plane : search.get_search_plans()) {
        rows += plane.dm_grid.size();
    }
    state.counters["search_rows"] = static_cast<double>(rows);
    ctx->mem.report(state);
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(kNchans * p.nsamps));
}

[[maybe_unused]] const bool kRegistered = [] {
    const auto cfg                              = suite_config().name;
    const std::vector<BoxcarScreenMode> screens = boxcar_screen_modes();
    const std::vector<bool> trapezoids          = boxcar_trapezoid_modes();
    // A transform that cannot fit the device (FDMT-FFT on UWL needs
    // 510k-sample segments) is reported as skipped, not aborted.
    const auto reg = [](std::string name, auto fn) {
        benchmark::RegisterBenchmark(name,
                                     [fn](benchmark::State& s) {
                                         try {
                                             fn(s);
                                         } catch (const std::exception& e) {
                                             s.SkipWithError(e.what());
                                         }
                                     })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond)
            ->MinWarmUpTime(0.2);
    };
    reg(fmt::format("BM_fdmt_boxcar_cuda/{}/transform", cfg),
        [](benchmark::State& s) { bench_transform(s); });
    reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fft_transform", cfg),
        [](benchmark::State& s) { bench_fft_transform(s); });
    for (const float wtsp : wtsp_sweep()) {
        for (const bool trap : trapezoids) {
            for (const auto screen : screens) {
                const auto suffix = boxcar_mode_suffix(screen, trap);
                reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fft_fused/wtsp:{:g}{}",
                                cfg, wtsp, suffix),
                    [wtsp, screen, trap](benchmark::State& s) {
                        bench_fft_fused(s, wtsp, screen, trap);
                    });
                reg(fmt::format("BM_fdmt_boxcar_cuda/{}/separate/wtsp:{:g}{}",
                                cfg, wtsp, suffix),
                    [wtsp, screen, trap](benchmark::State& s) {
                        bench_separate(s, wtsp, screen, trap);
                    });
            }
        }
        std::vector<SizeType> depths{0};
        if (suite_config().subband_depth != 0) {
            depths.push_back(suite_config().subband_depth);
        }
        // DMT_BENCH_SB adds one more subband depth (developer runs).
        if (const char* env = std::getenv("DMT_BENCH_SB")) { // NOLINT
            const auto sb = static_cast<SizeType>(std::stoul(env));
            if (std::ranges::find(depths, sb) == depths.end()) {
                depths.push_back(sb);
            }
        }
        for (const auto sb : depths) {
            for (const bool trap : trapezoids) {
                for (const auto screen : screens) {
                    const auto suffix = boxcar_mode_suffix(screen, trap);
                    reg(fmt::format(
                            "BM_fdmt_boxcar_cuda/{}/fused/wtsp:{:g}/sb:{}{}",
                            cfg, wtsp, sb, suffix),
                        [wtsp, sb, screen, trap](benchmark::State& s) {
                            bench_fused(s, wtsp, sb, false, screen, trap);
                        });
                    reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fused_uint8/"
                                    "wtsp:{:g}/sb:{}{}",
                                    cfg, wtsp, sb, suffix),
                        [wtsp, sb, screen, trap](benchmark::State& s) {
                            bench_fused(s, wtsp, sb, true, screen, trap);
                        });
                }
            }
        }
    }
    if (rt_sweep_enabled()) {
        for (const auto& p : rt_points()) {
            for (const bool packed : {true, false}) {
                for (const bool trap : trapezoids) {
                    for (const auto screen : screens) {
                        reg(fmt::format("{}{}",
                                        rt_name(DMT_GPU_NAME, p, packed),
                                        boxcar_mode_suffix(screen, trap)),
                            [p, packed, screen, trap](benchmark::State& s) {
                                bench_rt(s, p, packed, screen, trap);
                            });
                    }
                }
            }
        }
    }
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
