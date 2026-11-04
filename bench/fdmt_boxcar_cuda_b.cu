#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
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

// Streaming FDMTBoxcar on the GPU at the reference point of the selected
// suite configuration (DMT_BENCH_CONFIG), device-resident unless noted:
// - transform: FDMT alone, writing the DM-time plane (the floor to beat);
// - separate: FDMT to the plane, then the boxcar scorer over the
//   materialised root (full band only);
// - fused: FDMTBoxcar, float32 or uint8 input, device or host API;
// - fft_transform / fft_fused: FDMT-FFT (fractional delays) to the plane,
//   and FDMTFFTBoxcar searching its segments.
// Widths follow the 100 ms rule, max_boxcar = floor(0.1 s / tsamp), at two
// spacings per configuration. Input is Gaussian noise at threshold 6.
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

thrust::device_vector<uint8_t>
make_packed(const thrust::device_vector<float>& x) {
    thrust::device_vector<uint8_t> v(x.size());
    thrust::transform(x.begin(), x.end(), v.begin(), [] __device__(float f) {
        return static_cast<uint8_t>(
            fminf(fmaxf(rintf(127.0F + 12.0F * f), 0.0F), 255.0F));
    });
    return v;
}

BoxcarConfig make_config(float wtsp, bool packed) {
    return BoxcarConfig(max_boxcar_100ms(), wtsp, 6.0F, packed ? 12.0F : 1.0F,
                        PeakFindingMode::kContiguousPeakPerWidth, 1U << 20U, 0,
                        true, packed ? 8 : 32, packed ? 127.0F : 0.0F);
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
    const auto plan = make_plan(ref_point());
    FDMT fdmt(kFMin, kFMax, kNchans, kNsampsRef, kTsamp, dm_grid(ref_point()),
              true, "valid", bench_gpu_exec(), 1, algorithms::kFDMTAutoFuse,
              true, bench_chunk());
    const auto wf = make_noise(kNchans * kNsampsRef);
    thrust::device_vector<float> dmt(plan.get_dmt_size());
    fdmt.execute(dspan_c(wf), dspan(dmt));
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        fdmt.execute(dspan_c(wf), dspan(dmt));
    }
    const auto mem = fdmt.get_memory_usage();
    common_counters(state, plan,
                    mem.total() + dmt.size() * sizeof(float) +
                        wf.size() * sizeof(float),
                    0);
    state.counters["chunk_nsamps"] =
        static_cast<double>(fdmt.get_chunk_nsamps());
}

// FDMT into the plane, then the same GPU scorer over the materialised root.
void bench_separate(benchmark::State& state, float wtsp) {
    const auto plan   = make_plan(ref_point());
    const auto config = make_config(wtsp, false);
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
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        run();
    }
    const auto transform = fdmt.get_memory_usage().total();
    const auto score     = scorer->memory_usage().total();
    common_counters(state, plan,
                    transform + score + dmt.size() * sizeof(float) +
                        wf.size() * sizeof(float),
                    planes.front().widths.size());
}

void bench_fft_transform(benchmark::State& state) {
    const auto plan = make_plan(ref_point());
    algorithms::FDMTFFT fft(kFMin, kFMax, kNchans, kNsampsRef, kTsamp,
                            dm_grid(ref_point()), true, "valid",
                            bench_gpu_exec());
    const auto wf = make_noise(kNchans * kNsampsRef);
    thrust::device_vector<float> dmt(plan.get_dmt_size());
    fft.execute(dspan_c(wf), dspan(dmt));
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        fft.execute(dspan_c(wf), dspan(dmt));
    }
    common_counters(state, plan,
                    fft.get_memory_usage().total() +
                        dmt.size() * sizeof(float) + wf.size() * sizeof(float),
                    0);
}

void bench_fft_fused(benchmark::State& state, float wtsp) {
    auto plan         = make_plan(ref_point());
    const auto config = make_config(wtsp, false);
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
    for (auto _ : state) {
        const GPUEventTimer timer{state, stream};
        run();
    }
    const auto emitted = search.copy_result(Stream{stream}).emitted_count;
    BENCH_GPU_TRY(cudaStreamDestroy(stream));
    common_counters(state, search.get_plan(),
                    search.get_memory_usage().total() +
                        wf.size() * sizeof(float),
                    search.get_search_plans().front().widths.size());
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    state.counters["candidates_per_block"] = static_cast<double>(emitted);
}

enum class Api { kDevice, kHost };

void bench_fused(benchmark::State& state,
                 float wtsp,
                 SizeType subband_depth,
                 bool packed,
                 Api api) {
    auto plan         = make_plan(ref_point(), subband_depth);
    const auto config = make_config(wtsp, packed);
    FDMTBoxcar search(std::move(plan), config, true, bench_gpu_exec(), 1,
                      algorithms::kFDMTAutoFuse, bench_chunk());
    const auto wf = make_noise(kNchans * kNsampsRef);
    const auto wf8 =
        packed ? make_packed(wf) : thrust::device_vector<uint8_t>{};
    std::vector<float> host_wf;
    std::vector<uint8_t> host_wf8;
    if (api == Api::kHost) {
        if (packed) {
            host_wf8.resize(wf8.size());
            thrust::copy(wf8.begin(), wf8.end(), host_wf8.begin());
        } else {
            host_wf.resize(wf.size());
            thrust::copy(wf.begin(), wf.end(), host_wf.begin());
        }
    }
    cudaStream_t stream{};
    BENCH_GPU_TRY(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    SizeType emitted = 0;
    const auto run   = [&] {
        if (api == Api::kHost) {
            const auto r =
                packed ? search.execute(std::span<const uint8_t>(host_wf8))
                       : search.execute(std::span<const float>(host_wf));
            emitted = r.emitted_count;
        } else if (packed) {
            (void)search.execute(dspan_c(wf8), Stream{stream});
        } else {
            (void)search.execute(dspan_c(wf), Stream{stream});
        }
    };
    for (uint64_t filled = 0;
         filled < warmup_samples(search.get_search_plans());
         filled += kNsampsRef) {
        run();
    }
    BENCH_GPU_TRY(cudaStreamSynchronize(stream));
    for (auto _ : state) {
        const GPUEventTimer timer{state, api == Api::kHost ? nullptr : stream};
        run();
    }
    if (api == Api::kDevice) {
        emitted = search.copy_result(Stream{stream}).emitted_count;
    }
    BENCH_GPU_TRY(cudaStreamDestroy(stream));
    const auto mem = search.get_memory_usage();
    const SizeType input_bytes =
        api == Api::kHost
            ? (packed ? host_wf8.size() : host_wf.size() * sizeof(float))
            : (packed ? wf8.size() : wf.size() * sizeof(float));
    common_counters(state, search.get_plan(), mem.total() + input_bytes,
                    search.get_search_plans().front().widths.size());
    state.counters["input_nbits"]   = packed ? 8 : 32;
    state.counters["subband_depth"] = static_cast<double>(subband_depth);
    state.counters["chunk_nsamps"] =
        static_cast<double>(search.get_chunk_nsamps());
    state.counters["fuse_levels"] =
        static_cast<double>(search.get_fuse_levels());
    state.counters["candidates_per_block"] = static_cast<double>(emitted);
}

[[maybe_unused]] const bool kRegistered = [] {
    const auto cfg = suite_config().name;
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
        reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fft_fused/wtsp:{:g}", cfg,
                        wtsp),
            [wtsp](benchmark::State& s) { bench_fft_fused(s, wtsp); });
        reg(fmt::format("BM_fdmt_boxcar_cuda/{}/separate/wtsp:{:g}", cfg, wtsp),
            [wtsp](benchmark::State& s) { bench_separate(s, wtsp); });
        std::vector<SizeType> depths{0};
        if (suite_config().subband_depth != 0) {
            depths.push_back(suite_config().subband_depth);
        }
        for (const auto sb : depths) {
            reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fused/wtsp:{:g}/sb:{}", cfg,
                            wtsp, sb),
                [wtsp, sb](benchmark::State& s) {
                    bench_fused(s, wtsp, sb, false, Api::kDevice);
                });
            reg(fmt::format(
                    "BM_fdmt_boxcar_cuda/{}/fused_uint8/wtsp:{:g}/sb:{}", cfg,
                    wtsp, sb),
                [wtsp, sb](benchmark::State& s) {
                    bench_fused(s, wtsp, sb, true, Api::kDevice);
                });
            reg(fmt::format("BM_fdmt_boxcar_cuda/{}/fused_host/wtsp:{:g}/sb:{}",
                            cfg, wtsp, sb),
                [wtsp, sb](benchmark::State& s) {
                    bench_fused(s, wtsp, sb, false, Api::kHost);
                });
        }
    }
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
