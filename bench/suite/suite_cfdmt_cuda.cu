// CohFDMT GPU benchmarks of the published suite (see suite_cfdmt_common.hpp):
// "<gpu>" is device-resident (input and output on the GPU, timed with
// events); "<gpu>_host" takes host arrays, so it includes the PCIe
// transfers of the input block and the result.

#include <chrono>
#include <cstdint>
#include <format>
#include <span>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include <benchmark/benchmark.h>

#include "dmt/algorithms/cfdmt.hpp"

#include "bench_gpu_utils.cuh"
#include "suite_cfdmt_common.hpp"

namespace dmt::bench_suite::cfdmt {
namespace {

bool over_device_budget(benchmark::State& state,
                        const plans::CohFDMTPlan& plan) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    BENCH_GPU_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    const double need   = estimate_bytes(plan);
    const double budget = 0.9 * static_cast<double>(free_bytes);
    if (need > budget) {
        state.SkipWithError(
            std::format("skipped: needs ~{:.1f} GB of device memory, {:.1f} "
                        "GB free",
                        need / 1.0737e9, budget / 1.0737e9));
        return true;
    }
    return false;
}

template <bool Host> void bench_gpu(benchmark::State& state, const Point& p) {
    const plans::CohFDMTPlan plan(make_config(p));
    if (over_device_budget(state, plan)) {
        return;
    }
    const algorithms::CohFDMT search(make_config(p), bench_gpu_exec());
    const auto in = make_packed_input(search.get_input_size());
    if constexpr (Host) {
        std::vector<float> dmt(search.get_dmt_size());
        search.execute<uint8_t>(std::span<const uint8_t>(in), dmt);
        for (auto _ : state) {
            const auto t0 = std::chrono::steady_clock::now();
            search.execute<uint8_t>(std::span<const uint8_t>(in), dmt);
            state.SetIterationTime(
                std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              t0)
                    .count());
        }
    } else {
        const thrust::device_vector<uint8_t> d_in(in.begin(), in.end());
        thrust::device_vector<float> d_dmt(search.get_dmt_size());
        const DeviceSpan<const uint8_t> din(
            thrust::raw_pointer_cast(d_in.data()), d_in.size());
        const DeviceSpan<float> dout(thrust::raw_pointer_cast(d_dmt.data()),
                                     d_dmt.size());
        search.execute<uint8_t>(din, dout);
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        cudaEvent_t start{};
        cudaEvent_t stop{};
        BENCH_GPU_TRY(cudaEventCreate(&start));
        BENCH_GPU_TRY(cudaEventCreate(&stop));
        for (auto _ : state) {
            BENCH_GPU_TRY(cudaEventRecord(start));
            search.execute<uint8_t>(din, dout);
            BENCH_GPU_TRY(cudaEventRecord(stop));
            BENCH_GPU_TRY(cudaEventSynchronize(stop));
            float ms = 0.0F;
            BENCH_GPU_TRY(cudaEventElapsedTime(&ms, start, stop));
            state.SetIterationTime(static_cast<double>(ms) / 1000.0);
        }
        BENCH_GPU_TRY(cudaEventDestroy(start));
        BENCH_GPU_TRY(cudaEventDestroy(stop));
    }
    set_counters(state, plan, 1);
}

void register_gpu() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        return;
    }
    for (const auto& p : points()) {
        benchmark::RegisterBenchmark(
            name(DMT_GPU_NAME, p),
            [p](benchmark::State& s) { bench_gpu<false>(s, p); })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);
        if (p.sweep == Sweep::kRef || p.sweep == Sweep::kNbits) {
            benchmark::RegisterBenchmark(
                name(DMT_GPU_NAME "_host", p),
                [p](benchmark::State& s) { bench_gpu<true>(s, p); })
                ->UseManualTime()
                ->Unit(benchmark::kMillisecond);
        }
    }
}

[[maybe_unused]] const bool kRegistered = [] {
    register_gpu();
    return true;
}();

} // namespace
} // namespace dmt::bench_suite::cfdmt
