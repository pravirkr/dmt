// GPU half of dmt_bench_suite (CUDA or HIP; backend names "cuda" / "hip");
// see suite_common.hpp for the configuration.
//
// "cuda" benchmarks are device-resident (input and output already on the
// GPU, timed with CUDA events). "cuda_host" benchmarks take host arrays, so
// they include the PCIe transfers a streaming pipeline would pay.

#include <cstdint>
#include <format>
#include <new>
#include <span>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include <benchmark/benchmark.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

#include "bench_gpu_utils.cuh"
#include "suite_common.hpp"

namespace dmt::bench_suite {
namespace {

using algorithms::DDMT;
using algorithms::FDMT;
using algorithms::FDMTFFT;

template <typename T> DeviceSpan<T> dspan(thrust::device_vector<T>& v) {
    return {thrust::raw_pointer_cast(v.data()), v.size()};
}
template <typename T>
DeviceSpan<const T> dspan_c(const thrust::device_vector<T>& v) {
    return {thrust::raw_pointer_cast(v.data()), v.size()};
}

// Device memory check: the estimate must fit 90% of the free device memory.
bool skip_if_over_device_budget(benchmark::State& state,
                                Algo algo,
                                const plans::FDMTPlan& plan,
                                const Point& p) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    BENCH_GPU_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    const double need   = estimate_bytes(algo, plan, p);
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

void bench_fdmt_cuda(benchmark::State& state, Point p) {
    const auto plan = make_plan(p);
    if (skip_if_over_device_budget(state, Algo::kFDMT, plan, p)) {
        return;
    }
    FDMT fdmt(kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1, true,
              "valid", bench_gpu_exec());
    thrust::device_vector<float> dmt(plan.get_buffer_size());
    if (p.nbits == 32) {
        const auto host = make_float_input(kNchans * p.nsamps);
        const thrust::device_vector<float> wf(host.begin(), host.end());
        fdmt.execute(dspan_c(wf), dspan(dmt));
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            fdmt.execute(dspan_c(wf), dspan(dmt));
        }
    } else {
        const auto host = make_packed_input(packed_bytes(p.nsamps, p.nbits));
        const thrust::device_vector<uint8_t> wf(host.begin(), host.end());
        fdmt.execute(dspan_c(wf), p.nbits, dspan(dmt));
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            fdmt.execute(dspan_c(wf), p.nbits, dspan(dmt));
        }
    }
    set_counters(state, plan, p, 0);
}

// Host arrays in and out: includes H2D of the (packed) input and D2H of the
// result, as a pipeline without GPU-resident input would see.
void bench_fdmt_cuda_host(benchmark::State& state, Point p) {
    const auto plan = make_plan(p);
    if (skip_if_over_device_budget(state, Algo::kFDMT, plan, p)) {
        return;
    }
    FDMT fdmt(kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1, true,
              "valid", bench_gpu_exec());
    std::vector<float> dmt(plan.get_buffer_size());
    if (p.nbits == 32) {
        const auto wf = make_float_input(kNchans * p.nsamps);
        fdmt.execute(wf, dmt); // warm-up, also allocates the staging
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            fdmt.execute(wf, dmt);
        }
    } else {
        const auto wf = make_packed_input(packed_bytes(p.nsamps, p.nbits));
        fdmt.execute(std::span<const uint8_t>(wf), p.nbits, dmt);
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            fdmt.execute(std::span<const uint8_t>(wf), p.nbits, dmt);
        }
    }
    set_counters(state, plan, p, 0);
}

void bench_fdmt_fft_cuda(benchmark::State& state, Point p) {
    const auto plan = make_plan(p);
    if (skip_if_over_device_budget(state, Algo::kFDMTFFT, plan, p)) {
        return;
    }
    try {
        FDMTFFT fdmt(kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1,
                     true, "valid", bench_gpu_exec());
        const auto host = make_float_input(kNchans * p.nsamps);
        const thrust::device_vector<float> wf(host.begin(), host.end());
        thrust::device_vector<float> dmt(plan.get_dmt_size());
        fdmt.execute(dspan_c(wf), dspan(dmt));
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            fdmt.execute(dspan_c(wf), dspan(dmt));
        }
    } catch (const std::bad_alloc&) {
        state.SkipWithError("skipped: out of device memory");
        return;
    }
    set_counters(state, plan, p, 0);
}

void bench_ddmt_cuda(benchmark::State& state, Point p) {
    const auto plan = make_plan(p);
    if (skip_if_over_device_budget(state, Algo::kDDMT, plan, p)) {
        return;
    }
    const auto dms = plan.get_dm_grid_final();
    DDMT ddmt(kFMin, kFMax, kNchans, kTsamp, dms, bench_gpu_exec(), p.nbits);
    const auto ndms   = dms.size();
    const auto n_cold = ddmt.get_output_nsamps(p.nsamps);
    if (p.nbits == 32) {
        const auto host = make_float_input(kNchans * p.nsamps);
        const thrust::device_vector<float> wf(host.begin(), host.end());
        thrust::device_vector<float> dmt(ndms * p.nsamps);
        ddmt.execute(dspan_c(wf),
                     DeviceSpan<float>(dspan(dmt).data(), ndms * n_cold));
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            ddmt.execute(dspan_c(wf), dspan(dmt));
        }
    } else {
        const auto host = make_packed_input(packed_bytes(p.nsamps, p.nbits));
        const thrust::device_vector<uint8_t> wf(host.begin(), host.end());
        thrust::device_vector<int32_t> dmt(ndms * p.nsamps);
        ddmt.execute(dspan_c(wf), p.nsamps,
                     DeviceSpan<int32_t>(dspan(dmt).data(), ndms * n_cold));
        BENCH_GPU_TRY(cudaDeviceSynchronize());
        for (auto _ : state) {
            const GPUEventTimer timer{state};
            ddmt.execute(dspan_c(wf), p.nsamps, dspan(dmt));
        }
    }
    set_counters(state, plan, p, 0);
}

bool gpu_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void register_cuda() {
    if (!gpu_device_available()) {
        return;
    }
    const auto add = [](const std::string& name, auto fn, Point p) {
        benchmark::RegisterBenchmark(name,
                                     [fn, p](benchmark::State& s) { fn(s, p); })
            ->UseManualTime()
            ->Unit(benchmark::kMillisecond);
    };
    for (const auto sweep : {Sweep::kNsamps, Sweep::kNdms}) {
        for (const auto& p : sweep_points(sweep)) {
            add(bench_name(Algo::kFDMT, DMT_GPU_NAME, p), bench_fdmt_cuda, p);
            add(bench_name(Algo::kFDMTFFT, DMT_GPU_NAME, p),
                bench_fdmt_fft_cuda, p);
            add(bench_name(Algo::kDDMT, DMT_GPU_NAME, p), bench_ddmt_cuda, p);
        }
    }
    for (const auto& p : sweep_points(Sweep::kNbits)) {
        add(bench_name(Algo::kFDMT, DMT_GPU_NAME, p), bench_fdmt_cuda, p);
        add(bench_name(Algo::kFDMT, DMT_GPU_NAME "_host", p),
            bench_fdmt_cuda_host, p);
        add(bench_name(Algo::kDDMT, DMT_GPU_NAME, p), bench_ddmt_cuda, p);
    }
}

[[maybe_unused]] const bool kRegistered = [] {
    register_cuda();
    return true;
}();

} // namespace
} // namespace dmt::bench_suite
