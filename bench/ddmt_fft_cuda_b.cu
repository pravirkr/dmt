#include <span>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include <benchmark/benchmark.h>

#include "bench_gpu_utils.cuh"
#include "dmt/algorithms/ddmt_fft.hpp"

namespace dmt {
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

// Device-resident DDMT-FFT (range(1): 0 = NUFFT, 1 = brute force); 1024
// channels over 1000-1500 MHz, DM 0-50 in steps of 0.5.
void BM_ddmt_fft_gpu(benchmark::State& state) {
    const auto nsamps = static_cast<SizeType>(state.range(0));
    const auto method =
        state.range(1) == 0 ? DDMTFFTMethod::kNUFFT : DDMTFFTMethod::kBrute;
    DDMTFFT e(1000.0F, 1500.0F, 1024, 6.4e-5F, 50.0F, 0.5F, 0.0F,
              bench_gpu_exec(), 32, {}, 1, DDMTFFTOptions{.method = method});
    const auto ndm = e.get_plan().get_dm_arr().size();
    const thrust::device_vector<float> wf(1024 * nsamps, 1.0F);
    thrust::device_vector<float> out(ndm * nsamps);
    const auto cold = e.get_output_nsamps(nsamps);
    e.execute(DeviceSpan<const float>(thrust::raw_pointer_cast(wf.data()),
                                      wf.size()),
              DeviceSpan<float>(thrust::raw_pointer_cast(out.data()),
                                ndm * cold));
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    for (auto _ : state) {
        const GPUEventTimer timer{state};
        e.execute(DeviceSpan<const float>(thrust::raw_pointer_cast(wf.data()),
                                          wf.size()),
                  DeviceSpan<float>(thrust::raw_pointer_cast(out.data()),
                                    out.size()));
    }
}

BENCHMARK(BM_ddmt_fft_gpu) // NOLINT
    ->ArgsProduct({{8192, 32768}, {0, 1}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

} // namespace dmt
