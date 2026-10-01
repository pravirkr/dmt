#include <cstdint>
#include <span>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include <benchmark/benchmark.h>

#include "bench_gpu_utils.cuh"

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/cfdmt_fused_gpu.cuh"
#include "dmt/engines.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/unpacker_cuda.cuh"

// CohFDMT GPU benchmarks: the device twin of cfdmt_cpu_b.cpp. Reference
// search: one GUPPI node (64 x 2.9296875 MHz at 1312.5-1500 MHz, int8
// FTPRI), t_p = 10 us, dt_step 16, DM 50-60 (one coarse trial) or 50-250
// (four). All device-resident, timed with events.
//   execute: end to end, fused (path 0) or cuFFT (path 1) coherent stage;
//   stage_*: unpack, forward FFT (whole block or L2-sized chunks as the
//     engine runs it), the fused coherent kernel of one trial against the
//     device copy bandwidth ("roofline" = its achieved bytes/s over a
//     device-to-device copy's), and the fine FDMT of one trial.

namespace dmt {
namespace {

using algorithms::detail::CohFDMTCoherentPath;

CohFDMTConfig guppi_config(SizeType dm_width) {
    return {.f_center = 1406.25F,
            .bw_sub   = 1500.0F / 512.0F,
            .nsub     = 64,
            .t_p      = 10.0E-6F,
            .dm_min   = 50.0F,
            .dm_max   = 50.0F + static_cast<float>(dm_width),
            .dt_step  = 16,
            .format   = BasebandFormat{.order = "FTPRI"}};
}

// Times body() with events on the legacy default stream.
template <typename Body>
void run_timed(benchmark::State& state, const Body& body) {
    body(); // warm-up
    BENCH_GPU_TRY(cudaDeviceSynchronize());
    cudaEvent_t start{};
    cudaEvent_t stop{};
    BENCH_GPU_TRY(cudaEventCreate(&start));
    BENCH_GPU_TRY(cudaEventCreate(&stop));
    for (auto _ : state) {
        BENCH_GPU_TRY(cudaEventRecord(start));
        body();
        BENCH_GPU_TRY(cudaEventRecord(stop));
        BENCH_GPU_TRY(cudaEventSynchronize(stop));
        float ms = 0.0F;
        BENCH_GPU_TRY(cudaEventElapsedTime(&ms, start, stop));
        state.SetIterationTime(static_cast<double>(ms) / 1000.0);
    }
    BENCH_GPU_TRY(cudaEventDestroy(start));
    BENCH_GPU_TRY(cudaEventDestroy(stop));
}

void set_counters(benchmark::State& state, const plans::CohFDMTPlan& plan) {
    const double data_s =
        static_cast<double>(plan.get_stride_nsamps()) * plan.get_tbin();
    state.counters["rtf"] = benchmark::Counter(
        data_s, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["ndm"]     = static_cast<double>(plan.get_ndm());
    state.counters["ndm_coh"] = static_cast<double>(plan.get_ndm_coh());
    state.counters["mbin"]    = static_cast<double>(plan.get_mbin());
}

SizeType log2_of(SizeType n) {
    SizeType l = 0;
    while ((SizeType{1} << l) < n) {
        ++l;
    }
    return l;
}

// Args: {path (0 fused, 1 cuFFT), DM width}.
void BM_cfdmt_cuda_execute(benchmark::State& state) {
    const auto cfg = guppi_config(static_cast<SizeType>(state.range(1)));
    const plans::CohFDMTPlan plan(cfg);
    const auto engine = algorithms::detail::make_cfdmt_gpu(
        plan,
        {.exec          = bench_gpu_exec(),
         .coherent_path = state.range(0) == 0 ? CohFDMTCoherentPath::kFused
                                              : CohFDMTCoherentPath::kCuFFT});
    const thrust::device_vector<uint8_t> d_in(plan.get_input_size(0), 3);
    thrust::device_vector<float> d_out(plan.get_dmt_size());
    const DeviceSpan<const uint8_t> din(thrust::raw_pointer_cast(d_in.data()),
                                        d_in.size());
    const DeviceSpan<float> dout(thrust::raw_pointer_cast(d_out.data()),
                                 d_out.size());
    run_timed(state,
              [&] { engine->execute(std::span(&din, 1), dout, Stream{}); });
    set_counters(state, plan);
}

void BM_cfdmt_cuda_stage_unpack(benchmark::State& state) {
    const plans::CohFDMTPlan plan(
        guppi_config(static_cast<SizeType>(state.range(1))));
    utils::BasebandUnpackerCUDA unpacker(
        plan.get_format(), plan.get_subband_groups(), plan.get_nbin(),
        plan.get_nfft(), plan.get_noverlap());
    const thrust::device_vector<uint8_t> d_in(unpacker.input_size(0), 3);
    thrust::device_vector<ComplexTypeGPU> d_out(unpacker.output_size());
    const cuda::std::span<const uint8_t> group(
        thrust::raw_pointer_cast(d_in.data()), d_in.size());
    const cuda::std::span<ComplexTypeGPU> out(
        thrust::raw_pointer_cast(d_out.data()), d_out.size());
    run_timed(state, [&] { unpacker.execute(std::span(&group, 1), out); });
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(d_in.size()));
}

// Arg 0: subbands per forward-FFT call (0 = the whole block at once).
void BM_cfdmt_cuda_stage_forward_fft(benchmark::State& state) {
    const plans::CohFDMTPlan plan(
        guppi_config(static_cast<SizeType>(state.range(1))));
    const auto nsub = plan.get_nsub();
    const auto per =
        state.range(0) == 0 ? nsub : static_cast<SizeType>(state.range(0));
    const auto rows = SizeType{2} * plan.get_nfft() * plan.get_nbin();
    const utils::CUFFTManager fwd(utils::FFTKind::kC2CForward, plan.get_nbin(),
                                  per * 2 * plan.get_nfft(), 0);
    thrust::device_vector<ComplexTypeGPU> data(nsub * rows);
    const cuda::std::span<ComplexTypeGPU> all(
        thrust::raw_pointer_cast(data.data()), data.size());
    run_timed(state, [&] {
        for (SizeType s0 = 0; s0 + per <= nsub; s0 += per) {
            fwd.execute(all.subspan(s0 * rows, per * rows));
        }
    });
    state.counters["chunk_mib"] =
        static_cast<double>(per * rows * sizeof(ComplexTypeGPU)) /
        (1024.0 * 1024.0);
}

// The fused coherent kernel of one trial, on synthetic buffers, against a
// device-to-device copy of the same traffic.
void BM_cfdmt_cuda_stage_coherent_fused(benchmark::State& state) {
    const plans::CohFDMTPlan plan(
        guppi_config(static_cast<SizeType>(state.range(1))));
    const auto nchans = plan.get_nchans();
    const auto mbin   = plan.get_mbin();
    const auto n_p    = plan.get_n_p();
    const auto novc   = plan.get_noverlap() / n_p;
    const auto lc     = mbin - (2 * novc);
    const auto nf     = plan.get_fdmt_nsamps();
    const auto nwin   = ((nf + lc - 1) / lc) + 1;
    const auto log2n  = static_cast<int>(log2_of(mbin));
    if (log2n != 10 && log2n != 8 && log2n != 9 && log2n != 11) {
        state.SkipWithError("benchmark instantiates mbin 256-2048 only");
        return;
    }
    thrust::device_vector<ComplexTypeGPU> spec(
        SizeType{2} * plan.get_nsub() * plan.get_nfft() * plan.get_nbin(),
        ComplexTypeGPU{1.0F, 0.5F});
    thrust::device_vector<ComplexTypeGPU> chirp(nchans * mbin,
                                                ComplexTypeGPU{0.5F, 0.5F});
    thrust::device_vector<ComplexTypeGPU> tw(mbin / 2,
                                             ComplexTypeGPU{1.0F, 0.0F});
    const auto sh = plan.get_channel_shifts(0);
    const thrust::device_vector<int64_t> shifts(sh.begin(), sh.end());
    const thrust::device_vector<float> mean(nchans, 0.0F);
    const thrust::device_vector<float> inv(nchans, 1.0F);
    thrust::device_vector<float> wf(nchans * nf);

    const auto f2 = [](const thrust::device_vector<ComplexTypeGPU>& v) {
        return reinterpret_cast<const float2*>(
            thrust::raw_pointer_cast(v.data()));
    };
    cfdmt_gpu::FusedArgs a{};
    a.spec      = f2(spec);
    a.chirp     = f2(chirp);
    a.twiddle   = f2(tw);
    a.shifts    = thrust::raw_pointer_cast(shifts.data());
    a.mean      = thrust::raw_pointer_cast(mean.data());
    a.inv_sigma = thrust::raw_pointer_cast(inv.data());
    a.waterfall = thrust::raw_pointer_cast(wf.data());
    a.nchans    = nchans;
    a.n_p       = n_p;
    a.nfft      = plan.get_nfft();
    a.nbin      = plan.get_nbin();
    a.novc      = novc;
    a.lc        = static_cast<int64_t>(lc);
    a.a         = plan.get_fdmt_window_start();
    a.nf        = static_cast<int64_t>(nf);
    a.msamp     = static_cast<int64_t>(plan.get_msamp());
    a.norm      = true;
    const auto round =
        static_cast<uint64_t>(cfdmt_gpu::fused_round_blocks(log2n));
    a.blocks_per_cta = ((8 + round - 1) / round) * round;
    a.ngroups        = (nwin + a.blocks_per_cta - 1) / a.blocks_per_cta;
    const auto smem  = cfdmt_gpu::fused_smem_bytes(log2n);
    const auto grid  = static_cast<unsigned>(nchans * a.ngroups);
    void (*kernel)(cfdmt_gpu::FusedArgs) =
        log2n == 8    ? &cfdmt_gpu::fused_coherent_kernel<8>
        : log2n == 9  ? &cfdmt_gpu::fused_coherent_kernel<9>
        : log2n == 10 ? &cfdmt_gpu::fused_coherent_kernel<10>
                      : &cfdmt_gpu::fused_coherent_kernel<11>;
    BENCH_GPU_TRY(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    run_timed(state,
              [&] { kernel<<<grid, cfdmt_gpu::kFusedThreads, smem>>>(a); });
    // Traffic: each channel's bins of its window blocks (both pols) in,
    // the aligned window out.
    const double bytes =
        (static_cast<double>(nchans) * static_cast<double>(nwin - 1) *
         static_cast<double>(mbin) * 2.0 * sizeof(ComplexTypeGPU)) +
        (static_cast<double>(nchans) * static_cast<double>(nf) * sizeof(float));
    state.counters["bytes_per_s"] = benchmark::Counter(
        bytes, benchmark::Counter::kIsIterationInvariantRate);

    // Roofline: a device-to-device copy of the same number of bytes (half
    // read, half written).
    thrust::device_vector<uint8_t> src(static_cast<SizeType>(bytes / 2));
    thrust::device_vector<uint8_t> dst(src.size());
    const auto seconds = [](const auto& body) {
        constexpr int kReps = 20;
        cudaEvent_t e0{};
        cudaEvent_t e1{};
        BENCH_GPU_TRY(cudaEventCreate(&e0));
        BENCH_GPU_TRY(cudaEventCreate(&e1));
        BENCH_GPU_TRY(cudaEventRecord(e0));
        for (int r = 0; r < kReps; ++r) {
            body();
        }
        BENCH_GPU_TRY(cudaEventRecord(e1));
        BENCH_GPU_TRY(cudaEventSynchronize(e1));
        float ms = 0.0F;
        BENCH_GPU_TRY(cudaEventElapsedTime(&ms, e0, e1));
        BENCH_GPU_TRY(cudaEventDestroy(e0));
        BENCH_GPU_TRY(cudaEventDestroy(e1));
        return static_cast<double>(ms) * 1.0E-3 / kReps;
    };
    const double t_copy = seconds([&] {
        BENCH_GPU_TRY(cudaMemcpyAsync(thrust::raw_pointer_cast(dst.data()),
                                      thrust::raw_pointer_cast(src.data()),
                                      src.size(), cudaMemcpyDeviceToDevice));
    });
    const double t_kernel =
        seconds([&] { kernel<<<grid, cfdmt_gpu::kFusedThreads, smem>>>(a); });
    state.counters["copy_bytes_per_s"] = bytes / t_copy;
    state.counters["roofline"]         = t_copy / t_kernel;
    state.counters["mbin"]             = static_cast<double>(mbin);
    state.counters["nchans"]           = static_cast<double>(nchans);
}

void BM_cfdmt_cuda_stage_fdmt_per_trial(benchmark::State& state) {
    const plans::CohFDMTPlan plan(
        guppi_config(static_cast<SizeType>(state.range(1))));
    algorithms::FDMT fdmt(plan.get_f_min(), plan.get_f_max(), plan.get_nchans(),
                          plan.get_fdmt_nsamps(), plan.get_tsamp(),
                          static_cast<IndexType>(plan.get_fine_dt_max()),
                          -static_cast<IndexType>(plan.get_fine_dt_max()),
                          plan.get_config().dt_step, true, "valid",
                          bench_gpu_exec());
    const thrust::device_vector<float> wf(
        plan.get_nchans() * plan.get_fdmt_nsamps(), 0.5F);
    thrust::device_vector<float> out(fdmt.get_plan().get_buffer_size());
    const DeviceSpan<const float> din(thrust::raw_pointer_cast(wf.data()),
                                      wf.size());
    const DeviceSpan<float> dout(thrust::raw_pointer_cast(out.data()),
                                 out.size());
    run_timed(state, [&] { fdmt.execute(din, dout, Stream{}); });
    state.counters["ndm_coh"] = static_cast<double>(plan.get_ndm_coh());
}

} // namespace

// {path or chunk, DM width [pc/cc] from DM 50}
BENCHMARK(BM_cfdmt_cuda_execute)
    ->ArgsProduct({{0, 1}, {10, 200}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cfdmt_cuda_stage_unpack)
    ->ArgsProduct({{0}, {10}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cfdmt_cuda_stage_forward_fft)
    ->ArgsProduct({{0, 2, 8}, {10}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cfdmt_cuda_stage_coherent_fused)
    ->ArgsProduct({{0}, {10, 200}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_cfdmt_cuda_stage_fdmt_per_trial)
    ->ArgsProduct({{0}, {10, 200}})
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

} // namespace dmt
