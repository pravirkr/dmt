#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/fft.hpp"
#include "dmt/unpacker.hpp"

#include <omp.h>

// CohFDMT CPU benchmarks. Reference search: one GUPPI node (64 x 2.9296875
// MHz at 1312.5-1500 MHz, int8 FTPRI), t_p = 10 us, DM 50-60 pc/cc (one
// coarse trial) or 50-250 (several). Args: {nthreads, t_p in us, dt_step,
// DM width}. Stage benchmarks time the pieces of one
// execute(): unpack, forward FFT, fine FDMT (per coarse trial) and the
// FFTW roofline of the per-trial coherent stage (the inverse FFTs alone).

namespace dmt {
using algorithms::CohFDMT;

namespace {

CohFDMTConfig guppi_config(const benchmark::State& state) {
    return {.f_center = 1406.25F,
            .bw_sub   = 1500.0F / 512.0F,
            .nsub     = 64,
            .t_p      = static_cast<float>(state.range(1)) * 1.0E-6F,
            .dm_min   = 50.0F,
            .dm_max   = 50.0F + static_cast<float>(state.range(3)),
            .dt_step  = static_cast<SizeType>(state.range(2)),
            .format   = BasebandFormat{.order = "FTPRI"}};
}

std::vector<uint8_t> random_block(SizeType n) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0F, 16.0F);
    std::vector<uint8_t> data(n);
    for (auto& x : data) {
        x = static_cast<uint8_t>(
            static_cast<int8_t>(std::clamp(dist(gen), -127.0F, 127.0F)));
    }
    return data;
}

void set_counters(benchmark::State& state,
                  const plans::CohFDMTPlan& plan,
                  double work_per_iter = 1.0) {
    const double data_s = static_cast<double>(plan.get_stride_nsamps()) *
                          plan.get_tbin() * work_per_iter;
    state.counters["rtf"] = benchmark::Counter(
        data_s, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["ndm"]     = static_cast<double>(plan.get_ndm());
    state.counters["ndm_coh"] = static_cast<double>(plan.get_ndm_coh());
    state.counters["nout"]    = static_cast<double>(plan.get_output_nsamps());
}

} // namespace

// End-to-end execute(); "rtf" = seconds of data per second (stride worth).
static void BM_cfdmt_execute(benchmark::State& state) {
    const CohFDMT search(guppi_config(state),
                         Exec::cpu(static_cast<int>(state.range(0))));
    const auto in = random_block(search.get_input_size());
    std::vector<float> dmt(search.get_dmt_size());
    search.execute<uint8_t>(std::span<const uint8_t>(in), dmt); // warm-up
    for (auto _ : state) {
        search.execute<uint8_t>(std::span<const uint8_t>(in), dmt);
        benchmark::DoNotOptimize(dmt.data());
    }
    set_counters(state, search.get_plan());
}

static void BM_cfdmt_stage_unpack(benchmark::State& state) {
    const plans::CohFDMTPlan plan(guppi_config(state));
    const utils::BasebandUnpackerCPU unpacker(
        plan.get_format(), plan.get_subband_groups(), plan.get_nbin(),
        plan.get_nfft(), plan.get_noverlap(),
        static_cast<int>(state.range(0)));
    const auto in = random_block(unpacker.input_size(0));
    utils::FFTVector<ComplexType> out(unpacker.output_size());
    const std::span<const uint8_t> group(in);
    for (auto _ : state) {
        unpacker.execute(std::span(&group, 1), out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(in.size()));
}

static void BM_cfdmt_stage_forward_fft(benchmark::State& state) {
    const plans::CohFDMTPlan plan(guppi_config(state));
    const utils::FFTWManager fwd(
        utils::FFTKind::kC2CForward, plan.get_nbin(),
        SizeType{2} * plan.get_nfft() * plan.get_nsub(),
        static_cast<int>(state.range(0)));
    utils::FFTVector<ComplexType> data(SizeType{2} * plan.get_nfft() *
                                           plan.get_nsub() * plan.get_nbin(),
                                       ComplexType{1.0F, 0.5F});
    for (auto _ : state) {
        fwd.execute(data);
        benchmark::DoNotOptimize(data.data());
    }
}

// Fine FDMT of one coarse trial (reset + execute), times ndm_coh per block.
static void BM_cfdmt_stage_fdmt_per_trial(benchmark::State& state) {
    const plans::CohFDMTPlan plan(guppi_config(state));
    algorithms::FDMT fdmt(plan.get_f_min(), plan.get_f_max(),
                          plan.get_nchans(), plan.get_fdmt_nsamps(),
                          plan.get_tsamp(),
                          static_cast<IndexType>(plan.get_fine_dt_max()),
                          -static_cast<IndexType>(plan.get_fine_dt_max()),
                          plan.get_config().dt_step, true, "valid",
                          Exec::cpu(static_cast<int>(state.range(0))));
    std::vector<float> wf(plan.get_nchans() * plan.get_fdmt_nsamps(), 0.5F);
    std::vector<float> out(fdmt.get_plan().get_buffer_size());
    for (auto _ : state) {
        fdmt.reset_history();
        fdmt.execute(wf, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.counters["ndm_coh"] = static_cast<double>(plan.get_ndm_coh());
}

// Roofline of one coarse trial's coherent stage: only the per-channel
// inverse FFTs (both pols) over each channel's window, on cached data.
static void BM_cfdmt_stage_inverse_fft_roofline(benchmark::State& state) {
    const plans::CohFDMTPlan plan(guppi_config(state));
    const auto mbin  = plan.get_mbin();
    const auto novc  = plan.get_noverlap() / plan.get_n_p();
    const auto lc    = mbin - (2 * novc);
    const auto nblk  = (plan.get_fdmt_nsamps() + lc - 1) / lc + 1;
    const auto rows  = plan.get_nchans() * nblk * 2;
    const auto nthr  = static_cast<int>(state.range(0));
    const auto batch = SizeType{64};
    const utils::FFTWRowPlan inv(utils::FFTKind::kC2CBackward, mbin, batch);
    utils::FFTVector<ComplexType> data(static_cast<SizeType>(nthr) * batch *
                                           mbin,
                                       ComplexType{1.0F, 0.5F});
    const auto nbatch = (rows + batch - 1) / batch;
    for (auto _ : state) {
#pragma omp parallel for num_threads(nthr) schedule(static)
        for (SizeType b = 0; b < nbatch; ++b) {
            const auto tid = static_cast<SizeType>(omp_get_thread_num());
            inv.c2c(data.data() + (tid * batch * mbin));
        }
        benchmark::DoNotOptimize(data.data());
    }
    state.counters["ndm_coh"] = static_cast<double>(plan.get_ndm_coh());
    state.counters["mbin"]    = static_cast<double>(mbin);
}

// {nthreads, t_p [us], dt_step, DM width [pc/cc] from DM 50}
#define CFDMT_ARGS                                                             \
    ArgsProduct({{1, 8}, {10}, {16}, {10, 200}})                               \
        ->Unit(benchmark::kMillisecond)                                        \
        ->UseRealTime()

BENCHMARK(BM_cfdmt_execute)->CFDMT_ARGS;
BENCHMARK(BM_cfdmt_stage_unpack)->CFDMT_ARGS;
BENCHMARK(BM_cfdmt_stage_forward_fft)->CFDMT_ARGS;
BENCHMARK(BM_cfdmt_stage_fdmt_per_trial)->CFDMT_ARGS;
BENCHMARK(BM_cfdmt_stage_inverse_fft_roofline)->CFDMT_ARGS;

} // namespace dmt
