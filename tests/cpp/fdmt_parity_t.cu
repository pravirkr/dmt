#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cstdint>
#include <cuda/std/span>
#include <span>
#include <thrust/device_vector.h>
#include <vector>

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::CohFDMTCPU;
using algorithms::CohFDMTCUDA;
using algorithms::FDMTCPU;
using algorithms::FDMTCUDA;
using algorithms::FDMTFFTCPU;
using algorithms::FDMTFFTCUDA;

TEST_CASE("parity: FDMTCUDA execute and stepper match FDMTCPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    auto waterfall        = test::sequential_waterfall(nchans, nsamps);
    FDMTCPU cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    FDMTCUDA gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    const auto n = cpu.get_plan().get_buffer_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_gpu(n, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);
    test::require_approx(dmt_gpu, dmt_cpu, cpu.get_plan().get_dmt_size());

    std::vector<float> dmt_step(n, 0.0F);
    thrust::device_vector<float> d_wf(waterfall.begin(), waterfall.end());
    thrust::device_vector<float> d_dmt(n, 0.0F);
    auto d_wf_span = cuda::std::span<const float>(
        thrust::raw_pointer_cast(d_wf.data()), d_wf.size());
    auto d_dmt_span = cuda::std::span<float>(
        thrust::raw_pointer_cast(d_dmt.data()), d_dmt.size());

    gpu.reset_history();
    gpu.reset(d_wf_span, d_dmt_span);
    gpu.advance_until_remaining(0);
    gpu.finalize();
    thrust::copy(d_dmt.begin(), d_dmt.end(), dmt_step.begin());
    test::require_approx(dmt_step, dmt_cpu, cpu.get_plan().get_dmt_size());
}

TEST_CASE("parity: FDMTFFTCUDA execute matches FDMTFFTCPU",
          "[fdmt_fft][gpu][parity]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    auto waterfall        = test::sequential_waterfall(nchans, nsamps, 11, 1);
    FDMTFFTCPU cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 8, 0,
                   1, true, "roll");
    FDMTFFTCUDA gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 8,
                    0, 1, true, "roll");
    const auto n = cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_gpu(n, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);
    test::require_approx(dmt_gpu, dmt_cpu, 0.05);
}

TEST_CASE("parity: CohFDMTCUDA execute matches CohFDMTCPU",
          "[cfdmt][gpu][parity]") {
    plans::CohFDMTPlan plan(1250.0F, 25.0F, 4, 1.0E-6F, 1 << 10, 2, 4.0E-6F,
                            5.0F, 0.0F, 32, "PRITF", false);
    CohFDMTCPU cpu(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                   plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                   plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                   plan.get_noverlap());
    CohFDMTCUDA gpu(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                    plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                    plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                    plan.get_noverlap());
    const SizeType in_size =
        SizeType{2} * SizeType{2} * plan.get_nsamp() * plan.get_nsub();
    std::vector<uint8_t> data_in(in_size);
    for (SizeType i = 0; i < in_size; ++i) {
        data_in[i] = static_cast<uint8_t>((i * 13) % 251);
    }
    std::vector<float> dmt_cpu(cpu.get_dmt_size(), 0.0F);
    std::vector<float> dmt_gpu(gpu.get_dmt_size(), 0.0F);
    cpu.execute<uint8_t>(data_in, dmt_cpu);
    gpu.execute<uint8_t>(data_in, dmt_gpu);
    REQUIRE_THAT(
        dmt_gpu,
        Catch::Matchers::Approx(dmt_cpu).epsilon(1.0E-2).margin(1.0E-2));
}

TEST_CASE("parity: FDMTCUDA add_frb_track recovery matches FDMTCPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 32;
    const SizeType nsamps = 128;
    FDMTCPU cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16,
                -16);
    FDMTCUDA gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16,
                 -16);
    const auto& plan = cpu.get_plan();
    std::vector<float> waterfall(nchans * nsamps, 0.0F);
    algorithms::add_frb_track(waterfall, plan, 8, 1.0F, 40, 1);
    const auto n = plan.get_buffer_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_gpu(n, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);
    test::require_approx(dmt_gpu, dmt_cpu, plan.get_dmt_size());
    const auto ns = plan.get_dmt_nsamps();
    CHECK(dmt_cpu[(8 * ns) + 40] == Catch::Approx(static_cast<float>(nchans)));
    CHECK(dmt_gpu[(8 * ns) + 40] == Catch::Approx(static_cast<float>(nchans)));
}

TEST_CASE("parity: FDMTCUDA valid-mode second block matches FDMTCPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 32;
    FDMTCPU cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    FDMTCUDA gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    const auto n = cpu.get_plan().get_buffer_size();
    auto block1  = test::sequential_waterfall(nchans, nsamps, 17, 1);
    auto block2  = test::sequential_waterfall(nchans, nsamps, 13, 3);
    std::vector<float> cpu1(n, 0.0F);
    std::vector<float> gpu1(n, 0.0F);
    std::vector<float> cpu2(n, 0.0F);
    std::vector<float> gpu2(n, 0.0F);
    cpu.execute(block1, cpu1);
    gpu.execute(block1, gpu1);
    cpu.execute(block2, cpu2);
    gpu.execute(block2, gpu2);
    test::require_approx(gpu1, cpu1, cpu.get_plan().get_dmt_size());
    test::require_approx(gpu2, cpu2, cpu.get_plan().get_dmt_size());
}

} // namespace dmt
