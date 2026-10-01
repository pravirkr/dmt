#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cstdint>
#include <span>
#include <thrust/device_vector.h>
#include <vector>
#include "dmt/gpu_compat.cuh"

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::CohFDMT;
using algorithms::FDMT;
using algorithms::FDMTFFT;

TEST_CASE("parity: FDMT (gpu) execute and stepper match CPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 64;
    auto waterfall        = test::sequential_waterfall(nchans, nsamps);
    FDMT cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    FDMT gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16, 0, 1,
             true, "valid", test::gpu_exec());
    const auto n = cpu.get_plan().get_buffer_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_gpu(n, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);
    test::require_approx(dmt_gpu, dmt_cpu, cpu.get_plan().get_dmt_size());

    std::vector<float> dmt_step(n, 0.0F);
    thrust::device_vector<float> d_wf(waterfall.begin(), waterfall.end());
    thrust::device_vector<float> d_dmt(n, 0.0F);
    auto d_wf_span = DeviceSpan<const float>(
        thrust::raw_pointer_cast(d_wf.data()), d_wf.size());
    auto d_dmt_span =
        DeviceSpan<float>(thrust::raw_pointer_cast(d_dmt.data()), d_dmt.size());

    gpu.reset_history();
    gpu.reset(d_wf_span, d_dmt_span);
    gpu.advance_until_remaining(0);
    gpu.finalize();
    thrust::copy(d_dmt.begin(), d_dmt.end(), dmt_step.begin());
    test::require_approx(dmt_step, dmt_cpu, cpu.get_plan().get_dmt_size());
}

TEST_CASE("parity: FDMTFFT (gpu) execute matches CPU",
          "[fdmt_fft][gpu][parity]") {
    const SizeType nchans = 8;
    const SizeType nsamps = 64;
    auto waterfall        = test::sequential_waterfall(nchans, nsamps, 11, 1);
    FDMTFFT cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 8, 0, 1,
                true, "roll");
    FDMTFFT gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 8, 0, 1,
                true, "roll", test::gpu_exec());
    const auto n = cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_gpu(n, 0.0F);
    cpu.execute(waterfall, dmt_cpu);
    gpu.execute(waterfall, dmt_gpu);
    test::require_approx(dmt_gpu, dmt_cpu, 0.05);
}

TEST_CASE("parity: CohFDMT (gpu) execute matches CPU", "[cfdmt][gpu][parity]") {
    const CohFDMTConfig cfg{.f_center = 1250.0F,
                            .bw_sub   = 4.0F,
                            .nsub     = 4,
                            .t_p      = 4.0E-6F,
                            .dm_min   = 0.0F,
                            .dm_max   = 5.0F,
                            .format   = BasebandFormat{.order = "PRITF"}};
    const CohFDMT cpu(cfg);
    const CohFDMT gpu(cfg, test::gpu_exec());
    std::vector<uint8_t> data_in(cpu.get_input_size());
    for (SizeType i = 0; i < data_in.size(); ++i) {
        data_in[i] = static_cast<uint8_t>((i * 13) % 251);
    }
    std::vector<float> dmt_cpu(cpu.get_dmt_size(), 0.0F);
    std::vector<float> dmt_gpu(gpu.get_dmt_size(), 0.0F);
    cpu.execute<uint8_t>(data_in, dmt_cpu);
    gpu.execute<uint8_t>(data_in, dmt_gpu);
    REQUIRE_THAT(
        dmt_gpu,
        Catch::Matchers::Approx(dmt_cpu).epsilon(1.0E-4).margin(1.0E-3));
}

TEST_CASE("parity: FDMT (gpu) add_frb_track recovery matches CPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 32;
    const SizeType nsamps = 128;
    FDMT cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16, -16);
    FDMT gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16, -16, 1,
             true, "valid", test::gpu_exec());
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

TEST_CASE("parity: FDMT (gpu) valid-mode second block matches CPU",
          "[fdmt][gpu][parity]") {
    const SizeType nchans = 16;
    const SizeType nsamps = 32;
    FDMT cpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16);
    FDMT gpu(test::kFMin, test::kFMax, nchans, nsamps, test::kTsamp, 16, 0, 1,
             true, "valid", test::gpu_exec());
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
