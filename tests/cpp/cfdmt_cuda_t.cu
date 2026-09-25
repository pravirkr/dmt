#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <ranges>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cuda/std/span>
#include <thrust/device_vector.h>

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/common/plans.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::CohFDMTCPU;
using algorithms::CohFDMTCUDA;
using plans::CohFDMTPlan;

namespace {
CohFDMTPlan make_cuda_test_plan() {
    const SizeType chan_per_sub = 4;
    const float f_center        = 1250.0F;
    const float bw_sub          = 25.0F;
    const SizeType nsub         = 4;
    const float tbin            = 1.0E-6F;
    const SizeType nbin         = 1 << 10;
    const SizeType nfft         = 2;
    const float t_p             = tbin * static_cast<float>(chan_per_sub);
    const float dm_max          = 5.0F;
    const float dm_min          = 0.0F;
    return {f_center, bw_sub, nsub,   tbin, nbin,    nfft,
            t_p,      dm_max, dm_min, 32,   "PRITF", false};
}
} // namespace

TEST_CASE("CohFDMTCUDA Constructor and Plan Invariants", "[cfdmt_gpu][gpu]") {

    const auto plan = make_cuda_test_plan();
    CohFDMTCUDA coh_fdmt_cuda(
        plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(), plan.get_t_p(),
        plan.get_dm_max(), plan.get_dm_min(), plan.get_noverlap());

    CHECK(coh_fdmt_cuda.get_dmt_size() == plan.get_dmt_size());
    CHECK(coh_fdmt_cuda.get_plan().get_ndm() == plan.get_ndm());
    CHECK(coh_fdmt_cuda.get_plan().get_dmt_nsamps() == plan.get_dmt_nsamps());
}

TEST_CASE("CohFDMTCUDA Host and Device Execution Parity",
          "[cfdmt_gpu][gpu][parity]") {

    const auto plan = make_cuda_test_plan();

    CohFDMTCPU coh_fdmt_cpu(plan.get_f_center(), plan.get_bw_sub(),
                            plan.get_nsub(), plan.get_tbin(), plan.get_nbin(),
                            plan.get_nfft(), plan.get_t_p(), plan.get_dm_max(),
                            plan.get_dm_min(), plan.get_noverlap());

    CohFDMTCUDA coh_fdmt_cuda(
        plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(), plan.get_t_p(),
        plan.get_dm_max(), plan.get_dm_min(), plan.get_noverlap());

    const SizeType in_size =
        SizeType{2} * SizeType{2} * plan.get_nsamp() * plan.get_nsub();
    std::vector<uint8_t> data_in_h(in_size);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data_in_h) {
        v = static_cast<uint8_t>(dist(rng));
    }

    const SizeType dmt_size = plan.get_dmt_size();
    std::vector<float> dmt_cpu(dmt_size, 0.0F);
    std::vector<float> dmt_cuda_h(dmt_size, 0.0F);

    // CPU execution
    coh_fdmt_cpu.execute<uint8_t>(data_in_h, dmt_cpu);

    // CUDA execution on host buffers
    coh_fdmt_cuda.execute<uint8_t>(data_in_h, dmt_cuda_h);

    // Device execution via cuda::std::span
    thrust::device_vector<uint8_t> data_in_d = data_in_h;
    thrust::device_vector<float> dmt_d(dmt_size, 0.0F);

    coh_fdmt_cuda.reset_history();
    coh_fdmt_cuda.execute<uint8_t>(
        cuda::std::span<const uint8_t>(
            thrust::raw_pointer_cast(data_in_d.data()), data_in_d.size()),
        cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                               dmt_d.size()));

    std::vector<float> dmt_cuda_dev(dmt_size, 0.0F);
    thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_cuda_dev.begin());

    test::require_approx(dmt_cuda_h, dmt_cuda_dev, 1.0E-3);
    REQUIRE_THAT(
        dmt_cuda_dev,
        Catch::Matchers::Approx(dmt_cpu).epsilon(1.0E-2).margin(1.0E-2));
}

TEST_CASE("CohFDMTCUDA Multi-block Streaming and History Reset",
          "[cfdmt_gpu][gpu]") {

    const auto plan = make_cuda_test_plan();

    CohFDMTCUDA coh_fdmt(plan.get_f_center(), plan.get_bw_sub(),
                         plan.get_nsub(), plan.get_tbin(), plan.get_nbin(),
                         plan.get_nfft(), plan.get_t_p(), plan.get_dm_max(),
                         plan.get_dm_min(), plan.get_noverlap());

    const SizeType in_size =
        SizeType{2} * SizeType{2} * plan.get_nsamp() * plan.get_nsub();
    std::vector<uint8_t> block1(in_size);
    std::vector<uint8_t> block2(in_size);

    std::mt19937 rng(54321);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : block1) {
        v = static_cast<uint8_t>(dist(rng));
    }
    for (auto& v : block2) {
        v = static_cast<uint8_t>(dist(rng));
    }

    std::vector<float> dmt_b1_initial(coh_fdmt.get_dmt_size(), 0.0F);
    std::vector<float> dmt_b2_streamed(coh_fdmt.get_dmt_size(), 0.0F);
    std::vector<float> dmt_b2_cold(coh_fdmt.get_dmt_size(), 0.0F);
    std::vector<float> dmt_b1_repeated(coh_fdmt.get_dmt_size(), 0.0F);

    // 1. Process block 1 cold
    coh_fdmt.execute<uint8_t>(block1, dmt_b1_initial);

    // 2. Process block 2 continuous (warm history)
    coh_fdmt.execute<uint8_t>(block2, dmt_b2_streamed);

    // 3. Reset history and process block 2 cold
    coh_fdmt.reset_history();
    coh_fdmt.execute<uint8_t>(block2, dmt_b2_cold);

    CHECK_FALSE(std::equal(
        dmt_b2_streamed.begin(), dmt_b2_streamed.end(), dmt_b2_cold.begin(),
        [](float a, float b) { return std::abs(a - b) <= 1.0E-4F; }));

    // 4. Reset history and re-process block 1
    coh_fdmt.reset_history();
    coh_fdmt.execute<uint8_t>(block1, dmt_b1_repeated);
    test::require_approx(dmt_b1_initial, dmt_b1_repeated, 1.0E-5);
}

TEST_CASE("CohFDMTCUDA synthetic impulse peaks near DM 0", "[cfdmt_gpu][gpu]") {
    const auto plan = make_cuda_test_plan();
    CohFDMTCUDA coh_fdmt(plan.get_f_center(), plan.get_bw_sub(),
                         plan.get_nsub(), plan.get_tbin(), plan.get_nbin(),
                         plan.get_nfft(), plan.get_t_p(), plan.get_dm_max(),
                         plan.get_dm_min(), plan.get_noverlap());
    CohFDMTCPU coh_cpu(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                       plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                       plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                       plan.get_noverlap());

    const SizeType nsamp   = plan.get_nsamp();
    const SizeType nsub    = plan.get_nsub();
    const SizeType in_size = SizeType{2} * SizeType{2} * nsamp * nsub;
    std::vector<uint8_t> data_in(in_size, 0);
    const SizeType pulse_t = nsamp / 2;
    for (SizeType ipol = 0; ipol < 2; ++ipol) {
        for (SizeType isub = 0; isub < nsub; ++isub) {
            const SizeType idx_re =
                ipol * 2 * nsub * nsamp + isub * nsamp + pulse_t;
            data_in[idx_re] = 127;
        }
    }

    std::vector<float> dmt_cuda(coh_fdmt.get_dmt_size(), 0.0F);
    std::vector<float> dmt_cpu(coh_cpu.get_dmt_size(), 0.0F);
    coh_fdmt.execute<uint8_t>(data_in, dmt_cuda);
    coh_cpu.execute<uint8_t>(data_in, dmt_cpu);
    test::require_approx(dmt_cuda, dmt_cpu, 1.0E-2);

    const auto& dm_grid = plan.get_dm_grid_final();
    const auto peak     = static_cast<SizeType>(
        std::distance(dmt_cuda.begin(), std::ranges::max_element(dmt_cuda)));
    const auto peak_dm_idx = peak / plan.get_dmt_nsamps();
    CHECK(std::abs(dm_grid[peak_dm_idx]) <= 1.5F);
    CHECK(dmt_cuda[peak] > 0.0F);
}

} // namespace dmt
