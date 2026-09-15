#ifdef DMT_ENABLE_CUDA

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cstddef>
#include <cuda/std/span>
#include <random>
#include <span>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTFFTCPU;
using algorithms::FDMTFFTCUDA;

TEST_CASE("FDMTFFTCUDA vs FDMTFFTCPU roll", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(1.0F, 5.0F);
    std::vector<float> waterfall(nchans * nsamps);
    for (auto& v : waterfall) {
        v = dist(rng);
    }

    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "roll");
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "roll");
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_cuda(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_cuda.execute(waterfall, dmt_cuda);
    REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.05));
}

TEST_CASE("FDMTFFTCUDA device execute", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;

    std::vector<float> waterfall(nchans * nsamps, 1.0F);
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "roll");
    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "roll");
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);

    thrust::device_vector<float> wf_d = waterfall;
    thrust::device_vector<float> dmt_d(n, 0.0F);
    fdmt_cuda.execute(
        cuda::std::span<const float>(thrust::raw_pointer_cast(wf_d.data()),
                                     wf_d.size()),
        cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                               dmt_d.size()));
    cudaDeviceSynchronize();
    std::vector<float> dmt_h(n, 0.0F);
    thrust::copy(dmt_d.begin(), dmt_d.end(), dmt_h.begin());
    REQUIRE_THAT(dmt_h, Catch::Matchers::Approx(dmt_cpu).margin(0.05));
}

TEST_CASE("FDMTFFTCUDA multi-beam", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;
    const size_t nbeams = 2;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(1.0F, 5.0F);
    std::vector<float> waterfall(nbeams * nchans * nsamps);
    for (auto& v : waterfall) {
        v = dist(rng);
    }
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "roll", false, 0, nbeams);
    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "roll", false, 1, nbeams);
    const auto n = nbeams * fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_cuda(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_cuda.execute(waterfall, dmt_cuda);
    REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.05));
}

TEST_CASE("FDMTFFTCUDA vs FDMTFFTCPU valid first block", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;

    std::vector<float> waterfall(nchans * nsamps, 1.25F);
    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "valid");
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "valid");
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(n, 0.0F);
    std::vector<float> dmt_cuda(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_cuda.execute(waterfall, dmt_cuda);
    REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.05));
}

TEST_CASE("FDMTFFTCUDA valid streaming nsamps < dt_max", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 4;
    const float tsamp   = 0.001F;
    const size_t dt_max = 12;

    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "valid");
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "valid");
    fdmt_cpu.reset_history();
    fdmt_cuda.reset_history();
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(0.5F, 2.0F);
    for (int blk = 0; blk < 4; ++blk) {
        std::vector<float> wf(nchans * nsamps);
        for (auto& v : wf) {
            v = dist(rng);
        }
        std::vector<float> dmt_cpu(n, 0.0F);
        std::vector<float> dmt_cuda(n, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_cuda.execute(wf, dmt_cuda);
        REQUIRE_THAT(dmt_cuda, Catch::Matchers::Approx(dmt_cpu).margin(0.1));
    }
}

TEST_CASE("FDMTFFTCUDA full mode interior vs CPU FFT", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;

    std::vector<float> waterfall(nchans * nsamps, 0.75F);
    FDMTFFTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "full");
    FDMTFFTCUDA fdmt_cuda(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                          "full");
    const auto ndms       = fdmt_cpu.get_plan().get_dmt_ndms();
    const auto nsamps_out = fdmt_cpu.get_plan().get_dmt_nsamps();
    std::vector<float> dmt_cpu(ndms * nsamps_out, 0.0F);
    std::vector<float> dmt_cuda(ndms * nsamps_out, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_cuda.execute(waterfall, dmt_cuda);
    for (size_t dm = 0; dm < ndms; ++dm) {
        for (size_t t = 0; t < nsamps; ++t) {
            const auto idx = (dm * nsamps_out) + t;
            REQUIRE(dmt_cuda[idx] ==
                    Catch::Approx(dmt_cpu[idx]).margin(0.05));
        }
    }
}

TEST_CASE("FDMTFFTCUDA host stepper matches execute", "[fdmt_fft_gpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;

    std::vector<float> waterfall(nchans * nsamps, 1.1F);
    FDMTFFTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "roll");
    const auto n = fdmt.get_plan().get_dmt_size();
    std::vector<float> dmt_one(n, 0.0F);
    fdmt.execute(waterfall, dmt_one);

    std::vector<float> dmt_step(n, 0.0F);
    fdmt.reset(std::span<const float>(waterfall), std::span<float>(dmt_step));
    fdmt.advance_until_remaining(1);
    CHECK(fdmt.num_subbands() == 2);
    fdmt.finalize();
    REQUIRE_THAT(dmt_step, Catch::Matchers::Approx(dmt_one).margin(0.05));
}

} // namespace dmt

#endif // DMT_ENABLE_CUDA
