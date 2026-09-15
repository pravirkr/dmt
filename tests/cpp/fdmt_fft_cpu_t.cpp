#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTFFTCPU;
using Catch::Approx;

namespace {

std::vector<float> random_waterfall(SizeType nchans, SizeType nsamps,
                                    unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(1.0F, 10.0F);
    std::vector<float> waterfall(nchans * nsamps);
    for (auto& v : waterfall) {
        v = dist(rng);
    }
    return waterfall;
}

void require_close(const std::vector<float>& a, const std::vector<float>& b,
                   SizeType n, float margin) {
    REQUIRE(a.size() >= n);
    REQUIRE(b.size() >= n);
    for (SizeType i = 0; i < n; ++i) {
        REQUIRE(a[i] == Approx(b[i]).margin(margin));
    }
}

} // namespace

TEST_CASE("FDMTFFTCPU basic constructor and properties", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;

    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);

    SECTION("Plan getters") {
        const auto& plan           = fdmt_fft.get_plan();
        const auto ndms_expected =
            static_cast<SizeType>(dt_max - dt_min + 1);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(fdmt_fft.get_nbeams() == 1);
        CHECK(plan.get_mode() == "valid");
        CHECK(plan.get_dmt_nsamps() == nsamps);
        CHECK(plan.get_fft_size() ==
              nsamps + plan.get_fft_overlap() + plan.get_max_shift());
        CHECK(plan.get_fft_n_bins() == (plan.get_fft_size() / 2) + 1);
    }

    SECTION("Invalid buffer sizes throw") {
        std::vector<float> bad_wf(10, 1.0F);
        std::vector<float> dmt(fdmt_fft.get_plan().get_dmt_size(), 0.0F);
        CHECK_THROWS_AS(fdmt_fft.execute(bad_wf, dmt), std::invalid_argument);

        std::vector<float> good_wf(nchans * nsamps, 1.0F);
        std::vector<float> bad_dmt(5, 0.0F);
        CHECK_THROWS_AS(fdmt_fft.execute(good_wf, bad_dmt),
                        std::invalid_argument);
    }
}

TEST_CASE("FDMTFFTCPU vs FDMTCPU roll mode numerical equivalence",
          "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;
    auto waterfall      = random_waterfall(nchans, nsamps);

    SECTION("With box smearing") {
        FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                         true, "roll");
        FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
                            1, true, "roll");
        const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_cpu);
        fdmt_fft.execute(waterfall, dmt_fft);
        require_close(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
    }

    SECTION("Without box smearing") {
        FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                         false, "roll");
        FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min,
                            1, false, "roll");
        const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_cpu);
        fdmt_fft.execute(waterfall, dmt_fft);
        require_close(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
    }
}

TEST_CASE("FDMTFFTCPU vs FDMTCPU full mode", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    auto waterfall      = random_waterfall(nchans, nsamps, 7);

    FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "full");
    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "full");
    const auto ndms       = fdmt_cpu.get_plan().get_dmt_ndms();
    const auto nsamps_out = fdmt_cpu.get_plan().get_dmt_nsamps();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(ndms * nsamps_out, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    // Linear (padded) FFT matches time-domain full on the input-aligned
    // region t < nsamps. The extra delay tail t >= nsamps is produced by
    // per-level buffer growth in FDMTCPU and is not required to match.
    for (SizeType dm = 0; dm < ndms; ++dm) {
        for (SizeType t = 0; t < nsamps; ++t) {
            const auto idx = (dm * nsamps_out) + t;
            REQUIRE(dmt_fft[idx] == Approx(dmt_cpu[idx]).margin(1e-3F));
        }
    }
}

TEST_CASE("FDMTFFTCPU vs FDMTCPU valid first block", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    auto waterfall      = random_waterfall(nchans, nsamps, 11);

    FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid");
    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "valid");
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    require_close(dmt_fft, dmt_cpu, n, 0.08F);
}

TEST_CASE("FDMTFFTCPU valid streaming vs FDMTCPU valid", "[fdmt_fft_cpu]") {
    const float f_min   = 1200.0F;
    const float f_max   = 1600.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 24;
    const size_t nblocks = 3;

    FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid");
    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "valid");
    const auto nout = fdmt_cpu.get_plan().get_dmt_size();
    fdmt_cpu.reset_history();
    fdmt_fft.reset_history();

    for (size_t b = 0; b < nblocks; ++b) {
        auto wf = random_waterfall(nchans, nsamps, 100 + static_cast<unsigned>(b));
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        require_close(dmt_fft, dmt_cpu, nout, 0.1F);
    }
}

TEST_CASE("FDMTFFTCPU odd channel counts", "[fdmt_fft_cpu]") {
    const float f_min   = 1200.0F;
    const float f_max   = 1600.0F;
    const size_t nchans = 13;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 24;
    auto waterfall      = random_waterfall(nchans, nsamps, 123);

    FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "roll");
    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "roll");
    const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    require_close(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
}

TEST_CASE("FDMTFFTCPU multi-beam processing", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    const size_t nbeams = 3;
    auto waterfall      = random_waterfall(nbeams * nchans, nsamps, 999);

    FDMTFFTCPU fdmt_multi(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1,
                          true, "roll", false, 1, nbeams);
    FDMTFFTCPU fdmt_single(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1,
                           true, "roll", false, 1, 1);
    const auto ndms = fdmt_multi.get_plan().get_dmt_ndms();
    std::vector<float> dmt_multi(nbeams * ndms * nsamps, 0.0F);
    fdmt_multi.execute(waterfall, dmt_multi);

    for (size_t b = 0; b < nbeams; ++b) {
        std::vector<float> single_wf(
            waterfall.begin() +
                static_cast<std::ptrdiff_t>(b * nchans * nsamps),
            waterfall.begin() +
                static_cast<std::ptrdiff_t>((b + 1) * nchans * nsamps));
        std::vector<float> single_dmt(ndms * nsamps, 0.0F);
        fdmt_single.execute(single_wf, single_dmt);
        for (size_t i = 0; i < single_dmt.size(); ++i) {
            REQUIRE(dmt_multi[(b * ndms * nsamps) + i] ==
                    Approx(single_dmt[i]).margin(1e-4F));
        }
    }
}

TEST_CASE("FDMTFFTCPU stepper matches execute", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    auto waterfall      = random_waterfall(nchans, nsamps, 5);

    FDMTFFTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                    "roll");
    const auto n = fdmt.get_plan().get_dmt_size();
    std::vector<float> dmt_one(n, 0.0F);
    fdmt.execute(waterfall, dmt_one);

    std::vector<float> dmt_step(n, 0.0F);
    fdmt.reset(waterfall, dmt_step);
    CHECK(fdmt.current_level() == 0);
    fdmt.advance_until_remaining(1);
    CHECK(fdmt.remaining_levels() == 1);
    CHECK(fdmt.num_subbands() == 2);
    auto sub0 = fdmt.view_subband(0);
    CHECK(sub0.ndt > 0);
    fdmt.finalize();
    require_close(dmt_step, dmt_one, n, 1e-4F);
}

TEST_CASE("FDMTFFTCPU variance matches plan", "[fdmt_fft_cpu]") {
    FDMTFFTCPU fdmt(1000.0F, 1500.0F, 16, 128, 0.001F, 16);
    const auto& plan = fdmt.get_plan();
    CHECK(fdmt.get_effective_variance(0, 1) ==
          Approx(plan.get_effective_variance(0, 1, true)));
    auto grid = fdmt.get_effective_sigma_grid(2);
    CHECK(grid.size() == plan.get_dmt_ndms());
}

TEST_CASE("FDMTFFTCPU convenience function compute_fdmt_fft", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    std::vector<float> waterfall(nchans * nsamps, 2.0F);

    auto [dmt, plan] = algorithms::compute_fdmt_fft(
        waterfall, f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
        "valid");
    CHECK(dmt.size() == plan.get_dmt_ndms() * plan.get_dmt_nsamps());
}

TEST_CASE("FDMTFFTCPU valid streaming with nsamps < dt_max", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 8;
    const float tsamp   = 0.001F;
    const size_t dt_max = 24;
    const size_t nblocks = 5;

    FDMTCPU fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid");
    FDMTFFTCPU fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "valid");
    const auto nout = fdmt_cpu.get_plan().get_dmt_size();
    fdmt_cpu.reset_history();
    fdmt_fft.reset_history();
    for (size_t b = 0; b < nblocks; ++b) {
        auto wf = random_waterfall(nchans, nsamps,
                                   200 + static_cast<unsigned>(b));
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        require_close(dmt_fft, dmt_cpu, nout, 0.15F);
    }
}

TEST_CASE("FDMTFFTCPU multi-beam stepper matches execute", "[fdmt_fft_cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;
    const size_t nbeams = 3;
    auto waterfall      = random_waterfall(nbeams * nchans, nsamps, 44);

    FDMTFFTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                    "roll", false, 1, nbeams);
    const auto n = nbeams * fdmt.get_plan().get_dmt_size();
    std::vector<float> dmt_one(n, 0.0F);
    fdmt.execute(waterfall, dmt_one);

    std::vector<float> dmt_step(n, 0.0F);
    fdmt.reset(waterfall, dmt_step);
    fdmt.finalize();
    require_close(dmt_step, dmt_one, n, 1e-4F);
}

} // namespace dmt
