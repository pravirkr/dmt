#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/common/fft_config.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "test_helpers.hpp"

#include <catch2/matchers/catch_matchers_all.hpp>

namespace dmt {

using algorithms::FDMT;
using algorithms::FDMTFFT;
using Catch::Approx;

namespace {

std::vector<float>
random_waterfall(SizeType nchans, SizeType nsamps, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(1.0F, 10.0F);
    std::vector<float> waterfall(nchans * nsamps);
    for (auto& v : waterfall) {
        v = dist(rng);
    }
    return waterfall;
}

// max |a - b| / max |b| over the first n values.
double rel_max_err(const std::vector<float>& a,
                   const std::vector<float>& b,
                   SizeType n) {
    double diff = 0.0;
    double ref  = 0.0;
    for (SizeType i = 0; i < n; ++i) {
        diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
        ref  = std::max(ref, static_cast<double>(std::abs(b[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

} // namespace

TEST_CASE("FDMTFFT basic constructor and properties", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;

    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min);
    // Fractional delays are the default (integer delays: equivalence tests).
    CHECK(fdmt_fft.fractional_delays());
    CHECK(fdmt_fft.get_output_latency() ==
          algorithms::fdmt_fft::kFractionalGuard);

    SECTION("Plan getters") {
        const auto& plan         = fdmt_fft.get_plan();
        const auto ndms_expected = static_cast<SizeType>(dt_max - dt_min + 1);
        CHECK(plan.get_dt_grid_final().size() == ndms_expected);
        CHECK(plan.get_dm_grid_final().size() == ndms_expected);
        CHECK(fdmt_fft.get_nbeams() == 1);
        CHECK(plan.get_mode() == "valid");
        CHECK(plan.get_dmt_nsamps() == nsamps);
        const auto n_min =
            nsamps + std::max(plan.get_fft_overlap(), plan.get_fft_support());
        CHECK(plan.get_fft_size() >= n_min);
        CHECK(plan.get_fft_size() % 2 == 0);
        CHECK(plan.get_fft_support() >= dt_max);
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

TEST_CASE("FDMTFFT vs FDMT roll mode numerical equivalence",
          "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    const size_t dt_min = 0;
    auto waterfall      = random_waterfall(nchans, nsamps);

    SECTION("With box smearing") {
        FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                      true, "roll");
        FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                         true, "roll", Exec::cpu(1), 1,
                         /*fractional_delays=*/false);
        const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_cpu);
        fdmt_fft.execute(waterfall, dmt_fft);
        test::require_approx(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
    }

    SECTION("Without box smearing") {
        FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                      false, "roll");
        FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                         false, "roll", Exec::cpu(1), 1,
                         /*fractional_delays=*/false);
        const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
        fdmt_cpu.execute(waterfall, dmt_cpu);
        fdmt_fft.execute(waterfall, dmt_fft);
        test::require_approx(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
    }
}

TEST_CASE("FDMTFFT vs FDMT full mode", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    auto waterfall      = random_waterfall(nchans, nsamps, 7);

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "full");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "full", Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto ndms       = fdmt_cpu.get_plan().get_dmt_ndms();
    const auto nsamps_out = fdmt_cpu.get_plan().get_dmt_nsamps();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(ndms * nsamps_out, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    // Linear (padded) FFT matches time-domain full on the input-aligned
    // region t < nsamps. The extra delay tail t >= nsamps is produced by
    // per-level buffer growth in FDMT and is not required to match.
    for (SizeType dm = 0; dm < ndms; ++dm) {
        for (SizeType t = 0; t < nsamps; ++t) {
            const auto idx = (dm * nsamps_out) + t;
            REQUIRE(dmt_fft[idx] == Approx(dmt_cpu[idx]).margin(1e-3F));
        }
    }
}

TEST_CASE("FDMTFFT vs FDMT valid first block", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 32;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 32;
    auto waterfall      = random_waterfall(nchans, nsamps, 11);

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "valid");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid", Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto n = fdmt_cpu.get_plan().get_dmt_size();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(n, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    test::require_approx(dmt_fft, dmt_cpu, n, 0.08F);
}

TEST_CASE("FDMTFFT valid streaming vs FDMT valid", "[fdmt_fft_cpu][cpu]") {
    const float f_min    = 1200.0F;
    const float f_max    = 1600.0F;
    const size_t nchans  = 16;
    const size_t nsamps  = 128;
    const float tsamp    = 0.001F;
    const size_t dt_max  = 24;
    const size_t nblocks = 3;

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "valid");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid", Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto nout = fdmt_cpu.get_plan().get_dmt_size();
    fdmt_cpu.reset_history();
    fdmt_fft.reset_history();

    for (size_t b = 0; b < nblocks; ++b) {
        auto wf =
            random_waterfall(nchans, nsamps, 100 + static_cast<unsigned>(b));
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        test::require_approx(dmt_fft, dmt_cpu, nout, 0.1F);
    }
}

TEST_CASE("FDMTFFT vs FDMT with negative delays", "[fdmt_fft_cpu][cpu]") {
    // Asymmetric grid: the Fourier support and overlap must cover both
    // signs of shift.
    const auto mode =
        GENERATE(std::string_view("roll"), std::string_view("full"),
                 std::string_view("valid"));
    CAPTURE(mode);
    const float f_min      = 1000.0F;
    const float f_max      = 1500.0F;
    const size_t nchans    = 32;
    const size_t nsamps    = 256;
    const float tsamp      = 0.001F;
    const IndexType dt_min = -16;
    const size_t dt_max    = 40;
    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1, true,
                  mode);
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, 1,
                     true, mode, Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
    const auto nout = fdmt_cpu.get_plan().get_dmt_nsamps();
    REQUIRE(fdmt_fft.get_plan().get_dmt_ndms() == ndms);
    // Full mode: FDMT's tail past nsamps is not required to match.
    const auto ncmp      = mode == "full" ? nsamps : nout;
    const size_t nblocks = mode == "valid" ? 3 : 1;
    for (size_t b = 0; b < nblocks; ++b) {
        auto wf =
            random_waterfall(nchans, nsamps, 70 + static_cast<unsigned>(b));
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(ndms * nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        for (SizeType dm = 0; dm < ndms; ++dm) {
            for (SizeType t = 0; t < ncmp; ++t) {
                const auto idx = (dm * nout) + t;
                REQUIRE(dmt_fft[idx] == Approx(dmt_cpu[idx]).margin(2e-3F));
            }
        }
    }
}

TEST_CASE("FDMTFFT odd channel counts", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1200.0F;
    const float f_max   = 1600.0F;
    const size_t nchans = 13;
    const size_t nsamps = 256;
    const float tsamp   = 0.001F;
    const size_t dt_max = 24;
    auto waterfall      = random_waterfall(nchans, nsamps, 123);

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "roll");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "roll", Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto ndms = fdmt_cpu.get_plan().get_dmt_ndms();
    std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
    std::vector<float> dmt_fft(ndms * nsamps, 0.0F);
    fdmt_cpu.execute(waterfall, dmt_cpu);
    fdmt_fft.execute(waterfall, dmt_fft);
    test::require_approx(dmt_fft, dmt_cpu, ndms * nsamps, 0.05F);
}

TEST_CASE("FDMTFFT multi-beam processing", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    const size_t nbeams = 3;
    auto waterfall      = random_waterfall(nbeams * nchans, nsamps, 999);

    FDMTFFT fdmt_multi(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                       "roll", Exec::cpu(1), nbeams);
    FDMTFFT fdmt_single(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                        "roll", Exec::cpu(1), 1);
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
        test::require_approx(
            std::vector<float>(
                dmt_multi.begin() +
                    static_cast<std::ptrdiff_t>(b * ndms * nsamps),
                dmt_multi.begin() +
                    static_cast<std::ptrdiff_t>((b + 1) * ndms * nsamps)),
            single_dmt, 1e-4);
    }
}

TEST_CASE("FDMTFFT stepper matches execute", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    auto waterfall      = random_waterfall(nchans, nsamps, 5);

    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                 "roll", Exec::cpu(1), 1, /*fractional_delays=*/false);
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
    test::require_approx(dmt_step, dmt_one, n, 1e-4F);
}

TEST_CASE("FDMTFFT variance matches plan", "[fdmt_fft_cpu][cpu]") {
    FDMTFFT fdmt(1000.0F, 1500.0F, 16, 128, 0.001F, 16);
    const auto& plan = fdmt.get_plan();
    CHECK(fdmt.get_effective_variance(0, 1) ==
          Approx(plan.get_effective_variance(0, 1, true)));
    auto grid = fdmt.get_effective_sigma_grid(2);
    REQUIRE_THAT(
        grid, Catch::Matchers::Approx(plan.get_effective_sigma_grid(2, true)));
}

TEST_CASE("FDMTFFT convenience function compute_fdmt_fft",
          "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 16;
    const size_t nsamps = 128;
    const float tsamp   = 0.001F;
    const size_t dt_max = 16;
    std::vector<float> waterfall(nchans * nsamps, 2.0F);

    auto [dmt, plan] = algorithms::compute_fdmt_fft(
        waterfall, f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
        "valid", Exec::cpu(1), 1, /*fractional_delays=*/false);
    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                 "valid", Exec::cpu(1), 1, /*fractional_delays=*/false);
    std::vector<float> dmt_class(plan.get_dmt_size(), 0.0F);
    fdmt.execute(waterfall, dmt_class);
    CHECK(dmt.size() == plan.get_dmt_ndms() * plan.get_dmt_nsamps());
    test::require_approx(dmt, dmt_class, plan.get_dmt_size(), 1e-4F);
}

TEST_CASE("FDMTFFT valid streaming with nsamps < dt_max",
          "[fdmt_fft_cpu][cpu]") {
    const float f_min    = 1000.0F;
    const float f_max    = 1500.0F;
    const size_t nchans  = 16;
    const size_t nsamps  = 8;
    const float tsamp    = 0.001F;
    const size_t dt_max  = 24;
    const size_t nblocks = 5;

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "valid");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid", Exec::cpu(1), 1, /*fractional_delays=*/false);
    const auto nout = fdmt_cpu.get_plan().get_dmt_size();
    fdmt_cpu.reset_history();
    fdmt_fft.reset_history();
    for (size_t b = 0; b < nblocks; ++b) {
        auto wf =
            random_waterfall(nchans, nsamps, 200 + static_cast<unsigned>(b));
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        test::require_approx(dmt_fft, dmt_cpu, nout, 0.15F);
    }
}

TEST_CASE("FDMTFFT multi-beam stepper matches execute", "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1000.0F;
    const float f_max   = 1500.0F;
    const size_t nchans = 8;
    const size_t nsamps = 64;
    const float tsamp   = 0.001F;
    const size_t dt_max = 8;
    const size_t nbeams = 3;
    auto waterfall      = random_waterfall(nbeams * nchans, nsamps, 44);

    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                 "roll", Exec::cpu(1), nbeams);
    const auto n = nbeams * fdmt.get_plan().get_dmt_size();
    std::vector<float> dmt_one(n, 0.0F);
    fdmt.execute(waterfall, dmt_one);

    std::vector<float> dmt_step(n, 0.0F);
    fdmt.reset(waterfall, dmt_step);
    fdmt.finalize();
    test::require_approx(dmt_step, dmt_one, n, 1e-4F);
}

// Long blocks with a short delay support are transformed in overlap-save
// segments by execute(); the stepper always uses the plan's single
// transform. Both compute the same linear convolution on the kept samples.
TEST_CASE("FDMTFFT segmented execute matches single transform",
          "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1200.0F;
    const float f_max   = 1600.0F;
    const size_t nchans = 32;
    const size_t nsamps = 16384;
    const float tsamp   = 0.001F;
    const size_t dt_max = 64;
    const auto mode =
        GENERATE(std::string_view("valid"), std::string_view("full"));
    const auto box = GENERATE(true, false);
    CAPTURE(mode, box);

    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, box, mode,
                 Exec::cpu(4), 1, /*fractional_delays=*/false);
    const auto n = fdmt.get_plan().get_dmt_size();
    for (unsigned blk = 0; blk < 3; ++blk) {
        auto wf = random_waterfall(nchans, nsamps, 700 + blk);
        std::vector<float> dmt_exec(n, 0.0F);
        std::vector<float> dmt_step(n, 0.0F);
        // Same history for both paths: step first (it updates the overlap
        // in finalize()), then rewind by replaying on a twin.
        FDMTFFT twin(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, box,
                     mode, Exec::cpu(4), 1, /*fractional_delays=*/false);
        for (unsigned prev = 0; prev < blk; ++prev) {
            auto wp = random_waterfall(nchans, nsamps, 700 + prev);
            std::vector<float> scratch(n, 0.0F);
            twin.execute(wp, scratch);
        }
        twin.reset(wf, dmt_step);
        twin.finalize();
        fdmt.execute(wf, dmt_exec);
        CHECK(rel_max_err(dmt_exec, dmt_step, n) < 2e-5);
    }
}

TEST_CASE("FDMTFFT segmented execute vs FDMT valid streaming",
          "[fdmt_fft_cpu][cpu]") {
    const float f_min   = 1200.0F;
    const float f_max   = 1600.0F;
    const size_t nchans = 32;
    const size_t nsamps = 8192;
    const float tsamp   = 0.001F;
    const size_t dt_max = 48;

    FDMT fdmt_cpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                  "valid");
    FDMTFFT fdmt_fft(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                     "valid", Exec::cpu(2), 1, /*fractional_delays=*/false);
    const auto nout = fdmt_cpu.get_plan().get_dmt_size();
    for (unsigned b = 0; b < 3; ++b) {
        auto wf = random_waterfall(nchans, nsamps, 300 + b);
        std::vector<float> dmt_cpu(fdmt_cpu.get_plan().get_buffer_size(), 0.0F);
        std::vector<float> dmt_fft(nout, 0.0F);
        fdmt_cpu.execute(wf, dmt_cpu);
        fdmt_fft.execute(wf, dmt_fft);
        CHECK(rel_max_err(dmt_fft, dmt_cpu, nout) < 2e-5);
    }
}

TEST_CASE("FDMTFFT results do not depend on the thread count",
          "[fdmt_fft_cpu][cpu]") {
    const size_t nchans = 64;
    const size_t nsamps = 4096;
    auto waterfall      = random_waterfall(nchans, nsamps, 17);
    FDMTFFT one(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 128, 0, 1, true,
                "valid", Exec::cpu(1));
    FDMTFFT many(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 128, 0, 1, true,
                 "valid", Exec::cpu(7));
    const auto n = one.get_plan().get_dmt_size();
    std::vector<float> a(n, 0.0F);
    std::vector<float> b(n, 0.0F);
    one.execute(waterfall, a);
    many.execute(waterfall, b);
    test::require_exact(a, b);
}

TEST_CASE("FFT planner effort and wisdom", "[fdmt_fft_cpu][cpu]") {
    const size_t nchans = 16;
    const size_t nsamps = 1000; // non-smooth block length
    auto waterfall      = random_waterfall(nchans, nsamps, 3);
    FDMTFFT est(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 40);
    const auto n = est.get_plan().get_dmt_size();
    std::vector<float> ref(n, 0.0F);
    est.execute(waterfall, ref);

    const auto saved = fft::get_planner();
    fft::set_planner(fft::Planner::kMeasure);
    CHECK(fft::get_planner() == fft::Planner::kMeasure);
    FDMTFFT meas(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 40);
    std::vector<float> got(n, 0.0F);
    meas.execute(waterfall, got);
    CHECK(rel_max_err(got, ref, n) < 2e-5);

    const auto path = std::filesystem::temp_directory_path() /
                      "dmt_fdmt_fft_cpu_t_wisdom.txt";
    CHECK(fft::export_wisdom(path.string()));
    fft::forget_wisdom();
    CHECK(fft::import_wisdom(path.string()));
    CHECK_FALSE(fft::import_wisdom((path.string() + ".missing")));
    std::filesystem::remove(path);
    fft::set_planner(saved);
}

TEST_CASE("FDMTFFT fractional delays: stepper matches execute",
          "[fdmt_fft_cpu][cpu]") {
    const auto mode =
        GENERATE(std::string_view("roll"), std::string_view("valid"),
                 std::string_view("full"));
    CAPTURE(mode);
    const size_t nchans = 32;
    const size_t nsamps = 512;
    auto waterfall      = random_waterfall(nchans, nsamps, 31);
    FDMTFFT fdmt(1100.0F, 1500.0F, nchans, nsamps, 0.001F, 40, 0, 1, true, mode,
                 Exec::cpu(3), 1, /*fractional_delays=*/true);
    CHECK(fdmt.fractional_delays());
    const auto n = fdmt.get_plan().get_dmt_size();
    std::vector<float> one(n, 0.0F);
    std::vector<float> step(n, 0.0F);
    FDMTFFT twin(1100.0F, 1500.0F, nchans, nsamps, 0.001F, 40, 0, 1, true, mode,
                 Exec::cpu(3), 1, true);
    fdmt.execute(waterfall, one);
    twin.reset(waterfall, step);
    twin.advance(2);
    CHECK(twin.view_subband(0).ndt > 0);
    twin.finalize();
    CHECK(rel_max_err(step, one, n) < 2e-5);

    // Integer and fractional modes differ, but only slightly.
    FDMTFFT integer(1100.0F, 1500.0F, nchans, nsamps, 0.001F, 40, 0, 1, true,
                    mode, Exec::cpu(3), 1, /*fractional_delays=*/false);
    std::vector<float> ref(n, 0.0F);
    integer.execute(waterfall, ref);
    // Valid mode lags by the look-ahead: output t is integer output t - lag.
    const auto lag  = fdmt.get_output_latency();
    const auto ndms = fdmt.get_plan().get_dmt_ndms();
    const auto nout = fdmt.get_plan().get_dmt_nsamps();
    CHECK(lag ==
          (mode == "valid" ? algorithms::fdmt_fft::kFractionalGuard : 0));
    std::vector<float> a;
    std::vector<float> b;
    for (size_t d = 0; d < ndms; ++d) {
        for (size_t t = lag; t < nout; ++t) {
            a.push_back(one[(d * nout) + t]);
            b.push_back(ref[(d * nout) + t - lag]);
        }
    }
    const auto e = rel_max_err(a, b, a.size());
    CHECK(e > 1e-6);
    CHECK(e < 0.5); // sub-sample shifts decorrelate white noise
}

TEST_CASE("FDMTFFT fractional delays stream like one long block",
          "[fdmt_fft_cpu][cpu]") {
    const size_t nchans  = 32;
    const size_t blk     = 2048;
    const size_t nblocks = 4;
    auto all             = random_waterfall(nchans, blk * nblocks, 41);
    FDMTFFT longer(1100.0F, 1500.0F, nchans, blk * nblocks, 0.001F, 48, 0, 1,
                   true, "valid", Exec::cpu(2), 1, true);
    FDMTFFT stream(1100.0F, 1500.0F, nchans, blk, 0.001F, 48, 0, 1, true,
                   "valid", Exec::cpu(2), 1, true);
    const auto ndms = longer.get_plan().get_dmt_ndms();
    std::vector<float> ref(ndms * blk * nblocks, 0.0F);
    longer.execute(all, ref);
    // Both lag the input by the guard; with that look-ahead every block
    // sample sees the whole interpolation kernel.
    REQUIRE(stream.get_output_latency() ==
            algorithms::fdmt_fft::kFractionalGuard);
    CHECK(stream.get_suggested_nsamps() >= 4 * blk / 16);
    FDMTFFT roll(1100.0F, 1500.0F, nchans, blk, 0.001F, 48, 0, 1, true, "roll");
    CHECK(roll.get_suggested_nsamps() == blk);
    CHECK(roll.get_output_latency() == 0);
    REQUIRE(longer.get_output_latency() ==
            algorithms::fdmt_fft::kFractionalGuard);
    double worst = 0.0;
    double scale = 0.0;
    for (size_t b = 0; b < nblocks; ++b) {
        std::vector<float> wf(nchans * blk);
        for (size_t c = 0; c < nchans; ++c) {
            std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(
                                          (c * blk * nblocks) + (b * blk)),
                        blk, wf.begin() + static_cast<std::ptrdiff_t>(c * blk));
        }
        std::vector<float> out(ndms * blk, 0.0F);
        stream.execute(wf, out);
        for (size_t d = 0; d < ndms; ++d) {
            for (size_t t = 0; t < blk; ++t) {
                const float a = out[(d * blk) + t];
                const float r = ref[(d * blk * nblocks) + (b * blk) + t];
                worst = std::max(worst, static_cast<double>(std::abs(a - r)));
                scale = std::max(scale, static_cast<double>(std::abs(r)));
            }
        }
    }
    CAPTURE(worst, scale);
    CHECK(worst / scale < 5e-3);
}

// Band-limited pulses dispersed along the continuous FDMT delay law (bottom
// edge reference; without box smearing a channel samples its upper edge):
// exact merge shifts realign them better than rounded ones.
TEST_CASE("FDMTFFT fractional delays improve pulse recovery",
          "[fdmt_fft_cpu][cpu]") {
    const size_t nchans = 64;
    const size_t nsamps = 1024;
    const float f_min   = 1100.0F;
    const float f_max   = 1500.0F;
    const double df     = (static_cast<double>(f_max) - f_min) / nchans;
    const auto law      = [&](double f, double dt) {
        const double a = 1.0 / (static_cast<double>(f_min) * f_min);
        const double b = 1.0 / (static_cast<double>(f_max) * f_max);
        return dt * (a - (1.0 / (f * f))) / (a - b);
    };
    FDMTFFT integer(f_min, f_max, nchans, nsamps, 0.001F, 160, 0, 1, false,
                    "full", Exec::cpu(4), 1, /*fractional_delays=*/false);
    FDMTFFT frac(f_min, f_max, nchans, nsamps, 0.001F, 160, 0, 1, false, "full",
                 Exec::cpu(4), 1, true);
    const auto& grid = integer.get_plan().get_dt_grid_final();
    const auto nout  = integer.get_plan().get_dmt_nsamps();
    std::mt19937 rng(8);
    std::uniform_real_distribution<double> phase(0.0, 1.0);
    double sum_int  = 0.0;
    double sum_frac = 0.0;
    for (const IndexType dt : {37, 73, 111, 149}) {
        const auto row =
            static_cast<SizeType>(std::ranges::find(grid, dt) - grid.begin());
        REQUIRE(row < grid.size());
        // Many arrival phases: the output is sampled on whole samples, so a
        // single pulse's peak also depends on where it falls between them.
        for (int rep = 0; rep < 24; ++rep) {
            const double t0 = 500.0 + phase(rng);
            std::vector<float> wf(nchans * nsamps, 0.0F);
            for (size_t c = 0; c < nchans; ++c) {
                const double f  = f_min + (df * static_cast<double>(c + 1));
                const double tc = t0 - law(f, static_cast<double>(dt));
                for (size_t t = 0; t < nsamps; ++t) {
                    const double z = (static_cast<double>(t) - tc) / 1.2;
                    wf[(c * nsamps) + t] =
                        static_cast<float>(std::exp(-0.5 * z * z));
                }
            }
            std::vector<float> a(integer.get_plan().get_dmt_size(), 0.0F);
            std::vector<float> b(a.size(), 0.0F);
            integer.execute(wf, a);
            frac.execute(wf, b);
            const auto peak = [&](const std::vector<float>& v) {
                return *std::max_element(
                    v.begin() + static_cast<std::ptrdiff_t>(row * nout),
                    v.begin() + static_cast<std::ptrdiff_t>((row + 1) * nout));
            };
            sum_int += peak(a);
            sum_frac += peak(b);
        }
    }
    CAPTURE(sum_int, sum_frac);
    CHECK(sum_frac > sum_int);
}

// Per-channel total shift of every root path against the continuous delay
// law: the least-squares fractional shifts roughly halve the rms
// misalignment of the rounded integer tree.
TEST_CASE("FDMTFFT fractional delays halve the path misalignment",
          "[fdmt_fft_cpu][cpu]") {
    const bool box      = GENERATE(false, true);
    const size_t nchans = 64;
    const float f_min   = 1100.0F;
    const float f_max   = 1500.0F;
    plans::FDMTPlan plan(f_min, f_max, nchans, 1024, 0.001F, 160, 0, 1, "full");
    const auto frac = algorithms::fdmt_fft::fractional_delays(plan, box);
    const auto& pc  = plan.get_container();
    const auto L    = plan.get_niters();
    const auto g    = [](double f) { return 1.0 / (f * f); };
    const double df = (static_cast<double>(f_max) - f_min) / nchans;
    std::vector<std::vector<long>> sum_of(L + 1);
    for (SizeType l = 1; l <= L; ++l) {
        sum_of[l].assign(pc.state_shape[l].ncoords, -1);
        for (SizeType i = 0; i < pc.coordinates_sum[l].size(); ++i) {
            const auto& c                      = pc.coordinates_sum[l][i];
            sum_of[l][c.buf_offset / c.nsamps] = static_cast<long>(i);
        }
    }
    double rms_int  = 0.0;
    double rms_frac = 0.0;
    for (SizeType r = 1; r < pc.state_shape[L].ncoords; ++r) {
        std::vector<double> si(nchans, 0.0);
        std::vector<double> sf(nchans, 0.0);
        std::function<void(SizeType, SizeType, double, double)> walk =
            [&](SizeType l, SizeType node, double a, double b) {
                if (l == 0) {
                    for (SizeType c = 0; c < nchans; ++c) {
                        const auto& gr = pc.grids[0][c];
                        if (node >= gr.coord_offset &&
                            node < gr.coord_offset + gr.ndt) {
                            const double dt0 = std::abs(static_cast<double>(
                                gr.dt_grid[node - gr.coord_offset]));
                            si[c]            = a + (box ? dt0 / 2 : dt0);
                            sf[c]            = b + (box ? dt0 / 2 : dt0);
                        }
                    }
                    return;
                }
                const auto i = sum_of[l][node];
                REQUIRE(i >= 0); // power-of-two channels: no copies
                const auto& c = pc.coordinates_sum[l][static_cast<SizeType>(i)];
                walk(l - 1, c.tail_buf_offset / c.tail_nsamps, a, b);
                walk(l - 1, c.head_buf_offset / c.head_nsamps,
                     a + static_cast<double>(c.delay),
                     b + frac.shift[l][static_cast<SizeType>(i)]);
            };
        walk(L, r, 0.0, 0.0);
        const double dt = static_cast<double>(pc.grids[L][0].dt_grid[r]);
        const auto rms  = [&](const std::vector<double>& sh) {
            std::vector<double> e(nchans);
            double mean = 0.0;
            for (SizeType c = 0; c < nchans; ++c) {
                const double fr = box ? 0.5 * (g(f_min + (df * c)) +
                                               g(f_min + (df * (c + 1))))
                                      : g(f_min + (df * (c + 1)));
                e[c] = sh[c] - (dt * (g(f_min) - fr) / (g(f_min) - g(f_max)));
                mean += e[c];
            }
            mean /= static_cast<double>(nchans);
            double acc = 0.0;
            for (const auto v : e) {
                acc += (v - mean) * (v - mean);
            }
            return std::sqrt(acc / static_cast<double>(nchans));
        };
        rms_int += rms(si);
        rms_frac += rms(sf);
    }
    CAPTURE(box, rms_int, rms_frac);
    CHECK(rms_frac < 0.7 * rms_int);
}

} // namespace dmt
