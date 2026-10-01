#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "cfdmt_behaviour.hpp"
#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using algorithms::CohFDMT;
using plans::CohFDMTPlan;

using namespace test::cfdmt; // NOLINT(google-build-using-namespace)

TEST_CASE("CohFDMTPlan geometry and exact coarse grid", "[cfdmt][cpu]") {
    const CohFDMTPlan plan(small_config());
    const auto n_p = plan.get_n_p();
    CHECK(n_p == 4);
    CHECK(plan.get_nchans() == 64);
    CHECK(plan.get_tbin() == Catch::Approx(1.0E-6));
    CHECK(plan.get_tsamp() == Catch::Approx(4.0E-6));
    REQUIRE(plan.get_ndm_coh() > 1);

    SECTION("coarse windows tile [dm_min, dm_max]") {
        const auto& coh = plan.get_dm_grid_coh();
        const auto step = plan.get_dm_step_coh();
        CHECK(coh.front() - (step / 2) == Catch::Approx(10.0F));
        CHECK(coh.back() + (step / 2) == Catch::Approx(11.0F));
        for (SizeType k = 1; k < coh.size(); ++k) {
            CHECK(coh[k] - coh[k - 1] == Catch::Approx(step));
        }
        // No gaps between consecutive trials' fine rows.
        const auto nfine      = plan.get_ndm_fine();
        const auto& fin       = plan.get_dm_grid_final();
        const float fine_step = fin[1] - fin[0];
        for (SizeType k = 1; k < coh.size(); ++k) {
            CHECK(fin[(k * nfine) - 1] >= fin[k * nfine] - fine_step);
        }
    }

    SECTION("residual intra-channel smearing <= smear_tol in every channel") {
        const double half = plan.get_dm_step_coh() / 2.0;
        double worst      = 0.0;
        for (SizeType c = 0; c < plan.get_nchans(); ++c) {
            const double lo = plan.get_f_min() +
                              (static_cast<double>(c) * plan.get_bw_chan());
            const double smear =
                (disp(half, lo) - disp(half, lo + plan.get_bw_chan())) /
                plan.get_tsamp();
            worst = std::max(worst, smear);
        }
        CHECK(worst <= 1.0 + 1.0E-4);
        CHECK(worst ==
              Catch::Approx(plan.get_intra_channel_smear()).epsilon(1e-3));
        // The grid is as coarse as the criterion allows: one trial fewer
        // would exceed it.
        const double wider = (plan.get_dm_max() - plan.get_dm_min()) /
                             static_cast<double>(plan.get_ndm_coh() - 1) / 2.0;
        const double lo    = plan.get_f_min();
        CHECK((disp(wider, lo) - disp(wider, lo + plan.get_bw_chan())) /
                  plan.get_tsamp() >
              1.0);
    }

    SECTION("block bookkeeping") {
        const auto nov = plan.get_noverlap();
        const auto lc  = plan.get_mbin() - (2 * nov / n_p);
        CHECK(nov % n_p == 0);
        CHECK(plan.get_nbin() == n_p * plan.get_mbin());
        CHECK(plan.get_block_nsamps() ==
              (plan.get_nfft() * lc * n_p) + (2 * nov));
        CHECK(plan.get_stride_nsamps() + plan.get_overlap_nsamps() ==
              plan.get_block_nsamps());
        CHECK(plan.get_stride_nsamps() == plan.get_output_nsamps() * n_p);
        CHECK(plan.get_output_nsamps() % lc == 0);
        CHECK(plan.get_output_nsamps() > 0);
        CHECK(plan.get_msamp() >=
              plan.get_output_nsamps() + plan.get_max_delay());
        CHECK(plan.get_dmt_size() == plan.get_ndm() * plan.get_output_nsamps());
        CHECK(plan.get_input_size() == 4 * plan.get_block_nsamps() * 16);
    }

    SECTION("noise statistics grids") {
        const auto var   = plan.get_effective_variance_grid();
        const auto var16 = plan.get_effective_variance_grid(16);
        const auto cnt   = plan.get_cumulative_count_grid();
        REQUIRE(var.size() == plan.get_ndm());
        for (SizeType i = 0; i < var.size(); ++i) {
            CHECK(var[i] >= static_cast<float>(plan.get_nchans()));
            CHECK(cnt[i] >= static_cast<float>(plan.get_nchans()));
            // Positive lag correlation: more than 16x the w=1 variance.
            CHECK(var16[i] > 16.0F * var[i]);
        }
        const auto r = plan.get_lag_correlation(4);
        CHECK(r[0] == Catch::Approx(1.0));
        CHECK(r[1] > 0.0);
        CHECK(r[1] < 0.01);
    }
}

TEST_CASE("CohFDMTPlan multi-subband grid is ~nsub/2 coarser than the "
          "single-band rule",
          "[cfdmt][cpu]") {
    // GUPPI node: 64 x 2.9296875 MHz at 1.4 GHz, t_p = 10 us, DM 50-60.
    const CohFDMTConfig cfg{.f_center     = 1406.25F,
                            .bw_sub       = 1500.0F / 512.0F,
                            .nsub         = 64,
                            .t_p          = 10.0E-6F,
                            .dm_min       = 0.0F,
                            .dm_max       = 500.0F,
                            .block_nsamps = 0};
    const CohFDMTPlan plan(cfg);
    // Old rule: ceil(delay * tbin / (2 t_p^2)) with the subband tbin.
    const double delay =
        disp(500.0, plan.get_f_min()) - disp(500.0, plan.get_f_max());
    const double old =
        std::ceil(delay * plan.get_tbin() / (2.0 * 1.0E-5 * 1.0E-5));
    INFO("old " << old << " new " << plan.get_ndm_coh());
    CHECK(old / static_cast<double>(plan.get_ndm_coh()) > 64.0 / 4.0);
    CHECK(plan.get_intra_channel_smear() <= 1.0F);
}

TEST_CASE("CohFDMTPlan rejects invalid configurations", "[cfdmt][cpu]") {
    auto bad = small_config();
    bad.nsub = 0;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad        = small_config();
    bad.dm_max = 5.0F;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad              = small_config();
    bad.format.order = "PRIT";
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad              = small_config();
    bad.format.nbits = 16;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad                = small_config();
    bad.subband_groups = {4, 4};
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad              = small_config();
    bad.block_nsamps = 1000; // shorter than the dispersion sweep
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad      = small_config();
    bad.nbin = 1001;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad          = small_config();
    bad.f_center = 5.0F;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
    bad                = small_config();
    bad.filter_leakage = 0.0F;
    CHECK_THROWS_AS(CohFDMTPlan(bad), std::invalid_argument);
}

TEST_CASE("CohFDMT recovers an injected dispersed impulse", "[cfdmt][cpu]") {
    check_impulse_recovery(Exec::cpu(4));
}

TEST_CASE("CohFDMT impulse response does not depend on position in the block",
          "[cfdmt][cpu]") {
    check_position_independence(Exec::cpu(4));
}

TEST_CASE("CohFDMT skipback blocks tile one long block", "[cfdmt][cpu]") {
    check_skipback_tiling(Exec::cpu(4));
}

TEST_CASE("CohFDMT normalised noise matches the variance grid",
          "[cfdmt][cpu]") {
    check_noise_statistics(Exec::cpu(4));
}

TEST_CASE("CohFDMT output is independent of layout, encoding and grouping",
          "[cfdmt][cpu]") {
    check_layout_independence(Exec::cpu(4));
}

TEST_CASE("CohFDMT 4-bit input and dt_step recover the pulse", "[cfdmt][cpu]") {
    check_4bit_dt_step(Exec::cpu(4));
}

TEST_CASE("CohFDMT execute validates buffers", "[cfdmt][cpu]") {
    check_validation(Exec::cpu(4));
}
TEST_CASE("CohFDMT trimmed filter margin matches a 1e-6 margin",
          "[cfdmt][cpu]") {
    check_filter_leakage(Exec::cpu(4));
}
TEST_CASE("CohFDMT concurrent calls on one engine", "[cfdmt][cpu]") {
    check_concurrent_calls(Exec::cpu(2));
}

} // namespace dmt
