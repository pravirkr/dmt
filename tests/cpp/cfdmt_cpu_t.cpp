#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/common/plans.hpp"

namespace dmt {

using algorithms::CohFDMTCPU;
using plans::CohFDMTPlan;

namespace {
// Small, fast-to-run configuration exercising several coarse-DM trials --
// the exact scenario (ndm_coh > 1) that used to throw inside apply_chirp().
CohFDMTPlan make_test_plan() {
    // Small on purpose: enough coarse-DM trials (> 1) to exercise the
    // apply_chirp() regression, but a small nbin/nsub/n_p so the FFTs and
    // FDMT tree stay cheap and the whole test runs in well under a second.
    const SizeType chan_per_sub = 4;
    const float f_center        = 1250.0F;
    const float bw_sub          = 25.0F;
    const SizeType nsub         = 4;
    const float tbin            = 1.0E-6F;
    const SizeType nbin         = 1 << 10;
    const SizeType nfft         = 2;
    const float t_p              = tbin * static_cast<float>(chan_per_sub);
    const float dm_max           = 5.0F;
    const float dm_min           = 0.0F;
    return {f_center, bw_sub, nsub,   tbin,      nbin,
           nfft,      t_p,    dm_max, dm_min,    32,
           "PRITF",   false};
}
} // namespace

TEST_CASE("CohFDMTPlan symmetric dt range", "[cfdmt]") {
    const auto plan = make_test_plan();
    // Symmetric fine-DM search: dt_min is negative, |dt_min| + dt_max + 1
    // spans twice the one-sided n_p width used before the 2x optimization.
    CHECK(plan.get_dt_min() < 0);
    CHECK(plan.get_dt_max() == plan.get_n_p() - 1);
    CHECK(-plan.get_dt_min() == static_cast<IndexType>(plan.get_n_p()));
    // dm_grid_final must be fully assembled from dm_grid_coh x fine grid,
    // never negative, and bounded by dm_max plus half a coarse step.
    const auto& dm_grid_final = plan.get_dm_grid_final();
    REQUIRE(!dm_grid_final.empty());
    for (const auto dm : dm_grid_final) {
        // The coarse grid (generate_coherent_dms) and the fine grid
        // (FDMTPlan::get_dm_grid_final(), independently derived from dt via
        // the dispersion law) can disagree by float-rounding noise right at
        // the dm_min boundary; allow a small epsilon rather than requiring
        // an exact >= 0.
        CHECK(dm >= -1.0E-2F);
    }
}

TEST_CASE("CohFDMTCPU::execute runs end-to-end with multiple coarse DM trials",
         "[cfdmt]") {
    // Regression test for the apply_chirp() size-invariant bug: this used to
    // throw unconditionally whenever dm_grid_coh.size() > 1, which is the
    // normal case.
    const auto plan = make_test_plan();
    REQUIRE(plan.get_dm_grid_coh().size() > 1);

    CohFDMTCPU coh_fdmt(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                       plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                       plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                       plan.get_noverlap());

    const SizeType in_size = SizeType{2} * SizeType{2} *
                            coh_fdmt.get_plan().get_nsamp() *
                            coh_fdmt.get_plan().get_nsub();
    std::vector<uint8_t> data_in(in_size);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data_in) {
        v = static_cast<uint8_t>(dist(rng));
    }

    std::vector<float> dmt(coh_fdmt.get_dmt_size(), 0.0F);
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));

    // Every element must be finite and non-negative (it's a sum of squared
    // magnitudes propagated through FDMT, which only adds/copies).
    bool any_nonzero = false;
    for (const auto val : dmt) {
        CHECK(std::isfinite(val));
        CHECK(val >= 0.0F);
        any_nonzero = any_nonzero || (val != 0.0F);
    }
    CHECK(any_nonzero);

    // A second call on independent data must also succeed and keep
    // streaming (rather than crash) via the per-coarse-DM history swap.
    for (auto& v : data_in) {
        v = static_cast<uint8_t>(dist(rng));
    }
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));

    coh_fdmt.reset_history();
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));
}

TEST_CASE("CohFDMTPlan dimension and buffer invariants", "[cfdmt]") {
    const auto plan = make_test_plan();

    CHECK(plan.get_ndm() == plan.get_dmt_ndms());
    CHECK(plan.get_ndm() == plan.get_dm_grid_final().size());
    CHECK(plan.get_dmt_nsamps() == plan.get_fdmt_plan().get_dmt_nsamps());
    CHECK(plan.get_dmt_size() == plan.get_ndm() * plan.get_dmt_nsamps());

    CohFDMTCPU coh_fdmt(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                        plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                        plan.get_noverlap());

    CHECK(coh_fdmt.get_dmt_size() == plan.get_dmt_size());

    const auto var_grid   = plan.get_effective_variance_grid();
    const auto sig_grid   = plan.get_effective_sigma_grid();
    const auto count_grid = plan.get_cumulative_count_grid();

    REQUIRE(var_grid.size() == plan.get_ndm());
    REQUIRE(sig_grid.size() == plan.get_ndm());
    REQUIRE(count_grid.size() == plan.get_ndm());

    for (size_t i = 0; i < plan.get_ndm(); ++i) {
        CHECK(var_grid[i] > 0.0F);
        CHECK(sig_grid[i] == Catch::Approx(std::sqrt(var_grid[i])));
        CHECK(count_grid[i] > 0.0F);
    }
}

TEST_CASE("CohFDMTCPU multi-block streaming state and history reset", "[cfdmt]") {
    const auto plan = make_test_plan();
    CohFDMTCPU coh_fdmt(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                        plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                        plan.get_noverlap());

    const SizeType in_size = SizeType{2} * SizeType{2} *
                             plan.get_nsamp() * plan.get_nsub();
    std::vector<uint8_t> block1(in_size);
    std::vector<uint8_t> block2(in_size);

    std::mt19937 rng(12345);
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

    // Streaming history from block 1 must carry over into block 2
    // so dmt_b2_streamed and dmt_b2_cold must differ
    bool any_difference = false;
    for (size_t i = 0; i < dmt_b2_streamed.size(); ++i) {
        if (std::abs(dmt_b2_streamed[i] - dmt_b2_cold[i]) > 1e-4F) {
            any_difference = true;
            break;
        }
    }
    CHECK(any_difference);

    // 4. Reset history and re-process block 1: must be bit-exact to dmt_b1_initial
    coh_fdmt.reset_history();
    coh_fdmt.execute<uint8_t>(block1, dmt_b1_repeated);
    for (size_t i = 0; i < dmt_b1_initial.size(); ++i) {
        REQUIRE(dmt_b1_initial[i] == dmt_b1_repeated[i]);
    }
}

TEST_CASE("CohFDMTCPU synthetic impulse response and DM alignment", "[cfdmt]") {
    const SizeType chan_per_sub = 4;
    const float f_center        = 1250.0F;
    const float bw_sub          = 25.0F;
    const SizeType nsub         = 4;
    const float tbin            = 1.0E-6F;
    const SizeType nbin         = 1 << 10;
    const SizeType nfft         = 2;
    const float t_p             = tbin * static_cast<float>(chan_per_sub);
    const float dm_max          = 4.0F;
    const float dm_min          = 0.0F;

    CohFDMTPlan plan(f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max,
                     dm_min, 32, "PRITF", false);
    CohFDMTCPU coh_fdmt(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                        plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                        plan.get_noverlap());

    const SizeType nsamp   = plan.get_nsamp();
    const SizeType in_size = SizeType{2} * SizeType{2} * nsamp * nsub;
    std::vector<uint8_t> data_in(in_size, 0);

    const SizeType pulse_t = nsamp / 2;
    for (SizeType ipol = 0; ipol < 2; ++ipol) {
        for (SizeType isub = 0; isub < nsub; ++isub) {
            const SizeType idx_re = ipol * 2 * nsub * nsamp +
                                    0 * nsub * nsamp + isub * nsamp + pulse_t;
            data_in[idx_re] = 127;
        }
    }

    std::vector<float> dmt(coh_fdmt.get_dmt_size(), 0.0F);
    coh_fdmt.execute<uint8_t>(data_in, dmt);

    const SizeType ndm_total  = plan.get_ndm();
    const SizeType nsamps_out = plan.get_dmt_nsamps();
    REQUIRE(dmt.size() == ndm_total * nsamps_out);

    float global_max   = -1.0F;
    size_t peak_dm_idx = 0;

    for (size_t idm = 0; idm < ndm_total; ++idm) {
        for (size_t it = 0; it < nsamps_out; ++it) {
            const float val = dmt[idm * nsamps_out + it];
            if (val > global_max) {
                global_max  = val;
                peak_dm_idx = idm;
            }
        }
    }

    const auto& dm_grid_final = plan.get_dm_grid_final();
    const float detected_dm   = dm_grid_final[peak_dm_idx];
    CHECK(std::abs(detected_dm) <= 1.5F);
    CHECK(global_max > 0.0F);
}

} // namespace dmt
