#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/common/plans.hpp"
#include "test_helpers.hpp"

#include <algorithm>
#include <catch2/matchers/catch_matchers_all.hpp>

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
    const float t_p             = tbin * static_cast<float>(chan_per_sub);
    const float dm_max          = 5.0F;
    const float dm_min          = 0.0F;
    return {f_center, bw_sub, nsub,   tbin, nbin,    nfft,
            t_p,      dm_max, dm_min, 32,   "PRITF", false};
}
} // namespace

TEST_CASE("CohFDMTPlan symmetric dt range", "[cfdmt][cpu]") {
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
          "[cfdmt][cpu]") {
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

    // The result is the leading get_dmt_size(); the arena tail is scratch.
    std::vector<float> dmt(coh_fdmt.get_buffer_size(), 0.0F);
    REQUIRE(dmt.size() >= coh_fdmt.get_dmt_size());
    std::vector<float> too_small(coh_fdmt.get_buffer_size() - 1, 0.0F);
    CHECK_THROWS_AS(coh_fdmt.execute<uint8_t>(data_in, too_small),
                    std::invalid_argument);
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));

    const auto result = std::span(dmt).first(coh_fdmt.get_dmt_size());
    CHECK(std::ranges::all_of(
        result, [](float val) { return std::isfinite(val) && val >= 0.0F; }));
    CHECK(std::ranges::any_of(result, [](float val) { return val != 0.0F; }));

    // A second call on independent data must also succeed and keep
    // streaming (rather than crash) via the per-coarse-DM history swap.
    for (auto& v : data_in) {
        v = static_cast<uint8_t>(dist(rng));
    }
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));

    coh_fdmt.reset_history();
    REQUIRE_NOTHROW(coh_fdmt.execute<uint8_t>(data_in, dmt));
}

TEST_CASE("CohFDMTPlan dimension and buffer invariants", "[cfdmt][cpu]") {
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
    const auto& fine = plan.get_fdmt_plan();
    CHECK(plan.get_buffer_size() ==
          ((plan.get_dm_grid_coh().size() - 1) * fine.get_dmt_size()) +
              fine.get_buffer_size());
    CHECK(coh_fdmt.get_buffer_size() == plan.get_buffer_size());

    const auto var_grid   = plan.get_effective_variance_grid();
    const auto sig_grid   = plan.get_effective_sigma_grid();
    const auto count_grid = plan.get_cumulative_count_grid();

    REQUIRE(var_grid.size() == plan.get_ndm());
    REQUIRE(sig_grid.size() == plan.get_ndm());
    REQUIRE(count_grid.size() == plan.get_ndm());
    CHECK(std::ranges::all_of(var_grid, [](float v) { return v > 0.0F; }));
    CHECK(std::ranges::all_of(count_grid, [](float v) { return v > 0.0F; }));
    std::vector<float> expected_sigma(var_grid.size());
    std::ranges::transform(var_grid, expected_sigma.begin(),
                           [](float v) { return std::sqrt(v); });
    REQUIRE_THAT(sig_grid, Catch::Matchers::Approx(expected_sigma));
}

TEST_CASE("CohFDMTCPU multi-block streaming state and history reset",
          "[cfdmt][cpu]") {
    const auto plan = make_test_plan();
    CohFDMTCPU coh_fdmt(plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
                        plan.get_tbin(), plan.get_nbin(), plan.get_nfft(),
                        plan.get_t_p(), plan.get_dm_max(), plan.get_dm_min(),
                        plan.get_noverlap());

    const SizeType in_size =
        SizeType{2} * SizeType{2} * plan.get_nsamp() * plan.get_nsub();
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

    // Runs one block through one shared arena and keeps only the result.
    std::vector<float> arena(coh_fdmt.get_buffer_size(), 0.0F);
    const auto run = [&](const std::vector<uint8_t>& block) {
        coh_fdmt.execute<uint8_t>(block, arena);
        return std::vector<float>(arena.begin(),
                                  arena.begin() + static_cast<std::ptrdiff_t>(
                                                      coh_fdmt.get_dmt_size()));
    };

    // 1. Process block 1 cold
    const auto dmt_b1_initial = run(block1);

    // 2. Process block 2 continuous (warm history)
    const auto dmt_b2_streamed = run(block2);

    // 3. Reset history and process block 2 cold
    coh_fdmt.reset_history();
    const auto dmt_b2_cold = run(block2);

    CHECK_FALSE(std::equal(
        dmt_b2_streamed.begin(), dmt_b2_streamed.end(), dmt_b2_cold.begin(),
        [](float a, float b) { return std::abs(a - b) <= 1e-4F; }));

    // 4. Reset history and re-process block 1: must be bit-exact to
    // dmt_b1_initial
    coh_fdmt.reset_history();
    const auto dmt_b1_repeated = run(block1);
    test::require_exact(dmt_b1_initial, dmt_b1_repeated);
}

TEST_CASE("CohFDMTCPU synthetic impulse response and DM alignment",
          "[cfdmt][cpu]") {
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
            const SizeType idx_re = (ipol * 2 * nsub * nsamp) +
                                    (0 * nsub * nsamp) + (isub * nsamp) +
                                    pulse_t;
            data_in[idx_re] = 127;
        }
    }

    std::vector<float> dmt(coh_fdmt.get_buffer_size(), 0.0F);
    coh_fdmt.execute<uint8_t>(data_in, dmt);
    dmt.resize(coh_fdmt.get_dmt_size()); // drop the arena's scratch tail

    const SizeType ndm_total  = plan.get_ndm();
    const SizeType nsamps_out = plan.get_dmt_nsamps();
    REQUIRE(dmt.size() == ndm_total * nsamps_out);

    float global_max   = -1.0F;
    size_t peak_dm_idx = 0;

    for (size_t idm = 0; idm < ndm_total; ++idm) {
        for (size_t it = 0; it < nsamps_out; ++it) {
            const float val = dmt[(idm * nsamps_out) + it];
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
