// Public DM and delay grid generators (dmt/common/dm_grids.hpp).

#include "dmt/common/dm_grids.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/common/plans.hpp"

namespace dmt {

namespace {

using plans::DMGridConfig;
using plans::DMGridMethod;

DMGridConfig uwl(DMGridMethod method) {
    DMGridConfig c;
    c.f_min       = 704.0;
    c.f_max       = 4032.0;
    c.nchans      = 6656;
    c.tsamp       = 64e-6;
    c.pulse_width = 40e-6;
    c.method      = method;
    return c;
}

// Step of the grid at the first trial at or above dm.
double step_at(const std::vector<double>& g, double dm) {
    const auto it = std::ranges::lower_bound(g, dm);
    REQUIRE(it != g.end());
    REQUIRE(it + 1 != g.end());
    return *(it + 1) - *it;
}

} // namespace

TEST_CASE("generate_dm_grid: Levin rules match DDMTPlan's", "[grid][cpu]") {
    for (const auto method :
         {DMGridMethod::kLevin, DMGridMethod::kLevinPiecewise}) {
        const auto c    = uwl(method);
        const auto grid = plans::generate_dm_grid(c, 0.0, 300.0);
        const auto want =
            method == DMGridMethod::kLevin
                ? plans::DDMTPlan::generate_levin_dm_grid(0.0F, 300.0F, 64e-6F,
                                                          40e-6F, 704.0F,
                                                          4032.0F, 6656, 1.25F)
                : plans::DDMTPlan::generate_levin_dm_grid_piecewise(
                      0.0F, 300.0F, 64e-6F, 40e-6F, 704.0F, 4032.0F, 6656,
                      1.25F);
        REQUIRE(grid.size() == want.size());
        for (std::size_t i = 0; i < grid.size(); ++i) {
            CHECK(grid[i] == static_cast<double>(want[i]));
        }
    }
}

TEST_CASE("generate_dm_grid: chan_dm makes trials denser around it",
          "[grid][cpu]") {
    // Coherently dedispersed at 413: no intra-channel smearing there, so the
    // step at 413 is the step of an unsmeared pulse (as at DM 0 without
    // coherent dedispersion), and away from 413 the smearing returns.
    auto plain    = uwl(DMGridMethod::kSnrLoss);
    auto coh      = plain;
    coh.chan_dm   = 413.0;
    const auto g0 = plans::generate_dm_grid(plain, 0.0, 1100.0);
    const auto g1 = plans::generate_dm_grid(coh, 0.0, 1100.0);
    CHECK(step_at(g1, 413.0) < 0.1 * step_at(g0, 413.0));
    CHECK(step_at(g1, 413.0) == Catch::Approx(step_at(g0, 0.0)).epsilon(0.01));
    CHECK(step_at(g1, 0.0) == Catch::Approx(step_at(g0, 413.0)).epsilon(0.05));
    CHECK(g1.size() > g0.size());
    CHECK(std::ranges::is_sorted(g1));
    CHECK(g1.front() == 0.0);
    CHECK(g1.back() == 1100.0);
    // The legacy Levin rules keep their classic formula: no chan_dm.
    auto levin   = coh;
    levin.method = DMGridMethod::kLevin;
    CHECK_THROWS_AS(plans::generate_dm_grid(levin, 0.0, 1100.0),
                    std::invalid_argument);
}

TEST_CASE("generate_dm_grid: tol is the S/N-loss rule's knob too",
          "[grid][cpu]") {
    auto by_loss         = uwl(DMGridMethod::kSnrLoss);
    by_loss.max_snr_loss = 0.2;
    auto by_tol          = uwl(DMGridMethod::kSnrLoss);
    by_tol.tol           = 1.25; // 1 / (1 - 0.2)
    by_tol.max_snr_loss  = 0.01; // ignored
    const auto a         = plans::generate_dm_grid(by_loss, 0.0, 500.0);
    const auto b         = plans::generate_dm_grid(by_tol, 0.0, 500.0);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i] == Catch::Approx(b[i]).epsilon(1e-12));
    }
    // A looser tolerance, fewer trials.
    by_tol.tol = 1.5;
    CHECK(plans::generate_dm_grid(by_tol, 0.0, 500.0).size() < a.size());
}

TEST_CASE("generate_dm_grid: Levin steps keep the broadening at tol",
          "[grid][cpu]") {
    // Each step x from p solves a^2 x^2 + b^2 (x - p)^2 + w0^2
    // = tol^2 (a^2 p^2 + w0^2), with Levin's a and b (band centre).
    auto c          = uwl(DMGridMethod::kLevin);
    c.tol           = 1.25;
    const auto g    = plans::generate_dm_grid(c, 300.0, 600.0);
    const double df = (c.f_max - c.f_min) / static_cast<double>(c.nchans);
    const double fc = (c.f_min + c.f_max) * 0.5e-3;
    const double a  = 8.3 * df / (fc * fc * fc);
    const double a2 = a * a;
    const double b2 = a2 * static_cast<double>(c.nchans * c.nchans) / 16.0;
    const double w0 =
        std::pow(c.tsamp * 1e6, 2) + std::pow(c.pulse_width * 1e6, 2);
    const double tol2 = c.tol * c.tol;
    REQUIRE(g.size() > 10);
    for (std::size_t i = 0; i + 2 < g.size(); ++i) {
        const double p   = g[i];
        const double x   = g[i + 1];
        const double lhs = (a2 * x * x) + (b2 * std::pow(x - p, 2)) + w0;
        const double rhs = tol2 * ((a2 * p * p) + w0);
        // Trials are stored as float: relative error ~1e-7 of the DM.
        CHECK(lhs == Catch::Approx(rhs).epsilon(1e-2));
    }
}

TEST_CASE("generate_dm_grid: S/N-loss rule over negative DMs", "[grid][cpu]") {
    auto c           = uwl(DMGridMethod::kSnrLoss);
    c.chan_dm        = 50.0;
    const auto pos   = plans::generate_dm_grid(c, 10.0, 300.0);
    auto mirrored    = c;
    mirrored.chan_dm = -50.0;
    const auto neg   = plans::generate_dm_grid(mirrored, -300.0, -10.0);
    REQUIRE(neg.size() == pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) {
        CHECK(neg[neg.size() - 1 - i] == -pos[i]);
    }
    const auto both = plans::generate_dm_grid(c, -100.0, 100.0);
    CHECK(std::ranges::is_sorted(both));
    CHECK(both.front() == -100.0);
    CHECK(both.back() == 100.0);
    CHECK(std::ranges::count(both, 0.0) == 1);
}

TEST_CASE("generate_dt_grid: integer delays with both ends", "[grid][cpu]") {
    for (const auto method : {DMGridMethod::kSnrLoss, DMGridMethod::kLevin}) {
        const auto c = uwl(method);
        const auto g = plans::generate_dt_grid(c, 0.0, 5000.0);
        REQUIRE(g.size() > 10);
        CHECK(g.front() == 0);
        CHECK(g.back() == 5000);
        CHECK(std::ranges::adjacent_find(g, std::greater_equal<>()) == g.end());
        const plans::FDMTPlan plan(704.0F, 4032.0F, 64, 8192, 64e-6F,
                                   std::vector<IndexType>(g.begin(), g.end()));
        CHECK(plan.get_dt_grid_final().size() == g.size());
    }
}

TEST_CASE("generate_dm_grid: invalid input throws", "[grid][cpu]") {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    auto c         = uwl(DMGridMethod::kSnrLoss);
    CHECK_THROWS_AS(plans::generate_dm_grid(c, 10.0, 5.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(plans::generate_dm_grid(c, 0.0, nan),
                    std::invalid_argument);
    auto bad    = c;
    bad.chan_dm = nan;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad              = c;
    bad.max_snr_loss = 1.0;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad       = c;
    bad.f_max = 500.0;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad        = c;
    bad.method = DMGridMethod::kLevin;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, -5.0, 10.0),
                    std::invalid_argument);
    bad     = c;
    bad.tol = 1.0;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
}

} // namespace dmt
