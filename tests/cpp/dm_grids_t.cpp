// Public DM grid generators (dmt/common/dm_grids.hpp).

#include "dmt/common/dm_grids.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/common/plans.hpp"

#include "dmt/ddmt_fft_common.hpp"

namespace dmt {

namespace {

using plans::DMGridConfig;

DMGridConfig band(double f_min, double f_max, SizeType nchans, double tsamp) {
    DMGridConfig c;
    c.f_min       = f_min;
    c.f_max       = f_max;
    c.nchans      = nchans;
    c.tsamp       = tsamp;
    c.pulse_width = tsamp;
    return c;
}

DMGridConfig uwl() { return band(704.0, 4032.0, 6656, 64e-6); }
DMGridConfig lband() { return band(1100.0, 1714.0, 1152, 455.6e-6); }

// Step of the grid at the first trial at or above dm.
double step_at(const std::vector<double>& g, double dm) {
    const auto it = std::ranges::lower_bound(g, dm);
    REQUIRE(it != g.end());
    REQUIRE(it + 1 != g.end());
    return *(it + 1) - *it;
}

std::vector<double> midpoints(const std::vector<double>& g) {
    std::vector<double> m;
    for (std::size_t i = 0; i + 1 < g.size(); ++i) {
        m.push_back(0.5 * (g[i] + g[i + 1]));
    }
    return m;
}

} // namespace

TEST_CASE("generate_levin_dm_grid matches DDMTPlan's", "[grid][cpu]") {
    auto c        = uwl();
    c.pulse_width = 40e-6;
    for (const bool piecewise : {false, true}) {
        c.piecewise     = piecewise;
        const auto grid = plans::generate_levin_dm_grid(c, 0.0, 300.0);
        const auto want =
            piecewise
                ? plans::DDMTPlan::generate_levin_dm_grid_piecewise(
                      0.0F, 300.0F, 64e-6F, 40e-6F, 704.0F, 4032.0F, 6656,
                      1.25F)
                : plans::DDMTPlan::generate_levin_dm_grid(0.0F, 300.0F, 64e-6F,
                                                          40e-6F, 704.0F,
                                                          4032.0F, 6656, 1.25F);
        REQUIRE(grid.size() == want.size());
        for (std::size_t i = 0; i < grid.size(); ++i) {
            CHECK(grid[i] == static_cast<double>(want[i]));
        }
    }
    auto bad    = c;
    bad.chan_dm = 100.0;
    CHECK_THROWS_AS(plans::generate_levin_dm_grid(bad, 0.0, 300.0),
                    std::invalid_argument);
    bad           = c;
    bad.kill_mask = std::vector<uint8_t>(c.nchans, 1);
    CHECK_THROWS_AS(plans::generate_levin_dm_grid(bad, 0.0, 300.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(plans::generate_levin_dm_grid(c, -5.0, 300.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        plans::DDMTPlan::generate_levin_dm_grid(0.0F, 300.0F, 64e-6F, 40e-6F,
                                                704.0F, 4032.0F, 6656, 0.0F),
        std::invalid_argument);
}

TEST_CASE("generate_levin_dm_grid steps keep the broadening at tol",
          "[grid][cpu]") {
    // Each step x from p solves a^2 x^2 + b^2 (x - p)^2 + w0^2
    // = tol^2 (a^2 p^2 + w0^2), with Levin's a and b (band centre).
    auto c          = uwl();
    c.pulse_width   = 40e-6;
    c.tol           = 1.25;
    const auto g    = plans::generate_levin_dm_grid(c, 300.0, 600.0);
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

TEST_CASE("generate_dm_grid: reference trial counts", "[grid][cpu]") {
    // DM 0-2000, pulse_width one sample, 5% loss.
    struct Case {
        DMGridConfig c;
        double chan_dm;
        std::size_t count;
    };
    const Case cases[] = {
        {band(704.0, 1216.0, 512, 512e-6), 0.0, 1817},
        {lband(), 0.0, 2658},
        {band(856.0, 1712.0, 4096, 612.48e-6), 0.0, 5627},
        {uwl(), 0.0, 12209},
        {uwl(), 413.0, 17809},
    };
    for (const auto& k : cases) {
        auto c       = k.c;
        c.chan_dm    = k.chan_dm;
        const auto g = plans::generate_dm_grid(c, 0.0, 2000.0);
        CHECK(g.size() == k.count);
        CHECK(g.front() == 0.0);
        CHECK(g.back() == 2000.0);
        CHECK(std::ranges::adjacent_find(g, std::greater_equal<>()) == g.end());
    }
}

TEST_CASE("generate_dm_grid: midpoints lose max_snr_loss", "[grid][cpu]") {
    for (const double loss : {0.05, 0.2}) {
        for (const double chan_dm : {0.0, 413.0}) {
            auto c         = lband();
            c.max_snr_loss = loss;
            c.chan_dm      = chan_dm;
            const auto g   = plans::generate_dm_grid(c, 0.0, 1000.0);
            const auto mid = midpoints(g);
            // Drop the last interval (clamped to dm_max).
            const std::vector<double> test(mid.begin(), mid.end() - 1);
            const auto l       = plans::calculate_snr_loss(c, g, test);
            const double worst = *std::ranges::max_element(l);
            const double best  = *std::ranges::min_element(l);
            CHECK(worst <= loss * 1.03);
            CHECK(best >= loss * 0.8);
            // Trials themselves lose nothing.
            const auto on = plans::calculate_snr_loss(c, g, g);
            CHECK(*std::ranges::max_element(on) == 0.0);
        }
    }
}

TEST_CASE("generate_dm_grid: chan_dm makes trials denser around it",
          "[grid][cpu]") {
    auto c           = uwl();
    c.chan_dm        = 413.0;
    const auto g     = plans::generate_dm_grid(c, 0.0, 1100.0);
    const auto plain = plans::generate_dm_grid(uwl(), 0.0, 1100.0);
    CHECK(g.size() > plain.size());
    CHECK(step_at(g, 413.0) < 0.5 * step_at(plain, 413.0));
    // The step depends on |DM - chan_dm| only.
    CHECK(step_at(g, 213.0) == Catch::Approx(step_at(g, 613.0)).epsilon(0.05));
}

TEST_CASE("generate_dm_grid: tol sets max_snr_loss = 1 - 1 / tol",
          "[grid][cpu]") {
    auto by_loss         = uwl();
    by_loss.max_snr_loss = 1.0 - (1.0 / 1.25);
    auto by_tol          = uwl();
    by_tol.tol           = 1.25;
    by_tol.max_snr_loss  = 0.5; // ignored
    CHECK(plans::generate_dm_grid(by_loss, 0.0, 500.0) ==
          plans::generate_dm_grid(by_tol, 0.0, 500.0));
}

TEST_CASE("generate_dm_grid: negative DMs", "[grid][cpu]") {
    const auto c   = lband();
    const auto pos = plans::generate_dm_grid(c, 10.0, 300.0);
    const auto neg = plans::generate_dm_grid(c, -300.0, -10.0);
    REQUIRE(neg.size() == pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) {
        CHECK(neg[neg.size() - 1 - i] == -pos[i]);
    }
    // Symmetric about 0, which is a trial.
    const auto both = plans::generate_dm_grid(c, -100.0, 100.0);
    REQUIRE(both.size() % 2 == 1);
    for (std::size_t i = 0; i < both.size(); ++i) {
        CHECK(both[i] == -both[both.size() - 1 - i]);
    }
    const auto to_zero = plans::generate_dm_grid(c, -100.0, 0.0);
    CHECK(to_zero.back() == 0.0);
    CHECK_FALSE(std::signbit(to_zero.back()));
    // chan_dm != 0 crosses zero in one sweep, symmetric about chan_dm.
    auto coh     = c;
    coh.chan_dm  = 50.0;
    const auto g = plans::generate_dm_grid(coh, -100.0, 200.0);
    CHECK(g.front() == -100.0);
    CHECK(g.back() == 200.0);
    CHECK(step_at(g, 0.0) == Catch::Approx(step_at(g, 100.0)).epsilon(0.05));
}

TEST_CASE("generate_dm_grid: kill_mask", "[grid][cpu]") {
    auto c           = lband();
    const auto plain = plans::generate_dm_grid(c, 0.0, 1000.0);
    c.kill_mask.assign(c.nchans, 1);
    CHECK(plans::generate_dm_grid(c, 0.0, 1000.0) == plain);
    // Without the bottom half the sweep and smearing are smaller: sparser.
    std::fill_n(c.kill_mask.begin(), c.nchans / 2, uint8_t{0});
    CHECK(plans::generate_dm_grid(c, 0.0, 1000.0).size() < plain.size());
    auto bad = c;
    bad.kill_mask.assign(c.nchans - 1, 1);
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad.kill_mask.assign(c.nchans, 0);
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
}

TEST_CASE("generate_dm_grid: piecewise grids are DDMTFFT NUFFT runs",
          "[grid][cpu]") {
    for (const double chan_dm : {0.0, 413.0}) {
        auto c          = uwl();
        c.chan_dm       = chan_dm;
        c.max_snr_loss  = 0.1;
        const auto cont = plans::generate_dm_grid(c, 0.0, 2000.0);
        c.piecewise     = true;
        const auto pw   = plans::generate_dm_grid(c, 0.0, 2000.0);
        CHECK(pw.front() == 0.0);
        CHECK(pw.back() >= 2000.0);
        // Never coarser than the continuous grid.
        const auto lc = plans::calculate_snr_loss(c, cont, midpoints(pw));
        const auto lp = plans::calculate_snr_loss(c, pw, midpoints(pw));
        for (std::size_t i = 0; i < lp.size(); ++i) {
            // (up to the interpolation of the step: 0.1% of the loss)
            CHECK(lp[i] <= std::max(lc[i], 0.1 * 1.001));
        }
        // DDMTFFT sums every trial by NUFFT runs.
        const std::vector<float> dms(pw.begin(), pw.end());
        const plans::DDMTPlan plan(704.0F, 4032.0F, 6656, 64e-6F, dms);
        const algorithms::ddmt_fft::DelayModel model(plan, 0);
        SizeType covered = 0;
        for (const auto& r : model.runs) {
            if (r.count >= algorithms::ddmt_fft::kMinNufftRun) {
                covered += r.count;
            }
        }
        CHECK(covered == dms.size());
    }
}

TEST_CASE("calculate_snr_loss", "[grid][cpu]") {
    const auto c = lband();
    const std::vector<double> trials{100.0, 110.0};
    const std::vector<double> test{100.0, 101.0, 103.0, 105.0, 107.0, 109.0};
    const auto l = plans::calculate_snr_loss(c, trials, test);
    CHECK(l[0] == 0.0);
    CHECK(l[1] < l[2]);
    CHECK(l[2] < l[3]);
    CHECK(l[4] == Catch::Approx(l[2]).margin(1e-3));
    CHECK(l[5] == Catch::Approx(l[1]).margin(1e-3));
    const std::vector<double> unsorted{110.0, 100.0};
    CHECK_THROWS_AS(plans::calculate_snr_loss(c, unsorted, test),
                    std::invalid_argument);
    CHECK_THROWS_AS(plans::calculate_snr_loss(c, {}, test),
                    std::invalid_argument);
}

TEST_CASE("generate_dm_grid: invalid input throws", "[grid][cpu]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto c           = uwl();
    CHECK_THROWS_AS(plans::generate_dm_grid(c, 10.0, 5.0),
                    std::invalid_argument);
    CHECK_THROWS_AS(plans::generate_dm_grid(c, 0.0, nan),
                    std::invalid_argument);
    auto bad  = c;
    bad.f_max = bad.f_min;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad     = c;
    bad.tol = 0.9;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad              = c;
    bad.max_snr_loss = 1.0;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    bad             = c;
    bad.pulse_width = -1.0;
    CHECK_THROWS_AS(plans::generate_dm_grid(bad, 0.0, 10.0),
                    std::invalid_argument);
    CHECK(plans::generate_dm_grid(c, 5.0, 5.0) == std::vector<double>{5.0});
}

} // namespace dmt
