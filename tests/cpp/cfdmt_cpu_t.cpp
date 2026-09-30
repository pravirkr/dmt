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

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using algorithms::CohFDMT;
using plans::CohFDMTPlan;

namespace {

// Low-frequency, multi-subband config: several coarse trials over a 1
// pc/cc range, small enough for sub-second tests (16 subbands x 1 MHz at
// 392-408 MHz, t_p = 4 us -> 64 channels, 5 coarse trials).
CohFDMTConfig small_config() {
    return {.f_center = 400.0F,
            .bw_sub   = 1.0F,
            .nsub     = 16,
            .t_p      = 4.0E-6F,
            .dm_min   = 10.0F,
            .dm_max   = 11.0F,
            .format   = BasebandFormat{.order = "FTPRI"}};
}

double disp(double dm, double f) {
    return static_cast<double>(kDispConst) * dm / (f * f);
}

// Largest |v| over the stream: quantisation scale that maps it to ~100.
float scale_for(const std::vector<ComplexType>& v) {
    float peak = 0.0F;
    for (const auto& x : v) {
        peak = std::max({peak, std::abs(x.real()), std::abs(x.imag())});
    }
    return peak > 0.0F ? 100.0F / peak : 1.0F;
}

std::vector<float> run(const CohFDMT& search,
                       const std::vector<ComplexType>& v,
                       SizeType nsamps_total,
                       SizeType t_begin,
                       float scale) {
    const auto& plan = search.get_plan();
    const auto bytes = utils::pack_baseband(
        v, plan.get_nsub(), nsamps_total, plan.get_format(), scale, 0, 0,
        t_begin, plan.get_block_nsamps());
    std::vector<float> dmt(search.get_dmt_size());
    search.execute<uint8_t>(std::span<const uint8_t>(bytes), dmt);
    return dmt;
}

struct Peak {
    SizeType row;
    SizeType t;
    float value;
};

Peak find_peak(const std::vector<float>& dmt, SizeType nout) {
    const auto it = std::ranges::max_element(dmt);
    const auto i  = static_cast<SizeType>(std::distance(dmt.begin(), it));
    return {i / nout, i % nout, *it};
}

// Detected energy of a noise-free stream (normalize = false units): each
// channel keeps mean(|H|^2) / n_p of its subband's energy.
double detected_energy(const CohFDMTPlan& plan,
                       const std::vector<ComplexType>& v,
                       float scale) {
    const auto& taper = plan.get_channel_taper();
    double w2         = 0.0;
    for (const float w : taper) {
        w2 += static_cast<double>(w) * static_cast<double>(w);
    }
    w2 /= static_cast<double>(taper.size());
    double e = 0.0;
    for (const auto& x : v) {
        e += std::norm(std::complex<double>(std::round(x.real() * scale),
                                            std::round(x.imag() * scale)));
    }
    return w2 * e / static_cast<double>(plan.get_n_p());
}

// Stream holding one pulse whose arrival time at f_ref lands on output
// sample j0 of the block starting at raw sample s0.
std::vector<ComplexType> pulse_stream(const CohFDMTPlan& plan,
                                      SizeType nsamps_total,
                                      double dm,
                                      double j0,
                                      SizeType s0 = 0) {
    const double t_ref = (static_cast<double>(s0) * plan.get_tbin()) +
                         plan.get_output_time_offset() +
                         (j0 * static_cast<double>(plan.get_tsamp()));
    const utils::BasebandPulse pulse{
        .dm        = dm,
        .t_arrival = t_ref - disp(dm, plan.get_f_ref()),
        .fluence   = 1.0E6};
    return utils::simulate_baseband(plan.get_f_center(), plan.get_bw_sub(),
                                    plan.get_nsub(), nsamps_total,
                                    std::span(&pulse, 1), 0.0F, 42, 4);
}

SizeType nearest_row(const CohFDMTPlan& plan, double dm) {
    const auto& grid = plan.get_dm_grid_final();
    SizeType best    = 0;
    for (SizeType i = 1; i < grid.size(); ++i) {
        if (std::abs(grid[i] - dm) < std::abs(grid[best] - dm)) {
            best = i;
        }
    }
    return best;
}

} // namespace

TEST_CASE("CohFDMTPlan geometry and exact coarse grid", "[cfdmt][cpu]") {
    const CohFDMTPlan plan(small_config());
    const auto n_p = plan.get_n_p();
    CHECK(n_p == 4);
    CHECK(plan.get_nchans() == 64);
    CHECK(plan.get_tbin() == Catch::Approx(1.0E-6));
    CHECK(plan.get_tsamp() == Catch::Approx(4.0E-6));
    REQUIRE(plan.get_ndm_coh() > 1);

    SECTION("coarse windows tile [dm_min, dm_max]") {
        const auto& coh  = plan.get_dm_grid_coh();
        const auto step  = plan.get_dm_step_coh();
        CHECK(coh.front() - (step / 2) == Catch::Approx(10.0F));
        CHECK(coh.back() + (step / 2) == Catch::Approx(11.0F));
        for (SizeType k = 1; k < coh.size(); ++k) {
            CHECK(coh[k] - coh[k - 1] == Catch::Approx(step));
        }
        // No gaps between consecutive trials' fine rows.
        const auto nfine = plan.get_ndm_fine();
        const auto& fin  = plan.get_dm_grid_final();
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
        CHECK(worst == Catch::Approx(plan.get_intra_channel_smear()).epsilon(1e-3));
        // The grid is as coarse as the criterion allows: one trial fewer
        // would exceed it.
        const double wider =
            (plan.get_dm_max() - plan.get_dm_min()) /
            static_cast<double>(plan.get_ndm_coh() - 1) / 2.0;
        const double lo = plan.get_f_min();
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
        CHECK(plan.get_msamp() >= plan.get_output_nsamps() +
                                      plan.get_max_delay());
        CHECK(plan.get_dmt_size() ==
              plan.get_ndm() * plan.get_output_nsamps());
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
    const CohFDMTConfig cfg{.f_center = 1406.25F,
                            .bw_sub   = 1500.0F / 512.0F,
                            .nsub     = 64,
                            .t_p      = 10.0E-6F,
                            .dm_min   = 0.0F,
                            .dm_max   = 500.0F,
                            .block_nsamps = 0};
    const CohFDMTPlan plan(cfg);
    // Old rule: ceil(delay * tbin / (2 t_p^2)) with the subband tbin.
    const double delay = disp(500.0, plan.get_f_min()) -
                         disp(500.0, plan.get_f_max());
    const double old = std::ceil(delay * plan.get_tbin() / (2.0 * 1.0E-5 * 1.0E-5));
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
}

TEST_CASE("CohFDMT recovers an injected dispersed impulse", "[cfdmt][cpu]") {
    auto cfg       = small_config();
    cfg.normalize  = false;
    const CohFDMT search(cfg, Exec::cpu(4));
    const auto& plan = search.get_plan();
    const auto nout  = plan.get_output_nsamps();
    const auto n     = plan.get_block_nsamps();
    const auto& coh  = plan.get_dm_grid_coh();
    const double half = plan.get_dm_step_coh() / 2.0;

    struct Case {
        double dm;
        double j0;
        double min_frac; // of the detected energy in the peak sample
    };
    const auto third = std::floor(static_cast<double>(nout) / 3.0);
    const auto mid   = std::floor(static_cast<double>(nout) / 2.0);
    const std::array<Case, 6> cases{
        Case{coh[2], mid, 0.7},                              // trial centre
        Case{coh[1] + (0.98 * half), 40.0, 0.7},             // window edge
        Case{coh[0] - (0.98 * half), 3.0, 0.7},              // dm_min edge
        Case{coh.back() + (0.98 * half),                     // dm_max edge,
             static_cast<double>(nout) - 4.0, 0.7},          // block end
        Case{coh[3] - (0.5 * half), third, 0.7},             // negative dt
        Case{coh[2] + (0.3 * half), third + 0.5, 0.45}};     // between samples
    for (const auto& cs : cases) {
        CAPTURE(cs.dm, cs.j0);
        const auto v     = pulse_stream(plan, n, cs.dm, cs.j0);
        const float sc   = scale_for(v);
        const auto dmt   = run(search, v, n, 0, sc);
        const auto peak  = find_peak(dmt, nout);
        const auto want  = nearest_row(plan, cs.dm);
        const double e   = detected_energy(plan, v, sc);
        const double fine_step =
            plan.get_dm_grid_final()[1] - plan.get_dm_grid_final()[0];
        CAPTURE(peak.row, want, peak.t, peak.value / e);
        CHECK(std::abs(plan.get_dm_grid_final()[peak.row] - cs.dm) <=
              1.5 * fine_step);
        CHECK(std::abs(static_cast<double>(peak.t) - cs.j0) <= 1.0);
        CHECK(peak.value >= cs.min_frac * e);
        CHECK(peak.value <= 1.05 * e);
    }
}

TEST_CASE("CohFDMT impulse response does not depend on position in the block",
          "[cfdmt][cpu]") {
    auto cfg      = small_config();
    cfg.normalize = false;
    const CohFDMT search(cfg, Exec::cpu(4));
    const auto& plan  = search.get_plan();
    const auto nout   = plan.get_output_nsamps();
    const auto n      = plan.get_block_nsamps();
    const auto lc     = plan.get_mbin() - (2 * plan.get_noverlap() / plan.get_n_p());
    const double dm   = plan.get_dm_grid_coh()[2] + 0.1 * plan.get_dm_step_coh();
    std::vector<float> peaks;
    // Shifts by whole FFT blocks keep the pulse's phase relative to the
    // block grid: identical response wherever it sits in the valid window.
    for (const double j0 : {5.0, 5.0 + static_cast<double>(lc),
                            5.0 + static_cast<double>(nout - (2 * lc))}) {
        const auto v   = pulse_stream(plan, n, dm, j0);
        const auto dmt = run(search, v, n, 0, 1000.0F);
        const auto pk  = find_peak(dmt, nout);
        CHECK(static_cast<double>(pk.t) == Catch::Approx(j0).margin(1.0));
        peaks.push_back(pk.value);
    }
    CHECK(peaks[1] == Catch::Approx(peaks[0]).epsilon(0.02));
    CHECK(peaks[2] == Catch::Approx(peaks[0]).epsilon(0.02));
}

TEST_CASE("CohFDMT skipback blocks tile one long block", "[cfdmt][cpu]") {
    auto cfg      = small_config();
    cfg.normalize = false;
    const CohFDMT small(cfg, Exec::cpu(4));
    const auto& ps     = small.get_plan();
    const SizeType nb  = 3;
    const auto stride  = ps.get_stride_nsamps();
    const auto total   = ps.get_block_nsamps() + ((nb - 1) * stride);
    auto big_cfg         = cfg;
    big_cfg.block_nsamps = total;
    const CohFDMT big(big_cfg, Exec::cpu(4));
    const auto& pb = big.get_plan();
    REQUIRE(pb.get_nbin() == ps.get_nbin());
    REQUIRE(pb.get_block_nsamps() == total);
    REQUIRE(pb.get_output_nsamps() == nb * ps.get_output_nsamps());

    // Noise plus a pulse straddling the first block boundary.
    const auto nout = ps.get_output_nsamps();
    const double dm = ps.get_dm_grid_coh()[1];
    const double t_ref =
        ps.get_output_time_offset() +
        (static_cast<double>(nout) * static_cast<double>(ps.get_tsamp()));
    const utils::BasebandPulse pulse{
        .dm = dm, .t_arrival = t_ref - disp(dm, ps.get_f_ref()), .fluence = 1e5};
    const auto v = utils::simulate_baseband(cfg.f_center, cfg.bw_sub, cfg.nsub,
                                            total, std::span(&pulse, 1), 8.0F,
                                            3, 4);
    const auto whole = run(big, v, total, 0, 1.0F);
    const auto ndm   = ps.get_ndm();
    std::vector<float> tiled(ndm * nb * nout);
    for (SizeType b = 0; b < nb; ++b) {
        const auto part = run(small, v, total, b * stride, 1.0F);
        for (SizeType r = 0; r < ndm; ++r) {
            std::copy_n(part.begin() + static_cast<std::ptrdiff_t>(r * nout),
                        nout,
                        tiled.begin() + static_cast<std::ptrdiff_t>(
                                            (r * nb * nout) + (b * nout)));
        }
    }
    double max_rel = 0.0;
    double scale   = 0.0;
    for (const float x : whole) {
        scale = std::max(scale, static_cast<double>(std::abs(x)));
    }
    for (SizeType i = 0; i < whole.size(); ++i) {
        max_rel = std::max(max_rel,
                           std::abs(static_cast<double>(whole[i] - tiled[i])) /
                               scale);
    }
    INFO("max |whole - tiled| / max|whole| = " << max_rel);
    CHECK(max_rel < 1.0E-5);

    // Stateless: a block gives bit-identical output on a fresh engine and
    // after other blocks (no state leaks between calls or coarse trials).
    const CohFDMT fresh(cfg, Exec::cpu(4));
    const auto first = run(fresh, v, total, stride, 1.0F);
    static_cast<void>(run(fresh, v, total, 0, 1.0F));
    const auto again = run(fresh, v, total, stride, 1.0F);
    CHECK(again == first);
}

TEST_CASE("CohFDMT normalised noise matches the variance grid",
          "[cfdmt][cpu]") {
    auto cfg         = small_config();
    cfg.block_nsamps = 1 << 18;
    const CohFDMT search(cfg, Exec::cpu(4));
    const auto& plan = search.get_plan();
    const auto nout  = plan.get_output_nsamps();
    const auto n     = plan.get_block_nsamps();
    const auto v     = utils::simulate_baseband(
        cfg.f_center, cfg.bw_sub, cfg.nsub, n, {}, 16.0F, 7, 4);
    const auto dmt   = run(search, v, n, 0, 1.0F);
    const auto ndm   = plan.get_ndm();
    for (const SizeType w : {SizeType{1}, SizeType{16}}) {
        const auto grid = plan.get_effective_variance_grid(w);
        double ratio    = 0.0;
        double mean_sum = 0.0;
        for (SizeType r = 0; r < ndm; ++r) {
            const float* row = dmt.data() + (r * nout);
            // Boxcar of w samples.
            std::vector<double> box(nout - w + 1, 0.0);
            double acc = 0.0;
            for (SizeType t = 0; t < nout; ++t) {
                acc += row[t];
                if (t >= w) {
                    acc -= row[t - w];
                }
                if (t + 1 >= w) {
                    box[t + 1 - w] = acc;
                }
            }
            const double m =
                std::accumulate(box.begin(), box.end(), 0.0) /
                static_cast<double>(box.size());
            double var = 0.0;
            for (const double x : box) {
                var += (x - m) * (x - m);
            }
            var /= static_cast<double>(box.size() - 1);
            ratio += var / static_cast<double>(grid[r]);
            mean_sum += m / std::sqrt(static_cast<double>(grid[r]));
        }
        ratio /= static_cast<double>(ndm);
        INFO("w=" << w << " mean(measured / predicted variance) = " << ratio
                  << ", mean(row mean / sigma) = "
                  << mean_sum / static_cast<double>(ndm));
        CHECK(ratio == Catch::Approx(1.0).epsilon(0.02));
        CHECK(std::abs(mean_sum / static_cast<double>(ndm)) < 0.05);
    }
}

TEST_CASE("CohFDMT output is independent of layout, encoding and grouping",
          "[cfdmt][cpu]") {
    auto cfg = small_config();
    const CohFDMTPlan plan(cfg);
    const auto n     = plan.get_block_nsamps();
    const double dm  = plan.get_dm_grid_coh()[1];
    const auto v     = pulse_stream(plan, n, dm, 100.0);
    auto noisy       = utils::simulate_baseband(cfg.f_center, cfg.bw_sub,
                                                cfg.nsub, n, {}, 5.0F, 9, 4);
    const float sc   = scale_for(v) * 0.5F;
    for (SizeType i = 0; i < v.size(); ++i) {
        noisy[i] += v[i] * sc;
    }
    const auto reference = [&] {
        const CohFDMT s(cfg, Exec::cpu(4));
        return run(s, noisy, n, 0, 1.0F);
    }();
    SECTION("orders PRITF and TFPRI, offset-binary uint8") {
        for (const auto* order : {"PRITF", "TFPRI"}) {
            for (const bool is_signed : {true, false}) {
                auto c = cfg;
                c.format.order     = order;
                c.format.is_signed = is_signed;
                const CohFDMT s(c, Exec::cpu(4));
                CAPTURE(order, is_signed);
                CHECK(run(s, noisy, n, 0, 1.0F) == reference);
            }
        }
    }
    SECTION("two subband groups") {
        auto c           = cfg;
        c.subband_groups = {6, 10};
        const CohFDMT s(c, Exec::cpu(4));
        const auto g0 = utils::pack_baseband(noisy, cfg.nsub, n, cfg.format,
                                             1.0F, 0, 6);
        const auto g1 = utils::pack_baseband(noisy, cfg.nsub, n, cfg.format,
                                             1.0F, 6, 10);
        const std::array<std::span<const uint8_t>, 2> groups{
            std::span<const uint8_t>(g0), std::span<const uint8_t>(g1)};
        std::vector<float> out(s.get_dmt_size());
        s.execute<uint8_t>(std::span<const std::span<const uint8_t>>(groups),
                           out);
        CHECK(out == reference);
    }
    SECTION("thread count does not change the result") {
        const CohFDMT s(cfg, Exec::cpu(1));
        CHECK(run(s, noisy, n, 0, 1.0F) == reference);
    }
}

TEST_CASE("CohFDMT 4-bit input and dt_step recover the pulse", "[cfdmt][cpu]") {
    auto cfg         = small_config();
    cfg.normalize    = false;
    cfg.format.nbits = 4;
    cfg.dt_step      = 4;
    const CohFDMT search(cfg, Exec::cpu(4));
    const auto& plan = search.get_plan();
    CHECK(plan.get_ndm_fine() == (2 * plan.get_fine_dt_max() / 4) + 1);
    const auto nout = plan.get_output_nsamps();
    const auto n    = plan.get_block_nsamps();
    const double dm = plan.get_dm_grid_coh()[2] + (0.3 * plan.get_dm_step_coh());
    const auto v    = pulse_stream(plan, n, dm, 200.0);
    const float sc  = scale_for(v) * 0.07F; // peak ~7 counts
    const auto dmt  = run(search, v, n, 0, sc);
    const auto pk   = find_peak(dmt, nout);
    const double fine_step =
        plan.get_dm_grid_final()[1] - plan.get_dm_grid_final()[0];
    CHECK(std::abs(plan.get_dm_grid_final()[pk.row] - dm) <= fine_step);
    CHECK(std::abs(static_cast<double>(pk.t) - 200.0) <= 2.0);
}

TEST_CASE("CohFDMT execute validates buffers", "[cfdmt][cpu]") {
    const CohFDMT search(small_config());
    std::vector<uint8_t> in(search.get_input_size());
    std::vector<float> out(search.get_dmt_size());
    std::vector<float> small_out(search.get_dmt_size() - 1);
    std::vector<uint8_t> small_in(search.get_input_size() - 4);
    CHECK_THROWS_AS(search.execute<uint8_t>(std::span<const uint8_t>(in),
                                            small_out),
                    std::invalid_argument);
    CHECK_THROWS_AS(search.execute<uint8_t>(std::span<const uint8_t>(small_in),
                                            out),
                    std::invalid_argument);
    CHECK_NOTHROW(search.execute<uint8_t>(std::span<const uint8_t>(in), out));
    CHECK(search.get_memory_usage().total() > 0);
    CHECK(search.get_buffer_size() == search.get_dmt_size());
}

} // namespace dmt
