#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "test_helpers.hpp"

// The balanced merge tree and its staggered subbands (FDMTPlan
// subband_depth): tree shape, the value of every node in every engine path,
// the dispersion delays a subband row applies, and the stepper view.
namespace dmt {
namespace {

using algorithms::FDMT;
using algorithms::FDMTFFT;
using Catch::Approx;
using plans::FDMTCoordGrid;
using plans::FDMTPlan;

SizeType width(const FDMTCoordGrid& g) { return g.chan_end - g.chan_begin; }

// Per-coordinate channel shifts of `level` row `idx` (no box smearing): the
// row at time t sums input channel c at t - shift[c], once per channel.
void leaf_shifts(const plans::FDMTPlanContainer& pc,
                 SizeType level,
                 SizeType idx,
                 SizeType shift,
                 std::vector<int64_t>& out) {
    const auto& c = pc.coordinates[level][idx];
    if (level == 0) {
        const auto dt = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
        REQUIRE(out[c.i_sub] < 0); // every channel reached once
        out[c.i_sub] = static_cast<int64_t>(shift) + std::abs(dt);
        return;
    }
    const auto n = pc.state_shape[level - 1].nsamps;
    leaf_shifts(pc, level - 1, c.tail_buf_offset / n, shift, out);
    if (c.i_coord_head != SIZE_MAX) {
        leaf_shifts(pc, level - 1, c.head_buf_offset / n, shift + c.delay, out);
    }
}

} // namespace

TEST_CASE("FDMT tree halves the band into equal subbands",
          "[fdmt][subband][cpu]") {
    // 1152 = 9 * 2^7: subbands of every depth up to 7 are exactly equal.
    const FDMTPlan plan(1100.0F, 1714.0F, 1152, 64, 455.6e-6F, 400, 0, 1,
                        "valid", 3);
    const auto& pc    = plan.get_container();
    const auto niters = plan.get_niters();
    REQUIRE(niters == 11);
    CHECK(plan.get_subband_depth() == 3);
    CHECK(plan.get_max_fuse_levels() == niters - 4);
    for (SizeType depth = 0; depth <= 7; ++depth) {
        const auto l = niters - depth;
        INFO("depth " << depth);
        const auto& grids = pc.grids[l];
        const auto ntree  = pc.state_shape[l].nchans;
        REQUIRE(ntree == (SizeType{1} << depth));
        const auto w = SizeType{1152} >> depth;
        for (SizeType i = 0; i < ntree; ++i) {
            CHECK(grids[i].chan_begin == i * w);
            CHECK(width(grids[i]) == w);
            CHECK_FALSE(grids[i].staggered);
        }
        // Staggered subbands: same width, centred on the tree boundaries.
        const auto nstag = depth >= 1 && depth <= 3 ? ntree - 1 : 0;
        CHECK(pc.state_shape[l].nstaggered == nstag);
        REQUIRE(grids.size() == ntree + nstag);
        for (SizeType k = 0; k < nstag; ++k) {
            const auto& g = grids[ntree + k];
            CHECK(g.staggered);
            CHECK(g.chan_begin == (k * w) + (w / 2));
            CHECK(width(g) == w);
            CHECK(g.f_start ==
                  plan.get_f_min() +
                      (plan.get_df() * static_cast<float>(g.chan_begin)));
        }
    }
    // Non-power-of-two remainders stay at the bottom: widths at any depth
    // differ by at most one channel.
    const FDMTPlan odd(700.0F, 1200.0F, 1000, 32, 0.001F, 64);
    const auto& po = odd.get_container();
    for (SizeType l = 0; l <= odd.get_niters(); ++l) {
        SizeType lo = SIZE_MAX, hi = 0, covered = 0;
        for (SizeType i = 0; i < po.state_shape[l].nchans; ++i) {
            lo = std::min(lo, width(po.grids[l][i]));
            hi = std::max(hi, width(po.grids[l][i]));
            CHECK(po.grids[l][i].chan_begin == covered);
            covered = po.grids[l][i].chan_end;
        }
        CHECK(covered == 1000);
        CHECK(hi - lo <= 1);
    }
}

TEST_CASE("FDMT subband depth leaves the root transform unchanged",
          "[fdmt][subband][cpu]") {
    for (const SizeType nchans : {SizeType{12}, SizeType{37}, SizeType{64}}) {
        const SizeType nsamps = 256;
        const FDMTPlan base(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 90, -30);
        std::mt19937 random(7);
        std::uniform_int_distribution<int> value(-8, 8);
        std::vector<float> input(nchans * nsamps);
        for (auto& x : input) {
            x = static_cast<float>(value(random));
        }
        for (SizeType depth = 1; depth < base.get_niters(); ++depth) {
            const FDMTPlan plan(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 90,
                                -30, 1, "valid", depth);
            CHECK(plan.get_dt_grid_final() == base.get_dt_grid_final());
            for (const bool smear : {true, false}) {
                for (const SizeType fuse :
                     {SizeType{0}, algorithms::kFDMTAutoFuse}) {
                    DYNAMIC_SECTION("nchans=" << nchans << " depth=" << depth
                                              << " smear=" << smear
                                              << " fuse=" << fuse) {
                        FDMT ref(base, smear, Exec::cpu(2), 1, fuse);
                        FDMT sub(plan, smear, Exec::cpu(2), 1, fuse);
                        CHECK(sub.get_fuse_levels() <=
                              plan.get_max_fuse_levels());
                        std::vector<float> a(base.get_dmt_size());
                        std::vector<float> b(plan.get_dmt_size());
                        for (int block = 0; block < 2; ++block) {
                            ref.execute(input, a);
                            sub.execute(input, b);
                            CHECK(a == b);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMT stepper computes every tree and staggered subband exactly",
          "[fdmt][subband][cpu]") {
    for (const SizeType nchans : {SizeType{7}, SizeType{24}, SizeType{37}}) {
        const SizeType nsamps = 200;
        const FDMTPlan probe(800.0F, 1400.0F, nchans, nsamps, 0.001F, 60, -20);
        const SizeType depth = probe.get_niters() - 1;
        const FDMTPlan plan(800.0F, 1400.0F, nchans, nsamps, 0.001F, 60, -20, 1,
                            "valid", depth);
        const auto& pc = plan.get_container();
        std::mt19937 random(11);
        std::uniform_int_distribution<int> value(0, 15);
        std::vector<float> input(nchans * nsamps);
        for (auto& x : input) {
            x = static_cast<float>(value(random));
        }
        DYNAMIC_SECTION("nchans=" << nchans << " depth=" << depth) {
            FDMT fdmt(plan, false, Exec::cpu(2), 1, 0, false, nsamps);
            std::vector<float> output(plan.get_dmt_size());
            fdmt.reset(input, output);
            for (SizeType l = 0;; ++l) {
                REQUIRE(fdmt.current_level() == l);
                REQUIRE(fdmt.num_subbands() == pc.grids[l].size());
                for (SizeType s = 0; s < fdmt.num_subbands(); ++s) {
                    const auto view = fdmt.view_subband(s);
                    const auto& g   = pc.grids[l][s];
                    CHECK(view.staggered == g.staggered);
                    CHECK(view.chan_begin == g.chan_begin);
                    CHECK(view.chan_end == g.chan_end);
                    for (SizeType d = 0; d < view.ndt; ++d) {
                        // Each subband sums exactly its own channels.
                        std::vector<int64_t> shift(nchans, -1);
                        leaf_shifts(pc, l, g.coord_offset + d, 0, shift);
                        for (SizeType c = 0; c < nchans; ++c) {
                            const bool inside =
                                c >= g.chan_begin && c < g.chan_end;
                            CHECK((shift[c] >= 0) == inside);
                        }
                        for (SizeType t = 0; t < nsamps; ++t) {
                            float expected = 0;
                            for (SizeType c = g.chan_begin; c < g.chan_end;
                                 ++c) {
                                const auto at =
                                    static_cast<int64_t>(t) - shift[c];
                                if (at >= 0) {
                                    expected +=
                                        input[(c * nsamps) +
                                              static_cast<SizeType>(at)];
                                }
                            }
                            REQUIRE(view.data[(d * view.nsamps) + t] ==
                                    expected);
                        }
                    }
                }
                if (fdmt.is_finished()) {
                    break;
                }
                fdmt.advance();
            }
        }
    }
}

TEST_CASE("FDMT stepper subbands stay exact across streamed blocks",
          "[fdmt][subband][cpu]") {
    // Valid-mode history carries every node, staggered ones included: the
    // second block's views equal the traced sums over the continuous
    // stream.
    for (const SizeType nchans : {SizeType{12}, SizeType{37}}) {
        const SizeType nsamps = 160;
        const FDMTPlan probe(800.0F, 1400.0F, nchans, nsamps, 0.001F, 50, -15);
        const SizeType depth = probe.get_niters() - 1;
        const FDMTPlan plan(800.0F, 1400.0F, nchans, nsamps, 0.001F, 50, -15, 1,
                            "valid", depth);
        const auto& pc = plan.get_container();
        std::mt19937 random(5);
        std::uniform_int_distribution<int> value(0, 15);
        std::vector<float> stream(nchans * 2 * nsamps);
        for (auto& x : stream) {
            x = static_cast<float>(value(random));
        }
        std::vector<float> blocks[2];
        for (SizeType b = 0; b < 2; ++b) {
            blocks[b].resize(nchans * nsamps);
            for (SizeType c = 0; c < nchans; ++c) {
                std::copy_n(stream.begin() +
                                static_cast<std::ptrdiff_t>((c * 2 * nsamps) +
                                                            (b * nsamps)),
                            nsamps,
                            blocks[b].begin() +
                                static_cast<std::ptrdiff_t>(c * nsamps));
            }
        }
        DYNAMIC_SECTION("nchans=" << nchans << " depth=" << depth) {
            FDMT fdmt(plan, false, Exec::cpu(2), 1, 0, false, nsamps);
            std::vector<float> output(plan.get_dmt_size());
            for (SizeType b = 0; b < 2; ++b) {
                fdmt.reset(blocks[b], output);
                for (SizeType l = 0;; ++l) {
                    for (SizeType sb = 0; b == 1 && sb < fdmt.num_subbands();
                         ++sb) {
                        const auto view = fdmt.view_subband(sb);
                        const auto& g   = pc.grids[l][sb];
                        for (SizeType d = 0; d < view.ndt; ++d) {
                            std::vector<int64_t> shift(nchans, -1);
                            leaf_shifts(pc, l, g.coord_offset + d, 0, shift);
                            for (SizeType t = 0; t < nsamps; ++t) {
                                float expected = 0;
                                for (SizeType c = g.chan_begin; c < g.chan_end;
                                     ++c) {
                                    const auto at =
                                        static_cast<int64_t>(nsamps + t) -
                                        shift[c];
                                    if (at >= 0) {
                                        expected +=
                                            stream[(c * 2 * nsamps) +
                                                   static_cast<SizeType>(at)];
                                    }
                                }
                                INFO("level " << l << " subband " << sb
                                              << " staggered " << g.staggered
                                              << " row " << d << " t " << t);
                                REQUIRE(view.data[(d * view.nsamps) + t] ==
                                        expected);
                            }
                        }
                    }
                    if (fdmt.is_finished()) {
                        break;
                    }
                    fdmt.advance();
                }
            }
        }
    }
}

TEST_CASE("FDMTFFT plan constructor exposes staggered subbands",
          "[fdmt][subband][cpu][fdmt_fft_cpu]") {
    const SizeType nchans = 24;
    const SizeType nsamps = 256;
    const FDMTPlan base(800.0F, 1400.0F, nchans, nsamps, 0.001F, 60, -20);
    const FDMTPlan plan(800.0F, 1400.0F, nchans, nsamps, 0.001F, 60, -20, 1,
                        "valid", 3);
    const auto& pc = plan.get_container();
    std::mt19937 random(9);
    std::uniform_int_distribution<int> value(0, 15);
    std::vector<float> input(nchans * nsamps);
    for (auto& x : input) {
        x = static_cast<float>(value(random));
    }
    // Rounded delays: every node equals FDMT's (up to FFT rounding).
    FDMTFFT fft(plan, false, Exec::cpu(2), 1, false);
    FDMT ref(plan, false, Exec::cpu(2), 1, 0, false, nsamps);
    CHECK(fft.get_plan().get_subband_depth() == 3);
    std::vector<float> out_a(plan.get_dmt_size());
    std::vector<float> out_b(plan.get_dmt_size());
    fft.reset(input, out_a);
    ref.reset(input, out_b);
    SizeType staggered = 0;
    for (SizeType l = 0;; ++l) {
        REQUIRE(fft.current_level() == l);
        REQUIRE(fft.num_subbands() == pc.grids[l].size());
        for (SizeType sb = 0; sb < fft.num_subbands(); ++sb) {
            const auto va = fft.view_subband(sb);
            const auto vb = ref.view_subband(sb);
            CHECK(va.staggered == pc.grids[l][sb].staggered);
            CHECK(va.chan_begin == vb.chan_begin);
            CHECK(va.chan_end == vb.chan_end);
            staggered += va.staggered ? 1 : 0;
            REQUIRE(va.data.size() == vb.data.size());
            for (SizeType i = 0; i < va.data.size(); ++i) {
                INFO("level " << l << " subband " << sb << " i " << i);
                REQUIRE(std::abs(va.data[i] - vb.data[i]) <=
                        2e-5F * std::max(1.0F, std::abs(vb.data[i])) *
                            static_cast<float>(nchans));
            }
        }
        if (fft.is_finished()) {
            break;
        }
        fft.advance();
        ref.advance();
    }
    CHECK(staggered == 1 + 3 + 7);
    // Fractional delays: the depth leaves the root transform unchanged.
    FDMTFFT a(base, true, Exec::cpu(2));
    FDMTFFT b(plan, true, Exec::cpu(2));
    std::vector<float> ya(base.get_dmt_size());
    std::vector<float> yb(plan.get_dmt_size());
    a.execute(input, ya);
    b.execute(input, yb);
    // Equal up to float FFT rounding (the depth changes level fusion).
    float scale = 0.0F;
    for (const auto v : ya) {
        scale = std::max(scale, std::abs(v));
    }
    for (SizeType i = 0; i < ya.size(); ++i) {
        REQUIRE(std::abs(ya[i] - yb[i]) <= 2e-5F * scale);
    }
}

TEST_CASE("FDMTPlan subband sigma matches the root and channel counts",
          "[fdmt][subband][cpu]") {
    const FDMTPlan plan(800.0F, 1400.0F, 37, 128, 0.001F, 60, -20, 1, "valid",
                        2);
    const auto niters = plan.get_niters();
    for (const SizeType w : {SizeType{1}, SizeType{4}}) {
        const auto root = plan.get_subband_variance_grid(niters, 0, w);
        CHECK(root == plan.get_effective_variance_grid(w));
        CHECK(plan.get_subband_sigma_grid(niters, 0, w) ==
              plan.get_effective_sigma_grid(w));
    }
    const auto& pc = plan.get_container();
    for (SizeType l = 0; l <= niters; ++l) {
        for (SizeType sb = 0; sb < pc.grids[l].size(); ++sb) {
            const auto& g  = pc.grids[l][sb];
            const auto var = plan.get_subband_variance_grid(l, sb, 3, false);
            REQUIRE(var.size() == g.ndt);
            for (const auto v : var) {
                CHECK(v == static_cast<float>(3 * (g.chan_end - g.chan_begin)));
            }
            // With box smearing each channel's box-mean weight lowers its
            // variance below the plain count.
            for (const auto v : plan.get_subband_variance_grid(l, sb)) {
                CHECK(v > 0.0F);
                CHECK(v <=
                      static_cast<float>(g.chan_end - g.chan_begin) + 1e-4F);
            }
        }
    }
    CHECK_THROWS_AS(plan.get_subband_variance_grid(niters + 1, 0),
                    std::out_of_range);
    CHECK_THROWS_AS(
        plan.get_subband_variance_grid(niters, pc.grids[niters].size()),
        std::out_of_range);
}

TEST_CASE("FDMT subband rows follow the dispersion law of their own band",
          "[fdmt][subband][cpu]") {
    // Each searched row's channel delays match the cold-plasma delay of its
    // DM between the subband's lower edge and each channel's upper edge (the
    // FDMT time reference and the box-smearing-off sample).
    const plans::FDMTPlan plan(1100.0F, 1714.0F, 48, 64, 455.6e-6F, 300, 0, 1,
                               "valid", 3);
    algorithms::FDMTBoxcar search(plan, BoxcarConfig(1), false);
    const auto& pc     = plan.get_container();
    SizeType staggered = 0;
    for (const auto& p : search.get_search_plans()) {
        staggered += p.staggered ? 1 : 0;
        const double f_low = p.f_low_mhz;
        for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
            std::vector<int64_t> shift(plan.get_nchans(), -1);
            leaf_shifts(pc, p.level, p.coord_offset + d, 0, shift);
            for (SizeType c = p.chan_begin; c < p.chan_end; ++c) {
                const double f_top =
                    plan.get_f_min() +
                    (plan.get_df() * static_cast<double>(c + 1));
                const double delay =
                    kDispConst * p.dm_grid[d] *
                    ((1.0 / (f_low * f_low)) - (1.0 / (f_top * f_top))) /
                    plan.get_tsamp();
                INFO("plane " << p.subband_level << "/" << p.subband_idx
                              << " dm " << p.dm_grid[d] << " chan " << c);
                CHECK(std::abs(static_cast<double>(shift[c]) - delay) <= 2.5);
            }
        }
    }
    CHECK(staggered == 1 + 3 + 7);
}

TEST_CASE("FDMT subband depth is validated", "[fdmt][subband][cpu]") {
    CHECK_NOTHROW(
        FDMTPlan(700.0F, 1200.0F, 5, 16, 0.001F, 8, 0, 1, "valid", 2));
    CHECK_THROWS_AS(
        FDMTPlan(700.0F, 1200.0F, 5, 16, 0.001F, 8, 0, 1, "valid", 3),
        std::invalid_argument);
    const FDMTPlan none(700.0F, 1200.0F, 5, 16, 0.001F, 8);
    CHECK(none.get_subband_depth() == 0);
    CHECK(none.get_max_fuse_levels() == none.get_niters());
}

TEST_CASE("FDMTPlan masked subband variance is the energy of the active "
          "channels",
          "[fdmt][subband][cpu]") {
    // Unit white noise through a linear map has variance sum(h^2). A row's
    // boxcar variance is therefore the sum, over the channels it reads, of
    // the energy of that channel's impulse response convolved with the
    // boxcar. A killed channel reads as zero, so the masked variance must
    // be the same sum over the active channels only.
    const SizeType nchans = 24;
    const SizeType nsamps = 160;
    const SizeType t0     = 8;
    const FDMTPlan probe(800.0F, 1400.0F, nchans, nsamps, 0.001F, 40);
    const FDMTPlan plan(800.0F, 1400.0F, nchans, nsamps, 0.001F, 40, 0, 1,
                        "valid", probe.get_niters() - 1);
    const auto& pc    = plan.get_container();
    const auto niters = plan.get_niters();
    std::vector<uint8_t> kill(nchans, 1);
    for (const SizeType c : {0, 5, 6, 13, 23}) { // both edges and interior
        kill[c] = 0;
    }
    const std::vector<uint8_t> none;
    for (const bool smear : {true, false}) {
        DYNAMIC_SECTION("smear=" << smear) {
            // resp[c][level][subband]: that view's rows for an impulse in
            // channel c at sample t0.
            std::vector<std::vector<std::vector<std::vector<float>>>> resp(
                nchans);
            std::vector<std::vector<SizeType>> view_nsamps(niters + 1);
            for (SizeType c = 0; c < nchans; ++c) {
                std::vector<float> input(nchans * nsamps, 0.0F);
                input[(c * nsamps) + t0] = 1.0F;
                FDMT fdmt(plan, smear, Exec::cpu(1), 1, 0, false, nsamps);
                std::vector<float> output(plan.get_dmt_size());
                fdmt.reset(input, output);
                resp[c].resize(niters + 1);
                for (SizeType l = 0;; ++l) {
                    for (SizeType s = 0; s < fdmt.num_subbands(); ++s) {
                        const auto view = fdmt.view_subband(s);
                        resp[c][l].emplace_back(view.data.begin(),
                                                view.data.end());
                        if (c == 0) {
                            view_nsamps[l].push_back(view.nsamps);
                        }
                    }
                    if (fdmt.is_finished()) {
                        break;
                    }
                    fdmt.advance();
                }
            }
            // Energy of y convolved with a boxcar of w samples.
            const auto energy = [](std::span<const float> y, SizeType w) {
                double e = 0;
                for (SizeType t = 0; t + 1 < y.size() + w; ++t) {
                    double v = 0;
                    for (SizeType k = 0; k < w && k <= t; ++k) {
                        if (t - k < y.size()) {
                            v += y[t - k];
                        }
                    }
                    e += v * v;
                }
                return e;
            };
            for (const SizeType w : {SizeType{1}, SizeType{3}, SizeType{8}}) {
                for (SizeType l = 0; l <= niters; ++l) {
                    for (SizeType s = 0; s < pc.grids[l].size(); ++s) {
                        const auto& g     = pc.grids[l][s];
                        const auto masked = plan.get_subband_variance_grid(
                            l, s, w, smear, kill);
                        const auto full =
                            plan.get_subband_variance_grid(l, s, w, smear);
                        REQUIRE(masked.size() == g.ndt);
                        const auto n = view_nsamps[l][s];
                        for (SizeType d = 0; d < g.ndt; ++d) {
                            double want_masked = 0;
                            double want_full   = 0;
                            for (SizeType c = 0; c < nchans; ++c) {
                                const std::span<const float> row(
                                    resp[c][l][s].data() + (d * n), n);
                                const double e = energy(row, w);
                                want_full += e;
                                want_masked += kill[c] != 0 ? e : 0.0;
                            }
                            CAPTURE(w, l, s, d);
                            CHECK(
                                masked[d] ==
                                Approx(want_masked).epsilon(1e-5).margin(1e-6));
                            CHECK(full[d] ==
                                  Approx(want_full).epsilon(1e-5).margin(1e-6));
                        }
                    }
                }
                // The root is the effective variance grid, mask included.
                CHECK(
                    plan.get_subband_variance_grid(niters, 0, w, smear, kill) ==
                    plan.get_effective_variance_grid(w, smear, kill));
                const auto grid =
                    plan.get_effective_variance_grid(w, smear, kill);
                for (SizeType d = 0; d < grid.size(); ++d) {
                    CHECK(plan.get_effective_variance(d, w, smear, kill) ==
                          grid[d]);
                    CHECK(plan.get_effective_sigma(d, w, smear, kill) ==
                          Approx(std::sqrt(grid[d])));
                }
            }
        }
    }
    const std::vector<uint8_t> wrong(nchans - 1, 1);
    CHECK_THROWS_AS(plan.get_effective_variance_grid(1, true, wrong),
                    std::invalid_argument);
    CHECK_THROWS_AS(plan.get_subband_variance_grid(niters, 0, 1, true, wrong),
                    std::invalid_argument);
    CHECK(plan.get_effective_variance_grid(1, true, none) ==
          plan.get_effective_variance_grid(1, true));
}

} // namespace dmt
