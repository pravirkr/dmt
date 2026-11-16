#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/common/plans.hpp"

#include "test_random.hpp"

// FDMT level-0 boxes for coherently dedispersed channels (FDMTPlan chan_dm):
// the plan's boxes, the CPU tree path against the box formula, FDMTFFT with
// integer delays against FDMT, and an FDMTBoxcar search on a chan_dm plan.

namespace dmt {

using algorithms::FDMT;
using algorithms::FDMTBoxcar;
using algorithms::FDMTFFT;

namespace {

constexpr float kFMin  = 704.0F;
constexpr float kFMax  = 1216.0F;
constexpr float kTsamp = 256e-6F;

std::vector<IndexType> grid(IndexType lo, IndexType hi, IndexType step) {
    std::vector<IndexType> g;
    for (IndexType dt = lo; dt <= hi; dt += step) {
        g.push_back(dt);
    }
    return g;
}

// Output rows t0 .. t0 + n of FDMT with box smearing from the plan's trace
// and box lengths: sum over channels of w(L_c) * x[c, t + trace_c .. +L_c).
std::vector<double> box_formula(const plans::FDMTPlan& plan,
                                const std::vector<float>& x,
                                SizeType total,
                                SizeType t0,
                                SizeType n) {
    const auto ndm    = plan.get_dmt_ndms();
    const auto nchans = plan.get_nchans();
    const auto smear  = plan.get_smearing_grid_final();
    const auto w      = plan.get_box_weights();
    const double unit =
        std::ldexp(1.0, -static_cast<int>(plan.get_box_fraction_bits()));
    std::vector<double> out(ndm * n, 0.0);
    for (SizeType d = 0; d < ndm; ++d) {
        const auto start = plan.trace_dm(d);
        for (SizeType c = 0; c < nchans; ++c) {
            const auto s     = static_cast<SizeType>(smear[(d * nchans) + c]);
            const double wgt = static_cast<double>(w[s]) * unit;
            for (SizeType i = 0; i < n; ++i) {
                const auto t = static_cast<IndexType>(t0 + i) + start[c];
                double sum   = 0.0;
                for (SizeType j = 0; j <= s; ++j) {
                    sum += x[(c * total) + static_cast<SizeType>(t) + j];
                }
                out[(d * n) + i] += sum * wgt;
            }
        }
    }
    return out;
}

} // namespace

TEST_CASE("FDMTPlan chan_dm level-0 boxes", "[fdmt_chan_dm][cpu]") {
    const auto g = grid(-60, 200, 4);
    const plans::FDMTPlan plain(kFMin, kFMax, 64, 256, kTsamp, g);
    REQUIRE(plain.get_level0_boxes().empty());
    REQUIRE(plain.get_level0_causal_delay() == 0);
    REQUIRE(plain.get_chan_dm() == 0.0);

    for (const double chan_dm : {413.0, -150.0}) {
        DYNAMIC_SECTION("chan_dm=" << chan_dm) {
            const plans::FDMTPlan plan(kFMin, kFMax, 64, 256, kTsamp, g,
                                       "valid", 0, chan_dm);
            const auto& pc    = plan.get_container();
            const auto& boxes = plan.get_level0_boxes();
            REQUIRE(boxes.size() == pc.state_shape[0].ncoords);
            REQUIRE(pc.level0_order.size() == boxes.size());
            int32_t min_lag = std::numeric_limits<int32_t>::max();
            SizeType reach  = 0;
            for (const auto& gr : pc.grids[0]) {
                const auto* b = boxes.data() + gr.coord_offset;
                const auto* o = pc.level0_order.data() + gr.coord_offset;
                for (SizeType i = 0; i < gr.ndt; ++i) {
                    REQUIRE(b[i].lag >= 0);
                    REQUIRE(b[i].s >= 0);
                    REQUIRE(b[i].point >= b[i].lag);
                    REQUIRE(b[i].point <= b[i].lag + b[i].s + 1);
                    min_lag = std::min(min_lag, b[i].lag);
                    reach   = std::max<SizeType>(reach, b[i].lag + b[i].s);
                    if (i > 0) {
                        // Sweep order: ascending (lag, s).
                        const auto& p = b[o[i - 1]];
                        const auto& q = b[o[i]];
                        REQUIRE(
                            (p.lag < q.lag || (p.lag == q.lag && p.s <= q.s)));
                    }
                }
            }
            // The causal delay makes every read causal, and no larger than
            // a sample beyond what the boxes need.
            REQUIRE(min_lag >= 0);
            REQUIRE(min_lag <= 1);
            REQUIRE(pc.state_shape[0].dt_max >= reach);
            REQUIRE(plan.get_history_size() ==
                    plan.get_nchans() * pc.state_shape[0].dt_max);
        }
    }
    REQUIRE_THROWS_AS(
        plans::FDMTPlan(kFMin, kFMax, 64, 256, kTsamp, g, "full", 0, 413.0),
        std::invalid_argument);
    const volatile std::uint64_t inf_bits = 0x7FF0000000000000ULL;
    const double inf = std::bit_cast<double>(std::uint64_t{inf_bits});
    REQUIRE_THROWS_AS(
        plans::FDMTPlan(kFMin, kFMax, 64, 256, kTsamp, g, "valid", 0, inf),
        std::invalid_argument);
}

TEST_CASE("FDMT chan_dm rows follow the box formula", "[fdmt_chan_dm][cpu]") {
    const SizeType nchans = 48;
    const SizeType block  = 256;
    const SizeType blocks = 4;
    const SizeType total  = block * blocks;
    for (const double chan_dm : {413.0, -150.0}) {
        for (const SizeType fuse : {SizeType{0}, SizeType{1}, SizeType{3}}) {
            DYNAMIC_SECTION("chan_dm=" << chan_dm << " fuse=" << fuse) {
                const plans::FDMTPlan plan(kFMin, kFMax, nchans, block, kTsamp,
                                           grid(-40, 300, 3), "valid", 0,
                                           chan_dm);
                FDMT fdmt(plan, true, Exec::cpu(2), 1,
                          fuse == 0 ? algorithms::kFDMTAutoFuse : fuse);
                std::vector<float> x(nchans * total);
                test_random::fill_normal<float>(x, 99);
                const auto ndm = plan.get_dmt_ndms();
                std::vector<float> out(ndm * total);
                std::vector<float> in(nchans * block);
                std::vector<float> dmt(plan.get_dmt_size());
                for (SizeType b = 0; b < blocks; ++b) {
                    for (SizeType c = 0; c < nchans; ++c) {
                        std::copy_n(x.data() + (c * total) + (b * block), block,
                                    in.data() + (c * block));
                    }
                    fdmt.execute(in, dmt);
                    for (SizeType d = 0; d < ndm; ++d) {
                        std::copy_n(dmt.data() + (d * block), block,
                                    out.data() + (d * total) + (b * block));
                    }
                }
                const auto t0 = plan.get_startup_samples() + 1;
                REQUIRE(t0 < total);
                const auto n   = total - t0;
                const auto ref = box_formula(plan, x, total, t0, n);
                double peak    = 0.0;
                for (const auto v : ref) {
                    peak = std::max(peak, std::abs(v));
                }
                for (SizeType d = 0; d < ndm; ++d) {
                    for (SizeType i = 0; i < n; ++i) {
                        REQUIRE(std::abs(out[(d * total) + t0 + i] -
                                         ref[(d * n) + i]) <= 2e-5 * peak);
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTFFT integer delays match FDMT with chan_dm",
          "[fdmt_chan_dm][cpu]") {
    const SizeType nchans = 48;
    const SizeType block  = 512;
    for (const bool smear : {true, false}) {
        DYNAMIC_SECTION("smear=" << smear) {
            const plans::FDMTPlan plan(kFMin, kFMax, nchans, block, kTsamp,
                                       grid(-40, 300, 3), "valid", 0, 413.0);
            FDMT fdmt(plan, smear);
            FDMTFFT fft(plan, smear, Exec::cpu(1), 1, false);
            std::vector<float> in(nchans * block);
            std::vector<float> a(plan.get_dmt_size());
            std::vector<float> b(plan.get_dmt_size());
            const auto ndm = plan.get_dmt_ndms();
            for (SizeType k = 0; k < 3; ++k) {
                test_random::fill_normal<float>(in, 7 + k);
                fdmt.execute(in, a);
                fft.execute(in, b);
                if ((k + 1) * block <= plan.get_startup_samples()) {
                    continue;
                }
                double peak = 0.0;
                double err  = 0.0;
                for (SizeType i = 0; i < ndm * block; ++i) {
                    peak = std::max(peak, std::abs(static_cast<double>(a[i])));
                    err  = std::max(err, std::abs(static_cast<double>(a[i]) -
                                                  static_cast<double>(b[i])));
                }
                REQUIRE(err <= 2e-5 * peak);
            }
        }
    }
}

TEST_CASE("FDMTBoxcar searches a chan_dm plan", "[fdmt_chan_dm][cpu]") {
    // A 4-sample burst at DM = chan_dm, coherently dedispersed about each
    // channel's centre (so it is sharp within every channel), in noise.
    const SizeType nchans = 128;
    const SizeType block  = 4096;
    const SizeType blocks = 4;
    const double chan_dm  = 413.0;
    const double k_dm     = 1.0 / 2.41e-4;
    const plans::FDMTPlan plan(kFMin, kFMax, nchans, block, kTsamp,
                               grid(0, 10000, 8), "valid", 0, chan_dm);
    BoxcarConfig cfg;
    cfg.max_boxcar    = 17;
    cfg.wtsp          = 1.5;
    cfg.detect_thresh = 8.0F;
    cfg.peak_mode     = PeakFindingMode::kContiguousPeakAcrossWidths;
    FDMTBoxcar search(plan, cfg, true, Exec::cpu(2));
    const SizeType total = block * blocks;
    std::vector<float> x(nchans * total);
    test_random::fill_normal<float>(x, 5);
    const double df  = (kFMax - kFMin) / static_cast<double>(nchans);
    const double amp = 48.0 / std::sqrt(static_cast<double>(nchans) * 4.0);
    for (SizeType c = 0; c < nchans; ++c) {
        const double f_c = kFMin + ((static_cast<double>(c) + 0.5) * df);
        const double t =
            600.0 + (k_dm * chan_dm *
                     ((1.0 / (f_c * f_c)) -
                      (1.0 / (kFMax * static_cast<double>(kFMax)))) /
                     kTsamp);
        const auto k = static_cast<SizeType>(std::floor(t));
        for (SizeType j = 0; j < 4; ++j) {
            x[(c * total) + k + j] += static_cast<float>(amp);
        }
    }
    std::vector<float> in(nchans * block);
    float best_snr    = 0.0F;
    uint32_t best_row = 0;
    const auto take   = [&](const CandidateResultView& r) {
        for (const auto& cand : r.candidates) {
            if (cand.snr > best_snr) {
                best_snr = cand.snr;
                best_row = cand.dm_idx;
            }
        }
    };
    for (SizeType b = 0; b < blocks; ++b) {
        for (SizeType c = 0; c < nchans; ++c) {
            std::copy_n(x.data() + (c * total) + (b * block), block,
                        in.data() + (c * block));
        }
        take(search.execute(in));
    }
    take(search.flush());
    const auto dms = plan.get_dm_grid_final();
    REQUIRE(best_snr > 36.0F);
    REQUIRE(best_snr < 52.0F);
    REQUIRE(std::abs(dms[best_row] - chan_dm) < 20.0);
}

TEST_CASE("FDMT chan_dm: box weights follow level0_max_box_s, not extent",
          "[fdmt_chan_dm][cpu]") {
    const plans::FDMTPlan plan(kFMin, kFMax, 64, 512, kTsamp,
                               grid(2100, 2240, 8), "valid", 0, 100.0);
    const auto& pc     = plan.get_container();
    const auto weights = plan.get_box_weights();
    REQUIRE(weights.size() == pc.level0_max_box_s + 1);
    REQUIRE(weights.size() < pc.state_shape[0].dt_max + 1);
}

TEST_CASE("FDMTFFT fractional chan_dm keeps wide-channel coherent pulses sharp",
          "[fdmt_chan_dm][cpu]") {
    const SizeType nchans = 4;
    const SizeType nsamps = 4096;
    const float f_min     = 300.0F;
    const float f_max     = 600.0F;
    const float tsamp     = 1e-3F;
    const double chan_dm  = 20.0;
    const double k_ts     = (1.0 / 2.41e-4) / static_cast<double>(tsamp);
    const SizeType dt_c   = static_cast<SizeType>(std::llround(
        k_ts * chan_dm * ((1.0 / (f_min * f_min)) - (1.0 / (f_max * f_max)))));
    const plans::FDMTPlan plan(f_min, f_max, nchans, nsamps, tsamp,
                               grid(static_cast<IndexType>(dt_c - 8),
                                    static_cast<IndexType>(dt_c + 8), 2),
                               "valid", 0, chan_dm);
    FDMTFFT fft(plan, true, Exec::cpu(1), 1, true);
    std::vector<float> wf(nchans * nsamps, 0.0F);
    const double df = static_cast<double>(f_max - f_min) / nchans;
    for (SizeType c = 0; c < nchans; ++c) {
        const double f_c = f_min + ((static_cast<double>(c) + 0.5) * df);
        const double t =
            1500.0 +
            (k_ts * chan_dm * ((1.0 / (f_c * f_c)) - (1.0 / (f_max * f_max))));
        for (SizeType i = 1400; i < 2300; ++i) {
            const double d = (static_cast<double>(i) - t) / 1.5;
            wf[(c * nsamps) + i] += static_cast<float>(std::exp(-0.5 * d * d));
        }
    }
    std::vector<float> out(plan.get_dmt_size());
    fft.execute(wf, out);
    const auto dts     = plan.get_dt_grid_final();
    SizeType best_row  = 0;
    IndexType best_err = std::numeric_limits<IndexType>::max();
    for (SizeType r = 0; r < dts.size(); ++r) {
        const IndexType err = std::abs(dts[r] - static_cast<IndexType>(dt_c));
        if (err < best_err) {
            best_err = err;
            best_row = r;
        }
    }
    float peak = 0.0F;
    for (SizeType i = plan.get_startup_samples(); i < nsamps; ++i) {
        peak = std::max(peak, out[(best_row * nsamps) + i]);
    }
    REQUIRE(peak > 3.7F);
}

TEST_CASE("FDMTPlan subband pruning skips single-channel copy subbands",
          "[grid][subband][cpu]") {
    const plans::FDMTPlan plan(1000.0F, 1500.0F, 5, 256, 1e-3F, 16, 0, 1,
                               "valid", 2, 0.0, 0.1F);
    REQUIRE(plan.get_nchans() == 5);
}

} // namespace dmt
