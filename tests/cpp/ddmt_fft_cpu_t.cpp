#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/fft.hpp"
#include "dmt/nufft.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::DDMT;
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

namespace {

constexpr float kFlo   = 1200.0F;
constexpr float kFhi   = 1600.0F;
constexpr float kTs    = 1.0E-3F;
constexpr SizeType kNc = 16;

std::vector<float> random_block(SizeType rows, SizeType nsamps, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(rows * nsamps);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

double rel_max_err(std::span<const float> a, std::span<const float> b) {
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    double ref  = 0.0;
    for (SizeType i = 0; i < a.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
        ref  = std::max(ref, static_cast<double>(std::abs(b[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

DDMTFFTOptions opts(DDMTFFTMethod m, SizeType guard = 16, double tol = 1e-6) {
    return DDMTFFTOptions{.method = m, .tolerance = tol, .guard = guard};
}

// Exact delay (samples) of trial dm in channel c, as DDMTFFT defines it.
double tau_of(double dm, SizeType c, SizeType nchans) {
    const double df = (static_cast<double>(kFhi) - kFlo) / nchans;
    const double f  = kFlo + (df * static_cast<double>(c));
    const double fr = kFlo + (df * static_cast<double>(nchans - 1));
    const double k  = static_cast<double>(kDispConst) / kTs;
    return dm * k * ((1.0 / (f * f)) - (1.0 / (fr * fr)));
}

// Double-precision single-transform reference of one cold-start call: the
// combined row is [guard zeros | block], one length-N transform with N the
// engine's choice for a call that fits one transform.
std::vector<float> reference_single(const std::vector<float>& wf,
                                    SizeType nchans,
                                    SizeType nsamps,
                                    std::span<const float> dms,
                                    const DDMTFFT& eng) {
    const auto guard = eng.get_options().guard;
    const auto ctx   = eng.get_max_delay() + guard;
    const auto total = guard + nsamps;
    REQUIRE(total > ctx);
    const auto n_out = total - ctx;
    const auto n     = utils::next_fft_size(n_out + ctx);
    const auto nb    = (n / 2) + 1;
    using C          = std::complex<double>;
    std::vector<C> tw(n);
    for (SizeType m = 0; m < n; ++m) {
        const double a = -2.0 * std::numbers::pi * static_cast<double>(m) /
                         static_cast<double>(n);
        tw[m]          = C(std::cos(a), std::sin(a));
    }
    // X_c(k) = sum_t row(t) W^(t k)
    std::vector<C> spec(nchans * nb);
    for (SizeType c = 0; c < nchans; ++c) {
        for (SizeType k = 0; k < nb; ++k) {
            C acc{};
            for (SizeType t = 0; t < std::min(n, total); ++t) {
                const double v =
                    t < guard ? 0.0 : wf[(c * nsamps) + (t - guard)];
                acc += v * tw[(t * k) % n];
            }
            spec[(c * nb) + k] = acc;
        }
    }
    std::vector<float> out(dms.size() * n_out);
    std::vector<C> y(nb);
    for (SizeType d = 0; d < dms.size(); ++d) {
        std::fill(y.begin(), y.end(), C{});
        for (SizeType c = 0; c < nchans; ++c) {
            const double tau = tau_of(dms[d], c, nchans);
            for (SizeType k = 0; k < nb; ++k) {
                const double ph = 2.0 * std::numbers::pi * tau *
                                  static_cast<double>(k) /
                                  static_cast<double>(n);
                y[k] += spec[(c * nb) + k] * C(std::cos(ph), std::sin(ph));
            }
        }
        // C2R: DC and Nyquist imaginary parts ignored.
        for (SizeType o = 0; o < n_out; ++o) {
            const auto t = guard + o;
            double v     = y[0].real();
            for (SizeType k = 1; k < nb; ++k) {
                const C e      = std::conj(tw[(t * k) % n]); // W^(-t k)
                const bool nyq = (2 * k == n);
                v += (nyq ? 1.0 : 2.0) *
                     (nyq ? (y[k].real() * e.real()) : (y[k] * e).real());
            }
            out[(d * n_out) + o] = static_cast<float>(v / n);
        }
    }
    return out;
}

} // namespace

TEST_CASE("DDMTFFT options, delays and streaming sizes",
          "[ddmt_fft_cpu][cpu]") {
    DDMTFFT lin(kFlo, kFhi, kNc, kTs, 40.0F, 0.5F, 0.0F, Exec::cpu(1));
    CHECK(lin.get_options().method == DDMTFFTMethod::kNUFFT);
    CHECK(lin.get_options().guard == 64);
    const double tmax = tau_of(40.0, 0, kNc);
    CHECK(lin.get_max_delay() == static_cast<SizeType>(std::ceil(tmax)) + 64);
    const auto d = lin.get_max_delay();
    CHECK(lin.get_output_nsamps(d + 10) == 10);
    CHECK(lin.get_output_nsamps(d) == 0);
    CHECK(lin.history_state_size() == kNc * (d + 64));
    const auto sug = lin.get_suggested_nsamps();
    CHECK(sug >= 4 * (d + 64));
    CHECK(utils::next_fft_size(sug + d + 64) == sug + d + 64);

    const std::vector<float> uneven = {0.0F, 1.0F, 3.0F, 7.0F};
    DDMTFFT ex(kFlo, kFhi, kNc, kTs, uneven, Exec::cpu(1));
    CHECK(ex.get_options().method == DDMTFFTMethod::kBrute);
    CHECK_THROWS_AS(DDMTFFT(kFlo, kFhi, kNc, kTs, uneven, Exec::cpu(1), 32, {},
                            1, opts(DDMTFFTMethod::kNUFFT)),
                    std::invalid_argument);
    CHECK_THROWS_AS(DDMTFFT(kFlo, kFhi, kNc, kTs, uneven, Exec::cpu(1), 32, {},
                            1, DDMTFFTOptions{.tolerance = 1.0}),
                    std::invalid_argument);
    // Non-finite input is rejected in every build type (Release uses
    // -ffast-math, which may fold NaN comparisons away).
    const std::vector<float> with_nan = {0.0F, std::nanf(""), 2.0F};
    CHECK_THROWS_AS(DDMTFFT(kFlo, kFhi, kNc, kTs, with_nan, Exec::cpu(1)),
                    std::invalid_argument);
    CHECK_THROWS_AS(DDMTFFT(kFlo, kFhi, kNc, kTs, uneven, Exec::cpu(1), 32, {},
                            1, DDMTFFTOptions{.tolerance = std::nan("")}),
                    std::invalid_argument);
    CHECK(algorithms::parse_ddmt_fft_method("nufft") == DDMTFFTMethod::kNUFFT);
    CHECK(algorithms::to_string(DDMTFFTMethod::kBrute) == "brute");
    CHECK_THROWS_AS(algorithms::parse_ddmt_fft_method("x"),
                    std::invalid_argument);

    // Wrong input size / warm-up
    std::vector<float> wf(kNc * 100, 0.0F);
    std::vector<float> bad(3);
    CHECK_THROWS_AS(ex.execute(wf, bad), std::invalid_argument);
    std::vector<float> hist(ex.history_state_size());
    CHECK_THROWS_AS(ex.save_history(hist), std::logic_error);
}

TEST_CASE("DDMTFFT matches a double-precision reference",
          "[ddmt_fft_cpu][cpu]") {
    const SizeType nsamps = 300;
    const auto method = GENERATE(DDMTFFTMethod::kBrute, DDMTFFTMethod::kNUFFT);
    CAPTURE(algorithms::to_string(method));
    std::vector<float> dms;
    for (int i = 0; i < 40; ++i) {
        dms.push_back(0.37F * static_cast<float>(i)); // uniform, fractional
    }
    DDMTFFT eng(kFlo, kFhi, kNc, kTs, dms, Exec::cpu(3), 32, {}, 1,
                opts(method, 16));
    auto wf          = random_block(kNc, nsamps, 11);
    const auto n_out = eng.get_output_nsamps(nsamps);
    std::vector<float> got(dms.size() * n_out);
    eng.execute(wf, got);
    const auto ref = reference_single(wf, kNc, nsamps, dms, eng);
    CHECK(rel_max_err(got, ref) < 1e-5);
}

TEST_CASE("DDMTFFT NUFFT tolerance and non-uniform brute force",
          "[ddmt_fft_cpu][cpu]") {
    const SizeType nsamps = 2000;
    auto wf               = random_block(kNc, nsamps, 3);
    DDMTFFT brute(kFlo, kFhi, kNc, kTs, 60.0F, 0.25F, 1.0F, Exec::cpu(4), 32,
                  {}, 1, opts(DDMTFFTMethod::kBrute));
    const auto n_out = brute.get_output_nsamps(nsamps);
    const auto ndm   = brute.get_plan().get_dm_arr().size();
    std::vector<float> ref(ndm * n_out);
    brute.execute(wf, ref);
    for (const double tol : {1e-3, 1e-6}) {
        DDMTFFT nu(kFlo, kFhi, kNc, kTs, 60.0F, 0.25F, 1.0F, Exec::cpu(4), 32,
                   {}, 1, opts(DDMTFFTMethod::kNUFFT, 16, tol));
        std::vector<float> got(ndm * n_out);
        nu.execute(wf, got);
        CAPTURE(tol);
        CHECK(rel_max_err(got, ref) < 10 * tol + 2e-6);
    }
    // A non-uniform (Levin) grid runs brute force and matches the reference.
    const plans::LevinConfig lev{
        .dm_start = 0.0F, .dm_end = 30.0F, .pulse_width = 2e-3F, .tol = 1.2F};
    DDMTFFT lv(kFlo, kFhi, kNc, kTs, lev, Exec::cpu(2), 32, {}, 1,
               opts(DDMTFFTMethod::kAuto, 16));
    CHECK(lv.get_options().method == DDMTFFTMethod::kBrute);
    const auto lv_dms = lv.get_plan().get_dm_arr();
    std::vector<float> lv_out(lv_dms.size() * lv.get_output_nsamps(300));
    auto wf2 = random_block(kNc, 300, 5);
    lv.execute(wf2, lv_out);
    const auto lv_ref = reference_single(wf2, kNc, 300, lv_dms, lv);
    CHECK(rel_max_err(lv_out, lv_ref) < 1e-5);
}

TEST_CASE("Piecewise-uniform Levin grid", "[ddmt_fft_cpu][cpu]") {
    // L-band-like set-up whose Levin step grows 20x over the range.
    const float dm_end = 2000.0F;
    const float ts     = 6.4e-5F;
    const float width  = 1e-4F;
    const SizeType nc  = 1024;
    const auto levin   = utils::generate_levin_dm_grid(0.0F, dm_end, ts, width,
                                                       kFlo, kFhi, nc, 1.2F);
    const auto pw      = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, dm_end, ts, width, kFlo, kFhi, nc, 1.2F);
    REQUIRE(pw.size() >= levin.size());
    CHECK(pw.size() < 2 * levin.size());
    CHECK(pw.front() == levin.front());
    CHECK(pw.back() >= dm_end);
    CHECK(std::ranges::is_sorted(pw));
    // Never coarser than Levin: every step <= the Levin step at its start.
    for (SizeType i = 0; i + 1 < pw.size(); ++i) {
        const auto it = std::ranges::upper_bound(levin, pw[i]);
        if (it == levin.end() || it == levin.begin()) {
            continue;
        }
        const auto j = static_cast<SizeType>(it - levin.begin()) - 1;
        CHECK(pw[i + 1] - pw[i] <= (levin[j + 1] - levin[j]) * 1.0001F);
    }
    // DDMTFFT finds the uniform runs (one NUFFT each).
    const plans::LevinConfig cfg{.dm_start          = 0.0F,
                                 .dm_end            = dm_end,
                                 .pulse_width       = width,
                                 .tol               = 1.2F,
                                 .piecewise_uniform = true};
    DDMTFFT eng(kFlo, kFhi, nc, ts, cfg, Exec::cpu(2));
    CHECK(eng.get_plan().get_dm_arr().size() == pw.size());
    CHECK(eng.method_used() == "piecewise_nufft");
    const plans::LevinConfig plain{
        .dm_start = 0.0F, .dm_end = dm_end, .pulse_width = width, .tol = 1.2F};
    CHECK(plans::DDMTPlan(kFlo, kFhi, nc, ts, plain).get_dm_arr().size() ==
          levin.size());
}

TEST_CASE("DDMTFFT piecewise NUFFT matches brute force",
          "[ddmt_fft_cpu][cpu]") {
    // Two uniform runs (40 and 64 trials) around a few scattered trials,
    // which go to brute force.
    std::vector<float> dms;
    for (int i = 0; i < 40; ++i) {
        dms.push_back(0.5F * static_cast<float>(i));
    }
    for (const float v : {20.3F, 21.7F, 23.0F}) {
        dms.push_back(v);
    }
    for (int i = 0; i < 64; ++i) {
        dms.push_back(24.0F + (0.75F * static_cast<float>(i)));
    }
    const SizeType nsamps = 1500;
    auto wf               = random_block(kNc, nsamps, 17);
    DDMTFFT brute(kFlo, kFhi, kNc, kTs, dms, Exec::cpu(3), 32, {}, 1,
                  opts(DDMTFFTMethod::kBrute));
    DDMTFFT mixed(kFlo, kFhi, kNc, kTs, dms, Exec::cpu(3), 32, {}, 1,
                  opts(DDMTFFTMethod::kAuto));
    CHECK(brute.method_used() == "brute");
    CHECK(mixed.method_used() == "piecewise_nufft");
    CHECK(mixed.get_options().method == DDMTFFTMethod::kNUFFT);
    const auto n_out = brute.get_output_nsamps(nsamps);
    std::vector<float> ref(dms.size() * n_out);
    std::vector<float> got(dms.size() * n_out);
    brute.execute(wf, ref);
    mixed.execute(wf, got);
    CHECK(rel_max_err(got, ref) < 1e-5);
    // Second (warm) call too.
    auto wf2 = random_block(kNc, nsamps, 18);
    std::vector<float> ref2(dms.size() * brute.get_output_nsamps(nsamps));
    std::vector<float> got2(ref2.size());
    brute.execute(wf2, ref2);
    mixed.execute(wf2, got2);
    CHECK(rel_max_err(got2, ref2) < 1e-5);
    // A linear grid is one run.
    DDMTFFT lin(kFlo, kFhi, kNc, kTs, 40.0F, 0.5F, 0.0F, Exec::cpu(1));
    CHECK(lin.method_used() == "nufft");
}

TEST_CASE("DDMTFFT streaming converges to one long call as the guard grows",
          "[ddmt_fft_cpu][cpu]") {
    const SizeType nblk = 4;
    const SizeType blk  = 1500;
    auto wf             = random_block(kNc, nblk * blk, 21);
    double err_prev     = 1e9;
    for (const SizeType guard : {8, 64, 512}) {
        DDMTFFT one(kFlo, kFhi, kNc, kTs, 30.0F, 1.0F, 0.0F, Exec::cpu(2), 32,
                    {}, 1, opts(DDMTFFTMethod::kBrute, guard));
        DDMTFFT str(kFlo, kFhi, kNc, kTs, 30.0F, 1.0F, 0.0F, Exec::cpu(2), 32,
                    {}, 1, opts(DDMTFFTMethod::kBrute, guard));
        const auto ndm = one.get_plan().get_dm_arr().size();
        const auto n1  = one.get_output_nsamps(nblk * blk);
        std::vector<float> full(ndm * n1);
        one.execute(wf, full);

        std::vector<float> joined(ndm * n1, 0.0F);
        SizeType o = 0;
        for (SizeType b = 0; b < nblk; ++b) {
            std::vector<float> part(kNc * blk);
            for (SizeType c = 0; c < kNc; ++c) {
                std::copy_n(
                    wf.begin() + static_cast<std::ptrdiff_t>((c * nblk * blk) +
                                                             (b * blk)),
                    blk, part.begin() + static_cast<std::ptrdiff_t>(c * blk));
            }
            const auto nb = str.get_output_nsamps(blk);
            std::vector<float> out(ndm * nb);
            str.execute(part, out);
            for (SizeType d = 0; d < ndm; ++d) {
                std::copy_n(
                    out.begin() + static_cast<std::ptrdiff_t>(d * nb), nb,
                    joined.begin() + static_cast<std::ptrdiff_t>((d * n1) + o));
            }
            o += nb;
        }
        REQUIRE(o == n1);
        const double err = rel_max_err(joined, full);
        CAPTURE(guard, err);
        CHECK(err < err_prev);
        err_prev = err;
    }
    CHECK(err_prev < 0.02);
}

TEST_CASE("DDMTFFT packed, time-major, kill mask, beams and history",
          "[ddmt_fft_cpu][cpu]") {
    const SizeType nsamps = 800;
    const auto method = GENERATE(DDMTFFTMethod::kBrute, DDMTFFTMethod::kNUFFT);
    CAPTURE(algorithms::to_string(method));
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> packed(kNc * nsamps);
    std::vector<float> asfloat(kNc * nsamps);
    for (SizeType i = 0; i < packed.size(); ++i) {
        packed[i]  = static_cast<uint8_t>(byte(rng));
        asfloat[i] = static_cast<float>(packed[i]);
    }

    SECTION("packed == float") {
        DDMTFFT p8(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 8, {},
                   1, opts(method));
        DDMTFFT pf(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 32,
                   {}, 1, opts(method));
        const auto n =
            p8.get_plan().get_dm_arr().size() * p8.get_output_nsamps(nsamps);
        std::vector<float> a(n);
        std::vector<float> b(n);
        p8.execute(packed, nsamps, a);
        pf.execute(asfloat, b);
        test::require_exact(a, b);

        // time-major (nsamps, nchans) of the same data
        std::vector<uint8_t> tm(nsamps * kNc);
        for (SizeType c = 0; c < kNc; ++c) {
            for (SizeType s = 0; s < nsamps; ++s) {
                tm[(s * kNc) + c] = packed[(c * nsamps) + s];
            }
        }
        DDMTFFT t8(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 8, {},
                   1, opts(method));
        std::vector<float> c(n);
        t8.execute_time_major(tm, nsamps, c);
        test::require_exact(c, a);
        CHECK_THROWS_AS(pf.execute(packed, nsamps, a), std::invalid_argument);
        CHECK_THROWS_AS(p8.execute(asfloat, b), std::invalid_argument);
    }

    SECTION("kill mask == zeroed channels") {
        std::vector<uint8_t> mask(kNc, 1);
        mask[3]  = 0;
        mask[10] = 0;
        DDMTFFT masked(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2),
                       32, mask, 1, opts(method));
        DDMTFFT plain(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 32,
                      {}, 1, opts(method));
        auto zeroed = asfloat;
        for (const SizeType c : {3U, 10U}) {
            std::fill_n(zeroed.begin() +
                            static_cast<std::ptrdiff_t>(c * nsamps),
                        nsamps, 0.0F);
        }
        const auto n = masked.get_plan().get_dm_arr().size() *
                       masked.get_output_nsamps(nsamps);
        std::vector<float> a(n);
        std::vector<float> b(n);
        masked.execute(asfloat, a);
        plain.execute(zeroed, b);
        CHECK(rel_max_err(a, b) < 1e-5);
    }

    SECTION("beams == independent engines") {
        const SizeType nbeams = 3;
        DDMTFFT multi(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(3), 32,
                      {}, nbeams, opts(method));
        const auto ndm  = multi.get_plan().get_dm_arr().size();
        const auto nout = multi.get_output_nsamps(nsamps);
        auto wf         = random_block(nbeams * kNc, nsamps, 77);
        std::vector<float> all(nbeams * ndm * nout);
        multi.execute(wf, all);
        for (SizeType b = 0; b < nbeams; ++b) {
            DDMTFFT single(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F,
                           Exec::cpu(1), 32, {}, 1, opts(method));
            std::vector<float> one(ndm * nout);
            single.execute(std::span<const float>(wf).subspan(b * kNc * nsamps,
                                                              kNc * nsamps),
                           one);
            test::require_exact(
                std::vector<float>(
                    all.begin() + static_cast<std::ptrdiff_t>(b * ndm * nout),
                    all.begin() +
                        static_cast<std::ptrdiff_t>((b + 1) * ndm * nout)),
                one);
        }
    }

    SECTION("save/load history resumes the stream; threads do not matter") {
        DDMTFFT a(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(1), 32, {},
                  1, opts(method));
        DDMTFFT b(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(5), 32, {},
                  1, opts(method));
        const auto ndm = a.get_plan().get_dm_arr().size();
        for (unsigned blk = 0; blk < 2; ++blk) {
            auto wf = random_block(kNc, nsamps, 100 + blk);
            std::vector<float> o(ndm * a.get_output_nsamps(nsamps));
            a.execute(wf, o);
        }
        std::vector<float> hist(a.history_state_size());
        a.save_history(hist);
        b.load_history(hist);
        auto wf = random_block(kNc, nsamps, 102);
        CHECK(a.get_output_nsamps(nsamps) == nsamps);
        std::vector<float> oa(ndm * nsamps);
        std::vector<float> ob(ndm * nsamps);
        a.execute(wf, oa);
        b.execute(wf, ob);
        test::require_exact(oa, ob);
        a.reset_history();
        CHECK(a.get_output_nsamps(nsamps) == nsamps - a.get_max_delay());
    }
}

// A pulse dispersed with exact (fractional) delays: DDMTFFT's phase-ramp
// shifts realign it, DDMT's rounded delays smear it by up to half a sample
// per channel.
TEST_CASE("DDMTFFT packed low-bit streams match float", "[ddmt_fft_cpu][cpu]") {
    // Odd block sizes put every sub-byte sample offset at a block start.
    const SizeType nbits = GENERATE(1, 2, 4, 8, 16);
    CAPTURE(nbits);
    const std::vector<SizeType> blocks = {301, 157, 211};
    DDMTFFT pk(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), nbits, {},
               1, opts(DDMTFFTMethod::kAuto));
    DDMTFFT fl(kFlo, kFhi, kNc, kTs, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 32, {}, 1,
               opts(DDMTFFTMethod::kAuto));
    std::mt19937 rng(21);
    const auto vmax = static_cast<int>(bit_pack_utils::max_sample_value(nbits));
    std::uniform_int_distribution<int> val(0, vmax);
    const auto ndm = pk.get_plan().get_dm_arr().size();
    for (const auto ns : blocks) {
        const auto rb = bit_pack_utils::packed_row_bytes(ns, nbits);
        std::vector<uint8_t> packed(kNc * rb, 0);
        std::vector<float> asfloat(kNc * ns);
        for (SizeType c = 0; c < kNc; ++c) {
            for (SizeType t = 0; t < ns; ++t) {
                const auto v          = static_cast<uint32_t>(val(rng));
                asfloat[(c * ns) + t] = static_cast<float>(v);
                uint8_t* row          = packed.data() + (c * rb);
                switch (nbits) {
                case 1:
                    bit_pack_utils::write_packed_sample<1>(row, t, v);
                    break;
                case 2:
                    bit_pack_utils::write_packed_sample<2>(row, t, v);
                    break;
                case 4:
                    bit_pack_utils::write_packed_sample<4>(row, t, v);
                    break;
                case 8:
                    bit_pack_utils::write_packed_sample<8>(row, t, v);
                    break;
                default:
                    bit_pack_utils::write_packed_sample<16>(row, t, v);
                    break;
                }
            }
        }
        const auto n = ndm * pk.get_output_nsamps(ns);
        std::vector<float> a(n);
        std::vector<float> b(n);
        pk.execute(packed, ns, a);
        fl.execute(asfloat, b);
        test::require_exact(a, b);
    }
}

TEST_CASE("DDMTFFT recovers fractional-delay pulses better than DDMT",
          "[ddmt_fft_cpu][cpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 1024;
    const double dm_true  = 17.3;
    const double t0       = 300.3;
    const double width    = 1.0; // samples (Gaussian sigma; ~band-limited)
    std::vector<float> wf(nchans * nsamps, 0.0F);
    for (SizeType c = 0; c < nchans; ++c) {
        const double tc = t0 + tau_of(dm_true, c, nchans);
        for (SizeType t = 0; t < nsamps; ++t) {
            const double z       = (static_cast<double>(t) - tc) / width;
            wf[(c * nsamps) + t] = static_cast<float>(std::exp(-0.5 * z * z));
        }
    }
    const std::vector<float> dms = {static_cast<float>(dm_true)};
    DDMTFFT fdd(kFlo, kFhi, nchans, kTs, dms, Exec::cpu(2), 32, {}, 1,
                opts(DDMTFFTMethod::kBrute, 32));
    DDMT ddmt(kFlo, kFhi, nchans, kTs, dms, Exec::cpu(2));
    std::vector<float> a(fdd.get_output_nsamps(nsamps));
    std::vector<float> b(ddmt.get_output_nsamps(nsamps));
    fdd.execute(wf, a);
    ddmt.execute(wf, b);
    const float pa = *std::ranges::max_element(a);
    const float pb = *std::ranges::max_element(b);
    // Ideal: every channel adds its sampled Gaussian peak at the same time.
    const double ideal = static_cast<double>(nchans) *
                         std::exp(-0.5 * std::pow(0.3 / width, 2.0));
    CAPTURE(pa, pb, ideal);
    CHECK(pa > pb);
    CHECK(pa > 0.97 * ideal);
}

TEST_CASE("NUFFT type 1 matches the direct sum", "[ddmt_fft_cpu][cpu]") {
    const SizeType m  = GENERATE(1, 7, 64, 513);
    const double tol  = GENERATE(1e-3, 1e-6);
    const SizeType np = 300;
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> ux(-3.0, 3.0);
    std::normal_distribution<float> ua(0.0F, 1.0F);
    std::vector<double> x(np);
    std::vector<ComplexType> a(np);
    double l1 = 0.0;
    for (SizeType j = 0; j < np; ++j) {
        x[j] = ux(rng);
        a[j] = ComplexType(ua(rng), ua(rng));
        l1 += std::abs(std::complex<double>(a[j]));
    }
    nufft::Type1Plan plan(m, tol);
    std::vector<ComplexType> f(m);
    utils::FFTVector<ComplexType> scratch;
    plan.execute(x, a, f, scratch);
    double worst = 0.0;
    for (SizeType d = 0; d < m; ++d) {
        std::complex<double> ref{};
        for (SizeType j = 0; j < np; ++j) {
            const double ph =
                2.0 * std::numbers::pi * static_cast<double>(d) * x[j];
            ref += std::complex<double>(a[j]) *
                   std::complex<double>(std::cos(ph), std::sin(ph));
        }
        worst =
            std::max(worst, std::abs(std::complex<double>(f[d]) - ref) / l1);
    }
    CAPTURE(m, tol, worst);
    CHECK(worst < 3 * tol + 1e-6);
}

} // namespace dmt
