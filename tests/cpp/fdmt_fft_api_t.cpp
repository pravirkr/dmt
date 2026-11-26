#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"

#include "dmt/engines.hpp"

#include "test_helpers.hpp"
#include "test_random.hpp"

// FDMTFFT packed and time-major input, kill mask and streaming history, and
// the shared overlap-save segment rule (forced by the testing hook), on the
// CPU. The GPU engine is checked against these in fft_engines_cuda_t.cu.

namespace dmt {

using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;
using algorithms::FDMTFFT;

namespace {

constexpr float kFMin   = 1100.0F;
constexpr float kFMax   = 1500.0F;
constexpr float kTsamp  = 0.001F;
constexpr SizeType kNch = 64;
constexpr SizeType kNs  = 1024;

std::vector<float> noise(SizeType n, unsigned seed) {
    return dmt::test_random::normal<float>(n, seed);
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    double ref  = 0.0;
    for (SizeType i = 0; i < a.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
        ref  = std::max(ref, static_cast<double>(std::abs(b[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

} // namespace

TEST_CASE("FDMTFFT packed and time-major input match float input",
          "[cpu][fdmt_fft_cpu]") {
    for (const auto* mode : {"valid", "full", "roll"}) {
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            CAPTURE(mode, nbits);
            const auto vals = dmt::test_random::uniform_int<uint32_t>(
                2 * kNch * kNs, nbits, 0,
                (nbits == 16) ? 65535U : ((1U << nbits) - 1U));
            const std::vector<float> wf(vals.begin(), vals.end());
            // Time-major values: (beam, t, chan).
            std::vector<uint32_t> tm(vals.size());
            for (SizeType b = 0; b < 2; ++b) {
                for (SizeType c = 0; c < kNch; ++c) {
                    for (SizeType t = 0; t < kNs; ++t) {
                        tm[(((b * kNs) + t) * kNch) + c] =
                            vals[(((b * kNch) + c) * kNs) + t];
                    }
                }
            }
            const auto packed    = test::pack_rows(vals, 2 * kNch, kNs, nbits);
            const auto tm_packed = test::pack_rows(tm, 2 * kNs, kNch, nbits);
            const auto make      = [&] {
                return FDMTFFT(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true,
                               mode, Exec::cpu(), 2);
            };
            auto f_float  = make();
            auto f_packed = make();
            auto f_tm     = make();
            const auto n  = 2 * f_float.get_plan().get_dmt_size();
            std::vector<float> a(n);
            std::vector<float> b(n);
            std::vector<float> c(n);
            for (int blk = 0; blk < 2; ++blk) { // valid: the history too
                f_float.execute(wf, a);
                f_packed.execute(packed, nbits, b);
                f_tm.execute_time_major(tm_packed, nbits, c);
                CHECK(max_rel(b, a) < 1e-6);
                CHECK(max_rel(c, a) < 1e-6);
            }
        }
    }
}

TEST_CASE("FDMTFFT packed stepper matches packed execute",
          "[cpu][fdmt_fft_cpu]") {
    const auto vals =
        dmt::test_random::uniform_int<uint32_t>(kNch * kNs, 9, 0, 15);
    const auto packed = test::pack_rows(vals, kNch, kNs, 4);
    FDMTFFT a(kFMin, kFMax, kNch, kNs, kTsamp, 48);
    FDMTFFT b(kFMin, kFMax, kNch, kNs, kTsamp, 48);
    const auto n = a.get_plan().get_dmt_size();
    std::vector<float> ra(n);
    std::vector<float> rb(n);
    for (int blk = 0; blk < 2; ++blk) {
        a.execute(packed, 4, ra);
        b.reset(packed, 4, rb);
        b.finalize();
        CHECK(max_rel(rb, ra) < 1e-5);
    }
}

TEST_CASE("FDMTFFT kill mask equals zeroed channels", "[cpu][fdmt_fft_cpu]") {
    std::vector<uint8_t> mask(kNch, 1);
    mask[0]  = 0;
    mask[17] = 0;
    mask[63] = 0;
    auto wf  = noise(kNch * kNs, 3);
    auto wz  = wf;
    for (SizeType c = 0; c < kNch; ++c) {
        if (mask[c] == 0) {
            std::fill_n(wz.begin() + static_cast<std::ptrdiff_t>(c * kNs), kNs,
                        0.0F);
        }
    }
    for (const auto* mode : {"valid", "full", "roll"}) {
        CAPTURE(mode);
        FDMTFFT killed(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true, mode,
                       Exec::cpu(), 1, true, mask);
        FDMTFFT zeroed(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true, mode,
                       Exec::cpu());
        const auto n = killed.get_plan().get_dmt_size();
        std::vector<float> a(n);
        std::vector<float> b(n);
        for (int blk = 0; blk < 2; ++blk) {
            killed.execute(wf, a);
            zeroed.execute(wz, b);
            CHECK(max_rel(a, b) < 1e-6);
        }
    }
    CHECK_THROWS_AS(FDMTFFT(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true,
                            "valid", Exec::cpu(1), 1, true,
                            std::vector<uint8_t>(kNch - 1, 1)),
                    std::invalid_argument);
}

TEST_CASE("FDMTFFT history save and load resume a stream",
          "[cpu][fdmt_fft_cpu]") {
    for (const bool frac : {true, false}) {
        CAPTURE(frac);
        FDMTFFT a(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true, "valid",
                  Exec::cpu(), 2, frac);
        FDMTFFT b(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true, "valid",
                  Exec::cpu(), 2, frac);
        REQUIRE(a.history_state_size() > 0);
        const auto n = 2 * a.get_plan().get_dmt_size();
        std::vector<float> ra(n);
        std::vector<float> rb(n);
        a.execute(noise(2 * kNch * kNs, 1), ra);
        std::vector<float> hist(a.history_state_size());
        a.save_history(hist);
        b.load_history(hist);
        const auto wf = noise(2 * kNch * kNs, 2);
        a.execute(wf, ra);
        b.execute(wf, rb);
        CHECK(ra == rb);
        std::vector<float> wrong(hist.size() + 1);
        CHECK_THROWS_AS(b.load_history(wrong), std::invalid_argument);
    }
    FDMTFFT roll(kFMin, kFMax, kNch, kNs, kTsamp, 48, 0, 1, true, "roll");
    CHECK(roll.history_state_size() == 0);
}

TEST_CASE("FDMTFFT rejects bad packed input", "[cpu][fdmt_fft_cpu]") {
    FDMTFFT f(kFMin, kFMax, kNch, kNs, kTsamp, 48);
    std::vector<float> out(f.get_plan().get_dmt_size());
    std::vector<uint8_t> bytes(kNch * kNs);
    CHECK_THROWS_AS(f.execute(bytes, 3, out), std::invalid_argument);
    CHECK_THROWS_AS(f.execute(bytes, 4, out), std::invalid_argument);
    CHECK_THROWS_AS(f.execute_time_major(bytes, 4, out), std::invalid_argument);
    CHECK_NOTHROW(f.execute(bytes, 8, out));
}

TEST_CASE("FDMTFFT forced segments equal the single transform",
          "[cpu][fdmt_fft_cpu]") {
    const SizeType ns = 8192;
    for (const auto* mode : {"valid", "full"}) {
        for (const bool frac : {true, false}) {
            CAPTURE(mode, frac);
            const plans::FDMTPlan plan(kFMin, kFMax, kNch, ns, kTsamp, 48, 0, 1,
                                       mode);
            FDMTFFT single(plan, true, Exec::cpu(), 1, frac);
            const auto fdmt_mode = std::string_view(mode) == "valid"
                                       ? FDMTMode::kValid
                                       : FDMTMode::kFull;
            const auto segmented = algorithms::detail::make_fdmt_fft_cpu(
                plan, {.use_box_smearing  = true,
                       .mode              = fdmt_mode,
                       .nbeams            = 1,
                       .exec              = Exec::cpu(),
                       .fractional_delays = frac,
                       .max_segment       = 1024});
            const auto n = plan.get_dmt_size();
            std::vector<float> a(n);
            std::vector<float> b(n);
            for (unsigned blk = 0; blk < 2; ++blk) {
                const auto wf = noise(kNch * ns, 20 + blk);
                single.execute(wf, a);
                segmented->execute(wf, b);
                // Integer delays: exact circular shifts, the same linear
                // convolution. Fractional: the segments truncate the
                // interpolation kernel elsewhere, a guard-level difference
                // (~1 / (pi sqrt(guard)) of the per-sample noise, a few
                // percent of the peak for 32 channels).
                CHECK(max_rel(b, a) < (frac ? 5e-2 : 1e-5));
            }
        }
    }
}

TEST_CASE("DDMTFFT forced segments equal the single transform",
          "[cpu][ddmt_fft_cpu]") {
    const SizeType nch = 32;
    const SizeType ns  = 6000;
    const plans::DDMTPlan plan(kFMin, kFMax, nch, kTsamp, 30.0F, 0.5F, 0.0F);
    for (const auto method : {DDMTFFTMethod::kBrute, DDMTFFTMethod::kNUFFT}) {
        CAPTURE(algorithms::to_string(method));
        const DDMTFFTOptions opts{.method = method};
        DDMTFFT single(plan, Exec::cpu(), 1, opts);
        const auto segmented =
            algorithms::detail::make_ddmt_fft_cpu(plan, {.nbeams  = 1,
                                                         .exec    = Exec::cpu(),
                                                         .options = opts,
                                                         .max_segment = 1024});
        const auto ndm = plan.get_dm_arr().size();
        for (unsigned blk = 0; blk < 2; ++blk) {
            const auto wf = noise(nch * ns, 30 + blk);
            const auto n  = ndm * single.get_output_nsamps(ns);
            std::vector<float> a(n);
            std::vector<float> b(n);
            single.execute(wf, a);
            segmented->execute(wf, b);
            // Guard-level (sinc truncation at the segment edges).
            CHECK(max_rel(b, a) < 5e-2);
        }
    }
}

} // namespace dmt
