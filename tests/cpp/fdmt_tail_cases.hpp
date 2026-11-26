#pragma once

// A stream ending in a short block: FDMT::execute_tail() and
// FDMTBoxcar::drain(tail, n_tail) on k whole blocks plus a tail of n_tail
// samples give what one pass over the same k * nsamps + n_tail samples
// gives, whatever the tail block's padding holds.

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"

#include "fdmt_boxcar_reference.hpp"
#include "test_helpers.hpp"
#include "test_random.hpp"

namespace boxcar_test {

/// Rows [0, n) of each channel of `whole` (nchans x total) starting at t0, as
/// a (nchans, nsamps) block padded with `pad`.
template <typename T>
std::vector<T> slice_block(const std::vector<T>& whole,
                           SizeType nchans,
                           SizeType total,
                           SizeType t0,
                           SizeType n,
                           SizeType nsamps,
                           T pad) {
    std::vector<T> block(nchans * nsamps, pad);
    for (SizeType c = 0; c < nchans; ++c) {
        for (SizeType t = 0; t < n; ++t) {
            block[(c * nsamps) + t] = whole[(c * total) + t0 + t];
        }
    }
    return block;
}

inline void check_fdmt_tail(const Exec& exec) {
    constexpr SizeType kNchans = 16;
    constexpr SizeType kNsamps = 64;
    constexpr SizeType kBlocks = 3;
    constexpr SizeType kTail   = 37;
    constexpr SizeType kTotal  = (kBlocks * kNsamps) + kTail;
    const std::vector<IndexType> dts{0, 3, 8, 15, 24};
    const auto make = [&](SizeType nsamps) {
        return FDMT(plans::FDMTPlan(1200.0F, 1600.0F, kNchans, nsamps, 0.001F,
                                    dts, "valid"),
                    true, exec);
    };
    for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
        CAPTURE(nbits);
        const auto values = test_random::uniform_int<uint8_t>(
            kNchans * kTotal, 17 + nbits, 0, 255);
        std::vector<float> fvalues(values.begin(), values.end());
        auto whole      = make(kTotal);
        const auto nrow = whole.get_plan().get_dmt_ndms();
        std::vector<float> ref(whole.get_plan().get_dmt_size());
        if (nbits == 32) {
            whole.execute(fvalues, ref);
        } else {
            whole.execute(std::span<const uint8_t>(values), 8, ref);
        }
        auto stream = make(kNsamps);
        std::vector<float> got(nrow * kTotal, 0.0F);
        std::vector<float> out(stream.get_plan().get_dmt_size());
        for (SizeType k = 0; k <= kBlocks; ++k) {
            const SizeType n = k < kBlocks ? kNsamps : kTail;
            // Padding that must not matter: large values.
            const auto fb = slice_block(fvalues, kNchans, kTotal, k * kNsamps,
                                        n, kNsamps, 1.0e4F);
            const auto pb = slice_block(values, kNchans, kTotal, k * kNsamps, n,
                                        kNsamps, uint8_t{255});
            if (k < kBlocks) {
                if (nbits == 32) {
                    stream.execute(fb, out);
                } else {
                    stream.execute(std::span<const uint8_t>(pb), 8, out);
                }
            } else if (nbits == 32) {
                stream.execute_tail(std::span<const float>(fb), kTail, out);
            } else {
                stream.execute_tail(std::span<const uint8_t>(pb), 8, kTail,
                                    out);
            }
            for (SizeType r = 0; r < nrow; ++r) {
                for (SizeType t = 0; t < n; ++t) {
                    got[(r * kTotal) + (k * kNsamps) + t] =
                        out[(r * kNsamps) + t];
                }
            }
        }
        for (SizeType r = 0; r < nrow; ++r) {
            for (SizeType t = 0; t < kTotal; ++t) {
                INFO("row " << r << " sample " << t);
                const float a = got[(r * kTotal) + t];
                const float b = ref[(r * kTotal) + t];
                if (nbits == 32) {
                    REQUIRE(std::abs(a - b) <= 1e-4F * (1.0F + std::abs(b)));
                } else {
                    REQUIRE(a == b);
                }
            }
        }
        // The stream has ended.
        CHECK_THROWS_AS(stream.execute(fvalues, out), std::logic_error);
        CHECK_THROWS_AS(
            stream.execute_tail(std::span<const float>(out), 1, out),
            std::logic_error);
        stream.reset_history();
        CHECK_THROWS_AS(
            stream.execute_tail(
                std::span<const float>(fvalues).first(kNchans * kNsamps), 0,
                out),
            std::invalid_argument);
        CHECK_THROWS_AS(
            stream.execute_tail(
                std::span<const float>(fvalues).first(kNchans * kNsamps),
                kNsamps + 1, out),
            std::invalid_argument);
    }
}

inline void check_boxcar_tail(const Exec& exec) {
    constexpr SizeType kNchans = 16;
    constexpr SizeType kNsamps = 128;
    constexpr SizeType kBlocks = 3;
    constexpr SizeType kTail   = 45;
    constexpr SizeType kTotal  = (kBlocks * kNsamps) + kTail;
    const std::vector<IndexType> dts{0, 3, 8, 15, 24, 40};
    BoxcarConfig cfg;
    cfg.max_boxcar       = 16;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.5F;
    cfg.max_candidates   = SizeType{1} << 20;
    cfg.max_cands_per_dm = 0;
    cfg.input_nbits      = 8;
    cfg.input_mean       = 128.0F;
    cfg.noise_sigma      = 30.0F;
    const auto values =
        test_random::uniform_int<uint8_t>(kNchans * kTotal, 91, 76, 180);
    const auto plan = [&](SizeType nsamps) {
        return plans::FDMTPlan(1200.0F, 1600.0F, kNchans, nsamps, 0.001F, dts,
                               "valid");
    };
    std::vector<RawCandidate> want;
    {
        FDMTBoxcar whole(plan(kTotal), cfg, true, exec);
        append(want, whole.execute(std::span<const uint8_t>(values)));
        append(want, whole.drain());
        CHECK(whole.get_searched_nsamps() == kTotal);
    }
    REQUIRE(want.size() > 20);
    FDMTBoxcar stream(plan(kNsamps), cfg, true, exec);
    std::vector<RawCandidate> got;
    for (SizeType k = 0; k < kBlocks; ++k) {
        const auto b = slice_block(values, kNchans, kTotal, k * kNsamps,
                                   kNsamps, kNsamps, uint8_t{0});
        append(got, stream.execute(std::span<const uint8_t>(b)));
    }
    // Bright padding: never searched.
    const auto tail = slice_block(values, kNchans, kTotal, kBlocks * kNsamps,
                                  kTail, kNsamps, uint8_t{255});
    append(got, stream.drain(std::span<const uint8_t>(tail), kTail));
    CHECK(stream.get_searched_nsamps() == kTotal);
    for (const auto& c : got) {
        REQUIRE(c.end_sample < kTotal);
    }
    compare_exact(got, want);
    CHECK_THROWS_AS(stream.drain(std::span<const uint8_t>(tail), kTail),
                    std::logic_error);
    stream.reset_history();
    CHECK_THROWS_AS(stream.drain(std::span<const uint8_t>(tail), 0),
                    std::invalid_argument);
}

} // namespace boxcar_test
