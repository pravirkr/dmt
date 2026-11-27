#pragma once

// A transient NaN in one block must not blind the Kadane-screened search for
// the rest of the stream: the cross-call Kadane carry restarts at 0 when it
// is not finite, so a pulse a few blocks later is found as without the NaN.

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "test_random.hpp"

namespace dmt::test {

inline void check_nan_block_does_not_blind(const Exec& exec) {
    constexpr SizeType kNchans = 32;
    constexpr SizeType kNsamps = 256;
    constexpr SizeType kBlocks = 6;
    constexpr SizeType kPulse  = (4 * kNsamps) + 128; // in block 4
    const plans::FDMTPlan plan(1200.0F, 1600.0F, kNchans, kNsamps, 0.001F, 16);
    BoxcarConfig cfg;
    cfg.max_boxcar    = 16;
    cfg.detect_thresh = 8.0F;
    cfg.screen_mode   = BoxcarScreenMode::kKadane;
    const auto stream = [&](bool with_nan) {
        algorithms::FDMTBoxcar search(plan, cfg, true, exec);
        std::vector<RawCandidate> found;
        for (SizeType k = 0; k < kBlocks; ++k) {
            auto block = test_random::normal<float>(kNchans * kNsamps, 700 + k);
            if (with_nan && k == 1) {
                // Bits, not quiet_NaN(): the tests build with -ffast-math.
                block[(3 * kNsamps) + 100] = std::bit_cast<float>(0x7FC00000U);
                block[(9 * kNsamps) + 101] = std::bit_cast<float>(0x7F800000U);
            }
            if (k * kNsamps <= kPulse && kPulse < (k + 1) * kNsamps) {
                for (SizeType c = 0; c < kNchans; ++c) {
                    block[(c * kNsamps) + (kPulse - (k * kNsamps))] += 4.0F;
                }
            }
            const auto res = search.execute(std::span<const float>(block));
            found.insert(found.end(), res.candidates.begin(),
                         res.candidates.end());
        }
        const auto res = search.flush();
        found.insert(found.end(), res.candidates.begin(), res.candidates.end());
        return found;
    };
    const auto pulse_found = [&](const std::vector<RawCandidate>& found) {
        for (const auto& c : found) {
            if (c.peak_sample + 8 >= kPulse && c.peak_sample <= kPulse + 24 &&
                c.snr > 15.0F) {
                return true;
            }
        }
        return false;
    };
    REQUIRE(pulse_found(stream(false)));
    CHECK(pulse_found(stream(true)));
}

} // namespace dmt::test
