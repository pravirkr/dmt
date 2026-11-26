#pragma once

// FDMTFFT stepper in mode "valid" on any backend: streaming through
// reset()/finalize() matches execute() even when the caller frees its input
// right after reset(), and an unfinished block rejects execute()/reset()
// until it is finalized or abandoned with reset_history().

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/common/backend.hpp"

#include "test_random.hpp"

namespace dmt::test {

/// max |a - b| over max |b|.
inline double stepper_rel_error(const std::vector<float>& a,
                                const std::vector<float>& b) {
    double num = 0.0;
    double den = 0.0;
    for (SizeType i = 0; i < b.size(); ++i) {
        num = std::max(num, static_cast<double>(std::abs(a[i] - b[i])));
        den = std::max(den, static_cast<double>(std::abs(b[i])));
    }
    return den > 0.0 ? num / den : num;
}

inline void check_fdmt_fft_valid_stepper(const Exec& exec) {
    using algorithms::FDMTFFT;
    constexpr SizeType kNchans = 32;
    constexpr SizeType kNsamps = 256;
    const auto make            = [&] {
        return FDMTFFT(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, 40, 0, 1,
                       true, "valid", exec);
    };

    SECTION("streamed stepper blocks match execute(); input freed early") {
        auto ref     = make();
        auto step    = make();
        const auto n = ref.get_plan().get_dmt_size();
        for (unsigned blk = 0; blk < 4; ++blk) {
            CAPTURE(blk);
            auto wf = test_random::uniform_real<float>(kNchans * kNsamps,
                                                       300 + blk, -1.0F, 1.0F);
            std::vector<float> a(n);
            std::vector<float> b(n);
            ref.execute(wf, a);
            step.reset(wf, b);
            // reset() reads the block: release it before finalize().
            std::vector<float>(wf.size(), 1.0e6F).swap(wf);
            step.advance(1);
            step.finalize();
            CHECK(stepper_rel_error(b, a) < 1e-5);
        }
    }

    SECTION("packed stepper blocks match execute(); input freed early") {
        auto ref     = make();
        auto step    = make();
        const auto n = ref.get_plan().get_dmt_size();
        for (unsigned blk = 0; blk < 3; ++blk) {
            CAPTURE(blk);
            auto packed = test_random::uniform_int<uint8_t>(kNchans * kNsamps,
                                                            400 + blk, 0, 255);
            std::vector<float> a(n);
            std::vector<float> b(n);
            ref.execute(std::span<const uint8_t>(packed), 8, a);
            step.reset(std::span<const uint8_t>(packed), 8, b);
            std::vector<uint8_t>(packed.size(), 255).swap(packed);
            step.finalize();
            CHECK(stepper_rel_error(b, a) < 1e-5);
        }
    }

    SECTION("an unfinished block rejects execute() and reset()") {
        auto step    = make();
        const auto n = step.get_plan().get_dmt_size();
        const auto wf =
            test_random::uniform_real<float>(kNchans * kNsamps, 7, -1.0F, 1.0F);
        std::vector<float> out(n);
        step.reset(wf, out);
        CHECK_THROWS_AS(step.execute(wf, out), std::logic_error);
        CHECK_THROWS_AS(step.reset(wf, out), std::logic_error);
        // At the root but not finalized: the history is not committed yet.
        step.advance_until_remaining(0);
        CHECK_THROWS_AS(step.execute(wf, out), std::logic_error);
        step.finalize();
        CHECK_NOTHROW(step.execute(wf, out));
        // reset_history() abandons the block along with the stream.
        step.reset(wf, out);
        step.reset_history();
        CHECK_THROWS_AS(step.finalize(), std::logic_error);
        CHECK_NOTHROW(step.execute(wf, out));
    }
}

} // namespace dmt::test
