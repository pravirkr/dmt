#pragma once

// FDMTFFT::estimate_memory() cases shared by the CPU and GPU tests.

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_fft.hpp"

namespace dmt::fft_estimate {

using algorithms::FDMTFFT;
using algorithms::FDMTMemoryEstimate;

struct Case {
    std::string mode;
    SizeType beams;
    bool frac;
};

/// Modes, beams and integer / fractional delays.
inline std::vector<Case> cases() {
    std::vector<Case> out;
    for (const char* mode : {"valid", "full", "roll"}) {
        for (const SizeType beams : {1, 2}) {
            for (const bool frac : {true, false}) {
                out.push_back({mode, beams, frac});
            }
        }
    }
    return out;
}

inline void check(const FDMTMemoryEstimate& e, const FDMTFFT& f) {
    const auto got = f.get_memory_usage();
    CHECK(e.usage.plan == got.plan);
    CHECK(e.usage.state == got.state);
    CHECK(e.usage.history == got.history);
    CHECK(e.usage.workspace == got.workspace);
    CHECK(e.usage.output == got.output);
    CHECK(e.chunk_nsamps > 0);
}

/// Every overload and case on `exec` equals the built instance.
inline void check_all(const Exec& exec) {
    constexpr float kFMin  = 1000.0F;
    constexpr float kFMax  = 1500.0F;
    constexpr float kTsamp = 0.001F;
    for (const auto& c : cases()) {
        INFO(c.mode << " beams " << c.beams << " frac " << c.frac);
        const auto e =
            FDMTFFT::estimate_memory(kFMin, kFMax, 64, 1024, kTsamp, 80, 0, 1,
                                     true, c.mode, exec, c.beams, c.frac);
        const FDMTFFT f(kFMin, kFMax, 64, 1024, kTsamp, 80, 0, 1, true, c.mode,
                        exec, c.beams, c.frac);
        check(e, f);
        CHECK((e.host_staging == 0) == (exec.backend == Backend::kCPU));
    }
    std::vector<uint8_t> kill(64, 1);
    kill[5] = 0;
    const std::vector<IndexType> dt_grid{0, 3, 9, 20, 41, 80};
    check(FDMTFFT::estimate_memory(kFMin, kFMax, 64, 1024, kTsamp, dt_grid,
                                   true, "valid", exec, 1, true, kill),
          FDMTFFT(kFMin, kFMax, 64, 1024, kTsamp, dt_grid, true, "valid", exec,
                  1, true, kill));
    const std::vector<float> dm_grid{0.0F, 10.0F, 25.0F, 50.0F};
    check(
        FDMTFFT::estimate_memory(kFMin, kFMax, 64, 1024, kTsamp, dm_grid, false,
                                 "valid", exec),
        FDMTFFT(kFMin, kFMax, 64, 1024, kTsamp, dm_grid, false, "valid", exec));
    const plans::FDMTPlan plan(kFMin, kFMax, 64, 1024, kTsamp, 80, 0, 1,
                               "valid", 1);
    check(FDMTFFT::estimate_memory(plan, true, exec, 2),
          FDMTFFT(plan, true, exec, 2));
}

} // namespace dmt::fft_estimate
