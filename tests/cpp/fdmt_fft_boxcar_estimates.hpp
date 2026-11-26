#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt_fft_boxcar.hpp"

namespace boxcar_test {

/// FDMTFFTBoxcar::estimate_memory() equals the built instance on `exec`.
inline void check_fft_boxcar_estimates(const dmt::Exec& exec) {
    using namespace dmt; // NOLINT(google-build-using-namespace)
    using algorithms::FDMTFFTBoxcar;

    const plans::FDMTPlan plan(1000.0F, 1500.0F, 64, 2048, 0.001F, 80);
    BoxcarConfig cfg;
    cfg.max_boxcar        = 32;
    BoxcarConfig keep     = cfg;
    keep.overflow_policy  = OverflowPolicy::kKeepBrightest;
    keep.max_cands_per_dm = 4;
    std::vector<uint8_t> kill(plan.get_nchans(), 1);
    kill[7] = 0;
    struct Case {
        const char* name;
        BoxcarConfig config;
        SizeType beams;
        std::vector<uint8_t> kill;
    };
    const std::vector<Case> cases{{"default", cfg, 1, {}},
                                  {"keep brightest", keep, 2, {}},
                                  {"kill mask", cfg, 2, kill}};
    const auto check = [](const algorithms::FDMTBoxcarMemoryEstimate& e,
                          const FDMTFFTBoxcar& f) {
        const auto got = f.get_memory_usage();
        CHECK(e.usage.plan == got.plan);
        CHECK(e.usage.state == got.state);
        CHECK(e.usage.history == got.history);
        CHECK(e.usage.peaks == got.peaks);
        CHECK(e.usage.workspace == got.workspace);
        CHECK(e.usage.candidates == got.candidates);
        CHECK(e.usage.staging == got.staging);
        CHECK(e.usage.host == got.host);
        CHECK(e.chunk_nsamps == f.get_chunk_nsamps());
    };
    for (const auto& c : cases) {
        INFO(c.name);
        const auto e = FDMTFFTBoxcar::estimate_memory(
            plan, c.config, true, exec, c.beams, true,
            algorithms::kFDMTDefaultMaxMemory, c.kill);
        const FDMTFFTBoxcar f(plan, c.config, true, exec, c.beams, true,
                              algorithms::kFDMTDefaultMaxMemory, c.kill);
        check(e, f);
    }
    const plans::FDMTPlan long_plan(1000.0F, 1500.0F, 256, 32768, 0.001F, 80);
    BoxcarConfig small   = cfg;
    small.max_candidates = 1024;
    const auto full =
        FDMTFFTBoxcar::estimate_memory(long_plan, small, true, exec, 4);
    const SizeType budget = full.usage.budgeted() - 1;
    const auto e = FDMTFFTBoxcar::estimate_memory(long_plan, small, true, exec,
                                                  4, true, budget);
    check(e, FDMTFFTBoxcar(long_plan, small, true, exec, 4, true, budget));
    CHECK(e.chunk_nsamps < full.chunk_nsamps);
    CHECK(e.usage.budgeted() <= budget);
    CHECK_THROWS_AS(
        FDMTFFTBoxcar::estimate_memory(plan, cfg, true, exec, 1, true, 1024),
        std::invalid_argument);
    BoxcarConfig huge   = cfg;
    huge.max_candidates = SizeType{1} << 40;
    const plans::FDMTPlan big(1000.0F, 1500.0F, 256, 4096, 0.001F, 400);
    const auto h = FDMTFFTBoxcar::estimate_memory(
        big, huge, true, exec, SizeType{1} << 22, true, SIZE_MAX / 2);
    CHECK(h.usage.candidates >= huge.max_candidates * sizeof(RawCandidate));
    CHECK(h.usage.state + h.usage.workspace + h.usage.history >
          (SizeType{1} << 40));
}

} // namespace boxcar_test
