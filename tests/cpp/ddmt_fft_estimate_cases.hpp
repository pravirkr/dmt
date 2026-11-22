#pragma once

// DDMTFFT::estimate_memory() cases shared by the CPU and GPU tests: a cold
// stream of equal calls, run until warm, holds what the estimate says.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/ddmt_fft_common.hpp"
#include "dmt/engines.hpp"

namespace dmt::ddmt_fft_estimate {

using algorithms::DDMTFFT;
using algorithms::DDMTFFTMemoryEstimate;
using algorithms::DDMTFFTMemoryUsage;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

struct Case {
    std::string name;
    plans::DDMTPlan plan;
    SizeType nbeams{1};
    DDMTFFTOptions options;
    SizeType nsamps{};
    SizeType cap{0}; // segment cap (0: none)
};

// Engine factory for capped cases (c.cap > 0), where tests force a short
// segment directly through DDMTFFTEngineConfig::max_segment.
inline std::unique_ptr<algorithms::detail::DDMTFFTEngine>
make_capped_engine(const Case& c, const Exec& exec, bool defer) {
    const algorithms::detail::DDMTFFTEngineConfig cfg{
        .nbeams           = c.nbeams,
        .exec             = exec,
        .options          = c.options,
        .max_segment      = c.cap,
        .defer_allocation = defer};
    if (exec.backend == Backend::kCPU) {
        return algorithms::detail::make_ddmt_fft_cpu(c.plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (exec.backend == algorithms::detail::kGPUBackend) {
        return algorithms::detail::make_ddmt_fft_gpu(c.plan, cfg);
    }
#endif
    throw std::invalid_argument("DDMTFFT estimate test: unavailable backend");
}

inline DDMTFFTMemoryUsage usage_of(const DDMTFFT& eng) {
    return eng.get_memory_usage();
}
inline DDMTFFTMemoryUsage usage_of(const algorithms::detail::DDMTFFTEngine& eng) {
    return eng.memory_usage();
}

inline plans::DDMTPlan
plan_of(float dm_max, SizeType nbits, std::vector<uint8_t> mask = {}) {
    return {1200.0F, 1600.0F, 16, 1.0E-3F, dm_max, 0.5F, 0.0F, nbits, mask};
}

inline DDMTFFTOptions with(DDMTFFTMethod m) {
    return {.method = m, .tolerance = 1.0E-6, .guard = 16};
}

/// Methods, formats, beams, kill mask, calls shorter than the context
/// (several warm-up calls) and segmented transforms (warm-up and steady
/// state on different lengths).
inline std::vector<Case> cases() {
    std::vector<uint8_t> mask(16, 1);
    mask[3] = 0;
    return {
        {"brute float", plan_of(20.0F, 32), 1, with(DDMTFFTMethod::kBrute),
         800},
        {"nufft 4-bit 2 beams", plan_of(20.0F, 4), 2,
         with(DDMTFFTMethod::kNUFFT), 1000},
        {"auto masked 8-bit", plan_of(20.0F, 8, mask), 1,
         with(DDMTFFTMethod::kAuto), 700},
        {"short calls", plan_of(200.0F, 32), 1, with(DDMTFFTMethod::kBrute),
         100},
        {"segmented 2-bit 3 beams", plan_of(200.0F, 2), 3,
         with(DDMTFFTMethod::kNUFFT), 3000, 1024},
    };
}

inline void check_usage(const DDMTFFTMemoryUsage& want,
                        SizeType staging,
                        SizeType pinned,
                        const DDMTFFTMemoryUsage& got) {
    CHECK(want.plan == got.plan);
    CHECK(want.history == got.history);
    CHECK(want.transform == got.transform);
    CHECK(want.workspace == got.workspace);
    CHECK(staging == got.staging);
    CHECK(pinned == got.pinned);
    CHECK(want.pinned == 0);
    CHECK(want.total() > 0);
}

/// Calls `call(n_out)` from a cold start until its first warm call (the
/// estimate's stream), checks the usage, then once more: nothing changes.
template <typename Engine, typename F>
void check_stream(Engine& eng,
                  SizeType nsamps,
                  const DDMTFFTMemoryEstimate& e,
                  SizeType staging,
                  SizeType pinned,
                  const F& call) {
    for (int i = 0; i < 4000; ++i) {
        const auto out = eng.get_output_nsamps(nsamps);
        call(out);
        if (out == nsamps) {
            break;
        }
    }
    REQUIRE(eng.get_output_nsamps(nsamps) == nsamps);
    check_usage(e.usage, staging, pinned, usage_of(eng));
    call(nsamps);
    check_usage(e.usage, staging, pinned, usage_of(eng));
}

enum class HostCall : std::uint8_t { kChannelMajor, kTimeMajor };

/// A host-memory call of `nsamps` zero samples (float or packed, as the
/// plan; time-major packed when asked).
template <typename Engine>
void host_call(Engine& eng,
               const plans::DDMTPlan& plan,
               SizeType nbeams,
               SizeType nsamps,
               SizeType out,
               HostCall kind) {
    const auto nbits  = plan.get_nbits();
    const auto nchans = plan.get_nchans();
    const auto rows   = nbeams * nchans;
    std::vector<float> dmt(nbeams * plan.get_dm_arr().size() * out);
    if (nbits == 32) {
        const std::vector<float> wf(rows * nsamps, 0.0F);
        eng.execute(wf, dmt);
    } else if (kind == HostCall::kChannelMajor) {
        const std::vector<uint8_t> wf(
            rows * bit_pack_utils::packed_row_bytes(nsamps, nbits), 0);
        eng.execute(wf, nsamps, dmt);
    } else {
        const std::vector<uint8_t> wf(
            nbeams * nsamps *
                bit_pack_utils::packed_row_bytes(nchans, nbits),
            0);
        eng.execute_time_major(wf, nsamps, dmt);
    }
}

/// Every case through host-memory calls: channel-major (staging
/// `host_staging`) and, for packed plans, time-major.
inline void check_host(const Case& c, const Exec& exec) {
    INFO(c.name);
    const auto run = [&](auto& eng, const DDMTFFTMemoryEstimate& e,
                         HostCall kind, SizeType staging) {
        check_stream(eng, c.nsamps, e, staging, e.host_pinned,
                     [&](SizeType out) {
                         host_call(eng, c.plan, c.nbeams, c.nsamps, out, kind);
                     });
    };
    if (c.cap == 0) {
        const auto e = DDMTFFT::estimate_memory(c.plan, c.nsamps, exec,
                                                c.nbeams, c.options);
        {
            DDMTFFT eng(c.plan, exec, c.nbeams, c.options);
            run(eng, e, HostCall::kChannelMajor, e.host_staging);
        }
        if (c.plan.get_nbits() != 32) {
            DDMTFFT eng(c.plan, exec, c.nbeams, c.options);
            run(eng, e, HostCall::kTimeMajor, e.time_major_staging);
        } else {
            CHECK(e.time_major_staging == 0);
        }
        return;
    }
    const auto e = make_capped_engine(c, exec, true)->estimate(c.nsamps);
    {
        auto eng = make_capped_engine(c, exec, false);
        run(*eng, e, HostCall::kChannelMajor, e.host_staging);
    }
    if (c.plan.get_nbits() != 32) {
        auto eng = make_capped_engine(c, exec, false);
        run(*eng, e, HostCall::kTimeMajor, e.time_major_staging);
    } else {
        CHECK(e.time_major_staging == 0);
    }
}

/// Call lengths around the warm-up span (the max delay): 1, a divisor of
/// it, one off either side, the span itself and longer than the context,
/// each a cold stream checked against its estimate; and a 0-sample
/// estimate, which equals an unused instance.
inline void check_call_lengths(const Exec& exec) {
    const auto plan    = plan_of(200.0F, 32);
    const auto options = with(DDMTFFTMethod::kBrute);
    const auto span    = DDMTFFT(plan, exec, 1, options).get_max_delay();
    REQUIRE(span > 8);
    SizeType divisor = 2;
    while (span % divisor != 0 && divisor < span) {
        ++divisor;
    }
    for (const SizeType n : {SizeType{1}, SizeType{2}, divisor, span - 1, span,
                             span + 1, (4 * span) + 50}) {
        INFO("nsamps " << n << " (span " << span << ")");
        const Case c{"call length", plan, 1, options, n};
        check_host(c, exec);
    }
    // Under a segment cap, a first output call shorter than the warm ones
    // can take one transform longer than their segments, and so set the
    // memory: find such a call length (with output-free calls before it).
    const SizeType ctx = span + options.guard;
    bool found         = false;
    for (SizeType cap = ctx + 2; cap < 4 * ctx && !found; ++cap) {
        for (SizeType n = 2; n < span && !found; ++n) {
            const SizeType silent = span / n;
            const SizeType n_out  = options.guard + (silent * n) + n - ctx;
            if (silent == 0 || n_out == 0 || n_out >= n) {
                continue;
            }
            const auto first =
                algorithms::ddmt_fft::plan_segments(n_out, ctx, cap).n_fft;
            const auto warm =
                algorithms::ddmt_fft::plan_segments(n, ctx, cap).n_fft;
            if (first > warm && first <= cap) {
                found = true;
                INFO("first output call sets the memory: nsamps "
                     << n << " cap " << cap);
                check_host({"first output call", plan, 1, options, n, cap},
                           exec);
            }
        }
    }
    CHECK(found);
    const auto e = DDMTFFT::estimate_memory(plan, 0, exec, 1, options);
    const DDMTFFT unused(plan, exec, 1, options);
    check_usage(e.usage, 0, 0, unused.get_memory_usage());
}

/// A configuration whose history alone (two copies of nbeams x 16 channels
/// x context floats) is `nbeams` x 7.7 kB.
inline Case many_beams(SizeType nbeams) {
    return {"many beams", plan_of(20.0F, 32), nbeams,
            with(DDMTFFTMethod::kBrute), 1000};
}

} // namespace dmt::ddmt_fft_estimate
