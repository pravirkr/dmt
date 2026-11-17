#pragma once

// DDMTFFT::estimate_memory() cases shared by the CPU and GPU tests: a cold
// stream of equal calls, run until warm, holds what the estimate says.

#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"

namespace dmt::ddmt_fft_estimate {

using algorithms::DDMTFFT;
using algorithms::DDMTFFTMemoryEstimate;
using algorithms::DDMTFFTMemoryUsage;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

// Restores the segment-cap hook at scope exit.
struct SegmentCap {
    explicit SegmentCap(SizeType n) {
        algorithms::detail::set_fft_segment_cap(n);
    }
    ~SegmentCap() { algorithms::detail::set_fft_segment_cap(0); }
    SegmentCap(const SegmentCap&)            = delete;
    SegmentCap& operator=(const SegmentCap&) = delete;
    SegmentCap(SegmentCap&&)                 = delete;
    SegmentCap& operator=(SegmentCap&&)      = delete;
};

struct Case {
    std::string name;
    plans::DDMTPlan plan;
    SizeType nbeams{1};
    DDMTFFTOptions options;
    SizeType nsamps{};
    SizeType cap{0}; // segment cap (0: none)
};

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
                        const DDMTFFTMemoryUsage& got) {
    CHECK(want.plan == got.plan);
    CHECK(want.history == got.history);
    CHECK(want.transform == got.transform);
    CHECK(want.workspace == got.workspace);
    CHECK(staging == got.staging);
    CHECK(want.total() > 0);
}

/// Calls `call(n_out)` from a cold start until its first warm call (the
/// estimate's stream), checks the usage, then once more: nothing changes.
template <typename F>
void check_stream(DDMTFFT& eng,
                  SizeType nsamps,
                  const DDMTFFTMemoryEstimate& e,
                  SizeType staging,
                  const F& call) {
    for (int i = 0; i < 1000; ++i) {
        const auto out = eng.get_output_nsamps(nsamps);
        call(out);
        if (out == nsamps) {
            break;
        }
    }
    REQUIRE(eng.get_output_nsamps(nsamps) == nsamps);
    check_usage(e.usage, staging, eng.get_memory_usage());
    call(nsamps);
    check_usage(e.usage, staging, eng.get_memory_usage());
}

enum class HostCall : std::uint8_t { kChannelMajor, kTimeMajor };

/// A host-memory call of `nsamps` zero samples (float or packed, as the
/// plan; time-major packed when asked).
inline void
host_call(DDMTFFT& eng, SizeType nsamps, SizeType out, HostCall kind) {
    const auto& plan  = eng.get_plan();
    const auto nbits  = plan.get_nbits();
    const auto nchans = plan.get_nchans();
    const auto rows   = eng.get_nbeams() * nchans;
    std::vector<float> dmt(eng.get_nbeams() * plan.get_dm_arr().size() * out);
    if (nbits == 32) {
        const std::vector<float> wf(rows * nsamps, 0.0F);
        eng.execute(wf, dmt);
    } else if (kind == HostCall::kChannelMajor) {
        const std::vector<uint8_t> wf(
            rows * bit_pack_utils::packed_row_bytes(nsamps, nbits), 0);
        eng.execute(wf, nsamps, dmt);
    } else {
        const std::vector<uint8_t> wf(
            eng.get_nbeams() * nsamps *
                bit_pack_utils::packed_row_bytes(nchans, nbits),
            0);
        eng.execute_time_major(wf, nsamps, dmt);
    }
}

/// Every case through host-memory calls: channel-major (staging
/// `host_staging`) and, for packed plans, time-major.
inline void check_host(const Case& c, const Exec& exec) {
    INFO(c.name);
    const SegmentCap cap(c.cap);
    const auto e =
        DDMTFFT::estimate_memory(c.plan, c.nsamps, exec, c.nbeams, c.options);
    {
        DDMTFFT eng(c.plan, exec, c.nbeams, c.options);
        check_stream(eng, c.nsamps, e, e.host_staging, [&](SizeType out) {
            host_call(eng, c.nsamps, out, HostCall::kChannelMajor);
        });
    }
    if (c.plan.get_nbits() != 32) {
        DDMTFFT eng(c.plan, exec, c.nbeams, c.options);
        check_stream(eng, c.nsamps, e, e.time_major_staging, [&](SizeType out) {
            host_call(eng, c.nsamps, out, HostCall::kTimeMajor);
        });
    } else {
        CHECK(e.time_major_staging == 0);
    }
}

/// A configuration whose history alone (two copies of nbeams x 16 channels
/// x context floats) is `nbeams` x 7.7 kB.
inline Case many_beams(SizeType nbeams) {
    return {"many beams", plan_of(20.0F, 32), nbeams,
            with(DDMTFFTMethod::kBrute), 1000};
}

} // namespace dmt::ddmt_fft_estimate
