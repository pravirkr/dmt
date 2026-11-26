#pragma once

// DDMT/SDMT::estimate_memory() cases shared by the CPU and GPU tests: an
// instance making calls of the estimated length holds what the estimate
// says, after every call of a stream from a cold start.

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/sdmt.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"

namespace dmt::ddmt_estimate {

using algorithms::DDMT;
using algorithms::DDMTMemoryEstimate;
using algorithms::DDMTMemoryUsage;
using algorithms::SDMT;

struct Case {
    std::string name;
    plans::DDMTPlan plan;
    SizeType nbeams{1};
    SizeType nsamps{};
    SizeType gulp{0}; // set_gulp_size (0: default)
};

inline plans::DDMTPlan
plan_of(float dm_max, SizeType nbits, std::vector<uint8_t> mask = {}) {
    return {1200.0F, 1600.0F, 16, 1.0E-3F, dm_max, 0.5F, 0.0F, nbits, mask};
}

/// Formats, beams, kill mask, and the host path's chunking: an explicit
/// gulp below the call (not a multiple of 8), above it, and the default
/// gulp splitting a long call into three chunks.
inline std::vector<Case> cases() {
    std::vector<uint8_t> mask(16, 1);
    mask[5] = 0;
    return {
        {"float", plan_of(20.0F, 32), 1, 1000, 0},
        {"4-bit 2 beams gulp 300", plan_of(20.0F, 4), 2, 1000, 300},
        {"8-bit masked gulp 4096", plan_of(20.0F, 8, mask), 1, 1000, 4096},
        {"2-bit auto chunks", plan_of(20.0F, 2), 1, 60000, 0},
    };
}

// GPU SDMT engine with force_shared_sums = true (small plans otherwise run
// the DDMT kernel and upload no shared-sum tables).
inline std::unique_ptr<algorithms::detail::DDMTEngine>
make_forced_sdmt_gpu(const Case& c, const Exec& exec, bool defer) {
#ifdef DMT_ENABLE_GPU
    auto eng =
        algorithms::detail::make_sdmt_gpu(c.plan, {.nbeams           = c.nbeams,
                                                   .exec             = exec,
                                                   .defer_allocation = defer,
                                                   .force_shared_sums = true});
    eng->set_gulp_size(c.gulp);
    return eng;
#else
    (void)c;
    (void)exec;
    (void)defer;
    throw std::invalid_argument("SDMT GPU not enabled");
#endif
}

inline DDMTMemoryUsage usage_of(const DDMT& eng) {
    return eng.get_memory_usage();
}
inline DDMTMemoryUsage usage_of(const algorithms::detail::DDMTEngine& eng) {
    return eng.memory_usage();
}

inline std::unique_ptr<DDMT> make(bool sdmt, const Case& c, const Exec& exec) {
    std::unique_ptr<DDMT> e;
    if (sdmt) {
        e = std::make_unique<SDMT>(c.plan, exec, c.nbeams);
    } else {
        e = std::make_unique<DDMT>(c.plan, exec, c.nbeams);
    }
    e->set_gulp_size(c.gulp);
    return e;
}

inline DDMTMemoryEstimate estimate(bool sdmt, const Case& c, const Exec& exec) {
    return sdmt
               ? SDMT::estimate_memory(c.plan, c.nsamps, exec, c.nbeams, c.gulp)
               : DDMT::estimate_memory(c.plan, c.nsamps, exec, c.nbeams,
                                       c.gulp);
}

inline void check_usage(const DDMTMemoryUsage& want,
                        SizeType staging,
                        SizeType pinned,
                        const DDMTMemoryUsage& got) {
    CHECK(want.plan == got.plan);
    CHECK(want.history == got.history);
    CHECK(staging == got.staging);
    CHECK(pinned == got.pinned);
    CHECK(want.staging == 0);
    CHECK(want.pinned == 0);
    CHECK(want.total() > 0);
}

/// From a cold start until one call past warm-up, the usage after every
/// call equals the estimate plus `staging` and `pinned`.
template <typename Engine, typename F>
void check_stream(Engine& eng,
                  SizeType nsamps,
                  const DDMTMemoryEstimate& e,
                  SizeType staging,
                  SizeType pinned,
                  const F& call) {
    bool warm = false;
    for (int i = 0; i < 100 && !warm; ++i) {
        warm = eng.get_output_nsamps(nsamps) == nsamps;
        call(eng.get_output_nsamps(nsamps));
        check_usage(e.usage, staging, pinned, usage_of(eng));
    }
    REQUIRE(warm);
}

enum class HostCall : std::uint8_t { kChannelMajor, kTimeMajor };

/// A host-memory call of `nsamps` zero samples.
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
    const auto nout   = nbeams * plan.get_dm_arr().size() * out;
    if (nbits == 32) {
        const std::vector<float> wf(rows * nsamps, 0.0F);
        std::vector<float> dmt(nout);
        eng.execute(wf, dmt);
        return;
    }
    std::vector<int32_t> dmt(nout);
    if (kind == HostCall::kChannelMajor) {
        const std::vector<uint8_t> wf(
            rows * bit_pack_utils::packed_row_bytes(nsamps, nbits), 0);
        eng.execute(wf, nsamps, dmt);
    } else {
        const std::vector<uint8_t> wf(
            nbeams * nsamps * bit_pack_utils::packed_row_bytes(nchans, nbits),
            0);
        eng.execute_time_major(wf, nsamps, dmt);
    }
}

/// Every case, DDMT and SDMT, through host-memory calls: channel-major and,
/// for packed plans, time-major (each on a fresh instance).
inline void check_host(const Exec& exec) {
    const auto check_one = [&](const Case& c, const DDMTMemoryEstimate& e,
                               auto make_eng) {
        {
            auto eng = make_eng();
            check_stream(*eng, c.nsamps, e, e.host_staging, e.host_pinned,
                         [&](SizeType out) {
                             host_call(*eng, c.plan, c.nbeams, c.nsamps, out,
                                       HostCall::kChannelMajor);
                         });
        }
        if (c.plan.get_nbits() != 32) {
            auto eng = make_eng();
            check_stream(*eng, c.nsamps, e, e.time_major_staging,
                         e.time_major_pinned, [&](SizeType out) {
                             host_call(*eng, c.plan, c.nbeams, c.nsamps, out,
                                       HostCall::kTimeMajor);
                         });
        } else {
            CHECK(e.time_major_staging == 0);
            CHECK(e.time_major_pinned == 0);
            CHECK(e.device_time_major_staging == 0);
        }
    };
    for (const bool sdmt : {false, true}) {
        for (const auto& c : cases()) {
            INFO((sdmt ? "SDMT " : "DDMT ") << c.name);
            if (sdmt && exec.backend != Backend::kCPU) {
                const auto e =
                    make_forced_sdmt_gpu(c, exec, true)->estimate(c.nsamps);
                check_one(c, e,
                          [&] { return make_forced_sdmt_gpu(c, exec, false); });
            } else {
                const auto e = estimate(sdmt, c, exec);
                check_one(c, e, [&] { return make(sdmt, c, exec); });
            }
        }
    }
}

/// SDMT's estimate counts its own programs: on the same plan its plan
/// tables differ from DDMT's.
inline void check_sdmt_differs(const Exec& exec) {
    const Case c{"sdmt vs ddmt", plan_of(60.0F, 32), 1, 1000, 0};
    const auto d = estimate(false, c, exec);
    const auto s =
        exec.backend == Backend::kCPU
            ? estimate(true, c, exec)
            : make_forced_sdmt_gpu(c, exec, true)->estimate(c.nsamps);
    CHECK(d.usage.plan != s.usage.plan);
    CHECK(d.usage.history == s.usage.history);
}

/// A configuration whose history alone (two copies of nbeams x 16 channels
/// x max-delay floats) grows with `nbeams`; no plan table does.
inline Case many_beams(SizeType nbeams) {
    return {"many beams", plan_of(20.0F, 32), nbeams, 1000, 0};
}

} // namespace dmt::ddmt_estimate
