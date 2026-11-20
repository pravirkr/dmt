#pragma once

// CohFDMT::estimate_memory() cases shared by the CPU and GPU tests.

#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/cfdmt.hpp"

#include "cfdmt_behaviour.hpp"

namespace dmt::cfdmt_estimate {

using algorithms::CohFDMT;
using test::cfdmt::small_config;

inline void check_usage(const plans::CohFDMTMemoryUsage& e,
                        const plans::CohFDMTMemoryUsage& got) {
    CHECK(e.spectrum == got.spectrum);
    CHECK(e.waterfall == got.waterfall);
    CHECK(e.fdmt == got.fdmt);
    CHECK(e.workspace == got.workspace);
    CHECK(e.output == got.output);
    CHECK(e.total() > 0);
}

inline void check(const plans::CohFDMTMemoryUsage& e, const CohFDMT& c) {
    check_usage(e, c.get_memory_usage());
}

/// Configurations: formats, normalisation, coarse trials, block lengths.
inline std::vector<CohFDMTConfig> cases() {
    std::vector<CohFDMTConfig> out;
    out.push_back(small_config());
    auto raw      = small_config();
    raw.normalize = false;
    out.push_back(raw);
    auto packed         = small_config();
    packed.format.nbits = 4;
    packed.format.order = "TFPRI";
    out.push_back(packed);
    auto trials   = small_config();
    trials.dm_max = 13.0F;
    out.push_back(trials);
    auto longer         = small_config();
    longer.block_nsamps = SizeType{1} << 17;
    out.push_back(longer);
    return out;
}

/// Every case on `exec` equals the built instance.
inline void check_all(const Exec& exec) {
    for (const auto& cfg : cases()) {
        INFO("block " << cfg.block_nsamps << " nbits " << cfg.format.nbits
                      << " dm_max " << cfg.dm_max);
        check(CohFDMT::estimate_memory(cfg, exec), CohFDMT(cfg, exec));
    }
}

/// The estimate allocates nothing (CPU): a block whose spectrum alone
/// (over 2 TB: `nsub` 16, 2^33 samples) is beyond any host is sized, never
/// allocated (an allocation would throw).
inline void check_impossible_cpu() {
    auto cfg         = small_config();
    cfg.block_nsamps = SizeType{1} << 33;
    const auto e     = CohFDMT::estimate_memory(cfg, Exec::cpu());
    CHECK(e.spectrum > (SizeType{1} << 41));
}

/// The configuration of the GPU probe: about 48 GiB in all (32 GiB
/// spectrum, 8 GiB waterfall, 8 GiB input copy), the fine FDMT just within
/// its 32-bit GPU index range, which bounds every single buffer below any
/// device's memory.
inline CohFDMTConfig beyond_device_config() {
    auto cfg         = small_config();
    cfg.nsub         = 64;
    cfg.t_p          = 64.0E-6F;
    cfg.block_nsamps = SizeType{1} << 25;
    return cfg;
}

} // namespace dmt::cfdmt_estimate
