#pragma once

// CohFDMT section of the published suite (dmt_bench_suite). CohFDMT searches
// baseband voltages, not a filterbank, so it has its own fixed configuration
// and plots and is not compared with the other algorithms.
//
// Reference search: one GUPPI node (64 x 2.9296875 MHz, 1312.5-1500 MHz,
// int8 FTPRI), t_p = 10 us, DM 50-60 pc/cc, dt_step 16, automatic block.
// Sweeps (through the reference point): t_p, input bit width, DM range width
// (number of coarse trials) and block length.
//
// Name, parsed by bench/scripts/plot_suite.py:
//   "cfdmt/<sweep>/<backend>/tp:<us>/nbits:<b>/dmw:<w>/block:<n>"
// with backend "cpu<threads>", "cuda"/"hip" (device-resident) or
// "cuda_host"/"hip_host" (host arrays, PCIe included); block 0 = automatic.

#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/common/baseband.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "suite_common.hpp"

namespace dmt::bench_suite::cfdmt {

inline constexpr float kFCenter      = 1406.25F;
inline constexpr float kBwSub        = 1500.0F / 512.0F;
inline constexpr SizeType kNsub      = 64;
inline constexpr float kDmMin        = 50.0F;
inline constexpr SizeType kDtStep    = 16;
inline constexpr SizeType kTpRef     = 10; // us
inline constexpr SizeType kNbitsRef  = 8;
inline constexpr SizeType kDmWidthRef = 10;

inline const std::vector<SizeType> kTpSweep      = {5, 10, 20, 40};
inline const std::vector<SizeType> kNbitsSweep   = {2, 4, 8};
inline const std::vector<SizeType> kDmWidthSweep = {10, 50, 200};
// Multiples of a GUPPI block (2^19 samples per subband); 0 = automatic.
inline const std::vector<SizeType> kBlockSweep = {0, SizeType{1} << 19,
                                                  SizeType{1} << 20,
                                                  SizeType{1} << 21};

enum class Sweep { kRef, kTp, kNbits, kDmWidth, kBlock };

struct Point {
    Sweep sweep;
    SizeType tp_us;
    SizeType nbits;
    SizeType dm_width;
    SizeType block;
};

inline std::string_view sweep_name(Sweep s) {
    switch (s) {
    case Sweep::kRef:
        return "ref";
    case Sweep::kTp:
        return "tp";
    case Sweep::kNbits:
        return "nbits";
    case Sweep::kDmWidth:
        return "dmw";
    case Sweep::kBlock:
        return "block";
    }
    return "?";
}

inline std::vector<Point> points() {
    std::vector<Point> pts{{Sweep::kRef, kTpRef, kNbitsRef, kDmWidthRef, 0}};
    for (const auto tp : kTpSweep) {
        pts.push_back({Sweep::kTp, tp, kNbitsRef, kDmWidthRef, 0});
    }
    for (const auto b : kNbitsSweep) {
        pts.push_back({Sweep::kNbits, kTpRef, b, kDmWidthRef, 0});
    }
    for (const auto w : kDmWidthSweep) {
        pts.push_back({Sweep::kDmWidth, kTpRef, kNbitsRef, w, 0});
    }
    for (const auto n : kBlockSweep) {
        pts.push_back({Sweep::kBlock, kTpRef, kNbitsRef, kDmWidthRef, n});
    }
    return pts;
}

inline CohFDMTConfig make_config(const Point& p) {
    return {.f_center     = kFCenter,
            .bw_sub       = kBwSub,
            .nsub         = kNsub,
            .t_p          = static_cast<float>(p.tp_us) * 1.0E-6F,
            .dm_min       = kDmMin,
            .dm_max       = kDmMin + static_cast<float>(p.dm_width),
            .block_nsamps = p.block,
            .dt_step      = kDtStep,
            .format = BasebandFormat{.order = "FTPRI", .nbits = p.nbits}};
}

inline std::string name(std::string_view backend, const Point& p) {
    return std::format("cfdmt/{}/{}/tp:{}/nbits:{}/dmw:{}/block:{}",
                       sweep_name(p.sweep), backend, p.tp_us, p.nbits,
                       p.dm_width, p.block);
}

/// "rtf": seconds of baseband searched per second (one stride per call);
/// "efficiency": stride / block; the rest describe the search.
inline void set_counters(benchmark::State& state,
                         const plans::CohFDMTPlan& plan,
                         int nthreads) {
    const double data_s =
        static_cast<double>(plan.get_stride_nsamps()) * plan.get_tbin();
    state.counters["rtf"] = benchmark::Counter(
        data_s, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["efficiency"] =
        static_cast<double>(plan.get_stride_nsamps()) /
        static_cast<double>(plan.get_block_nsamps());
    state.counters["ndm"]      = static_cast<double>(plan.get_ndm());
    state.counters["ndm_coh"]  = static_cast<double>(plan.get_ndm_coh());
    state.counters["nchans"]   = static_cast<double>(plan.get_nchans());
    state.counters["block_s"]  = static_cast<double>(plan.get_block_nsamps()) *
                                plan.get_tbin();
    state.counters["mem_mib"]  =
        static_cast<double>(plan.get_memory_estimate().total()) /
        (1024.0 * 1024.0);
    state.counters["nthreads"] = static_cast<double>(nthreads);
}

inline double estimate_bytes(const plans::CohFDMTPlan& plan) {
    const auto m = plan.get_memory_estimate();
    return static_cast<double>(m.total() + m.output +
                               plan.get_input_size(0));
}

} // namespace dmt::bench_suite::cfdmt
