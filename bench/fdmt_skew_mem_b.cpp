// Peak memory of the skewed-window streaming FDMT prototype
// (lib/dmt/fdmt_skew.hpp) against the tree engine, at the active
// bench/suite preset's reference point (DMT_BENCH_CONFIG; 32768-sample
// blocks, DM 0-2000, 8 threads), for float and packed 8-bit input.
//
// The tree engine's state and history are aligned_alloc'd, which the
// operator new tracker does not see, so compare ProcessPeakRSS_MB, running
// one benchmark per process (--benchmark_filter), and the engines' own
// accounting (engine_MB).

#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt.hpp"

#include "dmt/fdmt_skew.hpp"

#include "bench_mem_utils.hpp"
#include "suite/suite_common.hpp"

namespace dmt {
namespace {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;
using algorithms::detail::FDMTSkewCpu;
using namespace bench_suite; // NOLINT

constexpr int kThreads = 8;

/// Chunk length of both engines: DMT_BENCH_CHUNK (default 4096).
SizeType chunk_nsamps() {
    const char* env = std::getenv("DMT_BENCH_CHUNK"); // NOLINT
    return (env != nullptr && *env != '\0')
               ? static_cast<SizeType>(std::stoul(env))
               : SizeType{4096};
}

Point ref_point(SizeType nbits) {
    return {Sweep::kNdms, kNsampsRef, kDmMaxRef, nbits};
}

void bm_tree(benchmark::State& state, SizeType nbits) {
    const auto p    = ref_point(nbits);
    const auto plan = make_plan(p);
    for (auto _ : state) {
        FDMT fdmt(plan, true, Exec::cpu(kThreads), 1, kFDMTAutoFuse, true,
                  chunk_nsamps());
        std::vector<float> dmt(plan.get_dmt_size());
        if (nbits == 32) {
            const auto wf = make_float_input(kNchans * p.nsamps);
            fdmt.execute(wf, dmt);
        } else {
            const auto wf = make_packed_input(packed_bytes(p.nsamps, nbits));
            fdmt.execute(std::span<const uint8_t>(wf), nbits, dmt);
        }
        const auto mem = fdmt.get_memory_usage();
        state.counters["engine_MB"] =
            static_cast<double>(mem.total()) / (1024.0 * 1024.0);
        state.counters["history_MB"] =
            static_cast<double>(mem.history) / (1024.0 * 1024.0);
    }
    bench::report_memory_counters(state);
}

void bm_skew(benchmark::State& state, SizeType nbits, SizeType level) {
    const auto p    = ref_point(nbits);
    const auto plan = make_plan(p);
    for (auto _ : state) {
        FDMTSkewCpu skew(plan, true, kThreads, chunk_nsamps(), level, nbits);
        std::vector<float> dmt(plan.get_dmt_size());
        if (nbits == 32) {
            const auto wf = make_float_input(kNchans * p.nsamps);
            skew.execute(wf, dmt);
        } else {
            const auto wf = make_packed_input(packed_bytes(p.nsamps, nbits));
            skew.execute(std::span<const uint8_t>(wf), nbits, dmt);
        }
        const auto mem = skew.memory_usage();
        state.counters["engine_MB"] =
            static_cast<double>(mem.total()) / (1024.0 * 1024.0);
        state.counters["rings_MB"] =
            static_cast<double>(mem.rings) / (1024.0 * 1024.0);
        state.counters["ring_MB"] =
            static_cast<double>(mem.ring) / (1024.0 * 1024.0);
        state.counters["chunk"] = static_cast<double>(chunk_nsamps());
    }
    bench::report_memory_counters(state);
}

} // namespace

BENCHMARK_CAPTURE(bm_tree, float32, 32)->Iterations(1)->UseRealTime();
BENCHMARK_CAPTURE(bm_tree, packed8, 8)->Iterations(1)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, float32_level4, 32, 4)->Iterations(1)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, packed8_level4, 8, 4)->Iterations(1)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, packed8_level3, 8, 3)->Iterations(1)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, packed8_level5, 8, 5)->Iterations(1)->UseRealTime();

} // namespace dmt
