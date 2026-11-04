// Throughput of the skewed-window streaming FDMT prototype
// (lib/dmt/fdmt_skew.hpp) against the tree engine, at the active
// bench/suite preset's reference point (DMT_BENCH_CONFIG; 32768-sample
// blocks, DM 0-2000), float and packed 8-bit input, steady-state streaming
// (one warm-up block, then one block per iteration).

#include <cstdint>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt.hpp"

#include "dmt/fdmt_skew.hpp"

#include "suite/suite_common.hpp"

namespace dmt {
namespace {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;
using algorithms::detail::FDMTSkewCpu;
using namespace bench_suite; // NOLINT

Point ref_point(SizeType nbits) {
    return {Sweep::kNdms, kNsampsRef, kDmMaxRef, nbits};
}

void bm_tree(benchmark::State& state,
             SizeType nbits,
             int nthreads,
             SizeType chunk) {
    const auto p    = ref_point(nbits);
    const auto plan = make_plan(p);
    FDMT fdmt(plan, true, Exec::cpu(nthreads), 1, kFDMTAutoFuse, true, chunk);
    std::vector<float> dmt(plan.get_dmt_size());
    const auto wf_f = nbits == 32 ? make_float_input(kNchans * p.nsamps)
                                  : std::vector<float>{};
    const auto wf_p = nbits == 32
                          ? std::vector<uint8_t>{}
                          : make_packed_input(packed_bytes(p.nsamps, nbits));
    const auto run  = [&] {
        if (nbits == 32) {
            fdmt.execute(wf_f, dmt);
        } else {
            fdmt.execute(std::span<const uint8_t>(wf_p), nbits, dmt);
        }
    };
    run();
    for (auto _ : state) {
        run();
    }
    state.counters["chunk"] = static_cast<double>(fdmt.get_chunk_nsamps());
    state.counters["fuse"]  = static_cast<double>(fdmt.get_fuse_levels());
    state.counters["engine_MB"] =
        static_cast<double>(fdmt.get_memory_usage().total()) /
        (1024.0 * 1024.0);
    set_counters(state, plan, p, nthreads);
}

void bm_skew(benchmark::State& state,
             SizeType nbits,
             int nthreads,
             SizeType chunk,
             SizeType level) {
    const auto p    = ref_point(nbits);
    const auto plan = make_plan(p);
    FDMTSkewCpu skew(plan, true, nthreads, chunk, level, nbits);
    std::vector<float> dmt(plan.get_dmt_size());
    const auto wf_f = nbits == 32 ? make_float_input(kNchans * p.nsamps)
                                  : std::vector<float>{};
    const auto wf_p = nbits == 32
                          ? std::vector<uint8_t>{}
                          : make_packed_input(packed_bytes(p.nsamps, nbits));
    const auto run  = [&] {
        if (nbits == 32) {
            skew.execute(wf_f, dmt);
        } else {
            skew.execute(std::span<const uint8_t>(wf_p), nbits, dmt);
        }
    };
    run();
    skew.reset_profile();
    for (auto _ : state) {
        run();
    }
    const auto prof              = skew.profile();
    const double iters           = static_cast<double>(state.iterations());
    state.counters["ms_ring"]    = 1e3 * prof.ring / iters;
    state.counters["ms_stripes"] = 1e3 * prof.stripes / iters;
    state.counters["ms_rows"]    = 1e3 * prof.rows / iters;
    state.counters["ms_merges"]  = 1e3 * prof.merges / iters;
    state.counters["chunk"]      = static_cast<double>(chunk);
    state.counters["level"]      = static_cast<double>(level);
    state.counters["engine_MB"] =
        static_cast<double>(skew.memory_usage().total()) / (1024.0 * 1024.0);
    set_counters(state, plan, p, nthreads);
}

} // namespace

// Tree engine: automatic chunk, and the prototype's 4096.
BENCHMARK_CAPTURE(bm_tree, float32_t8_auto, 32, 8, 0)->UseRealTime();
BENCHMARK_CAPTURE(bm_tree, float32_t8_c4096, 32, 8, 4096)->UseRealTime();
BENCHMARK_CAPTURE(bm_tree, packed8_t8_auto, 8, 8, 0)->UseRealTime();
BENCHMARK_CAPTURE(bm_tree, packed8_t8_c4096, 8, 8, 4096)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, float32_t8_c4096_l4, 32, 8, 4096, 4)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, float32_t8_c4096_l3, 32, 8, 4096, 3)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, float32_t8_c4096_l5, 32, 8, 4096, 5)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, packed8_t8_c4096_l4, 8, 8, 4096, 4)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, packed8_t8_c1024_l4, 8, 8, 1024, 4)->UseRealTime();
BENCHMARK_CAPTURE(bm_tree, float32_t1_c4096, 32, 1, 4096)->UseRealTime();
BENCHMARK_CAPTURE(bm_skew, float32_t1_c4096_l4, 32, 1, 4096, 4)->UseRealTime();

} // namespace dmt
