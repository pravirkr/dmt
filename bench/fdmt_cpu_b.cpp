#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"

namespace dmt {
using algorithms::FDMTCPU;
using algorithms::kFDMTAutoFuse;
using plans::FDMTPlan;

// Helper function to generate random data
template <typename T>
static std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

// Fixture for the FDMT CPU benchmarks
class FDMTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min  = 704.0F;
        f_max  = 1216.0F;
        nchans = 4096;
        tsamp  = 0.00008192F;
        dt_max = 2048;
        // If range is 0, fix nsamps for thread benchmarks
        nsamps = (state.range(0) == 0) ? 1 << 16 : state.range(0);
        // If second range is 0, use first range as nthreads
        nthreads  = (state.range(1) == 0) ? static_cast<int>(state.range(0))
                                          : static_cast<int>(state.range(1));
        gen       = std::mt19937(std::random_device()());
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{};
    size_t nchans{}, dt_max{}, nsamps{};
    int nthreads{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                     "full", false, nthreads);
    }
}

// Default execution config (automatic level fusion).
BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute)
(benchmark::State& state) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                 "full", false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
    state.counters["fuse"] = static_cast<double>(fdmt.get_fuse_levels());
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall)
(benchmark::State& state) {
    FDMTPlan tmp_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(tmp_plan.get_buffer_size(), 0.0F);
    for (auto _ : state) {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                     "full", false, nthreads);
        fdmt.execute(waterfall, dmt);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_threads)
(benchmark::State& state) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, false,
                 "full", false, nthreads);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
}

// Packed low-bit input. Args: (nsamps, nthreads, nbits [32 = float
// reference], int_tree, fuse_levels [-1 = kFDMTAutoFuse, the default]). Valid
// mode with box smearing -- the streaming default -- on the same
// 4096-channel band.
BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_packed)
(benchmark::State& state) {
    const auto nbits = static_cast<SizeType>(state.range(2));
    const auto fuse  = state.range(4) < 0
                           ? kFDMTAutoFuse
                           : static_cast<SizeType>(state.range(4));
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                 "valid", false, nthreads, 1, fuse, state.range(3) != 0);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    if (nbits == 32) {
        for (auto _ : state) {
            fdmt.execute(waterfall, dmt);
        }
    } else {
        const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
        std::vector<uint8_t> packed(nchans * row_bytes);
        std::uniform_int_distribution<int> dis(0, 255);
        std::generate(packed.begin(), packed.end(),
                      [&]() { return static_cast<uint8_t>(dis(gen)); });
        for (auto _ : state) {
            fdmt.execute(std::span<const uint8_t>(packed), nbits, dmt);
        }
    }
    state.counters["fuse"] = static_cast<double>(fdmt.get_fuse_levels());
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(nchans * nsamps));
}

// Level fusion. Args: (nsamps, nthreads, fuse_levels [-1 = kFDMTAutoFuse]).
// Valid mode with box smearing (the streaming default); fuse 0 is the
// original level-by-level path.
BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_fused)
(benchmark::State& state) {
    const auto fuse = state.range(2) < 0
                          ? kFDMTAutoFuse
                          : static_cast<SizeType>(state.range(2));
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1, true,
                 "valid", false, nthreads, 1, fuse);
    std::vector<float> dmt(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall, dmt);
    }
    state.counters["fuse"] = static_cast<double>(fdmt.get_fuse_levels());
    state.SetItemsProcessed(state.iterations() *
                            static_cast<int64_t>(nchans * nsamps));
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 16;

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_planBuffer) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_threads) // NOLINT
    ->ArgsProduct({{0}, {1, 2, 4, 8, 10, 12, 16}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_fused) // NOLINT
    ->ArgsProduct({{1 << 12, 1 << 14, 1 << 16}, {1, 8}, {0, 1, 2, 3, 4, -1}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_packed) // NOLINT
    ->ArgsProduct({{1 << 14}, {1, 8}, {32}, {0}, {0, -1}})
    ->ArgsProduct({{1 << 14}, {1, 8}, {1, 2, 4, 8, 16}, {0, 1}, {0, -1}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// BENCHMARK_MAIN();

} // namespace dmt
