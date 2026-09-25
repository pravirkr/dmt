#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"

namespace dmt {
using algorithms::DDMTCPU;

namespace {

template <typename T>
std::vector<T> generate_random_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

} // namespace

// ============================================================================
// DDMT CPU Float Block Execution Benchmark
// ============================================================================

class DDMTCPUFloatFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min    = 1000.0F;
        f_max    = 1500.0F;
        nchans   = 1024;
        tsamp    = 0.00008192F;
        dm_max   = 50.0F;
        dm_step  = 1.0F;
        nsamps   = static_cast<SizeType>(state.range(0));
        nthreads = static_cast<int>(state.range(1));
        nbeams = state.range(2) > 0 ? static_cast<SizeType>(state.range(2)) : 1;

        std::mt19937 gen(42);
        waterfall =
            generate_random_vector<float>(nbeams * nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{}, nbeams{1};
    int nthreads{1};
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(DDMTCPUFloatFixture, BM_ddmt_cpu_float_execute)
(benchmark::State& state) {
    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F, nthreads,
                 /*nbits=*/32, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    std::vector<float> dmt(nbeams * dm_count * nsamps_reduced, 0.0F);

    for (auto _ : state) {
        ddmt.reset_history();
        ddmt.execute(waterfall, dmt);
        benchmark::DoNotOptimize(dmt.data());
    }

    const int64_t items = static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(nbeams * nchans * nsamps);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * static_cast<int64_t>(sizeof(float)));
}

// ============================================================================
// DDMT CPU Packed Integer Block Execution Benchmark (1, 2, 4, 8, 16 bits)
// ============================================================================

class DDMTCPUPackedFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min    = 1000.0F;
        f_max    = 1500.0F;
        nchans   = 1024;
        tsamp    = 0.00008192F;
        dm_max   = 50.0F;
        dm_step  = 1.0F;
        nsamps   = static_cast<SizeType>(state.range(0));
        nbits    = static_cast<SizeType>(state.range(1));
        nthreads = static_cast<int>(state.range(2));
        nbeams = state.range(3) > 0 ? static_cast<SizeType>(state.range(3)) : 1;

        const auto in_rows   = nbeams * nchans;
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        waterfall_packed.resize(in_rows * row_bytes);

        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dis(0, 255);
        for (auto& b : waterfall_packed) {
            b = static_cast<uint8_t>(dis(gen));
        }
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{}, nbits{8}, nbeams{1};
    int nthreads{1};
    std::vector<uint8_t> waterfall_packed;
};

BENCHMARK_DEFINE_F(DDMTCPUPackedFixture, BM_ddmt_cpu_packed_execute)
(benchmark::State& state) {
    DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F, nthreads,
                 nbits, {}, nbeams);
    const auto max_delay =
        *std::ranges::max_element(ddmt.get_plan().get_container().delay_table);
    const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
    const auto dm_count       = ddmt.get_plan().get_container().dm_arr.size();
    std::vector<int32_t> dmt(nbeams * dm_count * nsamps_reduced, 0);

    for (auto _ : state) {
        ddmt.reset_history();
        ddmt.execute(waterfall_packed, nsamps, dmt);
        benchmark::DoNotOptimize(dmt.data());
    }

    const int64_t items = static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(nbeams * nchans * nsamps);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(waterfall_packed.size()));
}

// ============================================================================
// Registration
// ============================================================================

// Float execution: nsamps in {2048, 8192}, threads in {1, 8}, beams = 1
BENCHMARK_REGISTER_F(DDMTCPUFloatFixture, BM_ddmt_cpu_float_execute)
    ->ArgsProduct({{2048, 8192}, {1, 8}, {1}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Multi-beam float execution: nsamps = 4096, threads = 8, beams in {1, 4}
BENCHMARK_REGISTER_F(DDMTCPUFloatFixture, BM_ddmt_cpu_float_execute)
    ->ArgsProduct({{4096}, {8}, {1, 4}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Packed precision scaling: nsamps = 4096, nbits in {1, 2, 4, 8, 16}, threads =
// 8, beams = 1
BENCHMARK_REGISTER_F(DDMTCPUPackedFixture, BM_ddmt_cpu_packed_execute)
    ->ArgsProduct({{4096}, {1, 2, 4, 8, 16}, {8}, {1}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Multi-beam packed execution: nsamps = 4096, nbits = 8, threads = 8, beams in
// {1, 4}
BENCHMARK_REGISTER_F(DDMTCPUPackedFixture, BM_ddmt_cpu_packed_execute)
    ->ArgsProduct({{4096}, {8}, {8}, {1, 4}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

} // namespace dmt
