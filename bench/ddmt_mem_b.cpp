#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include "bench_mem_utils.hpp"
#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"

namespace dmt {
using algorithms::DDMTCPU;

namespace {

template <typename T>
std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

} // namespace

// ============================================================================
// DDMT (float) memory footprint
// ============================================================================

class DDMTMemFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min     = 1000.0F;
        f_max     = 1500.0F;
        nchans    = 1024;
        tsamp     = 0.00008192F;
        dm_max    = 50.0F;
        dm_step   = 1.0F;
        nsamps    = static_cast<SizeType>(state.range(0));
        gen       = std::mt19937(42);
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(DDMTMemFixture, BM_ddmt_memory_execute_float)
(benchmark::State& state) {
    for (auto _ : state) {
        DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
                     /*nthreads=*/1, /*nbits=*/32, {}, /*nbeams=*/1);
        state.PauseTiming();
        const auto nsamps_out = ddmt.get_output_nsamps(nsamps);
        const auto dm_count   = ddmt.get_plan().get_container().dm_arr.size();
        std::vector<float> dmt(dm_count * nsamps_out, 0.0F);
        state.ResumeTiming();
        ddmt.execute(waterfall, dmt);
    }
    bench::report_memory_counters(state);
}

BENCHMARK_REGISTER_F(DDMTMemFixture, BM_ddmt_memory_execute_float) // NOLINT
    ->Arg(2048)
    ->Arg(4096)
    ->Arg(8192)
    ->Iterations(4)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// ============================================================================
// DDMT (packed low-bit) memory footprint
// ============================================================================

class DDMTPackedMemFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min   = 1000.0F;
        f_max   = 1500.0F;
        nchans  = 1024;
        tsamp   = 0.00008192F;
        dm_max  = 50.0F;
        dm_step = 1.0F;
        nsamps  = static_cast<SizeType>(state.range(0));
        nbits   = static_cast<SizeType>(state.range(1));

        const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
        waterfall_packed.resize(nchans * row_bytes);
        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dis(0, 255);
        for (auto& b : waterfall_packed) {
            b = static_cast<uint8_t>(dis(gen));
        }
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{}, f_max{}, tsamp{}, dm_max{}, dm_step{};
    SizeType nchans{}, nsamps{}, nbits{8};
    std::vector<uint8_t> waterfall_packed;
};

BENCHMARK_DEFINE_F(DDMTPackedMemFixture, BM_ddmt_memory_execute_packed)
(benchmark::State& state) {
    for (auto _ : state) {
        DDMTCPU ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
                     /*nthreads=*/1, nbits, {}, /*nbeams=*/1);
        state.PauseTiming();
        const auto nsamps_out = ddmt.get_output_nsamps(nsamps);
        const auto dm_count   = ddmt.get_plan().get_container().dm_arr.size();
        std::vector<int32_t> dmt(dm_count * nsamps_out, 0);
        state.ResumeTiming();
        ddmt.execute(waterfall_packed, nsamps, dmt);
    }
    bench::report_memory_counters(state);
}

BENCHMARK_REGISTER_F(DDMTPackedMemFixture, BM_ddmt_memory_execute_packed) // NOLINT
    ->ArgsProduct({{2048, 4096, 8192}, {1, 2, 4, 8, 16}})
    ->Iterations(4)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

} // namespace dmt
