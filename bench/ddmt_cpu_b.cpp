#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include <benchmark/benchmark.h>

#include "cpu/ddmt_cpu_kernels.hpp"
#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/sdmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"

namespace dmt {
using algorithms::DDMT;
using algorithms::SDMT;

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
    DDMT ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
              Exec::cpu(nthreads),
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
    DDMT ddmt(f_min, f_max, nchans, tsamp, dm_max, dm_step, 0.0F,
              Exec::cpu(nthreads), nbits, {}, nbeams);
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

// ============================================================================
// Kernel roofline: sum_into() over cache-resident buffers
// ============================================================================

// dst rows (256 x 512 lanes) += nsrc windows read at per-row offsets, as the
// engine's accumulator passes do. adds_per_s is the per-core ceiling that
// load/store ports and cache-line splits allow.
template <typename T> void BM_ddmt_cpu_kernel_sum(benchmark::State& state) {
    constexpr int kLen  = 512;
    constexpr int kRows = 256;
    const int nsrc      = static_cast<int>(state.range(0));
    const int wlen      = kLen + 256 + algorithms::ddmt_cpu::kPad;
    std::vector<T> win(static_cast<SizeType>(nsrc * wlen), T{1});
    std::vector<T> rows(
        static_cast<SizeType>((kRows * kLen) + algorithms::ddmt_cpu::kPad),
        T{0});
    std::mt19937 gen(7);
    std::vector<int> offs(static_cast<SizeType>(kRows * nsrc));
    for (auto& o : offs) {
        o = static_cast<int>(gen() % 256);
    }
    const T* p[algorithms::ddmt_cpu::kMaxSources];
    for (auto _ : state) {
        for (int r = 0; r < kRows; ++r) {
            for (int k = 0; k < nsrc; ++k) {
                p[k] = win.data() + (k * wlen) + offs[(r * nsrc) + k];
            }
            algorithms::ddmt_cpu::sum_into<algorithms::ddmt_cpu::kMaxSources>(
                rows.data() + (r * kLen), p, nsrc, kLen, /*accumulate=*/true);
        }
        benchmark::DoNotOptimize(rows.data());
    }
    state.counters["adds_per_s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * kRows * kLen * nsrc,
        benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_ddmt_cpu_kernel_sum, float)->Arg(1)->Arg(4)->Arg(8);
BENCHMARK_TEMPLATE(BM_ddmt_cpu_kernel_sum, uint32_t)->Arg(1)->Arg(4)->Arg(8);
BENCHMARK_TEMPLATE(BM_ddmt_cpu_kernel_sum, uint16_t)->Arg(1)->Arg(4)->Arg(8);

// ============================================================================
// DM grid shape: shared partial sums (fine grids) vs direct sums (coarse or
// unsorted grids), and plan construction time
// ============================================================================

namespace {

/// 0: linear grid, ~1 sample per trial across the band; 1: every 16th of
/// those; 2: 256 random, unsorted trials over the same range.
std::vector<float> ddmt_bench_grid(int kind) {
    std::vector<float> dms;
    for (int i = 0; i < 1024; i += kind == 1 ? 16 : 1) {
        dms.push_back(0.035F * static_cast<float>(i));
    }
    if (kind == 2) {
        std::mt19937 gen(3);
        dms.resize(256);
        for (auto& v : dms) {
            v = std::uniform_real_distribution<float>(0.0F, 35.8F)(gen);
        }
    }
    return dms;
}

} // namespace

template <typename Engine> void BM_ddmt_cpu_dm_grid(benchmark::State& state) {
    const auto dms        = ddmt_bench_grid(static_cast<int>(state.range(0)));
    const SizeType nchans = 1024;
    const SizeType nsamps = 8192;
    Engine engine(1000.0F, 1500.0F, nchans, 0.00008192F, dms,
                  Exec::cpu(static_cast<int>(state.range(1))));
    std::mt19937 gen(42);
    const auto wf = generate_random_vector<float>(nchans * nsamps, gen);
    std::vector<float> dmt(dms.size() * nsamps);
    engine.execute(wf, std::span<float>(dmt).first(
                           dms.size() * engine.get_output_nsamps(nsamps)));
    for (auto _ : state) {
        engine.execute(wf, dmt);
        benchmark::DoNotOptimize(dmt.data());
    }
    state.counters["adds_per_s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) *
            static_cast<double>(nchans * dms.size() * nsamps),
        benchmark::Counter::kIsRate);
}

BENCHMARK_TEMPLATE(BM_ddmt_cpu_dm_grid, DDMT)
    ->ArgsProduct({{0, 1, 2}, {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_ddmt_cpu_dm_grid, SDMT)
    ->ArgsProduct({{0, 1, 2}, {1, 8}})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

template <typename Engine> void BM_ddmt_cpu_construct(benchmark::State& state) {
    const auto dms = ddmt_bench_grid(0);
    for (auto _ : state) {
        Engine engine(1000.0F, 1500.0F, 4096, 0.00008192F, dms,
                      Exec::cpu(static_cast<int>(state.range(0))));
        benchmark::DoNotOptimize(&engine);
    }
}

BENCHMARK_TEMPLATE(BM_ddmt_cpu_construct, DDMT)
    ->Arg(1)
    ->Arg(8)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_TEMPLATE(BM_ddmt_cpu_construct, SDMT)
    ->Arg(1)
    ->Arg(8)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

} // namespace dmt
