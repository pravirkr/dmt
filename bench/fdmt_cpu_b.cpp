#include <algorithm>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <dmt/common/plans.hpp>
#include <dmt/fdmt/fdmt_cpu.hpp>

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
        nsamps = state.range(0);
        std::random_device rd;
        gen       = std::mt19937(rd());
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{};
    float f_max{};
    size_t nchans{};
    float tsamp{};
    size_t dt_max{};
    size_t nsamps{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

class FDMTThreadsFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min    = 704.0F;
        f_max    = 1216.0F;
        nchans   = 4096;
        tsamp    = 0.00008192F;
        dt_max   = 2048;
        nsamps   = 65536;
        nthreads = state.range(0);
        std::random_device rd;
        gen       = std::mt19937(rd());
        waterfall = generate_vector<float>(nchans * nsamps, gen);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    float f_min{};
    float f_max{};
    size_t nchans{};
    float tsamp{};
    size_t dt_max{};
    size_t nsamps{};
    size_t nthreads{};
    std::mt19937 gen;
    std::vector<float> waterfall;
};

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer_seq)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTCPU::set_num_threads(1);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer_par)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTCPU::set_num_threads(8);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_initialise_seq)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(1);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> state_init(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.initialise(waterfall.data(), waterfall.size(), state_init.data(),
                        state_init.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_initialise_par)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(8);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> state_init(fdmt.get_plan().get_buffer_size(), 0.0F);
    for (auto _ : state) {
        fdmt.initialise(waterfall.data(), waterfall.size(), state_init.data(),
                        state_init.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_seq)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(1);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_par)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(8);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall_seq)
(benchmark::State& state) {
    FDMTPlan tmp_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(tmp_plan.get_dmt_size(), 0.0F);
    for (auto _ : state) {
        FDMTCPU::set_num_threads(1);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall_par)
(benchmark::State& state) {
    FDMTPlan tmp_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(tmp_plan.get_dmt_size(), 0.0F);
    for (auto _ : state) {
        FDMTCPU::set_num_threads(8);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTThreadsFixture, BM_fdmt_execute_threads)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(static_cast<int>(nthreads));
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 16;

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_planBuffer_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_planBuffer_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_initialise_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_initialise_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps)
    ->MeasureProcessCPUTime()
    ->UseRealTime();
BENCHMARK_REGISTER_F(FDMTThreadsFixture, BM_fdmt_execute_threads)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(10)
    ->Arg(12)
    ->Arg(16)
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// BENCHMARK_MAIN();
