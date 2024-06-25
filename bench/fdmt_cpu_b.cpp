#include <benchmark/benchmark.h>

#include <algorithm>
#include <dmt/fdmt_cpu.hpp>
#include <random>
#include <vector>

class FDMTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min  = 704.0F;
        f_max  = 1216.0F;
        nchans = 4096;
        tsamp  = 0.00008192F;
        dt_max = 2048;
        nsamps = state.range(0);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    template <typename T>
    std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
        std::vector<T> vec(size);
        std::uniform_real_distribution<T> dis(0.0, 1.0);
        std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
        return vec;
    }

    float f_min{};
    float f_max{};
    size_t nchans{};
    float tsamp{};
    size_t dt_max{};
    size_t nsamps{};
};

class FDMTThreadsFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min  = 704.0F;
        f_max  = 1216.0F;
        nchans = 4096;
        tsamp  = 0.00008192F;
        dt_max = 2048;
        nsamps = 65536;
        nthreads = state.range(0);
    }

    void TearDown(const ::benchmark::State& /*unused*/) override {}

    template <typename T>
    std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
        std::vector<T> vec(size);
        std::uniform_real_distribution<T> dis(0.0, 1.0);
        std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
        return vec;
    }

    float f_min{};
    float f_max{};
    size_t nchans{};
    float tsamp{};
    size_t dt_max{};
    size_t nsamps{};
    size_t nthreads{};
};

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer_seq)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_planBuffer_par)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(8);
    for (auto _ : state) {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_initialise_seq)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(1);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall        = generate_vector<float>(nchans * nsamps, gen);
    const auto& plan      = fdmt.get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    std::vector<float> state_init(state_size, 0.0F);
    for (auto _ : state) {
        fdmt.initialise(waterfall.data(), waterfall.size(), state_init.data(),
                        state_init.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_initialise_par)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(8);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall        = generate_vector<float>(nchans * nsamps, gen);
    const auto& plan      = fdmt.get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    std::vector<float> state_init(state_size, 0.0F);
    for (auto _ : state) {
        fdmt.initialise(waterfall.data(), waterfall.size(), state_init.data(),
                        state_init.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_seq)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(1);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall = generate_vector<float>(nchans * nsamps, gen);
    std::vector<float> dmt(fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_execute_par)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(8);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall = generate_vector<float>(nchans * nsamps, gen);
    std::vector<float> dmt(fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
    for (auto _ : state) {
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall_seq)
(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall = generate_vector<float>(nchans * nsamps, gen);

    for (auto _ : state) {
        FDMTCPU::set_num_threads(1);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        state.PauseTiming();
        std::vector<float> dmt(fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
        state.ResumeTiming();
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall_par)
(benchmark::State& state) {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall = generate_vector<float>(nchans * nsamps, gen);

    for (auto _ : state) {
        FDMTCPU::set_num_threads(8);
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);
        state.PauseTiming();
        std::vector<float> dmt(fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
        state.ResumeTiming();
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

BENCHMARK_DEFINE_F(FDMTThreadsFixture, BM_fdmt_execute_threads)
(benchmark::State& state) {
    FDMTCPU::set_num_threads(nthreads);
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max);

    std::random_device rd;
    std::mt19937 gen(rd());
    auto waterfall = generate_vector<float>(nchans * nsamps, gen);
    std::vector<float> dmt(fdmt.get_dt_grid_final().size() * nsamps, 0.0F);
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
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_initialise_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_initialise_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_execute_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall_seq)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall_par)
    ->RangeMultiplier(2)
    ->Range(kMinNsamps, kMaxNsamps);
BENCHMARK_REGISTER_F(FDMTThreadsFixture, BM_fdmt_execute_threads)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(10)->Arg(12)->Arg(16);

// BENCHMARK_MAIN();
