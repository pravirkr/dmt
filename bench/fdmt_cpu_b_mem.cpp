#include <cstdint>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <dmt/common/plans.hpp>
#include <dmt/fdmt/fdmt_cpu.hpp>

// Custom memory manager to track allocations
class CustomMemoryManager : public benchmark::MemoryManager {
public:
    void Start() BENCHMARK_OVERRIDE {
        m_bytes_allocated  = 0;
        m_allocation_count = 0;
    }

    void Stop(Result& result) BENCHMARK_OVERRIDE {
        result.num_allocs     = static_cast<int64_t>(m_allocation_count);
        result.max_bytes_used = static_cast<int64_t>(m_bytes_allocated);
    }

    std::pair<void*, std::size_t> allocate(std::size_t size) {
        void* ptr = std::malloc(size); // NOLINT
        if (ptr != nullptr) {
            m_bytes_allocated += size;
            ++m_allocation_count;
        }
        return {ptr, size};
    }

    void deallocate(void* ptr, std::size_t size) {
        std::free(ptr); // NOLINT
        m_bytes_allocated -= size;
    }

private:
    std::size_t m_bytes_allocated{};
    std::size_t m_allocation_count{};
};

// Global new and delete operators to use our custom memory manager
CustomMemoryManager custom_mm; // NOLINT

void* operator new(std::size_t sz) { // NOLINT
    auto [ptr, allocated_size] = custom_mm.allocate(sz);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void operator delete(void* ptr, std::size_t size) noexcept { // NOLINT
    custom_mm.deallocate(ptr, size);
}

// Helper function to generate random data
template <typename T>
static std::vector<T> generate_vector(size_t size, std::mt19937& gen) {
    std::vector<T> vec(size);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dis(gen); });
    return vec;
}

class FDMTCPUFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        f_min    = 704.0F;
        f_max    = 1216.0F;
        nchans   = 4096;
        tsamp    = 0.00008192F;
        dt_max   = 2048;
        nsamps   = state.range(0);
        nthreads = static_cast<int>(state.range(1));
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

BENCHMARK_DEFINE_F(FDMTCPUFixture, BM_fdmt_overall_memory_usage)
(benchmark::State& state) {
    for (auto _ : state) {
        FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, 1, 0,
                     nthreads);
        state.PauseTiming();
        std::vector<float> dmt(fdmt.get_plan().get_dmt_size());
        state.ResumeTiming();
        fdmt.execute(waterfall.data(), waterfall.size(), dmt.data(),
                     dmt.size());
    }
}

constexpr size_t kMinNsamps = 1 << 11;
constexpr size_t kMaxNsamps = 1 << 16;

BENCHMARK_REGISTER_F(FDMTCPUFixture, BM_fdmt_overall_memory_usage) // NOLINT
    ->ArgsProduct({benchmark::CreateRange(kMinNsamps, kMaxNsamps, 2), {1, 8}})
    ->MeasureProcessCPUTime()
    ->UseRealTime();

// Separate main function for memory benchmarks
// BENCHMARK_MAIN();
int main(int argc, char** argv) {
    ::benchmark::RegisterMemoryManager(&custom_mm);
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
}