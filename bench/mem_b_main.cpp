// Shared entry point and global operator new/delete overrides for the
// dmt_bench_mem executable. Kept in its own translation unit because a
// replacement allocation/deallocation function must have external linkage,
// must not be declared inline, and must be defined exactly once per
// [new.delete] -- it cannot live in bench_mem_utils.hpp, which is included
// by every algorithm-specific memory benchmark file below.
#include <cstdlib>
#include <new>

#include <benchmark/benchmark.h>

#include "bench_mem_utils.hpp"

namespace dmt::bench {
PeakMemoryManager g_mem_manager; // NOLINT
} // namespace dmt::bench

void* operator new(std::size_t sz) { // NOLINT
    void* ptr = std::malloc(sz);     // NOLINT
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    dmt::bench::g_mem_manager.record_alloc(sz);
    return ptr;
}

void operator delete(void* ptr) noexcept { // NOLINT
    if (ptr != nullptr) {
        std::free(ptr); // NOLINT
    }
}

void operator delete(void* ptr, std::size_t sz) noexcept { // NOLINT
    if (ptr != nullptr) {
        dmt::bench::g_mem_manager.record_free(sz);
        std::free(ptr); // NOLINT
    }
}

int main(int argc, char** argv) { // NOLINT
    ::benchmark::RegisterMemoryManager(&dmt::bench::g_mem_manager);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
