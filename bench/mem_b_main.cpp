// Shared entry point and global operator new/delete overrides for the
// dmt_bench_mem executable. Kept in its own translation unit because a
// replacement allocation/deallocation function must have external linkage,
// must not be declared inline, and must be defined exactly once per
// [new.delete] -- it cannot live in bench_mem_utils.hpp, which is included
// by every algorithm-specific memory benchmark file below.
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#include <benchmark/benchmark.h>

#include "bench_mem_utils.hpp"

namespace dmt::bench {
PeakMemoryManager g_mem_manager; // NOLINT
} // namespace dmt::bench

namespace {

constexpr std::size_t kHeaderBytes =
    std::max(sizeof(std::size_t), alignof(std::max_align_t));

void* alloc_tracked(std::size_t sz) noexcept {
    if (sz > std::numeric_limits<std::size_t>::max() - kHeaderBytes) {
        return nullptr;
    }
    void* raw = std::malloc(kHeaderBytes + sz); // NOLINT
    if (raw == nullptr) {
        return nullptr;
    }
    std::memcpy(raw, &sz, sizeof(sz));
    dmt::bench::g_mem_manager.record_alloc(sz);
    return static_cast<char*>(raw) + kHeaderBytes;
}

void free_tracked(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    void* raw      = static_cast<char*>(ptr) - kHeaderBytes;
    std::size_t sz = 0;
    std::memcpy(&sz, raw, sizeof(sz));
    dmt::bench::g_mem_manager.record_free(sz);
    std::free(raw); // NOLINT
}

} // namespace

void* operator new(std::size_t sz) { // NOLINT
    if (void* ptr = alloc_tracked(sz)) {
        return ptr;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t sz, const std::nothrow_t&) noexcept { // NOLINT
    return alloc_tracked(sz);
}

void* operator new[](std::size_t sz) { // NOLINT
    return ::operator new(sz);
}

void* operator new[](std::size_t sz,
                     const std::nothrow_t&) noexcept { // NOLINT
    return alloc_tracked(sz);
}

void operator delete(void* ptr) noexcept { // NOLINT
    free_tracked(ptr);
}

void operator delete(void* ptr, std::size_t /*sz*/) noexcept { // NOLINT
    free_tracked(ptr);
}

void operator delete(void* ptr, const std::nothrow_t&) noexcept { // NOLINT
    free_tracked(ptr);
}

void operator delete[](void* ptr) noexcept { // NOLINT
    free_tracked(ptr);
}

void operator delete[](void* ptr, std::size_t /*sz*/) noexcept { // NOLINT
    free_tracked(ptr);
}

void operator delete[](void* ptr, const std::nothrow_t&) noexcept { // NOLINT
    free_tracked(ptr);
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
