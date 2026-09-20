#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sys/resource.h>

#include <benchmark/benchmark.h>

namespace dmt::bench {

/**
 * @brief Tracks heap allocations routed through operator new/delete during
 * a benchmark::MemoryManager Start()/Stop() window.
 * @details
 * Reports the true high-water mark of outstanding bytes (`max_bytes_used`),
 * not just whatever is left over at Stop() time -- a naive tracker that
 * only nets allocations against deallocations collapses to ~0 for any
 * benchmark loop that frees everything before returning, which is most of
 * them. `total_allocated_bytes` is the cumulative sum of every allocation
 * request in the window (not netted against frees), i.e. the "total RAM
 * churned through" figure; `net_heap_growth` is what's still outstanding
 * at Stop().
 *
 * Holds only trivial std::size_t members (no mutex, no container) so the
 * process-wide instance is safe to touch from operator new/delete even
 * before its own static initialization has run -- other translation
 * units' static initializers can invoke global operator new before
 * `main()`, and a tracker with a non-trivial constructor would be
 * undefined behavior at that point.
 */
class PeakMemoryManager : public benchmark::MemoryManager {
public:
    void Start() override {
        m_current = 0;
        m_peak    = 0;
        m_total   = 0;
        m_allocs  = 0;
    }

    void Stop(Result& result) override {
        result.num_allocs            = static_cast<int64_t>(m_allocs);
        result.max_bytes_used        = static_cast<int64_t>(m_peak);
        result.total_allocated_bytes = static_cast<int64_t>(m_total);
        result.net_heap_growth       = static_cast<int64_t>(m_current);
    }

    void record_alloc(std::size_t n) noexcept {
        m_current += n;
        m_total += n;
        ++m_allocs;
        m_peak = std::max(m_peak, m_current);
    }

    void record_free(std::size_t n) noexcept { m_current -= std::min(n, m_current); }

    /// True high-water mark of outstanding bytes since the last Start(), in
    /// MiB. Iteration-count-invariant: every benchmark iteration's objects
    /// are destroyed before the next one begins, so this reflects a single
    /// iteration's peak working set regardless of how many iterations ran.
    [[nodiscard]] double peak_mb() const noexcept {
        return static_cast<double>(m_peak) / (1024.0 * 1024.0);
    }

    /// Cumulative bytes ever requested since the last Start(), summed over
    /// every iteration that ran -- divide by the iteration count for a
    /// per-iteration figure.
    [[nodiscard]] std::size_t total_bytes() const noexcept { return m_total; }

private:
    std::size_t m_current{0};
    std::size_t m_peak{0};
    std::size_t m_total{0};
    std::size_t m_allocs{0};
};

// Single process-wide instance, defined in mem_b_main.cpp and registered
// there with benchmark::RegisterMemoryManager. Replacement operator
// new/delete must live in exactly one non-inline translation unit per
// [new.delete] (they must not be declared inline, and defining them in a
// header included by multiple .cpp files would violate the ODR), so they
// are also defined there rather than here.
extern PeakMemoryManager g_mem_manager; // NOLINT

/// Process-wide peak resident set size (high-water mark since process
/// start), in MiB. Complements PeakMemoryManager: this is charged by the OS
/// for everything (thread stacks, mmap'd regions, allocations that bypass
/// operator new), at the cost of being a whole-process monotonic counter
/// rather than a per-iteration one.
inline double get_process_peak_rss_mb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    constexpr double kBytesPerUnit = 1.0; // Darwin reports ru_maxrss in bytes
#else
    constexpr double kBytesPerUnit = 1024.0; // Linux reports ru_maxrss in KiB
#endif
    return (static_cast<double>(usage.ru_maxrss) * kBytesPerUnit) / (1024.0 * 1024.0);
}

/// Reports the standard set of memory counters for a benchmark case: process
/// peak RSS, peak heap high-water mark, and total heap churned per
/// iteration. Call once after the timed `for (auto _ : state)` loop.
inline void report_memory_counters(benchmark::State& state) {
    state.counters["ProcessPeakRSS_MB"] = get_process_peak_rss_mb();
    state.counters["PeakHeap_MB"]       = g_mem_manager.peak_mb();
    state.counters["TotalAlloc_MB_per_iter"] =
        static_cast<double>(g_mem_manager.total_bytes()) /
        (1024.0 * 1024.0 * static_cast<double>(std::max<int64_t>(1, state.iterations())));
}

} // namespace dmt::bench
