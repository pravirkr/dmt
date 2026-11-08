#pragma once

// Peak memory of one suite point, recorded next to its timing (run_suite.py
// writes both into the same JSON):
// - peak_rss_bytes: the process's peak resident set size while the point
//   built its engine and input, warmed up and ran its timed blocks. On
//   Linux the kernel's high-water mark (VmHWM) is reset when the point
//   starts (writing 5 to /proc/self/clear_refs), so timing and memory come
//   from the same run; base_rss_bytes is the resident size at that reset
//   (the benchmark process itself: binary, libraries, runtime contexts;
//   freed heap is returned to the system first).
//   Where the reset is unavailable, the counter is the monotonic
//   getrusage() peak and rss_peak_resettable is 0.
// - peak_device_bytes (GPU suites, suite_memory_gpu.cuh): device memory in
//   use above the point's starting level, sampled every millisecond.

#include <cstdint>
#include <fstream>
#include <string>

#include <benchmark/benchmark.h>
#include <sys/resource.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace dmt::bench_suite {

/// A field of /proc/self/status in bytes ("VmHWM", "VmRSS"), or -1.
inline int64_t proc_status_bytes(const std::string& key) {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.size() > key.size() && line.starts_with(key) &&
            line[key.size()] == ':') {
            return std::stoll(line.substr(key.size() + 1)) * 1024; // kB
        }
    }
    return -1;
}

/// getrusage() peak RSS in bytes (monotonic over the process lifetime).
inline int64_t rusage_peak_bytes() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return usage.ru_maxrss; // NOLINT: bytes
#else
    return usage.ru_maxrss * 1024; // NOLINT: KiB
#endif
}

/// Peak resident set size of one benchmark point: begin() before the
/// engine is built, report() after the timed loop.
class PeakRss {
public:
    void begin() {
#if defined(__GLIBC__)
        // Earlier points' freed heap would otherwise stay resident.
        malloc_trim(0);
#endif
        std::ofstream refs("/proc/self/clear_refs");
        refs << "5";
        refs.flush();
        m_resettable = refs.good() && proc_status_bytes("VmHWM") >= 0;
        m_base       = m_resettable ? proc_status_bytes("VmRSS") : 0;
    }

    [[nodiscard]] int64_t peak_bytes() const {
        return m_resettable ? proc_status_bytes("VmHWM") : rusage_peak_bytes();
    }

    void report(benchmark::State& state) const {
        state.counters["peak_rss_bytes"] = static_cast<double>(peak_bytes());
        state.counters["base_rss_bytes"] = static_cast<double>(m_base);
        if (!m_resettable) {
            state.counters["rss_peak_resettable"] = 0;
        }
    }

private:
    int64_t m_base{0};
    bool m_resettable{false};
};

} // namespace dmt::bench_suite
