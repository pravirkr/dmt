#pragma once

// Peak device memory of one GPU suite point (see suite_memory.hpp for the
// host side). The CUDA and HIP runtimes keep no allocation high-water mark,
// so a thread samples the device's used memory (cudaMemGetInfo) every
// millisecond while the point builds its engine and input and warms up;
// pause() stops it before the timed loop (the engines allocate nothing in
// steady state, and the sampler must not compete with kernel launches), and
// report() takes one last sample. peak_device_bytes is the highest use above
// the level when the point began (the runtime context and anything still
// held by earlier points are excluded); allocations that live less than a
// millisecond can be missed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include <benchmark/benchmark.h>

#include "dmt/gpu_compat.cuh"

#include "suite_memory.hpp"

namespace dmt::bench_suite {

class DevicePeakSampler {
public:
    DevicePeakSampler()                                    = default;
    DevicePeakSampler(const DevicePeakSampler&)            = delete;
    DevicePeakSampler& operator=(const DevicePeakSampler&) = delete;
    ~DevicePeakSampler() { pause(); }

    void begin() {
        pause();
        cudaGetDevice(&m_device);
        m_base = used();
        m_peak.store(m_base);
        m_run.store(true);
        m_thread = std::thread([this] {
            cudaSetDevice(m_device);
            while (m_run.load(std::memory_order_relaxed)) {
                sample();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    void pause() {
        if (m_thread.joinable()) {
            m_run.store(false);
            m_thread.join();
            sample();
        }
    }

    [[nodiscard]] int64_t peak_bytes() {
        pause();
        sample();
        return std::max<int64_t>(0, m_peak.load() - m_base);
    }

    void report(benchmark::State& state) {
        state.counters["peak_device_bytes"] = static_cast<double>(peak_bytes());
    }

private:
    [[nodiscard]] static int64_t used() {
        std::size_t free_b  = 0;
        std::size_t total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            return 0;
        }
        return static_cast<int64_t>(total_b - free_b);
    }
    void sample() {
        const int64_t u = used();
        int64_t prev    = m_peak.load();
        while (u > prev && !m_peak.compare_exchange_weak(prev, u)) {
        }
    }

    int m_device{0};
    int64_t m_base{0};
    std::atomic<int64_t> m_peak{0};
    std::atomic<bool> m_run{false};
    std::thread m_thread;
};

/// Host peak RSS and device peak of one GPU point.
struct GpuPeakMemory {
    PeakRss host;
    DevicePeakSampler device;
    void begin() {
        host.begin();
        device.begin();
    }
    void pause() { device.pause(); }
    void report(benchmark::State& state) {
        host.report(state);
        device.report(state);
    }
};

} // namespace dmt::bench_suite
