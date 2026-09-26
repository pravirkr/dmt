#pragma once

// Published benchmark suite (dmt_bench_suite): one fixed data configuration
// shared by every algorithm and backend, so FDMT, FDMT-FFT and DDMT land on
// the same plots. Only the block length (nsamps), the number of DM trials
// (via dt_max) and the input bit width are swept. See bench/README.md.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::bench_suite {

// Fixed configuration: 4096 channels over 704-1216 MHz at 81.92 us.
inline constexpr float kFMin      = 704.0F;
inline constexpr float kFMax      = 1216.0F;
inline constexpr float kTsamp     = 8.192e-5F;
inline constexpr SizeType kNchans = 4096;

// Reference point every sweep passes through.
inline constexpr SizeType kNsampsRef = 16384;
inline constexpr IndexType kDtMaxRef = 2048; // 2049 DM trials

inline const std::vector<SizeType> kNsampsSweep = {4096, 8192, 16384, 32768,
                                                   65536};
inline const std::vector<IndexType> kDtMaxSweep = {256, 512, 1024, 2048, 4096};
inline const std::vector<SizeType> kNbitsSweep  = {1, 2, 4, 8, 16, 32};

enum class Sweep { kNsamps, kNdms, kNbits };
enum class Algo { kFDMT, kFDMTFFT, kDDMT };

struct Point {
    Sweep sweep;
    SizeType nsamps;
    IndexType dt_max;
    SizeType nbits;
};

inline std::string_view sweep_name(Sweep s) {
    switch (s) {
    case Sweep::kNsamps:
        return "nsamps";
    case Sweep::kNdms:
        return "ndms";
    case Sweep::kNbits:
        return "nbits";
    }
    return "?";
}

inline std::string_view algo_name(Algo a) {
    switch (a) {
    case Algo::kFDMT:
        return "FDMT";
    case Algo::kFDMTFFT:
        return "FDMT-FFT";
    case Algo::kDDMT:
        return "DDMT";
    }
    return "?";
}

/// Every point of one sweep (the other two axes at their reference values).
inline std::vector<Point> sweep_points(Sweep sweep) {
    std::vector<Point> points;
    switch (sweep) {
    case Sweep::kNsamps:
        for (const auto n : kNsampsSweep) {
            points.push_back({sweep, n, kDtMaxRef, 32});
        }
        break;
    case Sweep::kNdms:
        for (const auto dt : kDtMaxSweep) {
            points.push_back({sweep, kNsampsRef, dt, 32});
        }
        break;
    case Sweep::kNbits:
        for (const auto b : kNbitsSweep) {
            points.push_back({sweep, kNsampsRef, kDtMaxRef, b});
        }
        break;
    }
    return points;
}

/// CPU thread counts: DMT_BENCH_THREADS="1,8" (the default).
inline std::vector<int> cpu_threads() {
    const char* env        = std::getenv("DMT_BENCH_THREADS"); // NOLINT
    const std::string spec = (env != nullptr) ? env : "1,8";
    std::vector<int> threads;
    std::size_t start = 0;
    while (start < spec.size()) {
        const auto end = spec.find(',', start);
        threads.push_back(std::stoi(spec.substr(start, end - start)));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return threads;
}

/// The multi-threaded CPU count (largest requested) used for DDMT and
/// FDMT-FFT, whose single-thread runs are too slow to sweep.
inline int cpu_threads_max() {
    int best = 1;
    for (const int t : cpu_threads()) {
        best = std::max(best, t);
    }
    return best;
}

/**
 * Benchmark name, parsed by bench/scripts/plot_suite.py:
 * "suite/<sweep>/<algo>/<backend>/nsamps:<n>/dtmax:<dt>/nbits:<b>".
 * backend is "cpu<threads>", "cuda" (device-resident) or "cuda_host"
 * (host arrays, including PCIe transfers).
 */
inline std::string
bench_name(Algo algo, std::string_view backend, const Point& p) {
    return std::format("suite/{}/{}/{}/nsamps:{}/dtmax:{}/nbits:{}",
                       sweep_name(p.sweep), algo_name(algo), backend, p.nsamps,
                       p.dt_max, p.nbits);
}

/// The FDMT plan of a point; its final DM grid is given to DDMT too, so all
/// three algorithms compute the identical trial set.
inline plans::FDMTPlan make_plan(const Point& p) {
    return {kFMin, kFMax, kNchans, p.nsamps, kTsamp, p.dt_max, 0, 1, "valid"};
}

/// Memory budget in bytes: DMT_BENCH_MAX_GB (set by run_suite.py; 16 GB if
/// unset). Points estimated above it are skipped, not run.
inline double memory_budget_bytes() {
    const char* env = std::getenv("DMT_BENCH_MAX_GB"); // NOLINT
    const double gb = (env != nullptr) ? std::stod(env) : 16.0;
    return gb * 1024.0 * 1024.0 * 1024.0;
}

/// Estimated working-set bytes of one engine at point `p` (input, output and
/// the engine's own buffers), used only for the skip decision.
inline double
estimate_bytes(Algo algo, const plans::FDMTPlan& plan, const Point& p) {
    const double input =
        static_cast<double>(kNchans * p.nsamps) *
        ((p.nbits == 32) ? 4.0 : static_cast<double>(p.nbits) / 8.0);
    const double b = static_cast<double>(plan.get_buffer_size()) * 4.0;
    const double d = static_cast<double>(plan.get_dmt_size()) * 4.0;
    switch (algo) {
    case Algo::kFDMT: // caller buffer + internal ping-pong half
        return input + (2.0 * b);
    case Algo::kFDMTFFT: // 2 complex ping-pong + IFFT buffers + time rows
        return input + d +
               (32.0 * static_cast<double>(plan.get_fft_buffer_size()));
    case Algo::kDDMT: // output + one block of history per channel
        return input + d + (static_cast<double>(kNchans) * 4.0 * p.dt_max);
    }
    return 0.0;
}

/// Machine-readable counters (the name carries the same information).
inline void set_counters(benchmark::State& state,
                         const plans::FDMTPlan& plan,
                         const Point& p,
                         int nthreads) {
    state.counters["nsamps"]   = static_cast<double>(p.nsamps);
    state.counters["ndms"]     = static_cast<double>(plan.get_dmt_ndms());
    state.counters["nchans"]   = static_cast<double>(kNchans);
    state.counters["nbits"]    = static_cast<double>(p.nbits);
    state.counters["nthreads"] = static_cast<double>(nthreads);
    state.counters["block_s"]  = static_cast<double>(p.nsamps) * kTsamp;
}

/// Deterministic input: float samples in [0, 1) or packed random bytes.
inline std::vector<float> make_float_input(SizeType n) {
    std::vector<float> v(n);
    unsigned x = 12345U;
    for (auto& e : v) {
        x = (x * 1664525U) + 1013904223U;
        e = static_cast<float>(x >> 8) / 16777216.0F;
    }
    return v;
}

inline std::vector<uint8_t> make_packed_input(SizeType nbytes) {
    std::vector<uint8_t> v(nbytes);
    unsigned x = 67890U;
    for (auto& e : v) {
        x = (x * 1664525U) + 1013904223U;
        e = static_cast<uint8_t>(x >> 24);
    }
    return v;
}

inline SizeType packed_bytes(SizeType nsamps, SizeType nbits) {
    return kNchans * (((nsamps * nbits) + 7) / 8);
}

} // namespace dmt::bench_suite
