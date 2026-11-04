#pragma once

// Published benchmark suite (dmt_bench_suite): one data configuration shared
// by every algorithm and backend, so FDMT, FDMT-FFT (integer and fractional
// merge delays), DDMT, SDMT and DDMT-FFT (NUFFT and brute force) land on the
// same plots. DMT_BENCH_CONFIG selects cryopaf (512 channels),
// cryopaf_lband (1152 channels) or parkes_uwl (6656 channels). Only block
// length, DM range and input bit width are swept. See bench/README.md.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <benchmark/benchmark.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::bench_suite {

// Configurations selected with DMT_BENCH_CONFIG (run_suite.py --config),
// all on the piecewise-uniform Levin grid
// from DM 0 (every engine gets the same grid; the FDMT engines round it to
// integer delays, and DDMT-FFT runs one NUFFT per uniform segment):
// - "cryopaf" (default): 512 channels over 704-1216 MHz at 512 us,
//   DM 0-2000, tolerance 1.1.
// - "cryopaf_lband": 1152 channels over 1100-1714 MHz at 455.6 us,
//   DM 0-2000, tolerance 1.1, with one search subband level.
// - "parkes_uwl": 6656 channels over 704-4032 MHz at 64 us,
//   DM 0-2000, tolerance 1.6 (4749 trials; 1.1 would need 12941). The
//   dispersion sweep spans multiple blocks.
// The benchmark names do not carry the configuration; run_suite.py writes
// each one to its own results tree.
struct SuiteConfig {
    std::string_view name;
    float f_min;
    float f_max;
    float tsamp;
    SizeType nchans;
    float levin_tol;
    // Reference point every sweep passes through.
    SizeType nsamps_ref;
    float dm_max_ref;
    std::vector<SizeType> nsamps_sweep;
    std::vector<float> dm_max_sweep;
    /// FDMTBoxcar search depth (plan subband_depth); transform-only runs
    /// are unchanged.
    SizeType subband_depth{0};
};

inline const SuiteConfig& suite_config() {
    static const SuiteConfig cfg = [] {
        // Reference blocks retain 32768 samples for all presets.
        SuiteConfig cryopaf{"cryopaf",
                            704.0F,
                            1216.0F,
                            5.12e-4F,
                            512,
                            1.1F,
                            32768,
                            2000.0F,
                            {8192, 16384, 32768, 65536},
                            {125.0F, 250.0F, 500.0F, 1000.0F, 2000.0F}};
        // DM 2000 sweeps 4.03365 s (8853.49 samples); the covering
        // Levin grid ends at DM 2005.36 (8877 rounded delay samples).
        // The 32768-sample reference block spans 14.9291 seconds.
        SuiteConfig cryopaf_lband{"cryopaf_lband",
                                  1100.0F,
                                  1714.0F,
                                  455.6e-6F,
                                  1152,
                                  1.1F,
                                  32768,
                                  2000.0F,
                                  {8192, 16384, 32768, 65536},
                                  {125.0F, 250.0F, 500.0F, 1000.0F, 2000.0F},
                                  1};
        SuiteConfig parkes_uwl{"parkes_uwl",
                               704.0F,
                               4032.0F,
                               64e-6F,
                               6656,
                               1.6F,
                               32768,
                               2000.0F,
                               {8192, 16384, 32768, 65536},
                               {125.0F, 250.0F, 500.0F, 1000.0F, 2000.0F}};
        const char* env = std::getenv("DMT_BENCH_CONFIG"); // NOLINT
        const std::string_view sel =
            (env != nullptr && *env != '\0') ? env : "cryopaf";
        if (sel == "cryopaf") {
            return cryopaf;
        }
        if (sel == "parkes_uwl") {
            return parkes_uwl;
        }
        if (sel == "cryopaf_lband") {
            return cryopaf_lband;
        }
        throw std::invalid_argument(
            std::format("DMT_BENCH_CONFIG={}: expected cryopaf, cryopaf_lband "
                        "or parkes_uwl",
                        sel));
    }();
    return cfg;
}

// Shorthands for the active configuration.
inline const float kFMin                         = suite_config().f_min;
inline const float kFMax                         = suite_config().f_max;
inline const float kTsamp                        = suite_config().tsamp;
inline const SizeType kNchans                    = suite_config().nchans;
inline const float kLevinTol                     = suite_config().levin_tol;
inline const SizeType kNsampsRef                 = suite_config().nsamps_ref;
inline const float kDmMaxRef                     = suite_config().dm_max_ref;
inline const std::vector<SizeType>& kNsampsSweep = suite_config().nsamps_sweep;
inline const std::vector<float>& kDmMaxSweep     = suite_config().dm_max_sweep;
inline const std::vector<SizeType> kNbitsSweep   = {1, 2, 4, 8, 16, 32};

enum class Sweep { kNsamps, kNdms, kNbits };
enum class Algo {
    kFDMT,
    kFDMTFFT,
    kFDMTFFTFrac,
    kDDMT,
    kSDMT,
    kDDMTFFT,
    kDDMTFFTBrute,
};

struct Point {
    Sweep sweep;
    SizeType nsamps;
    float dm_max;
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
    case Algo::kSDMT:
        return "SDMT";
    case Algo::kFDMTFFTFrac:
        return "FDMT-FFT-frac";
    case Algo::kDDMTFFT:
        return "DDMT-FFT";
    case Algo::kDDMTFFTBrute:
        return "DDMT-FFT-brute";
    }
    return "?";
}

/// Every point of one sweep (the other two axes at their reference values).
inline std::vector<Point> sweep_points(Sweep sweep) {
    std::vector<Point> points;
    switch (sweep) {
    case Sweep::kNsamps:
        for (const auto n : kNsampsSweep) {
            points.push_back({sweep, n, kDmMaxRef, 32});
        }
        break;
    case Sweep::kNdms:
        for (const auto dm : kDmMaxSweep) {
            points.push_back({sweep, kNsampsRef, dm, 32});
        }
        break;
    case Sweep::kNbits:
        for (const auto b : kNbitsSweep) {
            points.push_back({sweep, kNsampsRef, kDmMaxRef, b});
        }
        break;
    }
    return points;
}

/// The reference point (both sweeps through it, float32 input): where the
/// verification-only engines (FDMT-FFT with integer delays, brute-force
/// DDMT-FFT) are timed.
inline bool is_reference(const Point& p) {
    return p.sweep == Sweep::kNdms && p.nsamps == kNsampsRef &&
           p.dm_max == kDmMaxRef && p.nbits == 32;
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
 * "suite/<sweep>/<algo>/<backend>/nsamps:<n>/dmmax:<dm>/nbits:<b>".
 * backend is "cpu<threads>", "cuda" (device-resident) or "cuda_host"
 * (host arrays, including PCIe transfers).
 */
inline std::string
bench_name(Algo algo, std::string_view backend, const Point& p) {
    return std::format("suite/{}/{}/{}/nsamps:{}/dmmax:{:g}/nbits:{}",
                       sweep_name(p.sweep), algo_name(algo), backend, p.nsamps,
                       p.dm_max, p.nbits);
}

/// The DM grid of a point: the piecewise-uniform Levin grid from 0 to
/// dm_max (pulse width one sample), given to every engine.
inline std::vector<float> dm_grid(const Point& p) {
    return plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, p.dm_max, kTsamp, kTsamp, kFMin, kFMax, kNchans, kLevinTol);
}

/// The FDMT plan of a point (its grid rounds dm_grid() to integer delays,
/// keeping every trial: the grid steps are at least 3 samples).
inline plans::FDMTPlan make_plan(const Point& p, SizeType subband_depth = 0) {
    return {kFMin,  kFMax,      kNchans, p.nsamps,
            kTsamp, dm_grid(p), "valid", subband_depth};
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
inline double estimate_bytes(Algo algo,
                             const plans::FDMTPlan& plan,
                             const Point& p,
                             bool device = false) {
    const double input =
        static_cast<double>(kNchans * p.nsamps) *
        ((p.nbits == 32) ? 4.0 : static_cast<double>(p.nbits) / 8.0);
    const double b = static_cast<double>(plan.get_buffer_size()) * 4.0;
    const double d = static_cast<double>(plan.get_dmt_size()) * 4.0;
    // The FDMT runs blocks in chunks whose two state halves fit 2 GiB.
    constexpr double kChunkStateBudget = 2.0 * 1024.0 * 1024.0 * 1024.0;
    switch (algo) {
    case Algo::kFDMT: // output + the engine's two state halves
        return input + d + std::min(2.0 * b, kChunkStateBudget) +
               static_cast<double>(plan.get_history_size() +
                                   plan.get_tree_history_size()) *
                   sizeof(float) * (device ? 2.0 : 1.0) +
               static_cast<double>(plan.get_container().get_memory_usage());
    case Algo::kFDMTFFT: // channel + DM rows, spectra, tree stage, FFT work
    case Algo::kFDMTFFTFrac:
    case Algo::kDDMTFFT:
    case Algo::kDDMTFFTBrute:
        return input + d +
               (10.0 * static_cast<double>(plan.get_fft_size()) *
                static_cast<double>(kNchans + plan.get_dmt_ndms()));
    case Algo::kDDMT: // output + one block of history per channel
    case Algo::kSDMT:
        return input + d +
               (static_cast<double>(kNchans) * 4.0 *
                static_cast<double>(plan.get_dt_max()));
    }
    return 0.0;
}

/// Machine-readable counters (the name carries the same information).
inline void set_counters(benchmark::State& state,
                         const plans::FDMTPlan& plan,
                         const Point& p,
                         int nthreads) {
    state.counters["nsamps"]    = static_cast<double>(p.nsamps);
    state.counters["ndms"]      = static_cast<double>(plan.get_dmt_ndms());
    state.counters["nchans"]    = static_cast<double>(kNchans);
    state.counters["f_min"]     = kFMin;
    state.counters["f_max"]     = kFMax;
    state.counters["tsamp"]     = kTsamp;
    state.counters["levin_tol"] = kLevinTol;
    state.counters["dm_max"]    = static_cast<double>(p.dm_max);
    state.counters["nbits"]     = static_cast<double>(p.nbits);
    state.counters["nthreads"]  = static_cast<double>(nthreads);
    state.counters["block_s"]   = static_cast<double>(p.nsamps) * kTsamp;
}

/// DDMT throughput in delay-and-sum additions per second (nchans x ndm per
/// output sample), comparable against the machine's load/add roofline. For
/// SDMT it is the equivalent brute-force rate (it performs fewer additions).
inline void set_ddmt_rate(benchmark::State& state,
                          const plans::FDMTPlan& plan,
                          const Point& p) {
    state.counters["adds_per_s"] =
        benchmark::Counter(static_cast<double>(kNchans) *
                               static_cast<double>(plan.get_dmt_ndms()) *
                               static_cast<double>(p.nsamps),
                           benchmark::Counter::kIsIterationInvariantRate);
}

/// Warm-up of a streaming direct engine (DDMT, SDMT, DDMT-FFT): runs
/// `run(n)` -- one block into an output of n samples per trial -- until a
/// call returns the whole block. Blocks shorter than the largest delay need
/// several calls before the history is full.
template <typename Engine, typename Run>
void warm_up_stream(const Engine& engine, SizeType nsamps, Run&& run) {
    for (int call = 0; call < 1024; ++call) {
        const SizeType n = engine.get_output_nsamps(nsamps);
        run(n);
        if (n == nsamps) {
            return;
        }
    }
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

/// Candidate-search benchmarks: boxcars up to 100 ms (floor(0.1 s / tsamp)
/// samples), at two width spacings per preset (every width and ~1.5x
/// spacing at coarse sampling; 1.5x and 2x for UWL's 64 us).
inline SizeType max_boxcar_100ms() {
    return static_cast<SizeType>(std::floor(0.1 / kTsamp));
}

inline std::vector<float> wtsp_sweep() {
    return suite_config().name == "parkes_uwl" ? std::vector<float>{1.5F, 2.0F}
                                               : std::vector<float>{1.0F, 1.5F};
}

} // namespace dmt::bench_suite
