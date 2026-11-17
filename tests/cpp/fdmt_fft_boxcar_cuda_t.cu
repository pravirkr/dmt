#include <thrust/device_vector.h>

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;
using algorithms::FDMTFFTBoxcar;

namespace {

struct SegmentCap {
    explicit SegmentCap(SizeType n) {
        algorithms::detail::set_fft_segment_cap(n);
    }
    ~SegmentCap() { algorithms::detail::set_fft_segment_cap(0); }
    SegmentCap(const SegmentCap&)            = delete;
    SegmentCap& operator=(const SegmentCap&) = delete;
    SegmentCap(SegmentCap&&)                 = delete;
    SegmentCap& operator=(SegmentCap&&)      = delete;
};

// Per-beam references from the GPU FDMTFFT itself (cuFFT rounding).
std::vector<RawCandidate>
gpu_reference(const plans::FDMTPlan& plan,
              const BoxcarConfig& cfg,
              bool smear,
              bool frac,
              std::span<const uint8_t> kill,
              SizeType beams,
              const std::vector<std::vector<float>>& input) {
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    std::vector<RawCandidate> expected;
    const SizeType size = plan.get_nchans() * plan.get_nsamps();
    for (SizeType b = 0; b < beams; ++b) {
        auto beam = input;
        for (SizeType k = 0; k < input.size(); ++k) {
            beam[k].assign(input[k].begin() + b * size,
                           input[k].begin() + (b + 1) * size);
        }
        auto c = fft_reference(plan, cfg, smear, frac, kill, exec, beam);
        for (auto& x : c) {
            x.beam_idx = static_cast<uint32_t>(b);
        }
        expected.insert(expected.end(), c.begin(), c.end());
    }
    return expected;
}

} // namespace

TEST_CASE("FDMTFFTBoxcar GPU host and device APIs match direct reference",
          "[gpu][parity][boxcar]") {
    const plans::FDMTPlan plan(700, 1200, 7, 733, 0.001F,
                               std::vector<IndexType>{0, 4, 17, 40, 97});
    const auto input = waterfalls(plan, 2, 4);
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    for (const SizeType cap : {SizeType{0}, SizeType{512}}) {
        const SegmentCap segments(cap);
        for (const bool frac : {true, false}) {
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                CAPTURE(cap, frac, mode);
                BoxcarConfig cfg;
                cfg.max_boxcar       = 37;
                cfg.wtsp             = 1;
                cfg.detect_thresh    = 2.7F;
                cfg.peak_mode        = mode;
                cfg.max_candidates   = 500000;
                cfg.max_cands_per_dm = 0;
                const auto expected =
                    gpu_reference(plan, cfg, true, frac, {}, 2, input);
                FDMTFFTBoxcar search(plan, cfg, true, exec, 2, frac);
                std::vector<RawCandidate> actual;
                for (const auto& block : input) {
                    append(actual, search.execute(block));
                }
                append(actual, search.flush());
                compare_near_threshold(actual, expected, cfg.detect_thresh);

                cudaStream_t stream;
                gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
                search.reset_history(Stream{stream});
                thrust::device_vector<float> device_input(input.front().size());
                auto* ptr = thrust::raw_pointer_cast(device_input.data());
                actual.clear();
                for (const auto& block : input) {
                    gpu_utils::check_gpu_call(cudaMemcpyAsync(
                        ptr, block.data(), block.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
                    (void)search.execute(
                        DeviceSpan<const float>(ptr, block.size(),
                                                {exec.backend, 0}),
                        Stream{stream});
                    append(actual, search.copy_result(Stream{stream}));
                }
                (void)search.flush_device(Stream{stream});
                append(actual, search.copy_result(Stream{stream}));
                compare_near_threshold(actual, expected, cfg.detect_thresh);
                gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
            }
        }
    }
}

TEST_CASE("FDMTFFTBoxcar GPU kill mask and wide windows match the reference",
          "[gpu][parity][boxcar]") {
    const plans::FDMTPlan plan(700, 1200, 8, 1031, 0.001F,
                               std::vector<IndexType>{0, 9, 33, 71});
    const auto input = waterfalls(plan, 1, 3);
    const std::vector<uint8_t> kill{1, 0, 1, 1, 0, 1, 1, 1};
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    BoxcarConfig cfg;
    cfg.max_boxcar       = 1500;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.max_candidates   = 500000;
    cfg.max_cands_per_dm = 0;
    FDMTFFTBoxcar search(plan, cfg, true, exec, 1, true, kill);
    std::vector<RawCandidate> actual;
    for (const auto& block : input) {
        append(actual, search.execute(block));
    }
    append(actual, search.flush());
    compare_near_threshold(actual,
                           gpu_reference(plan, cfg, true, true, kill, 1, input),
                           cfg.detect_thresh);
}

TEST_CASE("FDMTFFTBoxcar GPU memory budget shortens the segments",
          "[gpu][boxcar][budget]") {
    // MeerKAT L-band sized (4096 channels at 612.48 us): half the
    // unconstrained memory forces shorter segments; the candidates agree
    // with the unconstrained run within the fractional kernel's truncation
    // (0.5% of each row's peak; compare_clusters).
    // An impossible budget throws.
    const plans::FDMTPlan plan(856.0F, 1712.0F, 4096, 16384, 612.48e-6F, 800);
    const auto input = waterfalls(plan, 1, 2);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 64;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 6.0F;
    cfg.max_candidates   = 200000;
    cfg.max_cands_per_dm = 0;
    const Exec exec{algorithms::detail::kGPUBackend, 1, 0};
    FDMTFFTBoxcar open(plan, cfg, true, exec);
    const SizeType full   = open.get_memory_usage().budgeted();
    const SizeType budget = full / 2;
    FDMTFFTBoxcar capped(plan, cfg, true, exec, 1, true, {}, budget);
    INFO(open.summary() << capped.summary());
    // On a GPU the budget bounds device memory; host storage is reported.
    CHECK(capped.get_memory_usage().budgeted() <= budget);
    CHECK(capped.get_memory_usage().host > 0);
    CHECK(capped.get_chunk_nsamps() < open.get_chunk_nsamps());
    std::vector<RawCandidate> a;
    std::vector<RawCandidate> b;
    for (const auto& block : input) {
        append(a, open.execute(block));
        append(b, capped.execute(block));
    }
    append(a, open.drain());
    append(b, capped.drain());
    REQUIRE(a.size() > 20);
    compare_clusters(b, a, cfg.detect_thresh, 5e-3, 0.05);
    CHECK_THROWS_AS(
        FDMTFFTBoxcar(plan, cfg, true, exec, 1, true, {}, SizeType{1} << 20),
        std::invalid_argument);
}

namespace {
/// FDMTFFTBoxcar::estimate_memory() equals the built instance on `exec`.
void check_fft_boxcar_estimates(const Exec& exec) {
    const plans::FDMTPlan plan(1000.0F, 1500.0F, 64, 2048, 0.001F, 80);
    BoxcarConfig cfg;
    cfg.max_boxcar        = 32;
    BoxcarConfig keep     = cfg;
    keep.overflow_policy  = OverflowPolicy::kKeepBrightest;
    keep.max_cands_per_dm = 4;
    std::vector<uint8_t> kill(plan.get_nchans(), 1);
    kill[7] = 0;
    struct Case {
        const char* name;
        BoxcarConfig config;
        SizeType beams;
        std::vector<uint8_t> kill;
    };
    const std::vector<Case> cases{{"default", cfg, 1, {}},
                                  {"keep brightest", keep, 2, {}},
                                  {"kill mask", cfg, 2, kill}};
    const auto check = [](const algorithms::FDMTBoxcarMemoryEstimate& e,
                          const FDMTFFTBoxcar& f) {
        const auto got = f.get_memory_usage();
        CHECK(e.usage.plan == got.plan);
        CHECK(e.usage.state == got.state);
        CHECK(e.usage.history == got.history);
        CHECK(e.usage.peaks == got.peaks);
        CHECK(e.usage.workspace == got.workspace);
        CHECK(e.usage.candidates == got.candidates);
        CHECK(e.usage.staging == got.staging);
        CHECK(e.usage.host == got.host);
        CHECK(e.chunk_nsamps == f.get_chunk_nsamps());
    };
    for (const auto& c : cases) {
        INFO(c.name);
        const auto e = FDMTFFTBoxcar::estimate_memory(
            plan, c.config, true, exec, c.beams, true, c.kill);
        const FDMTFFTBoxcar f(plan, c.config, true, exec, c.beams, true,
                              c.kill);
        check(e, f);
        CHECK((e.host_staging == 0) == (exec.backend == Backend::kCPU));
    }
    // A budget the full segment does not fit: shorter segments (a long
    // block, so the transform dominates the boxcar buffers).
    const plans::FDMTPlan long_plan(1000.0F, 1500.0F, 256, 32768, 0.001F, 80);
    BoxcarConfig small   = cfg;
    small.max_candidates = 1024;
    const auto full =
        FDMTFFTBoxcar::estimate_memory(long_plan, small, true, exec, 4);
    const SizeType budget = full.usage.budgeted() - 1;
    const auto e = FDMTFFTBoxcar::estimate_memory(long_plan, small, true, exec,
                                                  4, true, {}, budget);
    check(e, FDMTFFTBoxcar(long_plan, small, true, exec, 4, true, {}, budget));
    CHECK(e.chunk_nsamps < full.chunk_nsamps);
    CHECK(e.usage.budgeted() <= budget);
    CHECK_THROWS_AS(FDMTFFTBoxcar::estimate_memory(plan, cfg, true, exec, 1,
                                                   true, {}, 1024),
                    std::invalid_argument);
    // 2^22 beams and 2^40 candidates (TBs on each side, beyond any host or
    // device): sized, never allocated (an allocation would throw).
    BoxcarConfig huge   = cfg;
    huge.max_candidates = SizeType{1} << 40;
    const plans::FDMTPlan big(1000.0F, 1500.0F, 256, 4096, 0.001F, 400);
    const auto h = FDMTFFTBoxcar::estimate_memory(
        big, huge, true, exec, SizeType{1} << 22, true, {}, SIZE_MAX / 2);
    CHECK(h.usage.candidates >= huge.max_candidates * sizeof(RawCandidate));
    CHECK(h.usage.state + h.usage.workspace + h.usage.history >
          (SizeType{1} << 40));
}
} // namespace

TEST_CASE("FDMTFFTBoxcar estimate_memory equals the constructed instance (gpu)",
          "[boxcar][gpu][budget]") {
    check_fft_boxcar_estimates(Exec{algorithms::detail::kGPUBackend, 1, 0});
}
