// FDMTBoxcar per-call kill mask on the GPU: parity with the direct
// reference (host and device input), calls with different masks queued on
// a stream without host synchronisation, and memory accounting.

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

#include "fdmt_boxcar_reference.hpp"

using namespace boxcar_test;

namespace {

using Masks = std::vector<std::vector<uint8_t>>;

plans::FDMTPlan gpu_mask_plan(SizeType depth) {
    return {700,     1200,   32,
            400,     0.001F, std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120},
            "valid", depth};
}

std::vector<uint8_t> gpu_random_mask(SizeType n, unsigned seed, double killed) {
    std::mt19937 gen(seed);
    std::bernoulli_distribution kill(killed);
    std::vector<uint8_t> m(n);
    for (auto& v : m) {
        v = kill(gen) ? 0 : 1;
    }
    return m;
}

Masks gpu_masks() {
    return {gpu_random_mask(32, 11, 0.2), gpu_random_mask(32, 11, 0.2),
            gpu_random_mask(32, 12, 0.5), {},
            std::vector<uint8_t>(32, 0),  gpu_random_mask(32, 13, 0.1),
            std::vector<uint8_t>(32, 1),  gpu_random_mask(32, 14, 0.6)};
}

const Exec kGpu{algorithms::detail::kGPUBackend, 1, 0};

} // namespace

TEST_CASE("GPU FDMTBoxcar per-call masks match the reference",
          "[gpu][parity][boxcar][kill][mask]") {
    const auto masks = gpu_masks();
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        const auto plan  = gpu_mask_plan(depth);
        const auto input = waterfalls(plan, 1, masks.size());
        for (const bool with_ctor : {false, true}) {
            const auto ctor = with_ctor ? gpu_random_mask(32, 99, 0.25)
                                        : std::vector<uint8_t>{};
            for (const auto mode :
                 {PeakFindingMode::kContiguousPeakPerWidth,
                  PeakFindingMode::kContiguousPeakAcrossWidths,
                  PeakFindingMode::kAllAboveThreshold}) {
                for (const bool trap : {false, true}) {
                    DYNAMIC_SECTION("depth " << depth << " ctor " << with_ctor
                                             << " mode "
                                             << static_cast<int>(mode)
                                             << " trapezoid " << trap) {
                        BoxcarConfig cfg;
                        cfg.max_boxcar       = 37;
                        cfg.wtsp             = 1.5F;
                        cfg.detect_thresh    = 2.7F;
                        cfg.peak_mode        = mode;
                        cfg.max_candidates   = 1U << 20;
                        cfg.max_cands_per_dm = 0;
                        cfg.use_trapezoid    = trap;
                        const auto expected  = masked_reference(
                            plan, cfg, true, input, ctor, masks);
                        REQUIRE(expected.candidates.size() > 50);
                        for (const SizeType chunk :
                             {SizeType{64}, plan.get_nsamps()}) {
                            FDMTBoxcar search(plan, cfg, true, kGpu, 1,
                                              algorithms::kFDMTAutoFuse, chunk,
                                              algorithms::kFDMTDefaultMaxMemory,
                                              ctor);
                            // Host input.
                            std::vector<RawCandidate> host;
                            for (SizeType k = 0; k < input.size(); ++k) {
                                append(host,
                                       search.execute(
                                           std::span<const float>(input[k]),
                                           std::span<const uint8_t>(masks[k])));
                            }
                            append(host, search.flush());
                            compare(host, expected.candidates);
                            // Device input on a nondefault stream.
                            cudaStream_t stream;
                            gpu_utils::check_gpu_call(
                                cudaStreamCreate(&stream));
                            search.reset_history(Stream{stream});
                            thrust::device_vector<float> buf(
                                input.front().size());
                            auto* ptr = thrust::raw_pointer_cast(buf.data());
                            std::vector<RawCandidate> dev;
                            for (SizeType k = 0; k < input.size(); ++k) {
                                gpu_utils::check_gpu_call(cudaMemcpyAsync(
                                    ptr, input[k].data(),
                                    input[k].size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
                                (void)search.execute(
                                    DeviceSpan<const float>(ptr,
                                                            input[k].size(),
                                                            {kGpu.backend, 0}),
                                    std::span<const uint8_t>(masks[k]),
                                    Stream{stream});
                                append(dev, search.copy_result(Stream{stream}));
                            }
                            (void)search.flush_device(Stream{stream});
                            append(dev, search.copy_result(Stream{stream}));
                            compare(dev, expected.candidates);
                            gpu_utils::check_gpu_call(
                                cudaStreamDestroy(stream));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU FDMTBoxcar per-call masks: calls queued without host sync",
          "[gpu][boxcar][kill][mask]") {
    // Every block in its own device buffer, all queued back to back with a
    // different mask each (the masks are host temporaries freed after each
    // call); only the last call's result and the flush are read. They must
    // equal a run that synchronises after every call.
    const auto plan  = gpu_mask_plan(2);
    const auto masks = gpu_masks();
    const auto input = waterfalls(plan, 1, masks.size());
    BoxcarConfig cfg;
    cfg.max_boxcar       = 37;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = 0;
    std::vector<thrust::device_vector<float>> bufs;
    for (const auto& block : input) {
        bufs.emplace_back(block.begin(), block.end());
    }
    const auto run = [&](bool sync) {
        FDMTBoxcar search(plan, cfg, true, kGpu);
        cudaStream_t stream;
        gpu_utils::check_gpu_call(cudaStreamCreate(&stream));
        std::vector<RawCandidate> out;
        for (SizeType k = 0; k < input.size(); ++k) {
            const auto mask = masks[k]; // freed after the call
            (void)search.execute(DeviceSpan<const float>(
                                     thrust::raw_pointer_cast(bufs[k].data()),
                                     bufs[k].size(), {kGpu.backend, 0}),
                                 std::span<const uint8_t>(mask),
                                 Stream{stream});
            if (sync || k + 1 == input.size()) {
                const auto r = search.copy_result(Stream{stream});
                if (k + 1 == input.size()) {
                    append(out, r);
                }
            }
        }
        (void)search.flush_device(Stream{stream});
        append(out, search.copy_result(Stream{stream}));
        gpu_utils::check_gpu_call(cudaStreamDestroy(stream));
        return out;
    };
    const auto synced = run(true);
    REQUIRE_FALSE(synced.empty());
    compare(run(false), synced);
}

TEST_CASE("GPU FDMTBoxcar per-call masks: memory is accounted",
          "[gpu][boxcar][kill][mask]") {
    const auto plan  = gpu_mask_plan(1);
    const auto input = waterfalls(plan, 1, 2);
    BoxcarConfig cfg;
    cfg.max_boxcar = 37;
    FDMTBoxcar search(plan, cfg, true, kGpu);
    // Device input: no host staging before the first mask.
    thrust::device_vector<float> buf(input[0].begin(), input[0].end());
    const DeviceSpan<const float> span(thrust::raw_pointer_cast(buf.data()),
                                       buf.size(), {kGpu.backend, 0});
    (void)search.execute(span);
    (void)search.copy_result();
    const auto before = search.get_memory_usage();
    CHECK(before.staging == 0);
    const auto mask = gpu_random_mask(32, 3, 0.3);
    (void)search.execute(span, std::span<const uint8_t>(mask));
    (void)search.copy_result();
    const auto after = search.get_memory_usage();
    CHECK(after.staging == input[0].size() * sizeof(float));
    CHECK(after.plan > before.plan);
    CHECK(after.workspace > before.workspace);
    CHECK(after.total() <= algorithms::kFDMTDefaultMaxMemory);
}
