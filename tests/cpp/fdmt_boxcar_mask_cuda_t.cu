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

TEST_CASE("GPU FDMTBoxcar masks device input without a block copy",
          "[gpu][boxcar][kill][mask]") {
    // 1024 channels x 65536 samples (256 MiB per block). The budget leaves
    // less than half a block above the unmasked search, so a staging copy
    // of the block (the 0.8.0 behaviour) could not fit; the transform reads
    // killed rows as zero instead. Constructor and per-call masks.
    const plans::FDMTPlan plan(1200.0F, 1500.0F, 1024, 65536, 0.000256F, 500);
    BoxcarConfig cfg;
    cfg.max_boxcar = 16;
    const SizeType block_bytes =
        plan.get_nchans() * plan.get_nsamps() * sizeof(float);
    const SizeType base =
        FDMTBoxcar(plan, cfg, true, kGpu).get_memory_usage().budgeted();
    const SizeType budget = base + (block_bytes / 2);
    std::vector<float> host(plan.get_nchans() * plan.get_nsamps());
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(-4, 4);
    for (auto& v : host) {
        v = static_cast<float>(dist(rng));
    }
    const thrust::device_vector<float> buf(host.begin(), host.end());
    const DeviceSpan<const float> span(thrust::raw_pointer_cast(buf.data()),
                                       buf.size(), {kGpu.backend, 0});
    const auto ctor_mask = gpu_random_mask(plan.get_nchans(), 5, 0.2);
    FDMTBoxcar search(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse, 0,
                      budget, ctor_mask);
    for (unsigned k = 0; k < 3; ++k) {
        const auto mask = gpu_random_mask(plan.get_nchans(), 11 + k, 0.1);
        REQUIRE_NOTHROW(search.execute(span, std::span<const uint8_t>(mask)));
        (void)search.copy_result();
    }
    const auto usage = search.get_memory_usage();
    CHECK(usage.staging == plan.get_nchans());
    CHECK(usage.budgeted() <= budget);
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
    // Device input with a mask: no staging copy of the block, only one keep
    // byte per channel read by the transform.
    CHECK(after.staging == plan.get_nchans());
    CHECK(after.plan > before.plan);
    CHECK(after.workspace > before.workspace);
    // The noise model lives in host memory: reported, not budgeted.
    CHECK(after.host > before.host);
    CHECK(after.budgeted() + after.host == after.total());
    CHECK(after.budgeted() <= algorithms::kFDMTDefaultMaxMemory);
}

TEST_CASE("GPU FDMTBoxcar per-call masks support kKadane, kNone, and "
          "kKadaneSurrogate",
          "[gpu][parity][boxcar][kill][mask][kadane]") {
    const auto masks = gpu_masks();
    const auto plan  = gpu_mask_plan(2);
    const auto input = waterfalls(plan, 1, masks.size());
    const auto ctor  = gpu_random_mask(32, 99, 0.25);

    for (const auto screen :
         {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone,
          BoxcarScreenMode::kKadaneSurrogate}) {
        for (const bool trap : {false, true}) {
            BoxcarConfig cfg;
            cfg.max_boxcar       = 37;
            cfg.wtsp             = 1.5F;
            cfg.detect_thresh    = 2.9F;
            cfg.peak_mode        = PeakFindingMode::kContiguousPeakAcrossWidths;
            cfg.use_trapezoid    = trap;
            cfg.screen_mode      = screen;
            cfg.max_candidates   = 1U << 20;
            cfg.max_cands_per_dm = 0;

            FDMTBoxcar cpu(plan, cfg, true, Exec::cpu(2), 1, 0, 64,
                           algorithms::kFDMTDefaultMaxMemory, ctor);
            FDMTBoxcar gpu(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                           64, algorithms::kFDMTDefaultMaxMemory, ctor);
            std::vector<RawCandidate> expected;
            std::vector<RawCandidate> actual;
            for (SizeType k = 0; k < input.size(); ++k) {
                append(expected,
                       cpu.execute(std::span<const float>(input[k]),
                                   std::span<const uint8_t>(masks[k])));
                append(actual, gpu.execute(std::span<const float>(input[k]),
                                           std::span<const uint8_t>(masks[k])));
            }
            append(expected, cpu.flush());
            append(actual, gpu.flush());
            if (screen == BoxcarScreenMode::kKadaneSurrogate) {
                compare_near_threshold(actual, expected, cfg.detect_thresh);
            } else {
                compare(actual, expected);
            }
        }
    }
}

TEST_CASE(
    "GPU FDMTBoxcar zero-copy kill masks and budgeted() accounting across "
    "Kadane modes",
    "[gpu][boxcar][kill][mask][kadane]") {
    const plans::FDMTPlan plan(1200.0F, 1500.0F, 1024, 65536, 0.000256F, 500);
    const SizeType block_bytes =
        plan.get_nchans() * plan.get_nsamps() * sizeof(float);
    std::vector<float> host(plan.get_nchans() * plan.get_nsamps());
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(-4, 4);
    for (auto& v : host) {
        v = static_cast<float>(dist(rng));
    }
    const thrust::device_vector<float> buf(host.begin(), host.end());
    const DeviceSpan<const float> span(thrust::raw_pointer_cast(buf.data()),
                                       buf.size(), {kGpu.backend, 0});
    const auto ctor_mask = gpu_random_mask(plan.get_nchans(), 5, 0.2);

    for (const auto screen :
         {BoxcarScreenMode::kKadane, BoxcarScreenMode::kNone,
          BoxcarScreenMode::kKadaneSurrogate}) {
        BoxcarConfig cfg;
        cfg.max_boxcar    = 16;
        cfg.use_trapezoid = true;
        cfg.screen_mode   = screen;
        const auto unmasked_usage =
            FDMTBoxcar(plan, cfg, true, kGpu).get_memory_usage();
        CHECK(unmasked_usage.budgeted() + unmasked_usage.host ==
              unmasked_usage.total());
        CHECK(unmasked_usage.host > 0);
        const SizeType budget = unmasked_usage.budgeted() + (block_bytes / 2);

        FDMTBoxcar search(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                          0, budget, ctor_mask);
        for (unsigned k = 0; k < 2; ++k) {
            const auto mask = gpu_random_mask(plan.get_nchans(), 21 + k, 0.1);
            REQUIRE_NOTHROW(
                search.execute(span, std::span<const uint8_t>(mask)));
            (void)search.copy_result();
        }
        const auto usage = search.get_memory_usage();
        CHECK(usage.staging == plan.get_nchans());
        CHECK(usage.budgeted() + usage.host == usage.total());
        CHECK(usage.budgeted() <= budget);
    }
}
