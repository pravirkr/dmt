// FDMTBoxcar per-call kill mask on the GPU: parity with the direct
// reference (host and device input), calls with different masks queued on
// a stream without host synchronisation, and memory accounting.

#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

#include "fdmt_boxcar_reference.hpp"
#include "skew_subband_search.hpp"

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

// The stream of `blocks` (one beam, nchans rows of `ns` samples each) cut
// into blocks of `short_ns` samples (the tail that fills no block dropped).
std::vector<std::vector<float>>
gpu_reblock(const std::vector<std::vector<float>>& blocks,
            SizeType nchans,
            SizeType ns,
            SizeType short_ns) {
    const SizeType total = blocks.size() * ns;
    std::vector<std::vector<float>> out(total / short_ns,
                                        std::vector<float>(nchans * short_ns));
    for (SizeType t = 0; t < (total / short_ns) * short_ns; ++t) {
        for (SizeType c = 0; c < nchans; ++c) {
            out[t / short_ns][(c * short_ns) + (t % short_ns)] =
                blocks[t / ns][(c * ns) + (t % ns)];
        }
    }
    return out;
}

void copy_device_lags(uint64_t* dst, const uint64_t* src, SizeType n) {
    gpu_utils::check_gpu_call(
        cudaMemcpy(dst, src, n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
}

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
    // A subband search reserves the keep bytes (one per channel) for
    // drain()'s zero blocks from the start; the first mask reuses them.
    REQUIRE(plan.get_subband_depth() > 0);
    CHECK(before.staging == plan.get_nchans());
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

TEST_CASE("GPU FDMTBoxcar declared per-call masks: reserved at construction, "
          "usage fixed",
          "[gpu][boxcar][kill][mask][memory]") {
    const auto masks = gpu_masks();
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        DYNAMIC_SECTION("depth " << depth) {
            const auto plan  = gpu_mask_plan(depth);
            const auto input = waterfalls(plan, 1, masks.size());
            BoxcarConfig cfg;
            cfg.max_boxcar          = 37;
            cfg.wtsp                = 1.5F;
            cfg.detect_thresh       = 2.7F;
            cfg.max_candidates      = 1U << 20;
            cfg.max_cands_per_dm    = 0;
            auto declared           = cfg;
            declared.per_call_masks = true;
            const auto est =
                FDMTBoxcar::estimate_memory(plan, declared, true, kGpu);
            const auto lazy_est =
                FDMTBoxcar::estimate_memory(plan, cfg, true, kGpu);
            // Device table slots and row supports, up front.
            CHECK(est.usage.budgeted() > lazy_est.usage.budgeted());
            CHECK(est.usage.host > lazy_est.usage.host);
            FDMTBoxcar search(plan, declared, true, kGpu);
            FDMTBoxcar lazy(plan, cfg, true, kGpu);
            const auto same = [&](const algorithms::FDMTBoxcarMemoryUsage& u) {
                CHECK(u.plan == est.usage.plan);
                CHECK(u.history == est.usage.history);
                CHECK(u.workspace == est.usage.workspace);
                CHECK(u.candidates == est.usage.candidates);
                CHECK(u.staging == est.usage.staging);
                CHECK(u.host == est.usage.host);
                CHECK(u.total() == est.usage.total());
            };
            same(search.get_memory_usage());
            CHECK(est.search_latency == search.get_search_latency());
            // Device input only, a mask per call.
            thrust::device_vector<float> buf(input[0].size());
            std::vector<RawCandidate> got;
            std::vector<RawCandidate> want;
            for (SizeType k = 0; k < input.size(); ++k) {
                thrust::copy(input[k].begin(), input[k].end(), buf.begin());
                const DeviceSpan<const float> span(
                    thrust::raw_pointer_cast(buf.data()), buf.size(),
                    {kGpu.backend, 0});
                (void)search.execute(span, std::span<const uint8_t>(masks[k]));
                gpu_utils::check_gpu_call(cudaDeviceSynchronize());
                append(got, search.copy_result());
                append(want, lazy.execute(std::span<const float>(input[k]),
                                          std::span<const uint8_t>(masks[k])));
                same(search.get_memory_usage());
            }
            append(got, search.drain());
            append(want, lazy.drain());
            same(search.get_memory_usage());
            REQUIRE_FALSE(want.empty());
            compare(got, want);
        }
    }
}

TEST_CASE("GPU FDMTBoxcar declared per-call masks: a search latency over one "
          "block builds, at its estimate",
          "[gpu][boxcar][kill][mask]") {
    // The windowed top of a 4096-channel subband search at DM 2000 on
    // 4096-sample blocks waits several blocks; the estimate reports it. The
    // older blocks share the catch-all, so declared masks build (they were
    // rejected while the GPU kept two whole blocks only).
    const float tsamp = 455.6e-6F;
    const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 2000.0F, tsamp, tsamp, 856.0F, 1712.0F, 4096, 1.1F);
    const plans::FDMTPlan plan(856.0F, 1712.0F, 4096, 4096, tsamp, grid,
                               "valid", 2);
    BoxcarConfig cfg(24);
    const SizeType budget =
        (plan.get_history_size() + plan.get_tree_history_size()) *
        sizeof(float) * 8 / 10;
    const auto est = FDMTBoxcar::estimate_memory(
        plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse, 0, budget);
    INFO(est.search_path);
    REQUIRE(est.search_path.find("windowed") != std::string::npos);
    REQUIRE(est.search_latency > plan.get_nsamps());
    cfg.per_call_masks  = true;
    const auto declared = FDMTBoxcar::estimate_memory(
        plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse, 0, budget);
    const FDMTBoxcar search(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                            0, budget);
    CHECK(search.get_memory_usage().total() == declared.usage.total());
    CHECK(search.get_search_latency() > plan.get_nsamps());
}

TEST_CASE("GPU FDMTBoxcar per-call masks on blocks shorter than a window",
          "[gpu][parity][boxcar][kill][mask]") {
    // As the CPU test: windows reach over several blocks and use the
    // history mask; search latency 0 (no catch-all).
    const plans::FDMTPlan long_plan(
        700, 1200, 32, 400, 0.001F,
        std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120}, "valid", 0);
    const auto stream = waterfalls(long_plan, 1, 2);
    BoxcarConfig cfg;
    cfg.max_boxcar       = 37;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = 0;
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        for (const SizeType ns : {SizeType{1}, SizeType{7}, SizeType{64}}) {
            DYNAMIC_SECTION("depth " << depth << " block " << ns) {
                const plans::FDMTPlan plan(
                    700, 1200, 32, ns, 0.001F,
                    std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120}, "valid",
                    depth);
                const auto input = gpu_reblock(stream, 32, 400, ns);
                Masks masks;
                std::mt19937 gen(7);
                const auto pool = gpu_masks();
                while (masks.size() < input.size()) {
                    const auto& m = pool[gen() % pool.size()];
                    for (SizeType i = 0;
                         i < 1 + (gen() % 6) && masks.size() < input.size();
                         ++i) {
                        masks.push_back(m);
                    }
                }
                const auto expected =
                    masked_reference(plan, cfg, true, input, {}, masks);
                REQUIRE(expected.candidates.size() > 20);
                FDMTBoxcar search(plan, cfg, true, kGpu);
                REQUIRE(search.get_search_latency() == 0);
                std::vector<RawCandidate> out;
                for (SizeType k = 0; k < input.size(); ++k) {
                    append(out,
                           search.execute(std::span<const float>(input[k]),
                                          std::span<const uint8_t>(masks[k])));
                }
                append(out, search.drain());
                compare(out, expected.candidates);
            }
        }
    }
}

TEST_CASE("GPU FDMTBoxcar windowed top: short blocks use the catch-all",
          "[gpu][parity][skew][boxcar][kill][mask]") {
    // Blocks much shorter than the search latency: the two kept entries
    // are the newest block and the catch-all, which stands for every older
    // block still searched. Exact (to float rounding) against the CPU
    // reference given each row's recorded lag.
    using skew_search_test::require_same_before;
    using skew_search_test::run_subband_search;
    skew_search_test::g_copy_device_lags = copy_device_lags;
    for (const SizeType depth : {SizeType{1}, SizeType{3}}) {
        for (const SizeType ns : {SizeType{128}, SizeType{512}}) {
            DYNAMIC_SECTION("depth " << depth << " block " << ns) {
                const auto grid =
                    plans::DDMTPlan::generate_levin_dm_grid_piecewise(
                        0.0F, 600.0F, 455.6e-6F, 455.6e-6F, 1100.0F, 1714.0F,
                        1152, 1.1F);
                const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, ns,
                                           455.6e-6F, grid, "valid", depth);
                const SizeType nb = 12288 / ns;
                const SizeType nc = plan.get_nchans();
                Masks masks;
                for (SizeType b = 0; b < nb; ++b) {
                    std::vector<uint8_t> m(nc, 1);
                    const auto kill = [&](double lo, double hi) {
                        for (auto c = static_cast<SizeType>(lo * nc);
                             c < static_cast<SizeType>(hi * nc); ++c) {
                            m[c] = 0;
                        }
                    };
                    if (b < nb / 2) {
                        kill(0.0, 0.3);
                        if (b % 5 == 0) {
                            kill(0.5, 0.6);
                        }
                    } else {
                        kill(0.7, 1.0);
                        if (b % 3 == 0) {
                            kill(0.0, 0.1);
                        }
                    }
                    masks.push_back(std::move(m));
                }
                auto box          = skew_search_test::search_config(false);
                box.detect_thresh = 3.5F;
                const auto planes =
                    algorithms::detail::make_boxcar_plans(plan, box, true);
                std::vector<uint64_t> lags;
                const auto gpu = run_subband_search(
                    plan, kGpu, true, false, nb, 128, 0, true, &masks, 1.5F,
                    &planes, SIZE_MAX, &box, nullptr, true, &lags);
                INFO("gpu path: " << gpu.path);
                REQUIRE(gpu.latency > ns);
                const SizeType needed =
                    1 + static_cast<SizeType>((gpu.latency + ns - 1) / ns);
                REQUIRE(needed > 2);
                std::vector<std::vector<float>> input;
                for (SizeType b = 0; b < nb; ++b) {
                    input.push_back(skew_search_test::search_block(
                                        nc * ns, 900 + static_cast<unsigned>(b))
                                        .values);
                }
                const auto expected = masked_reference(
                    plan, box, true, input, {}, masks, lags, 2, needed);
                require_same_before(gpu.candidates, expected.candidates,
                                    UINT64_MAX, 2e-5F);
                // A call failing after its commit leaves the stream as it
                // was (the replaced front and its slots restored).
                const auto failed = run_subband_search(
                    plan, kGpu, true, false, nb, 128, 0, true, &masks, 1.5F,
                    &planes, nb / 2 + 1, &box);
                require_same_before(failed.candidates, gpu.candidates,
                                    UINT64_MAX);
                // ... and so does one right before drain().
                const auto last =
                    run_subband_search(plan, kGpu, true, false, nb, 128, 0,
                                       true, &masks, 1.5F, &planes, nb, &box);
                require_same_before(last.candidates, gpu.candidates,
                                    UINT64_MAX);
            }
        }
    }
    skew_search_test::g_copy_device_lags = nullptr;
}

TEST_CASE("GPU FDMTBoxcar per-call mask: the catch-all through the facade",
          "[gpu][boxcar][kill][mask]") {
    // The windowed top of a 4096-channel DM 2000 subband search on
    // 256-sample blocks waits ~66 blocks: the facade asks for the
    // catch-all. A constant mask gives the same tables everywhere, so the
    // result is the constructor mask's (no cold-prefix skip).
    const float tsamp = 455.6e-6F;
    const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 2000.0F, tsamp, tsamp, 856.0F, 1712.0F, 4096, 1.1F);
    const plans::FDMTPlan plan(856.0F, 1712.0F, 4096, 256, tsamp, grid, "valid",
                               2);
    BoxcarConfig cfg(24);
    cfg.detect_thresh    = 4.5F;
    cfg.skip_cold_prefix = false;
    cfg.per_call_masks   = true;
    const auto min = FDMTBoxcar::estimate_minimum_memory(plan, cfg, true, kGpu);
    INFO(min.search_path);
    REQUIRE(min.search_latency > 2 * plan.get_nsamps());
    FDMTBoxcar call(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse, 0,
                    min.max_memory_bytes);
    REQUIRE(call.get_search_latency() > 2 * plan.get_nsamps());
    const SizeType nb   = 90;
    const SizeType size = 4096 * 256;
    const auto m        = gpu_random_mask(4096, 7, 0.3);
    cfg.per_call_masks  = false;
    FDMTBoxcar ctor(plan, cfg, true, kGpu, 1, algorithms::kFDMTAutoFuse, 0,
                    min.max_memory_bytes, m);
    std::vector<RawCandidate> base;
    std::vector<RawCandidate> out;
    for (SizeType k = 0; k < nb; ++k) {
        const auto block = test_random::normal<float>(size, 300 + k);
        append(base, ctor.execute(std::span<const float>(block)));
        append(out, call.execute(std::span<const float>(block),
                                 std::span<const uint8_t>(m)));
    }
    append(base, ctor.drain());
    append(out, call.drain());
    REQUIRE(base.size() > 50);
    compare(out, base);
}
