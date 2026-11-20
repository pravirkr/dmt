// FDMTBoxcar per-call kill mask on the GPU: parity with the direct
// reference (host and device input), calls with different masks queued on
// a stream without host synchronisation, and memory accounting.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_int_tree.hpp"
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

            FDMTBoxcar cpu(plan, cfg, true, Exec::cpu(), 1, 0, 64,
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

// --- Packed input under per-call masks: the GPU equals the CPU ---------

namespace {

// Codes round(mean + scale x) clamped to the depth, packed per row.
std::vector<std::vector<uint8_t>>
gpu_pack(const std::vector<std::vector<float>>& input,
         SizeType rows,
         SizeType ns,
         unsigned nbits,
         uint32_t mean,
         float scale) {
    const uint32_t top = nbits == 16 ? 65535U : (1U << nbits) - 1U;
    const SizeType rb  = bit_pack_utils::packed_row_bytes(ns, nbits);
    std::vector<std::vector<uint8_t>> out;
    for (const auto& block : input) {
        std::vector<uint8_t> p(rows * rb, 0);
        for (SizeType r = 0; r < rows; ++r) {
            for (SizeType t = 0; t < ns; ++t) {
                const float v   = std::round(static_cast<float>(mean) +
                                             (scale * block[(r * ns) + t]));
                const auto code = static_cast<uint32_t>(
                    std::clamp(v, 0.0F, static_cast<float>(top)));
                uint8_t* row = p.data() + (r * rb);
                switch (nbits) {
                case 1:
                    bit_pack_utils::write_packed_sample<1>(row, t, code);
                    break;
                case 2:
                    bit_pack_utils::write_packed_sample<2>(row, t, code);
                    break;
                case 4:
                    bit_pack_utils::write_packed_sample<4>(row, t, code);
                    break;
                case 8:
                    bit_pack_utils::write_packed_sample<8>(row, t, code);
                    break;
                default:
                    bit_pack_utils::write_packed_sample<16>(row, t, code);
                    break;
                }
            }
        }
        out.push_back(std::move(p));
    }
    return out;
}

// Exact equality (every field, S/N bit for bit).
void same_exact(std::vector<RawCandidate> a, std::vector<RawCandidate> b) {
    sort(a);
    sort(b);
    REQUIRE(a.size() == b.size());
    SizeType bad = 0;
    for (SizeType i = 0; i < a.size() && bad < 10; ++i) {
        if (key(a[i]) == key(b[i]) &&
            std::bit_cast<uint32_t>(a[i].snr) ==
                std::bit_cast<uint32_t>(b[i].snr) &&
            a[i].width == b[i].width) {
            continue;
        }
        ++bad;
        INFO("candidate " << i << " peak " << a[i].peak_sample << " / "
                          << b[i].peak_sample << " snr " << a[i].snr << " / "
                          << b[i].snr);
        CHECK(key(a[i]) == key(b[i]));
        CHECK(std::bit_cast<uint32_t>(a[i].snr) ==
              std::bit_cast<uint32_t>(b[i].snr));
    }
    REQUIRE(bad == 0);
}

BoxcarConfig
packed_cfg(PeakFindingMode mode, unsigned nbits = 8, SizeType width = 37) {
    BoxcarConfig cfg;
    cfg.max_boxcar       = width;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.7F;
    cfg.peak_mode        = mode;
    cfg.max_candidates   = 1U << 20;
    cfg.max_cands_per_dm = 0;
    cfg.input_nbits      = static_cast<int>(nbits);
    cfg.input_mean       = nbits == 16 ? 30000.0F : 128.0F;
    cfg.noise_sigma      = nbits == 16 ? 3000.0F : 10.0F;
    cfg.screen_mode      = BoxcarScreenMode::kPrefix;
    return cfg;
}

// Kept candidates of every call; `flags` records each call's overflow (only
// kKeepBrightest may overflow).
std::vector<RawCandidate>
host_run(FDMTBoxcar& search,
         const std::vector<std::vector<uint8_t>>& input,
         const Masks& masks,
         bool drain               = false,
         std::vector<bool>* flags = nullptr) {
    std::vector<RawCandidate> out;
    const auto add = [&](CandidateResultView view) {
        if (flags == nullptr) {
            append(out, view);
            return;
        }
        flags->push_back(view.overflowed());
        out.insert(out.end(), view.candidates.begin(), view.candidates.end());
    };
    for (SizeType k = 0; k < input.size(); ++k) {
        if (masks[k].empty() && k == 0) {
            add(search.execute(std::span<const uint8_t>(input[k])));
        } else {
            add(search.execute(std::span<const uint8_t>(input[k]),
                               std::span<const uint8_t>(masks[k])));
        }
    }
    add(drain ? search.drain() : search.flush());
    return out;
}

// The first block without a mask: per-call masks then start mid-stream.
Masks packed_masks() {
    auto m = gpu_masks();
    m[0]   = {};
    return m;
}

FDMTBoxcar make_packed(const plans::FDMTPlan& plan,
                       const BoxcarConfig& cfg,
                       Exec exec,
                       std::span<const uint8_t> ctor = {},
                       SizeType chunk                = 0) {
    return FDMTBoxcar(plan, cfg, true, exec, 1, algorithms::kFDMTAutoFuse,
                      chunk, algorithms::kFDMTDefaultMaxMemory, ctor);
}

} // namespace

TEST_CASE("GPU FDMTBoxcar packed per-call masks equal the CPU",
          "[gpu][parity][boxcar][kill][mask][packed]") {
    const auto masks = packed_masks();
    for (const SizeType depth : {SizeType{0}, SizeType{2}}) {
        const auto plan   = gpu_mask_plan(depth);
        const auto stream = waterfalls(plan, 1, masks.size());
        const auto input  = gpu_pack(stream, 32, 400, 8, 128, 10.0F);
        for (const bool with_ctor : {false, true}) {
            const auto ctor = with_ctor ? gpu_random_mask(32, 99, 0.25)
                                        : std::vector<uint8_t>{};
            for (const auto screen :
                 {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kNone,
                  BoxcarScreenMode::kKadane}) {
                for (const auto mode :
                     {PeakFindingMode::kContiguousPeakPerWidth,
                      PeakFindingMode::kContiguousPeakAcrossWidths,
                      PeakFindingMode::kAllAboveThreshold}) {
                    for (const bool trap : {false, true}) {
                        DYNAMIC_SECTION(
                            "depth " << depth << " ctor " << with_ctor
                                     << " screen " << static_cast<int>(screen)
                                     << " mode " << static_cast<int>(mode)
                                     << " trapezoid " << trap) {
                            auto cfg          = packed_cfg(mode);
                            cfg.use_trapezoid = trap;
                            cfg.screen_mode   = screen;
                            auto cpu =
                                make_packed(plan, cfg, Exec::cpu(), ctor);
                            const auto want = host_run(cpu, input, masks, true);
                            REQUIRE(want.size() > 50);
                            for (const SizeType chunk :
                                 {SizeType{128}, plan.get_nsamps()}) {
                                auto gpu =
                                    make_packed(plan, cfg, kGpu, ctor, chunk);
                                same_exact(host_run(gpu, input, masks, true),
                                           want);
                                // Device input on a nondefault stream.
                                cudaStream_t st;
                                gpu_utils::check_gpu_call(
                                    cudaStreamCreate(&st));
                                gpu.reset_history(Stream{st});
                                thrust::device_vector<uint8_t> buf(
                                    input.front().size());
                                auto* ptr =
                                    thrust::raw_pointer_cast(buf.data());
                                std::vector<RawCandidate> dev;
                                for (SizeType k = 0; k < input.size(); ++k) {
                                    gpu_utils::check_gpu_call(cudaMemcpyAsync(
                                        ptr, input[k].data(), input[k].size(),
                                        cudaMemcpyHostToDevice, st));
                                    const DeviceSpan<const uint8_t> span(
                                        ptr, input[k].size(),
                                        {kGpu.backend, 0});
                                    if (k == 0) {
                                        (void)gpu.execute(span, Stream{st});
                                    } else {
                                        (void)gpu.execute(
                                            span,
                                            std::span<const uint8_t>(masks[k]),
                                            Stream{st});
                                    }
                                    append(dev, gpu.copy_result(Stream{st}));
                                }
                                gpu_utils::check_gpu_call(
                                    cudaStreamSynchronize(st));
                                append(dev, gpu.drain());
                                same_exact(dev, want);
                                gpu_utils::check_gpu_call(
                                    cudaStreamDestroy(st));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("GPU FDMTBoxcar packed per-call masks: short blocks at 1-8 bits, "
          "keep-brightest, wide boxes and 16-bit",
          "[gpu][parity][boxcar][kill][mask][packed]") {
    const auto ctor = gpu_random_mask(32, 99, 0.25);
    SECTION("blocks shorter than a window, 1-8 bits") {
        const auto long_plan = gpu_mask_plan(0);
        const auto stream    = waterfalls(long_plan, 1, 2);
        struct Depth {
            unsigned nbits;
            uint32_t mean;
            float scale;
        };
        // The mean code of each depth (1-bit: 0), as the CPU test.
        for (const Depth d : {Depth{1, 0, 1.0F}, Depth{2, 1, 1.0F},
                              Depth{4, 8, 2.0F}, Depth{8, 128, 10.0F}}) {
            for (const SizeType ns : {SizeType{7}, SizeType{64}}) {
                DYNAMIC_SECTION("nbits " << d.nbits << " block " << ns) {
                    const plans::FDMTPlan plan(
                        700, 1200, 32, ns, 0.001F,
                        std::vector<IndexType>{0, 3, 9, 20, 41, 77, 120},
                        "valid", 0);
                    const auto blocks = gpu_reblock(stream, 32, 400, ns);
                    const auto input =
                        gpu_pack(blocks, 32, ns, d.nbits, d.mean, d.scale);
                    Masks masks;
                    const auto pool = gpu_masks();
                    for (SizeType k = 0; k < input.size(); ++k) {
                        masks.push_back(pool[(k / 5) % pool.size()]);
                    }
                    auto cfg = packed_cfg(
                        PeakFindingMode::kContiguousPeakPerWidth, d.nbits);
                    cfg.input_mean  = static_cast<float>(d.mean);
                    cfg.noise_sigma = d.scale;
                    auto cpu        = make_packed(plan, cfg, Exec::cpu(), ctor);
                    auto gpu        = make_packed(plan, cfg, kGpu, ctor);
                    REQUIRE(gpu.get_search_latency() == 0);
                    const auto want = host_run(cpu, input, masks);
                    REQUIRE(want.size() > 20);
                    same_exact(host_run(gpu, input, masks), want);
                }
            }
        }
    }
    SECTION("kKeepBrightest, max_boxcar over 4096, 16-bit (uint64 rows)") {
        const auto plan   = gpu_mask_plan(2);
        const auto masks  = packed_masks();
        const auto stream = waterfalls(plan, 1, masks.size());
        struct Case {
            const char* name;
            unsigned nbits;
            SizeType width;
            bool keep;
            BoxcarScreenMode screen;
        };
        // Kadane over 4096 takes the prefix kernel (the fallback).
        constexpr auto kP = BoxcarScreenMode::kPrefix;
        constexpr auto kK = BoxcarScreenMode::kKadane;
        for (const Case c :
             {Case{"keep", 8, 37, true, kP}, Case{"wide", 8, 4200, false, kP},
              Case{"16-bit", 16, 37, false, kP},
              Case{"keep kadane", 8, 37, true, kK},
              Case{"wide kadane", 8, 4200, false, kK},
              Case{"16-bit kadane", 16, 37, false, kK}}) {
            DYNAMIC_SECTION(c.name) {
                const auto input = gpu_pack(stream, 32, 400, c.nbits,
                                            c.nbits == 16 ? 30000 : 128,
                                            c.nbits == 16 ? 3000.0F : 10.0F);
                auto cfg = packed_cfg(PeakFindingMode::kContiguousPeakPerWidth,
                                      c.nbits, c.width);
                cfg.screen_mode = c.screen;
                // This plan's rows carry in uint32 (uint64: the next section).
                REQUIRE(algorithms::detail::int_search_fits_u32(plan, true,
                                                                c.nbits));
                if (c.keep) {
                    cfg.overflow_policy  = OverflowPolicy::kKeepBrightest;
                    cfg.max_cands_per_dm = 3;
                }
                auto cpu = make_packed(plan, cfg, Exec::cpu(), ctor);
                auto gpu = make_packed(plan, cfg, kGpu, ctor);
                std::vector<bool> cpu_flags;
                std::vector<bool> gpu_flags;
                const auto want =
                    host_run(cpu, input, masks, false, &cpu_flags);
                REQUIRE(want.size() > 20);
                same_exact(host_run(gpu, input, masks, false, &gpu_flags),
                           want);
                CHECK(gpu_flags == cpu_flags);
                if (c.keep) {
                    CHECK(std::ranges::count(cpu_flags, true) > 0);
                } else {
                    CHECK(std::ranges::count(cpu_flags, true) == 0);
                }
            }
        }
    }
    SECTION("uint64 carry (16-bit, smeared, delays to 8200)") {
        const plans::FDMTPlan plan(700, 1200, 32, 2048, 0.001F,
                                   std::vector<IndexType>{0, 4100, 8200});
        REQUIRE_FALSE(algorithms::detail::int_search_fits_u32(plan, true, 16));
        const auto masks  = packed_masks();
        const auto stream = waterfalls(plan, 1, masks.size());
        const auto input  = gpu_pack(stream, 32, 2048, 16, 30000, 3000.0F);
        const auto cfg =
            packed_cfg(PeakFindingMode::kContiguousPeakPerWidth, 16);
        auto cpu        = make_packed(plan, cfg, Exec::cpu(), ctor);
        auto gpu        = make_packed(plan, cfg, kGpu, ctor);
        const auto want = host_run(cpu, input, masks, true);
        REQUIRE(want.size() > 20);
        same_exact(host_run(gpu, input, masks, true), want);
    }
}

TEST_CASE("GPU FDMTBoxcar packed per-call masks: the screens skip tiles",
          "[gpu][parity][boxcar][kill][mask][packed]") {
    // Quiet noise and a high threshold: runs close, so the prefix screen
    // (per-call tables' group bounds) and the Kadane gate (per-call cuts)
    // skip tiles; the dense tests above keep runs open, so every tile takes
    // the bookkeeping there. 8-bit rows on 256 channels carry in uint32
    // (the int32 Kadane gate); 16-bit rows with delays to 8200 carry in
    // uint64 (the float gate on centred sums). Heavy kills move the cuts.
    struct Case {
        const char* name;
        SizeType nchans;
        std::vector<IndexType> dms;
        unsigned nbits;
        float mean;
        float sigma;
        SizeType ns;
        float f_lo;
        float f_hi;
        float tsamp;
    };
    const std::vector<Case> cases = {
        {"8-bit",
         256,
         {0, 2, 5, 9, 14, 20, 30, 45, 60},
         8,
         128.0F,
         10.0F,
         4096,
         1200.0F,
         1500.0F,
         64e-6F},
        {"16-bit uint64",
         32,
         {0, 4100, 8200},
         16,
         30000.0F,
         3000.0F,
         4096,
         700.0F,
         1200.0F,
         0.001F},
    };
    for (const auto& c : cases) {
        const plans::FDMTPlan plan(c.f_lo, c.f_hi, c.nchans, c.ns, c.tsamp,
                                   c.dms, "valid", 0);
        REQUIRE(algorithms::detail::int_search_fits_u32(plan, true, c.nbits) ==
                (c.nbits == 8));
        const SizeType nc = plan.get_nchans();
        const SizeType ns = plan.get_nsamps();
        const SizeType nb = 6;
        const SizeType rb = bit_pack_utils::packed_row_bytes(ns, c.nbits);
        const float top   = c.nbits == 16 ? 65535.0F : 255.0F;
        std::vector<std::vector<uint8_t>> input;
        Masks masks;
        for (SizeType b = 0; b < nb; ++b) {
            const auto x = test_random::normal<float>(nc * ns, 40 + b);
            std::vector<uint8_t> p(nc * rb);
            for (SizeType ch = 0; ch < nc; ++ch) {
                for (SizeType t = 0; t < ns; ++t) {
                    // Bright boxes, some crossing block boundaries.
                    const SizeType at = (b * ns) + t;
                    const bool pulse  = (at % 1500) < (3 + (at / 1500) % 20);
                    const float v     = c.mean + (c.sigma * x[(ch * ns) + t]) +
                                        (pulse ? 0.4F * c.sigma : 0.0F);
                    const auto code   = static_cast<uint32_t>(
                        std::clamp(std::round(v), 0.0F, top));
                    if (c.nbits == 16) {
                        bit_pack_utils::write_packed_sample<16>(
                            p.data() + (ch * rb), t, code);
                    } else {
                        bit_pack_utils::write_packed_sample<8>(
                            p.data() + (ch * rb), t, code);
                    }
                }
            }
            input.push_back(std::move(p));
            masks.push_back(
                gpu_random_mask(nc, 60 + static_cast<unsigned>(b),
                                0.1 + (0.12 * static_cast<double>(b))));
        }
        masks[0] = {};
        for (const auto screen :
             {BoxcarScreenMode::kPrefix, BoxcarScreenMode::kKadane}) {
            for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                    PeakFindingMode::kAllAboveThreshold}) {
                DYNAMIC_SECTION(c.name << " screen " << static_cast<int>(screen)
                                       << " mode " << static_cast<int>(mode)) {
                    // Dense widths: the prefix screen groups them (sparse
                    // widths leave each to the per-width check).
                    BoxcarConfig cfg(256);
                    cfg.wtsp             = 1.05F;
                    cfg.detect_thresh    = 6.0F;
                    cfg.noise_sigma      = c.sigma;
                    cfg.input_nbits      = static_cast<int>(c.nbits);
                    cfg.input_mean       = c.mean;
                    cfg.peak_mode        = mode;
                    cfg.screen_mode      = screen;
                    cfg.max_candidates   = 1U << 20;
                    cfg.max_cands_per_dm = 0;
                    auto cpu             = make_packed(plan, cfg, Exec::cpu());
                    auto gpu             = make_packed(plan, cfg, kGpu);
                    const auto& widths = gpu.get_search_plans().front().widths;
                    REQUIRE_FALSE(
                        algorithms::detail::boxcar_screen_groups(widths, false)
                            .empty());
                    const auto want = host_run(cpu, input, masks, true);
                    REQUIRE(want.size() > 20);
                    same_exact(host_run(gpu, input, masks, true), want);
                }
            }
        }
    }
}

TEST_CASE("GPU FDMTBoxcar packed per-call masks: the catch-all",
          "[gpu][parity][boxcar][kill][mask][packed]") {
    // The windowed top waits over a block: entries older than the newest
    // share the catch-all (the CPU keeps more entries, so it is no oracle
    // here). Killed channels read the mean code, so without box smearing
    // the packed search equals the float one on codes minus the mean, whose
    // catch-all is checked against the reference above.
    const float tsamp = 455.6e-6F;
    const auto grid   = plans::DDMTPlan::generate_levin_dm_grid_piecewise(
        0.0F, 600.0F, tsamp, tsamp, 1100.0F, 1714.0F, 1152, 1.1F);
    const plans::FDMTPlan plan(1100.0F, 1714.0F, 1152, 128, tsamp, grid,
                               "valid", 3);
    const SizeType nc = plan.get_nchans();
    const SizeType ns = plan.get_nsamps();
    const SizeType nb = 48;
    Masks masks;
    for (SizeType b = 0; b < nb; ++b) {
        std::vector<uint8_t> m(nc, 1);
        const auto kill = [&](double lo, double hi) {
            for (auto c = static_cast<SizeType>(lo * nc);
                 c < static_cast<SizeType>(hi * nc); ++c) {
                m[c] = 0;
            }
        };
        kill(b < nb / 2 ? 0.0 : 0.7, b < nb / 2 ? 0.3 : 1.0);
        if (b % 5 == 0) {
            kill(0.5, 0.6);
        }
        masks.push_back(std::move(m));
    }
    masks[0] = {};
    std::vector<std::vector<uint8_t>> packed;
    std::vector<std::vector<float>> centred;
    for (SizeType b = 0; b < nb; ++b) {
        const auto x = test_random::normal<float>(nc * ns, 900 + b);
        std::vector<uint8_t> p(nc * ns);
        std::vector<float> c(nc * ns);
        for (SizeType i = 0; i < x.size(); ++i) {
            const float v =
                std::clamp(std::round(128.0F + (10.0F * x[i])), 0.0F, 255.0F);
            p[i] = static_cast<uint8_t>(v);
            c[i] = v - 128.0F;
        }
        packed.push_back(std::move(p));
        centred.push_back(std::move(c));
    }
    for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                            PeakFindingMode::kAllAboveThreshold}) {
        DYNAMIC_SECTION("mode " << static_cast<int>(mode)) {
            BoxcarConfig fl(24);
            fl.detect_thresh    = 4.75F;
            fl.noise_sigma      = 10.0F;
            fl.peak_mode        = mode;
            fl.skip_cold_prefix = false;
            fl.per_call_masks   = true;
            fl.max_candidates   = 1U << 20;
            auto pk             = fl;
            pk.input_nbits      = 8;
            pk.input_mean       = 128.0F;
            auto kd             = pk;
            kd.screen_mode      = BoxcarScreenMode::kKadane;
            auto pf             = pk;
            pf.screen_mode      = BoxcarScreenMode::kPrefix;
            // The surrogate's catch-all and lagged rows (tiles across a
            // table change): packed against float on centred codes, to near
            // the threshold (its S/N is approximate either way).
            auto sp        = pk;
            sp.screen_mode = BoxcarScreenMode::kKadaneSurrogate;
            auto sf        = fl;
            sf.screen_mode = BoxcarScreenMode::kKadaneSurrogate;
            // The skewed path (search latency) on every engine: an explicit
            // (shortest) chunk keeps the path the budget gives (an automatic
            // one takes the faster direct path where it fits).
            const SizeType chunk = 256;
            const auto minimum   = [&](const BoxcarConfig& c, bool smear) {
                return FDMTBoxcar::estimate_minimum_memory(
                    plan, c, smear, kGpu, 1, algorithms::kFDMTAutoFuse, chunk);
            };
            const auto min  = minimum(pk, false);
            SizeType budget = 0;
            for (const auto& [c, smear] :
                 {std::pair{pk, false}, std::pair{fl, false},
                  std::pair{kd, false}, std::pair{kd, true},
                  std::pair{sp, false}, std::pair{sf, false},
                  std::pair{pf, true}}) {
                budget = std::max(budget, minimum(c, smear).max_memory_bytes);
            }
            FDMTBoxcar a(plan, pk, false, kGpu, 1, algorithms::kFDMTAutoFuse,
                         chunk, budget);
            FDMTBoxcar b(plan, fl, false, kGpu, 1, algorithms::kFDMTAutoFuse,
                         chunk, budget);
            FDMTBoxcar sa(plan, sp, false, kGpu, 1, algorithms::kFDMTAutoFuse,
                          chunk, budget);
            FDMTBoxcar sb(plan, sf, false, kGpu, 1, algorithms::kFDMTAutoFuse,
                          chunk, budget);
            // The Kadane gate's catch-all cuts: exact against the prefix
            // screen (both exact searches), with box smearing too.
            FDMTBoxcar ka(plan, kd, false, kGpu, 1, algorithms::kFDMTAutoFuse,
                          chunk, budget);
            FDMTBoxcar ks(plan, kd, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                          chunk, budget);
            FDMTBoxcar ps(plan, pf, true, kGpu, 1, algorithms::kFDMTAutoFuse,
                          chunk, budget);
            REQUIRE(sa.get_search_latency() == sb.get_search_latency());
            REQUIRE(sa.get_search_latency() > ns);
            INFO(min.search_path);
            REQUIRE(a.get_search_latency() > ns);
            REQUIRE(b.get_search_latency() == a.get_search_latency());
            REQUIRE(ka.get_search_latency() == a.get_search_latency());
            REQUIRE(ks.get_search_latency() == ps.get_search_latency());
            REQUIRE(ks.get_search_latency() > ns);
            std::vector<RawCandidate> got;
            std::vector<RawCandidate> want;
            std::vector<RawCandidate> kad;
            std::vector<RawCandidate> ksm;
            std::vector<RawCandidate> psm;
            std::vector<RawCandidate> spk;
            std::vector<RawCandidate> sfl;
            for (SizeType k = 0; k < nb; ++k) {
                const std::span<const uint8_t> in(packed[k]);
                if (k == 0) {
                    append(got, a.execute(in));
                    append(want, b.execute(std::span<const float>(centred[k])));
                    append(kad, ka.execute(in));
                    append(ksm, ks.execute(in));
                    append(psm, ps.execute(in));
                    append(spk, sa.execute(in));
                    append(sfl, sb.execute(std::span<const float>(centred[k])));
                    continue;
                }
                const std::span<const uint8_t> m(masks[k]);
                append(got, a.execute(in, m));
                append(want, b.execute(std::span<const float>(centred[k]), m));
                append(kad, ka.execute(in, m));
                append(ksm, ks.execute(in, m));
                append(psm, ps.execute(in, m));
                append(spk, sa.execute(in, m));
                append(sfl, sb.execute(std::span<const float>(centred[k]), m));
            }
            append(got, a.drain());
            append(want, b.drain());
            append(kad, ka.drain());
            append(ksm, ks.drain());
            append(psm, ps.drain());
            append(spk, sa.drain());
            append(sfl, sb.drain());
            REQUIRE(want.size() > 50);
            compare(got, want);
            same_exact(kad, got);
            REQUIRE(psm.size() > 50);
            same_exact(ksm, psm);
            REQUIRE(sfl.size() > 50);
            compare_near_threshold(spk, sfl, sp.detect_thresh);
        }
    }
}

TEST_CASE("GPU FDMTBoxcar packed per-call masks: queued calls, memory and "
          "limits",
          "[gpu][boxcar][kill][mask][packed]") {
    const auto plan   = gpu_mask_plan(2);
    const auto masks  = packed_masks();
    const auto stream = waterfalls(plan, 1, masks.size());
    const auto input  = gpu_pack(stream, 32, 400, 8, 128, 10.0F);
    const auto cfg    = packed_cfg(PeakFindingMode::kContiguousPeakPerWidth);

    SECTION("calls queued without host sync") {
        // Every block in its own device buffer, queued back to back; the
        // first per-call mask arrives right after an unmasked queued call.
        auto cpu        = make_packed(plan, cfg, Exec::cpu());
        const auto want = host_run(cpu, input, masks);
        auto gpu        = make_packed(plan, cfg, kGpu);
        cudaStream_t st;
        gpu_utils::check_gpu_call(cudaStreamCreate(&st));
        std::vector<thrust::device_vector<uint8_t>> bufs;
        for (const auto& b : input) {
            bufs.emplace_back(b.begin(), b.end());
        }
        gpu_utils::check_gpu_call(cudaDeviceSynchronize());
        std::vector<RawCandidate> got;
        for (SizeType k = 0; k < input.size(); ++k) {
            const DeviceSpan<const uint8_t> span(
                thrust::raw_pointer_cast(bufs[k].data()), bufs[k].size(),
                {kGpu.backend, 0});
            const auto view =
                k == 0 ? gpu.execute(span, Stream{st})
                       : gpu.execute(span, std::span<const uint8_t>(masks[k]),
                                     Stream{st});
            (void)view;
            append(got, gpu.copy_result(Stream{st}));
        }
        append(got, gpu.flush());
        same_exact(got, want);
        gpu_utils::check_gpu_call(cudaStreamDestroy(st));
    }

    SECTION("memory is accounted; declared masks keep the estimate") {
        auto gpu          = make_packed(plan, cfg, kGpu);
        const auto before = gpu.get_memory_usage();
        (void)gpu.execute(std::span<const uint8_t>(input[1]),
                          std::span<const uint8_t>(masks[1]));
        const auto after = gpu.get_memory_usage();
        CHECK(after.plan > before.plan);
        CHECK(after.host > before.host);
        CHECK(after.budgeted() <= algorithms::kFDMTDefaultMaxMemory);
        for (const auto [on_gpu, screen] :
             {std::pair{true, BoxcarScreenMode::kPrefix},
              std::pair{false, BoxcarScreenMode::kPrefix},
              std::pair{true, BoxcarScreenMode::kKadane},
              std::pair{false, BoxcarScreenMode::kKadane},
              std::pair{true, BoxcarScreenMode::kKadaneSurrogate},
              std::pair{false, BoxcarScreenMode::kKadaneSurrogate}}) {
            INFO("gpu " << on_gpu << " screen " << static_cast<int>(screen));
            const Exec ex           = on_gpu ? kGpu : Exec::cpu();
            auto base               = cfg;
            base.screen_mode        = screen;
            auto declared           = base;
            declared.per_call_masks = true;
            const auto est =
                FDMTBoxcar::estimate_memory(plan, declared, true, ex);
            auto search = make_packed(plan, declared, ex);
            CHECK(search.get_memory_usage().total() == est.usage.total());
            CHECK(search.get_memory_usage().host == est.usage.host);
            auto lazy = make_packed(plan, base, ex);
            // Device input on the GPU (host input stages the block).
            thrust::device_vector<uint8_t> buf(input[0].size());
            std::vector<RawCandidate> got;
            std::vector<RawCandidate> want;
            for (SizeType k = 1; k < input.size(); ++k) {
                const std::span<const uint8_t> mask(masks[k]);
                if (on_gpu) {
                    thrust::copy(input[k].begin(), input[k].end(), buf.begin());
                    const DeviceSpan<const uint8_t> span(
                        thrust::raw_pointer_cast(buf.data()), buf.size(),
                        {kGpu.backend, 0});
                    (void)search.execute(span, mask);
                    gpu_utils::check_gpu_call(cudaDeviceSynchronize());
                    append(got, search.copy_result());
                } else {
                    append(got, search.execute(
                                    std::span<const uint8_t>(input[k]), mask));
                }
                append(want,
                       lazy.execute(std::span<const uint8_t>(input[k]), mask));
                CHECK(search.get_memory_usage().total() == est.usage.total());
            }
            append(got, search.drain());
            append(want, lazy.drain());
            CHECK(search.get_memory_usage().total() == est.usage.total());
            REQUIRE(want.size() > 20);
            same_exact(got, want);
        }
    }
}

TEST_CASE("GPU FDMTBoxcar packed per-call masks: kKadaneSurrogate",
          "[gpu][parity][boxcar][kill][mask][packed][surrogate]") {
    const auto plan   = gpu_mask_plan(2);
    const auto masks  = packed_masks();
    const auto stream = waterfalls(plan, 1, masks.size());
    const auto input  = gpu_pack(stream, 32, 400, 8, 128, 10.0F);
    const auto ctor   = gpu_random_mask(32, 99, 0.25);

    SECTION("the CPU's candidates, to near the threshold") {
        // The GPU surrogate emits its own (approximate) S/N, as on float
        // input; the first block takes no mask, so excursions are open when
        // the per-call masks start.
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            DYNAMIC_SECTION("mode " << static_cast<int>(mode)) {
                auto cfg        = packed_cfg(mode);
                cfg.screen_mode = BoxcarScreenMode::kKadaneSurrogate;
                auto cpu        = make_packed(plan, cfg, Exec::cpu(), ctor);
                auto gpu        = make_packed(plan, cfg, kGpu, ctor);
                const auto want = host_run(cpu, input, masks, true);
                REQUIRE(want.size() > 20);
                compare_near_threshold(host_run(gpu, input, masks, true), want,
                                       cfg.detect_thresh);
            }
        }
    }

    SECTION("killed channels never reach the sums") {
        // A constant per-call mask over loud killed channels: the result is
        // the quiet stream's, bit for bit.
        const auto m = gpu_random_mask(32, 7, 0.4);
        auto loud    = stream;
        for (auto& block : loud) {
            for (SizeType c = 0; c < 32; ++c) {
                if (m[c] == 0) {
                    for (SizeType t = 0; t < 400; ++t) {
                        block[(c * 400) + t] = 12.0F; // clamps at 255
                    }
                }
            }
        }
        const auto noisy = gpu_pack(loud, 32, 400, 8, 128, 10.0F);
        const Masks first(input.size(), m);
        for (const auto mode : {PeakFindingMode::kContiguousPeakPerWidth,
                                PeakFindingMode::kContiguousPeakAcrossWidths,
                                PeakFindingMode::kAllAboveThreshold}) {
            DYNAMIC_SECTION("mode " << static_cast<int>(mode)) {
                auto cfg        = packed_cfg(mode);
                cfg.screen_mode = BoxcarScreenMode::kKadaneSurrogate;
                auto a          = make_packed(plan, cfg, kGpu);
                auto b          = make_packed(plan, cfg, kGpu);
                const auto base = host_run(a, input, first, true);
                REQUIRE(base.size() > 20);
                same_exact(host_run(b, noisy, first, true), base);
            }
        }
    }
}
