#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <thrust/device_vector.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"

#include "test_helpers.hpp"

// Skewed-window GPU path (fdmt_cuda.cu: skewed direct levels, batched mid
// levels and the target-ring direct top). Small plans take the unskewed
// direct top, so the engines here force the skewed path
// (FDMTEngineConfig::force_skewed_direct) and check that they took it.

namespace dmt {

using algorithms::kFDMTAutoFuse;

namespace {

// Engines on one plan for the skewed-path cases: the GPU engine forced onto
// the skewed direct levels + target-ring top (small plans otherwise take
// the unskewed direct top), the GPU tree, and CPU references.
struct SkewRig {
    static constexpr float kFMin     = 704.0F;
    static constexpr float kFMax     = 4032.0F;
    static constexpr SizeType kChans = 2048;
    static constexpr SizeType kSamps = 512;
    static constexpr float kTsamp    = 0.000064F;
    SizeType nbeams;
    SizeType chunk;
    plans::FDMTPlan plan;
    SkewRig(SizeType beams, SizeType ch)
        : nbeams(beams),
          chunk(ch),
          plan(kFMin,
               kFMax,
               kChans,
               kSamps,
               kTsamp,
               std::vector<IndexType>{0, 90000, 180000},
               "valid") {}

    [[nodiscard]] std::unique_ptr<algorithms::detail::FDMTEngine>
    gpu(bool skewed, SizeType ch = 0) const {
        algorithms::detail::FDMTEngineConfig cfg{};
        cfg.exec                = test::gpu_exec();
        cfg.nbeams              = nbeams;
        cfg.chunk_nsamps        = ch == 0 ? chunk : ch;
        cfg.fuse_levels         = skewed ? kFDMTAutoFuse : 0;
        cfg.force_skewed_direct = skewed;
        auto e                  = algorithms::detail::make_fdmt_gpu(plan, cfg);
        INFO("path: " << e->search_path());
        REQUIRE(e->search_path().starts_with("skewed direct levels") == skewed);
        return e;
    }
    [[nodiscard]] std::unique_ptr<algorithms::detail::FDMTEngine>
    cpu(bool skewed) const {
        algorithms::detail::FDMTEngineConfig cfg{};
        cfg.exec               = Exec::cpu(2);
        cfg.nbeams             = nbeams;
        cfg.chunk_nsamps       = chunk;
        cfg.skew_history_bytes = skewed ? 0 : SIZE_MAX;
        auto e                 = algorithms::detail::make_fdmt_cpu(plan, cfg);
        REQUIRE(e->search_path().empty() == !skewed);
        return e;
    }
    [[nodiscard]] SizeType dmt_size() const {
        return nbeams * plan.get_dmt_size();
    }
    // Values 0..7 keep every level's sum exact in float (the GPU's direct
    // top adds in a different order than the tree).
    [[nodiscard]] std::vector<uint8_t> block(std::mt19937& gen) const {
        std::uniform_int_distribution<uint32_t> dis(0, 7);
        std::vector<uint8_t> v(nbeams * kChans * kSamps);
        for (auto& x : v) {
            x = static_cast<uint8_t>(dis(gen));
        }
        return v;
    }
    [[nodiscard]] std::vector<float> run(algorithms::detail::FDMTEngine& e,
                                         const std::vector<uint8_t>& b,
                                         bool packed) const {
        std::vector<float> out(dmt_size(), -1.0F);
        if (packed) {
            e.execute(std::span<const uint8_t>(b), 8, std::span<float>(out));
        } else {
            const std::vector<float> wf(b.begin(), b.end());
            e.execute(std::span<const float>(wf), std::span<float>(out));
        }
        return out;
    }
};

} // namespace

TEST_CASE("FDMT (gpu) skewed direct and batched target-ring top match CPU",
          "[fdmt_gpu][gpu][skew]") {
    const SkewRig rig(2, 128);
    auto skew = rig.gpu(true);
    auto tree = rig.gpu(false);
    auto cpu  = rig.cpu(false);
    REQUIRE(skew->get_fuse_levels() == 3);
    std::mt19937 gen(1337);
    for (int blk = 0; blk < 3; ++blk) {
        const auto b = rig.block(gen);
        const auto x = rig.run(*skew, b, false);
        test::require_exact(x, rig.run(*tree, b, false));
        test::require_exact(x, rig.run(*cpu, b, false));
    }
    skew->reset_history();
    tree->reset_history();
    for (int blk = 0; blk < 2; ++blk) {
        const auto b = rig.block(gen);
        test::require_exact(rig.run(*skew, b, true), rig.run(*tree, b, true));
    }
}

TEST_CASE("FDMT (gpu) skewed path: packed input, history and input types",
          "[fdmt_gpu][gpu][skew]") {
    std::mt19937 gen(4242);

    SECTION("packed input keeps the tree's rounded box means") {
        const SkewRig rig(2, 128);
        auto gpu = rig.gpu(true);
        auto cpu = rig.cpu(false);
        for (int i = 0; i < 3; ++i) {
            const auto b = rig.block(gen);
            test::require_exact(rig.run(*gpu, b, true), rig.run(*cpu, b, true));
        }
    }

    SECTION("save/load resumes a stream exactly; layout is checked") {
        const SkewRig rig(2, 128);
        auto a               = rig.gpu(true);
        const SizeType hsize = a->history_state_size();
        REQUIRE(hsize > 0);
        const auto b0 = rig.block(gen);
        const auto b1 = rig.block(gen);
        const auto b2 = rig.block(gen);
        const auto bx = rig.block(gen);
        rig.run(*a, b0, false);
        rig.run(*a, b1, true);
        REQUIRE(a->history_state_size() == hsize);
        std::vector<float> hist(hsize);
        a->save_history(hist);
        const auto ref = rig.run(*a, b2, false);
        auto c         = rig.gpu(true);
        c->load_history(hist);
        test::require_exact(rig.run(*c, b2, false), ref);
        // Multiplexing: another stream in between, then resume.
        a->reset_history();
        rig.run(*a, bx, false);
        a->load_history(hist);
        test::require_exact(rig.run(*a, b2, false), ref);
        // Device-memory round trip gives the same buffer back.
        thrust::device_vector<float> dh(hist.begin(), hist.end());
        c->load_history(DeviceSpan<const float>(
                            thrust::raw_pointer_cast(dh.data()), dh.size()),
                        Stream{});
        thrust::device_vector<float> back(hsize, 0.0F);
        c->save_history(
            DeviceSpan<float>(thrust::raw_pointer_cast(back.data()), hsize),
            Stream{});
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<float> back_h(hsize);
        thrust::copy(back.begin(), back.end(), back_h.begin());
        test::require_exact(back_h, hist);
        // Another layout (chunk), the tree's format, or junk is rejected.
        auto other = rig.gpu(true, 2 * rig.chunk);
        std::vector<float> other_buf(other->history_state_size());
        if (other_buf.size() == hsize) {
            other_buf = hist;
        }
        REQUIRE_THROWS_AS(other->load_history(other_buf),
                          std::invalid_argument);
        auto tree = rig.gpu(false);
        std::vector<float> tree_buf(tree->history_state_size());
        if (tree_buf.size() == hsize) {
            tree->save_history(tree_buf);
            REQUIRE_THROWS_AS(a->load_history(tree_buf), std::invalid_argument);
        }
        std::vector<float> junk(hsize, 0.5F);
        REQUIRE_THROWS_AS(a->load_history(junk), std::invalid_argument);
    }

    SECTION("switching input type keeps the stream (matches CPU skew)") {
        // Rows straddling a switch use the current block's box-mean
        // convention on both skewed engines (recomputed from samples; the
        // tree keeps the previous block's for its stored rows), and with
        // delays far beyond a block every row straddles here: compare with
        // the CPU skewed engine (one beam).
        const SkewRig rig(1, 128);
        auto gpu = rig.gpu(true);
        auto cpu = rig.cpu(true);
        for (int i = 0; i < 5; ++i) {
            const bool packed = (i % 2 == 1);
            const auto b      = rig.block(gen);
            CAPTURE(i); // odd blocks are packed
            test::require_exact(rig.run(*gpu, b, packed),
                                rig.run(*cpu, b, packed));
        }
    }
}

} // namespace dmt
