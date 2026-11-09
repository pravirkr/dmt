#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/fdmt.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_chunk.hpp"

// Chunked execution (chunk_nsamps, mode "valid"): a block of nsamps runs as
// chunks of the given length through the engine's own state, which must give
// exactly the output of the whole block in one chunk, for every chunk length
// (remainder chunks included), over several streamed blocks, with every
// performance parameter and with packed input.

namespace dmt {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

std::vector<float> random_block(SizeType n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0F, 1.0F);
    std::vector<float> wf(n);
    for (auto& v : wf) {
        v = dis(gen);
    }
    return wf;
}

std::vector<uint8_t>
random_packed(SizeType nrows, SizeType nsamps, SizeType nbits, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(
        0, bit_pack_utils::max_sample_value(nbits));
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    std::vector<uint8_t> packed(nrows * row_bytes, 0);
    for (SizeType r = 0; r < nrows; ++r) {
        uint8_t* row = packed.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const uint32_t v = dis(gen);
            switch (nbits) {
            case 1:
                bit_pack_utils::write_packed_sample<1>(row, s, v);
                break;
            case 2:
                bit_pack_utils::write_packed_sample<2>(row, s, v);
                break;
            default:
                bit_pack_utils::write_packed_sample<8>(row, s, v);
                break;
            }
        }
    }
    return packed;
}

struct Engine {
    FDMT fdmt;

    std::vector<float> run(const std::vector<float>& wf) {
        std::vector<float> dmt(
            fdmt.get_nbeams() * fdmt.get_plan().get_dmt_size(), -9.0F);
        fdmt.execute(wf, dmt);
        return dmt;
    }
    std::vector<float> run(const std::vector<uint8_t>& wf, SizeType nbits) {
        std::vector<float> dmt(
            fdmt.get_nbeams() * fdmt.get_plan().get_dmt_size(), -9.0F);
        fdmt.execute(std::span<const uint8_t>(wf), nbits, dmt);
        return dmt;
    }
};

Engine make(SizeType nchans,
            SizeType nsamps,
            IndexType dt_max,
            IndexType dt_min,
            SizeType nbeams,
            SizeType fuse,
            bool int_tree,
            SizeType chunk,
            bool smear = true) {
    return {FDMT(kFMin, kFMax, nchans, nsamps, kTsamp, dt_max, dt_min, 1, smear,
                 "valid", Exec::cpu(2), nbeams, fuse, int_tree, chunk)};
}

} // namespace

TEST_CASE("FDMT chunked execution is bit-exact with whole blocks",
          "[fdmt_cpu][cpu][chunk]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    // Delays longer than the chunks (history spanning several chunks),
    // signed grids, odd channel counts and wide level-0 boxes.
    const std::vector<Case> cases = {
        {64, 1000, 300, 0}, {37, 777, 120, -60}, {16, 640, 400, 250}};
    for (const auto& c : cases) {
        for (const SizeType chunk : {64, 128, 192, 1000}) {
            for (const bool smear : {true, false}) {
                for (const SizeType fuse : {SizeType{0}, kFDMTAutoFuse}) {
                    DYNAMIC_SECTION("nchans=" << c.nchans << " dt=[" << c.dt_min
                                              << "," << c.dt_max << "] chunk="
                                              << chunk << " smear=" << smear
                                              << " fuse=" << fuse) {
                        auto whole =
                            make(c.nchans, c.nsamps, c.dt_max, c.dt_min, 2,
                                 fuse, true, c.nsamps, smear);
                        auto chunked =
                            make(c.nchans, c.nsamps, c.dt_max, c.dt_min, 2,
                                 fuse, true, chunk, smear);
                        REQUIRE(whole.fdmt.get_chunk_nsamps() == c.nsamps);
                        REQUIRE(chunked.fdmt.get_chunk_nsamps() ==
                                std::min<SizeType>(chunk, c.nsamps));
                        for (unsigned blk = 0; blk < 3; ++blk) {
                            const auto wf =
                                random_block(2 * c.nchans * c.nsamps, 11 + blk);
                            REQUIRE_THAT(
                                chunked.run(wf),
                                Catch::Matchers::Equals(whole.run(wf)));
                        }
                        // Same stream state, in the saved (FIFO) layout.
                        std::vector<float> h_whole(
                            whole.fdmt.history_state_size());
                        std::vector<float> h_chunked(
                            chunked.fdmt.history_state_size());
                        whole.fdmt.save_history(h_whole);
                        chunked.fdmt.save_history(h_chunked);
                        REQUIRE_THAT(h_chunked,
                                     Catch::Matchers::Equals(h_whole));
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMT chunked execution with packed input and int tree",
          "[fdmt_cpu][cpu][chunk]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 704;
    for (const SizeType nbits : {1, 2, 8}) {
        for (const bool int_tree : {true, false}) {
            DYNAMIC_SECTION("nbits=" << nbits << " int_tree=" << int_tree) {
                auto whole   = make(nchans, nsamps, 260, 0, 1, kFDMTAutoFuse,
                                    int_tree, nsamps);
                auto chunked = make(nchans, nsamps, 260, 0, 1, kFDMTAutoFuse,
                                    int_tree, 128);
                for (unsigned blk = 0; blk < 3; ++blk) {
                    const auto wf =
                        random_packed(nchans, nsamps, nbits, 40 + blk);
                    REQUIRE_THAT(chunked.run(wf, nbits),
                                 Catch::Matchers::Equals(whole.run(wf, nbits)));
                }
            }
        }
    }
}

TEST_CASE("FDMT chunk streams interoperate through saved history",
          "[fdmt_cpu][cpu][chunk]") {
    // A stream handed back and forth between a whole-block and a chunked
    // engine (and a different chunk) through save_history()/load_history().
    const SizeType nchans = 32;
    const SizeType nsamps = 500;
    auto whole            = make(nchans, nsamps, 333, -40, 1, 0, true, nsamps);
    auto a                = make(nchans, nsamps, 333, -40, 1, 0, true, 64);
    auto b                = make(nchans, nsamps, 333, -40, 1, 2, true, 320);
    std::vector<float> hist(whole.fdmt.history_state_size(), 0.0F);
    whole.fdmt.save_history(hist); // a cold stream's history
    std::vector<float> h_ref(hist.size());
    for (unsigned blk = 0; blk < 6; ++blk) {
        const auto wf = random_block(nchans * nsamps, 70 + blk);
        auto& engine  = (blk % 2 == 0) ? a : b;
        engine.fdmt.load_history(hist);
        const auto got = engine.run(wf);
        engine.fdmt.save_history(hist);
        REQUIRE_THAT(got, Catch::Matchers::Equals(whole.run(wf)));
        whole.fdmt.save_history(h_ref);
        REQUIRE_THAT(hist, Catch::Matchers::Equals(h_ref));
    }
}

TEST_CASE("FDMT chunk length rules", "[fdmt_cpu][cpu][chunk]") {
    // Explicit chunks round up to a multiple of 64 and cap at nsamps.
    CHECK(make(32, 1000, 50, 0, 1, 0, true, 100).fdmt.get_chunk_nsamps() ==
          128);
    CHECK(make(32, 1000, 50, 0, 1, 0, true, 5000).fdmt.get_chunk_nsamps() ==
          1000);
    // Automatic: small plans run whole blocks.
    CHECK(make(32, 1000, 50, 0, 1, 0, true, 0).fdmt.get_chunk_nsamps() == 1000);
    // A plan whose whole-block state is far over the budget is chunked to
    // a power of two >= 2048, with the state of one chunk within budget.
    FDMT big(704.0F, 1216.0F, 512, 1 << 16, 64e-6F, 20000, 0, 1, true, "valid",
             Exec::cpu(1));
    const auto chunk = big.get_chunk_nsamps();
    CHECK(chunk >= 2048);
    CHECK(chunk < (SizeType{1} << 16));
    CHECK((chunk & (chunk - 1)) == 0);
    CHECK(big.get_memory_usage().state <= (SizeType{2} << 30));
    // Chunks need mode "valid".
    REQUIRE_THROWS_AS(FDMT(kFMin, kFMax, 32, 1000, kTsamp, 50, 0, 1, true,
                           "full", Exec::cpu(1), 1, 0, true, 128),
                      std::invalid_argument);
    // The stepper runs whole blocks only.
    auto chunked  = make(32, 1000, 50, 0, 1, 0, true, 128);
    const auto wf = random_block(SizeType{32} * 1000, 3);
    std::vector<float> dmt(chunked.fdmt.get_plan().get_dmt_size());
    REQUIRE_THROWS_AS(chunked.fdmt.reset(wf, dmt), std::logic_error);
    // The output buffer needs only dmt_size.
    std::vector<float> small(chunked.fdmt.get_plan().get_dmt_size() - 1);
    REQUIRE_THROWS_AS(chunked.fdmt.execute(wf, small), std::invalid_argument);
}

TEST_CASE("FDMT chunk sizing does not wrap", "[fdmt_cpu][cpu][chunk]") {
    using algorithms::detail::fdmt_cache_chunk;
    using algorithms::detail::fdmt_chunk_nsamps;
    using algorithms::detail::kFDMTMinChunk;
    plans::FDMTPlan plan(kFMin, kFMax, 32, 4096, kTsamp, 50);
    CHECK_THROWS_AS(
        fdmt_chunk_nsamps(plan, FDMTMode::kValid, 1,
                          std::numeric_limits<SizeType>::max() - 10),
        std::invalid_argument);

    const auto huge_beams = std::numeric_limits<SizeType>::max() / 2;
    const auto chunk = fdmt_chunk_nsamps(plan, FDMTMode::kValid, huge_beams, 0);
    CHECK(chunk == std::min(kFDMTMinChunk, plan.get_nsamps()));
    CHECK(chunk != 0);
    CHECK(fdmt_cache_chunk(plan, huge_beams, sizeof(float),
                           SizeType{2} << 30) == 0);
}

} // namespace dmt
