#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <random>
#include <span>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bit_pack_utils.hpp"

// FDMTCUDA level fusion (the fuse_levels constructor parameter). The fused
// kernel does the same float additions as the level-by-level kernels, so fused
// vs unfused on the GPU is bit-exact for any input; against FDMTCPU the input
// is integer-valued, which keeps every sum exact under fast-math.

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTCUDA;
using algorithms::kFDMTAutoFuse;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;
// fuse_levels / int_tree of the original level-by-level float path.
constexpr SizeType kUnfused = 0;
constexpr bool kFloatTree   = false;

std::vector<float> random_floats(SizeType n, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis;
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dis(gen);
    }
    return v;
}

struct Packed2 {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

Packed2 random_2bit(SizeType nrows, SizeType nsamps, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, 3);
    const auto row_bytes = utils::packed_row_bytes(nsamps, 2);
    Packed2 out{.values = std::vector<float>(nrows * nsamps),
                .packed = std::vector<uint8_t>(nrows * row_bytes, 0)};
    for (SizeType r = 0; r < nrows; ++r) {
        for (SizeType s = 0; s < nsamps; ++s) {
            const auto v                 = dis(gen);
            out.values[(r * nsamps) + s] = static_cast<float>(v);
            utils::write_packed_sample<2>(out.packed.data() + (r * row_bytes),
                                          s, v);
        }
    }
    return out;
}

template <typename Fdmt>
std::vector<float> run(Fdmt& fdmt, const std::vector<float>& wf) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const float>(wf), std::span<float>(dmt));
    return dmt;
}

template <typename Fdmt>
std::vector<float> run_2bit(Fdmt& fdmt, const std::vector<uint8_t>& wf) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const uint8_t>(wf), 2, std::span<float>(dmt));
    return dmt;
}

// Leading dmt_size values of every beam, concatenated.
template <typename Fdmt>
std::vector<float> beams(const Fdmt& fdmt, const std::vector<float>& dmt) {
    const auto stride = fdmt.get_plan().get_buffer_size();
    const auto n      = fdmt.get_plan().get_dmt_size();
    std::vector<float> out;
    for (SizeType b = 0; b < fdmt.get_nbeams(); ++b) {
        out.insert(out.end(),
                   dmt.begin() + static_cast<std::ptrdiff_t>(b * stride),
                   dmt.begin() + static_cast<std::ptrdiff_t>((b * stride) + n));
    }
    return out;
}

} // namespace

TEST_CASE("FDMTCUDA fused levels are bit-exact with the unfused kernels",
          "[fdmt_gpu][gpu]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    const std::vector<Case> cases = {
        {64, 256, 48, 0},    // even tree
        {37, 100, 40, 0},    // odd channel counts -> copy sub-bands
        {64, 256, 32, -32},  // symmetric DM range
        {32, 128, -4, -40},  // purely negative
        {16, 128, 90, 60},   // wide level-0 boxes
        {4, 64, 8, 0},       // fewer levels than the requested depth
        {1024, 1500, 256, 0} // several tiles per row
    };
    for (const auto& c : cases) {
        for (const std::string mode : {"full", "roll", "valid"}) {
            for (const bool smearing : {true, false}) {
                DYNAMIC_SECTION("nchans=" << c.nchans << " dt=[" << c.dt_min
                                          << "," << c.dt_max << "] mode="
                                          << mode << " smearing=" << smearing) {
                    const auto wf =
                        random_floats(2 * c.nchans * c.nsamps,
                                      static_cast<unsigned>(c.nchans));
                    FDMTCUDA ref(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                 c.dt_max, c.dt_min, 1, smearing, mode, false,
                                 0, 2, kUnfused);
                    REQUIRE(ref.get_fuse_levels() == 0);
                    const auto expected = beams(ref, run(ref, wf));
                    for (const SizeType fuse :
                         {SizeType{1}, SizeType{2}, SizeType{3}, SizeType{5},
                          kFDMTAutoFuse}) {
                        FDMTCUDA gpu(kFMin, kFMax, c.nchans, c.nsamps, kTsamp,
                                     c.dt_max, c.dt_min, 1, smearing, mode,
                                     false, 0, 2, fuse);
                        INFO("fuse=" << fuse
                                     << " effective=" << gpu.get_fuse_levels());
                        REQUIRE_THAT(beams(gpu, run(gpu, wf)),
                                     Catch::Matchers::Equals(expected));
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTCUDA fused packed int tree matches FDMTCPU", "[fdmt_gpu][gpu]") {
    for (const std::string mode : {"full", "roll", "valid"}) {
        for (const bool smearing : {true, false}) {
            DYNAMIC_SECTION("mode=" << mode << " smearing=" << smearing) {
                const SizeType nchans = 256;
                const SizeType nsamps = 300;
                const auto wf         = random_2bit(nchans, nsamps, 5);
                FDMTCPU cpu(kFMin, kFMax, nchans, nsamps, kTsamp, 64, 0, 1,
                            smearing, mode, false, 1, 1, kUnfused, kFloatTree);
                const auto expected = beams(cpu, run(cpu, wf.values));
                for (const SizeType fuse :
                     {SizeType{2}, SizeType{4}, kFDMTAutoFuse}) {
                    FDMTCUDA gpu(kFMin, kFMax, nchans, nsamps, kTsamp, 64, 0, 1,
                                 smearing, mode, false, 0, 1, fuse, true);
                    REQUIRE_THAT(beams(gpu, run_2bit(gpu, wf.packed)),
                                 Catch::Matchers::Equals(expected));
                }
            }
        }
    }
}

TEST_CASE("FDMTCUDA fused valid-mode streaming matches FDMTCPU",
          "[fdmt_gpu][gpu]") {
    // Blocks shorter than the fused levels' delays exercise the history
    // shift written by the first tile. One stream alternates between a fused
    // and an unfused engine through save_history()/load_history().
    const SizeType nchans = 32;
    const SizeType nbeams = 2;
    for (const SizeType block : {4, 16, 64, 700}) {
        for (const bool smearing : {true, false}) {
            DYNAMIC_SECTION("block=" << block << " smearing=" << smearing) {
                FDMTCPU cpu(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1,
                            smearing, "valid", false, 1, nbeams, kUnfused,
                            kFloatTree);
                FDMTCUDA fused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1,
                               smearing, "valid", false, 0, nbeams, 3);
                FDMTCUDA unfused(kFMin, kFMax, nchans, block, kTsamp, 48, -8, 1,
                                 smearing, "valid", false, 0, nbeams, kUnfused);
                thrust::device_vector<float> hist(fused.history_state_size(),
                                                  0.0F);
                const cuda::std::span<float> hist_out(
                    thrust::raw_pointer_cast(hist.data()), hist.size());
                const cuda::std::span<const float> hist_in(
                    thrust::raw_pointer_cast(hist.data()), hist.size());
                for (SizeType b = 0; b < 8; ++b) {
                    FDMTCUDA& gpu = (b % 3 == 2) ? unfused : fused;
                    const auto wf = random_2bit(nbeams * nchans, block,
                                                static_cast<unsigned>(80 + b));
                    gpu.load_history(hist_in);
                    const auto got = beams(gpu, run_2bit(gpu, wf.packed));
                    gpu.save_history(hist_out);
                    cudaDeviceSynchronize();
                    REQUIRE_THAT(got, Catch::Matchers::Equals(
                                          beams(cpu, run(cpu, wf.values))));
                }
            }
        }
    }
}

TEST_CASE("FDMTCUDA fusion depth resolution and memory usage",
          "[fdmt_gpu][gpu]") {
    const auto make = [](SizeType fuse) {
        return FDMTCUDA(704.0F, 1216.0F, 4096, 2048, 0.00008192F, 2048, 0, 1,
                        true, "valid", false, 0, 1, fuse);
    };
    // Default: automatic, and a realistic band fuses at least one level.
    auto automatic    = make(kFDMTAutoFuse);
    const auto niters = automatic.get_plan().get_niters();
    CHECK(automatic.get_fuse_levels() >= 1);
    CHECK(automatic.get_fuse_levels() <= niters);
    CHECK(automatic.get_int_tree());
    CHECK(make(0).get_fuse_levels() == 0);
    CHECK(make(2).get_fuse_levels() == 2);
    const auto deep = make(99);
    CHECK(deep.get_fuse_levels() >= 1);
    CHECK(deep.get_fuse_levels() <= 8);

    const auto mem  = automatic.get_memory_usage();
    const auto& pln = automatic.get_plan();
    CHECK(mem.state == pln.get_buffer_size() * sizeof(float));
    CHECK(mem.output == pln.get_buffer_size() * sizeof(float));
    CHECK(mem.plan > 0);
    CHECK(mem.workspace > 0);                         // fused tables
    CHECK(make(0).get_memory_usage().workspace == 0); // no fusion

    // The stepper stays unfused and inspectable.
    const SizeType nchans = 64;
    const SizeType nsamps = 256;
    FDMTCUDA small(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, true,
                   "valid", false, 0, 1, 3);
    const auto wf = random_floats(nchans * nsamps, 3);
    thrust::device_vector<float> d_wf(wf.begin(), wf.end());
    thrust::device_vector<float> d_dmt(small.get_plan().get_buffer_size());
    small.reset(cuda::std::span<const float>(
                    thrust::raw_pointer_cast(d_wf.data()), d_wf.size()),
                cuda::std::span<float>(thrust::raw_pointer_cast(d_dmt.data()),
                                       d_dmt.size()));
    REQUIRE(small.current_level() == 0);
    REQUIRE_NOTHROW(small.view_level_data());
    small.finalize();
}

} // namespace dmt
