#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cuda/std/span>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thrust/device_vector.h>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bit_pack_utils.hpp"

// FDMTCUDA packed low-bit input and narrow-integer tree (FDMTExecConfig::
// int_tree). Integer-valued input keeps every partial sum an exact integer
// below 2^24, which stays exact under any float reassociation (Release
// builds use fast-math), so all comparisons here are bit-exact -- including
// CUDA vs CPU.

namespace dmt {

using algorithms::FDMTCPU;
using algorithms::FDMTCUDA;
using algorithms::FDMTExecConfig;

namespace {

constexpr float kFMin  = 1000.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

struct PackedWaterfall {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

PackedWaterfall
random_packed(SizeType nrows, SizeType nsamps, SizeType nbits, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0,
                                                utils::max_sample_value(nbits));
    const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
    PackedWaterfall out{.values = std::vector<float>(nrows * nsamps),
                        .packed = std::vector<uint8_t>(nrows * row_bytes, 0)};
    for (SizeType r = 0; r < nrows; ++r) {
        uint8_t* row = out.packed.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const uint32_t v             = dis(gen);
            out.values[(r * nsamps) + s] = static_cast<float>(v);
            switch (nbits) {
            case 1:
                utils::write_packed_sample<1>(row, s, v);
                break;
            case 2:
                utils::write_packed_sample<2>(row, s, v);
                break;
            case 4:
                utils::write_packed_sample<4>(row, s, v);
                break;
            case 8:
                utils::write_packed_sample<8>(row, s, v);
                break;
            default:
                utils::write_packed_sample<16>(row, s, v);
                break;
            }
        }
    }
    return out;
}

template <typename Fdmt>
std::vector<float> run_float(Fdmt& fdmt, const std::vector<float>& wf) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const float>(wf), std::span<float>(dmt));
    return dmt;
}

template <typename Fdmt>
std::vector<float>
run_packed(Fdmt& fdmt, const std::vector<uint8_t>& wf, SizeType nbits) {
    std::vector<float> dmt(
        fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size(), -777.0F);
    fdmt.execute(std::span<const uint8_t>(wf), nbits, std::span<float>(dmt));
    return dmt;
}

// Leading `n` values of beam `beam`, beams `stride` elements apart (both
// backends use stride = get_buffer_size()).
std::vector<float> beam_slice(const std::vector<float>& dmt,
                              SizeType stride,
                              SizeType beam,
                              SizeType n) {
    return {dmt.begin() + static_cast<std::ptrdiff_t>(beam * stride),
            dmt.begin() + static_cast<std::ptrdiff_t>((beam * stride) + n)};
}

} // namespace

TEST_CASE("FDMTCUDA packed input matches float input and FDMTCPU",
          "[fdmt_gpu][gpu]") {
    struct Case {
        SizeType nchans;
        SizeType nsamps;
        IndexType dt_max;
        IndexType dt_min;
    };
    const std::vector<Case> cases = {{64, 203, 40, 0},
                                     {37, 128, 24, -24},
                                     {256, 128, 64, 0},
                                     {16, 128, 90, 60}};
    for (const auto& c : cases) {
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            for (const std::string mode : {"full", "roll", "valid"}) {
                for (const bool smearing : {true, false}) {
                    DYNAMIC_SECTION("nchans=" << c.nchans << " nbits=" << nbits
                                              << " mode=" << mode
                                              << " smearing=" << smearing) {
                        const auto wf = random_packed(c.nchans, c.nsamps, nbits,
                                                      17 + nbits);
                        FDMTCUDA gpu_float(kFMin, kFMax, c.nchans, c.nsamps,
                                           kTsamp, c.dt_max, c.dt_min, 1,
                                           smearing, mode);
                        const auto expected = run_float(gpu_float, wf.values);
                        const auto n = gpu_float.get_plan().get_dmt_size();
                        for (const bool int_tree : {false, true}) {
                            FDMTCUDA gpu(kFMin, kFMax, c.nchans, c.nsamps,
                                         kTsamp, c.dt_max, c.dt_min, 1,
                                         smearing, mode);
                            gpu.set_exec_config({.int_tree = int_tree});
                            const auto got = run_packed(gpu, wf.packed, nbits);
                            REQUIRE_THAT(beam_slice(got, 0, 0, n),
                                         Catch::Matchers::Equals(
                                             beam_slice(expected, 0, 0, n)));

                            FDMTCPU cpu(kFMin, kFMax, c.nchans, c.nsamps,
                                        kTsamp, c.dt_max, c.dt_min, 1, smearing,
                                        mode);
                            cpu.set_exec_config({.int_tree = int_tree});
                            const auto cpu_out =
                                run_packed(cpu, wf.packed, nbits);
                            REQUIRE_THAT(beam_slice(got, 0, 0, n),
                                         Catch::Matchers::Equals(
                                             beam_slice(cpu_out, 0, 0, n)));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("FDMTCUDA packed valid-mode streaming matches FDMTCPU",
          "[fdmt_gpu][gpu]") {
    // Blocks smaller than the delay exercise the multi-block history FIFOs
    // (level 0 from packed samples, tree levels from integer state).
    const SizeType nchans = 64;
    for (const SizeType block : {16, 64}) {
        for (const SizeType nbits : {1, 2, 8}) {
            DYNAMIC_SECTION("block=" << block << " nbits=" << nbits) {
                FDMTCUDA gpu(kFMin, kFMax, nchans, block, kTsamp, 48, 0, 1,
                             true, "valid");
                FDMTCPU cpu(kFMin, kFMax, nchans, block, kTsamp, 48, 0, 1, true,
                            "valid");
                gpu.set_exec_config({.int_tree = true});
                cpu.set_exec_config({.int_tree = true});
                const auto n = cpu.get_plan().get_dmt_size();
                for (SizeType b = 0; b < 8; ++b) {
                    const auto wf = random_packed(
                        nchans, block, nbits, static_cast<unsigned>(90 + b));
                    const auto g = run_packed(gpu, wf.packed, nbits);
                    const auto h = run_packed(cpu, wf.packed, nbits);
                    REQUIRE_THAT(
                        beam_slice(g, 0, 0, n),
                        Catch::Matchers::Equals(beam_slice(h, 0, 0, n)));
                }
            }
        }
    }
}

TEST_CASE("FDMTCUDA packed multi-beam matches per-beam execution",
          "[fdmt_gpu][gpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 128;
    const SizeType nbeams = 3;
    const SizeType nbits  = 2;
    const auto wf         = random_packed(nbeams * nchans, nsamps, nbits, 5);
    FDMTCUDA multi(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, true, "full",
                   false, 0, nbeams);
    multi.set_exec_config({.int_tree = true});
    const auto got        = run_packed(multi, wf.packed, nbits);
    const auto& plan      = multi.get_plan();
    const auto n          = plan.get_dmt_size();
    const auto gpu_stride = plan.get_buffer_size(); // documented layout
    const auto row_bytes  = utils::packed_row_bytes(nsamps, nbits);
    for (SizeType b = 0; b < nbeams; ++b) {
        std::vector<uint8_t> one(
            wf.packed.begin() +
                static_cast<std::ptrdiff_t>(b * nchans * row_bytes),
            wf.packed.begin() +
                static_cast<std::ptrdiff_t>((b + 1) * nchans * row_bytes));
        FDMTCUDA single(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, true,
                        "full");
        single.set_exec_config({.int_tree = true});
        const auto ref = run_packed(single, one, nbits);
        REQUIRE_THAT(beam_slice(got, gpu_stride, b, n),
                     Catch::Matchers::Equals(beam_slice(ref, 0, 0, n)));
    }
}

TEST_CASE("FDMTCUDA packed device-memory execute and stepper",
          "[fdmt_gpu][gpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 128;
    const SizeType nbits  = 1;
    const auto wf         = random_packed(nchans, nsamps, nbits, 3);
    FDMTCUDA ref(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full");
    const auto expected = run_float(ref, wf.values);
    const auto n        = ref.get_plan().get_dmt_size();

    FDMTCUDA gpu(kFMin, kFMax, nchans, nsamps, kTsamp, 32, 0, 1, false, "full");
    gpu.set_exec_config({.int_tree = true});
    thrust::device_vector<uint8_t> wf_d(wf.packed.begin(), wf.packed.end());
    thrust::device_vector<float> dmt_d(gpu.get_plan().get_buffer_size(), 0.0F);
    const cuda::std::span<const uint8_t> wf_span(
        thrust::raw_pointer_cast(wf_d.data()), wf_d.size());
    const cuda::std::span<float> dmt_span(
        thrust::raw_pointer_cast(dmt_d.data()), dmt_d.size());

    gpu.execute(wf_span, nbits, dmt_span);
    cudaDeviceSynchronize();
    std::vector<float> got(dmt_d.size());
    thrust::copy(dmt_d.begin(), dmt_d.end(), got.begin());
    REQUIRE_THAT(beam_slice(got, 0, 0, n),
                 Catch::Matchers::Equals(beam_slice(expected, 0, 0, n)));

    // Stepper: 1-bit, no smearing -> level 0 is uint8 and not inspectable;
    // the root is float.
    gpu.reset(wf_span, nbits, dmt_span);
    REQUIRE_THROWS_AS(gpu.view_level_data(), std::logic_error);
    gpu.advance_until_remaining(0);
    REQUIRE_NOTHROW(gpu.view_level_data());
    gpu.finalize();
    cudaDeviceSynchronize();
    thrust::copy(dmt_d.begin(), dmt_d.end(), got.begin());
    REQUIRE_THAT(beam_slice(got, 0, 0, n),
                 Catch::Matchers::Equals(beam_slice(expected, 0, 0, n)));

    // Validation.
    REQUIRE_THROWS_AS(gpu.execute(wf_span, 3, dmt_span), std::invalid_argument);
    REQUIRE_THROWS_AS(gpu.execute(wf_span, 2, dmt_span), std::invalid_argument);
}

TEST_CASE("FDMTCUDA rejects per-beam extents beyond 32-bit indexing",
          "[fdmt_gpu][gpu]") {
    // The device plan holds per-beam offsets as int32. This block's state
    // buffer is ~5e10 elements per beam, so construction must fail cleanly
    // (before any device allocation) instead of wrapping indices.
    const float f_min = 704.0F;
    const float f_max = 1216.0F;
    const float tsamp = 0.00008192F;
    const auto nsamps = SizeType{1} << 23;
    const auto plan_sz =
        plans::FDMTPlan(f_min, f_max, 4096, nsamps, tsamp, 2048)
            .get_buffer_size();
    REQUIRE(plan_sz > static_cast<SizeType>(INT32_MAX));
    REQUIRE_THROWS_AS(FDMTCUDA(f_min, f_max, 4096, nsamps, tsamp, 2048),
                      std::invalid_argument);
}

} // namespace dmt
