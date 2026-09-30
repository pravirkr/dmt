#include <array>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/unpacker.hpp"
#include "dmt/unpacker_cuda.cuh"
#include "dmt/utils/simulate.hpp"
#include "test_helpers.hpp"

namespace dmt {

using utils::BasebandUnpackerCPU;
using utils::BasebandUnpackerCUDA;

TEST_CASE("BasebandUnpackerCUDA matches BasebandUnpackerCPU",
          "[unpacker][gpu][parity]") {
    const SizeType nbin     = 64;
    const SizeType nfft     = 3;
    const SizeType noverlap = 8;
    const SizeType nsub     = 5;
    const SizeType nsamps   = (nfft * (nbin - (2 * noverlap))) + (2 * noverlap);
    const auto order = GENERATE(std::string("FTPRI"), std::string("PRITF"),
                                std::string("TFPRI"), std::string("RITFP"));
    const auto nbits = GENERATE(SizeType{8}, SizeType{4}, SizeType{2});
    const auto is_signed = GENERATE(true, false);
    const BasebandFormat format{.order     = order,
                                .nbits     = nbits,
                                .is_signed = is_signed};
    CAPTURE(order, nbits, is_signed);
    std::mt19937 rng(3);
    std::vector<ComplexType> v(2 * nsub * nsamps);
    for (auto& x : v) {
        x = {static_cast<float>(static_cast<int>(rng() % 15) - 7),
             static_cast<float>(static_cast<int>(rng() % 15) - 7)};
    }
    const std::vector<SizeType> groups{2, 3};
    const auto g0 = utils::pack_baseband(v, nsub, nsamps, format, 1.0F, 0, 2);
    const auto g1 = utils::pack_baseband(v, nsub, nsamps, format, 1.0F, 2, 3);

    const BasebandUnpackerCPU cpu(format, groups, nbin, nfft, noverlap);
    std::vector<ComplexType> want(cpu.output_size());
    const std::array<std::span<const uint8_t>, 2> h_groups{
        std::span<const uint8_t>(g0), std::span<const uint8_t>(g1)};
    cpu.execute(h_groups, want);

    BasebandUnpackerCUDA gpu(format, groups, nbin, nfft, noverlap);
    thrust::device_vector<uint8_t> d0(g0.begin(), g0.end());
    thrust::device_vector<uint8_t> d1(g1.begin(), g1.end());
    const std::array<cuda::std::span<const uint8_t>, 2> d_groups{
        cuda::std::span<const uint8_t>(thrust::raw_pointer_cast(d0.data()),
                                       d0.size()),
        cuda::std::span<const uint8_t>(thrust::raw_pointer_cast(d1.data()),
                                       d1.size())};
    thrust::device_vector<ComplexTypeGPU> d_out(gpu.output_size());
    gpu.execute(d_groups,
                cuda::std::span<ComplexTypeGPU>(
                    thrust::raw_pointer_cast(d_out.data()), d_out.size()));
    const thrust::host_vector<ComplexTypeGPU> got = d_out;
    REQUIRE(got.size() == want.size());
    for (SizeType i = 0; i < want.size(); ++i) {
        if (got[i].real() != want[i].real() ||
            got[i].imag() != want[i].imag()) {
            FAIL("mismatch at element " << i);
        }
    }
}

} // namespace dmt
