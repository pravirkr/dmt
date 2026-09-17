#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cuda/std/span>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dmt/utils/unpacker.hpp"
#include "test_helpers.hpp"

namespace dmt {

using utils::DataUnpackerCPU;
using utils::DataUnpackerCUDA;

TEST_CASE("DataUnpackerCUDA matches DataUnpackerCPU",
          "[unpacker][gpu][parity]") {
    const SizeType nsub     = 2;
    const SizeType nbin     = 8;
    const SizeType noverlap = 2;
    const SizeType nfft     = 2;
    const SizeType nsamp    = nfft * (nbin - (2 * noverlap));
    const SizeType in_n     = 2 * 2 * nsamp * nsub;
    const SizeType out_n    = nfft * nsub * nbin;

    std::vector<uint8_t> data_in(in_n);
    for (SizeType i = 0; i < in_n; ++i) {
        data_in[i] = static_cast<uint8_t>((i % 17) + 1);
    }

    DataUnpackerCPU cpu(nsub, nbin, noverlap, nfft, "PRITF");
    DataUnpackerCUDA gpu(nsub, nbin, noverlap, nfft, "PRITF");
    std::vector<ComplexType> p1(out_n);
    std::vector<ComplexType> p2(out_n);
    cpu.execute<uint8_t>(data_in, p1, p2);

    thrust::device_vector<ComplexTypeCUDA> d_p1(out_n);
    thrust::device_vector<ComplexTypeCUDA> d_p2(out_n);
    gpu.execute<uint8_t>(
        std::span<const uint8_t>(data_in),
        cuda::std::span<ComplexTypeCUDA>(thrust::raw_pointer_cast(d_p1.data()),
                                         d_p1.size()),
        cuda::std::span<ComplexTypeCUDA>(thrust::raw_pointer_cast(d_p2.data()),
                                         d_p2.size()));

    thrust::host_vector<ComplexTypeCUDA> h_p1 = d_p1;
    thrust::host_vector<ComplexTypeCUDA> h_p2 = d_p2;
    for (SizeType i = 0; i < out_n; ++i) {
        CHECK(h_p1[i].real() == Catch::Approx(p1[i].real()));
        CHECK(h_p1[i].imag() == Catch::Approx(p1[i].imag()));
        CHECK(h_p2[i].real() == Catch::Approx(p2[i].real()));
        CHECK(h_p2[i].imag() == Catch::Approx(p2[i].imag()));
    }
}

} // namespace dmt
