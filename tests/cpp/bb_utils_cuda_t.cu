#ifdef DMT_ENABLE_CUDA

#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <cuda/std/span>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dmt/bb_utils.hpp"
#include "dmt/bb_utils_cuda.cuh"
#include "test_helpers.hpp"

namespace dmt {

TEST_CASE("bb_utils CUDA swap_spectrum matches CPU",
          "[bb_utils][gpu][parity][internal]") {
    std::vector<ComplexType> a = {ComplexType(0.F, 0.F), ComplexType(1.F, 0.F),
                                  ComplexType(2.F, 0.F), ComplexType(3.F, 0.F),
                                  ComplexType(4.F, 0.F), ComplexType(5.F, 0.F),
                                  ComplexType(6.F, 0.F), ComplexType(7.F, 0.F)};
    auto b                     = a;
    bb_utils::swap_spectrum(a, b, 4, 2, /*nthreads=*/1);

    thrust::device_vector<ComplexTypeCUDA> d1(8);
    thrust::device_vector<ComplexTypeCUDA> d2(8);
    for (int i = 0; i < 8; ++i) {
        d1[i] = ComplexTypeCUDA(static_cast<float>(i), 0.F);
        d2[i] = ComplexTypeCUDA(static_cast<float>(i), 0.F);
    }
    bb_utils::swap_spectrum(cuda::std::span<ComplexTypeCUDA>(
                                thrust::raw_pointer_cast(d1.data()), d1.size()),
                            cuda::std::span<ComplexTypeCUDA>(
                                thrust::raw_pointer_cast(d2.data()), d2.size()),
                            4, 2, nullptr);
    cudaDeviceSynchronize();
    thrust::host_vector<ComplexTypeCUDA> h1 = d1;
    for (int i = 0; i < 8; ++i) {
        CHECK(h1[i].real() == Catch::Approx(a[i].real()));
    }
}

TEST_CASE("bb_utils CUDA apply_chirp matches CPU",
          "[bb_utils][gpu][parity][internal]") {
    const SizeType nsub = 1;
    const SizeType nbin = 4;
    const SizeType nfft = 2;
    std::vector<ComplexType> chirp(nsub * nbin, ComplexType(0.5F, 0.0F));
    std::vector<ComplexType> in1(nsub * nbin * nfft, ComplexType(2.0F, 0.0F));
    std::vector<ComplexType> in2 = in1;
    std::vector<ComplexType> out1(in1.size());
    std::vector<ComplexType> out2(in2.size());
    bb_utils::apply_chirp(in1, in2, chirp, out1, out2, nsub, nbin, nfft, 0,
                          2.0F, /*nthreads=*/1);

    thrust::device_vector<ComplexTypeCUDA> d_in1(in1.size());
    thrust::device_vector<ComplexTypeCUDA> d_in2(in2.size());
    thrust::device_vector<ComplexTypeCUDA> d_ch(chirp.size());
    thrust::device_vector<ComplexTypeCUDA> d_out1(in1.size());
    thrust::device_vector<ComplexTypeCUDA> d_out2(in2.size());
    for (SizeType i = 0; i < in1.size(); ++i) {
        d_in1[i] = ComplexTypeCUDA(in1[i].real(), in1[i].imag());
        d_in2[i] = ComplexTypeCUDA(in2[i].real(), in2[i].imag());
    }
    for (SizeType i = 0; i < chirp.size(); ++i) {
        d_ch[i] = ComplexTypeCUDA(chirp[i].real(), chirp[i].imag());
    }
    bb_utils::apply_chirp(
        cuda::std::span<const ComplexTypeCUDA>(
            thrust::raw_pointer_cast(d_in1.data()), d_in1.size()),
        cuda::std::span<const ComplexTypeCUDA>(
            thrust::raw_pointer_cast(d_in2.data()), d_in2.size()),
        cuda::std::span<const ComplexTypeCUDA>(
            thrust::raw_pointer_cast(d_ch.data()), d_ch.size()),
        cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(d_out1.data()), d_out1.size()),
        cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(d_out2.data()), d_out2.size()),
        static_cast<int>(nsub), static_cast<int>(nbin), static_cast<int>(nfft),
        0, 2.0F, nullptr);
    cudaDeviceSynchronize();
    thrust::host_vector<ComplexTypeCUDA> h_out = d_out1;
    for (SizeType i = 0; i < out1.size(); ++i) {
        CHECK(h_out[i].real() == Catch::Approx(out1[i].real()));
        CHECK(h_out[i].imag() == Catch::Approx(out1[i].imag()));
    }
}

} // namespace dmt

#endif // DMT_ENABLE_CUDA
