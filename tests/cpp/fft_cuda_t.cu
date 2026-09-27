#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda/std/span>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dmt/bb_utils.hpp"
#include "dmt/bb_utils_cuda.cuh"
#include "dmt/utils/fft.hpp"
#include "test_helpers.hpp"

namespace dmt {

using utils::CUFFTManager;
using utils::FFTKind;
using utils::FFTWManager;

TEST_CASE("CUFFTManager forward matches FFTWManager", "[fft][gpu][parity]") {
    const int nfft   = 2;
    const int nsub   = 1;
    const int nbin   = 16;
    const SizeType n = static_cast<SizeType>(nfft * nsub * nbin);

    FFTWManager cpu(FFTKind::kC2CForward, static_cast<SizeType>(nbin),
                    static_cast<SizeType>(nfft * nsub), 1);

    std::vector<ComplexType> h1(n);
    std::vector<ComplexType> h2(n);
    for (SizeType i = 0; i < n; ++i) {
        h1[i] = ComplexType(static_cast<float>(i + 1), 0.0F);
        h2[i] = ComplexType(0.5F, static_cast<float>(i));
    }
    auto cpu1 = h1;
    auto cpu2 = h2;
    cpu.execute(cpu1);
    cpu.execute(cpu2);
    bb_utils::swap_spectrum(cpu1, cpu2, nbin, nfft * nsub, 1);

    CUFFTManager gpu(FFTKind::kC2CForward, static_cast<SizeType>(nbin),
                     static_cast<SizeType>(nfft * nsub), 0);
    thrust::device_vector<ComplexTypeCUDA> d1(n);
    thrust::device_vector<ComplexTypeCUDA> d2(n);
    for (SizeType i = 0; i < n; ++i) {
        d1[i] = ComplexTypeCUDA(h1[i].real(), h1[i].imag());
        d2[i] = ComplexTypeCUDA(h2[i].real(), h2[i].imag());
    }
    auto d1_span = cuda::std::span<ComplexTypeCUDA>(
        thrust::raw_pointer_cast(d1.data()), d1.size());
    auto d2_span = cuda::std::span<ComplexTypeCUDA>(
        thrust::raw_pointer_cast(d2.data()), d2.size());
    gpu.execute(d1_span);
    gpu.execute(d2_span);
    bb_utils::swap_spectrum(d1_span, d2_span, nbin, nfft * nsub, nullptr);
    cudaDeviceSynchronize();
    thrust::host_vector<ComplexTypeCUDA> g1 = d1;
    for (SizeType i = 0; i < n; ++i) {
        CHECK(g1[i].real() == Catch::Approx(cpu1[i].real()).margin(1e-3F));
        CHECK(g1[i].imag() == Catch::Approx(cpu1[i].imag()).margin(1e-3F));
    }
}

} // namespace dmt
