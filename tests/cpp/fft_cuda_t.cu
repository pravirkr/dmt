#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda/std/span>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dmt/utils/fft.hpp"
#include "test_helpers.hpp"

namespace dmt {

using utils::FFTManagerCPU;
using utils::FFTManagerCUDA;

TEST_CASE("FFTManagerCUDA forward matches FFTManagerCPU", "[fft][gpu][parity]") {
    const int nfft  = 2;
    const int nsub  = 1;
    const int nbin  = 16;
    const int mbin  = 16;
    const int nchan = 1;
    const SizeType n = static_cast<SizeType>(nfft * nsub * nbin);

    FFTManagerCPU cpu(nfft, nsub, nbin, mbin, nchan, 1);
    std::vector<ComplexType> plan_u(n);
    std::vector<ComplexType> plan_d(n);
    cpu.initialize_plans(plan_u, plan_d);

    std::vector<ComplexType> h1(n);
    std::vector<ComplexType> h2(n);
    for (SizeType i = 0; i < n; ++i) {
        h1[i] = ComplexType(static_cast<float>(i + 1), 0.0F);
        h2[i] = ComplexType(0.5F, static_cast<float>(i));
    }
    auto cpu1 = h1;
    auto cpu2 = h2;
    cpu.forward_fft(cpu1, cpu2);

    FFTManagerCUDA gpu(nfft, nsub, nbin, mbin, nchan, 0);
    thrust::device_vector<ComplexTypeCUDA> d1(n);
    thrust::device_vector<ComplexTypeCUDA> d2(n);
    for (SizeType i = 0; i < n; ++i) {
        d1[i] = ComplexTypeCUDA(h1[i].real(), h1[i].imag());
        d2[i] = ComplexTypeCUDA(h2[i].real(), h2[i].imag());
    }
    gpu.forward_fft(
        cuda::std::span<ComplexTypeCUDA>(thrust::raw_pointer_cast(d1.data()),
                                         d1.size()),
        cuda::std::span<ComplexTypeCUDA>(thrust::raw_pointer_cast(d2.data()),
                                         d2.size()));
    cudaDeviceSynchronize();
    thrust::host_vector<ComplexTypeCUDA> g1 = d1;
    for (SizeType i = 0; i < n; ++i) {
        CHECK(g1[i].real() == Catch::Approx(cpu1[i].real()).margin(1e-3F));
        CHECK(g1[i].imag() == Catch::Approx(cpu1[i].imag()).margin(1e-3F));
    }
}

} // namespace dmt
