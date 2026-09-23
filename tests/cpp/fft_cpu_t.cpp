#include <complex>
#include <span>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/utils/fft.hpp"

namespace dmt {

using utils::FFTManagerCPU;

TEST_CASE("FFTManagerCPU forward then backward recovers scaled input",
          "[fft][cpu]") {
    const int nfft  = 2;
    const int nsub  = 2;
    const int nbin  = 16;
    const int mbin  = 16;
    const int nchan = 1;

    FFTManagerCPU fft(nfft, nsub, nbin, mbin, nchan, 1);
    const auto unpack_n = nfft * nsub * nbin;
    const auto delay_n  = nfft * nsub * nchan * mbin;
    std::vector<ComplexType> unpack(unpack_n, ComplexType(0.0F, 0.0F));
    std::vector<ComplexType> delay(delay_n, ComplexType(0.0F, 0.0F));
    fft.initialize_plans(unpack, delay);

    // FFTW_MEASURE overwrites the planning buffers; fill afterwards.
    std::vector<ComplexType> data1(unpack_n);
    std::vector<ComplexType> data2(unpack_n);
    for (SizeType i = 0; i < unpack_n; ++i) {
        data1[i] = ComplexType(static_cast<float>(i + 1), 0.25F);
        data2[i] = ComplexType(static_cast<float>(i + 2), -0.5F);
    }
    auto orig1 = data1;
    auto orig2 = data2;

    fft.forward_fft(data1, data2);
    fft.backward_fft(data1, data2);

    // Unnormalized C2C: IFFT(FFT(x)) = nbin * x. Spectrum swap cancels.
    for (SizeType i = 0; i < unpack_n; ++i) {
        CHECK(data1[i].real() ==
              Catch::Approx(orig1[i].real() * static_cast<float>(nbin))
                  .margin(1e-3F));
        CHECK(data1[i].imag() ==
              Catch::Approx(orig1[i].imag() * static_cast<float>(nbin))
                  .margin(1e-3F));
        CHECK(data2[i].real() ==
              Catch::Approx(orig2[i].real() * static_cast<float>(nbin))
                  .margin(1e-3F));
        CHECK(data2[i].imag() ==
              Catch::Approx(orig2[i].imag() * static_cast<float>(nbin))
                  .margin(1e-3F));
    }
}

TEST_CASE("FFTManagerCPU throws if plans are used before initialize",
          "[fft][cpu]") {
    FFTManagerCPU fft(1, 1, 8, 8, 1, 1);
    std::vector<ComplexType> data(8, ComplexType(1.0F, 0.0F));
    CHECK_THROWS_AS(fft.forward_fft(data, data), std::logic_error);
    CHECK_THROWS_AS(fft.backward_fft(data, data), std::logic_error);
}

} // namespace dmt
