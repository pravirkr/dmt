#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/utils/fft.hpp"

namespace dmt {

using utils::FFTKind;
using utils::FFTWManager;

TEST_CASE("FFTWManager C2C forward then backward recovers scaled input",
          "[fft][cpu]") {
    const SizeType length  = 16;
    const SizeType howmany = 5;
    const int nthreads     = 3;

    FFTWManager forward(FFTKind::kC2CForward, length, howmany, nthreads);
    FFTWManager backward(FFTKind::kC2CBackward, length, howmany, nthreads);

    std::vector<ComplexType> data(length * howmany);
    for (SizeType i = 0; i < data.size(); ++i) {
        data[i] = ComplexType(static_cast<float>(i + 1), 0.25F);
    }
    const auto original = data;

    forward.execute(data);
    backward.execute(data);

    for (SizeType i = 0; i < data.size(); ++i) {
        CHECK(data[i].real() ==
              Catch::Approx(original[i].real() * static_cast<float>(length))
                  .margin(1e-3F));
        CHECK(data[i].imag() ==
              Catch::Approx(original[i].imag() * static_cast<float>(length))
                  .margin(1e-3F));
    }
}

TEST_CASE("FFTWManager R2C then C2R recovers scaled input", "[fft][cpu]") {
    const SizeType length    = 32;
    const SizeType howmany   = 7;
    const SizeType n_complex = (length / 2) + 1;
    const int nthreads       = 4;

    FFTWManager forward(FFTKind::kR2C, length, howmany, nthreads);
    FFTWManager backward(FFTKind::kC2R, length, howmany, nthreads);

    std::vector<float> real(length * howmany);
    std::vector<ComplexType> freq(howmany * n_complex);
    for (SizeType i = 0; i < real.size(); ++i) {
        real[i] = static_cast<float>(i + 1);
    }
    const auto original = real;

    forward.execute(real, freq);
    backward.execute(real, freq);

    for (SizeType i = 0; i < real.size(); ++i) {
        CHECK(real[i] == Catch::Approx(original[i] * static_cast<float>(length))
                             .margin(1e-2F));
    }
}

TEST_CASE("FFTWManager rejects a mismatched buffer and the wrong execute",
          "[fft][cpu]") {
    FFTWManager plan(FFTKind::kC2CForward, 8, 2, 1);
    std::vector<ComplexType> data(8, ComplexType(1.0F, 0.0F));
    std::vector<float> real(16, 1.0F);
    CHECK_THROWS_AS(plan.execute(data), std::invalid_argument);
    CHECK_THROWS_AS(plan.execute(real, data), std::logic_error);
    CHECK_THROWS_AS(FFTWManager(FFTKind::kR2C, 0, 4, 1), std::invalid_argument);
}

} // namespace dmt
