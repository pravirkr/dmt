#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/utils/unpacker.hpp"

namespace dmt {

using utils::DataUnpackerCPU;

namespace {
constexpr SizeType kNpol = 2;

SizeType input_size(SizeType nfft, SizeType nbin, SizeType noverlap,
                    SizeType nsub) {
    const auto nsamp = nfft * (nbin - (2 * noverlap));
    return kNpol * 2 * nsamp * nsub;
}

SizeType output_size(SizeType nfft, SizeType nsub, SizeType nbin) {
    return nfft * nsub * nbin;
}

template <typename T>
std::vector<T> sequential_baseband(SizeType n) {
    std::vector<T> data(n);
    for (SizeType i = 0; i < n; ++i) {
        data[i] = static_cast<T>(static_cast<int>(i % 17) + 1);
    }
    return data;
}
} // namespace

TEST_CASE("DataUnpackerCPU rejects invalid construction", "[unpacker][cpu]") {
    CHECK_THROWS_AS(DataUnpackerCPU(2, 8, 8, 1, "PRITF"), std::invalid_argument);
    CHECK_THROWS_AS(DataUnpackerCPU(2, 16, 2, 1, "NOPE"), std::invalid_argument);
}

TEST_CASE("DataUnpackerCPU unpacks PRITF/FTPRI/RITFP layouts", "[unpacker][cpu]") {
    const SizeType nsub     = 2;
    const SizeType nbin     = 8;
    const SizeType noverlap = 2;
    const SizeType nfft     = 2;
    const auto in_n         = input_size(nfft, nbin, noverlap, nsub);
    const auto out_n        = output_size(nfft, nsub, nbin);
    const auto nsamp        = nfft * (nbin - (2 * noverlap));

    const std::vector<std::string_view> orders = {"PRITF", "FTPRI", "RITFP"};
    for (const auto order : orders) {
        DYNAMIC_SECTION("order=" << order) {
            DataUnpackerCPU unpacker(nsub, nbin, noverlap, nfft, order);
            auto data_in = sequential_baseband<uint8_t>(in_n);
            std::vector<ComplexType> p1(out_n, ComplexType(99.0F, 99.0F));
            std::vector<ComplexType> p2(out_n, ComplexType(99.0F, 99.0F));
            unpacker.execute<uint8_t>(data_in, p1, p2);

            // Overlap pad at the start of FFT 0 is zeros.
            CHECK(p1[0] == ComplexType(0.0F, 0.0F));
            CHECK(p2[0] == ComplexType(0.0F, 0.0F));

            // A valid interior bin of FFT 0 / sub 0 is a finite conversion
            // of the packed integer input (not left as the dirty sentinel).
            const SizeType ibin = noverlap + 1;
            CHECK(p1[ibin].real() != 99.0F);
            CHECK(std::isfinite(p1[ibin].real()));
            CHECK(std::isfinite(p1[ibin].imag()));
            CHECK(p1.size() == out_n);
            CHECK(nsamp > 0);
        }
    }
}

TEST_CASE("DataUnpackerCPU uint8 and int8 agree up to signed conversion",
          "[unpacker][cpu]") {
    const SizeType nsub     = 1;
    const SizeType nbin     = 8;
    const SizeType noverlap = 2;
    const SizeType nfft     = 1;
    DataUnpackerCPU unpacker(nsub, nbin, noverlap, nfft, "PRITF");
    const auto in_n  = input_size(nfft, nbin, noverlap, nsub);
    const auto out_n = output_size(nfft, nsub, nbin);

    std::vector<uint8_t> u8(in_n, 7);
    std::vector<int8_t> i8(in_n, 7);
    std::vector<ComplexType> p1_u(out_n);
    std::vector<ComplexType> p2_u(out_n);
    std::vector<ComplexType> p1_i(out_n);
    std::vector<ComplexType> p2_i(out_n);
    unpacker.execute<uint8_t>(u8, p1_u, p2_u);
    unpacker.execute<int8_t>(i8, p1_i, p2_i);
    REQUIRE_THAT(p1_u, Catch::Matchers::Equals(p1_i));
    REQUIRE_THAT(p2_u, Catch::Matchers::Equals(p2_i));
}

TEST_CASE("DataUnpackerCPU throws on buffer size mismatch", "[unpacker][cpu]") {
    DataUnpackerCPU unpacker(2, 8, 2, 1, "PRITF");
    std::vector<uint8_t> bad_in(3, 1);
    std::vector<ComplexType> p1(16);
    std::vector<ComplexType> p2(16);
    CHECK_THROWS_AS(unpacker.execute<uint8_t>(bad_in, p1, p2), std::runtime_error);
}

} // namespace dmt
