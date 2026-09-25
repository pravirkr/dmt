#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/bb_utils.hpp"

namespace dmt {

TEST_CASE("bb_utils::swap_spectrum rotates each row by n/2",
          "[bb_utils][cpu][internal]") {
    std::vector<ComplexType> a = {
        ComplexType(0.F, 0.F), ComplexType(1.F, 0.F), ComplexType(2.F, 0.F),
        ComplexType(3.F, 0.F), ComplexType(4.F, 0.F), ComplexType(5.F, 0.F),
        ComplexType(6.F, 0.F), ComplexType(7.F, 0.F),
    };
    auto b = a;
    bb_utils::swap_spectrum(a, b, 4, 2, /*nthreads=*/1);
    const std::vector<ComplexType> expected = {
        ComplexType(2.F, 0.F), ComplexType(3.F, 0.F), ComplexType(0.F, 0.F),
        ComplexType(1.F, 0.F), ComplexType(6.F, 0.F), ComplexType(7.F, 0.F),
        ComplexType(4.F, 0.F), ComplexType(5.F, 0.F),
    };
    REQUIRE_THAT(a, Catch::Matchers::Equals(expected));
    REQUIRE_THAT(b, Catch::Matchers::Equals(expected));
}

TEST_CASE("bb_utils::swap_spectrum rejects invalid sizes",
          "[bb_utils][cpu][internal]") {
    std::vector<ComplexType> a(4);
    std::vector<ComplexType> b(4);
    CHECK_THROWS_AS(bb_utils::swap_spectrum(a, b, 0, 1, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(bb_utils::swap_spectrum(a, b, 3, 1, 1),
                    std::invalid_argument);
    CHECK_THROWS_AS(bb_utils::swap_spectrum(a, b, 4, 2, 1),
                    std::invalid_argument);
}

TEST_CASE("bb_utils::compute_chirp fills a unit-magnitude tapered table",
          "[bb_utils][cpu][internal]") {
    const std::vector<float> dm_grid = {0.0F, 5.0F};
    const SizeType nbin              = 8;
    const SizeType nsub              = 2;
    const SizeType nchan             = 2;
    std::vector<ComplexType> table(dm_grid.size() * nsub * nbin);
    bb_utils::compute_chirp(dm_grid, table, 1250.0F, 100.0F, nbin, nsub, nchan);
    CHECK(std::ranges::all_of(table, [](const ComplexType& z) {
        return std::isfinite(z.real()) && std::isfinite(z.imag()) &&
               std::abs(z) <= 1.0F + 1.0E-5F;
    }));
    std::vector<ComplexType> too_small(3);
    CHECK_THROWS_AS(bb_utils::compute_chirp(dm_grid, too_small, 1250.0F, 100.0F,
                                            nbin, nsub, nchan),
                    std::runtime_error);
}

TEST_CASE("bb_utils::apply_chirp scales both polarisations",
          "[bb_utils][cpu][internal]") {
    const SizeType nsub = 1;
    const SizeType nbin = 4;
    const SizeType nfft = 2;
    std::vector<ComplexType> chirp(nsub * nbin, ComplexType(0.5F, 0.0F));
    std::vector<ComplexType> in1(nsub * nbin * nfft, ComplexType(2.0F, 0.0F));
    std::vector<ComplexType> in2(nsub * nbin * nfft, ComplexType(0.0F, 3.0F));
    std::vector<ComplexType> out1(in1.size());
    std::vector<ComplexType> out2(in2.size());
    bb_utils::apply_chirp(in1, in2, chirp, out1, out2, nsub, nbin, nfft, 0,
                          2.0F, /*nthreads=*/1);
    REQUIRE_THAT(out1, Catch::Matchers::Equals(std::vector<ComplexType>(
                           in1.size(), ComplexType(2.0F, 0.0F))));
    REQUIRE_THAT(out2, Catch::Matchers::Equals(std::vector<ComplexType>(
                           in2.size(), ComplexType(0.0F, 3.0F))));
}

TEST_CASE("bb_utils::unpad_detect writes Stokes I and drops overlap",
          "[bb_utils][cpu][internal]") {
    const SizeType nchan    = 2;
    const SizeType nfft     = 1;
    const SizeType nsub     = 1;
    const SizeType mbin     = 8;
    const SizeType noverlap = 2; // per-channel overlap = 1
    std::vector<ComplexType> p1(nfft * nsub * nchan * mbin,
                                ComplexType(3.0F, 4.0F));
    std::vector<ComplexType> p2(nfft * nsub * nchan * mbin,
                                ComplexType(0.0F, 0.0F));
    const SizeType mbin_adj = mbin - 2;
    std::vector<float> intensity(nsub * nchan * nfft * mbin_adj, -1.0F);
    bb_utils::unpad_detect(p1, p2, intensity, nchan, nfft, nsub, mbin, noverlap,
                           /*nthreads=*/1);
    REQUIRE_THAT(intensity, Catch::Matchers::Equals(
                                std::vector<float>(intensity.size(), 25.0F)));
}

} // namespace dmt
