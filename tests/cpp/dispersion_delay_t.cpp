// dmt::dispersion_delay against dmtlib.dispersion_delay (values from Python).

#include <bit>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dmt/common/types.hpp"

TEST_CASE("dispersion_delay matches dmtlib.dispersion_delay", "[types]") {
    // Built from bits so -ffast-math does not warn on
    // std::numeric_limits::infinity().
    constexpr double kInf =
        std::bit_cast<double>(uint64_t{0x7FF0000000000000ULL});
    struct Case {
        double dm, f_lo, f_hi, seconds;
    };
    for (const auto& c : {Case{100.0, 1200.0, 1600.0, 0.1260661595205164},
                          Case{413.0, 704.0, 4032.0, 3.3522905689855267},
                          Case{411.0, 704.0, 1536.0, 2.7181184236106946},
                          Case{56.7, 400.0, kInf, 1.4704356846473032},
                          Case{0.0, 300.0, 800.0, 0.0}}) {
        INFO("dm " << c.dm << " f_lo " << c.f_lo << " f_hi " << c.f_hi);
        CHECK(dmt::dispersion_delay(c.dm, c.f_lo, c.f_hi) ==
              Catch::Approx(c.seconds).epsilon(1e-14).margin(1e-300));
    }
    // The library constant, and the sign: a lower f_hi than f_lo gives a
    // negative delay (f_lo arrives first).
    CHECK(dmt::dispersion_delay(1.0, 1.0, kInf) ==
          Catch::Approx(static_cast<double>(dmt::kDispConst)).epsilon(1e-7));
    CHECK(dmt::dispersion_delay(10.0, 1600.0, 1200.0) ==
          Catch::Approx(-dmt::dispersion_delay(10.0, 1200.0, 1600.0))
              .epsilon(1e-15));
    static_assert(dmt::dispersion_delay(0.0, 1.0, 2.0) == 0.0);
}
