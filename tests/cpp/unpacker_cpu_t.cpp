#include <array>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "dmt/baseband_layout.hpp"
#include "dmt/common/baseband.hpp"
#include "dmt/unpacker.hpp"
#include "dmt/utils/simulate.hpp"

namespace dmt {

using utils::BasebandUnpackerCPU;

namespace {

constexpr SizeType kNbin     = 64;
constexpr SizeType kNfft     = 3;
constexpr SizeType kNoverlap = 8;
constexpr SizeType kStep     = kNbin - (2 * kNoverlap);
constexpr SizeType kNsamps   = (kNfft * kStep) + (2 * kNoverlap);

// Random integer-valued voltages representable exactly by @p format.
std::vector<ComplexType> random_voltages(const BasebandFormat& format,
                                         SizeType nsub,
                                         uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<ComplexType> v(2 * nsub * kNsamps);
    const auto draw = [&]() -> float {
        if (format.nbits == 2) {
            return format.levels_2bit[rng() % 4];
        }
        const int half = 1 << (format.nbits - 1);
        return static_cast<float>(static_cast<int>(rng() % (2 * half)) - half);
    };
    for (auto& x : v) {
        const float re = draw();
        const float im = draw();
        x              = {re, im};
    }
    return v;
}

// out[s][j][p][i] must equal v[p][s][j * step + i].
void check_blocks(const std::vector<ComplexType>& v,
                  const std::vector<ComplexType>& out,
                  SizeType nsub) {
    for (SizeType p = 0; p < 2; ++p) {
        for (SizeType j = 0; j < kNfft; ++j) {
            for (SizeType s = 0; s < nsub; ++s) {
                for (SizeType i = 0; i < kNbin; ++i) {
                    const auto got =
                        out[(((((s * kNfft) + j) * 2) + p) * kNbin) + i];
                    const auto want =
                        v[(((p * nsub) + s) * kNsamps) + (j * kStep) + i];
                    if (got != want) {
                        FAIL("mismatch at pol " << p << " block " << j
                                                << " sub " << s << " bin "
                                                << i);
                    }
                }
            }
        }
    }
}

} // namespace

TEST_CASE("BasebandUnpackerCPU round-trips every order and bit width",
          "[unpacker][cpu]") {
    const std::array<std::string, 24> orders = {
        "PRITF", "PRIFT", "PTRIF", "PTFRI", "PFRIT", "PFTRI",
        "RIPTF", "RIPFT", "RITPF", "RITFP", "RIFPT", "RIFTP",
        "TPRIF", "TPFRI", "TRIPF", "TRIFP", "TFPRI", "TFRIP",
        "FPRIT", "FPTRI", "FRIPT", "FRITP", "FTPRI", "FTRIP"};
    const auto order = GENERATE_COPY(from_range(orders));
    struct Enc {
        SizeType nbits;
        bool is_signed;
        bool msb_first;
    };
    const auto enc = GENERATE(Enc{8, true, true}, Enc{8, false, true},
                              Enc{4, true, true}, Enc{4, false, false},
                              Enc{2, true, false}, Enc{2, true, true});
    const BasebandFormat format{.order     = order,
                                .nbits     = enc.nbits,
                                .is_signed = enc.is_signed,
                                .msb_first = enc.msb_first};
    const SizeType nsub = 3;
    CAPTURE(order, enc.nbits, enc.is_signed, enc.msb_first);
    const auto v     = random_voltages(format, nsub, 7);
    const auto bytes = utils::pack_baseband(v, nsub, kNsamps, format, 1.0F);
    const std::vector<SizeType> groups{nsub};
    const BasebandUnpackerCPU unpacker(format, groups, kNbin, kNfft,
                                       kNoverlap, 2);
    REQUIRE(unpacker.block_nsamps() == kNsamps);
    REQUIRE(bytes.size() == unpacker.input_size(0));
    std::vector<ComplexType> out(unpacker.output_size());
    const std::span<const uint8_t> in(bytes);
    unpacker.execute(std::span(&in, 1), out);
    check_blocks(v, out, nsub);
}

TEST_CASE("BasebandUnpackerCPU decodes hand-built GUPPI and LOFAR bytes",
          "[unpacker][cpu]") {
    const SizeType nsub = 2;
    // Distinct value per (pol, ri, t, s) within int8 range.
    const auto value = [](SizeType p, SizeType ri, SizeType t, SizeType s) {
        return static_cast<int>((((p * 2) + ri) * 29) + (t % 13) + (s * 5)) -
               64;
    };
    std::vector<ComplexType> v(2 * nsub * kNsamps);
    for (SizeType p = 0; p < 2; ++p) {
        for (SizeType s = 0; s < nsub; ++s) {
            for (SizeType t = 0; t < kNsamps; ++t) {
                v[(((p * nsub) + s) * kNsamps) + t] = {
                    static_cast<float>(value(p, 0, t, s)),
                    static_cast<float>(value(p, 1, t, s))};
            }
        }
    }
    const std::vector<SizeType> groups{nsub};
    SECTION("GUPPI FTPRI int8: Re(X), Im(X), Re(Y), Im(Y) per sample") {
        std::vector<int8_t> raw(4 * nsub * kNsamps);
        for (SizeType s = 0; s < nsub; ++s) {
            for (SizeType t = 0; t < kNsamps; ++t) {
                for (SizeType p = 0; p < 2; ++p) {
                    for (SizeType ri = 0; ri < 2; ++ri) {
                        raw[(s * kNsamps * 4) + (t * 4) + (p * 2) + ri] =
                            static_cast<int8_t>(value(p, ri, t, s));
                    }
                }
            }
        }
        const BasebandUnpackerCPU unpacker(BasebandFormat{.order = "FTPRI"},
                                           groups, kNbin, kNfft, kNoverlap);
        std::vector<ComplexType> out(unpacker.output_size());
        const std::span<const uint8_t> in(
            reinterpret_cast<const uint8_t*>(raw.data()), raw.size());
        unpacker.execute(std::span(&in, 1), out);
        check_blocks(v, out, nsub);
    }
    SECTION("GUPPI FTPRI 4-bit: real in the high nibble") {
        std::vector<ComplexType> v4(v.size());
        for (SizeType i = 0; i < v.size(); ++i) {
            v4[i] = {static_cast<float>((static_cast<int>(v[i].real()) & 7) -
                                        4),
                     static_cast<float>((static_cast<int>(v[i].imag()) & 7) -
                                        3)};
        }
        std::vector<uint8_t> raw(2 * nsub * kNsamps);
        for (SizeType s = 0; s < nsub; ++s) {
            for (SizeType t = 0; t < kNsamps; ++t) {
                for (SizeType p = 0; p < 2; ++p) {
                    const auto x  = v4[(((p * nsub) + s) * kNsamps) + t];
                    const auto re = static_cast<unsigned>(
                                        static_cast<int>(x.real())) &
                                    0xFU;
                    const auto im = static_cast<unsigned>(
                                        static_cast<int>(x.imag())) &
                                    0xFU;
                    raw[(s * kNsamps * 2) + (t * 2) + p] =
                        static_cast<uint8_t>((re << 4U) | im);
                }
            }
        }
        const BasebandUnpackerCPU unpacker(
            BasebandFormat{.order = "FTPRI", .nbits = 4}, groups, kNbin, kNfft,
            kNoverlap);
        std::vector<ComplexType> out(unpacker.output_size());
        const std::span<const uint8_t> in(raw);
        unpacker.execute(std::span(&in, 1), out);
        check_blocks(v4, out, nsub);
    }
    SECTION("LOFAR PRITF uint8 offset binary") {
        std::vector<uint8_t> raw(4 * nsub * kNsamps);
        for (SizeType p = 0; p < 2; ++p) {
            for (SizeType ri = 0; ri < 2; ++ri) {
                for (SizeType t = 0; t < kNsamps; ++t) {
                    for (SizeType s = 0; s < nsub; ++s) {
                        raw[(((((p * 2) + ri) * kNsamps) + t) * nsub) + s] =
                            static_cast<uint8_t>(value(p, ri, t, s) + 128);
                    }
                }
            }
        }
        const BasebandUnpackerCPU unpacker(
            BasebandFormat{.order = "PRITF", .is_signed = false}, groups,
            kNbin, kNfft, kNoverlap);
        std::vector<ComplexType> out(unpacker.output_size());
        const std::span<const uint8_t> in(raw);
        unpacker.execute(std::span(&in, 1), out);
        check_blocks(v, out, nsub);
    }
}

TEST_CASE("BasebandUnpackerCPU multi-group input equals one group",
          "[unpacker][cpu]") {
    const BasebandFormat format{.order = "TFPRI"};
    const SizeType nsub = 5;
    const auto v        = random_voltages(format, nsub, 11);
    const std::vector<SizeType> one{nsub};
    const std::vector<SizeType> two{2, 3};
    const BasebandUnpackerCPU u1(format, one, kNbin, kNfft, kNoverlap);
    const BasebandUnpackerCPU u2(format, two, kNbin, kNfft, kNoverlap);
    const auto all = utils::pack_baseband(v, nsub, kNsamps, format, 1.0F);
    const auto g0  = utils::pack_baseband(v, nsub, kNsamps, format, 1.0F, 0, 2);
    const auto g1  = utils::pack_baseband(v, nsub, kNsamps, format, 1.0F, 2, 3);
    std::vector<ComplexType> out1(u1.output_size());
    std::vector<ComplexType> out2(u2.output_size());
    const std::span<const uint8_t> in_all(all);
    u1.execute(std::span(&in_all, 1), out1);
    const std::array<std::span<const uint8_t>, 2> in_two{
        std::span<const uint8_t>(g0), std::span<const uint8_t>(g1)};
    u2.execute(in_two, out2);
    REQUIRE(out1 == out2);
    check_blocks(v, out1, nsub);
}

TEST_CASE("BasebandUnpackerCPU validation", "[unpacker][cpu]") {
    const std::vector<SizeType> groups{2};
    CHECK_THROWS_AS(utils::parse_baseband_order("PRIT"), std::invalid_argument);
    CHECK_THROWS_AS(utils::parse_baseband_order("PPRITF"),
                    std::invalid_argument);
    CHECK_THROWS_AS(utils::parse_baseband_order("PRXTF"),
                    std::invalid_argument);
    CHECK_THROWS_AS(
        BasebandUnpackerCPU(BasebandFormat{.nbits = 3}, groups, kNbin, kNfft,
                            kNoverlap),
        std::invalid_argument);
    CHECK_THROWS_AS(
        BasebandUnpackerCPU(BasebandFormat{}, groups, 16, kNfft, 8),
        std::invalid_argument);
    const BasebandUnpackerCPU unpacker(BasebandFormat{}, groups, kNbin, kNfft,
                                       kNoverlap);
    std::vector<uint8_t> bad(unpacker.input_size(0) - 1);
    std::vector<ComplexType> out(unpacker.output_size());
    const std::span<const uint8_t> in(bad);
    CHECK_THROWS_AS(unpacker.execute(std::span(&in, 1), out),
                    std::invalid_argument);
    std::vector<uint8_t> good(unpacker.input_size(0));
    const std::span<const uint8_t> in_good(good);
    std::vector<ComplexType> small(unpacker.output_size() - 1);
    CHECK_THROWS_AS(unpacker.execute(std::span(&in_good, 1), small),
                    std::invalid_argument);
}

} // namespace dmt
