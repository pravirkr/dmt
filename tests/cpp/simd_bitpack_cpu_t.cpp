#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/simd_intrinsics.hpp"

#include <catch2/matchers/catch_matchers_all.hpp>

// Kernel-level checks for the FDMT low-bit / SIMD helpers against their
// scalar reference definitions, across every remainder length and output
// alignment (the vector paths have head/tail loops that plain FDMT tests
// with round block sizes would never reach).

namespace dmt {

namespace {

template <typename T> void check_unpack(SizeType nbits) {
    std::mt19937 gen(static_cast<unsigned>(nbits * 131));
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<SizeType> lengths;
    for (SizeType n = 0; n <= 70; ++n) {
        lengths.push_back(n);
    }
    lengths.push_back(1001);
    lengths.push_back(4099);
    for (const auto n : lengths) {
        std::vector<uint8_t> row(utils::packed_row_bytes(n, nbits) + 1);
        for (auto& b : row) {
            b = static_cast<uint8_t>(byte(gen));
        }
        std::vector<T> out(n + 1, T{123});
        utils::unpack_row(row.data(), nbits, n, out.data());
        for (SizeType i = 0; i < n; ++i) {
            uint32_t ref = 0;
            switch (nbits) {
            case 1:
                ref = utils::read_packed_sample<1>(row.data(), i);
                break;
            case 2:
                ref = utils::read_packed_sample<2>(row.data(), i);
                break;
            case 4:
                ref = utils::read_packed_sample<4>(row.data(), i);
                break;
            case 8:
                ref = utils::read_packed_sample<8>(row.data(), i);
                break;
            default:
                ref = utils::read_packed_sample<16>(row.data(), i);
                break;
            }
            REQUIRE(out[i] == static_cast<T>(ref));
        }
        REQUIRE(out[n] == T{123}); // never writes past nsamps
    }
}

} // namespace

TEST_CASE("unpack_row matches read_packed_sample", "[bitpack][cpu]") {
    for (const SizeType nbits : {1, 2, 4, 8, 16}) {
        DYNAMIC_SECTION("nbits=" << nbits) {
            if (nbits <= 8) {
                check_unpack<uint8_t>(nbits);
            }
            check_unpack<uint16_t>(nbits);
            check_unpack<float>(nbits);
        }
    }
}

TEST_CASE("add_stream_f32 is bit-exact with the plain add", "[simd][cpu]") {
    INFO("backend: " << simd::backend_name());
    std::mt19937 gen(7);
    std::normal_distribution<float> dis;
    std::vector<std::size_t> counts;
    for (std::size_t n = 0; n <= 100; ++n) {
        counts.push_back(n);
    }
    counts.push_back(8192 + 3);
    counts.push_back(65536 + 17);
    for (const auto n : counts) {
        // Offsets 0..16 floats cover every alignment of `out` (and of the
        // unaligned-load operands) relative to a 64-byte vector.
        for (std::size_t off = 0; off <= 16; ++off) {
            std::vector<float> a(n + off);
            std::vector<float> b(n + off);
            for (auto& v : a) {
                v = dis(gen);
            }
            for (auto& v : b) {
                v = dis(gen);
            }
            std::vector<float> out(n + off + 1, -1.0F);
            std::vector<float> ref(n + off + 1, -1.0F);
            simd::add_stream_f32(a.data() + off, b.data() + off,
                                 out.data() + off, n);
            for (std::size_t i = 0; i < n; ++i) {
                ref[off + i] = a[off + i] + b[off + i];
            }
            REQUIRE_THAT(out, Catch::Matchers::Equals(ref));
        }
    }
}

} // namespace dmt
