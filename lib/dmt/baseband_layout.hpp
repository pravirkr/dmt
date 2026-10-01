#pragma once

// Baseband block layout (dmt::BasebandFormat) -> element strides and sample
// decoding, shared by the CPU and GPU unpackers.

#include <array>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

#include "dmt/common/baseband.hpp"
#include "dmt/common/types.hpp"

namespace dmt::utils {

/// Axes of a baseband block.
enum class BasebandAxis : uint8_t { kPol = 0, kRI = 1, kTime = 2, kFreq = 3 };

/// Parsed axis order, outermost first.
using BasebandAxisOrder = std::array<BasebandAxis, 4>;

[[nodiscard]] inline BasebandAxisOrder
parse_baseband_order(std::string_view order) {
    BasebandAxisOrder axes{};
    SizeType n    = 0;
    unsigned seen = 0;
    for (SizeType i = 0; i < order.size();) {
        BasebandAxis ax{};
        if (order.substr(i, 2) == "RI") {
            ax = BasebandAxis::kRI;
            i += 2;
        } else if (order[i] == 'P') {
            ax = BasebandAxis::kPol;
            ++i;
        } else if (order[i] == 'T') {
            ax = BasebandAxis::kTime;
            ++i;
        } else if (order[i] == 'F') {
            ax = BasebandAxis::kFreq;
            ++i;
        } else {
            n = 5; // invalid token
            break;
        }
        const unsigned bit = 1U << static_cast<unsigned>(ax);
        if (n >= 4 || (seen & bit) != 0) {
            n = 5;
            break;
        }
        seen |= bit;
        axes[n++] = ax;
    }
    if (n != 4) {
        throw std::invalid_argument(std::format(
            "BasebandFormat: invalid order '{}'; expected a permutation of "
            "the tokens P, RI, T and F (e.g. 'FTPRI', 'PRITF', 'TFPRI')",
            order));
    }
    return axes;
}

inline void validate_baseband_format(const BasebandFormat& format) {
    static_cast<void>(parse_baseband_order(format.order));
    if (format.nbits != 2 && format.nbits != 4 && format.nbits != 8) {
        throw std::invalid_argument(std::format(
            "BasebandFormat: nbits={} unsupported (2, 4 or 8)", format.nbits));
    }
}

/// Bytes of one block of @p nsamps samples and @p nsub subbands.
[[nodiscard]] inline SizeType baseband_block_bytes(const BasebandFormat& format,
                                                   SizeType nsamps,
                                                   SizeType nsub) {
    return (SizeType{4} * nsamps * nsub * format.nbits) / 8;
}

/// Element strides (in elements, not bytes) of each axis for a block of
/// @p nsamps samples and @p nsub subbands.
struct BasebandStrides {
    SizeType pol;
    SizeType ri;
    SizeType time;
    SizeType freq;
};

[[nodiscard]] inline BasebandStrides
baseband_strides(const BasebandFormat& format, SizeType nsamps, SizeType nsub) {
    const auto axes = parse_baseband_order(format.order);
    std::array<SizeType, 4> size{};
    size[static_cast<unsigned>(BasebandAxis::kPol)]  = 2;
    size[static_cast<unsigned>(BasebandAxis::kRI)]   = 2;
    size[static_cast<unsigned>(BasebandAxis::kTime)] = nsamps;
    size[static_cast<unsigned>(BasebandAxis::kFreq)] = nsub;
    std::array<SizeType, 4> stride{};
    SizeType s = 1;
    for (SizeType i = 4; i-- > 0;) {
        stride[static_cast<unsigned>(axes[i])] = s;
        s *= size[static_cast<unsigned>(axes[i])];
    }
    return {.pol  = stride[0],
            .ri   = stride[1],
            .time = stride[2],
            .freq = stride[3]};
}

/// Byte -> decoded element values. A byte holds 8 / nbits elements; entry
/// [byte][k] is the k-th element (in memory order) of that byte.
struct BasebandDecodeTable {
    SizeType per_byte{1};
    std::array<std::array<float, 4>, 256> value{};
};

[[nodiscard]] inline BasebandDecodeTable
make_decode_table(const BasebandFormat& format) {
    validate_baseband_format(format);
    BasebandDecodeTable table;
    table.per_byte       = 8 / format.nbits;
    const unsigned nbits = static_cast<unsigned>(format.nbits);
    const unsigned mask  = (1U << nbits) - 1U;
    for (unsigned byte = 0; byte < 256; ++byte) {
        for (unsigned k = 0; k < table.per_byte; ++k) {
            const unsigned shift =
                format.msb_first ? 8U - (nbits * (k + 1U)) : nbits * k;
            const unsigned code = (byte >> shift) & mask;
            float v             = 0.0F;
            if (nbits == 2) {
                v = format.levels_2bit[code];
            } else if (format.is_signed) {
                // Two's complement of nbits bits.
                const int half = 1 << (nbits - 1U);
                v = static_cast<float>(static_cast<int>(code) >= half
                                           ? static_cast<int>(code) - (2 * half)
                                           : static_cast<int>(code));
            } else {
                v = static_cast<float>(static_cast<int>(code) -
                                       (1 << (nbits - 1U)));
            }
            table.value[byte][k] = v;
        }
    }
    return table;
}

} // namespace dmt::utils
