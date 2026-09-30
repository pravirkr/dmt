#pragma once

// Input rows of one Fourier-engine call on the CPU: float32, or packed at
// nbits (LSB first), read straight into the transform rows (no unpacked copy
// of the block). Shared by the FDMTFFT and DDMTFFT CPU engines.

#include <algorithm>
#include <cstdint>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::fft_cpu {

struct Source {
    const float* f{nullptr};
    const uint8_t* p{nullptr};
    SizeType row_bytes{0};
    SizeType nbits{32};
    SizeType nsamps{0};
};

// Samples [s0, s0 + n) of packed row @p row as float.
template <unsigned NB>
inline void unpack_range(const uint8_t* row, SizeType s0, SizeType n, float* out) {
    constexpr SizeType kPer = (NB < 8) ? (8 / NB) : 1;
    SizeType i              = 0;
    if constexpr (NB < 8) {
        for (; i < n && ((s0 + i) % kPer) != 0; ++i) {
            out[i] = static_cast<float>(
                bit_pack_utils::read_packed_sample<NB>(row, s0 + i));
        }
    }
    if (i < n) {
        const auto byte0 = ((s0 + i) * NB) / 8;
        bit_pack_utils::unpack_row<NB>(row + byte0, n - i, out + i);
    }
}

inline void copy_samples(
    const Source& src, SizeType row, SizeType s0, SizeType n, float* out) {
    if (src.f != nullptr) {
        std::copy_n(src.f + (row * src.nsamps) + s0, n, out);
        return;
    }
    const uint8_t* r = src.p + (row * src.row_bytes);
    switch (src.nbits) {
    case 1:
        unpack_range<1>(r, s0, n, out);
        break;
    case 2:
        unpack_range<2>(r, s0, n, out);
        break;
    case 4:
        unpack_range<4>(r, s0, n, out);
        break;
    case 8:
        unpack_range<8>(r, s0, n, out);
        break;
    case 16:
        unpack_range<16>(r, s0, n, out);
        break;
    default:
        break;
    }
}

} // namespace dmt::algorithms::fft_cpu
