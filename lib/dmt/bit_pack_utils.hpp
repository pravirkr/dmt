#pragma once

#include <algorithm>
#include <cstdint>

#include "dmt/common/types.hpp"

#ifdef __CUDACC__
#define DMT_HOST_DEVICE __host__ __device__
#else
#define DMT_HOST_DEVICE
#endif

namespace dmt::utils {

/**
 * @brief Read the NBITS-wide sample at `sample_idx` from a packed row.
 * @details
 * Matches dedisp packing convention: for NBITS < 8, samples are packed
 * LSB-first within a byte (sample 0 occupies the low NBITS bits); for
 * NBITS == 8 a sample is exactly one byte; for NBITS == 16 a sample is two
 * bytes, little-endian. NBITS == 32 (float) never goes through this path.
 */
template <unsigned NBITS>
DMT_HOST_DEVICE inline uint32_t read_packed_sample(const uint8_t* row,
                                                   SizeType sample_idx) noexcept {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4 || NBITS == 8 ||
                      NBITS == 16,
                  "read_packed_sample: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 16) {
        const auto* p = row + (sample_idx * 2);
        return static_cast<uint32_t>(p[0]) |
               (static_cast<uint32_t>(p[1]) << 8U);
    } else if constexpr (NBITS == 8) {
        return row[sample_idx];
    } else {
        constexpr int kPerByte   = 8 / static_cast<int>(NBITS);
        constexpr uint32_t kMask = (1U << NBITS) - 1U;
        const auto byte_idx      = sample_idx / static_cast<SizeType>(kPerByte);
        const auto sub_idx = sample_idx % static_cast<SizeType>(kPerByte);
        return (static_cast<uint32_t>(row[byte_idx]) >>
                (static_cast<unsigned>(sub_idx) * NBITS)) &
               kMask;
    }
}

/**
 * @brief Write the NBITS-wide sample at `sample_idx` into a packed row.
 */
template <unsigned NBITS>
DMT_HOST_DEVICE inline void write_packed_sample(uint8_t* row,
                                                SizeType sample_idx,
                                                uint32_t val) noexcept {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4 || NBITS == 8 ||
                      NBITS == 16,
                  "write_packed_sample: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 16) {
        auto* p = row + (sample_idx * 2);
        p[0] = static_cast<uint8_t>(val & 0xFFU);
        p[1] = static_cast<uint8_t>((val >> 8U) & 0xFFU);
    } else if constexpr (NBITS == 8) {
        row[sample_idx] = static_cast<uint8_t>(val);
    } else {
        constexpr int kPerByte   = 8 / static_cast<int>(NBITS);
        constexpr uint32_t kMask = (1U << NBITS) - 1U;
        const auto byte_idx      = sample_idx / static_cast<SizeType>(kPerByte);
        const auto sub_idx       = sample_idx % static_cast<SizeType>(kPerByte);
        const auto shift         = static_cast<unsigned>(sub_idx) * NBITS;
        row[byte_idx] = static_cast<uint8_t>(
            (row[byte_idx] & ~(kMask << shift)) | ((val & kMask) << shift));
    }
}

/**
 * @brief Copy a slice of `count` packed samples from `src` to `dst`.
 * Handles sub-byte bit alignment for NBITS in {1, 2, 4} when offsets are not byte-aligned.
 */
template <unsigned NBITS>
inline void copy_packed_samples(const uint8_t* src, SizeType src_offset,
                                uint8_t* dst, SizeType dst_offset,
                                SizeType count) noexcept {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4 || NBITS == 8 ||
                      NBITS == 16,
                  "copy_packed_samples: NBITS must be 1, 2, 4, 8, or 16");
    if (count == 0) {
        return;
    }
    if constexpr (NBITS == 8) {
        std::copy_n(src + src_offset, count, dst + dst_offset);
    } else if constexpr (NBITS == 16) {
        std::copy_n(src + (src_offset * 2), count * 2, dst + (dst_offset * 2));
    } else {
        constexpr SizeType kPerByte = 8 / NBITS;
        if ((src_offset % kPerByte == 0) && (dst_offset % kPerByte == 0) &&
            (count % kPerByte == 0)) {
            std::copy_n(src + (src_offset / kPerByte), count / kPerByte,
                        dst + (dst_offset / kPerByte));
        } else {
            for (SizeType s = 0; s < count; ++s) {
                const auto val = read_packed_sample<NBITS>(src, src_offset + s);
                write_packed_sample<NBITS>(dst, dst_offset + s, val);
            }
        }
    }
}

/// @brief Bytes needed to store `nsamps` samples of `nbits` width,
/// LSB-first-packed for nbits < 8, byte/word-aligned otherwise.
constexpr SizeType packed_row_bytes(SizeType nsamps, SizeType nbits) noexcept {
    if (nbits < 8) {
        return ((nsamps * nbits) + 7) / 8;
    }
    return nsamps * (nbits / 8);
}

/// @brief Maximum value representable by an NBITS-wide unsigned sample.
constexpr uint32_t max_sample_value(SizeType nbits) noexcept {
    return (1U << nbits) - 1U;
}

} // namespace dmt::utils
