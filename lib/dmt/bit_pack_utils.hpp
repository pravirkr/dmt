#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "dmt/common/types.hpp"

namespace dmt::bit_pack_utils {

namespace detail {

/// Compile-time widths supported by the packed-sample helpers.
constexpr bool is_packed_nbits(unsigned nbits) noexcept {
    return nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8 || nbits == 16;
}

/** @brief One LUT row: unpacked lanes for a single packed byte (LSB-first). */
template <unsigned NBITS> struct UnpackLane {
    static constexpr unsigned kPerByte = 8U / NBITS;
    std::array<uint8_t, kPerByte> v;
};

/**
 * @brief 256-entry unpack LUT for sub-byte @c NBITS.
 *
 * The table blob is cache-line aligned.
 */
template <unsigned NBITS> struct UnpackTable {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4);

    static constexpr unsigned kPerByte = 8U / NBITS;
    static constexpr unsigned kMask    = (1U << NBITS) - 1U;

    struct alignas(64) Table {
        std::array<UnpackLane<NBITS>, 256> rows{};
    };

    static constexpr Table kTable = []() constexpr {
        Table t{};
        for (uint32_t byte = 0; byte < 256; ++byte) {
            for (uint32_t lane = 0; lane < kPerByte; ++lane) {
                t.rows[byte].v[lane] =
                    static_cast<uint8_t>((byte >> (lane * NBITS)) & kMask);
            }
        }
        return t;
    }();
};

} // namespace detail

/**
 * @brief Read one @c NBITS-wide unsigned sample from a packed channel row.
 * @param row Base of the row (@c packed_row_bytes(nsamps, NBITS) bytes).
 * @param sample_idx Sample index along time (0 .. nsamps-1).
 */
template <unsigned NBITS>
DMT_HD inline uint32_t read_packed_sample(const uint8_t* row,
                                          SizeType sample_idx) noexcept {
    static_assert(detail::is_packed_nbits(NBITS),
                  "read_packed_sample: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 16) {
        // Byte loads: a uint16_t load faults on the GPU when the row is not
        // 2-byte aligned (time-major samples are only byte-aligned).
        const SizeType byte = sample_idx * 2;
        return static_cast<uint32_t>(row[byte]) |
               (static_cast<uint32_t>(row[byte + 1]) << 8);
    } else if constexpr (NBITS == 8) {
        return row[sample_idx];
    } else {
        constexpr unsigned kPerByte = 8U / NBITS;
        constexpr uint32_t kMask    = (1U << NBITS) - 1U;
        const auto byte_idx = sample_idx / static_cast<SizeType>(kPerByte);
        const auto sub_idx  = sample_idx % static_cast<SizeType>(kPerByte);
        return (static_cast<uint32_t>(row[byte_idx]) >>
                (static_cast<unsigned>(sub_idx) * NBITS)) &
               kMask;
    }
}

/**
 * @brief Write one @c NBITS-wide sample into a packed channel row.
 * @param val Sample value; only the low @c NBITS bits are stored.
 */
template <unsigned NBITS>
DMT_HD inline void
write_packed_sample(uint8_t* row, SizeType sample_idx, uint32_t val) noexcept {
    static_assert(detail::is_packed_nbits(NBITS),
                  "write_packed_sample: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 16) {
        const SizeType byte = sample_idx * 2;
        row[byte]           = static_cast<uint8_t>(val & 0xFFU);
        row[byte + 1]       = static_cast<uint8_t>((val >> 8) & 0xFFU);
    } else if constexpr (NBITS == 8) {
        row[sample_idx] = static_cast<uint8_t>(val);
    } else {
        constexpr unsigned kPerByte = 8U / NBITS;
        constexpr uint32_t kMask    = (1U << NBITS) - 1U;
        const auto byte_idx = sample_idx / static_cast<SizeType>(kPerByte);
        const auto sub_idx  = sample_idx % static_cast<SizeType>(kPerByte);
        const auto shift    = static_cast<unsigned>(sub_idx) * NBITS;
        row[byte_idx]       = static_cast<uint8_t>(
            (row[byte_idx] & ~(kMask << shift)) | ((val & kMask) << shift));
    }
}

/**
 * @brief Copy @p count packed samples from @p src to @p dst at given offsets.
 *
 * For sub-byte widths, uses a byte @c memcpy when both offsets and @p count are
 * aligned to whole bytes; otherwise falls back to per-sample read/write.
 */
template <unsigned NBITS>
inline void copy_packed_samples(const uint8_t* src,
                                SizeType src_offset,
                                uint8_t* dst,
                                SizeType dst_offset,
                                SizeType count) noexcept {
    static_assert(detail::is_packed_nbits(NBITS),
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

/** @brief Byte length of one channel row holding @p nsamps samples at @p nbits.
 */
constexpr SizeType packed_row_bytes(SizeType nsamps, SizeType nbits) noexcept {
    if (nbits < 8) {
        return ((nsamps * nbits) + 7) / 8;
    }
    return nsamps * (nbits / 8);
}

/** @brief Largest unsigned value storable in @p nbits (@c (1&lt;&lt;nbits)-1).
 */
constexpr uint32_t max_sample_value(SizeType nbits) noexcept {
    return (1U << nbits) - 1U;
}

#ifndef __CUDA_ARCH__
namespace detail {

template <unsigned NBITS, typename T>
DMT_H void unpack_row_subbyte(const uint8_t* __restrict__ row,
                              SizeType nsamps,
                              T* __restrict__ out) noexcept {
    constexpr SizeType kPerByte = 8 / NBITS;
    const SizeType nfull        = nsamps / kPerByte;
    const SizeType tail_start   = nfull * kPerByte;

    if constexpr (std::is_same_v<T, uint8_t>) {
        const auto& lut_rows = UnpackTable<NBITS>::kTable.rows;
        for (SizeType b = 0; b < nfull; ++b) {
            std::memcpy(out + (b * kPerByte), lut_rows[row[b]].v.data(),
                        kPerByte);
        }
    } else {
        constexpr uint32_t kMask = (1U << NBITS) - 1U;
        for (SizeType b = 0; b < nfull; ++b) {
            const auto byte = static_cast<uint32_t>(row[b]);
#if defined(_OPENMP) && !defined(__CUDACC__)
#pragma omp simd
#endif
            for (SizeType k = 0; k < kPerByte; ++k) {
                out[(b * kPerByte) + k] = static_cast<T>(
                    (byte >> (static_cast<unsigned>(k) * NBITS)) & kMask);
            }
        }
    }
    for (SizeType i = tail_start; i < nsamps; ++i) {
        out[i] = static_cast<T>(read_packed_sample<NBITS>(row, i));
    }
}

} // namespace detail

/**
 * @brief Unpack the first @p nsamps samples of a packed row into @p out.
 *
 * Serial host implementation: FDMT level-0 already parallelizes over subbands
 * (@c omp for), so each thread unpacks its own row without nested parallelism.
 */
template <unsigned NBITS, typename T>
DMT_H inline void unpack_row(const uint8_t* __restrict__ row,
                             SizeType nsamps,
                             T* __restrict__ out) noexcept {
    static_assert(detail::is_packed_nbits(NBITS),
                  "unpack_row: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 8) {
#if defined(_OPENMP) && !defined(__CUDACC__)
#pragma omp simd
#endif
        for (SizeType i = 0; i < nsamps; ++i) {
            out[i] = static_cast<T>(row[i]);
        }
    } else if constexpr (NBITS == 16) {
        const auto* words = reinterpret_cast<const uint16_t*>(row);
#if defined(_OPENMP) && !defined(__CUDACC__)
#pragma omp simd
#endif
        for (SizeType i = 0; i < nsamps; ++i) {
            out[i] = static_cast<T>(words[i]);
        }
    } else {
        detail::unpack_row_subbyte<NBITS, T>(row, nsamps, out);
    }
}

/** @brief Runtime @p nbits dispatch for unpack_row (caller validates @p nbits).
 */
template <typename T>
DMT_H inline void unpack_row(const uint8_t* __restrict__ row,
                             SizeType nbits,
                             SizeType nsamps,
                             T* __restrict__ out) noexcept {
    switch (nbits) {
    case 1:
        unpack_row<1>(row, nsamps, out);
        break;
    case 2:
        unpack_row<2>(row, nsamps, out);
        break;
    case 4:
        unpack_row<4>(row, nsamps, out);
        break;
    case 8:
        unpack_row<8>(row, nsamps, out);
        break;
    case 16:
        unpack_row<16>(row, nsamps, out);
        break;
    default:
        break;
    }
}

#endif // !defined(__CUDA_ARCH__)

} // namespace dmt::bit_pack_utils
