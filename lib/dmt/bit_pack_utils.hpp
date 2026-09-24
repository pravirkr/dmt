#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

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

// Host-only below: bulk row unpacking (lookup tables, std::memcpy) is never
// called from device code, so it is kept out of the CUDA device pass.
#if !defined(__CUDA_ARCH__)
namespace detail {

struct alignas(8) Byte8 {
    uint8_t v[8];
};
struct alignas(4) Byte4 {
    uint8_t v[4];
};
struct alignas(2) Byte2 {
    uint8_t v[2];
};

template <unsigned NBITS> struct UnpackTable;

template <> struct UnpackTable<1> {
    static constexpr auto table = []() constexpr {
        std::array<Byte8, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            for (uint32_t k = 0; k < 8; ++k) {
                t[i].v[k] = static_cast<uint8_t>((i >> k) & 1U);
            }
        }
        return t;
    }();
};

template <> struct UnpackTable<2> {
    static constexpr auto table = []() constexpr {
        std::array<Byte4, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            for (uint32_t k = 0; k < 4; ++k) {
                t[i].v[k] = static_cast<uint8_t>((i >> (k * 2U)) & 3U);
            }
        }
        return t;
    }();
};

template <> struct UnpackTable<4> {
    static constexpr auto table = []() constexpr {
        std::array<Byte2, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            for (uint32_t k = 0; k < 2; ++k) {
                t[i].v[k] = static_cast<uint8_t>((i >> (k * 4U)) & 15U);
            }
        }
        return t;
    }();
};

} // namespace detail

/**
 * @brief Unpack the first `nsamps` NBITS-wide samples of a packed row into
 * `out`, converting each to `T`. Same packing convention as
 * read_packed_sample(); sub-byte widths are unpacked a whole byte at a time
 * (a 256-entry lookup table for uint8_t output, constant shift/mask lanes
 * otherwise).
 */
template <unsigned NBITS, typename T>
inline void unpack_row(const uint8_t* __restrict__ row,
                       SizeType nsamps,
                       T* __restrict__ out) noexcept {
    static_assert(NBITS == 1 || NBITS == 2 || NBITS == 4 || NBITS == 8 ||
                      NBITS == 16,
                  "unpack_row: NBITS must be 1, 2, 4, 8, or 16");
    if constexpr (NBITS == 8) {
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (SizeType i = 0; i < nsamps; ++i) {
            out[i] = static_cast<T>(row[i]);
        }
    } else if constexpr (NBITS == 16) {
#if defined(_OPENMP)
#pragma omp simd
#endif
        for (SizeType i = 0; i < nsamps; ++i) {
            out[i] =
                static_cast<T>(static_cast<uint32_t>(row[2 * i]) |
                               (static_cast<uint32_t>(row[(2 * i) + 1]) << 8U));
        }
    } else {
        constexpr SizeType kPerByte = 8 / NBITS;
        constexpr uint32_t kMask    = (1U << NBITS) - 1U;
        const SizeType nfull        = nsamps / kPerByte;
        if constexpr (std::is_same_v<T, uint8_t>) {
            // Byte output: one table lookup yields all kPerByte samples.
            const auto& table = detail::UnpackTable<NBITS>::table;
            for (SizeType b = 0; b < nfull; ++b) {
                std::memcpy(out + (b * kPerByte), table[row[b]].v, kPerByte);
            }
        } else {
            // Wider output: a constant-shift mask per lane vectorizes well
            // (measured faster than the table for float output).
            for (SizeType b = 0; b < nfull; ++b) {
                const auto byte = static_cast<uint32_t>(row[b]);
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (SizeType k = 0; k < kPerByte; ++k) {
                    out[(b * kPerByte) + k] = static_cast<T>(
                        (byte >> (static_cast<unsigned>(k) * NBITS)) & kMask);
                }
            }
        }
        for (SizeType i = nfull * kPerByte; i < nsamps; ++i) {
            out[i] = static_cast<T>(read_packed_sample<NBITS>(row, i));
        }
    }
}

/// @brief Runtime-nbits dispatch of unpack_row<NBITS>(); `nbits` must be one
/// of 1, 2, 4, 8, 16 (validated by the caller).
template <typename T>
inline void unpack_row(const uint8_t* __restrict__ row,
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

} // namespace dmt::utils
