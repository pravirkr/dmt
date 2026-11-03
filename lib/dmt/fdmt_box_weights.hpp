#pragma once

#include <algorithm>
#include <cstdint>

#include "dmt/common/types.hpp"

// Level-0 box means of the FDMT (use_box_smearing = true), shared by the
// plan (noise variance, int_tree bounds), the CPU and GPU engines and
// FDMT-FFT so all of them use the same integers.
//
// A level-0 row of |dt| = s holds the box sum S of L = s + 1 samples. The
// matched filter for a channel smeared over L samples is the box *mean* S/L,
// which keeps every channel's contribution in proportion to its S/N (a box
// sum weights noisier, more smeared channels as heavily as clean ones).
//
// Float input: the mean is S * w(L), with the dyadic weight
//
//     w(L) = round(2^b(L) / L) * 2^-b(L),   b(L) = ceil(log2 L) + 2,
//
// within 1/8 of 1/L (under 0.15% of S/N against the exact mean). It depends
// on L alone, so a trial's output does not depend on the rest of its plan.
//
// Packed low-bit input (exact integers): the same value rounded (half up) to
// f(L) fractional bits, the smallest f with 3 * 4^f >= 50 L. The rounding
// noise, 4^-f / 12, is then at most 2% of the box-mean noise sigma^2 / L of
// any input with sigma >= 0.5 LSB (S/N loss <= 0.5%), while the stored value
// stays near maxval * 2^f(L) ~ 4 sqrt(L) maxval instead of the sum's
// maxval * L: narrow integer tree levels (int_tree). Every row is written on
// the plan's common scale 2^-F, F = f(L_max), so the merges stay plain
// additions; integer state holds exact integers in units of 2^-F, float
// state the same values times 2^-F. For packed input int_tree on and off
// therefore agree bit for bit, and differ from float input with the same
// values by the rounding (at most 2^-(f(L) + 1) per channel).

namespace dmt::algorithms::detail {

/// ceil(log2(dt0_max + 1)) + 2: the dyadic weight bits b(L) of a box of
/// L = dt0_max + 1 samples.
DMT_HD constexpr SizeType box_fraction_bits(SizeType dt0_max) noexcept {
    SizeType bits = 0;
    while ((SizeType{1} << bits) < dt0_max + 1) {
        ++bits;
    }
    return bits + 2;
}

/// Dyadic weight round(2^b(L) / L) of a box of 1 <= L samples, shifted to
/// the scale 2^-bits (bits >= b(L)); rounded half up (an exact half never
/// occurs).
DMT_HD constexpr uint32_t box_weight(SizeType box_len, SizeType bits) noexcept {
    const SizeType own = box_fraction_bits(box_len - 1);
    const auto weight  = ((SizeType{1} << own) + (box_len / 2)) / box_len;
    return static_cast<uint32_t>(weight << (bits - own));
}

/// Fractional bits f(L) of the rounded box mean of packed input: the
/// smallest f with 3 * 4^f >= 50 L (see above).
DMT_HD constexpr SizeType box_mean_bits(SizeType box_len) noexcept {
    SizeType bits = 0;
    while (3 * (SizeType{1} << (2 * bits)) < 50 * box_len) {
        ++bits;
    }
    return bits;
}

/**
 * @brief Integer recipe of one packed-input level-0 box on the common scale
 * 2^-F: value = ((S * mul + add) >> rshift) << lshift, i.e. the dyadic mean
 * S * w(L) rounded half up to f(L) bits (no rounding when b(L) <= f(L)).
 */
struct FDMTBoxRound {
    uint32_t mul;    ///< round(2^b(L) / L)
    uint32_t add;    ///< 2^(rshift - 1), or 0
    uint32_t rshift; ///< max(b(L) - f(L), 0)
    uint32_t lshift; ///< F - min(b(L), f(L))
};

/// The recipe of a box of 1 <= L samples on the scale 2^-F (F >= f(L)).
DMT_HD constexpr FDMTBoxRound box_round(SizeType box_len,
                                        SizeType scale_bits) noexcept {
    const SizeType b   = box_fraction_bits(box_len - 1);
    const SizeType f   = box_mean_bits(box_len);
    const SizeType mul = ((SizeType{1} << b) + (box_len / 2)) / box_len;
    if (b <= f) {
        return {
            .mul    = static_cast<uint32_t>(mul),
            .add    = 0U,
            .rshift = 0U,
            .lshift = static_cast<uint32_t>(scale_bits - b),
        };
    }
    const SizeType r = b - f;
    return {
        .mul    = static_cast<uint32_t>(mul),
        .add    = static_cast<uint32_t>(SizeType{1} << (r - 1)),
        .rshift = static_cast<uint32_t>(r),
        .lshift = static_cast<uint32_t>(scale_bits - f),
    };
}

/// Applies a box recipe to the box sum `sum`. P is the product type: it
/// must hold sum * mul + add (see box_round_max()).
template <typename P>
DMT_HD constexpr P box_round_apply(P sum, const FDMTBoxRound& r) noexcept {
    // Every step truncates to P, and the shift counts are masked to P's
    // width (a no-op: P holds the result, so both counts are smaller), so
    // the compiler may keep narrow P in narrow SIMD lanes; C++ promotes
    // uint8_t / uint16_t arithmetic to int, and an unmasked count could
    // exceed the narrow width.
    constexpr uint32_t kMask = (8 * sizeof(P)) - 1;
    auto v                   = static_cast<P>(sum * static_cast<P>(r.mul));
    v                        = static_cast<P>(v + static_cast<P>(r.add));
    v                        = static_cast<P>(v >> (r.rshift & kMask));
    return static_cast<P>(v << (r.lshift & kMask));
}

/// Largest value of the recipes of boxes of up to `box_max` samples <=
/// maxval, intermediate sum * mul + add or result: the accumulator /
/// product type must hold it.
constexpr SizeType box_round_max(SizeType maxval, SizeType box_max) noexcept {
    SizeType largest = 0;
    for (SizeType len = 1; len <= box_max; ++len) {
        const auto r     = box_round(len, box_mean_bits(box_max));
        const auto inter = (maxval * len * r.mul) + r.add;
        largest = std::max({largest, inter, box_round_apply(maxval * len, r)});
    }
    return largest;
}

/// Whether box_round_max() fits uint32_t (else the product needs 64 bits).
constexpr bool box_round_fits_u32(SizeType maxval, SizeType box_max) noexcept {
    return box_round_max(maxval, box_max) <= SizeType{0xFFFFFFFF};
}

} // namespace dmt::algorithms::detail
