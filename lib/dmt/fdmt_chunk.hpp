#pragma once

#include <algorithm>
#include <format>
#include <stdexcept>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/modes.hpp"

// Chunked execution of the FDMT (mode "valid"), shared by the CPU and GPU
// engines.
//
// Valid mode is overlap-save inside the tree: every merge keeps the last
// `delay` samples of its head input across blocks, so splitting a block of N
// samples into chunks of c gives bit-identical output for any c. The tree
// state of one chunk is rows x c instead of rows x N -- at high DM the plan
// has ~10^5 level-0 rows, and the state of a whole block would be tens of GB
// while the block's input and output are a few hundred MB. Each chunk costs
// an extra O(min(delay, c)) history traffic per merge (a ring; see
// offset_add), a few per cent at c of a few thousand samples.

namespace dmt::algorithms::detail {

/// Automatic chunk length (chunk_nsamps = 0): the longest power of two, at
/// least kFDMTMinChunk, whose two state halves fit kFDMTChunkStateBudget, a
/// rule on the plan alone (no hardware detection).
inline constexpr SizeType kFDMTChunkStateBudget = SizeType{2} << 30; // 2 GiB
inline constexpr SizeType kFDMTMinChunk         = 2048;
/// An explicit chunk is rounded up to a multiple of this (whole bytes of
/// packed input, whole SIMD vectors and GPU blocks).
inline constexpr SizeType kFDMTChunkAlign = 64;

/// Largest level row count below the root (rows of the two state halves).
inline SizeType fdmt_state_rows(const plans::FDMTPlan& plan) noexcept {
    const auto& shapes = plan.get_container().state_shape;
    SizeType rows      = 0;
    for (SizeType l = 0; l < plan.get_niters(); ++l) {
        rows = std::max(rows, shapes[l].ncoords);
    }
    return rows;
}

/**
 * @brief Row length an engine executes blocks of `plan` in: the block's own
 * nsamps unless `mode` is "valid", else `requested` (0 = automatic) rounded
 * up to kFDMTChunkAlign and capped at nsamps.
 */
inline SizeType fdmt_chunk_nsamps(const plans::FDMTPlan& plan,
                                  FDMTMode mode,
                                  SizeType nbeams,
                                  SizeType requested) {
    const SizeType nsamps = plan.get_nsamps();
    if (mode != FDMTMode::kValid) {
        if (requested != 0 && requested < nsamps) {
            throw std::invalid_argument(std::format(
                "FDMT: chunk_nsamps={} needs mode='valid' (mode '{}' runs "
                "whole blocks of nsamps={})",
                requested, fdmt_mode_to_string(mode), nsamps));
        }
        return nsamps;
    }
    if (requested != 0) {
        const SizeType aligned =
            ((requested + kFDMTChunkAlign - 1) / kFDMTChunkAlign) *
            kFDMTChunkAlign;
        return std::min(aligned, nsamps);
    }
    const SizeType per_sample = 2 * fdmt_state_rows(plan) * sizeof(float) *
                                std::max<SizeType>(nbeams, 1);
    SizeType chunk            = kFDMTMinChunk;
    while (chunk < nsamps &&
           per_sample * (2 * chunk) <= kFDMTChunkStateBudget) {
        chunk *= 2;
    }
    return std::min(chunk, nsamps);
}

/**
 * @brief Copy of a valid-mode plan's tables re-laid out for rows of
 * `row_len` samples below the root (the chunk length): every level < niters
 * gets state rows of row_len, and coordinate offsets / lengths follow. The
 * root keeps the plan's own layout, which is the dmt output's (row stride
 * nsamps), so the last merge writes the output directly. Used by the GPU
 * engine, which uploads the plan's tables as they are.
 */
inline plans::FDMTPlanContainer fdmt_relayout_container(
    const plans::FDMTPlanContainer& pc, SizeType niters, SizeType row_len) {
    plans::FDMTPlanContainer out = pc;
    const auto rescale           = [&](SizeType offset, SizeType l) {
        if (offset == SIZE_MAX || l >= niters) {
            return offset;
        }
        return (offset / pc.state_shape[l].nsamps) * row_len;
    };
    for (SizeType l = 0; l < niters; ++l) {
        out.state_shape[l].nsamps    = row_len;
        out.state_shape[l].nelements = out.state_shape[l].ncoords * row_len;
    }
    for (SizeType l = 0; l <= niters; ++l) {
        for (auto* coords : {&out.coordinates[l], &out.coordinates_sum[l],
                             &out.coordinates_copy[l]}) {
            for (auto& c : *coords) {
                c.buf_offset = rescale(c.buf_offset, l);
                if (l < niters) {
                    c.nsamps = row_len;
                }
                if (l >= 1) {
                    c.tail_buf_offset = rescale(c.tail_buf_offset, l - 1);
                    c.head_buf_offset = rescale(c.head_buf_offset, l - 1);
                    if (c.tail_nsamps != SIZE_MAX) {
                        c.tail_nsamps = row_len;
                    }
                    if (c.head_nsamps != SIZE_MAX) {
                        c.head_nsamps = row_len;
                    }
                }
            }
        }
    }
    return out;
}

} // namespace dmt::algorithms::detail
