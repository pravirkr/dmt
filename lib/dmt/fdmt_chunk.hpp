#pragma once

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/modes.hpp"

// Chunked execution of the FDMT (mode "valid"), shared by the CPU and GPU
// engines.
//
// Valid mode is overlap-save inside the tree: every merge keeps the last
// `delay` samples of its head input across blocks, so splitting a block of N
// samples into chunks of c gives bit-identical output for any c on the CPU
// tree, packed paths, integer-valued input and explicit fusion depths.
// Automatic GPU direct levels on float input agree within float rounding and
// are not bit-identical across chunks. The tree
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

/// True when `a * b` fits in SizeType. A wrapping product must not be treated
/// as a small state, which would accept an impossibly large chunk.
inline bool product_fits(SizeType a, SizeType b) noexcept {
    return b == 0 || a <= std::numeric_limits<SizeType>::max() / b;
}

/// Bytes of the two state halves for one sample, or SizeType's maximum when
/// that product does not fit (callers then refuse to grow the chunk).
inline SizeType state_bytes_per_sample(SizeType rows,
                                       SizeType elem_bytes,
                                       SizeType nbeams) noexcept {
    const SizeType beams = std::max<SizeType>(nbeams, 1);
    if (!product_fits(SizeType{2}, rows)) {
        return std::numeric_limits<SizeType>::max();
    }
    const SizeType two_rows = SizeType{2} * rows;
    if (!product_fits(two_rows, elem_bytes)) {
        return std::numeric_limits<SizeType>::max();
    }
    const SizeType two_row_bytes = two_rows * elem_bytes;
    if (!product_fits(two_row_bytes, beams)) {
        return std::numeric_limits<SizeType>::max();
    }
    return two_row_bytes * beams;
}

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
        if (requested >
            std::numeric_limits<SizeType>::max() - (kFDMTChunkAlign - 1)) {
            throw std::invalid_argument(
                "FDMT: chunk_nsamps is too large to align");
        }
        const SizeType aligned =
            ((requested + kFDMTChunkAlign - 1) / kFDMTChunkAlign) *
            kFDMTChunkAlign;
        return std::min(aligned, nsamps);
    }
    const SizeType per_sample =
        state_bytes_per_sample(fdmt_state_rows(plan), sizeof(float), nbeams);
    SizeType chunk = kFDMTMinChunk;
    while (chunk < nsamps && product_fits(chunk, 2)) {
        const SizeType next = chunk * 2;
        if (!product_fits(per_sample, next) ||
            per_sample * next > kFDMTChunkStateBudget) {
            break;
        }
        chunk = next;
    }
    return std::min(chunk, nsamps);
}

/// Shortest chunk fdmt_cache_chunk() considers: below it the per-chunk
/// launches outweigh the cache gain.
inline constexpr SizeType kFDMTMinCacheChunk = 256;

/**
 * @brief GPU automatic chunk: the longest power of two, at least
 * kFDMTMinCacheChunk, whose two state halves (`elem_bytes` per element) fit
 * `cache_bytes` (the device's L2), so each merge reads the level below from
 * L2 rather than DRAM; capped at nsamps. 0 when even the shortest does not
 * fit: the merges are DRAM-bound at any chunk and fdmt_chunk_nsamps()'s
 * budget rule applies. On the L40S (96 MiB L2) this is 1.4-1.8x faster than
 * the budget rule on the cryopaf presets.
 */
inline SizeType fdmt_cache_chunk(const plans::FDMTPlan& plan,
                                 SizeType nbeams,
                                 SizeType elem_bytes,
                                 SizeType cache_bytes) noexcept {
    const SizeType per_sample =
        state_bytes_per_sample(fdmt_state_rows(plan), elem_bytes, nbeams);
    if (!product_fits(per_sample, kFDMTMinCacheChunk) ||
        per_sample * kFDMTMinCacheChunk > cache_bytes) {
        return 0;
    }
    SizeType chunk = kFDMTMinCacheChunk;
    while (chunk < plan.get_nsamps() && product_fits(chunk, 2)) {
        const SizeType next = chunk * 2;
        if (!product_fits(per_sample, next) ||
            per_sample * next > cache_bytes) {
            break;
        }
        chunk = next;
    }
    return std::min(chunk, plan.get_nsamps());
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
