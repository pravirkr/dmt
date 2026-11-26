#pragma once

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/dm_utils.hpp"

namespace dmt::algorithms::detail {

/**
 * @brief Header of a saved plain-tree FDMT history (save_history() off the
 * skewed-window paths), the same on the CPU and the GPU.
 *
 * kFDMTHistHeader float words, each holding a 16-bit piece exactly, as the
 * skewed-window histories do: word 0 the magic ("TR"), word 1 the version,
 * words 2-5 the layout hash (fdmt_tree_history_hash()), words 6-9 the stream
 * position (samples the stream has seen since its reset_history(), carried
 * across loads; informational), words 10-15 zero (reserved). The level-0
 * input history and the tree history (every merge's slot, oldest sample
 * first) follow.
 */
inline constexpr SizeType kFDMTHistHeader      = 16;
inline constexpr uint64_t kFDMTTreeHistMagic   = 0x5452U; // "TR"
inline constexpr uint64_t kFDMTTreeHistVersion = 1;
/// Magic of the skewed-window histories (fdmt_skew.hpp, CUDA skewed path).
inline constexpr uint64_t kFDMTSkewHistMagic = 0x534BU; // "SK"

/// Encodes `words` 16-bit chunks of `v` into `header[at .. at + words)` as
/// exact float32 integers in [0, 65535].
inline void put16(std::span<float> header,
                  SizeType at,
                  uint64_t v,
                  SizeType words) noexcept {
    for (SizeType i = 0; i < words; ++i) {
        header[at + i] = static_cast<float>((v >> (16 * i)) & 0xFFFFU);
    }
}

/// Decodes `words` 16-bit chunks from `header[at .. at + words)`. Each word
/// must be a finite exact integer in [0, 65535]; throws std::invalid_argument
/// with @p error_msg otherwise. Tests utils::is_finite_bits() first because
/// Release builds compile with -ffast-math.
[[nodiscard]] inline uint64_t get16(std::span<const float> header,
                                    SizeType at,
                                    SizeType words,
                                    const char* error_msg) {
    uint64_t v = 0;
    for (SizeType i = 0; i < words; ++i) {
        const float x = header[at + i];
        if (!utils::is_finite_bits(x) || x < 0.0F || x > 65535.0F ||
            x != std::floor(x)) {
            throw std::invalid_argument(error_msg);
        }
        v |= static_cast<uint64_t>(x) << (16 * i);
    }
    return v;
}

/**
 * @brief Identifies the plain-tree history layout: the beams, box smearing
 * (it scales every saved tree value) and, from the plan, the level-0 and
 * tree history sizes and every merge that keeps a history slot (level,
 * subband, delay trial, head row, delay and slot). Plans with the same
 * hash save the same quantities at the same places; chunk length, fusion
 * depth, threads and backend do not enter.
 */
[[nodiscard]] uint64_t fdmt_tree_history_hash(const plans::FDMTPlan& plan,
                                              SizeType nbeams,
                                              bool use_box_smearing);

/// Writes the header into header[0 .. kFDMTHistHeader).
void write_fdmt_tree_history_header(std::span<float> header,
                                    uint64_t layout_hash,
                                    uint64_t stream_pos) noexcept;

/**
 * @brief Checks a saved header against this engine's layout hash and
 * returns the stream position it holds.
 * @throws std::invalid_argument naming the mismatch: no header (not a
 * history, or one written before dmt 0.8), a skewed-window history, an
 * unknown version, or a different configuration.
 */
[[nodiscard]] uint64_t
check_fdmt_tree_history_header(std::span<const float> header,
                               uint64_t layout_hash);

/// The load_history() size error, with a hint for headerless buffers.
[[noreturn]] void throw_fdmt_history_size(SizeType expected, SizeType got);

} // namespace dmt::algorithms::detail
