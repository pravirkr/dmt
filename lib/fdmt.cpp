#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/omp_helper.hpp"
#include "dmt/simd_intrinsics.hpp"

namespace dmt::algorithms {

namespace {

enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

/**
 * @brief Element-wise converting copy (`std::copy_n` when the types match).
 */
template <typename TIn, typename TOut>
inline void fdmt_convert_copy(const TIn* __restrict__ src,
                              SizeType n,
                              TOut* __restrict__ dst) noexcept {
    if constexpr (std::is_same_v<TIn, TOut>) {
        std::copy_n(src, n, dst);
    } else {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) {
            dst[i] = static_cast<TOut>(src[i]);
        }
    }
}

/**
 * @brief The single FDMT merge addition, performed in the output storage
 * type. For TOut = float this is exactly `a + b`; for the narrow integer
 * types the plan-derived value bound guarantees no wrap-around.
 */
template <typename TOut, typename TA, typename TB>
inline TOut fdmt_add(TA a, TB b) noexcept {
    return static_cast<TOut>(static_cast<TOut>(a) + static_cast<TOut>(b));
}

/**
 * @brief Advances a size-`capacity` FIFO history window (holding the most
 * recent `capacity` samples of some stream) by `nsamps` new samples from
 * `new_data`, in place.
 *
 * When `nsamps >= capacity`, one block fully covers the window, so it's
 * simply replaced outright with the trailing `capacity` samples of
 * `new_data` (this is the only case supported before this generalization,
 * and this branch reproduces that exact formula bit-for-bit). When `nsamps <
 * capacity` (block smaller than the window), the oldest `nsamps` retained
 * samples are shifted out and this whole block is appended. Shared by both
 * the tree-level history in `offset_add<kValid>` and the level-0 per-channel
 * history in `fdmt_init_valid_update_history`, so the two mechanisms can't
 * drift apart. History is always stored as float, whatever the state type
 * (`TIn`) of the stream feeding it.
 */
template <typename TIn = float>
void fdmt_advance_history_window(const TIn* __restrict__ new_data,
                                 float* __restrict__ hist,
                                 SizeType nsamps,
                                 SizeType capacity) noexcept {
    if (capacity == 0) {
        return;
    }
    if (nsamps >= capacity) {
        fdmt_convert_copy(new_data + nsamps - capacity, capacity, hist);
    } else {
        std::copy(hist + nsamps, hist + capacity, hist);
        fdmt_convert_copy(new_data, nsamps, hist + capacity - nsamps);
    }
}

/**
 * @brief Writes the merge's overlap region out[i] = tail[i] + head[i] with
 * non-temporal stores when requested and supported (float -> float on x86
 * AVX2/AVX-512; see FDMTExecConfig::streaming_stores) and returns true;
 * otherwise returns false and the caller runs its ordinary loop. On builds
 * without streaming stores this folds to `return false`, leaving the
 * original loop as the only code path.
 */
template <typename TIn, typename TOut>
inline bool fdmt_try_stream_add(const TIn* __restrict__ tail,
                                const TIn* __restrict__ head,
                                TOut* __restrict__ out,
                                SizeType n,
                                bool stream_out) noexcept {
    if constexpr (simd::kHasStreamingStores && std::is_same_v<TIn, float> &&
                  std::is_same_v<TOut, float>) {
        if (stream_out) {
            simd::add_stream_f32(tail, head, out, n);
            return true;
        }
    }
    return false;
}

/**
 * @brief Performs offset addition of two input arrays with mode-specific
 * boundary handling. This is the innermost hot loop of the FDMT algorithm.
 *
 * Core operation: out[k] = tail[k] + head[k - delay_shift]
 *
 * The function always performs addition in the overlap region
 * [delay_shift, size_tail), then handles edges according to the mode:
 *
 * - Full: Extends output beyond input size, zero-filling if needed
 * - Valid: Only valid overlap region (same size in/out). If `hist` is
 *   given, the boundary region [0, delay_shift) reads the previous block's
 *   trailing `head` samples instead of zero-filling, and the current
 *   block's own trailing `delay_shift` samples of `head` are captured into
 *   `hist` for the next call -- this is what makes mode="valid" streaming
 *   bit-exact across block boundaries for every trial, not just dt=0 (see
 *   FDMTCoord::hist_offset and FDMTCPU::reset_history()). Each coordinate
 *   owns a disjoint slice of the shared history buffer, so this is safe
 *   across the `#pragma omp for` in fdmt_iter below with no extra locking.
 * - Roll: Cyclic boundary - head wraps around using modulo arithmetic
 *
 * @tparam Mode FDMTMode enum specifying boundary handling behavior
 * @tparam TIn Storage type of the input (previous level) state.
 * @tparam TOut Storage type of the output (current level) state.
 *
 * @param data_tail Input array (tail component) - lower frequency band
 * @param size_tail Size of tail array (must equal size_head)
 * @param data_head Input array (head component) - higher frequency band
 * @param size_head Size of head array (must equal size_tail)
 * @param out Output array receiving the sum
 * @param size_out Size of output array (mode-dependent constraints)
 * @param delay_shift Delay offset in samples (dispersion-induced shift)
 * @param hist Valid-mode-only cross-block history slice for this
 * coordinate (size >= delay_shift); nullptr for kFull/kRoll or when no
 * history is needed (delay_shift == 0).
 * @param stream_out Write the overlap region with non-temporal stores
 * (float -> float only; see FDMTExecConfig::streaming_stores).
 *
 */
template <FDMTMode Mode, typename TIn = float, typename TOut = float>
void offset_add(const TIn* __restrict__ data_tail,
                SizeType size_tail,
                const TIn* __restrict__ data_head,
                SizeType size_head,
                TOut* __restrict__ out,
                SizeType size_out,
                SizeType delay_shift,
                float* __restrict__ hist = nullptr,
                bool stream_out          = false) noexcept {
    assert(size_tail == size_head && "Input tail and head sizes must be equal");
    assert(size_out >= size_tail && "Output size must be >= Input tail size");
    // kValid supports delay_shift >= size_tail (block size smaller than this
    // coordinate's dispersion delay) via the cross-block history FIFO in
    // `hist`; kFull/kRoll have no history mechanism and always require the
    // delay to fit within one block.
    assert((Mode == FDMTMode::kValid || delay_shift < size_tail) &&
           "Offset must be < input tail size");

    if constexpr (Mode == FDMTMode::kFull) {
        // Full mode: Output can extend beyond input size
        // Part 1: Copy tail-only region [0, delay_shift)
        fdmt_convert_copy(data_tail, delay_shift, out);

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
        if (fdmt_try_stream_add(data_tail + delay_shift, data_head,
                                out + delay_shift, nsum, stream_out)) {
            // written with non-temporal stores
        } else {
#pragma omp simd
            for (SizeType i = 0; i < nsum; ++i) {
                out[delay_shift + i] =
                    fdmt_add<TOut>(data_tail[delay_shift + i], data_head[i]);
            }
        }

        // Part 3: Head-only region [size_tail, size_tail + nrest)
        const SizeType nrest = std::min(delay_shift, size_out - size_tail);
        if (nrest > 0) {
            fdmt_convert_copy(data_head + nsum, nrest, out + size_tail);
        }
        // Part 4: Zero-fill any remaining [size_tail + nrest, size_out)
        const SizeType filled = size_tail + nrest;
        if (filled < size_out) {
            std::fill(out + filled, out + size_out, TOut{0});
        }
    } else if constexpr (Mode == FDMTMode::kValid) {
        // Valid mode: only overlap region
        assert(size_out == size_tail &&
               "All sizes must be equal in valid mode");
        // Part 1: Boundary region [0, min(delay_shift, size_tail)). When
        // delay_shift >= size_tail (this coordinate's delay spans more than
        // one block), the whole output block comes from history + tail, and
        // Part 2 below contributes nothing this call.
        if (hist != nullptr && delay_shift > 0) {
            const SizeType boundary_len = std::min(delay_shift, size_tail);
            for (SizeType i = 0; i < boundary_len; ++i) {
                out[i] = fdmt_add<TOut>(data_tail[i], hist[i]);
            }
            // Update the size-delay_shift history FIFO with this block's own
            // head samples, so the next call sees a correctly-advanced
            // window regardless of how block size compares to delay_shift.
            fdmt_advance_history_window(data_head, hist, size_head,
                                        delay_shift);
        } else {
            fdmt_convert_copy(data_tail, delay_shift, out);
        }

        // Part 2: Overlap region [delay_shift, size_tail), empty once
        // delay_shift >= size_tail.
        const SizeType nsum =
            (delay_shift < size_tail) ? (size_tail - delay_shift) : 0;
        if (fdmt_try_stream_add(data_tail + delay_shift, data_head,
                                out + delay_shift, nsum, stream_out)) {
            // written with non-temporal stores
        } else {
#pragma omp simd
            for (SizeType i = 0; i < nsum; ++i) {
                out[delay_shift + i] =
                    fdmt_add<TOut>(data_tail[delay_shift + i], data_head[i]);
            }
        }
    } else if constexpr (Mode == FDMTMode::kRoll) {
        // Roll mode: cyclic addition
        assert(size_out == size_tail && "All sizes must be equal in roll mode");
        // Part 1: Wrapped region [0, delay_shift)
        for (SizeType k = 0; k < delay_shift; ++k) {
            out[k] = fdmt_add<TOut>(data_tail[k],
                                    data_head[size_tail - delay_shift + k]);
        }

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
        if (fdmt_try_stream_add(data_tail + delay_shift, data_head,
                                out + delay_shift, nsum, stream_out)) {
            // written with non-temporal stores
        } else {
#pragma omp simd
            for (SizeType i = 0; i < nsum; ++i) {
                out[delay_shift + i] =
                    fdmt_add<TOut>(data_tail[delay_shift + i], data_head[i]);
            }
        }
    }
}

/**
 * @brief Tile-restricted `offset_add`: writes only out[t] for t in
 * [t_begin, t_end), producing exactly the values the full-row
 * `offset_add<Mode>` would write there (each element is the same single
 * addition, so tiling is bit-exact by construction).
 *
 * Unlike `offset_add<kValid>`, this never advances the valid-mode history
 * FIFO: `hist` is only read, and the caller must advance it with
 * `fdmt_advance_history_window` once every tile of the coordinate is done.
 */
template <FDMTMode Mode, typename TIn, typename TOut>
void offset_add_range(const TIn* __restrict__ data_tail,
                      SizeType size_tail,
                      const TIn* __restrict__ data_head,
                      TOut* __restrict__ out,
                      SizeType size_out,
                      SizeType delay_shift,
                      const float* __restrict__ hist,
                      SizeType t_begin,
                      SizeType t_end) noexcept {
    t_end = std::min(t_end, size_out);
    // Intersect [lo, hi) with the tile; returns an empty range if disjoint.
    const auto clip = [t_begin, t_end](SizeType lo, SizeType hi) {
        const SizeType a = std::max(lo, t_begin);
        const SizeType b = std::min(hi, t_end);
        return std::pair<SizeType, SizeType>{a, std::max(a, b)};
    };
    const SizeType d = delay_shift;

    // Overlap region [d, size_tail): out[t] = tail[t] + head[t - d]
    const auto overlap = [&]() {
        const auto rg    = clip(std::min(d, size_tail), size_tail);
        const SizeType a = rg.first;
        const SizeType b = rg.second;
#pragma omp simd
        for (SizeType t = a; t < b; ++t) {
            out[t] = fdmt_add<TOut>(data_tail[t], data_head[t - d]);
        }
    };

    if constexpr (Mode == FDMTMode::kFull) {
        {
            const auto rg    = clip(0, d);
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            fdmt_convert_copy(data_tail + a, b - a, out + a);
        }
        overlap();
        const SizeType nrest  = std::min(d, size_out - size_tail);
        const SizeType filled = size_tail + nrest;
        {
            // Head-only region: out[t] = head[t - d]
            const auto rg    = clip(size_tail, filled);
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            fdmt_convert_copy(data_head + (a - d), b - a, out + a);
        }
        {
            const auto rg    = clip(filled, size_out);
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            std::fill(out + a, out + b, TOut{0});
        }
    } else if constexpr (Mode == FDMTMode::kValid) {
        if (hist != nullptr && d > 0) {
            const auto rg    = clip(0, std::min(d, size_tail));
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            for (SizeType t = a; t < b; ++t) {
                out[t] = fdmt_add<TOut>(data_tail[t], hist[t]);
            }
        } else {
            const auto rg    = clip(0, std::min(d, size_tail));
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            fdmt_convert_copy(data_tail + a, b - a, out + a);
        }
        overlap();
    } else if constexpr (Mode == FDMTMode::kRoll) {
        {
            const auto rg    = clip(0, d);
            const SizeType a = rg.first;
            const SizeType b = rg.second;
            for (SizeType t = a; t < b; ++t) {
                out[t] =
                    fdmt_add<TOut>(data_tail[t], data_head[size_tail - d + t]);
            }
        }
        overlap();
    }
}

/// @brief Copy coordinates (odd-channel padding): out = tail, zero-extended.
template <typename TIn, typename TOut>
void fdmt_copy_coords(const TIn* __restrict__ state_in,
                      TOut* __restrict__ state_out,
                      const plans::FDMTCoord* __restrict__ coords_copy_cur,
                      SizeType ncoords_copy_cur) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
    for (SizeType i_coord = 0; i_coord < ncoords_copy_cur; ++i_coord) {
        const auto* __restrict__ coord = &coords_copy_cur[i_coord];
        const TIn* __restrict__ tail   = &state_in[coord->tail_buf_offset];
        TOut* __restrict__ out         = &state_out[coord->buf_offset];
        fdmt_convert_copy(tail, coord->tail_nsamps, out);
        if (coord->nsamps > coord->tail_nsamps) {
            std::fill(out + coord->tail_nsamps, out + coord->nsamps, TOut{0});
        }
    }
}

template <FDMTMode Mode, typename TIn = float, typename TOut = float>
void fdmt_iter(const TIn* __restrict__ state_in,
               TOut* __restrict__ state_out,
               float* __restrict__ hist_ptr,
               const plans::FDMTCoord* __restrict__ coords_sum_cur,
               const plans::FDMTCoord* __restrict__ coords_copy_cur,
               SizeType ncoords_sum_cur,
               SizeType ncoords_copy_cur,
               bool stream_out = false) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, hist_ptr, coords_sum_cur, coords_copy_cur,     \
               ncoords_sum_cur, ncoords_copy_cur, stream_out)
#endif
    {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for nowait
#endif
        for (SizeType i_coord = 0; i_coord < ncoords_sum_cur; ++i_coord) {
            const auto* __restrict__ coord = &coords_sum_cur[i_coord];
            const TIn* __restrict__ tail   = &state_in[coord->tail_buf_offset];
            const TIn* __restrict__ head   = &state_in[coord->head_buf_offset];
            TOut* __restrict__ out         = &state_out[coord->buf_offset];
            float* __restrict__ hist = (hist_ptr != nullptr && coord->delay > 0)
                                           ? &hist_ptr[coord->hist_offset]
                                           : nullptr;
            offset_add<Mode>(tail, coord->tail_nsamps, head, coord->head_nsamps,
                             out, coord->nsamps, coord->delay, hist,
                             stream_out);
        }
        fdmt_copy_coords(state_in, state_out, coords_copy_cur,
                         ncoords_copy_cur);
    }
}

/// @brief A run of consecutive sum-coordinates [begin, end) of one output
/// sub-band, processed together by the kTiled schedule.
struct FDMTChunk {
    SizeType begin;
    SizeType end;
};

/**
 * @brief Cache-blocked (FDMTSchedule::kTiled) analogue of `fdmt_iter`.
 *
 * Work items are (chunk, time tile) pairs, chunk-major, so a static
 * schedule gives each thread consecutive tiles of the same chunk. Within an
 * item every coordinate of the chunk is merged over the same tile; the
 * chunk's distinct tail/head input tiles (consecutive dt trials share one
 * of their operands about half the time) are therefore loaded once and
 * reused from cache by every output that reads them, instead of each
 * full-row merge evicting the rows its neighbour is about to reuse.
 *
 * Valid-mode history is only read in the tile loop; the FIFO advance is a
 * separate pass after the tile loop's implicit barrier, so no tile can
 * observe an already-advanced history window.
 */
template <FDMTMode Mode, typename TIn, typename TOut>
void fdmt_iter_tiled(const TIn* __restrict__ state_in,
                     TOut* __restrict__ state_out,
                     float* __restrict__ hist_ptr,
                     const plans::FDMTCoord* __restrict__ coords_sum_cur,
                     const plans::FDMTCoord* __restrict__ coords_copy_cur,
                     SizeType ncoords_sum_cur,
                     SizeType ncoords_copy_cur,
                     const FDMTChunk* __restrict__ chunks,
                     SizeType nchunks,
                     SizeType level_nsamps,
                     SizeType tile_nsamps) noexcept {
    const SizeType ntiles = (level_nsamps + tile_nsamps - 1) / tile_nsamps;
    const SizeType nitems = nchunks * ntiles;
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, hist_ptr, coords_sum_cur, coords_copy_cur,     \
               ncoords_sum_cur, ncoords_copy_cur, chunks, tile_nsamps, ntiles, \
               nitems)
#endif
    {
        fdmt_copy_coords(state_in, state_out, coords_copy_cur,
                         ncoords_copy_cur);
#ifdef DMT_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
        for (SizeType item = 0; item < nitems; ++item) {
            const auto& chunk      = chunks[item / ntiles];
            const SizeType t_begin = (item % ntiles) * tile_nsamps;
            const SizeType t_end   = t_begin + tile_nsamps;
            for (SizeType i_coord = chunk.begin; i_coord < chunk.end;
                 ++i_coord) {
                const auto* __restrict__ coord = &coords_sum_cur[i_coord];
                const float* __restrict__ hist =
                    (hist_ptr != nullptr && coord->delay > 0)
                        ? &hist_ptr[coord->hist_offset]
                        : nullptr;
                offset_add_range<Mode>(
                    &state_in[coord->tail_buf_offset], coord->tail_nsamps,
                    &state_in[coord->head_buf_offset],
                    &state_out[coord->buf_offset], coord->nsamps, coord->delay,
                    hist, t_begin, t_end);
            }
        }
        if constexpr (Mode == FDMTMode::kValid) {
            if (hist_ptr != nullptr) {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
                for (SizeType i_coord = 0; i_coord < ncoords_sum_cur;
                     ++i_coord) {
                    const auto* __restrict__ coord = &coords_sum_cur[i_coord];
                    if (coord->delay > 0) {
                        fdmt_advance_history_window(
                            &state_in[coord->head_buf_offset],
                            &hist_ptr[coord->hist_offset], coord->head_nsamps,
                            coord->delay);
                    }
                }
            }
        }
    }
}

/// @brief Level-0 box-sum extension `prev + sample`, stored as TOut. Integer
/// state with integer samples (packed input, FDMTExecConfig::int_tree) adds
/// in integer lanes; the plan-derived bound keeps it from wrapping. Anything
/// else adds in float, the original arithmetic.
template <typename TOut, typename TS>
inline TOut fdmt_init_add(TOut prev, TS sample) noexcept {
    if constexpr (std::is_integral_v<TOut> && std::is_integral_v<TS>) {
        return static_cast<TOut>(prev + sample);
    } else {
        return static_cast<TOut>(static_cast<float>(prev) +
                                 static_cast<float>(sample));
    }
}

/// @brief No-smearing level-0 row: the waterfall shifted by `shift` samples
/// (wrapped for kRoll, zero-filled otherwise).
template <bool UseRoll, typename TOut = float, typename TS = float>
void fdmt_init_impl_row_shift(const TS* __restrict__ wf_sub,
                              TOut* __restrict__ buf_row,
                              SizeType shift,
                              SizeType nsamps) noexcept {
    if constexpr (UseRoll) {
        for (SizeType isamp = 0; isamp < shift; ++isamp) {
            buf_row[isamp] = static_cast<TOut>(wf_sub[nsamps - shift + isamp]);
        }
    } else {
        std::fill(buf_row, buf_row + shift, TOut{0});
    }
    for (SizeType isamp = shift; isamp < nsamps; ++isamp) {
        buf_row[isamp] = static_cast<TOut>(wf_sub[isamp - shift]);
    }
}

template <bool UseRoll,
          bool UseBoxSmearing,
          typename TOut = float,
          typename TS   = float>
void fdmt_init_impl_row(const TS* __restrict__ wf_sub,
                        const TOut* __restrict__ buf_prev,
                        TOut* __restrict__ buf_cur,
                        SizeType dt_cur,
                        SizeType nsamps) noexcept {
    if constexpr (UseBoxSmearing) {
        // Extend box sum: new_sum = prev_sum + waterfall[isamp - dt_cur]
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = fdmt_init_add<TOut>(
                    buf_prev[isamp], wf_sub[nsamps - dt_cur + isamp]);
            }
        } else {
            // No new sample available, just copy previous partial sum
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = buf_prev[isamp];
            }
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] =
                fdmt_init_add<TOut>(buf_prev[isamp], wf_sub[isamp - dt_cur]);
        }
    } else {
        // No smearing: just shift by dt_cur
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] =
                    static_cast<TOut>(wf_sub[nsamps - dt_cur + isamp]);
            }
        } else {
            std::fill(buf_cur, buf_cur + dt_cur, TOut{0});
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = static_cast<TOut>(wf_sub[isamp - dt_cur]);
        }
    }
}

/**
 * @brief First level-0 box row out[i] = sum_{k=0..s0} x[i - k], where
 * `x[j < 0]` is 0 (kFull), the wrapped waterfall (kRoll) or the cross-block
 * history (kValid).
 *
 * Evaluated as s0 + 1 vectorizable shifted adds (x[i], then + x[i-1], ...),
 * the same summation order as the CUDA init kernel, so CPU and GPU level-0
 * rows agree. This replaced a serial running sum (`+= x[i]; -= x[i-w]`)
 * whose loop-carried float dependency dominated FDMT runtime with box
 * smearing and accumulated rounding error along the block (it did not even
 * reproduce a width-1 box exactly: (x0 + x1) - x0 != x1). Every output here
 * is a sum of at most s0 + 1 samples, so its error is bounded per sample.
 * Integer-valued input (packed low-bit) is exact; with integer TOut the sums
 * run directly in uint8/uint16 lanes.
 */
template <FDMTMode Mode, typename TOut, typename TS>
void fdmt_init_row0_box(const TS* __restrict__ wf_sub,
                        const float* __restrict__ hist_sub,
                        TOut* __restrict__ out,
                        SizeType s0,
                        SizeType dt_max_final,
                        SizeType nsamps) noexcept {
    fdmt_convert_copy(wf_sub, nsamps, out);
    for (SizeType s = 1; s <= s0; ++s) {
        const SizeType nb = std::min(s, nsamps);
        if constexpr (Mode == FDMTMode::kRoll) {
            for (SizeType i = 0; i < nb; ++i) {
                out[i] = fdmt_init_add<TOut>(out[i], wf_sub[nsamps - s + i]);
            }
        } else if constexpr (Mode == FDMTMode::kValid) {
            for (SizeType i = 0; i < nb; ++i) {
                out[i] =
                    fdmt_init_add<TOut>(out[i], hist_sub[dt_max_final - s + i]);
            }
        }
#pragma omp simd
        for (SizeType i = s; i < nsamps; ++i) {
            out[i] = fdmt_init_add<TOut>(out[i], wf_sub[i - s]);
        }
    }
}

/**
 * @brief Initializes level-0 rows for one sub-band's local (possibly signed)
 * dt grid.
 *
 * `dt_grid_sub` is guaranteed dense, contiguous, and sorted ascending (built
 * by `make_plan_iter0`), but its values may be negative when the plan's
 * overall dt range is negative or straddles zero.
 *
 * Level 0 has no notion of merge *direction* — that is resolved entirely by
 * the sign-aware tail/head reference swap in `make_plan`, once per merge,
 * above this level (see the doc comment on `make_plan`'s negative-dt branch).
 * A level-0 row only ever represents "this single native channel's own
 * delay/smearing width of |dt| samples": rows +s and -s are always
 * identical, since a single leaf channel has no other channel to be
 * earlier/later than. This lets one magnitude-indexed algorithm cover both
 * the non-negative and mixed-sign/negative cases below, and is why every
 * lookup here uses `std::abs(dt)` rather than the signed value directly.
 */
template <bool UseRoll,
          bool UseBoxSmearing,
          typename TOut = float,
          typename TS   = float>
void fdmt_init_subband(const TS* __restrict__ wf_sub,
                       TOut* __restrict__ buf_base,
                       const std::vector<IndexType>& dt_grid_sub,
                       SizeType nsamps) noexcept {
    const auto ndt_sub  = dt_grid_sub.size();
    const auto dt_first = dt_grid_sub.front();
    const auto dt_last  = dt_grid_sub.back();
    // First box row of width s0 + 1 (see fdmt_init_row0_box).
    const auto row0_box = [&](TOut* __restrict__ row, SizeType s0) {
        fdmt_init_row0_box<UseRoll ? FDMTMode::kRoll : FDMTMode::kFull>(
            wf_sub, nullptr, row, s0, 0, nsamps);
    };

    if constexpr (!UseBoxSmearing) {
        // No smearing: each row is an independent shift of the raw
        // waterfall by |dt| samples (no accumulation between rows, so no
        // incremental sweep is needed here, unlike the box-smearing case
        // below).
        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto s = static_cast<SizeType>(std::abs(dt_grid_sub[i_dt]));
            fdmt_init_impl_row_shift<UseRoll>(
                wf_sub, buf_base + (i_dt * nsamps), s, nsamps);
        }
        return;
    }

    if (dt_first >= 0) {
        // All-non-negative sub-band grid: first box row, then each wider row
        // extends the previous one by a single sample.
        const auto dt_min_sub = static_cast<SizeType>(dt_first);
        row0_box(buf_base, dt_min_sub);
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur = static_cast<SizeType>(dt_grid_sub[i_dt]);
            TOut* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const TOut* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps);
            fdmt_init_impl_row<UseRoll, true>(wf_sub, buf_prev, buf_cur, dt_cur,
                                              nsamps);
        }
        return;
    }

    // Mixed-sign or purely-negative sub-band grid: sweep magnitude s from
    // the smallest value actually present up to the largest, reusing the
    // running box-sum incrementally, and write each computed row directly
    // into whichever dt index(es) match via O(1) index arithmetic (since
    // dt_grid_sub is dense, value v lives at index v - dt_first). This
    // avoids rescanning the whole sub-band grid for every magnitude, which
    // would otherwise make this O(ndt_sub^2).
    const auto s_lo = static_cast<SizeType>(
        (dt_last >= 0) ? 0 : std::min(std::abs(dt_first), std::abs(dt_last)));
    const auto s_hi =
        static_cast<SizeType>(std::max(std::abs(dt_first), std::abs(dt_last)));

    std::vector<TOut> row_prev(nsamps);
    std::vector<TOut> row_curr(nsamps);

    auto write_matches = [&](SizeType s, const TOut* __restrict__ row) {
        const auto s_signed = static_cast<IndexType>(s);
        if (s_signed >= dt_first && s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(s_signed - dt_first) *
                                    nsamps));
        }
        if (s_signed != 0 && -s_signed >= dt_first && -s_signed <= dt_last) {
            std::copy_n(
                row, nsamps,
                buf_base +
                    (static_cast<SizeType>(-s_signed - dt_first) * nsamps));
        }
    };

    if (s_lo == 0) {
        fdmt_convert_copy(wf_sub, nsamps, row_prev.data());
    } else {
        row0_box(row_prev.data(), s_lo);
    }
    write_matches(s_lo, row_prev.data());

    for (SizeType s = s_lo + 1; s <= s_hi; ++s) {
        fdmt_init_impl_row<UseRoll, true>(wf_sub, row_prev.data(),
                                          row_curr.data(), s, nsamps);
        write_matches(s, row_curr.data());
        row_prev.swap(row_curr);
    }
}

/**
 * @brief One beam's level-0 input: either a float waterfall or a packed
 * low-bit one (LSB-first rows of `row_bytes` bytes, see bit_pack_utils.hpp).
 * Packed rows are unpacked one sub-band at a time into a per-thread float
 * scratch row, so a float copy of the whole waterfall is never materialized.
 */
struct FDMTInput {
    const float* f32{nullptr};
    const uint8_t* packed{nullptr};
    SizeType nbits{32};
    SizeType row_bytes{0};

    // Row `i_sub` as samples of type TS: the float waterfall row itself, or
    // the packed row unpacked into `scratch`. TS other than float is only
    // requested for packed input (integer tree levels).
    template <typename TS>
    [[nodiscard]] const TS* row(SizeType i_sub,
                                SizeType nsamps,
                                std::vector<TS>& scratch) const noexcept {
        if (packed == nullptr) {
            if constexpr (std::is_same_v<TS, float>) {
                return f32 + (i_sub * nsamps);
            } else {
                return nullptr;
            }
        }
        scratch.resize(nsamps);
        utils::unpack_row(packed + (i_sub * row_bytes), nbits, nsamps,
                          scratch.data());
        return scratch.data();
    }
};

template <bool UseRoll, bool UseBoxSmearing, typename TOut = float>
void fdmt_init_impl(const FDMTInput& input,
                    TOut* __restrict__ init_buffer,
                    const plans::FDMTCoordGrid* __restrict__ grids_init,
                    SizeType nsubs,
                    SizeType nsamps) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(input, init_buffer, grids_init, nsubs, nsamps)
#endif
    {
        // Integer level-0 state (int_tree) is fed integer samples directly.
        using TS = std::conditional_t<std::is_integral_v<TOut>, TOut, float>;
        std::vector<TS> scratch;
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
        for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
            const TS* __restrict__ wf_sub =
                input.row<TS>(i_sub, nsamps, scratch);
            TOut* __restrict__ buf_base =
                init_buffer + (grids_init[i_sub].coord_offset * nsamps);
            fdmt_init_subband<UseRoll, UseBoxSmearing>(
                wf_sub, buf_base, grids_init[i_sub].dt_grid, nsamps);
        }
    }
}

template <typename TOut = float, typename TS = float>
void fdmt_init_valid_row0_shift(const TS* __restrict__ wf_sub,
                                const float* __restrict__ hist_sub,
                                TOut* __restrict__ buf_row0,
                                SizeType dt_min_sub,
                                SizeType dt_max_final,
                                SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) without smearing =====
    // For isamp in [0, min(dt_min_sub, nsamps)): read from history. Clamped
    // so dt_min_sub >= nsamps can't write past the end of buf_row0.
    const SizeType boundary_len = std::min(dt_min_sub, nsamps);
    for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
        buf_row0[isamp] =
            static_cast<TOut>(hist_sub[dt_max_final - dt_min_sub + isamp]);
    }
    // For isamp in [dt_min_sub, nsamps): read from waterfall
    for (SizeType isamp = dt_min_sub; isamp < nsamps; ++isamp) {
        buf_row0[isamp] = static_cast<TOut>(wf_sub[isamp - dt_min_sub]);
    }
}

template <bool UseBoxSmearing, typename TOut = float, typename TS = float>
void fdmt_init_valid_row(const TS* __restrict__ wf_sub,
                         const float* __restrict__ hist_sub,
                         const TOut* __restrict__ buf_prev,
                         TOut* __restrict__ buf_cur,
                         SizeType dt_cur,
                         SizeType dt_max_final,
                         SizeType nsamps) noexcept {
    // Clamped so dt_cur >= nsamps can't write past the end of buf_cur (or
    // read past the end of buf_prev, which is the same size).
    const SizeType boundary_len = std::min(dt_cur, nsamps);
    if constexpr (UseBoxSmearing) {
        // Extend box: new_sum[isamp] = prev_sum[isamp] + input[isamp - dt_cur]
        for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
            buf_cur[isamp] = fdmt_init_add<TOut>(
                buf_prev[isamp], hist_sub[dt_max_final + isamp - dt_cur]);
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] =
                fdmt_init_add<TOut>(buf_prev[isamp], wf_sub[isamp - dt_cur]);
        }
    } else {
        // No smearing: just shift by dt_cur
        for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
            buf_cur[isamp] =
                static_cast<TOut>(hist_sub[dt_max_final - dt_cur + isamp]);
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = static_cast<TOut>(wf_sub[isamp - dt_cur]);
        }
    }
}

/**
 * @brief Advances one sub-band's level-0 input history windows by this
 * block. Sub-bands own disjoint history slices, so this runs per sub-band
 * inside the parallel init loop, right after that sub-band's rows are
 * built (which is the only point a packed input's unpacked row exists).
 */
template <typename TS>
void fdmt_init_valid_update_history(const TS* __restrict__ wf_sub,
                                    float* __restrict__ hist_sub,
                                    float* __restrict__ hist_init_sub,
                                    SizeType nsamps,
                                    SizeType dt_max_init,
                                    SizeType dt_max_final) noexcept {
    fdmt_advance_history_window(wf_sub, hist_sub, nsamps, dt_max_final);
    if (hist_init_sub != nullptr && dt_max_init > 0) {
        fdmt_advance_history_window(wf_sub, hist_init_sub, nsamps, dt_max_init);
    }
}

/**
 * @brief `mode == "valid"` analogue of `fdmt_init_subband` (see its doc
 * comment for the magnitude-only rationale). Uses `hist_sub` to seed rows
 * that need samples from before the start of the current block instead of
 * zero-padding them.
 */
template <bool UseBoxSmearing, typename TOut = float, typename TS = float>
void fdmt_init_valid_subband(const TS* __restrict__ wf_sub,
                             const float* __restrict__ hist_sub,
                             TOut* __restrict__ buf_base,
                             const std::vector<IndexType>& dt_grid_sub,
                             SizeType dt_max_final,
                             SizeType nsamps) noexcept {
    const auto ndt_sub  = dt_grid_sub.size();
    const auto dt_first = dt_grid_sub.front();
    const auto dt_last  = dt_grid_sub.back();
    // First box row of width s0 + 1 (see fdmt_init_row0_box).
    const auto row0_box = [&](TOut* __restrict__ row, SizeType s0) {
        fdmt_init_row0_box<FDMTMode::kValid>(wf_sub, hist_sub, row, s0,
                                             dt_max_final, nsamps);
    };

    if constexpr (!UseBoxSmearing) {
        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto s = static_cast<SizeType>(std::abs(dt_grid_sub[i_dt]));
            fdmt_init_valid_row0_shift(wf_sub, hist_sub,
                                       buf_base + (i_dt * nsamps), s,
                                       dt_max_final, nsamps);
        }
        return;
    }

    if (dt_first >= 0) {
        const auto dt_min_sub = static_cast<SizeType>(dt_first);
        row0_box(buf_base, dt_min_sub);
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur = static_cast<SizeType>(dt_grid_sub[i_dt]);
            TOut* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const TOut* __restrict__ buf_prev =
                buf_base + ((i_dt - 1) * nsamps);
            fdmt_init_valid_row<true>(wf_sub, hist_sub, buf_prev, buf_cur,
                                      dt_cur, dt_max_final, nsamps);
        }
        return;
    }

    const auto s_lo = static_cast<SizeType>(
        (dt_last >= 0) ? 0 : std::min(std::abs(dt_first), std::abs(dt_last)));
    const auto s_hi =
        static_cast<SizeType>(std::max(std::abs(dt_first), std::abs(dt_last)));

    std::vector<TOut> row_prev(nsamps);
    std::vector<TOut> row_curr(nsamps);

    auto write_matches = [&](SizeType s, const TOut* __restrict__ row) {
        const auto s_signed = static_cast<IndexType>(s);
        if (s_signed >= dt_first && s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(s_signed - dt_first) *
                                    nsamps));
        }
        if (s_signed != 0 && -s_signed >= dt_first && -s_signed <= dt_last) {
            std::copy_n(
                row, nsamps,
                buf_base +
                    (static_cast<SizeType>(-s_signed - dt_first) * nsamps));
        }
    };

    if (s_lo == 0) {
        fdmt_convert_copy(wf_sub, nsamps, row_prev.data());
    } else {
        row0_box(row_prev.data(), s_lo);
    }
    write_matches(s_lo, row_prev.data());

    for (SizeType s = s_lo + 1; s <= s_hi; ++s) {
        fdmt_init_valid_row<true>(wf_sub, hist_sub, row_prev.data(),
                                  row_curr.data(), s, dt_max_final, nsamps);
        write_matches(s, row_curr.data());
        row_prev.swap(row_curr);
    }
}

// Specialized implementation for VALID mode with history
template <bool UseBoxSmearing, typename TOut = float>
void fdmt_init_valid_impl(const FDMTInput& input,
                          TOut* __restrict__ init_buffer,
                          float* __restrict__ hist_buffer,
                          float* __restrict__ hist_init_buffer,
                          const plans::FDMTCoordGrid* __restrict__ grids_init,
                          SizeType nsubs,
                          SizeType nsamps,
                          SizeType dt_max_init,
                          SizeType dt_max_final) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(input, init_buffer, hist_buffer, hist_init_buffer, grids_init,      \
               nsubs, nsamps, dt_max_init, dt_max_final)
#endif
    {
        // Integer level-0 state (int_tree) is fed integer samples directly.
        using TS = std::conditional_t<std::is_integral_v<TOut>, TOut, float>;
        std::vector<TS> scratch;
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
        for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
            const TS* __restrict__ wf_sub =
                input.row<TS>(i_sub, nsamps, scratch);
            float* __restrict__ hist_sub = hist_buffer + (i_sub * dt_max_final);
            TOut* __restrict__ buf_base =
                init_buffer + (grids_init[i_sub].coord_offset * nsamps);

            fdmt_init_valid_subband<UseBoxSmearing>(wf_sub, hist_sub, buf_base,
                                                    grids_init[i_sub].dt_grid,
                                                    dt_max_final, nsamps);
            fdmt_init_valid_update_history(wf_sub, hist_sub,
                                           (hist_init_buffer != nullptr)
                                               ? hist_init_buffer +
                                                     (i_sub * dt_max_init)
                                               : nullptr,
                                           nsamps, dt_max_init, dt_max_final);
        }
    }
}
} // namespace

class FDMTCPU::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         IndexType dt_max,
         IndexType dt_min,
         SizeType dt_step,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads,
         SizeType nbeams)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_nbeams(nbeams),
          m_plan(f_min,
                 f_max,
                 nchans,
                 nsamps,
                 tsamp,
                 dt_max,
                 dt_min,
                 dt_step,
                 mode,
                 verbose),
          m_state_internal(m_nbeams * m_plan.get_buffer_size() * sizeof(float)),
          m_history(m_mode == FDMTMode::kValid
                        ? m_nbeams * m_plan.get_history_size()
                        : 0,
                    0.0F),
          m_history_init(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_history_init_size()
                             : 0,
                         0.0F),
          m_tree_history(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_tree_history_size()
                             : 0,
                         0.0F) {
        set_dmt_openmp_threads(nthreads);
        set_exec_config(FDMTExecConfig{});
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads,
         SizeType nbeams)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_nbeams(nbeams),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, verbose),
          m_state_internal(m_nbeams * m_plan.get_buffer_size() * sizeof(float)),
          m_history(m_mode == FDMTMode::kValid
                        ? m_nbeams * m_plan.get_history_size()
                        : 0,
                    0.0F),
          m_history_init(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_history_init_size()
                             : 0,
                         0.0F),
          m_tree_history(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_tree_history_size()
                             : 0,
                         0.0F) {
        set_dmt_openmp_threads(nthreads);
        set_exec_config(FDMTExecConfig{});
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int nthreads,
         SizeType nbeams)
        : m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_nbeams(nbeams),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, verbose),
          m_state_internal(m_nbeams * m_plan.get_buffer_size() * sizeof(float)),
          m_history(m_mode == FDMTMode::kValid
                        ? m_nbeams * m_plan.get_history_size()
                        : 0,
                    0.0F),
          m_history_init(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_history_init_size()
                             : 0,
                         0.0F),
          m_tree_history(m_mode == FDMTMode::kValid
                             ? m_nbeams * m_plan.get_tree_history_size()
                             : 0,
                         0.0F) {
        set_dmt_openmp_threads(nthreads);
        set_exec_config(FDMTExecConfig{});
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    [[nodiscard]] SizeType get_nbeams() const noexcept { return m_nbeams; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        {
            const FuseRequest fuse(*this);
            reset(waterfall, dmt);
        }
        advance_until_remaining(0);
        finalize();
        spdlog::debug("FDMTCPU::Impl: Execution complete.");
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt) {
        {
            const FuseRequest fuse(*this);
            reset(waterfall_packed, nbits, dmt);
        }
        advance_until_remaining(0);
        finalize();
        spdlog::debug("FDMTCPU::Impl: Packed execution complete.");
    }

    void reset(std::span<const float> waterfall, std::span<float> dmt) {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall.size() != m_nbeams * nchans * nsamps) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of waterfall. Expected {}, got {}",
                m_nbeams * nchans * nsamps, waterfall.size()));
        }
        check_dmt_size(dmt);
        const FDMTInput input{.f32 = waterfall.data()};
        start(input, nchans * nsamps, all_float_levels(), dmt);
    }

    void reset(std::span<const uint8_t> waterfall_packed,
               SizeType nbits,
               std::span<float> dmt) {
        if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 &&
            nbits != 16) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: nbits={} must be one of 1, 2, 4, 8, 16", nbits));
        }
        const auto nchans    = m_plan.get_nchans();
        const auto nsamps    = m_plan.get_nsamps();
        const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
        if (waterfall_packed.size() != m_nbeams * nchans * row_bytes) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of packed waterfall (nbits={}). "
                "Expected {} bytes, got {}",
                nbits, m_nbeams * nchans * row_bytes, waterfall_packed.size()));
        }
        check_dmt_size(dmt);
        const FDMTInput input{.packed    = waterfall_packed.data(),
                              .nbits     = nbits,
                              .row_bytes = row_bytes};
        start(input, nchans * row_bytes,
              m_exec_config.int_tree ? int_tree_levels(nbits)
                                     : all_float_levels(),
              dmt);
    }

    void set_exec_config(const FDMTExecConfig& config) {
        if (m_is_initialized && !is_finished()) {
            throw std::logic_error(
                "FDMTCPU: set_exec_config() called mid-transform; finish the "
                "current block first.");
        }
        if (config.schedule != FDMTSchedule::kCoord &&
            config.schedule != FDMTSchedule::kTiled) {
            throw std::invalid_argument("FDMTCPU: invalid FDMTSchedule");
        }
        if (config.streaming_stores != FDMTStreamingStores::kOff &&
            config.streaming_stores != FDMTStreamingStores::kAuto &&
            config.streaming_stores != FDMTStreamingStores::kAlways) {
            throw std::invalid_argument("FDMTCPU: invalid FDMTStreamingStores");
        }
        m_exec_config = config;
        m_tile_ndt    = (config.tile_ndt > 0) ? config.tile_ndt : kAutoTileNdt;
        m_fuse_levels = (config.fuse_levels == FDMTExecConfig::kAutoFuse)
                            ? auto_fuse_levels()
                            : std::min(config.fuse_levels, m_plan.get_niters());
        m_tile_nsamps = (config.tile_nsamps > 0) ? config.tile_nsamps
                                                 : auto_tile_nsamps(m_tile_ndt);
        build_chunks();
    }

    [[nodiscard]] const FDMTExecConfig& get_exec_config() const noexcept {
        return m_exec_config;
    }

    [[nodiscard]] SizeType get_effective_tile_nsamps() const noexcept {
        return m_tile_nsamps;
    }

    [[nodiscard]] SizeType get_effective_fuse_levels() const noexcept {
        return m_fuse_levels;
    }

    void advance(SizeType levels = 1) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            execute_iter(next_level);
            m_current_level = next_level;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level);
        }
    }

    [[nodiscard]] std::span<const float> view_level_data() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& state_shape =
            m_plan.get_container().state_shape[m_current_level];
        return {current_level_f32(), state_shape.nelements};
    }

    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTCPU: Subband index {} out of range (current level has {} "
                "subbands)",
                subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return {current_level_f32() + offset, count};
    }

    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(std::format(
                "FDMTCPU: Subband index {} out of range (current level has {} "
                "subbands)",
                subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return FDMTSubbandView{
            .data = std::span<const float>(current_level_f32() + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = state_shape.nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    [[nodiscard]] SizeType current_level() const noexcept {
        return m_current_level;
    }

    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan.get_niters() + 1;
    }

    [[nodiscard]] SizeType remaining_levels() const noexcept {
        if (total_levels() <= 1 || m_current_level >= total_levels() - 1) {
            return 0;
        }
        return (total_levels() - 1) - m_current_level;
    }

    [[nodiscard]] SizeType num_subbands() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().state_shape[m_current_level].nchans;
    }

    [[nodiscard]] bool is_finished() const noexcept {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize() {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        if (!is_finished()) {
            advance_until_remaining(0);
        }
        m_is_initialized = false;
    }

    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width) const {
        return m_plan.get_effective_variance(dm_idx, boxcar_width,
                                             m_use_box_smearing);
    }

    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width) const {
        return m_plan.get_effective_sigma(dm_idx, boxcar_width,
                                          m_use_box_smearing);
    }

    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width) const {
        return m_plan.get_effective_variance_grid(boxcar_width,
                                                  m_use_box_smearing);
    }

    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width) const {
        return m_plan.get_effective_sigma_grid(boxcar_width,
                                               m_use_box_smearing);
    }

    void reset_history() noexcept {
        std::fill(m_history.begin(), m_history.end(), 0.0F);
        std::fill(m_history_init.begin(), m_history_init.end(), 0.0F);
        std::fill(m_tree_history.begin(), m_tree_history.end(), 0.0F);
    }

    [[nodiscard]] SizeType history_state_size() const noexcept {
        return m_history.size() + m_history_init.size() + m_tree_history.size();
    }

    void save_history(std::span<float> out) const {
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(
                std::format("FDMTCPU::save_history: Invalid output size. "
                            "Expected {}, got {}",
                            history_state_size(), out.size()));
        }
        auto it = out.begin();
        it      = std::copy(m_history.begin(), m_history.end(), it);
        it      = std::copy(m_history_init.begin(), m_history_init.end(), it);
        std::copy(m_tree_history.begin(), m_tree_history.end(), it);
    }

    void load_history(std::span<const float> in) {
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(
                std::format("FDMTCPU::load_history: Invalid input size. "
                            "Expected {}, got {}",
                            history_state_size(), in.size()));
        }
        auto it = in.begin();
        std::copy(it, it + static_cast<IndexType>(m_history.size()),
                  m_history.begin());
        it += static_cast<IndexType>(m_history.size());
        std::copy(it, it + static_cast<IndexType>(m_history_init.size()),
                  m_history_init.begin());
        it += static_cast<IndexType>(m_history_init.size());
        std::copy(it, it + static_cast<IndexType>(m_tree_history.size()),
                  m_tree_history.begin());
    }

private:
    // Small chunks + long tiles measured best (M1 sweep, tmp_files/
    // cache_optimization.md): few concurrent row streams keep hardware
    // prefetch effective while still capturing the shared-operand reuse.
    static constexpr SizeType kAutoTileNdt = 4;

    // Storage type of one tree level's state (FDMTExecConfig::int_tree).
    using Elem = detail::FDMTLevelType;

    // Where one level's state lives: `base` of beam 0, beams `beam_stride`
    // bytes apart.
    struct LevelBuf {
        std::byte* base{nullptr};
        Elem type{Elem::kF32};
        SizeType beam_stride{0};
    };

    // malloc-backed byte storage (implicit object creation makes it valid
    // storage for float, uint8_t and uint16_t state alike), zero-filled.
    class StateBytes {
    public:
        explicit StateBytes(SizeType nbytes)
            : m_ptr(static_cast<std::byte*>(
                  std::calloc(std::max<SizeType>(nbytes, 1), 1))),
              m_size(nbytes) {
            if (m_ptr == nullptr) {
                throw std::bad_alloc();
            }
        }
        [[nodiscard]] std::byte* data() const noexcept { return m_ptr.get(); }
        [[nodiscard]] SizeType size() const noexcept { return m_size; }

    private:
        struct FreeDeleter {
            void operator()(std::byte* p) const noexcept { std::free(p); }
        };
        std::unique_ptr<std::byte, FreeDeleter> m_ptr;
        SizeType m_size;
    };

    bool m_use_box_smearing;
    FDMTMode m_mode;
    SizeType m_nbeams;
    plans::FDMTPlan m_plan;
    // Internal state buffer (for ping-pong buffering). With int_tree, both
    // integer ping-pong halves (<= 2 bytes/element each) live in here too.
    StateBytes m_state_internal;
    // History buffers for valid-mode streaming across FDMT blocks. Always
    // float, whatever the input or tree state type.
    std::vector<float> m_history;
    std::vector<float> m_history_init; // only when use_box_smearing is true
    std::vector<float> m_tree_history;

    // Runtime execution switches
    FDMTExecConfig m_exec_config;
    SizeType m_tile_ndt{kAutoTileNdt};
    SizeType m_tile_nsamps{0};
    SizeType m_fuse_levels{0}; // resolved FDMTExecConfig::fuse_levels
    SizeType m_llc_bytes{detect_llc_bytes()};
    std::vector<std::vector<FDMTChunk>> m_chunks; // per level (index 0 unused)
    std::array<std::vector<Elem>, 17> m_int_levels_cache; // indexed by nbits

    // Stepper state
    std::vector<LevelBuf> m_levels;
    // Set only while execute() runs reset(): level fusion replaces levels
    // 0..fuse_levels there, never in the inspectable stepper path.
    bool m_fuse_requested{false};
    std::vector<std::vector<SizeType>> m_fused_sum_index; // per level

    struct FuseRequest {
        Impl& impl;
        explicit FuseRequest(Impl& i) : impl(i) {
            impl.m_fuse_requested = true;
        }
        ~FuseRequest() { impl.m_fuse_requested = false; }
        FuseRequest(const FuseRequest&)            = delete;
        FuseRequest& operator=(const FuseRequest&) = delete;
        FuseRequest(FuseRequest&&)                 = delete;
        FuseRequest& operator=(FuseRequest&&)      = delete;
    };
    SizeType m_current_level{0};
    bool m_is_initialized{false};

    static FDMTMode parse_mode(std::string_view mode) {
        if (mode == "full") {
            return FDMTMode::kFull;
        }
        if (mode == "roll") {
            return FDMTMode::kRoll;
        }
        if (mode == "valid") {
            return FDMTMode::kValid;
        }
        throw std::invalid_argument(std::format(
            "Invalid mode '{}'. Expected 'full', 'roll', or 'valid'", mode));
    }

    template <typename F> static void with_elem(Elem e, F&& f) {
        switch (e) {
        case Elem::kU8:
            f.template operator()<uint8_t>();
            break;
        case Elem::kU16:
            f.template operator()<uint16_t>();
            break;
        case Elem::kF32:
            f.template operator()<float>();
            break;
        }
    }

    template <typename T>
    [[nodiscard]] T* level_ptr(SizeType level, SizeType beam) const noexcept {
        const auto& lb = m_levels[level];
        return reinterpret_cast<T*>(lb.base + (beam * lb.beam_stride));
    }

    [[nodiscard]] const float* current_level_f32() const {
        if (m_levels[m_current_level].type != Elem::kF32) {
            throw std::logic_error(std::format(
                "FDMTCPU: level {} is stored as an integer type "
                "(FDMTExecConfig::int_tree); disable int_tree to inspect "
                "intermediate levels.",
                m_current_level));
        }
        return level_ptr<const float>(m_current_level, 0);
    }

    void check_dmt_size(std::span<float> dmt) const {
        if (dmt.size() < m_nbeams * m_plan.get_buffer_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of dmt. Expected at least {}, got {}",
                m_nbeams * m_plan.get_buffer_size(), dmt.size()));
        }
    }

    [[nodiscard]] std::vector<Elem> all_float_levels() const {
        return std::vector<Elem>(total_levels(), Elem::kF32);
    }

    // Per-level storage types for packed `nbits` input with int_tree (see
    // detail::int_tree_level_types, shared with the CUDA backend).
    [[nodiscard]] const std::vector<Elem>& int_tree_levels(SizeType nbits) {
        auto& cached = m_int_levels_cache[nbits];
        if (cached.empty()) {
            cached =
                detail::int_tree_level_types(m_plan, m_use_box_smearing, nbits);
        }
        return cached;
    }

    /**
     * Assigns each level's buffer. Float levels ping-pong between the
     * caller's dmt buffer and m_state_internal so the root lands in dmt
     * (level l is in dmt iff niters - l is even -- the original parity
     * rule). Integer levels alternate between the two halves of
     * m_state_internal; int_tree_levels() guarantees the first float level
     * after them is a dmt level.
     */
    void layout_levels(const std::vector<Elem>& types, std::span<float> dmt) {
        const SizeType niters = m_plan.get_niters();
        const SizeType bufsz  = m_plan.get_buffer_size();
        auto* dmt_bytes       = reinterpret_cast<std::byte*>(dmt.data());
        auto* internal        = m_state_internal.data();
        const SizeType half   = m_nbeams * bufsz * sizeof(uint16_t);
        m_levels.resize(niters + 1);
        for (SizeType l = 0; l <= niters; ++l) {
            if (types[l] == Elem::kF32) {
                m_levels[l] = {
                    .base = ((niters - l) % 2 == 0) ? dmt_bytes : internal,
                    .type = Elem::kF32,
                    .beam_stride = bufsz * sizeof(float),
                };
            } else {
                m_levels[l] = {
                    .base        = internal + ((l % 2 == 0) ? 0 : half),
                    .type        = types[l],
                    .beam_stride = bufsz * sizeof(uint16_t),
                };
            }
        }
    }

    void start(const FDMTInput& input,
               SizeType input_beam_stride,
               const std::vector<Elem>& types,
               std::span<float> dmt) {
        layout_levels(types, dmt);
        const SizeType fuse = m_fuse_requested ? m_fuse_levels : 0;
        if (fuse > 0) {
            initialise_fused(input, input_beam_stride, fuse);
            m_current_level = fuse;
        } else {
            m_current_level = 0;
            initialise(input, input_beam_stride);
        }
        m_is_initialized = true;
        spdlog::debug("FDMTCPU: Stepper initialized at level 0.");
    }

    static SizeType detect_l2_per_core() noexcept {
        constexpr SizeType kFallback = SizeType{1} << 20;
#if defined(__APPLE__)
        uint64_t l2 = 0;
        size_t len  = sizeof(l2);
        if (sysctlbyname("hw.perflevel0.l2cachesize", &l2, &len, nullptr, 0) !=
                0 ||
            l2 == 0) {
            len = sizeof(l2);
            if (sysctlbyname("hw.l2cachesize", &l2, &len, nullptr, 0) != 0) {
                return kFallback;
            }
        }
        uint64_t cpus = 0;
        len           = sizeof(cpus);
        if (sysctlbyname("hw.perflevel0.cpusperl2", &cpus, &len, nullptr, 0) !=
                0 ||
            cpus == 0) {
            cpus = 1;
        }
        return (l2 > 0) ? static_cast<SizeType>(l2 / cpus) : kFallback;
#elif defined(__linux__) && defined(_SC_LEVEL2_CACHE_SIZE)
        const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        return (l2 > 0) ? static_cast<SizeType>(l2) : kFallback;
#else
        return kFallback;
#endif
    }

    // Last-level cache size (bytes) gating FDMTExecConfig::streaming_stores.
    static SizeType detect_llc_bytes() noexcept {
        constexpr SizeType kFallback = SizeType{32} << 20;
#if defined(__APPLE__)
        uint64_t llc = 0;
        size_t len   = sizeof(llc);
        if (sysctlbyname("hw.l3cachesize", &llc, &len, nullptr, 0) == 0 &&
            llc > 0) {
            return static_cast<SizeType>(llc);
        }
        llc = 0;
        len = sizeof(llc);
        if (sysctlbyname("hw.perflevel0.l2cachesize", &llc, &len, nullptr, 0) ==
                0 &&
            llc > 0) {
            return static_cast<SizeType>(llc);
        }
        return kFallback;
#elif defined(__linux__) && defined(_SC_LEVEL3_CACHE_SIZE)
        const long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (l3 > 0) {
            return static_cast<SizeType>(l3);
        }
        const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        return (l2 > 0) ? static_cast<SizeType>(l2) : kFallback;
#else
        return kFallback;
#endif
    }

    // Largest power-of-two tile keeping one work item's working set -- the
    // chunk's (<= tile_ndt) output rows plus its (<= 2 * tile_ndt) distinct
    // input rows -- within about half the per-core L2.
    static SizeType auto_tile_nsamps(SizeType tile_ndt) noexcept {
        constexpr SizeType kMinTile = 512;
        const SizeType budget       = detect_l2_per_core() / 2;
        const SizeType per_sample   = 3 * tile_ndt * sizeof(float);
        const SizeType t = std::max<SizeType>(budget / per_sample, 1);
        return std::max(kMinTile, std::bit_floor(t));
    }

    // Splits each level's sum coordinates (contiguous per output sub-band,
    // dt-ascending) into runs of <= m_tile_ndt that never span sub-bands.
    void build_chunks() {
        const auto& pc = m_plan.get_container();
        m_chunks.assign(total_levels(), {});
        for (SizeType l = 1; l < total_levels(); ++l) {
            const auto& coords = pc.coordinates_sum[l];
            auto& chunks       = m_chunks[l];
            SizeType begin     = 0;
            for (SizeType k = 1; k <= coords.size(); ++k) {
                if (k == coords.size() ||
                    coords[k].i_sub != coords[begin].i_sub ||
                    k - begin == m_tile_ndt) {
                    chunks.push_back({.begin = begin, .end = k});
                    begin = k;
                }
            }
        }
    }

    // Beam-major layout: beam b's waterfall/state/history all live in
    // disjoint, per-beam-sized slices of the same flat buffers (the plan's
    // coordinate offsets are beam-agnostic and apply identically within each
    // slice). Looping over beams here -- rather than folding beam into the
    // OpenMP parallel-for inside fdmt_init_impl/fdmt_init_valid_impl/
    // fdmt_iter -- means those functions, and the nbeams=1 hot path, are
    // completely untouched by this generalization: at nbeams=1 the loop body
    // runs exactly once with a zero offset, reproducing today's single-beam
    // call byte-for-byte. (Folding beam into the parallel region for better
    // multi-beam throughput is a follow-up optimization once this is
    // benchmarked, not part of this change.)
    void initialise(const FDMTInput& input, SizeType input_beam_stride) {
        const auto& plan_c     = m_plan.get_container();
        const auto& grids_init = plan_c.grids[0];
        const auto nsamps      = plan_c.state_shape[0].nsamps;
        const auto dt_max_init = plan_c.state_shape[0].dt_max;
        const auto dt_max_final =
            plan_c.state_shape[m_plan.get_niters()].dt_max;
        const auto nsubs = plan_c.state_shape[0].nchans;

        const auto hist_beam_stride = m_plan.get_history_size();
        const auto hist_init_stride = m_plan.get_history_init_size();

        with_elem(m_levels[0].type, [&]<typename T>() {
            for (SizeType b = 0; b < m_nbeams; ++b) {
                FDMTInput input_b = input;
                if (input_b.packed != nullptr) {
                    input_b.packed += b * input_beam_stride;
                } else {
                    input_b.f32 += b * input_beam_stride;
                }
                T* __restrict__ init_buffer_b = level_ptr<T>(0, b);

                if (m_mode == FDMTMode::kFull) {
                    if (m_use_box_smearing) {
                        fdmt_init_impl<false, true>(input_b, init_buffer_b,
                                                    grids_init.data(), nsubs,
                                                    nsamps);
                    } else {
                        fdmt_init_impl<false, false>(input_b, init_buffer_b,
                                                     grids_init.data(), nsubs,
                                                     nsamps);
                    }
                } else if (m_mode == FDMTMode::kRoll) {
                    if (m_use_box_smearing) {
                        fdmt_init_impl<true, true>(input_b, init_buffer_b,
                                                   grids_init.data(), nsubs,
                                                   nsamps);
                    } else {
                        fdmt_init_impl<true, false>(input_b, init_buffer_b,
                                                    grids_init.data(), nsubs,
                                                    nsamps);
                    }
                } else {
                    float* __restrict__ hist_b =
                        m_history.data() + (b * hist_beam_stride);
                    float* __restrict__ hist_init_b =
                        m_history_init.data() + (b * hist_init_stride);
                    if (m_use_box_smearing) {
                        fdmt_init_valid_impl<true>(
                            input_b, init_buffer_b, hist_b, hist_init_b,
                            grids_init.data(), nsubs, nsamps, dt_max_init,
                            dt_max_final);
                    } else {
                        fdmt_init_valid_impl<false>(
                            input_b, init_buffer_b, hist_b, hist_init_b,
                            grids_init.data(), nsubs, nsamps, dt_max_init,
                            dt_max_final);
                    }
                }
            }
        });
    }

    // ---------------------------------------------------------------------
    // Level fusion (FDMTExecConfig::fuse_levels)
    // ---------------------------------------------------------------------

    // Per level: linear coordinate index -> its index in coordinates_sum, or
    // SIZE_MAX for a copy coordinate. Both lists are in coordinate order, so
    // one merge-style pass per level builds it. The plan never changes, so
    // this is built once.
    void build_fused_sum_index() {
        if (!m_fused_sum_index.empty()) {
            return;
        }
        const auto& pc = m_plan.get_container();
        m_fused_sum_index.assign(total_levels(), {});
        for (SizeType l = 1; l < total_levels(); ++l) {
            const auto& coords                      = pc.coordinates[l];
            [[maybe_unused]] const auto& coords_sum = pc.coordinates_sum[l];
            auto& index                             = m_fused_sum_index[l];
            index.assign(coords.size(), SIZE_MAX);
            SizeType j = 0;
            for (SizeType k = 0; k < coords.size(); ++k) {
                if (coords[k].i_coord_head != SIZE_MAX) {
                    index[k] = j++;
                }
            }
            assert(j == coords_sum.size());
        }
    }

    // Coordinate range [begin, end) of level `l` owned by fused group `g`:
    // the level-l sub-bands descending from level-F sub-band g, i.e.
    // [g * 2^(F-l), (g+1) * 2^(F-l)) clipped to the level's sub-band count
    // (odd counts end in a single-child copy sub-band, which still descends
    // from sub-band floor(j / 2)).
    [[nodiscard]] std::pair<SizeType, SizeType>
    fused_coord_range(SizeType l, SizeType g, SizeType fuse) const noexcept {
        const auto& grids    = m_plan.get_container().grids[l];
        const SizeType width = SizeType{1} << (fuse - l);
        const SizeType first = g * width;
        const SizeType last  = std::min(first + width, grids.size()) - 1;
        return {grids[first].coord_offset,
                grids[last].coord_offset + grids[last].ndt};
    }

    // Merges level-l coordinates [c_begin, c_end) of one fused group. `in`
    // / `out` hold the group's rows starting at coordinates in_base /
    // out_base; the merge itself is the unmodified offset_add / copy used by
    // fdmt_iter, so fused results are bit-identical.
    template <FDMTMode Mode, typename TIn, typename TOut>
    void fused_merge_group(const TIn* __restrict__ in,
                           SizeType in_base,
                           TOut* __restrict__ out,
                           SizeType out_base,
                           float* __restrict__ hist_b,
                           SizeType l,
                           SizeType c_begin,
                           SizeType c_end) const noexcept {
        const auto& pc            = m_plan.get_container();
        const auto& coords        = pc.coordinates[l];
        const auto& coords_sum    = pc.coordinates_sum[l];
        const auto& sum_index     = m_fused_sum_index[l];
        const SizeType in_origin  = in_base * pc.state_shape[l - 1].nsamps;
        const SizeType out_origin = out_base * pc.state_shape[l].nsamps;
        for (SizeType k = c_begin; k < c_end; ++k) {
            // Sum coordinates are read from coordinates_sum: only those
            // entries carry the valid-mode tree-history slot (hist_offset is
            // assigned there after the plan is built).
            const auto& coord    = (sum_index[k] == SIZE_MAX)
                                       ? coords[k]
                                       : coords_sum[sum_index[k]];
            const TIn* tail      = in + (coord.tail_buf_offset - in_origin);
            TOut* __restrict__ o = out + (coord.buf_offset - out_origin);
            if (coord.i_coord_head == SIZE_MAX) {
                fdmt_convert_copy(tail, coord.tail_nsamps, o);
                if (coord.nsamps > coord.tail_nsamps) {
                    std::fill(o + coord.tail_nsamps, o + coord.nsamps, TOut{0});
                }
                continue;
            }
            const TIn* head = in + (coord.head_buf_offset - in_origin);
            float* hist     = (hist_b != nullptr && coord.delay > 0)
                                  ? hist_b + coord.hist_offset
                                  : nullptr;
            offset_add<Mode>(tail, coord.tail_nsamps, head, coord.head_nsamps,
                             o, coord.nsamps, coord.delay, hist);
        }
    }

    template <FDMTMode Mode>
    void fused_merge_dispatch(const std::byte* in,
                              SizeType in_base,
                              std::byte* out,
                              SizeType out_base,
                              float* hist_b,
                              SizeType l,
                              SizeType c_begin,
                              SizeType c_end) const noexcept {
        with_elem(m_levels[l - 1].type, [&]<typename TIn>() {
            with_elem(m_levels[l].type, [&]<typename TOut>() {
                if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                              !(std::is_floating_point_v<TIn> &&
                                !std::is_floating_point_v<TOut>)) {
                    fused_merge_group<Mode>(
                        reinterpret_cast<const TIn*>(in), in_base,
                        reinterpret_cast<TOut*>(out), out_base, hist_b, l,
                        c_begin, c_end);
                }
            });
        });
    }

    // Level-0 rows of channels [ch_begin, ch_end) into `buf` (rows from
    // coordinate `base`), plus their valid-mode input history -- exactly
    // the per-sub-band work of initialise(), for one fused group.
    template <typename T0>
    void fused_init_group(
        const FDMTInput& input_b,
        T0* __restrict__ buf,
        SizeType base,
        SizeType ch_begin,
        SizeType ch_end,
        SizeType beam,
        std::vector<std::conditional_t<std::is_integral_v<T0>, T0, float>>&
            scratch) {
        using TS       = std::conditional_t<std::is_integral_v<T0>, T0, float>;
        const auto& pc = m_plan.get_container();
        const auto& grids       = pc.grids[0];
        const SizeType nsamps   = pc.state_shape[0].nsamps;
        const auto dt_max_init  = pc.state_shape[0].dt_max;
        const auto dt_max_final = pc.state_shape[m_plan.get_niters()].dt_max;
        for (SizeType c = ch_begin; c < ch_end; ++c) {
            const TS* __restrict__ wf = input_b.row<TS>(c, nsamps, scratch);
            T0* __restrict__ rows =
                buf + ((grids[c].coord_offset - base) * nsamps);
            const auto& dt_grid = grids[c].dt_grid;
            if (m_mode == FDMTMode::kFull) {
                if (m_use_box_smearing) {
                    fdmt_init_subband<false, true>(wf, rows, dt_grid, nsamps);
                } else {
                    fdmt_init_subband<false, false>(wf, rows, dt_grid, nsamps);
                }
            } else if (m_mode == FDMTMode::kRoll) {
                if (m_use_box_smearing) {
                    fdmt_init_subband<true, true>(wf, rows, dt_grid, nsamps);
                } else {
                    fdmt_init_subband<true, false>(wf, rows, dt_grid, nsamps);
                }
            } else {
                float* hist_sub    = m_history.data() +
                                     (beam * m_plan.get_history_size()) +
                                     (c * dt_max_final);
                float* hist_init_b = m_history_init.data() +
                                     (beam * m_plan.get_history_init_size());
                if (m_use_box_smearing) {
                    fdmt_init_valid_subband<true>(wf, hist_sub, rows, dt_grid,
                                                  dt_max_final, nsamps);
                } else {
                    fdmt_init_valid_subband<false>(wf, hist_sub, rows, dt_grid,
                                                   dt_max_final, nsamps);
                }
                fdmt_init_valid_update_history(
                    wf, hist_sub,
                    (hist_init_b != nullptr) ? hist_init_b + (c * dt_max_init)
                                             : nullptr,
                    nsamps, dt_max_init, dt_max_final);
            }
        }
    }

    // Largest per-group scratch level buffer for fusion depth `fuse` (4
    // bytes/element covers every storage type); two are live per thread.
    [[nodiscard]] SizeType fused_scratch_bytes(SizeType fuse) const noexcept {
        const auto& pc         = m_plan.get_container();
        const SizeType ngroups = pc.state_shape[fuse].nchans;
        SizeType max_bytes     = 0;
        for (SizeType l = 0; l < fuse; ++l) {
            for (SizeType g = 0; g < ngroups; ++g) {
                const auto [cb, ce] = fused_coord_range(l, g, fuse);
                max_bytes =
                    std::max(max_bytes, (ce - cb) * pc.state_shape[l].nsamps *
                                            sizeof(float));
            }
        }
        return max_bytes;
    }

    // kAutoFuse: the deepest fusion whose two per-thread scratch buffers fit
    // in ~2x the per-core L2 (M1 sweep: shallower past that point, deeper
    // fusion starts spilling and loses; see tmp_files/cache_optimization.md).
    // 0 when even one level would not fit.
    [[nodiscard]] SizeType auto_fuse_levels() const noexcept {
        const SizeType budget = 2 * detect_l2_per_core();
        SizeType best         = 0;
        for (SizeType f = 1; f <= m_plan.get_niters(); ++f) {
            if (2 * fused_scratch_bytes(f) > budget) {
                break;
            }
            best = f;
        }
        return best;
    }

    /**
     * Fused replacement for initialise() + levels 1..fuse. Level-F sub-band
     * g depends only on the 2^F input channels below it, so each thread
     * builds one such group at a time: level-0 rows into a per-thread
     * scratch buffer, then each merge level ping-pongs between two scratch
     * buffers, and only level F is written to its real state buffer. The
     * group's rows (a few rows per channel) stay cache-resident, removing
     * the main-memory write + read of every intermediate level.
     */
    void initialise_fused(const FDMTInput& input,
                          SizeType input_beam_stride,
                          SizeType fuse) {
        const auto& pc         = m_plan.get_container();
        const SizeType ngroups = pc.state_shape[fuse].nchans;
        const SizeType nchans  = pc.state_shape[0].nchans;
        build_fused_sum_index();
        const SizeType max_bytes    = fused_scratch_bytes(fuse);
        const auto tree_hist_stride = m_plan.get_tree_history_size();

        for (SizeType b = 0; b < m_nbeams; ++b) {
            FDMTInput input_b = input;
            if (input_b.packed != nullptr) {
                input_b.packed += b * input_beam_stride;
            } else {
                input_b.f32 += b * input_beam_stride;
            }
            float* hist_b = (m_mode == FDMTMode::kValid)
                                ? m_tree_history.data() + (b * tree_hist_stride)
                                : nullptr;
            std::byte* final_out = level_ptr<std::byte>(fuse, b);
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel
#endif
            {
                // Uninitialised per-thread scratch: every row a group reads
                // is written first.
                std::unique_ptr<std::byte[]> buf_a(new std::byte[max_bytes]);
                std::unique_ptr<std::byte[]> buf_b(new std::byte[max_bytes]);
                std::vector<float> scratch_f32;
                std::vector<uint8_t> scratch_u8;
                std::vector<uint16_t> scratch_u16;
#ifdef DMT_ENABLE_OPENMP
#pragma omp for schedule(static)
#endif
                for (SizeType g = 0; g < ngroups; ++g) {
                    const SizeType ch_begin = g << fuse;
                    const SizeType ch_end =
                        std::min(ch_begin + (SizeType{1} << fuse), nchans);
                    const auto [c0b, c0e] = fused_coord_range(0, g, fuse);
                    std::byte* cur        = buf_a.get();
                    std::byte* nxt        = buf_b.get();
                    with_elem(m_levels[0].type, [&]<typename T0>() {
                        auto& scratch = [&]() -> auto& {
                            if constexpr (std::is_same_v<T0, uint8_t>) {
                                return scratch_u8;
                            } else if constexpr (std::is_same_v<T0, uint16_t>) {
                                return scratch_u16;
                            } else {
                                return scratch_f32;
                            }
                        }();
                        fused_init_group<T0>(input_b,
                                             reinterpret_cast<T0*>(cur), c0b,
                                             ch_begin, ch_end, b, scratch);
                    });
                    SizeType in_base = c0b;
                    for (SizeType l = 1; l <= fuse; ++l) {
                        const auto [cb, ce]     = fused_coord_range(l, g, fuse);
                        const bool last         = (l == fuse);
                        std::byte* out          = last ? final_out : nxt;
                        const SizeType out_base = last ? 0 : cb;
                        if (m_mode == FDMTMode::kFull) {
                            fused_merge_dispatch<FDMTMode::kFull>(
                                cur, in_base, out, out_base, hist_b, l, cb, ce);
                        } else if (m_mode == FDMTMode::kRoll) {
                            fused_merge_dispatch<FDMTMode::kRoll>(
                                cur, in_base, out, out_base, hist_b, l, cb, ce);
                        } else {
                            fused_merge_dispatch<FDMTMode::kValid>(
                                cur, in_base, out, out_base, hist_b, l, cb, ce);
                        }
                        std::swap(cur, nxt);
                        in_base = cb;
                    }
                }
            }
        }
    }

    template <FDMTMode Mode, typename TIn, typename TOut>
    void execute_iter_typed(SizeType i_iter) {
        const auto& plan_c          = m_plan.get_container();
        const auto& coords_sum_cur  = plan_c.coordinates_sum[i_iter];
        const auto& coords_copy_cur = plan_c.coordinates_copy[i_iter];
        const auto ncoords_sum_cur  = coords_sum_cur.size();
        const auto ncoords_copy_cur = coords_copy_cur.size();
        const auto hist_beam_stride = m_plan.get_tree_history_size();
        const bool tiled = m_exec_config.schedule == FDMTSchedule::kTiled;
        // Streaming stores only help once this level's output cannot stay
        // cached until the next level reads it back.
        const auto streaming = m_exec_config.streaming_stores;
        const bool stream_out =
            streaming == FDMTStreamingStores::kAlways ||
            (streaming == FDMTStreamingStores::kAuto &&
             m_nbeams * plan_c.state_shape[i_iter].nelements * sizeof(TOut) >
                 m_llc_bytes);

        for (SizeType b = 0; b < m_nbeams; ++b) {
            const TIn* __restrict__ state_in_b = level_ptr<TIn>(i_iter - 1, b);
            TOut* __restrict__ state_out_b     = level_ptr<TOut>(i_iter, b);
            float* __restrict__ hist_b =
                (Mode == FDMTMode::kValid)
                    ? m_tree_history.data() + (b * hist_beam_stride)
                    : nullptr;
            if (tiled) {
                fdmt_iter_tiled<Mode>(
                    state_in_b, state_out_b, hist_b, coords_sum_cur.data(),
                    coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur,
                    m_chunks[i_iter].data(), m_chunks[i_iter].size(),
                    plan_c.state_shape[i_iter].nsamps, m_tile_nsamps);
            } else {
                fdmt_iter<Mode>(state_in_b, state_out_b, hist_b,
                                coords_sum_cur.data(), coords_copy_cur.data(),
                                ncoords_sum_cur, ncoords_copy_cur, stream_out);
            }
        }
    }

    void execute_iter(SizeType i_iter) {
        with_elem(m_levels[i_iter - 1].type, [&]<typename TIn>() {
            with_elem(m_levels[i_iter].type, [&]<typename TOut>() {
                // Levels only ever widen (see int_tree_levels()).
                if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                              !(std::is_floating_point_v<TIn> &&
                                !std::is_floating_point_v<TOut>)) {
                    if (m_mode == FDMTMode::kFull) {
                        execute_iter_typed<FDMTMode::kFull, TIn, TOut>(i_iter);
                    } else if (m_mode == FDMTMode::kRoll) {
                        execute_iter_typed<FDMTMode::kRoll, TIn, TOut>(i_iter);
                    } else {
                        execute_iter_typed<FDMTMode::kValid, TIn, TOut>(i_iter);
                    }
                } else {
                    throw std::logic_error(
                        "FDMTCPU: invalid narrowing level transition");
                }
            });
        });
    }

}; // End FDMTCPU::Impl definition

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 IndexType dt_max,
                 IndexType dt_min,
                 SizeType dt_step,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_min,
                                    dt_step,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads,
                                    nbeams)) {}
FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<IndexType>& dt_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads,
                                    nbeams)) {}

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 bool verbose,
                 int nthreads,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dm_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    nthreads,
                                    nbeams)) {}

FDMTCPU::~FDMTCPU()                                   = default;
FDMTCPU::FDMTCPU(FDMTCPU&& other) noexcept            = default;
FDMTCPU& FDMTCPU::operator=(FDMTCPU&& other) noexcept = default;
const plans::FDMTPlan& FDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
SizeType FDMTCPU::get_nbeams() const noexcept { return m_impl->get_nbeams(); }
void FDMTCPU::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}
void FDMTCPU::reset(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->reset(waterfall, dmt);
}
void FDMTCPU::execute(std::span<const uint8_t> waterfall_packed,
                      SizeType nbits,
                      std::span<float> dmt) {
    m_impl->execute(waterfall_packed, nbits, dmt);
}
void FDMTCPU::reset(std::span<const uint8_t> waterfall_packed,
                    SizeType nbits,
                    std::span<float> dmt) {
    m_impl->reset(waterfall_packed, nbits, dmt);
}
void FDMTCPU::set_exec_config(const FDMTExecConfig& config) {
    m_impl->set_exec_config(config);
}
const FDMTExecConfig& FDMTCPU::get_exec_config() const noexcept {
    return m_impl->get_exec_config();
}
SizeType FDMTCPU::get_effective_tile_nsamps() const noexcept {
    return m_impl->get_effective_tile_nsamps();
}
SizeType FDMTCPU::get_effective_fuse_levels() const noexcept {
    return m_impl->get_effective_fuse_levels();
}
void FDMTCPU::advance(SizeType levels) { m_impl->advance(levels); }
void FDMTCPU::advance_until_remaining(SizeType remaining_levels) {
    m_impl->advance_until_remaining(remaining_levels);
}
std::span<const float> FDMTCPU::view_level_data() const {
    return m_impl->view_level_data();
}
std::span<const float> FDMTCPU::view_subband_data(SizeType subband_idx) const {
    return m_impl->view_subband_data(subband_idx);
}
FDMTSubbandView FDMTCPU::view_subband(SizeType subband_idx) const {
    return m_impl->view_subband(subband_idx);
}
SizeType FDMTCPU::current_level() const noexcept {
    return m_impl->current_level();
}
SizeType FDMTCPU::total_levels() const noexcept {
    return m_impl->total_levels();
}
SizeType FDMTCPU::remaining_levels() const noexcept {
    return m_impl->remaining_levels();
}
SizeType FDMTCPU::num_subbands() const { return m_impl->num_subbands(); }
bool FDMTCPU::is_finished() const noexcept { return m_impl->is_finished(); }
void FDMTCPU::finalize() { m_impl->finalize(); }

float FDMTCPU::get_effective_variance(SizeType dm_idx,
                                      SizeType boxcar_width) const {
    return m_impl->get_effective_variance(dm_idx, boxcar_width);
}

float FDMTCPU::get_effective_sigma(SizeType dm_idx,
                                   SizeType boxcar_width) const {
    return m_impl->get_effective_sigma(dm_idx, boxcar_width);
}

std::vector<float>
FDMTCPU::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_variance_grid(boxcar_width);
}

std::vector<float>
FDMTCPU::get_effective_sigma_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_sigma_grid(boxcar_width);
}

void FDMTCPU::reset_history() noexcept { m_impl->reset_history(); }
SizeType FDMTCPU::history_state_size() const noexcept {
    return m_impl->history_state_size();
}
void FDMTCPU::save_history(std::span<float> out) const {
    m_impl->save_history(out);
}
void FDMTCPU::load_history(std::span<const float> in) {
    m_impl->load_history(in);
}

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             IndexType dt_max,
             IndexType dt_min,
             SizeType dt_step,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads,
             SizeType nbeams) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                 use_box_smearing, mode, verbose, nthreads, nbeams);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(nbeams * buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    // Compact each beam's leading get_dmt_size() samples (discarding the
    // buffer_size-sized scratch tail) into a contiguous (nbeams, dmt_size)
    // result.
    const auto dmt_size = fdmt_plan.get_dmt_size();
    std::vector<float> dmt_out(nbeams * dmt_size);
    for (SizeType b = 0; b < nbeams; ++b) {
        std::copy_n(dmt.data() + (b * buffer_size), dmt_size,
                    dmt_out.data() + (b * dmt_size));
    }
    return std::make_tuple(std::move(dmt_out), fdmt_plan);
}

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<IndexType>& dt_grid,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads,
             SizeType nbeams) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid, use_box_smearing,
                 mode, verbose, nthreads, nbeams);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(nbeams * buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    const auto dmt_size = fdmt_plan.get_dmt_size();
    std::vector<float> dmt_out(nbeams * dmt_size);
    for (SizeType b = 0; b < nbeams; ++b) {
        std::copy_n(dmt.data() + (b * buffer_size), dmt_size,
                    dmt_out.data() + (b * dmt_size));
    }
    return std::make_tuple(std::move(dmt_out), fdmt_plan);
}

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             bool use_box_smearing,
             std::string_view mode,
             bool verbose,
             int nthreads,
             SizeType nbeams) {
    FDMTCPU fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid, use_box_smearing,
                 mode, verbose, nthreads, nbeams);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(nbeams * buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    const auto dmt_size = fdmt_plan.get_dmt_size();
    std::vector<float> dmt_out(nbeams * dmt_size);
    for (SizeType b = 0; b < nbeams; ++b) {
        std::copy_n(dmt.data() + (b * buffer_size), dmt_size,
                    dmt_out.data() + (b * dmt_size));
    }
    return std::make_tuple(std::move(dmt_out), fdmt_plan);
}

void add_frb_track(std::span<float> waterfall,
                   const plans::FDMTPlan& plan,
                   SizeType dm_idx,
                   float amplitude,
                   IndexType toffset,
                   SizeType width) {
    if (width == 0) {
        throw std::invalid_argument(
            "add_frb_track: width must be greater than 0");
    }
    const auto nchans = plan.get_nchans();
    const auto nsamps = plan.get_nsamps();
    if (waterfall.size() != nchans * nsamps) {
        throw std::invalid_argument(std::format(
            "add_frb_track: Invalid size of waterfall. Expected {}, got {}",
            nchans * nsamps, waterfall.size()));
    }
    // trace_dm() throws std::out_of_range if dm_idx is invalid.
    const auto shifts = plan.trace_dm(dm_idx);

    for (SizeType c = 0; c < nchans; ++c) {
        const auto start = toffset + shifts[c];
        const auto end   = start + static_cast<IndexType>(width);
        if (start < 0 || end > static_cast<IndexType>(nsamps)) {
            throw std::out_of_range(std::format(
                "add_frb_track: injected pulse for channel {} falls outside "
                "the waterfall (start={}, end={}, nsamps={}); choose a "
                "smaller |toffset| or a dm_idx with less total delay",
                c, start, end, nsamps));
        }
        float* __restrict__ row = waterfall.data() + (c * nsamps);
        for (IndexType t = start; t < end; ++t) {
            row[static_cast<SizeType>(t)] += amplitude;
        }
    }
}

} // namespace dmt::algorithms
