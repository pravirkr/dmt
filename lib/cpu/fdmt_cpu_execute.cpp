#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <omp.h>

#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_search.hpp"
#include "dmt/fdmt_tree.hpp"

#include "fdmt_cpu_engine.hpp"

#if defined(__AVX__)
#include "dmt/x86_intrinsics.hpp"
#endif

// CPU FDMT engine: execute(), the boxcar search input and the tree
// levels, with their kernels.

namespace dmt::algorithms::detail::fdmt_cpu {

namespace {

/// Stores src[0 .. n) into a ring of `delay` slots at stream positions
/// pos .. pos + n (slot = position mod delay), n <= delay: at most two
/// contiguous copies instead of a remainder per sample.
template <typename T>
void ring_store(
    T* ring, SizeType delay, uint64_t pos, const T* src, SizeType n) {
    const auto slot   = static_cast<SizeType>(pos % delay);
    const SizeType n1 = std::min(n, delay - slot);
    std::copy_n(src, n1, ring + slot);
    std::copy_n(src + n1, n - n1, ring);
}

/**
 * @brief out[i] = a[i] + b[i] for i < m with non-temporal (streaming) stores
 * where the target supports them, for rows that go to main memory anyway (a
 * level far larger than the caches): it skips the read-for-ownership of
 * every output line. Same float additions as the plain loop, so the result
 * is bit-identical. Callers must fdmt_stream_fence() before another thread
 * reads `out`.
 */
inline void fdmt_stream_add(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ out,
                            SizeType m) noexcept {
#if defined(__AVX512F__) || defined(__AVX__)
#if defined(__AVX512F__)
    constexpr SizeType kLanes = 16;
#else
    constexpr SizeType kLanes = 8;
#endif
    constexpr std::uintptr_t kAlign = kLanes * sizeof(float);
    SizeType i                      = 0;
    for (; i < m && (reinterpret_cast<std::uintptr_t>(out + i) % kAlign) != 0;
         ++i) {
        out[i] = a[i] + b[i];
    }
    for (; i + kLanes <= m; i += kLanes) {
#if defined(__AVX512F__)
        _mm512_stream_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i),
                                                _mm512_loadu_ps(b + i)));
#else
        _mm256_stream_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i),
                                                _mm256_loadu_ps(b + i)));
#endif
    }
    for (; i < m; ++i) {
        out[i] = a[i] + b[i];
    }
#elif defined(__clang__) && __has_builtin(__builtin_nontemporal_store)
    for (SizeType i = 0; i < m; ++i) {
        __builtin_nontemporal_store(a[i] + b[i], out + i);
    }
#else
#pragma omp simd
    for (SizeType i = 0; i < m; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

/// Orders this thread's streaming stores before a following barrier.
inline void fdmt_stream_fence() noexcept {
#if defined(__AVX__)
    _mm_sfence();
#endif
}

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

// Units of the tree state. With box smearing every level-0 box carries the
// dyadic box-mean weight w(L) = round(2^b / L) (fdmt_box_weights.hpp).
// Integer levels (int_tree) hold exact integers in units of `int_unit` =
// 2^-b; float levels and every history buffer hold canonical box-mean units.
// The only conversions are integer -> float (the first float level) and the
// history boundary of integer levels, both exact power-of-two scalings,
// which commute with float rounding: the integer tree stays bit-identical to
// the float tree. Without box smearing int_unit is 1.

/// `a` of storage type TIn as storage type TOut (rescaled int -> float).
template <typename TOut, typename TIn>
inline TOut fdmt_convert(TIn a, float int_unit) noexcept {
    if constexpr (std::is_integral_v<TIn> && !std::is_integral_v<TOut>) {
        return static_cast<float>(a) * int_unit;
    } else {
        return static_cast<TOut>(a);
    }
}

/// The merge `a + b` of two TIn values stored as TOut (rescaled int ->
/// float after the addition, which is exact).
template <typename TOut, typename TIn>
inline TOut fdmt_merge(TIn a, TIn b, float int_unit) noexcept {
    if constexpr (std::is_integral_v<TIn> && !std::is_integral_v<TOut>) {
        return fdmt_add<float>(a, b) * int_unit;
    } else {
        return fdmt_add<TOut>(a, b);
    }
}

/**
 * @brief out[i] = fdmt_merge(a[i], b[i]) with streaming stores, for the
 * level types fdmt_stream_add does not cover (integer levels and the first
 * float level after them). Each 64-byte block of `out` is computed with the
 * plain merge expression into registers and stored non-temporally, so the
 * result is bit-identical to the plain loop. Besides skipping the
 * read-for-ownership, this keeps a level's lines from staying dirty in the
 * writing core's cache: the next level's threads read them from memory
 * instead of from another core (on a multi-CCD part such as EPYC, a slow
 * cross-die transfer). Callers must fdmt_stream_fence() before another
 * thread reads `out`.
 */
template <typename TOut, typename TIn>
inline void fdmt_stream_merge(const TIn* __restrict__ a,
                              const TIn* __restrict__ b,
                              TOut* __restrict__ out,
                              SizeType m,
                              float int_unit) noexcept {
#if defined(__AVX__)
    constexpr SizeType kBlockBytes = 64;
    constexpr SizeType kBlock      = kBlockBytes / sizeof(TOut);
    SizeType i                     = 0;
    for (; i < m &&
           (reinterpret_cast<std::uintptr_t>(out + i) % kBlockBytes) != 0;
         ++i) {
        out[i] = fdmt_merge<TOut>(a[i], b[i], int_unit);
    }
    for (; i + kBlock <= m; i += kBlock) {
        alignas(kBlockBytes) TOut blk[kBlock];
#pragma omp simd
        for (SizeType j = 0; j < kBlock; ++j) {
            blk[j] = fdmt_merge<TOut>(a[i + j], b[i + j], int_unit);
        }
#if defined(__AVX512F__)
        _mm512_stream_si512(reinterpret_cast<__m512i*>(out + i),
                            _mm512_load_si512(blk));
#else
        const auto* src = reinterpret_cast<const __m256i*>(blk);
        auto* dst       = reinterpret_cast<__m256i*>(out + i);
        _mm256_stream_si256(dst, _mm256_load_si256(src));
        _mm256_stream_si256(dst + 1, _mm256_load_si256(src + 1));
#endif
    }
    for (; i < m; ++i) {
        out[i] = fdmt_merge<TOut>(a[i], b[i], int_unit);
    }
#else
#pragma omp simd
    for (SizeType i = 0; i < m; ++i) {
        out[i] = fdmt_merge<TOut>(a[i], b[i], int_unit);
    }
#endif
}

/// fdmt_convert_copy, rescaling int -> float.
template <typename TIn, typename TOut>
inline void fdmt_convert_copy_unit(const TIn* __restrict__ src,
                                   SizeType n,
                                   TOut* __restrict__ dst,
                                   float int_unit) noexcept {
    if constexpr (std::is_integral_v<TIn> && !std::is_integral_v<TOut>) {
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) {
            dst[i] = fdmt_convert<TOut>(src[i], int_unit);
        }
    } else {
        fdmt_convert_copy(src, n, dst);
    }
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
 * drift apart. History is always stored as float in canonical units,
 * whatever the state type (`TIn`) of the stream feeding it: integer state is
 * rescaled by `int_unit` (see fdmt_convert).
 */
template <typename TIn = float>
void fdmt_advance_history_window(const TIn* __restrict__ new_data,
                                 float* __restrict__ hist,
                                 SizeType nsamps,
                                 SizeType capacity,
                                 float int_unit = 1.0F) noexcept {
    if (capacity == 0) {
        return;
    }
    if (nsamps >= capacity) {
        fdmt_convert_copy_unit(new_data + nsamps - capacity, capacity, hist,
                               int_unit);
    } else {
        std::copy(hist + nsamps, hist + capacity, hist);
        fdmt_convert_copy_unit(new_data, nsamps, hist + capacity - nsamps,
                               int_unit);
    }
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
 *   FDMTCoord::hist_offset and FDMT::reset_history()). `hist` is a ring of
 *   `delay_shift` slots: head sample of stream time tau lives in slot
 *   tau mod delay_shift, and `stream_pos` is the stream time of sample 0 of
 *   this block, so a block costs O(min(delay, block)) history traffic
 *   whatever the delay. Each coordinate owns a disjoint slice of the shared
 *   history buffer, so this is safe across the `#pragma omp for` in
 *   fdmt_iter below with no extra locking.
 * - Roll: Cyclic boundary - head wraps around using modulo arithmetic
 *
 * @tparam Mode FDMTMode enum specifying boundary handling behavior
 * @tparam TIn Storage type of the input (previous level) state.
 * @tparam TOut Storage type of the output (current level) state.
 * @tparam Stream Write the overlap region with streaming stores (see
 * fdmt_stream_add and fdmt_stream_merge).
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
 * history is needed (delay_shift == 0). Canonical units (see fdmt_convert).
 * @param int_unit Value of one integer state unit (2^-F with box smearing).
 * @param stream_pos Valid mode: stream time of this block's first sample.
 * @param ring_pos Valid mode: stream_pos % delay_shift if the caller tracks
 * it (saves a division per call), else SIZE_MAX.
 */
template <FDMTMode Mode,
          typename TIn  = float,
          typename TOut = float,
          bool Stream   = false>
void offset_add(const TIn* __restrict__ data_tail,
                SizeType size_tail,
                const TIn* __restrict__ data_head,
                [[maybe_unused]] SizeType size_head,
                TOut* __restrict__ out,
                SizeType size_out,
                SizeType delay_shift,
                float* __restrict__ hist = nullptr,
                float int_unit           = 1.0F,
                SizeType stream_pos      = 0,
                SizeType ring_pos        = SIZE_MAX) noexcept {
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
        fdmt_convert_copy_unit(data_tail, delay_shift, out, int_unit);

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = fdmt_merge<TOut>(data_tail[delay_shift + i],
                                                    data_head[i], int_unit);
        }

        // Part 3: Head-only region [size_tail, size_tail + nrest)
        const SizeType nrest = std::min(delay_shift, size_out - size_tail);
        if (nrest > 0) {
            fdmt_convert_copy_unit(data_head + nsum, nrest, out + size_tail,
                                   int_unit);
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
            const SizeType d = delay_shift;
            const SizeType n = size_tail;
            // Slot of the oldest sample (`ring_pos` when the caller tracks it).
            const SizeType pos =
                (ring_pos != SIZE_MAX) ? ring_pos : stream_pos % d;
            // Output i < min(d, n) adds head sample tau = stream_pos + i - d,
            // ring slot (pos + i) mod d: two contiguous runs.
            const SizeType nb    = std::min(d, n);
            const SizeType first = std::min(nb, d - pos);
            const float unit     = std::is_integral_v<TIn> ? int_unit : 1.0F;
            if (n <= d) {
                // The whole block comes from the ring, and its n head
                // samples refill exactly the n slots just read: one pass
                // per run (read the slot, store the head sample, add).
                const auto swap_add = [&](SizeType i0, SizeType i1,
                                          SizeType slot0) {
                    float* __restrict__ ring   = hist + slot0;
                    const TIn* __restrict__ tl = data_tail + i0;
                    const TIn* __restrict__ hd = data_head + i0;
                    TOut* __restrict__ o       = out + i0;
                    const SizeType m           = i1 - i0;
                    if constexpr (std::is_integral_v<TOut>) {
                        const float to_int = 1.0F / int_unit;
#pragma omp simd
                        for (SizeType j = 0; j < m; ++j) {
                            const float h = ring[j];
                            ring[j]       = fdmt_convert<float>(hd[j], unit);
                            o[j]          = fdmt_add<TOut>(
                                tl[j], static_cast<TOut>(h * to_int));
                        }
                    } else {
#pragma omp simd
                        for (SizeType j = 0; j < m; ++j) {
                            const float h = ring[j];
                            ring[j]       = fdmt_convert<float>(hd[j], unit);
                            o[j]          = fdmt_add<TOut>(
                                fdmt_convert<float>(tl[j], int_unit), h);
                        }
                    }
                };
                swap_add(0, first, pos);
                swap_add(first, n, 0);
                return;
            }
            const auto add_hist = [&](SizeType i0, SizeType i1,
                                      SizeType slot0) {
                if constexpr (std::is_integral_v<TOut>) {
                    // Canonical history -> integer units (exact).
                    const float to_int = 1.0F / int_unit;
                    for (SizeType i = i0; i < i1; ++i) {
                        out[i] = fdmt_add<TOut>(
                            data_tail[i],
                            static_cast<TOut>(hist[slot0 + i - i0] * to_int));
                    }
                } else {
                    for (SizeType i = i0; i < i1; ++i) {
                        out[i] = fdmt_add<TOut>(
                            fdmt_convert<float>(data_tail[i], int_unit),
                            hist[slot0 + i - i0]);
                    }
                }
            };
            add_hist(0, first, pos);
            add_hist(first, nb, 0);
            // Store this block's last min(d, n) head samples (canonical
            // units) over the slots just consumed: the sample of stream
            // time tau goes to slot tau mod d.
            const SizeType skip   = n - nb; // head samples not kept
            const SizeType slot0  = (pos + skip) % d;
            const SizeType first2 = std::min(nb, d - slot0);
            fdmt_convert_copy_unit(data_head + skip, first2, hist + slot0,
                                   unit);
            fdmt_convert_copy_unit(data_head + skip + first2, nb - first2, hist,
                                   unit);
        } else {
            fdmt_convert_copy_unit(data_tail, delay_shift, out, int_unit);
        }

        // Part 2: Overlap region [delay_shift, size_tail), empty once
        // delay_shift >= size_tail.
        const SizeType nsum =
            (delay_shift < size_tail) ? (size_tail - delay_shift) : 0;
        if constexpr (Stream && std::is_same_v<TIn, float> &&
                      std::is_same_v<TOut, float>) {
            fdmt_stream_add(data_tail + delay_shift, data_head,
                            out + delay_shift, nsum);
        } else if constexpr (Stream) {
            fdmt_stream_merge<TOut>(data_tail + delay_shift, data_head,
                                    out + delay_shift, nsum, int_unit);
        } else {
#pragma omp simd
            for (SizeType i = 0; i < nsum; ++i) {
                out[delay_shift + i] = fdmt_merge<TOut>(
                    data_tail[delay_shift + i], data_head[i], int_unit);
            }
        }
    } else if constexpr (Mode == FDMTMode::kRoll) {
        // Roll mode: cyclic addition
        assert(size_out == size_tail && "All sizes must be equal in roll mode");
        // Part 1: Wrapped region [0, delay_shift)
        for (SizeType k = 0; k < delay_shift; ++k) {
            out[k] = fdmt_merge<TOut>(
                data_tail[k], data_head[size_tail - delay_shift + k], int_unit);
        }

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = fdmt_merge<TOut>(data_tail[delay_shift + i],
                                                    data_head[i], int_unit);
        }
    }
}

/// @brief One merge op: out = tail + head shifted by the delay.
template <FDMTMode Mode, typename TIn, typename TOut, bool Stream = false>
inline void fdmt_merge_op(const TIn* __restrict__ state_in,
                          TOut* __restrict__ state_out,
                          float* __restrict__ hist_ptr,
                          const FDMTMergeOp& op,
                          FDMTRowLengths len,
                          float int_unit,
                          SizeType stream_pos) noexcept {
    float* hist =
        (hist_ptr != nullptr && op.delay > 0) ? hist_ptr + op.hist : nullptr;
    const auto n_in = len.tail(op);
    offset_add<Mode, TIn, TOut, Stream>(
        state_in + op.tail, n_in, state_in + op.head, n_in, state_out + op.out,
        len.out(op), op.delay, hist, int_unit, stream_pos);
}

/// @brief One copy op (odd-channel padding): out = tail, zero-extended.
template <typename TIn, typename TOut>
inline void fdmt_copy_op(const TIn* __restrict__ state_in,
                         TOut* __restrict__ state_out,
                         const FDMTMergeOp& op,
                         FDMTRowLengths len,
                         float int_unit) noexcept {
    TOut* out       = state_out + op.out;
    const auto n_in = len.tail(op);
    const auto n    = len.out(op);
    fdmt_convert_copy_unit(state_in + op.tail, n_in, out, int_unit);
    if (n > n_in) {
        std::fill(out + n_in, out + n, TOut{0});
    }
}

/// Calling thread's index inside a parallel region.
inline SizeType fdmt_thread_id() noexcept {
    return static_cast<SizeType>(omp_get_thread_num());
}

/// @brief One tree level: every op (a merge, or a copy when head is
/// SIZE_MAX), in parallel.
template <FDMTMode Mode, typename TIn = float, typename TOut = float>
void fdmt_iter(const TIn* __restrict__ state_in,
               TOut* __restrict__ state_out,
               float* __restrict__ hist_ptr,
               std::span<const FDMTMergeOp> level_ops,
               FDMTRowLengths len,
               float int_unit,
               SizeType stream_pos,
               [[maybe_unused]] int nthreads) noexcept {
    const FDMTMergeOp* ops = level_ops.data();
    const SizeType nops    = level_ops.size();
    // A level of the unfused path is far larger than the caches: stream it.
#pragma omp parallel default(none) num_threads(nthreads) shared(               \
        state_in, state_out, hist_ptr, ops, nops, len, int_unit, stream_pos)
    {
#pragma omp for nowait
        for (SizeType i = 0; i < nops; ++i) {
            if (ops[i].head == SIZE_MAX) {
                fdmt_copy_op(state_in, state_out, ops[i], len, int_unit);
            } else {
                fdmt_merge_op<Mode, TIn, TOut, true>(state_in, state_out,
                                                     hist_ptr, ops[i], len,
                                                     int_unit, stream_pos);
            }
        }
        fdmt_stream_fence();
    }
}

/// @brief Level-0 box-sum extension `prev + sample`, stored as TOut. Integer
/// state adds in integer lanes (a float sample, the cross-block history of
/// packed input, holds an exact integer); the plan-derived bound keeps it
/// from wrapping. Anything else adds in float, the original arithmetic.
template <typename TOut, typename TS>
inline TOut fdmt_init_add(TOut prev, TS sample) noexcept {
    if constexpr (std::is_integral_v<TOut>) {
        return static_cast<TOut>(prev + static_cast<TOut>(sample));
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

/// Float input: the box sum times its dyadic weight.
struct FDMTBoxScaleF32 {
    float w;
    [[nodiscard]] float operator()(float acc) const noexcept { return acc * w; }
};

/// Packed input: the box sum (an exact integer, accumulated in P) through
/// its rounding recipe, stored as integer state (units 2^-F) or float state
/// (times 2^-F). P is the narrowest unsigned type holding the recipe's
/// product (detail::box_round_max), so the sums run in the widest SIMD lanes
/// the plan allows.
template <typename TOut, typename P> struct FDMTBoxScaleExact;

template <typename TOut, typename P> struct FDMTBoxScaleRound {
    detail::FDMTBoxRound r;
    float unit;
    [[nodiscard]] TOut operator()(P acc) const noexcept {
        const P v = detail::box_round_apply<P>(acc, r);
        if constexpr (std::is_integral_v<TOut>) {
            return static_cast<TOut>(v);
        } else {
            return static_cast<float>(v) * unit;
        }
    }
    /// Rows with b(L) <= f(L) (L = 1, 2, 4) need no rounding: one multiply.
    [[nodiscard]] bool exact() const noexcept { return r.rshift == 0; }
    [[nodiscard]] FDMTBoxScaleExact<TOut, P> as_exact() const noexcept;
};

/// An exact packed-input row: sum * (mul << lshift), bit-identical to
/// FDMTBoxScaleRound when its recipe does not round, in one multiply.
template <typename TOut, typename P> struct FDMTBoxScaleExact {
    P m;
    float unit;
    [[nodiscard]] TOut operator()(P acc) const noexcept {
        const auto v = static_cast<P>(acc * m);
        if constexpr (std::is_integral_v<TOut>) {
            return static_cast<TOut>(v);
        } else {
            return static_cast<float>(v) * unit;
        }
    }
};

template <typename TOut, typename P>
FDMTBoxScaleExact<TOut, P>
FDMTBoxScaleRound<TOut, P>::as_exact() const noexcept {
    return {.m = static_cast<P>(r.mul << r.lshift), .unit = unit};
}

/// Calls f(scale), or f(scale.as_exact()) when the row needs no rounding.
template <typename Scale, typename F>
inline void fdmt_with_scale(const Scale& scale, F&& f) {
    if constexpr (requires { scale.exact(); }) {
        if (scale.exact()) {
            f(scale.as_exact());
            return;
        }
    }
    f(scale);
}

/**
 * @brief First level-0 box sum acc[i] = sum_{k=0..s0} x[i - k], where
 * `x[j < 0]` is 0 (kFull), the wrapped waterfall (kRoll) or the cross-block
 * history (kValid; `hist_sub` holds the channel's last `hist_len` input
 * samples).
 *
 * Evaluated as s0 + 1 vectorizable shifted adds (x[i], then + x[i-1], ...),
 * the same summation order as the CUDA init kernel, so CPU and GPU level-0
 * rows agree. This replaced a serial running sum (`+= x[i]; -= x[i-w]`)
 * whose loop-carried float dependency dominated FDMT runtime with box
 * smearing and accumulated rounding error along the block (it did not even
 * reproduce a width-1 box exactly: (x0 + x1) - x0 != x1). Every output here
 * is a sum of at most s0 + 1 samples, so its error is bounded per sample.
 * Integer-valued input (packed low-bit) is summed exactly in integer lanes.
 */
template <FDMTMode Mode,
          typename TAcc,
          typename TOut,
          typename TS,
          typename Scale>
void fdmt_init_row0_box(const TS* __restrict__ wf_sub,
                        const float* __restrict__ hist_sub,
                        TAcc* __restrict__ acc,
                        TOut* __restrict__ out,
                        Scale scale,
                        SizeType s0,
                        SizeType hist_len,
                        SizeType nsamps,
                        bool keep_acc) noexcept {
    // The last pass also writes the row's box mean out[i] = scale(acc[i]);
    // the running sum is kept only if a wider row extends it.
    if (s0 == 0) {
        // NOLINTNEXTLINE(bugprone-branch-clone): differs; omp simd compares equal
        if (keep_acc) {
#pragma omp simd
            for (SizeType i = 0; i < nsamps; ++i) {
                acc[i] = static_cast<TAcc>(wf_sub[i]);
                out[i] = scale(acc[i]);
            }
        } else {
#pragma omp simd
            for (SizeType i = 0; i < nsamps; ++i) {
                out[i] = scale(static_cast<TAcc>(wf_sub[i]));
            }
        }
        return;
    }
    fdmt_convert_copy(wf_sub, nsamps, acc);
    for (SizeType s = 1; s <= s0; ++s) {
        const SizeType nb = std::min(s, nsamps);
        if constexpr (Mode == FDMTMode::kRoll) {
            for (SizeType i = 0; i < nb; ++i) {
                acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[nsamps - s + i]);
            }
        } else if constexpr (Mode == FDMTMode::kValid) {
            for (SizeType i = 0; i < nb; ++i) {
                acc[i] =
                    fdmt_init_add<TAcc>(acc[i], hist_sub[hist_len - s + i]);
            }
        }
        if (s < s0) {
#pragma omp simd
            for (SizeType i = s; i < nsamps; ++i) {
                acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]);
            }
        } else {
            for (SizeType i = 0; i < nb; ++i) {
                out[i] = scale(acc[i]);
            }
            // NOLINTNEXTLINE(bugprone-branch-clone): differs; omp simd compares equal
            if (keep_acc) {
#pragma omp simd
                for (SizeType i = s; i < nsamps; ++i) {
                    acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]);
                    out[i] = scale(acc[i]);
                }
            } else {
#pragma omp simd
                for (SizeType i = s; i < nsamps; ++i) {
                    out[i] = scale(fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]));
                }
            }
        }
    }
}

/**
 * @brief Extends the running box sum `acc` (width s) by one sample to width
 * s + 1, acc[i] += x[i - s] (boundary as in fdmt_init_row0_box), and writes
 * the box mean out[i] = scale(acc[i]). The running sum stays unscaled: the
 * next row extends it again.
 */
template <FDMTMode Mode,
          typename TAcc,
          typename TOut,
          typename TS,
          typename Scale>
void fdmt_init_box_extend(const TS* __restrict__ wf_sub,
                          const float* __restrict__ hist_sub,
                          TAcc* __restrict__ acc,
                          TOut* __restrict__ out,
                          SizeType s,
                          Scale scale,
                          SizeType hist_len,
                          SizeType nsamps,
                          bool keep_acc) noexcept {
    const SizeType nb = std::min(s, nsamps);
    if constexpr (Mode == FDMTMode::kRoll) {
        for (SizeType i = 0; i < nb; ++i) {
            acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[nsamps - s + i]);
        }
    } else if constexpr (Mode == FDMTMode::kValid) {
        for (SizeType i = 0; i < nb; ++i) {
            acc[i] = fdmt_init_add<TAcc>(acc[i], hist_sub[hist_len - s + i]);
        }
    }
    // kFull: no sample before the block, the box keeps its partial sum.
    for (SizeType i = 0; i < nb; ++i) {
        out[i] = scale(acc[i]);
    }
    // NOLINTNEXTLINE(bugprone-branch-clone): differs; omp simd compares equal
    if (keep_acc) {
#pragma omp simd
        for (SizeType i = s; i < nsamps; ++i) {
            acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]);
            out[i] = scale(acc[i]);
        }
    } else {
        // The channel's widest row: the running sum is not needed again.
#pragma omp simd
        for (SizeType i = s; i < nsamps; ++i) {
            out[i] = scale(fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]));
        }
    }
}

/// Box widths extended per pass of fdmt_init_box_extend_block.
inline constexpr SizeType kBoxBlock = 4;

/**
 * @brief fdmt_init_box_extend for the kBoxBlock consecutive widths s0 ..
 * s0 + kBoxBlock - 1 in one pass over the samples: the running sum stays in
 * a register across the block, so it is loaded and stored once per block
 * instead of once per row, and every row is written exactly once. Each
 * sample is still extended in order of increasing width, so the result is
 * bit-identical to kBoxBlock single-width calls.
 */
template <FDMTMode Mode,
          typename TAcc,
          typename TOut,
          typename TS,
          typename Scale>
void fdmt_init_box_extend_block(const TS* __restrict__ wf_sub,
                                const float* __restrict__ hist_sub,
                                TAcc* __restrict__ acc,
                                TOut* const* outs,
                                const Scale* scales,
                                SizeType s0,
                                SizeType hist_len,
                                SizeType nsamps,
                                bool keep_acc) noexcept {
    constexpr SizeType kK = kBoxBlock;
    // Samples i < s0 + K - 1 need, for some width of the block, a sample
    // before the block (history / wrap / none): row by row.
    const SizeType head = std::min(s0 + kK - 1, nsamps);
    for (SizeType k = 0; k < kK; ++k) {
        const SizeType s  = s0 + k;
        TOut* out         = outs[k];
        const Scale scale = scales[k];
        const SizeType nb = std::min(s, head);
        if constexpr (Mode == FDMTMode::kRoll) {
            for (SizeType i = 0; i < nb; ++i) {
                acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[nsamps - s + i]);
            }
        } else if constexpr (Mode == FDMTMode::kValid) {
            for (SizeType i = 0; i < nb; ++i) {
                acc[i] =
                    fdmt_init_add<TAcc>(acc[i], hist_sub[hist_len - s + i]);
            }
        }
        for (SizeType i = 0; i < nb; ++i) {
            out[i] = scale(acc[i]);
        }
        for (SizeType i = s; i < head; ++i) {
            acc[i] = fdmt_init_add<TAcc>(acc[i], wf_sub[i - s]);
            out[i] = scale(acc[i]);
        }
    }
    TOut* o[kK];
    Scale sk[kK];
    for (SizeType k = 0; k < kK; ++k) {
        o[k]  = outs[k];
        sk[k] = scales[k];
    }
    const TS* __restrict__ x = wf_sub - s0;
    const auto body          = [&](auto keep) {
#pragma omp simd
        for (SizeType i = head; i < nsamps; ++i) {
            TAcc a = acc[i];
            // `#pragma GCC unroll` is understood by GCC and Clang alike (the
            // count must be a literal for GCC < 14).
            static_assert(kK == 4, "update the unroll count below");
#pragma GCC unroll 4
            for (SizeType k = 0; k < kK; ++k) {
                a       = fdmt_init_add<TAcc>(a, x[i - k]);
                o[k][i] = sk[k](a);
            }
            if constexpr (decltype(keep)::value) {
                acc[i] = a;
            }
        }
    };
    if (keep_acc) {
        body(std::true_type{});
    } else {
        body(std::false_type{});
    }
}

/**
 * @brief Box-smearing level-0 rows of one sub-band (channel) for any mode.
 *
 * `dt_grid_sub` is dense, contiguous and ascending (built by
 * `make_plan_iter0`), but may be negative when the plan's dt range is
 * negative or straddles zero. Level 0 has no notion of merge *direction* --
 * that is resolved by the sign-aware tail/head reference swap in
 * `make_plan`. A level-0 row only ever represents "this single native
 * channel's own smearing width of |dt| samples": rows +s and -s are always
 * identical. So one sweep over the magnitude s covers every grid: the box
 * sum of width s + 1 is extended one sample at a time in `acc` (one nsamps
 * scratch row of TAcc), and each magnitude is written, as its box mean
 * `scale_at(s)`, to the +s and/or -s rows present. Every magnitude in
 * [s_lo, s_hi] has at least one row because the grid is dense.
 */
template <FDMTMode Mode,
          typename TAcc,
          typename TOut,
          typename TS,
          typename ScaleAt>
void fdmt_init_subband_box(const TS* __restrict__ wf_sub,
                           const float* __restrict__ hist_sub,
                           TOut* __restrict__ buf_base,
                           const std::vector<IndexType>& dt_grid_sub,
                           SizeType hist_len,
                           SizeType nsamps,
                           SizeType stride,
                           TAcc* __restrict__ acc,
                           const ScaleAt& scale_at) noexcept {
    using Scale          = decltype(scale_at(SizeType{0}));
    const auto dt_first  = dt_grid_sub.front();
    const auto dt_last   = dt_grid_sub.back();
    const auto abs_first = static_cast<SizeType>(std::abs(dt_first));
    const auto abs_last  = static_cast<SizeType>(std::abs(dt_last));
    // Sweep start: the smallest magnitude present.
    const SizeType s_lo = (dt_first >= 0)  ? abs_first
                          : (dt_last >= 0) ? 0
                                           : std::min(abs_first, abs_last);
    const SizeType s_hi = std::max(abs_first, abs_last);

    // The +s row (or, if absent, the -s row) receives the box mean; a second
    // matching row (-s, mixed-sign grids only) is a copy.
    const auto rows_for = [&](SizeType s) {
        const auto s_signed = static_cast<IndexType>(s);
        TOut* plus          = nullptr;
        TOut* minus         = nullptr;
        if (s_signed >= dt_first && s_signed <= dt_last) {
            plus = buf_base +
                   (static_cast<SizeType>(s_signed - dt_first) * stride);
        }
        if (s_signed != 0 && -s_signed >= dt_first && -s_signed <= dt_last) {
            minus = buf_base +
                    (static_cast<SizeType>(-s_signed - dt_first) * stride);
        }
        assert((plus != nullptr || minus != nullptr) &&
               "level-0 dt grids are dense");
        return (plus != nullptr) ? std::pair{plus, minus}
                                 : std::pair{minus, plus};
    };

    {
        // Not a structured binding: OpenMP cannot capture those.
        const auto rows = rows_for(s_lo);
        fdmt_with_scale(scale_at(s_lo), [&](const auto& scale) {
            fdmt_init_row0_box<Mode>(wf_sub, hist_sub, acc, rows.first, scale,
                                     s_lo, hist_len, nsamps, s_hi > s_lo);
        });
        if (rows.second != nullptr) {
            std::copy_n(rows.first, nsamps, rows.second);
        }
    }
    SizeType s = s_lo + 1;
    // Blocks of kBoxBlock widths, then the remaining widths one by one.
    for (; s + kBoxBlock - 1 <= s_hi; s += kBoxBlock) {
        TOut* outs[kBoxBlock];
        Scale scales[kBoxBlock];
        bool all_exact = true;
        for (SizeType k = 0; k < kBoxBlock; ++k) {
            outs[k]   = rows_for(s + k).first;
            scales[k] = scale_at(s + k);
            if constexpr (requires { scales[k].exact(); }) {
                all_exact = all_exact && scales[k].exact();
            } else {
                all_exact = false;
            }
        }
        const bool keep = s + kBoxBlock - 1 < s_hi;
        if constexpr (requires { scales[0].as_exact(); }) {
            if (all_exact) {
                decltype(scales[0].as_exact()) exact[kBoxBlock];
                for (SizeType k = 0; k < kBoxBlock; ++k) {
                    exact[k] = scales[k].as_exact();
                }
                fdmt_init_box_extend_block<Mode>(wf_sub, hist_sub, acc, outs,
                                                 exact, s, hist_len, nsamps,
                                                 keep);
            } else {
                fdmt_init_box_extend_block<Mode>(wf_sub, hist_sub, acc, outs,
                                                 scales, s, hist_len, nsamps,
                                                 keep);
            }
        } else {
            fdmt_init_box_extend_block<Mode>(wf_sub, hist_sub, acc, outs,
                                             scales, s, hist_len, nsamps, keep);
        }
        for (SizeType k = 0; k < kBoxBlock; ++k) {
            const auto rows = rows_for(s + k);
            if (rows.second != nullptr) {
                std::copy_n(rows.first, nsamps, rows.second);
            }
        }
    }
    for (; s <= s_hi; ++s) {
        // Not a structured binding: clang 16-18 cannot capture one in a
        // lambda when compiling with OpenMP.
        const auto rows = rows_for(s);
        auto* const out = rows.first;
        fdmt_with_scale(scale_at(s), [&](const auto& scale) {
            fdmt_init_box_extend<Mode>(wf_sub, hist_sub, acc, out, s, scale,
                                       hist_len, nsamps, s < s_hi);
        });
        if (rows.second != nullptr) {
            std::copy_n(out, nsamps, rows.second);
        }
    }
}

/// @brief No-smearing valid-mode level-0 row: the waterfall shifted by
/// `shift` samples, the first ones from the cross-block history.
template <typename TOut = float, typename TS = float>
void fdmt_init_valid_row0_shift(const TS* __restrict__ wf_sub,
                                const float* __restrict__ hist_sub,
                                TOut* __restrict__ buf_row0,
                                SizeType shift,
                                SizeType hist_len,
                                SizeType nsamps) noexcept {
    // Clamped so shift >= nsamps can't write past the end of buf_row0.
    const SizeType boundary_len = std::min(shift, nsamps);
    for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
        buf_row0[isamp] = static_cast<TOut>(hist_sub[hist_len - shift + isamp]);
    }
    for (SizeType isamp = shift; isamp < nsamps; ++isamp) {
        buf_row0[isamp] = static_cast<TOut>(wf_sub[isamp - shift]);
    }
}

/**
 * @brief Where one sub-band's level-0 rows come from and go to: its rows
 * (`buf_base`), its valid-mode input history (`hist_sub`, the last
 * `hist_len` raw samples; `hist_init_sub`, the last `dt_max_init`) and the
 * per-thread scratch.
 */
struct FDMTLevel0Channel {
    SizeType i_sub;
    const std::vector<IndexType>* dt_grid;
    float* hist_sub; ///< last `hist_len` raw input samples (valid mode)
    SizeType hist_len;
    SizeType nsamps;    ///< samples of this chunk
    SizeType stride;    ///< row stride of the level-0 rows
    std::byte* unpack;  ///< scratch row of nsamps samples (<= 4 bytes)
    std::byte* acc_row; ///< scratch row of nsamps samples (<= 8 bytes)
};

/**
 * @brief All level-0 rows of one sub-band, and in valid mode its input
 * history update: the shifted waterfall without box smearing, else the box
 * means (fdmt_box_weights.hpp) -- dyadic float weights for float input, the
 * rounded integer recipe for packed input (`round_bytes`: bytes of its
 * accumulator / product type, 1, 2, 4 or 8; the scratch row holds 8 per
 * sample; byte accumulators read the input unpacked to bytes).
 */
template <FDMTMode Mode, bool UseBoxSmearing, typename TOut>
void fdmt_init_channel(const FDMTInput& input,
                       TOut* __restrict__ buf_base,
                       const FDMTLevel0Channel& ch,
                       const FDMTBoxWeights& weights,
                       SizeType round_bytes) noexcept {
    const auto& dt_grid = *ch.dt_grid;
    const auto nsamps   = ch.nsamps;
    // Valid mode: advance the channel's raw-input history window by this
    // chunk, right after its rows are built (the only point a packed
    // input's unpacked row exists). Raw samples, never rescaled.
    const auto finish = [&]([[maybe_unused]] const auto* wf) {
        if constexpr (Mode == FDMTMode::kValid) {
            fdmt_advance_history_window(wf, ch.hist_sub, nsamps, ch.hist_len);
        }
    };
    if constexpr (UseBoxSmearing) {
        if (input.packed != nullptr) {
            const auto run = [&]<typename P>() {
                using TS =
                    std::conditional_t<sizeof(P) == 1, uint8_t, uint16_t>;
                const auto* wf = input.row<TS>(
                    ch.i_sub, nsamps, reinterpret_cast<TS*>(ch.unpack));
                fdmt_init_subband_box<Mode>(
                    wf, ch.hist_sub, buf_base, dt_grid, ch.hist_len, nsamps,
                    ch.stride, reinterpret_cast<P*>(ch.acc_row),
                    [&](SizeType s) {
                        return FDMTBoxScaleRound<TOut, P>{.r = weights.round[s],
                                                          .unit = weights.unit};
                    });
                finish(wf);
            };
            if (round_bytes <= 1) {
                run.template operator()<uint8_t>();
            } else if (round_bytes <= 2) {
                run.template operator()<uint16_t>();
            } else if (round_bytes <= 4) {
                run.template operator()<uint32_t>();
            } else {
                run.template operator()<uint64_t>();
            }
            return;
        }
        if constexpr (std::is_same_v<TOut, float>) {
            const auto* wf = input.row<float>(ch.i_sub, nsamps, nullptr);
            fdmt_init_subband_box<Mode>(
                wf, ch.hist_sub, buf_base, dt_grid, ch.hist_len, nsamps,
                ch.stride, reinterpret_cast<float*>(ch.acc_row),
                [&](SizeType s) {
                    return FDMTBoxScaleF32{.w = weights.w_f32[s]};
                });
            finish(wf);
        }
        return;
    } else {
        // No smearing: each row is an independent shift of the raw
        // waterfall by |dt| samples. Integer state is fed integer samples.
        using TS = std::conditional_t<std::is_integral_v<TOut>, TOut, float>;
        const auto* wf =
            input.row<TS>(ch.i_sub, nsamps, reinterpret_cast<TS*>(ch.unpack));
        for (SizeType i_dt = 0; i_dt < dt_grid.size(); ++i_dt) {
            const auto s = static_cast<SizeType>(std::abs(dt_grid[i_dt]));
            TOut* row    = buf_base + (i_dt * ch.stride);
            if constexpr (Mode == FDMTMode::kValid) {
                fdmt_init_valid_row0_shift(wf, ch.hist_sub, row, s, ch.hist_len,
                                           nsamps);
            } else {
                fdmt_init_impl_row_shift<Mode == FDMTMode::kRoll>(wf, row, s,
                                                                  nsamps);
            }
        }
        finish(wf);
    }
}

} // namespace

void FDMTCpuEngine::execute(std::span<const float> waterfall,
                            std::span<float> dmt) {
    if (m_skew) {
        m_skew->execute(waterfall, dmt);
        return;
    }
    run_block(float_input(waterfall), m_float_levels, dmt);
}

void FDMTCpuEngine::execute(std::span<const uint8_t> waterfall_packed,
                            SizeType nbits,
                            std::span<float> dmt) {
    if (m_skew) {
        m_skew->execute(waterfall_packed, nbits, dmt);
        return;
    }
    const auto input = packed_input(waterfall_packed, nbits);
    run_block(input, packed_levels(nbits), dmt);
}

void FDMTCpuEngine::execute_search(std::span<const float> waterfall,
                                   detail::FDMTSearchSink& sink) {
    if (m_skew) {
        m_skew->execute_search(waterfall, sink);
        return;
    }
    const auto input = float_input(waterfall);
    check_previous_block_finished();
    m_is_initialized    = false;
    const SizeType root = m_plan.get_niters();
    if (m_mode != FDMTMode::kValid || m_fuse_levels >= root) {
        throw std::logic_error(
            "FDMT search requires valid mode and a separate root");
    }
    for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
        start_chunk(input, m_float_levels, {}, t0,
                    std::min(m_chunk, m_plan.get_nsamps() - t0));
        SizeType level = m_fuse_levels;
        if (level > 0) {
            initialise_fused(level);
        } else {
            initialise();
        }
        const auto consume = [&](SizeType l, SizeType source) {
            sink.consume(
                l,
                {
                    .state       = level_ptr<float>(source, 0),
                    .prev_state  = l > 0 ? level_ptr<float>(l - 1, 0) : nullptr,
                    .history     = m_tree_history.data(),
                    .row_stride  = m_row_len[source],
                    .beam_stride = m_levels[source].beam_stride / sizeof(float),
                    .history_stride = m_plan.get_tree_history_size(),
                    .nsamps         = m_active,
                    .sample_begin   = m_stream_pos,
                    .history_ring   = true,
                });
        };
        const auto advance_op_history = [&](SizeType l, SizeType k_begin) {
            if (k_begin >= m_ops[l].size()) {
                return;
            }
            for (SizeType b = 0; b < m_nbeams; ++b) {
                const float* state = level_ptr<float>(l - 1, b);
                float* hist =
                    m_tree_history.data() + b * m_plan.get_tree_history_size();
                // Each coordinate owns a disjoint ring.
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                for (SizeType k = k_begin; k < m_ops[l].size(); ++k) {
                    const auto& op = m_ops[l][k];
                    if (op.head == SIZE_MAX || op.delay == 0) {
                        continue;
                    }
                    const SizeType keep = std::min(op.delay, m_active);
                    ring_store(hist + op.hist, op.delay,
                               m_stream_pos + m_active - keep,
                               state + op.head + m_active - keep, keep);
                }
            }
        };
        consume(level, level);
        for (++level; level < root; ++level) {
            execute_iter(level, /*regular_only=*/true);
            consume(level, level);
            advance_op_history(level, detail::fdmt_regular_coords(
                                          m_plan.get_container(), level));
        }
        consume(root, root - 1);
        // Every root coordinate owns its history slot. Save only the
        // trailing min(delay, chunk) input samples, after all readers.
        advance_op_history(root, 0);
        end_chunk();
    }
}

void FDMTCpuEngine::execute_search(std::span<const uint8_t> waterfall,
                                   SizeType nbits,
                                   detail::FDMTSearchSink& sink) {
    if (nbits != m_search_nbits || nbits == 32) {
        throw std::invalid_argument(
            "FDMT integer search: input bit depth mismatch");
    }
    if (m_skew) {
        m_skew->execute_search(waterfall, nbits, sink);
        return;
    }
    if (m_search_u32) {
        execute_search_typed<uint32_t>(waterfall, nbits, sink,
                                       m_integer32_history.data());
    } else {
        execute_search_typed<uint64_t>(waterfall, nbits, sink,
                                       m_integer_history.data());
    }
}

template <typename TInt>
void FDMTCpuEngine::execute_search_typed(std::span<const uint8_t> waterfall,
                                         SizeType nbits,
                                         detail::FDMTSearchSink& sink,
                                         TInt* history) {
    // The sink reads uint32_t state through the integer32_* fields.
    const auto as_u64 = [](const TInt* p) -> const uint64_t* {
        if constexpr (std::is_same_v<TInt, uint64_t>) {
            return p;
        } else {
            return nullptr;
        }
    };
    const auto as_u32 = [](const TInt* p) -> const uint32_t* {
        if constexpr (std::is_same_v<TInt, uint32_t>) {
            return p;
        } else {
            return nullptr;
        }
    };
    const auto input    = packed_input(waterfall, nbits);
    const SizeType root = m_plan.get_niters();
    const SizeType half = m_nbeams * m_half_elems * sizeof(TInt);
    for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
        m_input    = input;
        m_input.t0 = t0;
        m_active   = std::min(m_chunk, m_plan.get_nsamps() - t0);
        for (SizeType l = 0; l < root; ++l) {
            m_levels[l] = {.base = m_state_internal.data() + (l % 2) * half,
                           .type =
                               Elem::kF32, // private path uses level_ptr<TInt>
                           .beam_stride = m_half_elems * sizeof(TInt)};
        }
        const auto& grids = m_plan.get_container().grids[0];
        for (SizeType b = 0; b < m_nbeams; ++b) {
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
            for (SizeType c = 0; c < grids.size(); ++c) {
                init_channel<TInt>(beam_input(b),
                                   level_ptr<TInt>(0, b) +
                                       grids[c].coord_offset * m_row_len[0],
                                   c, b, fdmt_thread_id());
            }
        }
        for (SizeType l = 0; l <= root; ++l) {
            if (l > 0 && l < root) {
                const auto reg_coords =
                    detail::fdmt_regular_coords(m_plan.get_container(), l);
                for (SizeType b = 0; b < m_nbeams; ++b) {
                    const auto* in = level_ptr<TInt>(l - 1, b);
                    auto* out      = level_ptr<TInt>(l, b);
                    const auto* hist =
                        history + b * m_plan.get_tree_history_size();
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                    for (SizeType k = 0; k < reg_coords; ++k) {
                        const auto& op   = m_ops[l][k];
                        const TInt* tail = in + op.tail;
                        TInt* dst        = out + op.out;
                        if (op.head == SIZE_MAX) {
                            std::copy_n(tail, m_active, dst);
                            continue;
                        }
                        // Samples before the delay read the ring (slot
                        // = stream position mod delay, wrapping once);
                        // the rest read the head row.
                        const SizeType split = std::min(op.delay, m_active);
                        if (split > 0) {
                            const TInt* ring = hist + op.hist;
                            SizeType slot =
                                static_cast<SizeType>(m_stream_pos % op.delay);
                            for (SizeType t = 0; t < split; ++t) {
                                dst[t] = tail[t] + ring[slot];
                                slot   = slot + 1 == op.delay ? 0 : slot + 1;
                            }
                        }
                        // Sample t >= split reads head row sample
                        // t - delay (split == delay here, so the head
                        // pointer starts at the row). Streamed, like
                        // the transform's levels (fdmt_stream_merge).
                        if (split < m_active) {
                            fdmt_stream_merge<TInt>(tail + split, in + op.head,
                                                    dst + split,
                                                    m_active - split, 1.0F);
                            // Before the loop's barrier, on this thread.
                            fdmt_stream_fence();
                        }
                    }
                }
            }
            const SizeType source = l == root ? root - 1 : l;
            sink.consume(
                l, {.row_stride     = m_row_len[source],
                    .beam_stride    = m_half_elems,
                    .history_stride = m_plan.get_tree_history_size(),
                    .nsamps         = m_active,
                    .sample_begin   = m_stream_pos,
                    .history_ring   = true,
                    .integer_state  = as_u64(level_ptr<TInt>(source, 0)),
                    .integer_prev_state =
                        l > 0 ? as_u64(level_ptr<TInt>(l - 1, 0)) : nullptr,
                    .integer_history = as_u64(history),
                    .integer32_state = as_u32(level_ptr<TInt>(source, 0)),
                    .integer32_prev_state =
                        l > 0 ? as_u32(level_ptr<TInt>(l - 1, 0)) : nullptr,
                    .integer32_history = as_u32(history)});
            // Advance only after the sink and the merge have read the
            // old history. Each coordinate owns a disjoint ring.
            if (l > 0) {
                for (SizeType b = 0; b < m_nbeams; ++b) {
                    const auto* in = level_ptr<TInt>(l - 1, b);
                    auto* hist = history + b * m_plan.get_tree_history_size();
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                    for (SizeType k = 0; k < m_ops[l].size(); ++k) {
                        const auto& op = m_ops[l][k];
                        if (op.head == SIZE_MAX || op.delay == 0) {
                            continue;
                        }
                        const SizeType keep = std::min(op.delay, m_active);
                        ring_store(hist + op.hist, op.delay,
                                   m_stream_pos + m_active - keep,
                                   in + op.head + m_active - keep, keep);
                    }
                }
            }
        }
        m_stream_pos += m_active;
    }
}

FDMTInput FDMTCpuEngine::float_input(std::span<const float> waterfall) {
    const auto nchans = m_plan.get_nchans();
    const auto nsamps = m_plan.get_nsamps();
    if (waterfall.size() != m_nbeams * nchans * nsamps) {
        throw std::invalid_argument(
            fmt::format("FDMT: Invalid size of waterfall. Expected {}, got {}",
                        m_nbeams * nchans * nsamps, waterfall.size()));
    }
    m_input_beam_stride = nchans * nsamps;
    return {.f32 = waterfall.data(), .row_samps = nsamps};
}

FDMTInput FDMTCpuEngine::packed_input(std::span<const uint8_t> packed,
                                      SizeType nbits) {
    if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 && nbits != 16) {
        throw std::invalid_argument(
            fmt::format("FDMT: nbits={} must be one of 1, 2, 4, 8, 16", nbits));
    }
    const auto nchans    = m_plan.get_nchans();
    const auto nsamps    = m_plan.get_nsamps();
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    if (packed.size() != m_nbeams * nchans * row_bytes) {
        throw std::invalid_argument(
            fmt::format("FDMT: Invalid size of packed waterfall (nbits={}). "
                        "Expected {} bytes, got {}",
                        nbits, m_nbeams * nchans * row_bytes, packed.size()));
    }
    m_input_beam_stride = nchans * row_bytes;
    return {.packed    = packed.data(),
            .nbits     = nbits,
            .row_bytes = row_bytes,
            .row_samps = nsamps};
}

void FDMTCpuEngine::check_dmt_size(std::span<float> dmt) const {
    const auto need = m_nbeams * m_plan.get_dmt_size();
    if (dmt.size() < need) {
        throw std::invalid_argument(fmt::format(
            "FDMT: Invalid size of dmt. Expected at least {} (nbeams * "
            "plan.get_dmt_size()), got {}",
            need, dmt.size()));
    }
}

void FDMTCpuEngine::check_previous_block_finished() const {
    if (m_mode == FDMTMode::kValid && m_is_initialized && !is_finished()) {
        throw std::logic_error(fmt::format(
            "FDMT: the previous block's stepper run stopped at level "
            "{} of {}. In mode='valid' every block must be finalized "
            "before the next reset()/execute(), or its cross-block "
            "history is lost. Call finalize(), or reset_history() to "
            "abandon the stream.",
            m_current_level, total_levels() - 1));
    }
}

void FDMTCpuEngine::layout_levels(const std::vector<Elem>& types,
                                  std::span<float> dmt,
                                  SizeType t0) {
    const SizeType niters = m_plan.get_niters();
    const SizeType half   = m_nbeams * m_half_elems * sizeof(float);
    for (SizeType l = 0; l < niters; ++l) {
        const SizeType elem = (types[l] == Elem::kF32)   ? sizeof(float)
                              : (types[l] == Elem::kU16) ? sizeof(uint16_t)
                                                         : sizeof(uint8_t);
        m_levels[l]         = {
            .base        = m_state_internal.data() + ((l % 2) * half),
            .type        = types[l],
            .beam_stride = m_half_elems * elem,
        };
    }
    m_levels[niters] = {
        .base = reinterpret_cast<std::byte*>(dmt.empty() ? nullptr
                                                         : dmt.data() + t0),
        .type = Elem::kF32,
        .beam_stride = m_plan.get_dmt_size() * sizeof(float),
    };
}

void FDMTCpuEngine::start_chunk(const FDMTInput& input,
                                const std::vector<Elem>& types,
                                std::span<float> dmt,
                                SizeType t0,
                                SizeType n) {
    m_input    = input;
    m_input.t0 = t0;
    m_active   = n;
    layout_levels(types, dmt, t0);
}

void FDMTCpuEngine::run_block(const FDMTInput& input,
                              const std::vector<Elem>& types,
                              std::span<float> dmt) {
    check_dmt_size(dmt);
    check_previous_block_finished();
    m_is_initialized      = false;
    const SizeType nsamps = m_plan.get_nsamps();
    const SizeType niters = m_plan.get_niters();
    for (SizeType t0 = 0; t0 < nsamps; t0 += m_chunk) {
        start_chunk(input, types, dmt, t0, std::min(m_chunk, nsamps - t0));
        SizeType level = 0;
        if (m_fuse_levels > 0) {
            initialise_fused(m_fuse_levels);
            level = m_fuse_levels;
        } else {
            initialise();
        }
        for (SizeType l = level + 1; l <= niters; ++l) {
            execute_iter(l);
        }
        if (m_fuse_levels == niters) {
            end_chunk();
        }
    }
}

FDMTInput FDMTCpuEngine::beam_input(SizeType beam) const noexcept {
    FDMTInput input_b = m_input;
    if (input_b.packed != nullptr) {
        input_b.packed += beam * m_input_beam_stride;
    } else {
        input_b.f32 += beam * m_input_beam_stride;
    }
    return input_b;
}

void FDMTCpuEngine::initialise() {
    const auto& grids_init = m_plan.get_container().grids[0];
    const auto nsubs       = grids_init.size();
    const auto len         = m_row_len[0];
    with_elem(m_levels[0].type, [&]<typename T>() {
        for (SizeType b = 0; b < m_nbeams; ++b) {
            const FDMTInput input_b = beam_input(b);
            T* init_buffer_b        = level_ptr<T>(0, b);
#pragma omp parallel num_threads(m_ws.nthreads)
            {
                const SizeType tid = fdmt_thread_id();
#pragma omp for
                for (SizeType c = 0; c < nsubs; ++c) {
                    init_channel(input_b,
                                 init_buffer_b +
                                     (grids_init[c].coord_offset * len),
                                 c, b, tid);
                }
            }
        }
    });
}

template <typename T>
void FDMTCpuEngine::init_channel(const FDMTInput& input_b,
                                 T* rows,
                                 SizeType c,
                                 SizeType beam,
                                 SizeType tid,
                                 SizeType nsamps,
                                 SizeType stride) {
    const auto& pc      = m_plan.get_container();
    const auto hist_len = pc.state_shape[0].dt_max;
    const bool valid    = m_mode == FDMTMode::kValid;
    if (nsamps == 0) {
        nsamps = valid ? m_active : pc.state_shape[0].nsamps;
        stride = m_row_len[0];
    }
    const FDMTLevel0Channel ch{
        .i_sub    = c,
        .dt_grid  = &pc.grids[0][c].dt_grid,
        .hist_sub = valid ? m_history.data() +
                                (beam * m_plan.get_history_size()) +
                                (c * hist_len)
                          : nullptr,
        .hist_len = hist_len,
        .nsamps   = nsamps,
        .stride   = stride,
        .unpack   = m_ws.slice(tid),
        .acc_row  = m_ws.slice(tid) + m_ws.row_bytes,
    };
    const SizeType round_bytes =
        m_box_round_bytes[input_b.packed != nullptr ? input_b.nbits
                                                    : SizeType{0}];
    const auto go = [&]<FDMTMode Mode>() {
        if (m_use_box_smearing) {
            fdmt_init_channel<Mode, true>(input_b, rows, ch, m_box_weights,
                                          round_bytes);
        } else {
            fdmt_init_channel<Mode, false>(input_b, rows, ch, m_box_weights,
                                           round_bytes);
        }
    };
    if (m_mode == FDMTMode::kFull) {
        go.template operator()<FDMTMode::kFull>();
    } else if (m_mode == FDMTMode::kRoll) {
        go.template operator()<FDMTMode::kRoll>();
    } else {
        go.template operator()<FDMTMode::kValid>();
    }
}

template <FDMTMode Mode, typename TIn, typename TOut>
void FDMTCpuEngine::fused_merge_group(const TIn* __restrict__ in,
                                      SizeType in_base,
                                      TOut* __restrict__ out,
                                      SizeType out_base,
                                      bool to_level,
                                      float* __restrict__ hist_b,
                                      SizeType l,
                                      SizeType c_begin,
                                      SizeType c_end) const noexcept {
    const auto len  = row_lengths();
    const TIn* in0  = in - (in_base * m_row_len[l - 1]);
    TOut* out0      = to_level ? out : out - (out_base * m_row_len[l]);
    const auto& ops = m_ops[l];
    for (SizeType k = c_begin; k < c_end; ++k) {
        const auto& op = ops[k];
        if (op.head == SIZE_MAX) {
            fdmt_copy_op(in0, out0, op, len, m_int_unit);
        } else {
            fdmt_merge_op<Mode>(in0, out0, hist_b, op, len, m_int_unit,
                                m_stream_pos);
        }
    }
}

template <FDMTMode Mode>
void FDMTCpuEngine::fused_merge_dispatch(const std::byte* in,
                                         SizeType in_base,
                                         std::byte* out,
                                         SizeType out_base,
                                         bool to_level,
                                         float* hist_b,
                                         SizeType l,
                                         SizeType c_begin,
                                         SizeType c_end) const noexcept {
    with_elem(m_levels[l - 1].type, [&]<typename TIn>() {
        with_elem(m_levels[l].type, [&]<typename TOut>() {
            if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                          !(std::is_floating_point_v<TIn> &&
                            !std::is_floating_point_v<TOut>)) {
                fused_merge_group<Mode>(reinterpret_cast<const TIn*>(in),
                                        in_base, reinterpret_cast<TOut*>(out),
                                        out_base, to_level, hist_b, l, c_begin,
                                        c_end);
            }
        });
    });
}

template <typename T0>
void FDMTCpuEngine::fused_init_group(const FDMTInput& input_b,
                                     T0* __restrict__ buf,
                                     SizeType base,
                                     SizeType ch_begin,
                                     SizeType ch_end,
                                     SizeType beam,
                                     SizeType tid) {
    const auto& grids = m_plan.get_container().grids[0];
    const auto len    = m_row_len[0];
    for (SizeType c = ch_begin; c < ch_end; ++c) {
        init_channel(input_b, buf + ((grids[c].coord_offset - base) * len), c,
                     beam, tid);
    }
}

void FDMTCpuEngine::initialise_fused(SizeType fuse) {
    if (m_fuse_tile > 0) {
        initialise_fused_tiled(fuse);
        return;
    }
    const auto& pc              = m_plan.get_container();
    const SizeType ngroups      = pc.state_shape[fuse].nchans;
    const auto tree_hist_stride = m_plan.get_tree_history_size();

    for (SizeType b = 0; b < m_nbeams; ++b) {
        const FDMTInput input_b = beam_input(b);
        float* hist_b = (m_mode == FDMTMode::kValid)
                            ? m_tree_history.data() + (b * tree_hist_stride)
                            : nullptr;
        std::byte* final_out = level_ptr<std::byte>(fuse, b);
#pragma omp parallel num_threads(m_ws.nthreads)
        {
            const SizeType tid = fdmt_thread_id();
            std::byte* buf_a   = m_ws.fused_a(tid);
            std::byte* buf_b   = m_ws.fused_b(tid);
            // Groups at the bottom of the band carry more dt rows than
            // those at the top, so hand them out dynamically.
#pragma omp for schedule(dynamic)
            for (SizeType g = 0; g < ngroups; ++g) {
                const auto channels = detail::fdmt_group_channels(pc, fuse, g);
                const SizeType ch_begin = channels.first;
                const SizeType ch_end   = channels.second;
                const SizeType c0b      = fused_coord_range(0, g, fuse).first;
                std::byte* cur          = buf_a;
                std::byte* nxt          = buf_b;
                with_elem(m_levels[0].type, [&]<typename T0>() {
                    fused_init_group<T0>(input_b, reinterpret_cast<T0*>(cur),
                                         c0b, ch_begin, ch_end, b, tid);
                });
                SizeType in_base = c0b;
                for (SizeType l = 1; l <= fuse; ++l) {
                    const auto l_range = fused_coord_range(l, g, fuse);
                    const SizeType cb  = l_range.first;
                    const SizeType ce  = l_range.second;
                    const bool last    = (l == fuse);
                    std::byte* out     = last ? final_out : nxt;
                    if (m_mode == FDMTMode::kFull) {
                        fused_merge_dispatch<FDMTMode::kFull>(
                            cur, in_base, out, cb, last, hist_b, l, cb, ce);
                    } else if (m_mode == FDMTMode::kRoll) {
                        fused_merge_dispatch<FDMTMode::kRoll>(
                            cur, in_base, out, cb, last, hist_b, l, cb, ce);
                    } else {
                        fused_merge_dispatch<FDMTMode::kValid>(
                            cur, in_base, out, cb, last, hist_b, l, cb, ce);
                    }
                    std::swap(cur, nxt);
                    in_base = cb;
                }
            }
        }
    }
}

template <typename TIn, typename TOut>
void FDMTCpuEngine::fused_merge_tile(const TIn* __restrict__ in,
                                     SizeType in_base,
                                     TOut* __restrict__ out,
                                     SizeType out_base,
                                     bool to_level,
                                     float* __restrict__ hist_b,
                                     SizeType l,
                                     SizeType c_begin,
                                     SizeType c_end,
                                     SizeType ts,
                                     SizeType n) noexcept {
    const SizeType tile = m_fuse_tile;
    const auto& ops     = m_ops[l];
    const SizeType pos  = m_stream_pos + ts;
    SizeType* ring_pos  = m_tile_ring_pos[l].data();
    for (SizeType k = c_begin; k < c_end; ++k) {
        const auto& op  = ops[k];
        const TIn* tail = in + ((op.tail_row - in_base) * tile);
        TOut* o         = to_level ? out + op.out + ts
                                   : out + ((op.out_row - out_base) * tile);
        if (op.head == SIZE_MAX) {
            fdmt_convert_copy_unit(tail, n, o, m_int_unit);
            continue;
        }
        const TIn* head  = in + ((op.head_row - in_base) * tile);
        const SizeType d = op.delay;
        // Level F goes to main memory: stream it (see fdmt_stream_add).
        const auto merge = [&](SizeType dd, float* hist, SizeType rp0) {
            if (to_level) {
                offset_add<FDMTMode::kValid, TIn, TOut, true>(
                    tail, n, head, n, o, n, dd, hist, m_int_unit, pos, rp0);
            } else {
                offset_add<FDMTMode::kValid, TIn, TOut, false>(
                    tail, n, head, n, o, n, dd, hist, m_int_unit, pos, rp0);
            }
        };
        if (d == 0) {
            merge(0, nullptr, SIZE_MAX);
            continue;
        }
        // pos % d, carried from the previous tile without a division
        // whenever the delay exceeds the tile.
        SizeType& rp = ring_pos[k];
        if (ts == 0) {
            rp = m_stream_pos % d;
        }
        merge(d, hist_b + op.hist, rp);
        rp += n;
        if (rp >= d) {
            rp -= d;
            if (rp >= d) {
                rp %= d;
            }
        }
    }
}

void FDMTCpuEngine::fused_tile_dispatch(const std::byte* in,
                                        SizeType in_base,
                                        std::byte* out,
                                        bool to_level,
                                        float* hist_b,
                                        SizeType l,
                                        SizeType c_begin,
                                        SizeType c_end,
                                        SizeType ts,
                                        SizeType n) noexcept {
    with_elem(m_levels[l - 1].type, [&]<typename TIn>() {
        with_elem(m_levels[l].type, [&]<typename TOut>() {
            if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                          !(std::is_floating_point_v<TIn> &&
                            !std::is_floating_point_v<TOut>)) {
                fused_merge_tile(reinterpret_cast<const TIn*>(in), in_base,
                                 reinterpret_cast<TOut*>(out), c_begin,
                                 to_level, hist_b, l, c_begin, c_end, ts, n);
            }
        });
    });
}

void FDMTCpuEngine::initialise_fused_tiled(SizeType fuse) {
    const auto& pc              = m_plan.get_container();
    const SizeType ngroups      = pc.state_shape[fuse].nchans;
    const auto tree_hist_stride = m_plan.get_tree_history_size();
    const auto& grids0          = pc.grids[0];
    const SizeType tile         = m_fuse_tile;
    const SizeType n_active     = m_active;

    for (SizeType b = 0; b < m_nbeams; ++b) {
        const FDMTInput input_b = beam_input(b);
        float* hist_b        = m_tree_history.data() + (b * tree_hist_stride);
        std::byte* final_out = level_ptr<std::byte>(fuse, b);
#pragma omp parallel num_threads(m_ws.nthreads)
        {
            const SizeType tid = fdmt_thread_id();
#pragma omp for schedule(dynamic) nowait
            for (SizeType g = 0; g < ngroups; ++g) {
                const auto channels = detail::fdmt_group_channels(pc, fuse, g);
                const SizeType ch_begin = channels.first;
                const SizeType ch_end   = channels.second;
                const SizeType c0b      = fused_coord_range(0, g, fuse).first;
                for (SizeType ts = 0; ts < n_active; ts += tile) {
                    const SizeType n  = std::min(tile, n_active - ts);
                    FDMTInput input_t = input_b;
                    input_t.t0 += ts;
                    std::byte* cur = m_ws.fused_a(tid);
                    std::byte* nxt = m_ws.fused_b(tid);
                    with_elem(m_levels[0].type, [&]<typename T0>() {
                        auto* rows0 = reinterpret_cast<T0*>(cur);
                        for (SizeType c = ch_begin; c < ch_end; ++c) {
                            init_channel(
                                input_t,
                                rows0 + ((grids0[c].coord_offset - c0b) * tile),
                                c, b, tid, n, tile);
                        }
                    });
                    SizeType in_base = c0b;
                    for (SizeType l = 1; l <= fuse; ++l) {
                        const auto [cb, ce] = fused_coord_range(l, g, fuse);
                        const bool last     = (l == fuse);
                        std::byte* out      = last ? final_out : nxt;
                        fused_tile_dispatch(cur, in_base, out, last, hist_b, l,
                                            cb, ce, ts, n);
                        std::swap(cur, nxt);
                        in_base = cb;
                    }
                }
            }
            fdmt_stream_fence();
        }
    }
}

template <FDMTMode Mode, typename TIn, typename TOut>
void FDMTCpuEngine::execute_iter_typed(SizeType i_iter, bool regular_only) {
    const auto hist_beam_stride = m_plan.get_tree_history_size();
    const auto ops_span = regular_only
                              ? std::span<const FDMTMergeOp>(m_ops[i_iter])
                                    .first(detail::fdmt_regular_coords(
                                        m_plan.get_container(), i_iter))
                              : std::span<const FDMTMergeOp>(m_ops[i_iter]);
    for (SizeType b = 0; b < m_nbeams; ++b) {
        float* __restrict__ hist_b =
            (Mode == FDMTMode::kValid)
                ? m_tree_history.data() + (b * hist_beam_stride)
                : nullptr;
        fdmt_iter<Mode>(level_ptr<TIn>(i_iter - 1, b),
                        level_ptr<TOut>(i_iter, b), hist_b, ops_span,
                        row_lengths(), m_int_unit, m_stream_pos, m_nthreads);
    }
}

void FDMTCpuEngine::execute_iter(SizeType i_iter, bool regular_only) {
    with_elem(m_levels[i_iter - 1].type, [&]<typename TIn>() {
        with_elem(m_levels[i_iter].type, [&]<typename TOut>() {
            // Levels only ever widen (see int_tree_level_types()).
            if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                          !(std::is_floating_point_v<TIn> &&
                            !std::is_floating_point_v<TOut>)) {
                if (m_mode == FDMTMode::kFull) {
                    execute_iter_typed<FDMTMode::kFull, TIn, TOut>(
                        i_iter, regular_only);
                } else if (m_mode == FDMTMode::kRoll) {
                    execute_iter_typed<FDMTMode::kRoll, TIn, TOut>(
                        i_iter, regular_only);
                } else {
                    execute_iter_typed<FDMTMode::kValid, TIn, TOut>(
                        i_iter, regular_only);
                }
            } else {
                throw std::logic_error(
                    "FDMT: invalid narrowing level transition");
            }
        });
    });
    if (i_iter == m_plan.get_niters()) {
        end_chunk();
    }
}

} // namespace dmt::algorithms::detail::fdmt_cpu
