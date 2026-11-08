#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <omp.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_skew.hpp"
#include "dmt/fdmt_tree.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms {

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
 * @brief A large zero-initialised array for the engine's tree state and
 * tree history: 2 MiB-aligned, with transparent huge pages requested on
 * Linux, and zero-filled by `nthreads` threads with a static schedule.
 *
 * Huge pages: a fused time tile writes short runs of thousands of rows that
 * lie a chunk apart, one 4 KiB page each, so with small pages nearly every
 * run misses the TLB. The parallel fill first-touches the pages across the
 * threads (and so across NUMA nodes) instead of on the constructing thread.
 */
template <typename T> class FDMTLargeArray {
public:
    FDMTLargeArray() = default;
    ~FDMTLargeArray() { std::free(m_data); } // NOLINT
    FDMTLargeArray(const FDMTLargeArray&)            = delete;
    FDMTLargeArray& operator=(const FDMTLargeArray&) = delete;
    FDMTLargeArray(FDMTLargeArray&&)                 = delete;
    FDMTLargeArray& operator=(FDMTLargeArray&&)      = delete;

    void assign(SizeType n, int nthreads) {
        std::free(m_data); // NOLINT
        m_data = nullptr;
        m_size = n;
        if (n == 0) {
            return;
        }
        constexpr SizeType kAlign = SizeType{2} << 20;
        const SizeType bytes =
            (((n * sizeof(T)) + kAlign - 1) / kAlign) * kAlign;
        m_data = static_cast<T*>(std::aligned_alloc(kAlign, bytes)); // NOLINT
        if (m_data == nullptr) {
            throw std::bad_alloc();
        }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(m_data, bytes, MADV_HUGEPAGE); // a hint; failure is harmless
#endif
        auto* raw            = reinterpret_cast<std::byte*>(m_data);
        const SizeType pages = bytes / kAlign;
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < pages; ++i) {
            std::memset(raw + (i * kAlign), 0, kAlign);
        }
    }
    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] SizeType size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] T* begin() noexcept { return m_data; }
    [[nodiscard]] T* end() noexcept { return m_data + m_size; }
    [[nodiscard]] const T* begin() const noexcept { return m_data; }
    [[nodiscard]] const T* end() const noexcept { return m_data + m_size; }

private:
    T* m_data{nullptr};
    SizeType m_size{0};
};

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

/**
 * @brief One merge (or copy) of the engine's tree, from the plan's
 * coordinate: element offsets of the output row and of its tail / head
 * child rows within their levels' buffers (re-laid out for the engine's row
 * length when it executes in chunks), row lengths, the merge delay and the
 * valid-mode tree-history slot.
 */
struct FDMTMergeOp {
    SizeType out;
    SizeType tail;
    SizeType head; ///< unused for a copy
    SizeType nsamps;
    SizeType tail_nsamps;
    SizeType delay;
    SizeType hist;
    // Row (coordinate) indices of out / tail / head within their levels, for
    // the fused time tiles, whose scratch rows have their own length.
    SizeType out_row;
    SizeType tail_row;
    SizeType head_row; ///< unused for a copy
};

/// Row lengths of one call: the ops' own, or `active` samples of every row
/// (a valid-mode chunk, where every row has the same length).
struct FDMTRowLengths {
    SizeType active{0};
    [[nodiscard]] SizeType out(const FDMTMergeOp& op) const noexcept {
        return active != 0 ? active : op.nsamps;
    }
    [[nodiscard]] SizeType tail(const FDMTMergeOp& op) const noexcept {
        return active != 0 ? active : op.tail_nsamps;
    }
};

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

/**
 * @brief Per-thread scratch carved out of one engine-owned buffer allocated
 * at construction (see FDMTCpuEngine::m_workspace), so execute() never
 * allocates. Thread t's slice starts at base + t * stride and holds, at
 * fixed offsets: one unpacked packed-input row (`unpack`, nsamps samples of
 * <= 4 bytes), two running level-0 box rows for mixed-sign dt grids
 * (`rows`), and the level-fusion ping-pong buffers (`fused_a`/`fused_b`).
 * Every parallel region that uses it runs with exactly `nthreads` threads.
 */
struct FDMTWorkspace {
    std::byte* base{nullptr};
    SizeType stride{0};
    SizeType row_bytes{0};   ///< one row of nsamps samples, <= 4 bytes each
    SizeType fused_bytes{0}; ///< one fusion ping-pong buffer
    int nthreads{1};

    [[nodiscard]] std::byte* slice(SizeType tid) const noexcept {
        return base + (tid * stride);
    }
    template <typename T> [[nodiscard]] T* unpack(SizeType tid) const noexcept {
        return reinterpret_cast<T*>(slice(tid));
    }
    template <typename T> [[nodiscard]] T* rows(SizeType tid) const noexcept {
        return reinterpret_cast<T*>(slice(tid) + row_bytes);
    }
    [[nodiscard]] std::byte* fused_a(SizeType tid) const noexcept {
        return slice(tid) + (3 * row_bytes);
    }
    [[nodiscard]] std::byte* fused_b(SizeType tid) const noexcept {
        return fused_a(tid) + fused_bytes;
    }
};

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

/**
 * @brief Level-0 box means (fdmt_box_weights.hpp), indexed by |dt| = s (box
 * length s + 1): the dyadic float weight w(s + 1) for float input, and for
 * packed input the integer rounding recipe on the plan's common scale 2^-F
 * (`unit` = 2^-F).
 */
struct FDMTBoxWeights {
    const float* w_f32{nullptr};
    const detail::FDMTBoxRound* round{nullptr};
    float unit{1.0F};
};

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

/**
 * @brief One beam's level-0 input: either a float waterfall or a packed
 * low-bit one (LSB-first rows of `row_bytes` bytes, see bit_pack_utils.hpp).
 * Packed rows are unpacked one sub-band at a time into a per-thread scratch
 * row, so a copy of the whole waterfall is never materialized.
 */
struct FDMTInput {
    const float* f32{nullptr};
    const uint8_t* packed{nullptr};
    SizeType nbits{32};
    SizeType row_bytes{0}; ///< packed bytes per channel row of the block
    SizeType row_samps{0}; ///< float samples per channel row of the block
    SizeType t0{0};        ///< first sample of the current chunk (packed:
                           ///< a whole number of bytes into the row)

    // Samples [t0, t0 + nsamps) of row `i_sub` as type TS: the float
    // waterfall row itself, or the packed row unpacked into `scratch`
    // (nsamps elements). TS other than float is only requested for packed
    // input.
    template <typename TS>
    [[nodiscard]] const TS* row(SizeType i_sub,
                                SizeType nsamps,
                                TS* __restrict__ scratch) const noexcept {
        if (packed == nullptr) {
            if constexpr (std::is_same_v<TS, float>) {
                return f32 + (i_sub * row_samps) + t0;
            } else {
                return nullptr;
            }
        }
        bit_pack_utils::unpack_row(packed + (i_sub * row_bytes) +
                                       ((t0 * nbits) / 8),
                                   nbits, nsamps, scratch);
        return scratch;
    }
};

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
    const auto finish = [&](const auto* wf) {
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

class FDMTCpuEngine final : public detail::FDMTEngine {
public:
    FDMTCpuEngine(const plans::FDMTPlan& plan,
                  const detail::FDMTEngineConfig& cfg)
        : m_use_box_smearing(cfg.use_box_smearing),
          m_int_tree(cfg.int_tree),
          m_mode(cfg.mode),
          m_nbeams(cfg.nbeams),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_plan(plan) {
        if (m_nbeams == 0) {
            throw std::invalid_argument("FDMT: nbeams must be at least 1");
        }
        m_search_nbits = cfg.search_nbits;
        m_search_upper = cfg.search_upper_levels;
        // Packed search state in uint32_t when the plan bounds every tree
        // value below 2^32 (as the GPU engine): half the bytes per merge.
        m_search_u32 = detail::int_search_fits_u32(m_plan, m_use_box_smearing,
                                                   m_search_nbits);
        m_chunk = detail::fdmt_chunk_nsamps(m_plan, m_mode, m_nbeams,
                                            cfg.chunk_nsamps, m_search_upper);
        if (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0 &&
            (m_plan.get_history_size() + m_plan.get_tree_history_size()) *
                    sizeof(float) >
                detail::kFDMTChunkStateBudget) {
            m_chunk = std::min<SizeType>(1024, m_plan.get_nsamps());
        } else if (m_search_nbits != 32 && !m_search_u32 &&
                   cfg.chunk_nsamps == 0) {
            m_chunk = std::min(m_plan.get_nsamps(),
                               std::max<SizeType>(64, m_chunk / 2));
        }
        const bool skew = choose_skew(cfg);
        allocate_working_memory(cfg.fuse_levels, cfg.max_fuse_levels, skew);
        const SizeType used = get_memory_usage().total();
        if (used > cfg.max_memory_bytes) {
            throw std::invalid_argument(fmt::format(
                "FDMT (cpu): this configuration needs {:.2f} GiB, over "
                "max_memory_bytes ({:.2f} GiB){}",
                static_cast<double>(used) / (1 << 30),
                static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
                skew ? "; use a shorter chunk_nsamps" : ""));
        }
    }

    ~FDMTCpuEngine() override                      = default;
    FDMTCpuEngine(const FDMTCpuEngine&)            = delete;
    FDMTCpuEngine& operator=(const FDMTCpuEngine&) = delete;
    FDMTCpuEngine(FDMTCpuEngine&&)                 = delete;
    FDMTCpuEngine& operator=(FDMTCpuEngine&&)      = delete;

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        if (m_skew) {
            m_skew->execute(waterfall, dmt);
            return;
        }
        run_block(float_input(waterfall), m_float_levels, dmt);
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt) override {
        if (m_skew) {
            m_skew->execute(waterfall_packed, nbits, dmt);
            return;
        }
        const auto input = packed_input(waterfall_packed, nbits);
        run_block(input, packed_levels(nbits), dmt);
    }

    void execute_search(std::span<const float> waterfall,
                        detail::FDMTSearchSink& sink) override {
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
                    l, {
                           .state = level_ptr<float>(source, 0),
                           .prev_state =
                               l > 0 ? level_ptr<float>(l - 1, 0) : nullptr,
                           .history    = m_tree_history.data(),
                           .row_stride = m_row_len[source],
                           .beam_stride =
                               m_levels[source].beam_stride / sizeof(float),
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
                    float* hist        = m_tree_history.data() +
                                         b * m_plan.get_tree_history_size();
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

    void execute_search(std::span<const uint8_t> waterfall,
                        SizeType nbits,
                        detail::FDMTSearchSink& sink) override {
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
    void execute_search_typed(std::span<const uint8_t> waterfall,
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
                m_levels[l] = {
                    .base = m_state_internal.data() + (l % 2) * half,
                    .type = Elem::kF32, // private path uses level_ptr<TInt>
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
                                SizeType slot    = static_cast<SizeType>(
                                    m_stream_pos % op.delay);
                                for (SizeType t = 0; t < split; ++t) {
                                    dst[t] = tail[t] + ring[slot];
                                    slot = slot + 1 == op.delay ? 0 : slot + 1;
                                }
                            }
                            // Sample t >= split reads head row sample
                            // t - delay (split == delay here, so the head
                            // pointer starts at the row). Streamed, like
                            // the transform's levels (fdmt_stream_merge).
                            if (split < m_active) {
                                fdmt_stream_merge<TInt>(
                                    tail + split, in + op.head, dst + split,
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
                        auto* hist =
                            history + b * m_plan.get_tree_history_size();
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

    void reset(std::span<const float> waterfall,
               std::span<float> dmt) override {
        start_stepper(float_input(waterfall), m_float_levels, dmt);
    }

    void reset(std::span<const uint8_t> waterfall_packed,
               SizeType nbits,
               std::span<float> dmt) override {
        const auto input = packed_input(waterfall_packed, nbits);
        start_stepper(input, packed_levels(nbits), dmt);
    }

    [[nodiscard]] SizeType get_fuse_levels() const noexcept override {
        return m_fuse_levels;
    }

    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept override {
        return m_chunk;
    }

    [[nodiscard]] std::string search_path() const override {
        if (m_skew) {
            return fmt::format(
                "skewed direct 0-{}, fused {}-{}", m_skew->get_leaf_level(),
                m_skew->get_leaf_level(), m_skew->get_fuse_level());
        }
        return {};
    }

    [[nodiscard]] std::string summary() const override {
        const auto mem = get_memory_usage();
        const auto mib = [](SizeType b) {
            return static_cast<double>(b) / 1048576.0;
        };
        const auto summary = fmt::format(
            "*** FDMT (cpu) ***\n"
            "Mode: {}, {}, nbeams {}, chunk {} of {} samples, fuse_levels "
            "{}{}, int_tree {}\n"
            "Host memory: plan {:.1f} MiB, state {:.1f} MiB, history {:.1f} "
            "MiB, workspace {:.1f} MiB (total {:.1f} MiB); output buffer "
            "{:.1f} MiB per execute()\n",
            fdmt_mode_to_string(m_mode), fmt::format("{} threads", m_nthreads),
            m_nbeams, m_chunk, m_plan.get_nsamps(), m_fuse_levels,
            (m_fuse_tile > 0) ? fmt::format(" (time tiles of {})", m_fuse_tile)
                              : std::string{},
            m_int_tree ? "on" : "off", mib(mem.plan), mib(mem.state),
            mib(mem.history), mib(mem.workspace), mib(mem.total()),
            mib(mem.output));
        return m_plan.summary() + summary;
    }

    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept override {
        if (m_skew) {
            const auto sm = m_skew->memory_usage();
            return {
                .plan      = m_plan.get_container().get_memory_usage() +
                             m_skew->plan_bytes(),
                .state     = sm.state,
                .history   = sm.ring + sm.rings,
                .workspace = sm.stripes,
                .output    = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
            };
        }
        return {
            .plan      = m_plan.get_container().get_memory_usage() +
                         (m_ops_elems * sizeof(FDMTMergeOp)),
            .state     = m_state_internal.size(),
            .history   = history_state_size() * sizeof(float) +
                         m_integer_history.size() * sizeof(uint64_t) +
                         m_integer32_history.size() * sizeof(uint32_t),
            .workspace = m_workspace.size(),
            .output    = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
        };
    }

    void advance(SizeType levels, Stream /*stream*/) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            execute_iter(next_level);
            m_current_level = next_level;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        const auto& shape = m_plan.get_container().state_shape[m_current_level];
        return {current_level_f32(),
                shape.ncoords * m_row_len[m_current_level]};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c = m_plan.get_container();
        if (subband_idx >= plan_c.grids[m_current_level].size()) {
            throw std::out_of_range(fmt::format(
                "FDMT: Subband index {} out of range (current level has {} "
                "subbands)",
                subband_idx, plan_c.grids[m_current_level].size()));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto len    = m_row_len[m_current_level];
        const auto offset = grid.coord_offset * len;
        const auto count  = grid.ndt * len;
        return FDMTSubbandView{
            .data = std::span<const float>(current_level_f32() + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = len,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
            .chan_begin  = grid.chan_begin,
            .chan_end    = grid.chan_end,
            .staggered   = grid.staggered,
        };
    }

    [[nodiscard]] SizeType current_level() const noexcept override {
        return m_current_level;
    }

    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan.get_niters() + 1;
    }

    [[nodiscard]] SizeType num_subbands() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }

    [[nodiscard]] bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        m_is_initialized = false;
    }

    void reset_history() noexcept override {
        if (m_skew) {
            m_skew->reset_history();
            return;
        }
        std::ranges::fill(m_history, 0.0F);
        std::ranges::fill(m_tree_history, 0.0F);
        std::ranges::fill(m_integer_history, uint64_t{0});
        std::ranges::fill(m_integer32_history, uint32_t{0});
        m_stream_pos = 0;
        // Abandons any unfinished stepper block along with its stream.
        m_is_initialized = false;
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        if (m_skew) {
            return m_skew->history_state_size();
        }
        return m_history.size() + m_tree_history.size();
    }

    // Layout: the level-0 input history, then the tree history of every
    // beam with each coordinate's slot in FIFO order (oldest sample first),
    // whatever the ring position.
    void save_history(std::span<float> out) const override {
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(
                fmt::format("FDMT::save_history: Invalid output size. "
                            "Expected {}, got {}",
                            history_state_size(), out.size()));
        }
        if (m_skew) { // the skewed-window engine's own layout
            m_skew->save_history(out);
            return;
        }
        std::ranges::copy(m_history, out.begin());
        const auto tree = out.subspan(m_history.size());
        for_each_ring(
            [&](SizeType beam_off, SizeType off, SizeType d, SizeType pos) {
                const float* ring = m_tree_history.data() + beam_off + off;
                float* fifo       = tree.data() + beam_off + off;
                std::copy(ring + pos, ring + d, fifo);
                std::copy(ring, ring + pos, fifo + (d - pos));
            });
    }

    void load_history(std::span<const float> in) override {
        if (m_skew) { // either ring layout; the engine checks the size
            m_skew->load_history(in);
            m_is_initialized = false;
            return;
        }
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(
                fmt::format("FDMT::load_history: Invalid input size. "
                            "Expected {}, got {}",
                            history_state_size(), in.size()));
        }
        std::copy_n(in.begin(), m_history.size(), m_history.begin());
        // FIFO order is the ring at stream position 0.
        std::copy_n(in.begin() + static_cast<std::ptrdiff_t>(m_history.size()),
                    m_tree_history.size(), m_tree_history.begin());
        m_stream_pos     = 0;
        m_is_initialized = false;
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    // Storage type of one tree level's state (int_tree).
    using Elem = detail::FDMTLevelType;

    // Where one level's state lives: `base` of beam 0, beams `beam_stride`
    // bytes apart.
    struct LevelBuf {
        std::byte* base{nullptr};
        Elem type{Elem::kF32};
        SizeType beam_stride{0};
    };

    // Everything below is allocated in the constructor (see
    // allocate_working_memory()); execute() and the stepper only index it.
    bool m_use_box_smearing;
    bool m_int_tree;
    FDMTMode m_mode;
    SizeType m_nbeams;
    int m_nthreads;
    const plans::FDMTPlan& m_plan; // owned by the FDMT facade
    // Row length of the tree state: blocks run in chunks of m_chunk samples
    // (valid mode; otherwise one chunk of nsamps).
    SizeType m_chunk{0};
    // Per level: row length in the engine's buffers, and the merge / copy
    // ops by coordinate (levels >= 1; see FDMTMergeOp).
    std::vector<SizeType> m_row_len;
    std::vector<std::vector<FDMTMergeOp>> m_ops;
    SizeType m_ops_elems{0};
    // The two state halves: level l < niters lives in half l % 2 (the root
    // is written straight into the caller's dmt buffer). Byte storage holds
    // float, uint8_t and uint16_t levels alike. Zero-filled.
    FDMTLargeArray<std::byte> m_state_internal;
    SizeType m_half_elems{0}; // elements of one beam's level in a half
    // Valid-mode streaming state, always float: the last dt0_max raw input
    // samples of every channel, and each merge's tree-history ring (see
    // offset_add), at stream position m_stream_pos.
    std::vector<float> m_history;
    FDMTLargeArray<float> m_tree_history;
    SizeType m_stream_pos{0};
    SizeType m_search_nbits{32};
    bool m_search_upper{false};
    bool m_search_u32{false};
    std::unique_ptr<detail::FDMTSkewCpu> m_skew;
    /// Leaf level of the skewed engine (choose_skew; 0: automatic).
    SizeType m_skew_leaf{0};
    FDMTLargeArray<uint64_t> m_integer_history;
    FDMTLargeArray<uint32_t> m_integer32_history;

    // Level-0 box means (use_box_smearing; see fdmt_box_weights.hpp): dyadic
    // float weights, the packed-input rounding recipes on the common scale
    // 2^-F, whether a recipe's product needs 64 bits (by nbits), and the
    // value of one integer state unit, 2^-F (1 without box smearing).
    std::vector<float> m_box_w_f32;
    std::vector<detail::FDMTBoxRound> m_box_round;
    std::array<SizeType, 17> m_box_round_bytes{};
    FDMTBoxWeights m_box_weights;
    float m_int_unit{1.0F};

    SizeType m_fuse_levels{0}; // resolved fuse_levels
    // Valid mode: the fused levels run in time tiles of this many samples
    // (0: one tile per chunk, the whole-chunk fused path).
    SizeType m_fuse_tile{0};
    // Time-tiled fusion: ring position of every fused merge (levels
    // 1..fuse, by op index), advanced tile by tile.
    std::vector<std::vector<SizeType>> m_tile_ring_pos;
    // Per-level storage types: all float, and per nbits for int_tree.
    std::vector<Elem> m_float_levels;
    std::array<std::vector<Elem>, 17> m_int_levels; // indexed by nbits
    // Per-thread scratch (see FDMTWorkspace) and its layout.
    std::vector<std::byte> m_workspace;
    FDMTWorkspace m_ws;

    // Current block / chunk.
    std::vector<LevelBuf> m_levels;
    FDMTInput m_input{};
    SizeType m_input_beam_stride{0};
    SizeType m_active{0}; // samples of the current chunk (valid mode)
    SizeType m_current_level{0};
    bool m_is_initialized{false};

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
            throw std::logic_error(fmt::format(
                "FDMT: level {} is stored as an integer type "
                "(int_tree); construct with int_tree=false to inspect "
                "intermediate levels.",
                m_current_level));
        }
        return level_ptr<const float>(m_current_level, 0);
    }

    // ---------------------------------------------------------------------
    // Inputs and blocks
    // ---------------------------------------------------------------------

    [[nodiscard]] FDMTInput float_input(std::span<const float> waterfall) {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall.size() != m_nbeams * nchans * nsamps) {
            throw std::invalid_argument(fmt::format(
                "FDMT: Invalid size of waterfall. Expected {}, got {}",
                m_nbeams * nchans * nsamps, waterfall.size()));
        }
        m_input_beam_stride = nchans * nsamps;
        return {.f32 = waterfall.data(), .row_samps = nsamps};
    }

    [[nodiscard]] FDMTInput packed_input(std::span<const uint8_t> packed,
                                         SizeType nbits) {
        if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 &&
            nbits != 16) {
            throw std::invalid_argument(fmt::format(
                "FDMT: nbits={} must be one of 1, 2, 4, 8, 16", nbits));
        }
        const auto nchans    = m_plan.get_nchans();
        const auto nsamps    = m_plan.get_nsamps();
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        if (packed.size() != m_nbeams * nchans * row_bytes) {
            throw std::invalid_argument(fmt::format(
                "FDMT: Invalid size of packed waterfall (nbits={}). "
                "Expected {} bytes, got {}",
                nbits, m_nbeams * nchans * row_bytes, packed.size()));
        }
        m_input_beam_stride = nchans * row_bytes;
        return {.packed    = packed.data(),
                .nbits     = nbits,
                .row_bytes = row_bytes,
                .row_samps = nsamps};
    }

    [[nodiscard]] const std::vector<Elem>&
    packed_levels(SizeType nbits) const noexcept {
        return m_int_tree ? m_int_levels[nbits] : m_float_levels;
    }

    void check_dmt_size(std::span<float> dmt) const {
        const auto need = m_nbeams * m_plan.get_dmt_size();
        if (dmt.size() < need) {
            throw std::invalid_argument(fmt::format(
                "FDMT: Invalid size of dmt. Expected at least {} (nbeams * "
                "plan.get_dmt_size()), got {}",
                need, dmt.size()));
        }
    }

    // In "valid" mode every block must reach the root before the next one
    // starts: levels a stepper run never executed would otherwise skip this
    // block's cross-block history update and silently corrupt the stream.
    void check_previous_block_finished() const {
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

    /**
     * Assigns each level's buffer for this block: level l < niters in state
     * half l % 2 (so a merge never reads and writes the same half), the root
     * in the caller's dmt buffer at the chunk's first sample. Integer levels
     * use the halves' leading bytes.
     */
    void layout_levels(const std::vector<Elem>& types,
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

    // One chunk [t0, t0 + n) of the block: input view, level layout, length.
    void start_chunk(const FDMTInput& input,
                     const std::vector<Elem>& types,
                     std::span<float> dmt,
                     SizeType t0,
                     SizeType n) {
        m_input    = input;
        m_input.t0 = t0;
        m_active   = n;
        layout_levels(types, dmt, t0);
    }

    // execute(): the block's chunks, each from level 0 (fused when
    // fuse_levels > 0) to the root.
    void run_block(const FDMTInput& input,
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

    // reset(): one whole block, level by level (inspectable).
    void start_stepper(const FDMTInput& input,
                       const std::vector<Elem>& types,
                       std::span<float> dmt) {
        if (m_chunk < m_plan.get_nsamps()) {
            throw std::logic_error(fmt::format(
                "FDMT: the stepper runs whole blocks, but this engine "
                "executes blocks of {} samples in chunks of {} "
                "(chunk_nsamps); construct with chunk_nsamps >= nsamps to "
                "step through levels.",
                m_plan.get_nsamps(), m_chunk));
        }
        check_dmt_size(dmt);
        check_previous_block_finished();
        start_chunk(input, types, dmt, 0, m_plan.get_nsamps());
        initialise();
        m_current_level  = 0;
        m_is_initialized = true;
    }

    // After the root of a chunk: the stream moves on.
    void end_chunk() noexcept {
        if (m_mode == FDMTMode::kValid) {
            m_stream_pos += m_active;
        }
    }

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    // Automatic fusion depth (kFDMTAutoFuse): the deepest depth whose two
    // per-thread scratch buffers fit max(kFuseBudgetTotal / nthreads,
    // kFuseBudgetPerThread) -- a rule on the plan and thread count only, no
    // hardware detection. Fitted on an ARM64 (NEON) laptop and a Xeon Gold
    // 6348H (4096 channels, dt_max 2048, 4K-64K samples, 1-8 threads): within
    // 5% of the best fixed depth except 4K-sample blocks on the Xeon (6-8.5%).
    // The per-thread floor of 8 MB keeps the same choices there and lets a
    // high-DM plan (1024 channels at 512 us, DM 2000: ~22 rows per channel,
    // 2-8K-sample chunks) fuse at 8 threads, within 1% of its best depth
    // on the ARM64 laptop, where 5 MB fell back to no fusion (+18%).
    static constexpr SizeType kFuseBudgetTotal     = SizeType{36} << 20;
    static constexpr SizeType kFuseBudgetPerThread = SizeType{8} << 20;

    // Valid mode fuses in time tiles instead (initialise_fused_tiled): the
    // deepest depth, then the longest tile of kFuseTiles, whose mean
    // per-group working set (fused_tile_working_set) is at most
    // kFuseTileBudget, with at least kFuseGroupsPerThread groups per thread
    // for load balance -- again from the plan and thread count only. Tiles
    // shorter than 256 samples lose more to per-merge overhead than they
    // gain in locality. Fitted on a Xeon Gold 6348H at 8 threads (1 MiB L2
    // per core): float input, 512 us, 1024 channels to DM 2000 (F=4, T=512,
    // -31% against the untiled F=1) and 4096 channels to DM 1000 (F=7,
    // T=256, -27%), each within 1% of the best (F, T) measured.
    static constexpr std::array<SizeType, 2> kFuseTiles = {512, 256};
    static constexpr SizeType kFuseTileBudget           = SizeType{1792} << 10;
    static constexpr SizeType kFuseGroupsPerThread      = 2;

    static constexpr SizeType round_up_64(SizeType n) noexcept {
        return ((n + 63) / 64) * 64;
    }

    /**
     * Resolves fuse_levels and allocates every buffer execute() and the
     * stepper use: the merge ops, the state halves and histories, per-level
     * type tables, the box means and the per-thread workspace (see
     * FDMTWorkspace). Nothing is allocated after this.
     */
    /// Fuse level requested of the skewed-window engine (0: automatic).
    [[nodiscard]] SizeType skew_fuse_request(SizeType fuse_levels,
                                             SizeType max_fuse_levels) const {
        return (fuse_levels == kFDMTAutoFuse)
                   ? SizeType{0}
                   : std::min({fuse_levels, max_fuse_levels,
                               m_plan.get_max_fuse_levels()});
    }

    /**
     * (chunk, leaf level) candidates of the skewed-window engine, fastest
     * first. Measured on parkes_uwl (6656 channels, 8 threads one per CCD
     * on an EPYC 9654, s per 2.1 s block): float ring K=1 4.96 at chunk
     * 4096 and 7.4 at 2048, K=2 6.6 / 9.4, K=3 8.8 / 11.1, K=4 13.3 /
     * 15.3, K=1 13.6 at 1024 (levels 1..K-1 of a direct leaf cost double
     * running-sum gathers, the binary merges above them are cheap); 8-bit
     * ring K=2 4.08, K=3 4.38, K=4 4.69, K=1 4.77 at 4096, K=4 5.99 and
     * K=2 6.28 at 2048. Leaf 0: the automatic level (<= 4).
     */
    static std::vector<std::pair<SizeType, SizeType>>
    skew_candidates(SizeType ring_elem) {
        if (ring_elem == sizeof(float)) {
            return {{4096, 1}, {4096, 2}, {2048, 1}, {4096, 3}, {2048, 2},
                    {2048, 3}, {4096, 4}, {1024, 1}, {2048, 4}, {1024, 0},
                    {512, 1},  {512, 0},  {256, 1},  {256, 0}};
        }
        return {{4096, 2}, {4096, 3}, {4096, 4}, {4096, 1},
                {2048, 4}, {2048, 2}, {2048, 3}, {2048, 1},
                {1024, 0}, {512, 0},  {256, 0}};
    }

    /**
     * Tree path or skewed-window engine (fdmt_skew.hpp), under the memory
     * budget: the tree path (fastest) when its history and state fit
     * max_memory_bytes, else the skewed engine at the fastest (chunk, leaf
     * level) of skew_candidates() whose estimate fits (an explicit
     * chunk_nsamps is kept, with the fastest leaf level that fits it).
     * Sets m_chunk and m_skew_leaf. Throws when nothing fits.
     */
    bool choose_skew(const detail::FDMTEngineConfig& cfg) {
        const SizeType niters = m_plan.get_niters();
        const SizeType nsamps = m_plan.get_nsamps();
        const SizeType depth =
            m_search_upper ? m_plan.get_subband_depth() : SizeType{0};
        const bool eligible =
            m_mode == FDMTMode::kValid && m_nbeams == 1 && niters >= 5 &&
            cfg.fuse_levels != 0 && cfg.max_fuse_levels >= 4 &&
            // Searched subband levels sit above the leaf and fused levels.
            (!m_search_upper || niters >= depth + 5);
        if (!eligible) {
            return false;
        }
        const SizeType hist_floats =
            m_plan.get_history_size() + m_plan.get_tree_history_size();
        if (cfg.skew_history_bytes !=
            detail::FDMTEngineConfig::kSkewHistoryAuto) {
            m_skew_leaf = cfg.skew_leaf;
            return m_chunk < nsamps &&
                   hist_floats * sizeof(float) > cfg.skew_history_bytes;
        }
        const auto& pc            = m_plan.get_container();
        const bool wide           = m_search_nbits != 32 && !m_search_u32;
        const SizeType state_elem = wide ? sizeof(uint64_t) : sizeof(float);
        const SizeType plan_bytes = pc.get_memory_usage();
        SizeType max_rows         = 0;
        for (SizeType l = 0; l < niters; ++l) {
            max_rows = std::max(
                max_rows, m_search_upper ? detail::fdmt_regular_coords(pc, l)
                                         : pc.state_shape[l].ncoords);
        }
        const SizeType tree_bytes = (hist_floats * state_elem) +
                                    (2 * max_rows * m_chunk * state_elem) +
                                    plan_bytes;
        if (tree_bytes <= cfg.max_memory_bytes) {
            return false;
        }
        const SizeType ring_elem = m_search_nbits == 32   ? sizeof(float)
                                   : m_search_nbits == 16 ? 2
                                                          : 1;
        const SizeType req_fuse =
            skew_fuse_request(cfg.fuse_levels, cfg.max_fuse_levels);
        const auto fits = [&](SizeType c, SizeType leaf) {
            const auto e = detail::estimate_skew_memory(
                m_plan, 4, c, ring_elem, m_nthreads, m_use_box_smearing,
                req_fuse, depth, leaf);
            const SizeType bytes =
                e.ring + ((e.state + e.rings) * state_elem / sizeof(float)) +
                e.stripes + plan_bytes;
            // A tenth of headroom for the engine's merge tables.
            return bytes + (bytes / 10) <= cfg.max_memory_bytes;
        };
        if (cfg.chunk_nsamps != 0) {
            // Explicit chunk: the fastest leaf level that fits it (else the
            // automatic one); checked against the budget once allocated.
            if (m_chunk >= nsamps) {
                return false;
            }
            m_skew_leaf = 0;
            for (const auto& [c, leaf] : skew_candidates(ring_elem)) {
                if (c == m_chunk && leaf != 0 && fits(c, leaf)) {
                    m_skew_leaf = leaf;
                    break;
                }
            }
            return true;
        }
        for (const auto& [c, leaf] : skew_candidates(ring_elem)) {
            if (c < nsamps && fits(c, leaf)) {
                m_chunk     = c;
                m_skew_leaf = leaf;
                return true;
            }
        }
        throw std::invalid_argument(fmt::format(
            "FDMT (cpu): no streaming path fits max_memory_bytes ({:.2f} GiB): "
            "the tree path needs {:.2f} GiB and the skewed-window path more "
            "than the budget at every chunk length",
            static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
            static_cast<double>(tree_bytes) / (1 << 30)));
    }

    void allocate_working_memory(SizeType fuse_levels,
                                 SizeType max_fuse_levels,
                                 bool skew) {
        const SizeType niters = m_plan.get_niters();
        if (skew) {
            const SizeType req_fuse =
                skew_fuse_request(fuse_levels, max_fuse_levels);
            m_skew = std::make_unique<detail::FDMTSkewCpu>(
                m_plan, m_use_box_smearing, m_nthreads, m_chunk, SizeType{4},
                m_search_nbits, req_fuse, m_search_nbits != 32, m_search_u32,
                m_search_upper ? m_plan.get_subband_depth() : SizeType{0},
                m_skew_leaf);
            m_fuse_levels = m_skew->get_fuse_level();
            return;
        }
        max_fuse_levels = (m_search_nbits == 32) ? max_fuse_levels : 0;
        build_merge_ops();
        const SizeType state_elem = (m_search_nbits == 32 || m_search_u32)
                                        ? sizeof(uint32_t)
                                        : sizeof(uint64_t);
        m_state_internal.assign(2 * m_nbeams * m_half_elems * state_elem,
                                m_nthreads);
        if (m_mode == FDMTMode::kValid) {
            m_history.assign(m_nbeams * m_plan.get_history_size(), 0.0F);
            if (m_search_nbits == 32) {
                m_tree_history.assign(m_nbeams * m_plan.get_tree_history_size(),
                                      m_nthreads);
            } else if (m_search_u32) {
                m_integer32_history.assign(
                    m_nbeams * m_plan.get_tree_history_size(), m_nthreads);
            } else {
                m_integer_history.assign(
                    m_nbeams * m_plan.get_tree_history_size(), m_nthreads);
            }
        }
        if (m_use_box_smearing) {
            const auto weights = m_plan.get_box_weights();
            const auto box_max = weights.size(); // L_max
            const auto scale_f = m_plan.get_box_mean_bits();
            const float w_unit = std::ldexp(
                1.0F, -static_cast<int>(m_plan.get_box_fraction_bits()));
            m_int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
            m_box_w_f32.resize(box_max);
            m_box_round.resize(box_max);
            for (SizeType s = 0; s < box_max; ++s) {
                m_box_w_f32[s] = static_cast<float>(weights[s]) * w_unit;
                m_box_round[s] = m_search_nbits == 32
                                     ? detail::box_round(s + 1, scale_f)
                                     : detail::FDMTBoxRound{.mul = weights[s],
                                                            .add = 0,
                                                            .rshift = 0,
                                                            .lshift = 0};
            }
            for (const SizeType nbits : {1, 2, 4, 8, 16}) {
                const auto largest = detail::box_round_max(
                    bit_pack_utils::max_sample_value(nbits), box_max);
                m_box_round_bytes[nbits] = (largest <= 0xFF)         ? 1
                                           : (largest <= 0xFFFF)     ? 2
                                           : (largest <= 0xFFFFFFFF) ? 4
                                                                     : 8;
            }
            if (m_search_nbits != 32) {
                m_box_round_bytes[m_search_nbits] = 8;
            }
            m_box_weights = {.w_f32 = m_box_w_f32.data(),
                             .round = m_box_round.data(),
                             .unit  = m_int_unit};
        }
        m_float_levels.assign(niters + 1, Elem::kF32);
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            m_int_levels[nbits] =
                detail::int_tree_level_types(m_plan, m_use_box_smearing, nbits);
        }
        m_levels.resize(niters + 1);

        if (fuse_levels == kFDMTAutoFuse) {
            m_fuse_levels = auto_fuse_levels();
        } else {
            m_fuse_levels = std::min(fuse_levels, niters);
        }

        m_fuse_levels = std::min(
            {m_fuse_levels, max_fuse_levels, m_plan.get_max_fuse_levels()});

        m_ws.row_bytes = round_up_64(m_row_len[0] * (m_search_nbits == 32
                                                         ? sizeof(float)
                                                         : sizeof(uint64_t)));
        m_fuse_tile    = (m_mode == FDMTMode::kValid && m_fuse_levels > 0)
                             ? fuse_tile_for(m_fuse_levels)
                             : 0;
        if (m_fuse_tile > 0) {
            m_tile_ring_pos.assign(m_fuse_levels + 1, {});
            for (SizeType l = 1; l <= m_fuse_levels; ++l) {
                m_tile_ring_pos[l].assign(m_ops[l].size(), 0);
            }
        }
        m_ws.fused_bytes =
            (m_fuse_levels > 0)
                ? round_up_64(fused_scratch_bytes(m_fuse_levels, m_fuse_tile))
                : 0;
        m_ws.stride   = (3 * m_ws.row_bytes) + (2 * m_ws.fused_bytes);
        m_ws.nthreads = m_nthreads;
        m_workspace.assign(m_nthreads * m_ws.stride, std::byte{0});
        m_ws.base = m_workspace.data();
    }

    /**
     * The engine's merge ops, from the plan's coordinates. Levels below the
     * root are re-laid out for rows of m_row_len[l] samples (the chunk
     * length in valid mode; the plan's own rows otherwise); the root keeps
     * the plan's layout, which is the dmt output's.
     */
    void build_merge_ops() {
        const auto& pc        = m_plan.get_container();
        const SizeType niters = m_plan.get_niters();
        const bool valid      = m_mode == FDMTMode::kValid;
        m_row_len.resize(niters + 1);
        for (SizeType l = 0; l <= niters; ++l) {
            m_row_len[l] =
                (valid && l < niters) ? m_chunk : pc.state_shape[l].nsamps;
        }
        m_half_elems = 0;
        for (SizeType l = 0; l < niters; ++l) {
            const SizeType n = m_search_upper
                                   ? detail::fdmt_regular_coords(pc, l)
                                   : pc.state_shape[l].ncoords;
            m_half_elems     = std::max(m_half_elems, n * m_row_len[l]);
        }
        const auto relayout = [&](SizeType offset, SizeType l) {
            const auto plan_len = pc.state_shape[l].nsamps;
            return (l == niters) ? offset : (offset / plan_len) * m_row_len[l];
        };
        m_ops.assign(niters + 1, {});
        m_ops_elems = 0;
        for (SizeType l = 1; l <= niters; ++l) {
            const auto& coords     = pc.coordinates[l];
            const auto& coords_sum = pc.coordinates_sum[l];
            auto& ops              = m_ops[l];
            ops.reserve(coords.size());
            SizeType j = 0;
            for (const auto& c : coords) {
                const bool is_sum = c.i_coord_head != SIZE_MAX;
                // Only coordinates_sum carries the tree-history slot.
                const auto& src = is_sum ? coords_sum[j++] : c;
                const auto row  = [&](SizeType offset, SizeType lv) {
                    return offset / pc.state_shape[lv].nsamps;
                };
                ops.push_back({
                    .out         = relayout(src.buf_offset, l),
                    .tail        = relayout(src.tail_buf_offset, l - 1),
                    .head        = is_sum ? relayout(src.head_buf_offset, l - 1)
                                          : SIZE_MAX,
                    .nsamps      = src.nsamps,
                    .tail_nsamps = src.tail_nsamps,
                    .delay       = is_sum ? src.delay : 0,
                    .hist        = is_sum ? src.hist_offset : 0,
                    .out_row     = row(src.buf_offset, l),
                    .tail_row    = row(src.tail_buf_offset, l - 1),
                    .head_row    = is_sum ? row(src.head_buf_offset, l - 1) : 0,
                });
            }
            m_ops_elems += ops.size();
        }
    }

    // Calls f(beam_offset, slot_offset, delay, ring_position) for every
    // tree-history ring of every beam.
    template <typename F> void for_each_ring(F&& f) const {
        if (m_tree_history.empty()) {
            return;
        }
        const auto stride = m_plan.get_tree_history_size();
        for (SizeType b = 0; b < m_nbeams; ++b) {
            for (SizeType l = 1; l < m_ops.size(); ++l) {
                for (const auto& op : m_ops[l]) {
                    if (op.head != SIZE_MAX && op.delay > 0) {
                        f(b * stride, op.hist, op.delay,
                          m_stream_pos % op.delay);
                    }
                }
            }
        }
    }

    // Row lengths of the current chunk's merges.
    [[nodiscard]] FDMTRowLengths row_lengths() const noexcept {
        return {.active = (m_mode == FDMTMode::kValid) ? m_active : 0};
    }

    // ---------------------------------------------------------------------
    // Level 0
    // ---------------------------------------------------------------------

    [[nodiscard]] FDMTInput beam_input(SizeType beam) const noexcept {
        FDMTInput input_b = m_input;
        if (input_b.packed != nullptr) {
            input_b.packed += beam * m_input_beam_stride;
        } else {
            input_b.f32 += beam * m_input_beam_stride;
        }
        return input_b;
    }

    // Beam-major layout: beam b's waterfall/state/history all live in
    // disjoint, per-beam-sized slices of the same flat buffers (the plan's
    // coordinate offsets are beam-agnostic and apply identically within each
    // slice), so the beams run one after the other.
    void initialise() {
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

    // Level-0 rows of channel `c` of beam `beam` into `rows`, plus its
    // valid-mode input history, using thread `tid`'s scratch: the current
    // chunk, or (valid mode, `nsamps` > 0) its `nsamps` samples from
    // `input_b.t0` into rows of `stride` samples (a fused time tile).
    template <typename T>
    void init_channel(const FDMTInput& input_b,
                      T* rows,
                      SizeType c,
                      SizeType beam,
                      SizeType tid,
                      SizeType nsamps = 0,
                      SizeType stride = 0) {
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

    // ---------------------------------------------------------------------
    // Level fusion (fuse_levels)
    // ---------------------------------------------------------------------

    // Coordinate range [begin, end) of level `l` owned by fused group `g`:
    // the level-l sub-bands descending from level-F sub-band g (see
    // fdmt_tree.hpp).
    [[nodiscard]] std::pair<SizeType, SizeType>
    fused_coord_range(SizeType l, SizeType g, SizeType fuse) const noexcept {
        return detail::fdmt_group_coords(m_plan.get_container(), fuse, g, l);
    }

    // Merges level-l coordinates [c_begin, c_end) of one fused group. `in`
    // / `out` hold the group's rows starting at coordinates in_base /
    // out_base (out_base 0 with `to_level`: the level's own buffer); the
    // merge itself is the unmodified offset_add / copy used by fdmt_iter,
    // so fused results are bit-identical.
    template <FDMTMode Mode, typename TIn, typename TOut>
    void fused_merge_group(const TIn* __restrict__ in,
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
    void fused_merge_dispatch(const std::byte* in,
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
                    fused_merge_group<Mode>(
                        reinterpret_cast<const TIn*>(in), in_base,
                        reinterpret_cast<TOut*>(out), out_base, to_level,
                        hist_b, l, c_begin, c_end);
                }
            });
        });
    }

    // Level-0 rows of channels [ch_begin, ch_end) into `buf` (rows from
    // coordinate `base`), plus their valid-mode input history -- exactly
    // the per-sub-band work of initialise(), for one fused group.
    template <typename T0>
    void fused_init_group(const FDMTInput& input_b,
                          T0* __restrict__ buf,
                          SizeType base,
                          SizeType ch_begin,
                          SizeType ch_end,
                          SizeType beam,
                          SizeType tid) {
        const auto& grids = m_plan.get_container().grids[0];
        const auto len    = m_row_len[0];
        for (SizeType c = ch_begin; c < ch_end; ++c) {
            init_channel(input_b, buf + ((grids[c].coord_offset - base) * len),
                         c, beam, tid);
        }
    }

    // Largest per-group scratch level buffer for fusion depth `fuse` (4
    // bytes/element covers every storage type); two are live per thread.
    // `tile` > 0: rows of `tile` samples (the time-tiled path).
    [[nodiscard]] SizeType
    fused_scratch_bytes(SizeType fuse, SizeType tile = 0) const noexcept {
        const auto& pc         = m_plan.get_container();
        const SizeType ngroups = pc.state_shape[fuse].nchans;
        SizeType max_bytes     = 0;
        for (SizeType l = 0; l < fuse; ++l) {
            const SizeType len = (tile > 0) ? tile : m_row_len[l];
            for (SizeType g = 0; g < ngroups; ++g) {
                const auto [cb, ce] = fused_coord_range(l, g, fuse);
                max_bytes =
                    std::max(max_bytes, (ce - cb) * len * sizeof(float));
            }
        }
        return max_bytes;
    }

    // Mean over the level-F groups of the bytes one thread touches while it
    // walks a group in tiles of `tile` samples: its two scratch buffers
    // (float-sized) and the tree-history rings of its fused merges.
    [[nodiscard]] double fused_tile_working_set(SizeType fuse,
                                                SizeType tile) const noexcept {
        const auto& pc         = m_plan.get_container();
        const SizeType ngroups = pc.state_shape[fuse].nchans;
        double total           = 0.0;
        for (SizeType g = 0; g < ngroups; ++g) {
            SizeType rows = 0;
            SizeType ring = 0;
            for (SizeType l = 0; l <= fuse; ++l) {
                const auto [cb, ce] = fused_coord_range(l, g, fuse);
                if (l < fuse) {
                    rows = std::max(rows, ce - cb);
                }
                if (l > 0) {
                    for (SizeType k = cb; k < ce; ++k) {
                        ring += m_ops[l][k].delay;
                    }
                }
            }
            total += static_cast<double>((2 * rows * tile) + ring) *
                     static_cast<double>(sizeof(float));
        }
        return total / static_cast<double>(ngroups);
    }

    // Tile length for valid-mode fusion depth `fuse`: the longest of
    // kFuseTiles within kFuseTileBudget (else the shortest), capped at the
    // chunk.
    [[nodiscard]] SizeType fuse_tile_for(SizeType fuse) const noexcept {
        SizeType tile = kFuseTiles.back();
        for (const SizeType t : kFuseTiles) {
            if (fused_tile_working_set(fuse, t) <=
                static_cast<double>(kFuseTileBudget)) {
                tile = t;
                break;
            }
        }
        return std::min(tile, m_chunk);
    }

    // kFDMTAutoFuse (see kFuseBudgetTotal and kFuseTileBudget).
    [[nodiscard]] SizeType auto_fuse_levels() const noexcept {
        if (m_mode == FDMTMode::kValid) {
            const auto& pc   = m_plan.get_container();
            const auto min_g = kFuseGroupsPerThread *
                               static_cast<SizeType>(std::max(m_nthreads, 1));
            SizeType best    = 0;
            for (SizeType f = 1; f <= m_plan.get_niters(); ++f) {
                if (fused_tile_working_set(f, kFuseTiles.back()) >
                    static_cast<double>(kFuseTileBudget)) {
                    break;
                }
                if (f > 1 && pc.state_shape[f].nchans < min_g) {
                    break;
                }
                best = f;
            }
            return best;
        }
        const SizeType budget =
            std::max(kFuseBudgetTotal / std::max<SizeType>(m_nthreads, 1),
                     kFuseBudgetPerThread);
        SizeType best = 0;
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
    void initialise_fused(SizeType fuse) {
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
                    const auto channels =
                        detail::fdmt_group_channels(pc, fuse, g);
                    const SizeType ch_begin = channels.first;
                    const SizeType ch_end   = channels.second;
                    const SizeType c0b = fused_coord_range(0, g, fuse).first;
                    std::byte* cur     = buf_a;
                    std::byte* nxt     = buf_b;
                    with_elem(m_levels[0].type, [&]<typename T0>() {
                        fused_init_group<T0>(input_b,
                                             reinterpret_cast<T0*>(cur), c0b,
                                             ch_begin, ch_end, b, tid);
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

    // Merges level-l coordinates [c_begin, c_end) of one fused group over the
    // time tile [ts, ts + n) of the chunk. `in` / `out` are scratch rows of
    // m_fuse_tile samples starting at coordinates in_base / out_base, or
    // (`to_level`) `out` is the level's own state buffer (rows of the chunk
    // length, written at column ts). Valid mode only: each merge is the
    // unmodified offset_add, whose tree-history ring carries the head
    // samples across tiles exactly as it does across chunks.
    template <typename TIn, typename TOut>
    void fused_merge_tile(const TIn* __restrict__ in,
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

    // fused_merge_tile for the storage types of levels l - 1 and l.
    void fused_tile_dispatch(const std::byte* in,
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
                                     to_level, hist_b, l, c_begin, c_end, ts,
                                     n);
                }
            });
        });
    }

    /**
     * Time-tiled level fusion (valid mode, m_fuse_tile > 0). Like
     * initialise_fused(), one thread owns one level-F channel group at a
     * time, but walks the chunk in tiles of m_fuse_tile samples: level 0 and
     * levels 1..F of the tile live in two scratch buffers of group rows x
     * tile, and level F is written to its state buffer. A merge whose delay
     * exceeds the tile reads its head samples from its tree-history ring,
     * which the previous tile of the same group has just written, so a
     * group's rings stay cache-resident while it walks the chunk. The
     * results equal the untiled path bit for bit (valid mode is independent
     * of the block length).
     */
    void initialise_fused_tiled(SizeType fuse) {
        const auto& pc              = m_plan.get_container();
        const SizeType ngroups      = pc.state_shape[fuse].nchans;
        const auto tree_hist_stride = m_plan.get_tree_history_size();
        const auto& grids0          = pc.grids[0];
        const SizeType tile         = m_fuse_tile;
        const SizeType n_active     = m_active;

        for (SizeType b = 0; b < m_nbeams; ++b) {
            const FDMTInput input_b = beam_input(b);
            float* hist_b = m_tree_history.data() + (b * tree_hist_stride);
            std::byte* final_out = level_ptr<std::byte>(fuse, b);
#pragma omp parallel num_threads(m_ws.nthreads)
            {
                const SizeType tid = fdmt_thread_id();
#pragma omp for schedule(dynamic) nowait
                for (SizeType g = 0; g < ngroups; ++g) {
                    const auto channels =
                        detail::fdmt_group_channels(pc, fuse, g);
                    const SizeType ch_begin = channels.first;
                    const SizeType ch_end   = channels.second;
                    const SizeType c0b = fused_coord_range(0, g, fuse).first;
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
                                    rows0 +
                                        ((grids0[c].coord_offset - c0b) * tile),
                                    c, b, tid, n, tile);
                            }
                        });
                        SizeType in_base = c0b;
                        for (SizeType l = 1; l <= fuse; ++l) {
                            const auto [cb, ce] = fused_coord_range(l, g, fuse);
                            const bool last     = (l == fuse);
                            std::byte* out      = last ? final_out : nxt;
                            fused_tile_dispatch(cur, in_base, out, last, hist_b,
                                                l, cb, ce, ts, n);
                            std::swap(cur, nxt);
                            in_base = cb;
                        }
                    }
                }
                fdmt_stream_fence();
            }
        }
    }

    // ---------------------------------------------------------------------
    // Tree levels
    // ---------------------------------------------------------------------

    template <FDMTMode Mode, typename TIn, typename TOut>
    void execute_iter_typed(SizeType i_iter, bool regular_only = false) {
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
                            row_lengths(), m_int_unit, m_stream_pos,
                            m_nthreads);
        }
    }

    void execute_iter(SizeType i_iter, bool regular_only = false) {
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

}; // End FDMTCpuEngine definition

} // namespace

std::unique_ptr<detail::FDMTEngine>
detail::make_fdmt_cpu(const plans::FDMTPlan& plan,
                      const detail::FDMTEngineConfig& cfg) {
    return std::make_unique<FDMTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
