#include "dmt/algorithms/fdmt.hpp"

#include <cstddef>
#include <format>
#include <stdexcept>
#include <utility>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"

namespace dmt::algorithms {

namespace {

enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

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
 * drift apart.
 */
void fdmt_advance_history_window(const float* __restrict__ new_data,
                                 float* __restrict__ hist,
                                 SizeType nsamps,
                                 SizeType capacity) noexcept {
    if (capacity == 0) {
        return;
    }
    if (nsamps >= capacity) {
        std::copy_n(new_data + nsamps - capacity, capacity, hist);
    } else {
        std::copy(hist + nsamps, hist + capacity, hist);
        std::copy_n(new_data, nsamps, hist + capacity - nsamps);
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
 *   FDMTCoord::hist_offset and FDMTCPU::reset_history()). Each coordinate
 *   owns a disjoint slice of the shared history buffer, so this is safe
 *   across the `#pragma omp for` in fdmt_iter below with no extra locking.
 * - Roll: Cyclic boundary - head wraps around using modulo arithmetic
 *
 * @tparam Mode FDMTMode enum specifying boundary handling behavior
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
 *
 */
template <FDMTMode Mode>
void offset_add(const float* __restrict__ data_tail,
                SizeType size_tail,
                const float* __restrict__ data_head,
                SizeType size_head,
                float* __restrict__ out,
                SizeType size_out,
                SizeType delay_shift,
                float* __restrict__ hist = nullptr) noexcept {
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
        std::copy_n(data_tail, delay_shift, out);

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }

        // Part 3: Head-only region [size_tail, size_tail + nrest)
        const SizeType nrest = std::min(delay_shift, size_out - size_tail);
        if (nrest > 0) {
            std::copy_n(data_head + nsum, nrest, out + size_tail);
        }
        // Part 4: Zero-fill any remaining [size_tail + nrest, size_out)
        const SizeType filled = size_tail + nrest;
        if (filled < size_out) {
            std::fill(out + filled, out + size_out, 0.0F);
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
                out[i] = data_tail[i] + hist[i];
            }
            // Update the size-delay_shift history FIFO with this block's own
            // head samples, so the next call sees a correctly-advanced
            // window regardless of how block size compares to delay_shift.
            fdmt_advance_history_window(data_head, hist, size_head,
                                        delay_shift);
        } else {
            std::copy_n(data_tail, delay_shift, out);
        }

        // Part 2: Overlap region [delay_shift, size_tail), empty once
        // delay_shift >= size_tail.
        const SizeType nsum =
            (delay_shift < size_tail) ? (size_tail - delay_shift) : 0;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }
    } else if constexpr (Mode == FDMTMode::kRoll) {
        // Roll mode: cyclic addition
        assert(size_out == size_tail && "All sizes must be equal in roll mode");
        // Part 1: Wrapped region [0, delay_shift)
        for (SizeType k = 0; k < delay_shift; ++k) {
            out[k] = data_tail[k] + data_head[size_tail - delay_shift + k];
        }

        // Part 2: Overlap region [delay_shift, size_tail)
        const SizeType nsum = size_tail - delay_shift;
#pragma omp simd
        for (SizeType i = 0; i < nsum; ++i) {
            out[delay_shift + i] = data_tail[delay_shift + i] + data_head[i];
        }
    }
}

template <FDMTMode Mode>
void fdmt_iter(const float* __restrict__ state_in,
               float* __restrict__ state_out,
               float* __restrict__ hist_ptr,
               const plans::FDMTCoord* __restrict__ coords_sum_cur,
               const plans::FDMTCoord* __restrict__ coords_copy_cur,
               SizeType ncoords_sum_cur,
               SizeType ncoords_copy_cur) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, hist_ptr, coords_sum_cur, coords_copy_cur,     \
               ncoords_sum_cur, ncoords_copy_cur)
#endif
    {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for nowait
#endif
        for (SizeType i_coord = 0; i_coord < ncoords_sum_cur; ++i_coord) {
            const auto* __restrict__ coord = &coords_sum_cur[i_coord];
            const float* __restrict__ tail = &state_in[coord->tail_buf_offset];
            const float* __restrict__ head = &state_in[coord->head_buf_offset];
            float* __restrict__ out        = &state_out[coord->buf_offset];
            float* __restrict__ hist = (hist_ptr != nullptr && coord->delay > 0)
                                           ? &hist_ptr[coord->hist_offset]
                                           : nullptr;
            offset_add<Mode>(tail, coord->tail_nsamps, head, coord->head_nsamps,
                             out, coord->nsamps, coord->delay, hist);
        }
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
        for (SizeType i_coord = 0; i_coord < ncoords_copy_cur; ++i_coord) {
            const auto* __restrict__ coord = &coords_copy_cur[i_coord];
            const float* __restrict__ tail = &state_in[coord->tail_buf_offset];
            float* __restrict__ out        = &state_out[coord->buf_offset];
            std::copy_n(tail, coord->tail_nsamps, out);
            if (coord->nsamps > coord->tail_nsamps) {
                std::fill(out + coord->tail_nsamps, out + coord->nsamps, 0.0F);
            }
        }
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl_row0(const float* __restrict__ wf_sub,
                         float* __restrict__ buf_row0,
                         SizeType dt_min_sub,
                         SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) =====
    if constexpr (UseBoxSmearing) {
        float running_sum = 0.0F;

        // Initialize sum for isamp=0
        if constexpr (UseRoll) {
            // Sum samples at wrapped indices: [nsamps - dt_min_sub, nsamps)
            // and [0]
            for (SizeType i = 0; i < dt_min_sub; ++i) {
                running_sum += wf_sub[nsamps - dt_min_sub + i];
            }
            running_sum += wf_sub[0];
        } else {
            running_sum = wf_sub[0];
        }
        buf_row0[0] = running_sum;

        // Triangle region [1, dt_min_sub]: window slides into valid data
        for (SizeType isamp = 1; isamp <= dt_min_sub; ++isamp) {
            running_sum += wf_sub[isamp];
            if constexpr (UseRoll) {
                running_sum -= wf_sub[nsamps - dt_min_sub - 1 + isamp];
            }
            buf_row0[isamp] = running_sum;
        }

        // Main region [dt_min_sub + 1, nsamps): fully within bounds
        for (SizeType isamp = dt_min_sub + 1; isamp < nsamps; ++isamp) {
            running_sum += wf_sub[isamp];
            running_sum -= wf_sub[isamp - dt_min_sub - 1];
            buf_row0[isamp] = running_sum;
        }
    } else {
        // No smearing: just shift by dt_min_sub
        if constexpr (UseRoll) {
            // Triangle: wrap indices
            for (SizeType isamp = 0; isamp < dt_min_sub; ++isamp) {
                buf_row0[isamp] = wf_sub[nsamps - dt_min_sub + isamp];
            }
        } else {
            // Triangle: partial (no history)
            std::fill(buf_row0, buf_row0 + dt_min_sub, 0.0F);
        }
        // Main region
        for (SizeType isamp = dt_min_sub; isamp < nsamps; ++isamp) {
            buf_row0[isamp] = wf_sub[isamp - dt_min_sub];
        }
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl_row(const float* __restrict__ wf_sub,
                        const float* __restrict__ buf_prev,
                        float* __restrict__ buf_cur,
                        SizeType dt_cur,
                        SizeType nsamps) noexcept {
    if constexpr (UseBoxSmearing) {
        // Extend box sum: new_sum = prev_sum + waterfall[isamp - dt_cur]
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] =
                    buf_prev[isamp] + wf_sub[nsamps - dt_cur + isamp];
            }
        } else {
            // No new sample available, just copy previous partial sum
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = buf_prev[isamp];
            }
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = buf_prev[isamp] + wf_sub[isamp - dt_cur];
        }
    } else {
        // No smearing: just shift by dt_cur
        if constexpr (UseRoll) {
            for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                buf_cur[isamp] = wf_sub[nsamps - dt_cur + isamp];
            }
        } else {
            std::fill(buf_cur, buf_cur + dt_cur, 0.0F);
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = wf_sub[isamp - dt_cur];
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
template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_subband(const float* __restrict__ wf_sub,
                       float* __restrict__ buf_base,
                       const std::vector<IndexType>& dt_grid_sub,
                       SizeType nsamps) noexcept {
    const auto ndt_sub  = dt_grid_sub.size();
    const auto dt_first = dt_grid_sub.front();
    const auto dt_last  = dt_grid_sub.back();

    if constexpr (!UseBoxSmearing) {
        // No smearing: each row is an independent shift of the raw
        // waterfall by |dt| samples (no accumulation between rows, so no
        // incremental sweep is needed here, unlike the box-smearing case
        // below).
        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto s = static_cast<SizeType>(std::abs(dt_grid_sub[i_dt]));
            fdmt_init_impl_row0<UseRoll, false>(
                wf_sub, buf_base + (i_dt * nsamps), s, nsamps);
        }
        return;
    }

    if (dt_first >= 0) {
        // All-non-negative sub-band grid: original incremental algorithm,
        // unchanged (numerically identical to the pre-negative-DM code path).
        const auto dt_min_sub = static_cast<SizeType>(dt_first);
        fdmt_init_impl_row0<UseRoll, true>(wf_sub, buf_base, dt_min_sub,
                                           nsamps);
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur = static_cast<SizeType>(dt_grid_sub[i_dt]);
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const float* __restrict__ buf_prev =
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

    std::vector<float> row_prev(nsamps);
    std::vector<float> row_curr(nsamps);

    auto write_matches = [&](SizeType s, const float* __restrict__ row) {
        const auto s_signed = static_cast<IndexType>(s);
        if (s_signed >= dt_first && s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(s_signed - dt_first) *
                                       nsamps));
        }
        if (s_signed != 0 && -s_signed >= dt_first && -s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(-s_signed - dt_first) *
                                       nsamps));
        }
    };

    if (s_lo == 0) {
        std::copy_n(wf_sub, nsamps, row_prev.data());
    } else {
        fdmt_init_impl_row0<UseRoll, true>(wf_sub, row_prev.data(), s_lo,
                                           nsamps);
    }
    write_matches(s_lo, row_prev.data());

    for (SizeType s = s_lo + 1; s <= s_hi; ++s) {
        fdmt_init_impl_row<UseRoll, true>(wf_sub, row_prev.data(),
                                          row_curr.data(), s, nsamps);
        write_matches(s, row_curr.data());
        row_prev.swap(row_curr);
    }
}

template <bool UseRoll, bool UseBoxSmearing>
void fdmt_init_impl(const float* __restrict__ waterfall,
                    float* __restrict__ init_buffer,
                    const plans::FDMTCoordGrid* __restrict__ grids_init,
                    SizeType nsubs,
                    SizeType nsamps) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, init_buffer, grids_init, nsubs, nsamps)
#endif
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        float* __restrict__ buf_base =
            init_buffer + (grids_init[i_sub].coord_offset * nsamps);
        fdmt_init_subband<UseRoll, UseBoxSmearing>(
            wf_sub, buf_base, grids_init[i_sub].dt_grid, nsamps);
    }
}

void fdmt_init_valid_row0_box(const float* __restrict__ wf_sub,
                              const float* __restrict__ hist_sub,
                              float* __restrict__ buf_row0,
                              SizeType dt_min_sub,
                              SizeType dt_max_final,
                              SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) =====
    // Box sum of (dt_min_sub + 1) samples ending at current position
    float running_sum = 0.0F;

    // --- isamp = 0: sum of logical indices [-dt_min_sub, 0] ---
    // Samples [-dt_min_sub, -1] from hist_sub, [0] from wf_sub
    for (SizeType i = 0; i < dt_min_sub; ++i) {
        running_sum += hist_sub[dt_max_final - dt_min_sub + i];
    }
    running_sum += wf_sub[0];
    buf_row0[0] = running_sum;

    // --- isamp in [1, min(dt_min_sub, nsamps-1)]: window slides into valid
    // data. Clamped to nsamps-1 so a per-channel delay exceeding the block
    // size (dt_min_sub >= nsamps) can't write past the end of buf_row0; the
    // main region loop below is then naturally empty in that case.
    const SizeType boundary_end = std::min(dt_min_sub + 1, nsamps);
    for (SizeType isamp = 1; isamp < boundary_end; ++isamp) {
        running_sum += wf_sub[isamp];
        running_sum -= hist_sub[dt_max_final + isamp - dt_min_sub - 1];
        buf_row0[isamp] = running_sum;
    }

    // --- isamp in [dt_min_sub + 1, nsamps): fully within waterfall ---
    // Add from wf_sub, remove from wf_sub
    for (SizeType isamp = dt_min_sub + 1; isamp < nsamps; ++isamp) {
        running_sum += wf_sub[isamp];
        running_sum -= wf_sub[isamp - dt_min_sub - 1];
        buf_row0[isamp] = running_sum;
    }
}

void fdmt_init_valid_row0_shift(const float* __restrict__ wf_sub,
                                const float* __restrict__ hist_sub,
                                float* __restrict__ buf_row0,
                                SizeType dt_min_sub,
                                SizeType dt_max_final,
                                SizeType nsamps) noexcept {
    // ===== First DT row (dt_min_sub) without smearing =====
    // For isamp in [0, min(dt_min_sub, nsamps)): read from history. Clamped
    // so dt_min_sub >= nsamps can't write past the end of buf_row0.
    const SizeType boundary_len = std::min(dt_min_sub, nsamps);
    for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
        buf_row0[isamp] = hist_sub[dt_max_final - dt_min_sub + isamp];
    }
    // For isamp in [dt_min_sub, nsamps): read from waterfall
    for (SizeType isamp = dt_min_sub; isamp < nsamps; ++isamp) {
        buf_row0[isamp] = wf_sub[isamp - dt_min_sub];
    }
}

template <bool UseBoxSmearing>
void fdmt_init_valid_row(const float* __restrict__ wf_sub,
                         const float* __restrict__ hist_sub,
                         const float* __restrict__ buf_prev,
                         float* __restrict__ buf_cur,
                         SizeType dt_cur,
                         SizeType dt_max_final,
                         SizeType nsamps) noexcept {
    // Clamped so dt_cur >= nsamps can't write past the end of buf_cur (or
    // read past the end of buf_prev, which is the same size).
    const SizeType boundary_len = std::min(dt_cur, nsamps);
    if constexpr (UseBoxSmearing) {
        // Extend box: new_sum[isamp] = prev_sum[isamp] + input[isamp - dt_cur]
        for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
            buf_cur[isamp] =
                buf_prev[isamp] + hist_sub[dt_max_final + isamp - dt_cur];
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = buf_prev[isamp] + wf_sub[isamp - dt_cur];
        }
    } else {
        // No smearing: just shift by dt_cur
        for (SizeType isamp = 0; isamp < boundary_len; ++isamp) {
            buf_cur[isamp] = hist_sub[dt_max_final - dt_cur + isamp];
        }
        for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
            buf_cur[isamp] = wf_sub[isamp - dt_cur];
        }
    }
}

void fdmt_init_valid_update_history(const float* __restrict__ waterfall,
                                    float* __restrict__ hist_buffer,
                                    float* __restrict__ hist_init_buffer,
                                    SizeType nsubs,
                                    SizeType nsamps,
                                    SizeType dt_max_init,
                                    SizeType dt_max_final) noexcept {
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        float* __restrict__ hist_sub     = hist_buffer + (i_sub * dt_max_final);
        fdmt_advance_history_window(wf_sub, hist_sub, nsamps, dt_max_final);

        if (hist_init_buffer != nullptr && dt_max_init > 0) {
            float* __restrict__ hist_init_sub =
                hist_init_buffer + (i_sub * dt_max_init);
            fdmt_advance_history_window(wf_sub, hist_init_sub, nsamps,
                                        dt_max_init);
        }
    }
}

/**
 * @brief `mode == "valid"` analogue of `fdmt_init_subband` (see its doc
 * comment for the magnitude-only rationale). Uses `hist_sub` to seed rows
 * that need samples from before the start of the current block instead of
 * zero-padding them.
 */
template <bool UseBoxSmearing>
void fdmt_init_valid_subband(const float* __restrict__ wf_sub,
                             const float* __restrict__ hist_sub,
                             float* __restrict__ buf_base,
                             const std::vector<IndexType>& dt_grid_sub,
                             SizeType dt_max_final,
                             SizeType nsamps) noexcept {
    const auto ndt_sub  = dt_grid_sub.size();
    const auto dt_first = dt_grid_sub.front();
    const auto dt_last  = dt_grid_sub.back();

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
        fdmt_init_valid_row0_box(wf_sub, hist_sub, buf_base, dt_min_sub,
                                 dt_max_final, nsamps);
        for (SizeType i_dt = 1; i_dt < ndt_sub; ++i_dt) {
            const auto dt_cur = static_cast<SizeType>(dt_grid_sub[i_dt]);
            float* __restrict__ buf_cur = buf_base + (i_dt * nsamps);
            const float* __restrict__ buf_prev =
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

    std::vector<float> row_prev(nsamps);
    std::vector<float> row_curr(nsamps);

    auto write_matches = [&](SizeType s, const float* __restrict__ row) {
        const auto s_signed = static_cast<IndexType>(s);
        if (s_signed >= dt_first && s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(s_signed - dt_first) *
                                       nsamps));
        }
        if (s_signed != 0 && -s_signed >= dt_first && -s_signed <= dt_last) {
            std::copy_n(row, nsamps,
                        buf_base + (static_cast<SizeType>(-s_signed - dt_first) *
                                       nsamps));
        }
    };

    if (s_lo == 0) {
        std::copy_n(wf_sub, nsamps, row_prev.data());
    } else {
        fdmt_init_valid_row0_box(wf_sub, hist_sub, row_prev.data(), s_lo,
                                 dt_max_final, nsamps);
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
template <bool UseBoxSmearing>
void fdmt_init_valid_impl(const float* __restrict__ waterfall,
                          float* __restrict__ init_buffer,
                          float* __restrict__ hist_buffer,
                          float* __restrict__ hist_init_buffer,
                          const plans::FDMTCoordGrid* __restrict__ grids_init,
                          SizeType nsubs,
                          SizeType nsamps,
                          SizeType dt_max_init,
                          SizeType dt_max_final) noexcept {
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, init_buffer, hist_buffer, grids_init, nsubs, nsamps,     \
               dt_max_final)
#endif
    for (SizeType i_sub = 0; i_sub < nsubs; ++i_sub) {
        const float* __restrict__ wf_sub = waterfall + (i_sub * nsamps);
        const float* __restrict__ hist_sub =
            hist_buffer + (i_sub * dt_max_final);
        float* __restrict__ buf_base =
            init_buffer + (grids_init[i_sub].coord_offset * nsamps);

        fdmt_init_valid_subband<UseBoxSmearing>(wf_sub, hist_sub, buf_base,
                                                grids_init[i_sub].dt_grid,
                                                dt_max_final, nsamps);
    }

    fdmt_init_valid_update_history(waterfall, hist_buffer, hist_init_buffer,
                                   nsubs, nsamps, dt_max_init, dt_max_final);
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
          m_state_internal(m_nbeams * m_plan.get_buffer_size(), 0.0F),
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
          m_state_internal(m_nbeams * m_plan.get_buffer_size(), 0.0F),
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
          m_state_internal(m_nbeams * m_plan.get_buffer_size(), 0.0F),
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
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    [[nodiscard]] SizeType get_nbeams() const noexcept { return m_nbeams; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        reset(waterfall, dmt);
        advance_until_remaining(0);
        finalize();
        spdlog::debug("FDMTCPU::Impl: Execution complete.");
    }

    void reset(std::span<const float> waterfall, std::span<float> dmt) {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall.size() != m_nbeams * nchans * nsamps) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of waterfall. Expected {}, got {}",
                m_nbeams * nchans * nsamps, waterfall.size()));
        }
        if (dmt.size() < m_nbeams * m_plan.get_buffer_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of dmt. Expected at least {}, got {}",
                m_nbeams * m_plan.get_buffer_size(), dmt.size()));
        }

        m_dmt_target_ptr = dmt.data();
        m_current_level  = 0;

        // Number of internal ping-pong iterations (excluding the final write)
        const SizeType internal_iters =
            total_levels() >= 2 ? total_levels() - 2 : 0;
        const bool odd_swaps = (internal_iters % 2) == 1;
        if (odd_swaps) {
            m_current_in_ptr  = dmt.data();
            m_current_out_ptr = m_state_internal.data();
        } else {
            m_current_in_ptr  = m_state_internal.data();
            m_current_out_ptr = dmt.data();
        }

        initialise(waterfall, m_current_in_ptr);
        m_is_initialized = true;
        spdlog::debug("FDMTCPU: Stepper initialized at level 0.");
    }

    void advance(SizeType levels = 1) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCPU: Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            execute_iter(m_current_in_ptr, m_current_out_ptr, next_level);
            std::swap(m_current_in_ptr, m_current_out_ptr);
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
        return {m_current_in_ptr, state_shape.nelements};
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
        return {m_current_in_ptr + offset, count};
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
            .data = std::span<const float>(m_current_in_ptr + offset, count),
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

private:
    bool m_use_box_smearing;
    FDMTMode m_mode;
    SizeType m_nbeams;
    plans::FDMTPlan m_plan;
    // Internal state buffer (for ping-pong buffering)
    std::vector<float> m_state_internal;
    // History buffers for valid-mode streaming across FDMT blocks
    std::vector<float> m_history;
    std::vector<float> m_history_init; // only when use_box_smearing is true
    std::vector<float> m_tree_history;

    // Stepper state
    float* m_current_in_ptr{nullptr};
    float* m_current_out_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
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
    void initialise(std::span<const float> waterfall,
                    float* __restrict__ init_buffer) {
        const auto& plan_c     = m_plan.get_container();
        const auto& grids_init = plan_c.grids[0];
        const auto nsamps      = plan_c.state_shape[0].nsamps;
        const auto dt_max_init = plan_c.state_shape[0].dt_max;
        const auto dt_max_final =
            plan_c.state_shape[m_plan.get_niters()].dt_max;
        const auto nsubs = plan_c.state_shape[0].nchans;

        const auto wf_beam_stride   = nsubs * nsamps;
        const auto buf_beam_stride  = m_plan.get_buffer_size();
        const auto hist_beam_stride = m_plan.get_history_size();
        const auto hist_init_stride = m_plan.get_history_init_size();

        for (SizeType b = 0; b < m_nbeams; ++b) {
            const float* __restrict__ wf_b =
                waterfall.data() + (b * wf_beam_stride);
            float* __restrict__ init_buffer_b =
                init_buffer + (b * buf_beam_stride);

            if (m_mode == FDMTMode::kFull) {
                if (m_use_box_smearing) {
                    fdmt_init_impl<false, true>(
                        wf_b, init_buffer_b, grids_init.data(), nsubs, nsamps);
                } else {
                    fdmt_init_impl<false, false>(
                        wf_b, init_buffer_b, grids_init.data(), nsubs, nsamps);
                }
            } else if (m_mode == FDMTMode::kRoll) {
                if (m_use_box_smearing) {
                    fdmt_init_impl<true, true>(
                        wf_b, init_buffer_b, grids_init.data(), nsubs, nsamps);
                } else {
                    fdmt_init_impl<true, false>(
                        wf_b, init_buffer_b, grids_init.data(), nsubs, nsamps);
                }
            } else {
                float* __restrict__ hist_b =
                    m_history.data() + (b * hist_beam_stride);
                float* __restrict__ hist_init_b =
                    m_history_init.data() + (b * hist_init_stride);
                if (m_use_box_smearing) {
                    fdmt_init_valid_impl<true>(wf_b, init_buffer_b, hist_b,
                                               hist_init_b, grids_init.data(),
                                               nsubs, nsamps, dt_max_init,
                                               dt_max_final);
                } else {
                    fdmt_init_valid_impl<false>(wf_b, init_buffer_b, hist_b,
                                                hist_init_b, grids_init.data(),
                                                nsubs, nsamps, dt_max_init,
                                                dt_max_final);
                }
            }
        }
    }

    void execute_iter(const float* __restrict__ state_in,
                      float* __restrict__ state_out,
                      SizeType i_iter) {
        const auto& plan_c          = m_plan.get_container();
        const auto& coords_sum_cur  = plan_c.coordinates_sum[i_iter];
        const auto& coords_copy_cur = plan_c.coordinates_copy[i_iter];
        const auto ncoords_sum_cur  = coords_sum_cur.size();
        const auto ncoords_copy_cur = coords_copy_cur.size();

        const auto buf_beam_stride  = m_plan.get_buffer_size();
        const auto hist_beam_stride = m_plan.get_tree_history_size();

        for (SizeType b = 0; b < m_nbeams; ++b) {
            const float* __restrict__ state_in_b =
                state_in + (b * buf_beam_stride);
            float* __restrict__ state_out_b = state_out + (b * buf_beam_stride);

            if (m_mode == FDMTMode::kFull) {
                fdmt_iter<FDMTMode::kFull>(
                    state_in_b, state_out_b, nullptr, coords_sum_cur.data(),
                    coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
            } else if (m_mode == FDMTMode::kRoll) {
                fdmt_iter<FDMTMode::kRoll>(
                    state_in_b, state_out_b, nullptr, coords_sum_cur.data(),
                    coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
            } else {
                float* __restrict__ hist_b =
                    m_tree_history.data() + (b * hist_beam_stride);
                fdmt_iter<FDMTMode::kValid>(
                    state_in_b, state_out_b, hist_b, coords_sum_cur.data(),
                    coords_copy_cur.data(), ncoords_sum_cur, ncoords_copy_cur);
            }
        }
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