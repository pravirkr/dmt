#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_tree.hpp"

// Level fusion for the CUDA backend (FDMTExecConfig::fuse_levels): level-0
// initialisation plus the first F tree merges of one channel group, over one
// time tile, computed entirely in on-chip (shared) memory by one thread
// block. Written against a small `Block` policy (row-wise for_each_row +
// barrier)
// so the identical code also runs sequentially on the host, where the tests
// check it bit-for-bit against the CPU FDMT (see fdmt_fused_tile_cpu_t.cpp).
//
// Geometry. Level-F sub-band g depends only on the 2^F input channels below
// it (the same sub-tree grouping as the CPU FDMT's fused path). A tile produces
// level-F samples [t0, t0 + T). Level l of the group is computed over the
// window [t0 - H_l, t0 + T), where H_F = 0 and H_{l-1} = H_l + D_l, D_l being
// the largest merge delay among the group's level-l coordinates: every head
// sample t - d a level-l output needs is then inside the level-(l-1) window.
// The input is staged once per tile, S extra samples to the left (the widest
// level-0 box of the group). Halo samples are recomputed by neighbouring
// tiles, which is cheap because the fused levels have small delays.
//
// Boundaries follow the unfused kernels exactly: cells outside a level's
// sample range [0, L_l) are zero (kFull), unused (kValid, where t - d < 0
// reads the tree history instead), or wrapped (kRoll, windows are kept in
// unwrapped time, and the input is read modulo the block). Every cell is the
// same single float addition of the same operands as in the level-by-level
// kernels, so fused and unfused results are bit-identical; integer-valued
// state (packed input, int_tree) is exact in float at these bounds.

namespace dmt::algorithms::detail {

/// Deepest fusion the CUDA backend supports (fixed-size per-level arrays).
inline constexpr int kMaxFusedLevels = 8;

/// @brief Level-0 input rows from a float waterfall, beam-major
/// (nbeams * nsubs rows of nsamps).
struct FDMTInputF32 {
    const float* data;
    int nsamps;

    DMT_HD float load(int64_t row, int t) const {
        return data[(static_cast<size_t>(row) * nsamps) + t];
    }
};

/// @brief Level-0 input rows from a packed low-bit waterfall (nbeams * nsubs
/// rows of row_bytes, LSB-first; see bit_pack_utils.hpp). `nbits` is a
/// runtime value: the branch is uniform across the grid, and it keeps the
/// kernel instantiation count down.
struct FDMTInputPacked {
    const uint8_t* data;
    int row_bytes;
    int nbits;

    DMT_HD float load(int64_t row, int t) const {
        const uint8_t* r = data + (static_cast<size_t>(row) * row_bytes);
        const auto st    = static_cast<SizeType>(t);
        switch (nbits) {
        case 1:
            return static_cast<float>(
                bit_pack_utils::read_packed_sample<1>(r, st));
        case 2:
            return static_cast<float>(
                bit_pack_utils::read_packed_sample<2>(r, st));
        case 4:
            return static_cast<float>(
                bit_pack_utils::read_packed_sample<4>(r, st));
        case 8:
            return static_cast<float>(
                bit_pack_utils::read_packed_sample<8>(r, st));
        default:
            return static_cast<float>(
                bit_pack_utils::read_packed_sample<16>(r, st));
        }
    }
};

/**
 * @brief One level-0 column: every dt row of one channel at sample `t`.
 *
 * Shared by the unfused init kernel and the fused tile so both produce the
 * same bits. `dt_first..dt_last` is the channel's dense (possibly signed) dt
 * grid; a level-0 row only depends on |dt| (see kernel_init_fdmt), so each
 * magnitude s is computed once and written to the +s / -s rows present.
 * `get_sample(tt)` returns the input at absolute sample tt <= t (history,
 * wrap or zero padding for tt < 0 is the caller's business);
 * `write_row(i_dt, value)` stores row i_dt's value at t. With box smearing
 * the box sum of |dt| = s, accumulated in TAcc (float, or uint32_t for the
 * exact integers of packed input), is written as `box_mean(s, sum)`, its box
 * mean in the destination's units (fdmt_box_weights.hpp); the running sum
 * itself stays unscaled.
 */
template <bool UseBoxSmearing,
          typename TAcc,
          typename GetSample,
          typename WriteRow,
          typename BoxMean>
DMT_HD inline void fdmt_init_column(int dt_first,
                                    int dt_last,
                                    int t,
                                    GetSample&& get_sample,
                                    WriteRow&& write_row,
                                    BoxMean&& box_mean) {
    const auto write_matches = [&](int s, auto val) {
        if (s >= dt_first && s <= dt_last) {
            write_row(s - dt_first, val);
        }
        const int neg = -s;
        if (s != 0 && neg >= dt_first && neg <= dt_last) {
            write_row(neg - dt_first, val);
        }
    };
    const int abs_first = dt_first < 0 ? -dt_first : dt_first;
    const int abs_last  = dt_last < 0 ? -dt_last : dt_last;
    // Sweep start: a grid reaching dt >= 0 sweeps up from s = 0 (writing
    // only the magnitudes present); a purely negative grid seeds at its
    // smallest magnitude, the end nearest zero.
    const int s_lo =
        (dt_last >= 0) ? 0 : (abs_first < abs_last ? abs_first : abs_last);
    const int s_hi = abs_first > abs_last ? abs_first : abs_last;
    if constexpr (UseBoxSmearing) {
        TAcc prev_val;
        if (s_lo == 0) {
            prev_val = static_cast<TAcc>(get_sample(t));
        } else {
            TAcc sum = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 4
#endif
            for (int d = 0; d <= s_lo; ++d) {
                sum += static_cast<TAcc>(get_sample(t - d));
            }
            prev_val = sum;
        }
        write_matches(s_lo, box_mean(s_lo, prev_val));
        for (int s = s_lo + 1; s <= s_hi; ++s) {
            const TAcc cur_val =
                prev_val + static_cast<TAcc>(get_sample(t - s));
            write_matches(s, box_mean(s, cur_val));
            prev_val = cur_val;
        }
    } else {
        for (int s = s_lo; s <= s_hi; ++s) {
            write_matches(s, get_sample(t - s));
        }
    }
}

/**
 * @brief The level-0 box means of one engine (fdmt_box_weights.hpp), by |dt|
 * = s: dyadic float weights `w_f32` for float input; for packed input the
 * integer rounding recipes on the common scale 2^-F (`unit` = 2^-F), with
 * 64-bit products when `round_u64`.
 */
struct FDMTBoxMeans {
    const float* w_f32;
    const FDMTBoxRound* round;
    float unit;
    int round_u64;

    /// Box mean of float-input box sum `sum`.
    [[nodiscard]] DMT_HD float of_float(int s, float sum) const {
        return sum * w_f32[s];
    }
    /// Rounded box mean of packed-input box sum `sum`: in integer units
    /// (exact as float) when `int_units`, else times 2^-F.
    [[nodiscard]] DMT_HD float
    of_packed(int s, uint32_t sum, bool int_units) const {
        const auto& r = round[s];
        const float v =
            round_u64 != 0
                ? static_cast<float>(box_round_apply<uint64_t>(sum, r))
                : static_cast<float>(box_round_apply<uint32_t>(sum, r));
        return int_units ? v : v * unit;
    }
};

/// Whether level-0 input type Input is packed low-bit (exact integers).
template <typename Input> inline constexpr bool kFDMTPackedInput    = false;
template <> inline constexpr bool kFDMTPackedInput<FDMTInputPacked> = true;

/**
 * @brief Kernel arguments of one fusion depth (plain data + device pointers).
 *
 * `group_info` holds, per channel group, kGroupHeader ints (first channel,
 * channel count, level-0 staging halo S) followed by kLevelInfo ints per
 * level 0..F (first coordinate, row count, window halo H_l, max merge delay
 * D_l). `coords` holds kCoordInfo ints per coordinate of levels 1..F,
 * indexed by global coordinate from `coords_offset[l]`: tail row and head
 * row (group-local rows of level l-1; head -1 for a copy coordinate), merge
 * delay, and valid-mode tree-history offset.
 */
struct FDMTFusedTileArgs {
    static constexpr int kGroupHeader = 3;
    static constexpr int kLevelInfo   = 4;
    static constexpr int kCoordInfo   = 4;

    int fuse;        ///< F (>= 1)
    int tile_nsamps; ///< T: level-F samples per tile
    int cap_a;       ///< shared floats of buffer A (buffer B follows)
    int nsubs;       ///< level-0 channels
    int hist_len;    ///< level-0 history per channel: last dt0_max samples
    /// Box smearing: the level-0 box means; box.unit is also the value 2^-F
    /// of one integer state unit (1 without box smearing).
    FDMTBoxMeans box;
    int level_nsamps[kMaxFusedLevels + 1]; ///< L_l, samples per row
    /// Valid-mode chunk: samples of this chunk in every row (rows keep their
    /// stride L_l); 0 = each level's whole row L_l.
    int active;
    int coords_offset[kMaxFusedLevels + 1];
    const int* group_info;
    const int* coords;
    const int* dt_grid0;      ///< level-0 dt grids, flat (by coord offset)
    const int* ndt0;          ///< per channel
    const int* coord_offset0; ///< per channel

    [[nodiscard]] DMT_HD int group_stride() const {
        return kGroupHeader + (kLevelInfo * (fuse + 1));
    }
};

/// @brief Whether sample t of a level with L samples per row is computed.
template <FDMTMode Mode> DMT_HD inline bool fdmt_in_range(int t, int L) {
    if constexpr (Mode == FDMTMode::kRoll) {
        return t < L; // negative t is the wrapped end of the block
    } else {
        return t >= 0 && t < L;
    }
}

/**
 * @brief One merge cell out(t) of a group-local coordinate, from the
 * previous level's window `in` (rows of width w_in starting at sample
 * lo_in). Mirrors kernel_execute_iter's arithmetic for each mode.
 */
/**
 * @brief One group-local merge coordinate, resolved against the previous
 * level's window (rows of width w_in starting at sample lo_in): cell(t) is
 * the output at sample t. Mirrors kernel_execute_iter's arithmetic for each
 * mode.
 */
template <FDMTMode Mode> struct FDMTFusedMergeRow {
    const float* tail; // in-window sample 0 of the tail row
    const float* head; // of the head row, shifted by the delay; or nullptr
    int d;
    const float* thist; // valid-mode tree history of the coordinate
    float hist_scale;   // canonical history -> tile units

    DMT_HD FDMTFusedMergeRow(const float* in,
                             int w_in,
                             int lo_in,
                             const int* coord,
                             const float* thist_in,
                             float hist_scale_in)
        : tail(in + (coord[0] * w_in) - lo_in),
          head(coord[1] < 0 ? nullptr
                            : in + (coord[1] * w_in) - lo_in - coord[2]),
          d(coord[2]),
          thist(thist_in != nullptr ? thist_in + coord[3] : nullptr),
          hist_scale(hist_scale_in) {}

    [[nodiscard]] DMT_HD float cell(int t) const {
        const float tail_val = tail[t];
        if (head == nullptr) {
            return tail_val; // copy coordinate (zero beyond the tail in kFull)
        }
        float head_val;
        if constexpr (Mode == FDMTMode::kValid) {
            if (t >= d) {
                head_val = head[t];
            } else {
                head_val = (thist != nullptr) ? thist[t] * hist_scale : 0.0F;
            }
        } else {
            // kFull: cells outside [0, L_in) are zero; kRoll: unwrapped.
            head_val = head[t];
        }
        return tail_val + head_val;
    }
};

/**
 * @brief Valid-mode tree history of the group's level-l coordinates (same
 * FIFO update as kernel_advance_tree_history), from level l-1 in `in`. Each
 * history sample taken from the block is written by the tile whose output
 * range [t0, t0 + T) contains it; samples shifted from the previous history
 * (blocks shorter than the delay) by tile 0. The history is in canonical
 * units: tile values are scaled by `to_canonical`.
 */
template <typename Block>
DMT_HD inline void fdmt_fused_tree_history(const Block& blk,
                                           const float* in,
                                           int w_in,
                                           int lo_in,
                                           const int* coords,
                                           int nrows,
                                           int max_delay,
                                           int nsamps,
                                           int tile,
                                           int t0,
                                           int t_end,
                                           const float* thist_in,
                                           float* thist_out,
                                           float to_canonical) {
    if (thist_out == nullptr || max_delay <= 0) {
        return;
    }
    blk.for_each_row(nrows, max_delay, [&](int r) {
        const int* coord = coords + (FDMTFusedTileArgs::kCoordInfo * r);
        const int head   = coord[1];
        const int d      = coord[2];
        const int hoff   = coord[3];
        return [=](int k) {
            if (head < 0 || k >= d) {
                return;
            }
            float val;
            if (nsamps >= d || k >= d - nsamps) {
                const int p =
                    (nsamps >= d) ? (nsamps - d + k) : (k - (d - nsamps));
                if (p < t0 || p >= t_end) {
                    return;
                }
                val = in[(head * w_in) + (p - lo_in)] * to_canonical;
            } else {
                if (tile != 0) {
                    return;
                }
                val =
                    (thist_in != nullptr) ? thist_in[hoff + k + nsamps] : 0.0F;
            }
            thist_out[hoff + k] = val;
        };
    });
}

/**
 * @brief Fused levels 0..F of channel group `group` over time tile `tile`.
 *
 * @param blk     Execution policy: for_each_row(n, w, row) runs row(r) once
 *                per row r in [0, n) and the returned cell(j) for j in
 *                [0, w) (rows over warps, cells over lanes on the GPU);
 *                sync() is a block barrier.
 * @param input   Level-0 input rows; this beam's channel c is row
 *                row_base + c.
 * @param out     This beam's level-F state (row k at k * L_F), stored as
 *                TOut.
 * @param smem    cap_a + cap_b floats of scratch (see
 *                FDMTFusedTilePlan::smem_floats()).
 * @param hist0   This beam's level-0 input history (valid mode) or nullptr.
 * @param thist_in / thist_out This beam's tree history, previous / next block
 *                (valid mode) or nullptr.
 *
 * Shared memory holds levels 0..F-1 in level F's units: integer units
 * (exact integers, see fdmt_box_weights.hpp) when TOut is an integer type,
 * canonical box-mean units otherwise. The tree history is canonical.
 */
template <FDMTMode Mode,
          bool UseBoxSmearing,
          typename Block,
          typename Input,
          typename TOut>
DMT_HD void fdmt_fused_tile(const Block& blk,
                            const Input& input,
                            int64_t row_base,
                            TOut* out,
                            const FDMTFusedTileArgs& a,
                            int tile,
                            int group,
                            float* smem,
                            const float* hist0,
                            const float* thist_in,
                            float* thist_out) {
    constexpr int kLI = FDMTFusedTileArgs::kLevelInfo;
    constexpr int kCI = FDMTFusedTileArgs::kCoordInfo;
    const int* gi     = a.group_info + (group * a.group_stride());
    const int ch0     = gi[0];
    const int nch     = gi[1];
    const int s_halo  = gi[2];
    const int* lvl    = gi + FDMTFusedTileArgs::kGroupHeader;
    const int fuse    = a.fuse;
    const int t0      = tile * a.tile_nsamps;
    const int t_end   = t0 + a.tile_nsamps;
    const auto row0   = [&](int l) { return lvl[(kLI * l) + 0]; };
    const auto nrows  = [&](int l) { return lvl[(kLI * l) + 1]; };
    const auto lo     = [&](int l) { return t0 - lvl[(kLI * l) + 2]; };
    const auto width  = [&](int l) { return t_end - lo(l); };
    float* buf_a      = smem;
    float* buf_b      = smem + a.cap_a;
    const auto buf    = [&](int l) { return (l % 2 == 0) ? buf_a : buf_b; };
    // Value of one tile unit in canonical units: integer level-F storage
    // keeps integer units 2^-F (packed input only).
    constexpr bool kIntUnits = std::is_integral_v<TOut>;
    const float tile_unit    = kIntUnits ? a.box.unit : 1.0F;
    const float hist_scale   = 1.0F / tile_unit;

    // Samples of level l computed this call (rows keep their stride L_l).
    const auto range = [&](int l) {
        return a.active > 0 ? a.active : a.level_nsamps[l];
    };

    // Phase 1: stage the group's input window [lo_0 - S, t_end) as float in
    // buffer B (free until level 1 is written).
    const int l0      = range(0);
    const int x_lo    = lo(0) - s_halo;
    const int x_width = t_end - x_lo;
    float* x          = buf_b;
    blk.for_each_row(nch, x_width, [&](int c) {
        const int64_t row = row_base + ch0 + c;
        float* xr         = x + (c * x_width);
        const float* h0   = (hist0 != nullptr)
                                ? hist0 + ((ch0 + c) * a.hist_len) + a.hist_len
                                : nullptr;
        return [=, &input, &a](int j) {
            (void)a;
            const int tt = x_lo + j;
            float v      = 0.0F;
            if constexpr (Mode == FDMTMode::kRoll) {
                const int tw = ((tt % l0) + l0) % l0;
                v            = input.load(row, tw);
            } else {
                if (tt >= 0 && tt < l0) {
                    v = input.load(row, tt);
                } else if (Mode == FDMTMode::kValid && tt < 0 &&
                           tt >= -a.hist_len && h0 != nullptr) {
                    v = h0[tt];
                }
            }
            xr[j] = v;
        };
    });
    blk.sync();

    // Phase 2: level 0 into buffer A.
    {
        const int w0  = width(0);
        const int lo0 = lo(0);
        float* dst    = buf_a;
        blk.for_each_row(nch, w0, [&](int c) {
            const int ch    = ch0 + c;
            const int coff  = a.coord_offset0[ch];
            const int ndt   = a.ndt0[ch];
            const int dt_lo = a.dt_grid0[coff];
            const int dt_hi = a.dt_grid0[coff + ndt - 1];
            float* dr       = dst + ((coff - row0(0)) * w0);
            const float* xc = x + (c * x_width) - x_lo;
            return [=](int j) {
                const int t = lo0 + j;
                if (!fdmt_in_range<Mode>(t, l0)) {
                    for (int k = 0; k < ndt; ++k) {
                        dr[(k * w0) + j] = 0.0F;
                    }
                    return;
                }
                const auto get = [&](int tt) { return xc[tt]; };
                const auto put = [&](int i_dt, float v) {
                    dr[(i_dt * w0) + j] = v;
                };
                if constexpr (kFDMTPackedInput<Input>) {
                    fdmt_init_column<UseBoxSmearing, uint32_t>(
                        dt_lo, dt_hi, t, get, put, [&](int s, uint32_t sum) {
                            return a.box.of_packed(s, sum, kIntUnits);
                        });
                } else {
                    fdmt_init_column<UseBoxSmearing, float>(
                        dt_lo, dt_hi, t, get, put, [&](int s, float sum) {
                            return a.box.of_float(s, sum);
                        });
                }
            };
        });
        blk.sync();
    }

    // Phase 3: merge levels 1..F-1 in shared memory, ping-ponging A/B.
    for (int l = 1; l < fuse; ++l) {
        const float* in  = buf(l - 1);
        float* dst       = buf(l);
        const int w_in   = width(l - 1);
        const int lo_in  = lo(l - 1);
        const int w_out  = width(l);
        const int lo_out = lo(l);
        const int n_l    = range(l);
        const int* cl    = a.coords + (kCI * (a.coords_offset[l] + row0(l)));
        blk.for_each_row(nrows(l), w_out, [&](int r) {
            const FDMTFusedMergeRow<Mode> m(in, w_in, lo_in, cl + (kCI * r),
                                            thist_in, hist_scale);
            float* dr = dst + (r * w_out);
            return [=](int j) {
                const int t = lo_out + j;
                dr[j]       = fdmt_in_range<Mode>(t, n_l) ? m.cell(t) : 0.0F;
            };
        });
        if constexpr (Mode == FDMTMode::kValid) {
            fdmt_fused_tree_history(blk, in, w_in, lo_in, cl, nrows(l),
                                    lvl[(kLI * l) + 3], range(l - 1), tile, t0,
                                    t_end, thist_in, thist_out, tile_unit);
        }
        blk.sync();
    }

    // Phase 4: level F straight to its state buffer.
    {
        const float* in    = buf(fuse - 1);
        const int w_in     = width(fuse - 1);
        const int lo_in    = lo(fuse - 1);
        const int n_f      = range(fuse);
        const int stride_f = a.level_nsamps[fuse];
        const int tn       = a.tile_nsamps;
        const int first    = row0(fuse);
        const int* cl      = a.coords + (kCI * (a.coords_offset[fuse] + first));
        blk.for_each_row(nrows(fuse), tn, [&](int r) {
            const FDMTFusedMergeRow<Mode> m(in, w_in, lo_in, cl + (kCI * r),
                                            thist_in, hist_scale);
            TOut* orow = out + (static_cast<int64_t>(first + r) * stride_f);
            return [=](int j) {
                const int t = t0 + j;
                if (t < n_f) {
                    orow[t] = static_cast<TOut>(m.cell(t));
                }
            };
        });
        if constexpr (Mode == FDMTMode::kValid) {
            fdmt_fused_tree_history(blk, in, w_in, lo_in, cl, nrows(fuse),
                                    lvl[(kLI * fuse) + 3], range(fuse - 1),
                                    tile, t0, t_end, thist_in, thist_out,
                                    tile_unit);
        }
    }
}

/**
 * @brief Host-side tables of one fusion depth (see FDMTFusedTileArgs); the
 * CUDA backend uploads them, the tests use them in place.
 */
struct FDMTFusedTilePlan {
    int fuse{0};
    int ngroups{0};
    std::vector<int> group_info;
    std::vector<int> coords;
    std::vector<int> dt_grid0;
    std::vector<int> ndt0;
    std::vector<int> coord_offset0;
    std::array<int, kMaxFusedLevels + 1> level_nsamps{};
    std::array<int, kMaxFusedLevels + 1> coords_offset{};
    int ntiles_nsamps{0};   ///< level-F samples the tiles must cover (L_F)
    int max_level0_halo{0}; ///< max over groups of H_0 + S

    /// Shared floats {buffer A, buffer B} needed for tile width T.
    [[nodiscard]] std::pair<SizeType, SizeType> smem_floats(int tile) const {
        const int stride = FDMTFusedTileArgs::kGroupHeader +
                           (FDMTFusedTileArgs::kLevelInfo * (fuse + 1));
        SizeType cap_a   = 0;
        SizeType cap_b   = 0;
        for (int g = 0; g < ngroups; ++g) {
            const int* gi     = group_info.data() + (g * stride);
            const int* lvl    = gi + FDMTFusedTileArgs::kGroupHeader;
            const auto window = [&](int l) {
                return static_cast<SizeType>(
                    tile + lvl[(FDMTFusedTileArgs::kLevelInfo * l) + 2]);
            };
            for (int l = 0; l < fuse; ++l) {
                const SizeType need =
                    static_cast<SizeType>(
                        lvl[(FDMTFusedTileArgs::kLevelInfo * l) + 1]) *
                    window(l);
                (l % 2 == 0 ? cap_a : cap_b) =
                    std::max((l % 2 == 0 ? cap_a : cap_b), need);
            }
            cap_b =
                std::max(cap_b, static_cast<SizeType>(gi[1]) *
                                    (window(0) + static_cast<SizeType>(gi[2])));
        }
        return {cap_a, cap_b};
    }

    /// Largest tile width (a multiple of 32, <= max_tile) whose scratch fits
    /// `smem_bytes`; 0 if even 32 does not fit.
    [[nodiscard]] int max_tile_nsamps(SizeType smem_bytes, int max_tile) const {
        int best = 0;
        for (int tile = 32; tile <= max_tile; tile += 32) {
            const auto [ca, cb] = smem_floats(tile);
            if ((ca + cb) * sizeof(float) > smem_bytes) {
                break;
            }
            best = tile;
        }
        return best;
    }
};

/// @brief Builds the fusion tables of depth `fuse` (1 <= fuse <=
/// min(niters, kMaxFusedLevels)) for plan container `pc`.
inline FDMTFusedTilePlan build_fused_tile_plan(
    const plans::FDMTPlanContainer& pc, SizeType niters, SizeType fuse) {
    if (fuse < 1 || fuse > niters ||
        fuse > static_cast<SizeType>(kMaxFusedLevels)) {
        throw std::invalid_argument("build_fused_tile_plan: invalid depth");
    }
    constexpr auto kIntMax =
        static_cast<SizeType>(std::numeric_limits<int32_t>::max());
    const auto to_int = [&](SizeType v) {
        if (v > kIntMax) {
            throw std::invalid_argument(
                "build_fused_tile_plan: extent exceeds 32-bit range");
        }
        return static_cast<int>(v);
    };
    FDMTFusedTilePlan p;
    const auto f = static_cast<int>(fuse);
    p.fuse       = f;
    p.ngroups    = to_int(pc.state_shape[fuse].nchans);
    for (SizeType l = 0; l <= fuse; ++l) {
        p.level_nsamps[l] = to_int(pc.state_shape[l].nsamps);
    }
    p.ntiles_nsamps = p.level_nsamps[fuse];

    const auto& grids0 = pc.grids[0];
    for (const auto& grid : grids0) {
        p.ndt0.push_back(to_int(grid.ndt));
        p.coord_offset0.push_back(to_int(grid.coord_offset));
        for (const auto dt : grid.dt_grid) {
            p.dt_grid0.push_back(static_cast<int>(dt));
        }
    }

    // Group-local coordinate range of level l for group g: the level-l
    // sub-bands descending from level-F sub-band g (see fdmt_tree.hpp).
    const auto range = [&](SizeType l, SizeType g) {
        return fdmt_group_coords(pc, fuse, g, l);
    };

    // Coordinate tables: sum coordinates carry the history slot, which is
    // only assigned on coordinates_sum (same order as in coordinates).
    std::vector<std::vector<int>> group_of(fuse + 1);
    for (SizeType l = 1; l <= fuse; ++l) {
        group_of[l - 1].assign(pc.coordinates[l - 1].size(), -1);
    }
    for (SizeType g = 0; g < static_cast<SizeType>(p.ngroups); ++g) {
        for (SizeType l = 0; l < fuse; ++l) {
            const auto [cb, ce] = range(l, g);
            for (SizeType k = cb; k < ce; ++k) {
                group_of[l][k] = static_cast<int>(cb);
            }
        }
    }
    int offset = 0;
    std::vector<std::vector<int>> delay_of(fuse + 1);
    for (SizeType l = 1; l <= fuse; ++l) {
        p.coords_offset[l]     = offset;
        const auto& coords     = pc.coordinates[l];
        const auto& coords_sum = pc.coordinates_sum[l];
        const auto n_in        = pc.state_shape[l - 1].nsamps;
        const auto n_out       = pc.state_shape[l].nsamps;
        delay_of[l].assign(coords.size(), 0);
        SizeType j = 0;
        for (SizeType k = 0; k < coords.size(); ++k) {
            const auto& c = coords[k];
            if (c.buf_offset != k * n_out) {
                throw std::logic_error(
                    "build_fused_tile_plan: unexpected layout");
            }
            const bool is_sum = c.i_coord_head != SIZE_MAX;
            const auto& src   = is_sum ? coords_sum[j++] : c;
            const int base    = group_of[l - 1][src.tail_buf_offset / n_in];
            if (is_sum && group_of[l - 1][src.head_buf_offset / n_in] != base) {
                throw std::logic_error("build_fused_tile_plan: merge operands "
                                       "in different groups");
            }
            p.coords.push_back(to_int(src.tail_buf_offset / n_in) - base);
            p.coords.push_back(
                is_sum ? to_int(src.head_buf_offset / n_in) - base : -1);
            p.coords.push_back(is_sum ? to_int(src.delay) : 0);
            p.coords.push_back(is_sum ? to_int(src.hist_offset) : 0);
            delay_of[l][k] = is_sum ? to_int(src.delay) : 0;
        }
        offset += static_cast<int>(coords.size());
    }

    for (SizeType g = 0; g < static_cast<SizeType>(p.ngroups); ++g) {
        const auto [ch_begin, ch_end] = fdmt_group_channels(pc, fuse, g);
        int s_halo                    = 0;
        for (SizeType c = ch_begin; c < ch_end; ++c) {
            const auto& dts = grids0[c].dt_grid;
            s_halo = std::max({s_halo, static_cast<int>(std::abs(dts.front())),
                               static_cast<int>(std::abs(dts.back()))});
        }
        std::vector<int> halo(fuse + 1, 0);
        std::vector<int> max_delay(fuse + 1, 0);
        for (SizeType l = fuse; l >= 1; --l) {
            const auto [cb, ce] = range(l, g);
            for (SizeType k = cb; k < ce; ++k) {
                max_delay[l] = std::max(max_delay[l], delay_of[l][k]);
            }
            halo[l - 1] = halo[l] + max_delay[l];
        }
        p.group_info.push_back(to_int(ch_begin));
        p.group_info.push_back(to_int(ch_end - ch_begin));
        p.group_info.push_back(s_halo);
        for (SizeType l = 0; l <= fuse; ++l) {
            const auto [cb, ce] = range(l, g);
            p.group_info.push_back(to_int(cb));
            p.group_info.push_back(to_int(ce - cb));
            p.group_info.push_back(halo[l]);
            p.group_info.push_back(max_delay[l]);
        }
        p.max_level0_halo = std::max(p.max_level0_halo, halo[0] + s_halo);
    }
    return p;
}

/// @brief Kernel arguments for `plan` with tile width `tile`, pointing at
/// the given (device or host) copies of the plan's tables.
/// `box`: see FDMTFusedTileArgs (its tables are unused without box
/// smearing).
inline FDMTFusedTileArgs make_fused_tile_args(const FDMTFusedTilePlan& plan,
                                              int tile,
                                              int hist_len,
                                              const int* group_info,
                                              const int* coords,
                                              const int* dt_grid0,
                                              const int* ndt0,
                                              const int* coord_offset0,
                                              const FDMTBoxMeans& box) {
    FDMTFusedTileArgs a{};
    a.fuse        = plan.fuse;
    a.tile_nsamps = tile;
    a.cap_a       = static_cast<int>(plan.smem_floats(tile).first);
    a.nsubs       = static_cast<int>(plan.ndt0.size());
    a.hist_len    = hist_len;
    a.box         = box;
    a.active      = 0;
    for (int l = 0; l <= kMaxFusedLevels; ++l) {
        a.level_nsamps[l]  = plan.level_nsamps[l];
        a.coords_offset[l] = plan.coords_offset[l];
    }
    a.group_info    = group_info;
    a.coords        = coords;
    a.dt_grid0      = dt_grid0;
    a.ndt0          = ndt0;
    a.coord_offset0 = coord_offset0;
    return a;
}

} // namespace dmt::algorithms::detail
