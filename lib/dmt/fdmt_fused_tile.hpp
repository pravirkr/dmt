#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

// Level fusion for the CUDA backend (FDMTExecConfig::fuse_levels): level-0
// initialisation plus the first F tree merges of one channel group, over one
// time tile, computed entirely in on-chip (shared) memory by one thread
// block. Written against a small `Block` policy (strided for_each + barrier)
// so the identical code also runs sequentially on the host, where the tests
// check it bit-for-bit against FDMTCPU (see fdmt_fused_tile_cpu_t.cpp).
//
// Geometry. Level-F sub-band g depends only on the 2^F input channels below
// it (the same sub-tree grouping as FDMTCPU's fused path). A tile produces
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
 * `write_row(i_dt, value)` stores row i_dt's value at t.
 */
template <bool UseBoxSmearing, typename GetSample, typename WriteRow>
DMT_HD inline void fdmt_init_column(int dt_first,
                                    int dt_last,
                                    int t,
                                    GetSample&& get_sample,
                                    WriteRow&& write_row) {
    const auto write_matches = [&](int s, float val) {
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
        float prev_val;
        if (s_lo == 0) {
            prev_val = get_sample(t);
        } else {
            float sum = 0.0F;
#ifdef __CUDA_ARCH__
#pragma unroll 4
#endif
            for (int d = 0; d <= s_lo; ++d) {
                sum += get_sample(t - d);
            }
            prev_val = sum;
        }
        write_matches(s_lo, prev_val);
        for (int s = s_lo + 1; s <= s_hi; ++s) {
            const float cur_val = prev_val + get_sample(t - s);
            write_matches(s, cur_val);
            prev_val = cur_val;
        }
    } else {
        for (int s = s_lo; s <= s_hi; ++s) {
            write_matches(s, get_sample(t - s));
        }
    }
}

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

    int fuse;         ///< F (>= 1)
    int tile_nsamps;  ///< T: level-F samples per tile
    int cap_a;        ///< shared floats of buffer A (buffer B follows)
    int nsubs;        ///< level-0 channels
    int dt_max_final; ///< level-0 history stride per channel (valid mode)
    int level_nsamps[kMaxFusedLevels + 1]; ///< L_l, samples per row
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
template <FDMTMode Mode>
DMT_HD inline float fdmt_fused_merge_cell(const float* in,
                                          int w_in,
                                          int lo_in,
                                          const int* coord,
                                          int t,
                                          const float* thist_in) {
    const int tail       = coord[0];
    const int head       = coord[1];
    const float tail_val = in[(tail * w_in) + (t - lo_in)];
    if (head < 0) {
        return tail_val; // copy coordinate (zero beyond the tail in kFull)
    }
    const int d = coord[2];
    float head_val;
    if constexpr (Mode == FDMTMode::kValid) {
        if (t >= d) {
            head_val = in[(head * w_in) + (t - d - lo_in)];
        } else {
            head_val = (thist_in != nullptr) ? thist_in[coord[3] + t] : 0.0F;
        }
    } else {
        // kFull: cells outside [0, L_in) are zero; kRoll: unwrapped window.
        head_val = in[(head * w_in) + (t - d - lo_in)];
    }
    return tail_val + head_val;
}

/**
 * @brief Valid-mode tree history of the group's level-l coordinates (same
 * FIFO update as kernel_advance_tree_history), from level l-1 in `in`. Each
 * history sample taken from the block is written by the tile whose output
 * range [t0, t0 + T) contains it; samples shifted from the previous history
 * (blocks shorter than the delay) by tile 0.
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
                                           float* thist_out) {
    if (thist_out == nullptr || max_delay <= 0) {
        return;
    }
    blk.for_each(nrows * max_delay, [&](int i) {
        const int* coord =
            coords + (FDMTFusedTileArgs::kCoordInfo * (i / max_delay));
        const int k    = i % max_delay;
        const int head = coord[1];
        const int d    = coord[2];
        if (head < 0 || k >= d) {
            return;
        }
        float val;
        if (nsamps >= d || k >= d - nsamps) {
            const int p = (nsamps >= d) ? (nsamps - d + k) : (k - (d - nsamps));
            if (p < t0 || p >= t_end) {
                return;
            }
            val = in[(head * w_in) + (p - lo_in)];
        } else {
            if (tile != 0) {
                return;
            }
            val =
                (thist_in != nullptr) ? thist_in[coord[3] + k + nsamps] : 0.0F;
        }
        thist_out[coord[3] + k] = val;
    });
}

/**
 * @brief Fused levels 0..F of channel group `group` over time tile `tile`.
 *
 * @param blk     Execution policy: for_each(n, f) runs f(i) for i in [0, n)
 *                (strided over the thread block on the GPU), sync() is a
 *                block barrier.
 * @param input   Level-0 input rows; this beam's channel c is row
 *                row_base + c.
 * @param out     This beam's level-F state (row k at k * L_F), stored as
 *                TOut.
 * @param smem    cap_a + cap_b floats of scratch (see
 *                FDMTFusedTilePlan::smem_floats()).
 * @param hist0   This beam's level-0 input history (valid mode) or nullptr.
 * @param thist_in / thist_out This beam's tree history, previous / next block
 *                (valid mode) or nullptr.
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

    // Phase 1: stage the group's input window [lo_0 - S, t_end) as float in
    // buffer B (free until level 1 is written).
    const int l0      = a.level_nsamps[0];
    const int x_lo    = lo(0) - s_halo;
    const int x_width = t_end - x_lo;
    float* x          = buf_b;
    blk.for_each(nch * x_width, [&](int i) {
        const int c  = i / x_width;
        const int tt = x_lo + (i % x_width);
        float v      = 0.0F;
        if constexpr (Mode == FDMTMode::kRoll) {
            const int tw = ((tt % l0) + l0) % l0;
            v            = input.load(row_base + ch0 + c, tw);
        } else {
            if (tt >= 0 && tt < l0) {
                v = input.load(row_base + ch0 + c, tt);
            } else if (Mode == FDMTMode::kValid && tt < 0 &&
                       tt >= -a.dt_max_final && hist0 != nullptr) {
                v = hist0[((ch0 + c) * a.dt_max_final) + a.dt_max_final + tt];
            }
        }
        x[i] = v;
    });
    blk.sync();

    // Phase 2: level 0 into buffer A.
    {
        const int w0  = width(0);
        const int lo0 = lo(0);
        float* dst    = buf_a;
        blk.for_each(nch * w0, [&](int i) {
            const int c    = i / w0;
            const int j    = i % w0;
            const int t    = lo0 + j;
            const int ch   = ch0 + c;
            const int coff = a.coord_offset0[ch];
            const int r0   = coff - row0(0);
            const int ndt  = a.ndt0[ch];
            const int* dts = a.dt_grid0 + coff;
            if (!fdmt_in_range<Mode>(t, l0)) {
                for (int k = 0; k < ndt; ++k) {
                    dst[((r0 + k) * w0) + j] = 0.0F;
                }
                return;
            }
            const float* xc = x + (c * x_width);
            fdmt_init_column<UseBoxSmearing>(
                dts[0], dts[ndt - 1], t, [&](int tt) { return xc[tt - x_lo]; },
                [&](int i_dt, float v) { dst[((r0 + i_dt) * w0) + j] = v; });
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
        const int n_l    = a.level_nsamps[l];
        const int* cl    = a.coords + (kCI * (a.coords_offset[l] + row0(l)));
        blk.for_each(nrows(l) * w_out, [&](int i) {
            const int r = i / w_out;
            const int t = lo_out + (i % w_out);
            dst[i]      = fdmt_in_range<Mode>(t, n_l)
                              ? fdmt_fused_merge_cell<Mode>(
                               in, w_in, lo_in, cl + (kCI * r), t, thist_in)
                              : 0.0F;
        });
        if constexpr (Mode == FDMTMode::kValid) {
            fdmt_fused_tree_history(blk, in, w_in, lo_in, cl, nrows(l),
                                    lvl[(kLI * l) + 3], a.level_nsamps[l - 1],
                                    tile, t0, t_end, thist_in, thist_out);
        }
        blk.sync();
    }

    // Phase 4: level F straight to its state buffer.
    {
        const float* in = buf(fuse - 1);
        const int w_in  = width(fuse - 1);
        const int lo_in = lo(fuse - 1);
        const int n_f   = a.level_nsamps[fuse];
        const int tn    = a.tile_nsamps;
        const int first = row0(fuse);
        const int* cl   = a.coords + (kCI * (a.coords_offset[fuse] + first));
        blk.for_each(nrows(fuse) * tn, [&](int i) {
            const int r = i / tn;
            const int t = t0 + (i % tn);
            if (t >= n_f) {
                return;
            }
            out[(static_cast<int64_t>(first + r) * n_f) + t] =
                static_cast<TOut>(fdmt_fused_merge_cell<Mode>(
                    in, w_in, lo_in, cl + (kCI * r), t, thist_in));
        });
        if constexpr (Mode == FDMTMode::kValid) {
            fdmt_fused_tree_history(
                blk, in, w_in, lo_in, cl, nrows(fuse), lvl[(kLI * fuse) + 3],
                a.level_nsamps[fuse - 1], tile, t0, t_end, thist_in, thist_out);
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
        SizeType cap_a = 0;
        SizeType cap_b = 0;
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
    const auto f     = static_cast<int>(fuse);
    p.fuse           = f;
    p.ngroups        = to_int(pc.state_shape[fuse].nchans);
    const auto nsubs = pc.state_shape[0].nchans;
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
    // sub-bands [g * 2^(F-l), (g+1) * 2^(F-l)), clipped (odd counts end in a
    // single-child copy sub-band that still descends from floor(j / 2)).
    const auto range = [&](SizeType l, SizeType g) {
        const auto& grids    = pc.grids[l];
        const SizeType w     = SizeType{1} << (fuse - l);
        const SizeType first = g * w;
        const SizeType last  = std::min(first + w, grids.size()) - 1;
        return std::pair<SizeType, SizeType>{grids[first].coord_offset,
                                             grids[last].coord_offset +
                                                 grids[last].ndt};
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
        const SizeType ch_begin = g << fuse;
        const SizeType ch_end =
            std::min(ch_begin + (SizeType{1} << fuse), nsubs);
        int s_halo = 0;
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
inline FDMTFusedTileArgs make_fused_tile_args(const FDMTFusedTilePlan& plan,
                                              int tile,
                                              int dt_max_final,
                                              const int* group_info,
                                              const int* coords,
                                              const int* dt_grid0,
                                              const int* ndt0,
                                              const int* coord_offset0) {
    FDMTFusedTileArgs a{};
    a.fuse         = plan.fuse;
    a.tile_nsamps  = tile;
    a.cap_a        = static_cast<int>(plan.smem_floats(tile).first);
    a.nsubs        = static_cast<int>(plan.ndt0.size());
    a.dt_max_final = dt_max_final;
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
