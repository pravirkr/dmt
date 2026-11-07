#pragma once

#include <cstdint>
#include <type_traits>

#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_direct.hpp"
#include "dmt/gpu_compat.cuh"

// Device side of the direct low levels (see fdmt_direct.hpp). Per chunk:
// kernel_direct_prefix writes every channel's running sum over the input
// window (history + chunk) once; kernel_direct_level stages a subband's
// windows in shared memory and evaluates its level-K rows.

namespace dmt::algorithms::detail {

/// Output samples per block of kernel_direct_level, warps per block, and
/// level-K rows per warp (accumulated in registers across channel passes).
inline constexpr int kDirectTile        = 512;
inline constexpr int kDirectWarps       = 8;
inline constexpr int kDirectRowsPerWarp = 2;

/// Threads per block of kernel_direct_prefix, and samples each sums per
/// segment (a segment of kPrefixThreads * kPrefixRun samples is staged in
/// shared memory).
inline constexpr int kPrefixThreads = 256;
inline constexpr int kPrefixRun     = 8;

/**
 * @brief Running sums of every channel over [-window, active) of the chunk:
 * prefix[row][j + 1] = sum of x(-window .. -window + j), prefix[row][0] = 0,
 * rows (beam * nchans + c) `pitch` apart. Samples before the chunk come from
 * `win_in` (the last `window` input samples of each row; zeros if null).
 * One block per row (grid (nchans, 1, beams)): each segment is staged in
 * shared memory, every thread sums a run of kPrefixRun samples, and a
 * block-wide scan of the runs (plus the previous segments' total) gives the
 * running sums.
 */
template <typename Input, typename TP = float>
__global__ __launch_bounds__(kPrefixThreads) void kernel_direct_prefix(
    const Input waterfall,
    const float* __restrict__ win_in,
    int window,
    int nchans,
    int active,
    TP* __restrict__ prefix,
    int pitch) {
    // TP: float, or uint32_t for packed input (exact integer sums; the
    // engine checks (window + chunk) * maxval fits).
    constexpr int kWarp    = 32;
    constexpr int kSegment = kPrefixThreads * kPrefixRun;
    constexpr int kWarps   = kPrefixThreads / kWarp;
    __shared__ TP seg[kSegment];
    __shared__ TP warp_sums[kWarps];
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % kWarp;
    const int warp = tid / kWarp;
    const auto row = (static_cast<int64_t>(blockIdx.z) * nchans) +
                     static_cast<int64_t>(blockIdx.x);
    TP* p          = prefix + (row * pitch);
    if (tid == 0) {
        p[0] = TP{0};
    }
    const int n = window + active;
    TP carry    = TP{0};
    for (int s0 = 0; s0 < n; s0 += kSegment) {
        for (int j = tid; j < kSegment; j += kPrefixThreads) {
            const int tau = s0 + j - window;
            TP v          = TP{0};
            if (s0 + j < n) {
                if (tau >= 0) {
                    v = static_cast<TP>(waterfall.load(row, tau));
                } else if (win_in != nullptr) {
                    v = static_cast<TP>(win_in[(row * window) + s0 + j]);
                }
            }
            seg[j] = v;
        }
        __syncthreads();
        // This thread's run, then an exclusive scan of the runs.
        TP run = TP{0};
#pragma unroll
        for (int k = 0; k < kPrefixRun; ++k) {
            run += seg[(tid * kPrefixRun) + k];
        }
        TP incl = run;
#pragma unroll
        for (int o = 1; o < kWarp; o <<= 1) {
            const TP u = __shfl_up_sync(0xffffffffU, incl, o);
            if (lane >= o) {
                incl += u;
            }
        }
        if (lane == kWarp - 1) {
            warp_sums[warp] = incl;
        }
        __syncthreads();
        TP before = carry;
        TP total  = TP{0};
        for (int w = 0; w < kWarps; ++w) {
            before += w < warp ? warp_sums[w] : TP{0};
            total += warp_sums[w];
        }
        TP acc = before + incl - run;
#pragma unroll
        for (int k = 0; k < kPrefixRun; ++k) {
            acc += seg[(tid * kPrefixRun) + k];
            seg[(tid * kPrefixRun) + k] = acc;
        }
        __syncthreads();
        for (int j = tid; j < kSegment && s0 + j < n; j += kPrefixThreads) {
            p[s0 + j + 1] = seg[j];
        }
        carry += total;
        __syncthreads();
    }
}

/**
 * @brief Level-K rows of one chunk from the running sums. Block: kRows rows
 * of one subband (blockIdx.y) and one kDirectTile tile of one beam
 * (blockIdx.z = beam * ntiles + tile). The subband's channels are staged in
 * passes of `cpass` channels; each lane accumulates its samples of every
 * row in registers across the passes.
 */
template <typename TP = float, typename TOut = float>
__global__ __launch_bounds__(kDirectWarps * 32) void kernel_direct_level(
    const TP* __restrict__ prefix,
    int pitch,
    int window,
    int nchans,
    const FDMTDirectSub* __restrict__ subs,
    const int32_t* __restrict__ leaf_off,
    const FDMTDirectLeaf* __restrict__ leaves,
    TOut* __restrict__ out,
    int row_stride,
    int64_t out_beam_stride,
    int cpass,
    int active,
    int ntiles) {
    // Float, or packed input: uint32_t running sums, integer leaf weights
    // (the exact dyadic box weights, held exactly in the float `w`) and
    // uint32_t / uint64_t rows; integer sums are exact.
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    TP* smem               = reinterpret_cast<TP*>(smem_bytes);
    constexpr int kWarp    = 32;
    constexpr int kPer     = kDirectTile / kWarp;
    constexpr int kRows    = kDirectRowsPerWarp * kDirectWarps;
    const FDMTDirectSub sb = subs[blockIdx.y];
    const int r0           = static_cast<int>(blockIdx.x) * kRows;
    const auto beam        = static_cast<int64_t>(blockIdx.z) / ntiles;
    const int t0 = (static_cast<int>(blockIdx.z) % ntiles) * kDirectTile;
    if (r0 >= sb.nrows || t0 >= active) {
        return;
    }
    const int n     = min(kDirectTile, active - t0);
    const int width = sb.span + n + 1; // staged running sums per channel
    const int lane  = static_cast<int>(threadIdx.x) % kWarp;
    const int warp  = static_cast<int>(threadIdx.x) / kWarp;

    TOut acc[kDirectRowsPerWarp][kPer];
    int cursor[kDirectRowsPerWarp];
    int row_end[kDirectRowsPerWarp];
#pragma unroll
    for (int q = 0; q < kDirectRowsPerWarp; ++q) {
        const int r = r0 + warp + (q * kDirectWarps);
        cursor[q]   = r < sb.nrows ? leaf_off[sb.row0 + r] : 0;
        row_end[q]  = r < sb.nrows ? leaf_off[sb.row0 + r + 1] : 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            acc[q][j] = TOut{0};
        }
    }
    for (int cb = 0; cb < sb.nc; cb += cpass) {
        const int ncp = min(cpass, sb.nc - cb);
        if (cb > 0) {
            __syncthreads();
        }
        for (int c = 0; c < ncp; ++c) {
            const TP* src = prefix +
                            ((beam * nchans + sb.c0 + cb + c) * pitch) +
                            (window - sb.span) + t0;
            for (int j = static_cast<int>(threadIdx.x); j < width;
                 j += static_cast<int>(blockDim.x)) {
                smem[(c * width) + j] = src[j];
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < kDirectRowsPerWarp; ++q) {
            for (; cursor[q] < row_end[q]; ++cursor[q]) {
                const FDMTDirectLeaf lf = leaves[cursor[q]];
                if (lf.c >= cb + ncp) {
                    break;
                }
                // Sample t reads the window at e = t + span - shift.
                const TP* p =
                    smem + ((lf.c - cb) * width) + sb.span - lf.shift + lane;
                if constexpr (std::is_same_v<TOut, float>) {
#pragma unroll
                    for (int j = 0; j < kPer; ++j) {
                        acc[q][j] +=
                            lf.w * (p[(kWarp * j) + 1] - p[(kWarp * j) - lf.s]);
                    }
                } else {
                    const auto w = static_cast<TOut>(lf.w);
#pragma unroll
                    for (int j = 0; j < kPer; ++j) {
                        acc[q][j] +=
                            w * static_cast<TOut>(p[(kWarp * j) + 1] -
                                                  p[(kWarp * j) - lf.s]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kDirectRowsPerWarp; ++q) {
        const int r = r0 + warp + (q * kDirectWarps);
        if (r >= sb.nrows) {
            continue;
        }
        TOut* o = out + (beam * out_beam_stride) +
                  (static_cast<int64_t>(sb.row0 + r) * row_stride) + t0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (lane + (kWarp * j) < n) {
                o[lane + (kWarp * j)] = acc[q][j];
            }
        }
    }
}

/**
 * @brief Rebuilds one level's tree history (ring layout, see
 * kernel_execute_iter) from the input window: coordinate i's head row over
 * the `delay` samples before stream position `pos`, its leaves summed as
 * the level-0 initialisation sums a box. Grid (delay / 256, coords, beams).
 */
__global__ void
kernel_direct_rebuild_history(const float* __restrict__ win,
                              int window,
                              int nchans,
                              const FDMTDirectHistCoord* __restrict__ coords,
                              int ncoords,
                              const FDMTDirectLeaf* __restrict__ leaves,
                              float* __restrict__ ring,
                              int64_t ring_stride,
                              int64_t ring_base,
                              uint64_t pos) {
    const auto k    = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto beam = static_cast<int64_t>(blockIdx.z);
    for (int i = static_cast<int>(blockIdx.y); i < ncoords;
         i += static_cast<int>(gridDim.y)) {
        const FDMTDirectHistCoord hc = coords[i];
        if (k >= hc.delay) {
            continue;
        }
        const int tau = k - hc.delay; // sample time relative to pos
        float value   = 0.0F;
        for (int m = hc.leaf_lo; m < hc.leaf_hi; ++m) {
            const FDMTDirectLeaf lf = leaves[m];
            const float* x = win + ((beam * nchans + lf.c) * window) + window +
                             tau - lf.shift;
            float sum      = x[0];
            for (int j = 1; j <= lf.s; ++j) {
                sum += x[-j];
            }
            value += sum * lf.w;
        }
        int slot = static_cast<int>(pos % static_cast<uint64_t>(hc.delay)) + k;
        slot -= slot >= hc.delay ? hc.delay : 0;
        ring[(beam * ring_stride) + hc.hist_offset - ring_base + slot] = value;
    }
}

/// Stream sample u's slot in a ring of h samples (u may precede the stream:
/// its zero-initialised slots).
__device__ __forceinline__ int64_t direct_ring_slot(int64_t u, int32_t h) {
    const int64_t slot = u % h;
    return slot < 0 ? slot + h : slot;
}

/**
 * @brief Appends a chunk's `active` samples of every channel into the ragged
 * per-channel circular ring `win_ring` at stream position `pos`.
 */
template <typename Input, typename TW = float>
__global__ void kernel_direct_write_ring(TW* __restrict__ win_ring,
                                         int64_t ring_beam_stride,
                                         const int64_t* __restrict__ ring_off,
                                         const int32_t* __restrict__ ring_len,
                                         const Input waterfall,
                                         int nchans,
                                         int active,
                                         uint64_t pos) {
    const int t     = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int c     = static_cast<int>(blockIdx.y);
    const auto beam = static_cast<int64_t>(blockIdx.z);
    if (t >= active || c >= nchans) {
        return;
    }
    const int32_t h = ring_len[c];
    const auto slot = direct_ring_slot(static_cast<int64_t>(pos) + t, h);
    win_ring[(beam * ring_beam_stride) + ring_off[c] + slot] =
        static_cast<TW>(waterfall.load((beam * nchans) + c, t));
}

/**
 * @brief Skewed-window level-K evaluation from the ragged per-channel circular
 * ring `win_ring`: each block stages its slice's shifted channel windows into
 * shared memory, computes warp-parallel prefix sums in shared memory, and
 * accumulates the slice's rows. Grid: (nsubs, ntiles, beams).
 */
/// Packed input to a float-output engine (FDMTDirectRound): the tree's
/// level-0 box means of packed input are the integer box sums rounded by
/// their recipe on 2^-F (fdmt_box_weights.hpp), not the float weights, so
/// each leaf applies the recipe to its exact uint32 box sum; rows accumulate
/// in integer units and are stored as float times `unit` (2^-F, or 1
/// without box smearing, when `round` is null and the leaf is one sample).
struct FDMTDirectRound {
    const FDMTBoxRound* round{nullptr};
    int round_u64{0};
    float unit{1.0F};
};

template <typename TW     = float,
          typename TP     = float,
          typename TOut   = float,
          bool kRound     = false,
          typename TStore = TOut>
__global__ __launch_bounds__(kDirectWarps * 32) void kernel_direct_level_skewed(
    const TW* __restrict__ win_ring,
    int64_t ring_beam_stride,
    const int64_t* __restrict__ ring_off,
    const int32_t* __restrict__ ring_len,
    const FDMTDirectSub* __restrict__ subs,
    const int32_t* __restrict__ leaf_off,
    const FDMTDirectLeaf* __restrict__ leaves,
    TStore* __restrict__ out,
    int row_stride,
    int64_t out_beam_stride,
    int cpass,
    uint64_t pos,
    int active,
    FDMTDirectRound rnd = {}) {
    static_assert(!kRound || (std::is_same_v<TP, uint32_t> &&
                              std::is_same_v<TOut, uint32_t> &&
                              std::is_same_v<TStore, float>),
                  "packed rounding: uint32 sums and rows, float output");
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    TP* smem               = reinterpret_cast<TP*>(smem_bytes);
    constexpr int kWarp    = 32;
    constexpr int kPer     = kDirectTile / kWarp;
    const FDMTDirectSub sb = subs[blockIdx.x];
    const int t0           = static_cast<int>(blockIdx.y) * kDirectTile;
    const auto beam        = static_cast<int64_t>(blockIdx.z);
    const int n_total      = active + sb.extra;
    if (sb.nrows == 0 || t0 >= n_total) {
        return;
    }
    const int n           = min(kDirectTile, n_total - t0);
    const int width       = sb.span + n + 1;
    const int lane        = static_cast<int>(threadIdx.x) % kWarp;
    const int warp        = static_cast<int>(threadIdx.x) / kWarp;
    const int64_t pos_end = static_cast<int64_t>(pos) + active;

    TOut acc[kDirectRowsPerWarp][kPer];
    int cursor[kDirectRowsPerWarp];
    int row_end[kDirectRowsPerWarp];
#pragma unroll
    for (int q = 0; q < kDirectRowsPerWarp; ++q) {
        const int r = warp + (q * kDirectWarps);
        cursor[q]   = r < sb.nrows ? leaf_off[sb.row0 + r] : 0;
        row_end[q]  = r < sb.nrows ? leaf_off[sb.row0 + r + 1] : 0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            acc[q][j] = TOut{0};
        }
    }
    for (int cb = 0; cb < sb.nc; cb += cpass) {
        const int ncp = min(cpass, sb.nc - cb);
        if (cb > 0) {
            __syncthreads();
        }
        for (int c = 0; c < ncp; ++c) {
            const int ch = sb.c0 + cb + c;
            const TW* chan_ring =
                win_ring + (beam * ring_beam_stride) + ring_off[ch];
            const int32_t h = ring_len[ch];
            const int64_t u0 =
                static_cast<int64_t>(pos) - sb.c_min[cb + c] - sb.span + t0;
            const auto slot0 = static_cast<int32_t>(direct_ring_slot(u0, h));
            // The slice's span is its widest channel's, so a narrower channel
            // stages samples older than its ring keeps (u < pos_end - h, its
            // slots already rewritten): stage zeros there. No leaf reads
            // them (each sums inside its own reach), so the box sums are
            // unchanged, and the slot stays inside this channel's ring.
            const int64_t u_old = pos_end - h;
            for (int j = static_cast<int>(threadIdx.x); j < width - 1;
                 j += static_cast<int>(blockDim.x)) {
                const int64_t u = u0 + j;
                TP val          = TP{0};
                if (u >= 0 && u < pos_end && u >= u_old) {
                    int64_t slot = static_cast<int64_t>(slot0) + j;
                    if (slot >= h) {
                        slot -= h;
                        if (slot >= h) {
                            slot %= h;
                        }
                    }
                    val = static_cast<TP>(chan_ring[slot]);
                }
                smem[(c * width) + 1 + j] = val;
            }
        }
        __syncthreads();
        for (int c = warp; c < ncp; c += kDirectWarps) {
            TP* s = smem + (c * width);
            if (lane == 0) {
                s[0] = TP{0};
            }
            TP carry = TP{0};
            for (int base = 1; base < width; base += kWarp) {
                const int idx = base + lane;
                TP v          = idx < width ? s[idx] : TP{0};
#pragma unroll
                for (int o = 1; o < kWarp; o <<= 1) {
                    const TP u = __shfl_up_sync(0xffffffffU, v, o);
                    if (lane >= o) {
                        v += u;
                    }
                }
                v += carry;
                if (idx < width) {
                    s[idx] = v;
                }
                carry = __shfl_sync(0xffffffffU, v, kWarp - 1);
            }
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < kDirectRowsPerWarp; ++q) {
            for (; cursor[q] < row_end[q]; ++cursor[q]) {
                const FDMTDirectLeaf lf = leaves[cursor[q]];
                if (lf.c >= cb + ncp) {
                    break;
                }
                const TP* p =
                    smem + ((lf.c - cb) * width) + sb.span - lf.shift + lane;
                if constexpr (kRound) {
                    if (rnd.round != nullptr) {
                        const FDMTBoxRound r = rnd.round[lf.s];
#pragma unroll
                        for (int j = 0; j < kPer; ++j) {
                            const uint32_t sum =
                                p[(kWarp * j) + 1] - p[(kWarp * j) - lf.s];
                            acc[q][j] +=
                                rnd.round_u64 != 0
                                    ? static_cast<uint32_t>(
                                          box_round_apply<uint64_t>(sum, r))
                                    : box_round_apply<uint32_t>(sum, r);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < kPer; ++j) {
                            acc[q][j] +=
                                p[(kWarp * j) + 1] - p[(kWarp * j) - lf.s];
                        }
                    }
                } else if constexpr (std::is_same_v<TOut, float>) {
#pragma unroll
                    for (int j = 0; j < kPer; ++j) {
                        acc[q][j] +=
                            lf.w * (p[(kWarp * j) + 1] - p[(kWarp * j) - lf.s]);
                    }
                } else {
                    const auto w = static_cast<TOut>(lf.w);
#pragma unroll
                    for (int j = 0; j < kPer; ++j) {
                        acc[q][j] +=
                            w * static_cast<TOut>(p[(kWarp * j) + 1] -
                                                  p[(kWarp * j) - lf.s]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kDirectRowsPerWarp; ++q) {
        const int r = warp + (q * kDirectWarps);
        if (r >= sb.nrows) {
            continue;
        }
        TStore* o = out + (beam * out_beam_stride) + sb.out_off +
                    (static_cast<int64_t>(r) * (row_stride + sb.extra)) + t0;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            if (lane + (kWarp * j) < n) {
                if constexpr (kRound) {
                    o[lane + (kWarp * j)] =
                        static_cast<float>(acc[q][j]) * rnd.unit;
                } else {
                    o[lane + (kWarp * j)] = acc[q][j];
                }
            }
        }
    }
}

/// Samples per lane of kernel_direct_top (a warp covers 32 * this).
inline constexpr int kTopSampsPerLane = 8;

/**
 * @brief The direct top (fdmt_direct.hpp): final trial f over this chunk as
 * the sum of its level-k rows at their shifts, from the level's state for
 * t >= shift and the rows' rings (stream position `pos` = chunk start)
 * before. One warp per trial and 32 * kTopSampsPerLane samples; grid
 * (trials / warps, samples, beams).
 */
template <typename T = float>
__global__ __launch_bounds__(256) void kernel_direct_top(
    const T* __restrict__ state,
    int64_t state_beam_stride,
    int row_stride,
    const T* __restrict__ ring,
    int64_t ring_beam_stride,
    const int64_t* __restrict__ ring_off,
    const int32_t* __restrict__ ring_len,
    const int32_t* __restrict__ leaf_off,
    const FDMTDirectTopLeaf* __restrict__ leaves,
    const int32_t* __restrict__ out_row,
    int ntrials,
    T* __restrict__ out,
    int64_t out_beam_stride,
    int out_row_stride,
    uint64_t pos,
    int active) {
    constexpr int kWarp = 32;
    const int lane      = static_cast<int>(threadIdx.x) % kWarp;
    const int f =
        static_cast<int>(((blockIdx.x * blockDim.x) + threadIdx.x) / kWarp);
    if (f >= ntrials) {
        return;
    }
    const auto beam = static_cast<int64_t>(blockIdx.z);
    const int t0    = static_cast<int>(blockIdx.y) * kWarp * kTopSampsPerLane;
    const T* st     = state + (beam * state_beam_stride);
    const T* rg     = ring + (beam * ring_beam_stride);
    T acc[kTopSampsPerLane];
#pragma unroll
    for (int j = 0; j < kTopSampsPerLane; ++j) {
        acc[j] = T{0};
    }
    for (int m = leaf_off[f]; m < leaf_off[f + 1]; ++m) {
        const FDMTDirectTopLeaf lf = leaves[m];
        const T* row   = st + (static_cast<int64_t>(lf.row) * row_stride);
        const int tau0 = t0 + lane - lf.shift;
        if (tau0 >= 0) { // the whole tile reads the state
#pragma unroll
            for (int j = 0; j < kTopSampsPerLane; ++j) {
                const int t = t0 + lane + (kWarp * j);
                if (t < active) {
                    acc[j] += row[tau0 + (kWarp * j)];
                }
            }
            continue;
        }
        // Ring slot of sample tau0, then stepped by a warp's stride: one
        // 64-bit remainder per leaf instead of one per sample.
        const int32_t h = ring_len[lf.row];
        const T* rr     = rg + ring_off[lf.row];
        auto slot       = static_cast<int32_t>(
            direct_ring_slot(static_cast<int64_t>(pos) + tau0, h));
        const int32_t step = kWarp % h;
#pragma unroll
        for (int j = 0; j < kTopSampsPerLane; ++j) {
            const int t   = t0 + lane + (kWarp * j);
            const int tau = tau0 + (kWarp * j);
            if (t < active) {
                acc[j] += tau >= 0 ? row[tau] : rr[slot];
            }
            slot += step;
            slot -= slot >= h ? h : 0;
        }
    }
    T* o = out + (beam * out_beam_stride) +
           (static_cast<int64_t>(out_row[f]) * out_row_stride);
#pragma unroll
    for (int j = 0; j < kTopSampsPerLane; ++j) {
        const int t = t0 + lane + (kWarp * j);
        if (t < active) {
            o[t] = acc[j];
        }
    }
}

/**
 * @brief Stores each level-k row's newest min(ring_len, active) samples of
 * this chunk into its ring (after the chunk's trials have read it). One warp
 * per (row, segment of kAdvanceSegment-like 256 samples); grid (rows /
 * warps, segments, beams).
 */
template <typename T = float>
__global__ __launch_bounds__(256) void kernel_direct_top_ring(
    const T* __restrict__ state,
    int64_t state_beam_stride,
    int row_stride,
    T* __restrict__ ring,
    int64_t ring_beam_stride,
    const int64_t* __restrict__ ring_off,
    const int32_t* __restrict__ ring_len,
    int nrows,
    uint64_t pos,
    int active) {
    constexpr int kWarp    = 32;
    constexpr int kSegment = 256;
    const int lane         = static_cast<int>(threadIdx.x) % kWarp;
    const int r =
        static_cast<int>(((blockIdx.x * blockDim.x) + threadIdx.x) / kWarp);
    if (r >= nrows) {
        return;
    }
    const int32_t h  = ring_len[r];
    const int newest = h < active ? h : active;
    const int seg0   = static_cast<int>(blockIdx.y) * kSegment;
    if (seg0 >= newest) {
        return;
    }
    const auto beam = static_cast<int64_t>(blockIdx.z);
    const T* row    = state + (beam * state_beam_stride) +
                      (static_cast<int64_t>(r) * row_stride);
    T* rr           = ring + (beam * ring_beam_stride) + ring_off[r];
    const int first = active - newest;
    const int end   = seg0 + kSegment < newest ? seg0 + kSegment : newest;
    auto slot       = static_cast<int32_t>(
        direct_ring_slot(static_cast<int64_t>(pos) + first + seg0 + lane, h));
    const int32_t step = kWarp % h;
    for (int j = seg0 + lane; j < end; j += kWarp) {
        rr[slot] = row[first + j];
        slot += step;
        slot -= slot >= h ? h : 0;
    }
}

/**
 * @brief Target-side ring direct top: one 256-thread block exclusively owns
 * final trial f (blockIdx.x), accumulates its level-k leaves into trial f's
 * ring at (pos + u + lf.shift), and reads out and zeroes the completed slots
 * [pos, pos + active).
 */
template <typename T = float>
__global__ __launch_bounds__(256) void kernel_direct_top_target(
    const T* __restrict__ state,
    int64_t state_beam_stride,
    int row_stride,
    T* __restrict__ ring,
    int64_t ring_beam_stride,
    const int64_t* __restrict__ ring_off,
    const int32_t* __restrict__ ring_len,
    const int32_t* __restrict__ leaf_off,
    const FDMTDirectTopLeaf* __restrict__ leaves,
    const int32_t* __restrict__ out_row,
    int ntrials,
    T* __restrict__ out,
    int64_t out_beam_stride,
    int out_row_stride,
    uint64_t pos,
    int active,
    int m_lo          = 0,
    int m_hi          = -1,
    bool finalize_out = true) {
    constexpr int kBlock = 256;
    const int tid        = static_cast<int>(threadIdx.x);
    const int f          = static_cast<int>(blockIdx.x);
    if (f >= ntrials) {
        return;
    }
    const auto beam    = static_cast<int64_t>(blockIdx.z);
    const int32_t h    = ring_len[f];
    const T* st        = state + (beam * state_beam_stride);
    T* tr              = ring + (beam * ring_beam_stride) + ring_off[f];
    const int32_t step = kBlock % h;
    const int base_m   = leaf_off[f];
    const int end_m    = (m_hi >= 0) ? (base_m + m_hi) : leaf_off[f + 1];
    for (int m = base_m + m_lo; m < end_m; ++m) {
        const FDMTDirectTopLeaf lf = leaves[m];
        const T* row = st + (static_cast<int64_t>(lf.row) * row_stride);
        auto slot    = static_cast<int32_t>(
            direct_ring_slot(static_cast<int64_t>(pos) + lf.shift + tid, h));
        for (int u = tid; u < active; u += kBlock) {
            tr[slot] += row[u];
            slot += step;
            slot -= slot >= h ? h : 0;
        }
        __syncthreads();
    }
    if (!finalize_out) {
        return;
    }
    T* o      = out + (beam * out_beam_stride) +
                (static_cast<int64_t>(out_row[f]) * out_row_stride);
    auto slot = static_cast<int32_t>(
        direct_ring_slot(static_cast<int64_t>(pos) + tid, h));
    for (int u = tid; u < active; u += kBlock) {
        o[u]     = tr[slot];
        tr[slot] = T{0};
        slot += step;
        slot -= slot >= h ? h : 0;
    }
}

/**
 * @brief Rebuilds one level's tree history (ring layout) above the direct
 * top from the level-k rings, at stream position `pos` (as
 * kernel_direct_rebuild_history does for the low levels).
 */
__global__ void kernel_direct_top_rebuild_history(
    const float* __restrict__ top_ring,
    int64_t top_beam_stride,
    const int64_t* __restrict__ ring_off,
    const int32_t* __restrict__ ring_len,
    const FDMTDirectHistCoord* __restrict__ coords,
    int ncoords,
    const FDMTDirectTopLeaf* __restrict__ leaves,
    float* __restrict__ ring,
    int64_t ring_stride,
    int64_t ring_base,
    uint64_t pos) {
    const auto k    = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto beam = static_cast<int64_t>(blockIdx.z);
    const float* tr = top_ring + (beam * top_beam_stride);
    for (int i = static_cast<int>(blockIdx.y); i < ncoords;
         i += static_cast<int>(gridDim.y)) {
        const FDMTDirectHistCoord hc = coords[i];
        if (k >= hc.delay) {
            continue;
        }
        const int64_t u = static_cast<int64_t>(pos) + k - hc.delay;
        float value     = 0.0F;
        for (int m = hc.leaf_lo; m < hc.leaf_hi; ++m) {
            const FDMTDirectTopLeaf lf = leaves[m];
            value += tr[ring_off[lf.row] +
                        direct_ring_slot(u - lf.shift, ring_len[lf.row])];
        }
        int slot = static_cast<int>(pos % static_cast<uint64_t>(hc.delay)) + k;
        slot -= slot >= hc.delay ? hc.delay : 0;
        ring[(beam * ring_stride) + hc.hist_offset - ring_base + slot] = value;
    }
}

} // namespace dmt::algorithms::detail
