#pragma once

#include <cstdint>

#include "dmt/common/types.hpp"
#include "dmt/ddmt_kernel.cuh"
#include "dmt/sdmt_gpu_plan.hpp"

/**
 * @file sdmt_kernel.cuh
 * @brief SDMT (exact shared-sum dedispersion) GPU kernel.
 *
 * Runs the per-(DM tile, subband) programs of sdmt_gpu_plan.hpp. Integer
 * results equal DDMT exactly; float results differ only by the order of
 * additions.
 */

namespace dmt::algorithms::sdmt_gpu {

using ddmt_gpu::DDMTAcc;
using ddmt_gpu::DDMTSegments;
using ddmt_gpu::fetch_sample;

/// Arena element: float for float input, uint32 partial sums otherwise.
template <unsigned NBITS>
using Elem = std::conditional_t<NBITS == 32, float, uint32_t>;

/// Bytes of the per-block trial-address table (first in shared memory).
DMT_HD constexpr SizeType trial_table_bytes(int tdm) {
    return ((static_cast<SizeType>(tdm) * sizeof(int) + 15) / 16) * 16;
}

/// Bytes of the per-subband node table (after the trial table).
DMT_HD constexpr SizeType node_table_bytes(int max_nodes) {
    return static_cast<SizeType>(max_nodes) * sizeof(Node);
}

/// Dynamic shared memory of a plan: trial table, node table, arena.
template <unsigned NBITS>
DMT_HD constexpr SizeType smem_bytes(int tdm, int max_nodes, int arena) {
    return trial_table_bytes(tdm) + node_table_bytes(max_nodes) +
           (static_cast<SizeType>(arena) * sizeof(Elem<NBITS>));
}

/// Element @p k of a row whose samples start at @p rb.
template <unsigned NBITS>
__device__ __forceinline__ Elem<NBITS> load_elem(const uint8_t* rb, int k) {
    if constexpr (NBITS == 32) {
        return reinterpret_cast<const float*>(rb)[k];
    } else if constexpr (NBITS == 16) {
        // Device rows are 2-byte aligned (see fetch_sample).
        return reinterpret_cast<const uint16_t*>(rb)[k];
    } else if constexpr (NBITS == 8) {
        return rb[k];
    } else {
        constexpr int kPer   = 8 / NBITS;
        constexpr uint32_t M = (1U << NBITS) - 1U;
        return (static_cast<uint32_t>(rb[k / kPer]) >> ((k % kPer) * NBITS)) &
               M;
    }
}

/// Window sample i (stream sample g0 + i of @p row): from the resolved row
/// @p rb when the window lies in one segment, else through both segments.
template <unsigned NBITS>
__device__ __forceinline__ Elem<NBITS> window_sample(const DDMTSegments& seg,
                                                     SizeType row,
                                                     const uint8_t* rb,
                                                     int idx0,
                                                     int g0,
                                                     int i) {
    if (rb == nullptr) {
        return static_cast<Elem<NBITS>>(fetch_sample<NBITS>(seg, row, g0 + i));
    }
    return load_elem<NBITS>(rb, idx0 + i);
}

/// Row pointer, row stride and first index of stream samples
/// [g0, g0 + len) of @p row when they lie in one segment; rb = nullptr
/// otherwise (per-sample two-segment reads).
__device__ __forceinline__ void window_source(const DDMTSegments& seg,
                                              SizeType row,
                                              int g0,
                                              int len,
                                              const uint8_t*& rb,
                                              SizeType& row_bytes,
                                              int& idx0) {
    if (g0 + len <= seg.a_len) {
        row_bytes = seg.a_row_bytes;
        rb        = seg.a + (row * row_bytes);
        idx0      = seg.a_off + g0;
    } else if (g0 >= seg.a_len && g0 + len <= seg.total) {
        row_bytes = seg.b_row_bytes;
        rb        = seg.b + (row * row_bytes);
        idx0      = seg.b_off + (g0 - seg.a_len);
    } else {
        rb        = nullptr;
        row_bytes = 0;
        idx0      = 0;
    }
}

/// Node-table ints prefetched per thread, and window samples prefetched
/// per window (the rest load synchronously).
inline constexpr int kNodePrefetch   = 2;
inline constexpr int kWindowPrefetch = 256;

/**
 * @brief SDMT kernel: a block computes 32 * SPW output samples of TY * DPT
 * consecutive DM trials; warp ty owns trials ty*DPT .. ty*DPT + DPT - 1,
 * lane tx the samples tx + 32*s.
 * @details
 * Per subband: stage the channel windows, compute the L1 nodes, then the
 * L2 nodes (a warp per node, lanes along time), and add every trial's L2
 * node into its registers. Consecutive trials of a warp that read the same
 * address reuse the loaded values.
 *
 * Software pipelined: the global loads of subband sb + 1 (its windows,
 * node records and trial addresses) are issued into registers before
 * subband sb is computed, and written to shared memory after it, so their
 * latency overlaps the computation. Launch bounds keep two blocks resident
 * per SM (128 registers per thread at 256 threads); the planner sizes the
 * shared memory for two blocks too.
 */
template <unsigned NBITS, int TY, int DPT, int SPW>
__global__ __launch_bounds__(32 * TY, 512 / (32 * TY)) void sdmt_kernel(
    DDMTSegments seg,
    DDMTAcc<NBITS>* __restrict__ d_out,
    SizeType out_dm_stride,
    SizeType out_beam_stride,
    const SubProg* __restrict__ subs,
    const Window* __restrict__ wins,
    const Node* __restrict__ nodes,
    const int* __restrict__ trials,
    const int* __restrict__ tile_sub,
    int max_nodes,
    int nchans,
    int ndm,
    int n_out) {
    using E           = Elem<NBITS>;
    using Acc         = DDMTAcc<NBITS>;
    constexpr int TDM = TY * DPT;
    constexpr int NT  = 32 * TY;
    constexpr int BT  = 32 * SPW;
    static_assert(TDM <= NT, "one trial address per thread");
    static_assert(NT % kSubband == 0 && kWindowPrefetch % (NT / kSubband) == 0,
                  "whole threads per window");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* taddr = reinterpret_cast<int*>(smem_raw);
    auto* ntab  = reinterpret_cast<int*>(smem_raw + trial_table_bytes(TDM));
    auto* A     = reinterpret_cast<E*>(smem_raw + trial_table_bytes(TDM) +
                                       node_table_bytes(max_nodes));

    const int tx  = static_cast<int>(threadIdx.x);
    const int ty  = static_cast<int>(threadIdx.y);
    const int tid = (ty * 32) + tx;
    const int t0  = static_cast<int>(blockIdx.x) * BT;
    const int dm0 = static_cast<int>(blockIdx.y) * TDM;
    const auto row0 =
        static_cast<SizeType>(blockIdx.z) * static_cast<SizeType>(nchans);
    const int sub0    = tile_sub[blockIdx.y];
    const int nsub    = tile_sub[blockIdx.y + 1] - sub0;
    const SubProg* sp = subs + sub0;
    const auto* nints = reinterpret_cast<const int*>(nodes);

    Acc acc[DPT][SPW];
#pragma unroll
    for (int d = 0; d < DPT; ++d) {
#pragma unroll
        for (int s = 0; s < SPW; ++s) {
            acc[d][s] = 0;
        }
    }

    // Prefetch registers. Window staging: kTPW threads per window, thread
    // (wj, wq) holds elements wq + kTPW * k of window wj. Node-table ints
    // tid + k * NT, and trial address tid.
    constexpr int kTPW = NT / kSubband;
    constexpr int kPW  = kWindowPrefetch / kTPW;
    const int wj       = tid / kTPW;
    const int wq       = tid % kTPW;
    E pw[kPW];
    int pn[kNodePrefetch];
    int pt        = 0;
    auto prefetch = [&](const SubProg& p) {
#pragma unroll
        for (int k = 0; k < kPW; ++k) {
            pw[k] = 0;
        }
        // One segment check for all the subband's windows.
        const uint8_t* rb0;
        SizeType rbytes;
        int idx0;
        window_source(seg, row0, t0 + p.bmin, (p.bmax - p.bmin) + p.wlen, rb0,
                      rbytes, idx0);
        if (rb0 != nullptr && wj < p.nwin) {
            const Window wr   = wins[p.win0 + wj];
            const uint8_t* rb = rb0 + (static_cast<SizeType>(wr.chan) * rbytes);
            const int i0      = idx0 - p.bmin + wr.base + wq;
#pragma unroll
            for (int k = 0; k < kPW; ++k) {
                if (wq + (kTPW * k) < p.wlen) {
                    pw[k] = load_elem<NBITS>(rb, i0 + (kTPW * k));
                }
            }
        }
        const int nn = (p.n1 + p.n2) * 8;
#pragma unroll
        for (int k = 0; k < kNodePrefetch; ++k) {
            const int e = tid + (k * NT);
            pn[k]       = e < nn ? nints[(p.n1_0 * 8) + e] : 0;
        }
        pt = tid < TDM ? trials[p.trial0 + tid] : 0;
    };

    SubProg cur = sp[0];
    SubProg nxt = nsub > 1 ? sp[1] : cur;
    prefetch(cur);

    for (int sb = 0; sb < nsub; ++sb) {
        __syncthreads(); // the previous subband is done with the arena
        {
            // Write the prefetched data; load what did not fit.
            const uint8_t* rb0;
            SizeType rbytes;
            int idx0;
            window_source(seg, row0, t0 + cur.bmin,
                          (cur.bmax - cur.bmin) + cur.wlen, rb0, rbytes, idx0);
            // Windows that straddle the two segments were not prefetched.
            const bool slow = rb0 == nullptr;
            if (!slow && wj < cur.nwin) {
                E* wa = A + (wj * cur.wlen) + wq;
#pragma unroll
                for (int k = 0; k < kPW; ++k) {
                    if (wq + (kTPW * k) < cur.wlen) {
                        wa[kTPW * k] = pw[k];
                    }
                }
            }
            if (slow || cur.wlen > kWindowPrefetch) {
                const int i0 = slow ? 0 : kWindowPrefetch;
                for (int j = ty; j < cur.nwin; j += TY) {
                    const Window wr = wins[cur.win0 + j];
                    const auto row  = row0 + static_cast<SizeType>(wr.chan);
                    const int g0    = t0 + wr.base;
                    const uint8_t* rb;
                    SizeType rbytes;
                    int idx0;
                    window_source(seg, row, g0, cur.wlen, rb, rbytes, idx0);
                    for (int i = i0 + tx; i < cur.wlen; i += 32) {
                        A[(j * cur.wlen) + i] =
                            window_sample<NBITS>(seg, row, rb, idx0, g0, i);
                    }
                }
            }
            const int nn = (cur.n1 + cur.n2) * 8;
#pragma unroll
            for (int k = 0; k < kNodePrefetch; ++k) {
                const int e = tid + (k * NT);
                if (e < nn) {
                    ntab[e] = pn[k];
                }
            }
            for (int e = tid + (kNodePrefetch * NT); e < nn; e += NT) {
                ntab[e] = nints[(cur.n1_0 * 8) + e];
            }
            if (tid < TDM) {
                taddr[tid] = pt;
            }
        }
        __syncthreads();
        const int n1 = cur.n1;
        const int n2 = cur.n2;
        if (sb + 1 < nsub) {
            cur = nxt;
            if (sb + 2 < nsub) {
                nxt = sp[sb + 2];
            }
            prefetch(cur);
        }
        for (int level = 0; level < 2; ++level) {
            const int n0 = level == 0 ? 0 : n1;
            const int nn = level == 0 ? n1 : n2;
            for (int n = ty; n < nn; n += TY) {
                const int* rec = ntab + ((n0 + n) * 8);
                const int4 h   = *reinterpret_cast<const int4*>(rec);
                const int4 sr  = *reinterpret_cast<const int4*>(rec + 4);
                E* dst         = A + h.x;
                const E* s0    = A + sr.x;
                const E* s1    = A + sr.y;
                const E* s2    = A + sr.z;
                const E* s3    = A + sr.w;
                if (h.z == 4) {
                    for (int i = tx; i < h.y; i += 32) {
                        dst[i] = (s0[i] + s1[i]) + (s2[i] + s3[i]);
                    }
                } else if (h.z == 3) {
                    for (int i = tx; i < h.y; i += 32) {
                        dst[i] = (s0[i] + s1[i]) + s2[i];
                    }
                } else {
                    for (int i = tx; i < h.y; i += 32) {
                        dst[i] = s0[i] + s1[i];
                    }
                }
            }
            __syncthreads();
        }
        E w[SPW];
        int prev = -1;
#pragma unroll
        for (int d = 0; d < DPT; ++d) {
            const int a = taddr[(ty * DPT) + d];
            if (a != prev) {
                const E* src = A + a + tx;
#pragma unroll
                for (int s = 0; s < SPW; ++s) {
                    w[s] = src[s * 32];
                }
                prev = a;
            }
#pragma unroll
            for (int s = 0; s < SPW; ++s) {
                acc[d][s] += static_cast<Acc>(w[s]);
            }
        }
    }

    auto* out_b = d_out + (static_cast<SizeType>(blockIdx.z) * out_beam_stride);
#pragma unroll
    for (int d = 0; d < DPT; ++d) {
        const int dm = dm0 + (ty * DPT) + d;
        if (dm >= ndm) {
            continue;
        }
        auto* orow = out_b + (static_cast<SizeType>(dm) * out_dm_stride);
#pragma unroll
        for (int s = 0; s < SPW; ++s) {
            const int t = t0 + tx + (s * 32);
            if (t < n_out) {
                orow[t] = acc[d][s];
            }
        }
    }
}

} // namespace dmt::algorithms::sdmt_gpu
