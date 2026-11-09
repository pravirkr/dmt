#pragma once

// Private: kernels of the CUDA FDMT engine launched from more than one
// of its translation units (fdmt_cuda_engine.cuh).

#include <cstdint>

#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/modes.hpp"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms::detail::fdmt_cuda {

/**
 * @brief Advances the level-0 FIFO history window (the last `hist_len` =
 * dt0_max raw samples of every channel) by one chunk of `nsamps` samples on
 * the GPU, for any chunk length.
 */
template <typename Input>
__global__ void kernel_advance_history_window(float* __restrict__ hist_out,
                                              const float* __restrict__ hist_in,
                                              const Input waterfall,
                                              int nsubs,
                                              int nsamps,
                                              int dt_max_final) {
    // (dt_max_final: the window length, hist_len.)
    const auto t = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (t >= dt_max_final || i_sub >= nsubs) {
        return;
    }
    const auto hist_offset =
        (i_beam * nsubs * dt_max_final) + (i_sub * dt_max_final);
    const auto wf_row = (i_beam * nsubs) + i_sub;

    if (nsamps >= dt_max_final) {
        hist_out[hist_offset + t] =
            waterfall.load(wf_row, nsamps - dt_max_final + t);
    } else {
        if (t < dt_max_final - nsamps) {
            hist_out[hist_offset + t] =
                (hist_in != nullptr) ? hist_in[hist_offset + t + nsamps] : 0.0F;
        } else {
            hist_out[hist_offset + t] =
                waterfall.load(wf_row, t - (dt_max_final - nsamps));
        }
    }
}

/// Threads per block and output samples per thread of kernel_execute_iter.
inline constexpr int kIterThreads    = 256;
inline constexpr int kIterSampsPerTh = 4;
/// Largest grid.y of a launch (the kernels stride beyond it).
inline constexpr int kMaxGridY = 65535;
/// ring_pos of a tree history kept oldest-first (FIFO) instead of a ring.
inline constexpr uint64_t kFifoHistory = UINT64_MAX;

/**
 * @brief Per-level tree merge kernel with multi-beam support. A block covers
 * kIterThreads * kIterSampsPerTh samples of one coordinate (samples strided
 * by kIterThreads, so every load and store stays coalesced); the
 * coordinate's table entries are read once per thread.
 */
template <FDMTMode Mode,
          typename TIn   = float,
          typename TOut  = float,
          typename THist = float>
__global__ __launch_bounds__(kIterThreads) void kernel_execute_iter(
    const TIn* __restrict__ state_in,
    TOut* __restrict__ state_out,
    const plans::FDMTCoordDPtrs coords_sum,
    const plans::FDMTCoordDPtrs coords_copy,
    THist* __restrict__ hist_in,
    int nsamps,
    int ncoords_sum_cur,
    int ncoords_copy_cur,
    int in_state_nelements,
    int out_state_nelements,
    int64_t hist_stride,
    int64_t hist_base,
    uint64_t ring_pos,
    float int_unit,
    int active,
    bool update_ring = false,
    int chunk_nsamps = 0) {
    // `active` > 0: a valid-mode chunk of that many samples in every row
    // (rows keep the strides of the coordinate tables); else each
    // coordinate's own lengths.
    // Tree history (valid mode): a coordinate's slot starts at its
    // hist_offset - hist_base, beams hist_stride apart. With ring_pos ==
    // kFifoHistory it is oldest-first; else a ring holding the head sample
    // of stream time u at u mod delay, ring_pos being this chunk's start.
    const int block_samp0 =
        static_cast<int>(blockIdx.x) * kIterThreads * kIterSampsPerTh;
    if (block_samp0 >= nsamps) {
        return;
    }
    const int isamp0 = block_samp0 + static_cast<int>(threadIdx.x);
    // State units (see fdmt_box_weights.hpp): integer levels hold integers
    // in units of int_unit = 2^-b, float levels and the tree history hold
    // canonical box-mean units. conv() stores a TIn value as TOut (rescaling
    // int -> float), from_hist() a history value; both are exact.
    const auto conv = [int_unit](TIn v) -> TOut {
        if constexpr (std::is_integral_v<TIn> && !std::is_integral_v<TOut>) {
            return static_cast<float>(v) * int_unit;
        } else {
            return static_cast<TOut>(v);
        }
    };
    const auto from_hist = [int_unit](THist h) -> TOut {
        if constexpr (std::is_integral_v<THist>) {
            return static_cast<TOut>(h);
        } else if constexpr (std::is_integral_v<TOut>) {
            return static_cast<TOut>(h * (1.0F / int_unit));
        } else {
            return h;
        }
    };
    const float unit      = std::is_integral_v<TIn> ? int_unit : 1.0F;
    const auto to_history = [unit](TIn value) -> THist {
        if constexpr (std::is_integral_v<THist>) {
            return static_cast<THist>(value);
        } else {
            return static_cast<float>(value) * unit;
        }
    };
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    // Coordinates over grid.y with a stride (levels can hold more than the
    // 65535 rows grid.y allows).
    const int ncoords_max =
        ncoords_sum_cur > ncoords_copy_cur ? ncoords_sum_cur : ncoords_copy_cur;
    for (int i_coord = static_cast<int>(blockIdx.y); i_coord < ncoords_max;
         i_coord += static_cast<int>(gridDim.y)) {
        if (i_coord < ncoords_sum_cur) {
            const int extra = (active > 0 && chunk_nsamps > 0)
                                  ? (coords_sum.nsamps[i_coord] - chunk_nsamps)
                                  : 0;
            const auto nsamps_out =
                active > 0 ? (active + extra) : coords_sum.nsamps[i_coord];
            const auto offset = coords_sum.offset[i_coord];
            const auto nsamps_tail =
                active > 0 ? (active + extra) : coords_sum.tail_nsamps[i_coord];
            TOut* out       = state_out + (i_beam * out_state_nelements) +
                              coords_sum.buf_offset[i_coord];
            const TIn* tail = state_in + (i_beam * in_state_nelements) +
                              coords_sum.tail_buf_offset[i_coord];
            const TIn* head = state_in + (i_beam * in_state_nelements) +
                              coords_sum.head_buf_offset[i_coord];
            THist* hist     = hist_in != nullptr
                                  ? hist_in + (i_beam * hist_stride) +
                                        (static_cast<int64_t>(
                                             coords_sum.hist_offset[i_coord]) -
                                         hist_base)
                                  : nullptr;
            const bool ring = ring_pos != kFifoHistory;
            const int ring0 =
                ring && offset > 0
                    ? static_cast<int>(ring_pos % static_cast<uint64_t>(offset))
                    : 0;
#pragma unroll
            for (int k = 0; k < kIterSampsPerTh; ++k) {
                const int isamp = isamp0 + (k * kIterThreads);
                if (isamp >= nsamps_out) {
                    break;
                }
                // Summed in the output storage type: float as before, or the
                // narrow integer type (int_tree), whose plan bound cannot wrap.
                TOut tail_val = TOut{0};
                TOut head_val = TOut{0};
                if constexpr (Mode == FDMTMode::kFull) {
                    if (isamp < nsamps_tail) {
                        tail_val = conv(tail[isamp]);
                    }
                    if (isamp >= offset && (isamp - offset) < nsamps_tail) {
                        head_val = conv(head[isamp - offset]);
                    }
                } else if constexpr (Mode == FDMTMode::kValid) {
                    tail_val = conv(tail[isamp]);
                    if (isamp >= offset) {
                        head_val = conv(head[isamp - offset]);
                    } else if (hist != nullptr) {
                        int slot = isamp;
                        if (ring) {
                            slot = ring0 + isamp;
                            slot -= slot >= offset ? offset : 0;
                        }
                        head_val = from_hist(hist[slot]);
                    }
                } else if constexpr (Mode == FDMTMode::kRoll) {
                    tail_val            = conv(tail[isamp]);
                    const int head_samp = (isamp >= offset)
                                              ? (isamp - offset)
                                              : (nsamps_tail - offset + isamp);
                    head_val            = conv(head[head_samp]);
                }
                out[isamp] = static_cast<TOut>(tail_val + head_val);
            }
            if constexpr (Mode == FDMTMode::kValid) {
                if (update_ring && offset > 0 && hist != nullptr) {
                    const auto nsamps_head =
                        active > 0 ? active : coords_sum.head_nsamps[i_coord];
                    const int newest =
                        nsamps_head < offset ? nsamps_head : offset;
                    const int first = nsamps_head - newest;
                    const int slot0 = static_cast<int>(
                        (ring_pos + static_cast<uint64_t>(first)) %
                        static_cast<uint64_t>(offset));
                    __syncthreads();
                    for (int k = static_cast<int>(threadIdx.x); k < newest;
                         k += kIterThreads) {
                        int slot = slot0 + k;
                        slot -= slot >= offset ? offset : 0;
                        hist[slot] = to_history(head[first + k]);
                    }
                }
            }
        }

        if (i_coord < ncoords_copy_cur) {
            const int extra = (active > 0 && chunk_nsamps > 0)
                                  ? (coords_copy.nsamps[i_coord] - chunk_nsamps)
                                  : 0;
            const auto nsamps_out =
                active > 0 ? (active + extra) : coords_copy.nsamps[i_coord];
            const auto nsamps_tail = active > 0
                                         ? (active + extra)
                                         : coords_copy.tail_nsamps[i_coord];
            TOut* out       = state_out + (i_beam * out_state_nelements) +
                              coords_copy.buf_offset[i_coord];
            const TIn* tail = state_in + (i_beam * in_state_nelements) +
                              coords_copy.tail_buf_offset[i_coord];
#pragma unroll
            for (int k = 0; k < kIterSampsPerTh; ++k) {
                const int isamp = isamp0 + (k * kIterThreads);
                if (isamp < nsamps_tail) {
                    out[isamp] = conv(tail[isamp]);
                } else if (isamp < nsamps_out) {
                    out[isamp] = TOut{0};
                }
            }
        }
    } // i_coord
}

} // namespace dmt::algorithms::detail::fdmt_cuda
