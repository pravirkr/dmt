#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_fused_tile.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"
#include "dmt/plans_cuda.cuh"

#include "fdmt_direct.cuh"

namespace dmt::algorithms {

namespace {

using detail::FDMTInputF32;
using detail::FDMTInputPacked;

/**
 * @brief Level-0 (per-channel) FDMT state initialization kernel.
 *
 * `dt_grid_sub` (this sub-band's local dt grid) is dense, contiguous, and
 * sorted ascending by construction (see `make_plan_iter0` in plans.cpp), but
 * may start below zero when the plan's overall dt range is negative or
 * straddles zero. Level 0 has no notion of merge *direction* -- that is
 * resolved entirely by the sign-aware tail/head reference swap in
 * `make_plan` (see its doc comment), once per merge, above this level. A
 * row here only ever represents "this single channel's own delay/smearing
 * width of |dt| samples": rows +s and -s are always identical, since a
 * single leaf channel has no other channel to be earlier/later than. This
 * mirrors the CPU implementation's `fdmt_init_subband` in fdmt.cpp exactly
 * (see its doc comment for the full rationale) -- every row lookup below is
 * indexed by magnitude `s`, and `write_matches` fans a computed row out to
 * whichever of the +s / -s dt indices are actually present in this
 * sub-band's grid.
 */
template <FDMTMode Mode, bool UseBoxSmearing, typename Input, typename TOut>
__global__ void
kernel_init_fdmt(const Input waterfall,
                 TOut* __restrict__ state,
                 const int* __restrict__ grids0_dt_grid_ptr,
                 const int* __restrict__ grids0_ndt_ptr,
                 const int* __restrict__ grids0_coord_offset_ptr,
                 int nsubs,
                 int nsamps,
                 int row_stride,
                 int hist_len,
                 int state0_nelements,
                 const float* __restrict__ hist,
                 const detail::FDMTBoxMeans box) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (i_sub >= nsubs || isamp >= nsamps) {
        return;
    }

    const auto* dt_grid_sub =
        &grids0_dt_grid_ptr[grids0_coord_offset_ptr[i_sub]];
    // `nsamps` samples of this chunk; rows `row_stride` apart.
    const auto buffer_offset = (i_beam * state0_nelements) +
                               (grids0_coord_offset_ptr[i_sub] * row_stride);
    const auto ndt_grid_sub  = grids0_ndt_ptr[i_sub];
    const auto wf_row        = (i_beam * nsubs) + i_sub;
    const auto hist_offset   = (i_beam * nsubs * hist_len) + (i_sub * hist_len);
    const auto dt_first      = dt_grid_sub[0];
    const auto dt_last       = dt_grid_sub[ndt_grid_sub - 1];

    // Lambda to fetch sample at relative index (isamp - shift), shift >= 0.
    auto get_sample = [&](int t) -> float {
        if (t >= 0) {
            return waterfall.load(wf_row, t);
        }
        if constexpr (Mode == FDMTMode::kRoll) {
            return waterfall.load(wf_row, t + nsamps);
        } else if constexpr (Mode == FDMTMode::kValid) {
            return (hist != nullptr) ? hist[hist_offset + hist_len + t] : 0.0F;
        } else {
            return 0.0F; // kFull: zero-padded before t=0
        }
    };

    // Box smearing writes each box's mean (fdmt_box_weights.hpp): the
    // dyadic float mean for float input; for packed input the exact integer
    // box sum (uint32) rounded by its recipe, as integer units for integer
    // TOut (int_tree, exact within the type's plan bound) or times 2^-F. The
    // column helper is shared with the fused kernel, so both agree bit for
    // bit.
    const auto put = [&](int i_dt, auto val) {
        state[buffer_offset + (static_cast<int64_t>(i_dt) * row_stride) +
              isamp] = static_cast<TOut>(val);
    };
    if constexpr (detail::kFDMTPackedInput<Input> &&
                  std::is_same_v<TOut, uint64_t>) {
        detail::fdmt_init_column<UseBoxSmearing, uint64_t>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, uint64_t sum) {
                return detail::box_round_apply<uint64_t>(sum, box.round[s]);
            });
    } else if constexpr (detail::kFDMTPackedInput<Input>) {
        detail::fdmt_init_column<UseBoxSmearing, uint32_t>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, uint32_t sum) {
                return box.of_packed(s, sum, std::is_integral_v<TOut>);
            });
    } else {
        detail::fdmt_init_column<UseBoxSmearing, float>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, float sum) { return box.of_float(s, sum); });
    }
}

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

/// kernel_advance_tree_history: one warp per (coordinate, segment of
/// kAdvanceSegment samples), kAdvanceWarps coordinates per block, so the many
/// short-delay rows of the low levels do not idle a block each.
inline constexpr int kAdvanceWarps   = 8;
inline constexpr int kAdvanceSegment = 256;

/**
 * @brief Advances tree-level cross-block history on GPU (layout as in
 * kernel_execute_iter). A FIFO is shifted from hist_in into hist_out; a
 * ring (hist_in == hist_out) only receives the chunk's newest
 * min(delay, chunk) head samples in place, after the level's merges have
 * read it. Supports arbitrary block sizes and offsets.
 * Grid: x over coordinate groups (strided), y over segments, z over beams.
 */
template <typename TIn = float, typename THist = float>
__global__ __launch_bounds__(
    kAdvanceWarps *
    32) void kernel_advance_tree_history(const TIn* __restrict__ state_in,
                                         const plans::FDMTCoordDPtrs coords_sum,
                                         const THist* hist_in,
                                         THist* hist_out,
                                         int ncoords_sum_cur,
                                         int in_state_nelements,
                                         int64_t hist_stride,
                                         int64_t hist_base,
                                         uint64_t ring_pos,
                                         float int_unit,
                                         int active) {
    constexpr int kWarp = 32;
    const int lane      = static_cast<int>(threadIdx.x) % kWarp;
    const int warp      = static_cast<int>(threadIdx.x) / kWarp;
    const int seg0      = static_cast<int>(blockIdx.y) * kAdvanceSegment;
    // Existing FDMT keeps canonical float history; packed search keeps
    // exact integer history without a conversion through float.
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
    for (int i_coord = (static_cast<int>(blockIdx.x) * kAdvanceWarps) + warp;
         i_coord < ncoords_sum_cur;
         i_coord += static_cast<int>(gridDim.x) * kAdvanceWarps) {
        const auto offset = coords_sum.offset[i_coord];
        if (offset <= 0 || seg0 >= offset) {
            continue;
        }
        // `active` > 0: this chunk's length (see kernel_execute_iter).
        const auto nsamps_head =
            active > 0 ? active : coords_sum.head_nsamps[i_coord];
        const auto hist_off =
            (i_beam * hist_stride) +
            (static_cast<int64_t>(coords_sum.hist_offset[i_coord]) - hist_base);
        const auto head_idx_base =
            (i_beam * in_state_nelements) + coords_sum.head_buf_offset[i_coord];
        const bool ring  = ring_pos != kFifoHistory;
        const int newest = nsamps_head < offset ? nsamps_head : offset;
        const int count  = ring ? newest : offset;
        const int end =
            seg0 + kAdvanceSegment < count ? seg0 + kAdvanceSegment : count;
        const int first = nsamps_head - newest;
        const int slot0 =
            ring ? static_cast<int>((ring_pos + static_cast<uint64_t>(first)) %
                                    static_cast<uint64_t>(offset))
                 : 0;
        for (int k = seg0 + lane; k < end; k += kWarp) {
            if (ring) {
                int slot = slot0 + k;
                slot -= slot >= offset ? offset : 0;
                hist_out[hist_off + slot] =
                    to_history(state_in[head_idx_base + first + k]);
            } else if (nsamps_head >= offset) {
                hist_out[hist_off + k] = to_history(
                    state_in[head_idx_base + (nsamps_head - offset + k)]);
            } else if (k < offset - nsamps_head) {
                hist_out[hist_off + k] =
                    (hist_in != nullptr) ? hist_in[hist_off + k + nsamps_head]
                                         : THist{0};
            } else {
                hist_out[hist_off + k] = to_history(
                    state_in[head_idx_base + (k - (offset - nsamps_head))]);
            }
        }
    } // i_coord
}

/**
 * @brief Linearizes ring tree histories (kernel_execute_iter) into the
 * oldest-first layout of save_history(): element k of a coordinate with
 * delay d is the ring slot (pos + k) mod d, pos being the next chunk's
 * stream sample.
 */
__global__ void kernel_tree_ring_to_fifo(const float* __restrict__ ring,
                                         float* __restrict__ fifo,
                                         const plans::FDMTCoordDPtrs coords,
                                         int ncoords,
                                         int64_t ring_stride,
                                         int64_t fifo_stride,
                                         int64_t ring_base,
                                         uint64_t pos) {
    const auto k = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    for (int i = static_cast<int>(blockIdx.y); i < ncoords;
         i += static_cast<int>(gridDim.y)) {
        const int d = coords.offset[i];
        if (d <= 0 || k >= d) {
            continue;
        }
        const auto off = static_cast<int64_t>(coords.hist_offset[i]);
        int slot       = static_cast<int>(pos % static_cast<uint64_t>(d)) + k;
        slot -= slot >= d ? d : 0;
        fifo[(i_beam * fifo_stride) + off + k] =
            ring[(i_beam * ring_stride) + off - ring_base + slot];
    }
}

/// @brief Thread-block execution policy of detail::fdmt_fused_tile: rows
/// strided over the block's warps, a row's cells over the warp's lanes, and
/// __syncthreads() as the barrier.
struct DeviceBlock {
    template <typename Row>
    __host__ __device__ void
    for_each_row(int nrows, int width, Row&& row) const {
#if DMT_GPU_DEVICE_PASS
        constexpr int kWarp = 32;
        const int lane      = static_cast<int>(threadIdx.x) % kWarp;
        const int nwarps    = static_cast<int>(blockDim.x) / kWarp;
        for (int r = static_cast<int>(threadIdx.x) / kWarp; r < nrows;
             r += nwarps) {
            auto cell = row(r);
            for (int j = lane; j < width; j += kWarp) {
                cell(j);
            }
        }
#else
        (void)nrows;
        (void)width;
        (void)row;
#endif
    }
    __host__ __device__ void sync() const {
#if DMT_GPU_DEVICE_PASS
        __syncthreads();
#endif
    }
};

/**
 * @brief Fused level-0 initialisation + tree merges 1..F (see
 * fdmt_fused_tile.hpp). One block per (time tile, channel group, beam);
 * dynamic shared memory holds the group's intermediate levels. Writes level
 * F, and in valid mode the fused levels' tree history, exactly as the
 * unfused kernels would.
 */
template <FDMTMode Mode, bool UseBoxSmearing, typename Input, typename TOut>
__global__ void kernel_fused_levels(const Input waterfall,
                                    TOut* __restrict__ state_out,
                                    const detail::FDMTFusedTileArgs args,
                                    int64_t out_beam_stride,
                                    const float* __restrict__ hist0,
                                    const float* __restrict__ thist_in,
                                    float* __restrict__ thist_out,
                                    int64_t tree_hist_size) {
    extern __shared__ float smem[];
    const auto i_beam       = static_cast<int64_t>(blockIdx.z);
    const auto hist0_stride = static_cast<int64_t>(args.nsubs) * args.hist_len;
    detail::fdmt_fused_tile<Mode, UseBoxSmearing>(
        DeviceBlock{}, waterfall, i_beam * args.nsubs,
        state_out + (i_beam * out_beam_stride), args,
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), smem,
        (hist0 != nullptr) ? hist0 + (i_beam * hist0_stride) : nullptr,
        (thist_in != nullptr) ? thist_in + (i_beam * tree_hist_size) : nullptr,
        (thist_out != nullptr) ? thist_out + (i_beam * tree_hist_size)
                               : nullptr);
}

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

class FDMTCudaEngine final : public detail::FDMTEngine {
public:
    // Keep the host-span overloads visible next to the device ones below.
    using detail::FDMTEngine::execute_search;

    FDMTCudaEngine(const plans::FDMTPlan& plan,
                   const detail::FDMTEngineConfig& cfg)
        : m_device_id(cfg.exec.device),
          m_use_box_smearing(cfg.use_box_smearing),
          m_int_tree(cfg.int_tree),
          m_mode(cfg.mode),
          m_nbeams(cfg.nbeams),
          m_plan(plan) {
        if (m_nbeams == 0) {
            throw std::invalid_argument(
                "FDMT (cuda): nbeams must be at least 1");
        }
        m_search_nbits = cfg.search_nbits;
        m_search_u32   = detail::int_search_fits_u32(m_plan, m_use_box_smearing,
                                                     m_search_nbits);
        // Staggered subbands cross fused-group boundaries: fuse (and compute
        // direct levels) only below them.
        m_search_max_fuse =
            std::min(cfg.max_fuse_levels, m_plan.get_max_fuse_levels());
        m_search_upper = cfg.search_upper_levels;
        m_force_skewed = cfg.force_skewed_direct;
        m_chunk = detail::fdmt_chunk_nsamps(m_plan, m_mode, m_nbeams,
                                            cfg.chunk_nsamps, m_search_upper);
        // Automatic chunk: L2-resident state when it fits, else the budget
        // rule (halved only when the packed search needs uint64 state).
        const SizeType elem = (m_search_nbits == 32 || m_search_u32)
                                  ? sizeof(uint32_t)
                                  : sizeof(uint64_t);
        const SizeType cached =
            (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0)
                ? detail::fdmt_cache_chunk(m_plan, m_nbeams, elem, l2_bytes(),
                                           m_search_upper)
                : 0;
        if (cached != 0) {
            m_chunk = cached;
        } else if (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0 &&
                   (m_plan.get_history_size() +
                    m_plan.get_tree_history_size()) *
                           sizeof(float) >
                       detail::kFDMTChunkStateBudget) {
            m_chunk = std::min<SizeType>(1024, m_plan.get_nsamps());
        } else if (elem == sizeof(uint64_t) && cfg.chunk_nsamps == 0) {
            m_chunk = std::min(m_plan.get_nsamps(),
                               std::max<SizeType>(64, m_chunk / 2));
        }
        m_requested_fuse = cfg.fuse_levels;
        init_device_structures(m_search_nbits == 32 ? cfg.fuse_levels : 0);
    }

    void execute_search(DeviceSpan<const float> waterfall,
                        detail::FDMTSearchSink& sink,
                        Stream queue) override {
        check_inputs(waterfall.size(), m_nbeams * m_plan.get_dmt_size());
        const SizeType root = m_plan.get_niters();
        if (m_mode != FDMTMode::kValid || get_fuse_levels() >= root) {
            throw std::logic_error(
                "FDMT search requires valid mode and a separate root");
        }
        const auto stream = to_cuda(queue);
        const detail::FDMTInputF32 input{waterfall.data(),
                                         static_cast<int>(m_plan.get_nsamps())};
        for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
            start(chunk_input(input, t0), m_float_levels, nullptr,
                  std::min(m_chunk, m_plan.get_nsamps() - t0), stream, true);
            const auto consume = [&](SizeType l, SizeType source) {
                const auto hist = tree_history(l);
                sink.consume(
                    l, {
                           .state = reinterpret_cast<const float*>(
                               m_levels_d[source].base),
                           .prev_state = l > 0 ? reinterpret_cast<const float*>(
                                                     m_levels_d[l - 1].base)
                                               : nullptr,
                           .history    = hist.in,
                           .row_stride =
                               m_chunk + (source < m_level_max_extra.size()
                                              ? m_level_max_extra[source]
                                              : 0),
                           .beam_stride =
                               static_cast<SizeType>(level_beam_stride(source)),
                           .history_stride = static_cast<SizeType>(hist.stride),
                           .history_base   = static_cast<SizeType>(hist.base),
                           .nsamps         = m_active,
                           .sample_begin   = m_chunk_pos,
                           .history_ring   = hist.ring != kFifoHistory,
                           .stream         = queue,
                       });
            };
            consume(m_current_level, m_current_level);
            if (m_top_chunk && m_top->target < root) {
                while (m_current_level < m_top->level) {
                    const SizeType next = m_current_level + 1;
                    execute_iter_device(next, stream, /*regular_only=*/true,
                                        /*advance_hist=*/true);
                    m_current_level = next;
                }
                run_top(
                    reinterpret_cast<float*>(m_levels_d[m_top->target].base),
                    level_beam_stride(m_top->target), static_cast<int>(m_chunk),
                    stream);
            }
            const SizeType last = (m_top_chunk && m_top->target == root)
                                      ? m_top->level
                                      : root - 1;
            while (m_current_level < last) {
                const SizeType next = m_current_level + 1;
                // The merge kernel may have advanced the regular rows'
                // rings itself; advance only the rest after the search read
                // the history.
                const int coord_begin =
                    execute_iter_device(next, stream, /*regular_only=*/true,
                                        /*advance_hist=*/false);
                m_current_level = next;
                consume(next, next);
                advance_tree_history<float>(
                    reinterpret_cast<const float*>(m_levels_d[next - 1].base),
                    next, stream, coord_begin);
            }
            if (m_top_chunk && m_top->target == root) {
                // The direct top materialises this chunk's trials.
                auto* trials = thrust::raw_pointer_cast(m_top->trials.data());
                const auto trial_stride =
                    m_plan.get_container().coordinates[root].size() * m_chunk;
                run_top(trials, static_cast<int64_t>(trial_stride),
                        static_cast<int>(m_chunk), stream);
                sink.consume(root, {
                                       .state             = trials,
                                       .row_stride        = m_chunk,
                                       .beam_stride       = trial_stride,
                                       .nsamps            = m_active,
                                       .sample_begin      = m_chunk_pos,
                                       .materialized_root = true,
                                       .stream            = queue,
                                   });
            } else {
                consume(root, root - 1);
                advance_tree_history<float>(
                    reinterpret_cast<const float*>(m_levels_d[root - 1].base),
                    root, stream);
            }
            m_current_level  = root;
            m_is_initialized = false;
        }
        m_device_work.mark(stream);
    }

    void execute_search(DeviceSpan<const uint8_t> waterfall,
                        SizeType nbits,
                        detail::FDMTSearchSink& sink,
                        Stream queue) override {
        if (nbits == 32 || nbits != m_search_nbits) {
            throw std::invalid_argument(
                "FDMT integer search: input bit depth mismatch");
        }
        check_packed_inputs(waterfall.size(), nbits,
                            m_nbeams * m_plan.get_dmt_size());
        gpu_utils::set_device(m_device_id);
        const auto stream   = to_cuda(queue);
        const SizeType root = m_plan.get_niters();
        const FDMTInputPacked full{
            .data      = waterfall.data(),
            .row_bytes = static_cast<int>(
                bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits)),
            .nbits = static_cast<int>(nbits)};
        if (m_search_u32) {
            execute_search_int_typed<uint32_t>(
                full, thrust::raw_pointer_cast(m_integer32_ring.data()), sink,
                queue, stream);
        } else {
            execute_search_int_typed<uint64_t>(
                full, thrust::raw_pointer_cast(m_integer_ring.data()), sink,
                queue, stream);
        }
        m_device_work.mark(stream);
    }

    template <typename TInt>
    void execute_search_int_typed(const FDMTInputPacked& full,
                                  TInt* ring,
                                  detail::FDMTSearchSink& sink,
                                  Stream queue,
                                  cudaStream_t stream) {
        const SizeType root = m_plan.get_niters();
        const auto tree = static_cast<int64_t>(m_plan.get_tree_history_size());
        if (m_direct) {
            execute_search_int_direct<TInt>(full, ring, sink, queue, stream);
            return;
        }
        for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
            m_active    = std::min(m_chunk, m_plan.get_nsamps() - t0);
            m_chunk_pos = m_next_pos;
            m_next_pos += m_active;
            m_level0_hist_in_ptr = thrust::raw_pointer_cast(
                (m_parity ? m_history_b_d : m_history_a_d).data());
            m_level0_hist_out_ptr = thrust::raw_pointer_cast(
                (m_parity ? m_history_a_d : m_history_b_d).data());
            m_parity    = !m_parity;
            auto* state = reinterpret_cast<TInt*>(
                thrust::raw_pointer_cast(m_state_internal_d.data()));
            const auto level_ptr = [&](SizeType l) {
                return state + (l % 2) * m_nbeams * m_half_elems;
            };
            initialise_device<FDMTInputPacked, TInt>(chunk_input(full, t0),
                                                     level_ptr(0), stream);
            for (SizeType l = 0; l <= root; ++l) {
                const auto& shape = m_plan.get_container().state_shape[l];
                auto sums         = m_plan_d.coordinates_sum.get_raw_ptrs();
                auto copies       = m_plan_d.coordinates_copy.get_raw_ptrs();
                sums.update_offsets(m_coords_sum_offsets[l]);
                copies.update_offsets(m_coords_copy_offsets[l]);
                int adv_begin = 0;
                if (l > 0 && l < root) {
                    const auto reg_sum = detail::fdmt_regular_sum_coords(
                        m_plan.get_container(), l);
                    const int count =
                        static_cast<int>(std::max(reg_sum, shape.ncoords_copy));
                    const dim3 block(kIterThreads);
                    const dim3 grid(
                        (m_active + kIterThreads * kIterSampsPerTh - 1) /
                            (kIterThreads * kIterSampsPerTh),
                        std::min(count, kMaxGridY), m_nbeams);
                    gpu_utils::check_kernel_launch_params(grid, block);
                    const bool fuse_ring = grid.x == 1;
                    kernel_execute_iter<FDMTMode::kValid, TInt, TInt, TInt>
                        <<<grid, block, 0, stream>>>(
                            level_ptr(l - 1), level_ptr(l), sums, copies, ring,
                            static_cast<int>(m_active),
                            static_cast<int>(reg_sum),
                            static_cast<int>(shape.ncoords_copy),
                            static_cast<int>(m_half_elems),
                            static_cast<int>(m_half_elems), tree, 0,
                            m_chunk_pos, 1.0F, static_cast<int>(m_active),
                            fuse_ring);
                    if (fuse_ring) {
                        adv_begin = static_cast<int>(reg_sum);
                    }
                }
                detail::FDMTSearchInput in{
                    .row_stride     = m_chunk,
                    .beam_stride    = m_half_elems,
                    .history_stride = m_plan.get_tree_history_size(),
                    .nsamps         = m_active,
                    .sample_begin   = m_chunk_pos,
                    .history_ring   = true,
                    .stream         = queue,
                };
                if constexpr (std::is_same_v<TInt, uint32_t>) {
                    in.integer32_state = level_ptr(l == root ? root - 1 : l);
                    in.integer32_prev_state =
                        l > 0 ? level_ptr(l - 1) : nullptr;
                    in.integer32_history = ring;
                } else {
                    in.integer_state      = level_ptr(l == root ? root - 1 : l);
                    in.integer_prev_state = l > 0 ? level_ptr(l - 1) : nullptr;
                    in.integer_history    = ring;
                }
                sink.consume(l, in);
                const int adv_count =
                    static_cast<int>(shape.ncoords_sum) - adv_begin;
                if (l > 0 && adv_count > 0 && shape.dt_max > 0) {
                    // Ring: only the chunk's newest min(delay, chunk) samples.
                    const auto newest = static_cast<int>(
                        std::min<SizeType>(shape.dt_max, m_active));
                    sums.update_offsets(adv_begin);
                    launch_advance_tree_history(
                        level_ptr(l - 1), sums, ring, ring, adv_count, newest,
                        static_cast<int>(m_half_elems), tree, 0, m_chunk_pos,
                        1.0F, stream);
                }
                gpu_utils::check_last_gpu_error(
                    "FDMT integer search launch failed");
            }
        }
    }

    /**
     * The packed search with the direct levels (and the direct top when
     * configured), as the float search runs them: level K from the input's
     * uint32_t running sums, the tree from K + 1, and either the trials (a
     * full-band search: materialised, as the sink's root) or level
     * root - subband_depth - 1 summed from level k's rings. Both start at
     * stream position 0 with all-zero windows and rings, so the tree
     * histories of the levels they replace are never read (the integer
     * search keeps no stepper, history export or tree fallback).
     */
    template <typename TInt>
    void execute_search_int_direct(const FDMTInputPacked& full,
                                   TInt* ring,
                                   detail::FDMTSearchSink& sink,
                                   Stream queue,
                                   cudaStream_t stream) {
        const SizeType root = m_plan.get_niters();
        const auto h_stride = static_cast<int64_t>(m_ring_size);
        auto* state         = reinterpret_cast<TInt*>(
            thrust::raw_pointer_cast(m_state_internal_d.data()));
        const auto level_ptr = [&](SizeType l) {
            return state + (l % 2) * m_nbeams * m_half_elems;
        };
        const auto beam_stride = static_cast<int64_t>(m_half_elems);
        const auto consume = [&](SizeType l, const TInt* src, const TInt* prev,
                                 int64_t src_beam_stride, bool materialized) {
            const SizeType src_lvl = l == root ? root - 1 : l;
            detail::FDMTSearchInput in{
                .row_stride     = m_chunk + (src_lvl < m_level_max_extra.size()
                                                 ? m_level_max_extra[src_lvl]
                                                 : 0),
                .beam_stride    = static_cast<SizeType>(src_beam_stride),
                .history_stride = m_ring_size,
                .history_base   = ring_base_for_level(l),
                .nsamps         = m_active,
                .sample_begin   = m_chunk_pos,
                .materialized_root = materialized,
                .history_ring      = true,
                .stream            = queue,
            };
            if constexpr (std::is_same_v<TInt, uint32_t>) {
                in.integer32_state      = src;
                in.integer32_prev_state = prev;
                in.integer32_history    = ring;
            } else {
                in.integer_state      = src;
                in.integer_prev_state = prev;
                in.integer_history    = ring;
            }
            sink.consume(l, in);
        };
        const auto merge = [&](SizeType l) {
            const auto& shape = m_plan.get_container().state_shape[l];
            auto sums         = m_plan_d.coordinates_sum.get_raw_ptrs();
            auto copies       = m_plan_d.coordinates_copy.get_raw_ptrs();
            sums.update_offsets(m_coords_sum_offsets[l]);
            copies.update_offsets(m_coords_copy_offsets[l]);
            const auto reg_sum =
                detail::fdmt_regular_sum_coords(m_plan.get_container(), l);
            const int count =
                static_cast<int>(std::max(reg_sum, shape.ncoords_copy));
            const int nsamps_l = static_cast<int>(
                m_active +
                (l < m_level_max_extra.size() ? m_level_max_extra[l] : 0));
            const dim3 block(kIterThreads);
            const dim3 grid((nsamps_l + kIterThreads * kIterSampsPerTh - 1) /
                                (kIterThreads * kIterSampsPerTh),
                            std::min(count, kMaxGridY), m_nbeams);
            gpu_utils::check_kernel_launch_params(grid, block);
            const bool skewed_mid =
                m_direct->skewed && m_top && l <= m_top->level;
            const bool fuse_ring = !skewed_mid && grid.x == 1;
            const auto h_base    = static_cast<int64_t>(ring_base_for_level(l));
            kernel_execute_iter<FDMTMode::kValid, TInt, TInt, TInt>
                <<<grid, block, 0, stream>>>(
                    level_ptr(l - 1), level_ptr(l), sums, copies,
                    skewed_mid ? nullptr : ring, nsamps_l,
                    static_cast<int>(reg_sum),
                    static_cast<int>(shape.ncoords_copy),
                    static_cast<int>(m_half_elems),
                    static_cast<int>(m_half_elems), h_stride, h_base,
                    m_chunk_pos, 1.0F, static_cast<int>(m_active), fuse_ring,
                    skewed_mid ? static_cast<int>(m_chunk) : 0);
            return (skewed_mid || fuse_ring) ? static_cast<int>(reg_sum) : 0;
        };
        const auto advance = [&](SizeType l, int adv_begin) {
            if (m_direct->skewed && m_top && l <= m_top->level) {
                return;
            }
            const auto& shape = m_plan.get_container().state_shape[l];
            const int adv_count =
                static_cast<int>(shape.ncoords_sum) - adv_begin;
            if (adv_count > 0 && shape.dt_max > 0 && ring != nullptr) {
                auto sums = m_plan_d.coordinates_sum.get_raw_ptrs();
                sums.update_offsets(m_coords_sum_offsets[l] + adv_begin);
                const auto newest = static_cast<int>(
                    std::min<SizeType>(shape.dt_max, m_active));
                const auto h_base =
                    static_cast<int64_t>(ring_base_for_level(l));
                launch_advance_tree_history(
                    level_ptr(l - 1), sums, ring, ring, adv_count, newest,
                    static_cast<int>(m_half_elems), h_stride, h_base,
                    m_chunk_pos, 1.0F, stream);
            }
        };
        const SizeType levels = m_direct->levels;
        for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
            m_active    = std::min(m_chunk, m_plan.get_nsamps() - t0);
            m_chunk_pos = m_next_pos;
            m_next_pos += m_active;
            const int in = m_parity ? 1 : 0;
            if (!m_direct->skewed) {
                m_direct_win_in =
                    thrust::raw_pointer_cast(m_direct->win[in].data());
                m_direct_win_out =
                    thrust::raw_pointer_cast(m_direct->win[1 - in].data());
            }
            m_parity         = !m_parity;
            const auto input = chunk_input(full, t0);
            launch_direct_int<TInt>(input, level_ptr(levels), stream);
            advance_direct_window(input, stream);
            SizeType level =
                (m_skew_nbatches > 1 && m_top) ? m_top->level : levels;
            consume(level, level_ptr(level), nullptr, beam_stride, false);
            const SizeType last = m_top ? m_top->level : root - 1;
            while (level < last) {
                ++level;
                const int adv_begin = merge(level);
                consume(level, level_ptr(level), level_ptr(level - 1),
                        beam_stride, false);
                advance(level, adv_begin);
            }
            if (m_top) {
                if (m_top->target == root) {
                    // The trials, materialised: the sink's root.
                    auto* trials = reinterpret_cast<TInt*>(
                        thrust::raw_pointer_cast(m_top->trials.data()));
                    const auto trial_stride = static_cast<int64_t>(
                        m_plan.get_container().coordinates[root].size() *
                        m_chunk);
                    run_top_int<TInt>(level_ptr(m_top->level), trials,
                                      trial_stride, stream);
                    consume(root, trials, nullptr, trial_stride, true);
                    gpu_utils::check_last_gpu_error(
                        "FDMT integer search (direct) launch failed");
                    continue;
                }
                run_top_int<TInt>(level_ptr(m_top->level),
                                  level_ptr(m_top->target), beam_stride,
                                  stream);
                level = m_top->target;
                consume(level, level_ptr(level), nullptr, beam_stride, false);
            }
            while (level + 1 < root) {
                ++level;
                const int adv_begin = merge(level);
                consume(level, level_ptr(level), level_ptr(level - 1),
                        beam_stride, false);
                advance(level, adv_begin);
            }
            // The root: merged on the fly by the sink from level root - 1.
            consume(root, level_ptr(root - 1), level_ptr(root - 1), beam_stride,
                    false);
            advance(root, 0);
            gpu_utils::check_last_gpu_error(
                "FDMT integer search (direct) launch failed");
        }
    }

    /// Level K of the packed search from the input (uint32_t running sums).
    template <typename TInt>
    void launch_direct_int(const FDMTInputPacked& input,
                           TInt* out,
                           cudaStream_t stream) {
        auto& d          = *m_direct;
        const int nchans = static_cast<int>(m_plan.get_nchans());
        const int active = static_cast<int>(m_active);
        if (d.skewed) {
            const dim3 wgrid((active + 255) / 256, nchans, m_nbeams);
            const int ntiles =
                (active + d.max_extra + detail::kDirectTile - 1) /
                detail::kDirectTile;
            const int row_stride = static_cast<int>(m_chunk);
            const auto b_stride_k =
                static_cast<int64_t>(level_beam_stride(d.levels));
            auto* state_int = reinterpret_cast<TInt*>(
                thrust::raw_pointer_cast(m_state_internal_d.data()));
            const auto lvl_ptr = [&](SizeType l) -> TInt* {
                return state_int + (l % 2) * m_nbeams * m_half_elems;
            };
            if (m_search_nbits <= 8) {
                auto* wring = thrust::raw_pointer_cast(d.win_ring_u8.data());
                detail::kernel_direct_write_ring<FDMTInputPacked, uint8_t>
                    <<<wgrid, 256, 0, stream>>>(
                        wring, d.win_ring_size,
                        thrust::raw_pointer_cast(d.chan_ring_off.data()),
                        thrust::raw_pointer_cast(d.chan_ring_len.data()), input,
                        nchans, active, m_chunk_pos);
                for (SizeType b = 0; b < m_skew_nbatches; ++b) {
                    const int32_t sub0    = d.batch_sub_off[b];
                    const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
                    if (nsubs_b > 0) {
                        const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                        gpu_utils::check_kernel_launch_params(
                            lgrid, dim3(detail::kDirectWarps * 32));
                        detail::kernel_direct_level_skewed<uint8_t, uint32_t,
                                                           TInt><<<
                            lgrid, detail::kDirectWarps * 32, d.smem, stream>>>(
                            wring, d.win_ring_size,
                            thrust::raw_pointer_cast(d.chan_ring_off.data()),
                            thrust::raw_pointer_cast(d.chan_ring_len.data()),
                            thrust::raw_pointer_cast(d.subs.data()) + sub0,
                            thrust::raw_pointer_cast(d.leaf_off.data()),
                            thrust::raw_pointer_cast(d.leaves.data()), out,
                            row_stride, b_stride_k, d.cpass, m_chunk_pos,
                            active);
                    }
                    if (m_skew_nbatches > 1 && m_top) {
                        launch_skewed_batch_mid<TInt>(b, lvl_ptr, stream);
                    }
                }
            } else {
                auto* wring = thrust::raw_pointer_cast(d.win_ring_u16.data());
                detail::kernel_direct_write_ring<FDMTInputPacked, uint16_t>
                    <<<wgrid, 256, 0, stream>>>(
                        wring, d.win_ring_size,
                        thrust::raw_pointer_cast(d.chan_ring_off.data()),
                        thrust::raw_pointer_cast(d.chan_ring_len.data()), input,
                        nchans, active, m_chunk_pos);
                for (SizeType b = 0; b < m_skew_nbatches; ++b) {
                    const int32_t sub0    = d.batch_sub_off[b];
                    const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
                    if (nsubs_b > 0) {
                        const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                        gpu_utils::check_kernel_launch_params(
                            lgrid, dim3(detail::kDirectWarps * 32));
                        detail::kernel_direct_level_skewed<uint16_t, uint32_t,
                                                           TInt><<<
                            lgrid, detail::kDirectWarps * 32, d.smem, stream>>>(
                            wring, d.win_ring_size,
                            thrust::raw_pointer_cast(d.chan_ring_off.data()),
                            thrust::raw_pointer_cast(d.chan_ring_len.data()),
                            thrust::raw_pointer_cast(d.subs.data()) + sub0,
                            thrust::raw_pointer_cast(d.leaf_off.data()),
                            thrust::raw_pointer_cast(d.leaves.data()), out,
                            row_stride, b_stride_k, d.cpass, m_chunk_pos,
                            active);
                    }
                    if (m_skew_nbatches > 1 && m_top) {
                        launch_skewed_batch_mid<TInt>(b, lvl_ptr, stream);
                    }
                }
            }
            gpu_utils::check_last_gpu_error(
                "FDMT (cuda): integer skewed direct levels failed");
            return;
        }
        auto* prefix = reinterpret_cast<uint32_t*>(
            thrust::raw_pointer_cast(d.prefix.data()));
        detail::kernel_direct_prefix<FDMTInputPacked, uint32_t>
            <<<dim3(nchans, 1, m_nbeams), detail::kPrefixThreads, 0, stream>>>(
                input, m_direct_win_in, d.window, nchans, active, prefix,
                d.pitch);
        constexpr int kRows = detail::kDirectRowsPerWarp * detail::kDirectWarps;
        const int ntiles =
            (active + detail::kDirectTile - 1) / detail::kDirectTile;
        const dim3 grid((d.max_rows + kRows - 1) / kRows, d.nsubs,
                        m_nbeams * ntiles);
        gpu_utils::check_kernel_launch_params(grid,
                                              dim3(detail::kDirectWarps * 32));
        detail::kernel_direct_level<uint32_t, TInt>
            <<<grid, detail::kDirectWarps * 32, d.smem, stream>>>(
                prefix, d.pitch, d.window, nchans,
                thrust::raw_pointer_cast(d.subs.data()),
                thrust::raw_pointer_cast(d.leaf_off.data()),
                thrust::raw_pointer_cast(d.leaves.data()), out,
                static_cast<int>(m_chunk), static_cast<int64_t>(m_half_elems),
                d.cpass, active, ntiles);
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): integer direct levels failed");
    }

    /// The direct top of the packed search: trials (rows m_chunk apart,
    /// beams `out_beam_stride` apart) from level k's state and rings, then
    /// level k's rings (run_top's integer analogue).
    template <typename TInt>
    void run_top_int(const TInt* level_state,
                     TInt* out,
                     int64_t out_beam_stride,
                     cudaStream_t stream) {
        auto& t              = *m_top;
        constexpr int kWarps = 8;
        constexpr int kSamps = 32 * detail::kTopSampsPerLane;
        const int active     = static_cast<int>(m_active);
        auto* rings =
            reinterpret_cast<TInt*>(thrust::raw_pointer_cast(t.ring.data()));
        if (t.target_ring) {
            const dim3 grid(t.ntrials, 1, m_nbeams);
            gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
            detail::kernel_direct_top_target<TInt>
                <<<grid, kWarps * 32, 0, stream>>>(
                    level_state, static_cast<int64_t>(m_half_elems),
                    static_cast<int>(m_chunk), rings, t.ring_size,
                    thrust::raw_pointer_cast(t.ring_off.data()),
                    thrust::raw_pointer_cast(t.ring_len.data()),
                    thrust::raw_pointer_cast(t.leaf_off.data()),
                    thrust::raw_pointer_cast(t.leaves.data()),
                    thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
                    out_beam_stride, static_cast<int>(m_chunk), m_chunk_pos,
                    active, 0, m_skew_nbatches > 1 ? 0 : -1, true);
            gpu_utils::check_last_gpu_error(
                "FDMT (cuda): integer direct top target failed");
            return;
        }
        const dim3 grid((t.ntrials + kWarps - 1) / kWarps,
                        (active + kSamps - 1) / kSamps, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
        detail::kernel_direct_top<TInt><<<grid, kWarps * 32, 0, stream>>>(
            level_state, static_cast<int64_t>(m_half_elems),
            static_cast<int>(m_chunk), rings, t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()),
            thrust::raw_pointer_cast(t.leaf_off.data()),
            thrust::raw_pointer_cast(t.leaves.data()),
            thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
            out_beam_stride, static_cast<int>(m_chunk), m_chunk_pos, active);
        constexpr int kSegment = 256;
        const int newest       = std::min(t.max_ring, active);
        if (newest > 0) {
            const dim3 rgrid((t.nrows + kWarps - 1) / kWarps,
                             (newest + kSegment - 1) / kSegment, m_nbeams);
            detail::kernel_direct_top_ring<TInt>
                <<<rgrid, kWarps * 32, 0, stream>>>(
                    level_state, static_cast<int64_t>(m_half_elems),
                    static_cast<int>(m_chunk), rings, t.ring_size,
                    thrust::raw_pointer_cast(t.ring_off.data()),
                    thrust::raw_pointer_cast(t.ring_len.data()), t.nrows,
                    m_chunk_pos, active);
        }
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): integer direct top failed");
    }

    void reset_search_history(Stream queue) override {
        gpu_utils::set_device(m_device_id);
        const auto stream = to_cuda(queue);
        for (auto* v : {&m_history_a_d, &m_history_b_d, &m_tree_fifo_d[0],
                        &m_tree_fifo_d[1], &m_tree_ring_d}) {
            if (!v->empty()) {
                gpu_utils::check_gpu_call(
                    cudaMemsetAsync(thrust::raw_pointer_cast(v->data()), 0,
                                    v->size() * sizeof(float), stream));
            }
        }
        if (m_integer_ring.size() > 0) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                thrust::raw_pointer_cast(m_integer_ring.data()), 0,
                m_integer_ring.size() * sizeof(uint64_t), stream));
        }
        if (m_integer32_ring.size() > 0) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                thrust::raw_pointer_cast(m_integer32_ring.data()), 0,
                m_integer32_ring.size() * sizeof(uint32_t), stream));
        }
        if (m_direct) {
            for (auto& w : m_direct->win) {
                if (!w.empty()) {
                    gpu_utils::check_gpu_call(
                        cudaMemsetAsync(thrust::raw_pointer_cast(w.data()), 0,
                                        w.size() * sizeof(float), stream));
                }
            }
            if (!m_direct->win_ring.empty()) {
                gpu_utils::check_gpu_call(cudaMemsetAsync(
                    thrust::raw_pointer_cast(m_direct->win_ring.data()), 0,
                    m_direct->win_ring.size() * sizeof(float), stream));
            }
            if (!m_direct->win_ring_u8.empty()) {
                gpu_utils::check_gpu_call(cudaMemsetAsync(
                    thrust::raw_pointer_cast(m_direct->win_ring_u8.data()), 0,
                    m_direct->win_ring_u8.size() * sizeof(uint8_t), stream));
            }
            if (!m_direct->win_ring_u16.empty()) {
                gpu_utils::check_gpu_call(cudaMemsetAsync(
                    thrust::raw_pointer_cast(m_direct->win_ring_u16.data()), 0,
                    m_direct->win_ring_u16.size() * sizeof(uint16_t), stream));
            }
            m_direct_filled = static_cast<SizeType>(m_direct->window);
        }
        m_direct_stale = false;
        if (m_top && !m_top->ring.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemsetAsync(thrust::raw_pointer_cast(m_top->ring.data()), 0,
                                m_top->ring.size() * sizeof(float), stream));
            m_top_filled = static_cast<SizeType>(m_top->max_ring);
        }
        m_top_stale      = false;
        m_parity         = false;
        m_is_initialized = false;
        m_chunk_pos      = 0;
        m_next_pos       = 0;
        m_device_work.mark(stream);
    }

    // Allocates every device buffer and resolves the fusion depth: nothing is
    // allocated after construction except the per-call staging of the
    // host-memory execute() overloads.
    void init_device_structures(SizeType fuse_levels) {
        // Chunked execution (fdmt_chunk.hpp): the device tables describe rows
        // of the chunk length below the root (the root keeps the dmt
        // output's layout), so the kernels run unchanged on every chunk.
        const SizeType niters = m_plan.get_niters();
        const bool chunked    = m_chunk < m_plan.get_nsamps();
        plans::FDMTPlanContainer relaid =
            chunked ? detail::fdmt_relayout_container(m_plan.get_container(),
                                                      niters, m_chunk)
                    : plans::FDMTPlanContainer{};
        gpu_utils::set_device(m_device_id);
        if (m_use_box_smearing) {
            // Level-0 box means (fdmt_box_weights.hpp), by |dt|: dyadic float
            // weights, and the packed-input rounding recipes on 2^-F.
            const auto weights = m_plan.get_box_weights();
            const auto box_max = weights.size();
            const auto scale_f = m_plan.get_box_mean_bits();
            const float w_unit = std::ldexp(
                1.0F, -static_cast<int>(m_plan.get_box_fraction_bits()));
            m_int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
            std::vector<float> box_w(box_max);
            std::vector<detail::FDMTBoxRound> box_round(box_max);
            for (SizeType s = 0; s < box_max; ++s) {
                box_w[s]     = static_cast<float>(weights[s]) * w_unit;
                box_round[s] = m_search_nbits == 32
                                   ? detail::box_round(s + 1, scale_f)
                                   : detail::FDMTBoxRound{.mul    = weights[s],
                                                          .add    = 0,
                                                          .rshift = 0,
                                                          .lshift = 0};
            }
            m_box_w_d     = box_w;
            m_box_round_d = box_round;
            for (const SizeType nbits : {1, 2, 4, 8, 16}) {
                // The rounding product's width is independent of the u32/u64
                // search state: it must hold S * mul + add (box_round_max()).
                m_box_round_u64[nbits] = !detail::box_round_fits_u32(
                    bit_pack_utils::max_sample_value(nbits), box_max);
            }
        }

        // Precompute cumulative coordinate offsets per iteration
        const auto niters_plus_one = m_plan.get_niters() + 1;
        m_coords_sum_offsets.resize(niters_plus_one, 0);
        m_coords_copy_offsets.resize(niters_plus_one, 0);
        int sum_off  = 0;
        int copy_off = 0;
        for (size_t i = 0; i < niters_plus_one; ++i) {
            m_coords_sum_offsets[i]  = sum_off;
            m_coords_copy_offsets[i] = copy_off;
            sum_off += static_cast<int>(
                m_plan.get_container().state_shape[i].ncoords_sum);
            copy_off += static_cast<int>(
                m_plan.get_container().state_shape[i].ncoords_copy);
        }

        m_float_levels.assign(m_plan.get_niters() + 1, Elem::kF32);
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            m_int_levels[nbits] =
                detail::int_tree_level_types(m_plan, m_use_box_smearing, nbits);
        }
        m_levels_d.resize(m_plan.get_niters() + 1);
        const auto& pc_init = chunked ? relaid : m_plan.get_container();
        configure_fusion(pc_init, fuse_levels);
        configure_direct(fuse_levels);
        configure_top(relaid);
        if (m_direct && m_direct->skewed) {
            for (const SizeType nbits : {1, 2, 4, 8, 16}) {
                m_int_levels[nbits].assign(niters + 1, Elem::kF32);
            }
        }
        const auto& pc =
            !relaid.state_shape.empty() ? relaid : m_plan.get_container();
        m_row_len0   = pc.state_shape[0].nsamps;
        m_half_elems = 0;
        for (SizeType l = 0; l < niters; ++l) {
            if (m_direct && m_direct->skewed && l < m_direct->levels) {
                continue;
            }
            if (m_direct && m_direct->skewed && m_top && m_top->skewed &&
                m_top->target == niters && l > m_top->level) {
                continue;
            }
            const SizeType elems =
                (m_direct && m_direct->skewed && m_top && l <= m_top->level)
                    ? pc.state_shape[l].nelements
                    : (m_search_upper ? detail::fdmt_regular_coords(pc, l)
                                      : pc.state_shape[l].ncoords) *
                          pc.state_shape[l].nsamps;
            m_half_elems = std::max(m_half_elems, elems);
        }
        check_int32_extents();
        // The two state halves: level l < niters lives in half l % 2; the
        // root is written straight into the caller's dmt buffer.
        m_state_internal_d.resize(
            ((m_search_nbits == 32 || m_search_u32) ? 2 : 4) * m_nbeams *
                m_half_elems,
            0.0F);
        if (m_mode == FDMTMode::kValid &&
            !((m_direct && m_direct->skewed) ||
              (m_search_nbits != 32 && m_direct))) {
            const auto hist_size = m_nbeams * m_plan.get_history_size();
            m_history_a_d.resize(hist_size, 0.0F);
            m_history_b_d.resize(hist_size, 0.0F);
        }
        plans::transfer_fdmt_plan_to_device(pc, m_plan_d);
        if (m_mode == FDMTMode::kValid) {
            // FIFOs for the fused levels, a ring above (see m_tree_fifo_d).
            // Only active tree ring levels (outside m_direct and m_top) are
            // allocated; save_history()/load_history() expand on demand.
            const auto tree = m_plan.get_tree_history_size();
            m_fifo_levels   = (m_search_nbits == 32 && m_fused)
                                  ? static_cast<SizeType>(m_fused->args.fuse)
                                  : 0;
            m_fifo_size     = tree_history_through(m_fifo_levels);
            const SizeType low_first =
                m_direct ? (m_direct->levels + 1) : (m_fifo_levels + 1);
            const SizeType low_last = m_top ? m_top->level : niters;
            m_ring_low_base         = tree_history_through(low_first - 1);
            m_ring_low_size =
                (m_direct && m_direct->skewed && m_top)
                    ? 0
                    : (low_last >= low_first
                           ? tree_history_through(low_last) - m_ring_low_base
                           : 0);
            m_ring_high_orig_base = (m_top && m_top->target < niters)
                                        ? tree_history_through(m_top->target)
                                        : tree;
            m_ring_high_size      = tree - m_ring_high_orig_base;
            m_ring_size           = m_ring_low_size + m_ring_high_size;
            m_full_tree_ring      = (m_ring_size == tree - m_fifo_size);
            if (m_search_nbits == 32) {
                for (auto& fifo : m_tree_fifo_d) {
                    fifo.resize(m_nbeams * m_fifo_size, 0.0F);
                }
                m_tree_ring_d.resize(m_nbeams * m_ring_size, 0.0F);
            } else if (m_search_u32) {
                m_integer32_ring.resize(m_nbeams * m_ring_size, 0);
            } else {
                m_integer_ring.resize(m_nbeams * m_ring_size, 0);
            }
        }
        gpu_utils::check_last_gpu_error("FDMT (gpu): constructor failed");
    }

    ~FDMTCudaEngine() override                       = default;
    FDMTCudaEngine(const FDMTCudaEngine&)            = delete;
    FDMTCudaEngine& operator=(const FDMTCudaEngine&) = delete;
    FDMTCudaEngine(FDMTCudaEngine&&)                 = delete;
    FDMTCudaEngine& operator=(FDMTCudaEngine&&)      = delete;

    void execute(std::span<const float> waterfall_h,
                 std::span<float> dmt_h) override {
        check_inputs(waterfall_h.size(), dmt_h.size());
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        stage_to_device(m_stage_wf_f32_d, waterfall_h);
        execute_d(cuda::std::span<const float>(
                      thrust::raw_pointer_cast(m_stage_wf_f32_d.data()),
                      waterfall_h.size()),
                  stage_dmt_span(), m_host.stream());
        copy_result_to_host(dmt_h);
    }

    void execute_d(cuda::std::span<const float> waterfall_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        check_inputs(waterfall_d.size(), dmt_d.size());
        const FDMTInputF32 input{
            .data   = waterfall_d.data(),
            .nsamps = static_cast<int>(m_plan.get_nsamps()),
        };
        run_block(input, m_float_levels, dmt_d, stream);
    }

    // execute(): the block's chunks, each from level 0 (fused when
    // configured) to the root; the root writes the chunk's columns of dmt.
    template <typename Input>
    void run_block(const Input& input,
                   const std::vector<detail::FDMTLevelType>& types,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        const SizeType nsamps = m_plan.get_nsamps();
        for (SizeType t0 = 0; t0 < nsamps; t0 += m_chunk) {
            start(chunk_input(input, t0), types, dmt_d.data() + t0,
                  std::min(m_chunk, nsamps - t0), stream, /*fuse=*/true);
            if (m_top_chunk) {
                advance(m_top->level - m_current_level, Stream{stream});
                run_top(dmt_d.data() + t0,
                        level_beam_stride(m_plan.get_niters()),
                        static_cast<int>(nsamps), stream);
            } else {
                advance_until_remaining(0, Stream{stream});
            }
        }
        finalize(Stream{stream});
    }

    // The input from sample t0 on (rows keep the block's stride; packed
    // chunks start on a whole byte: chunks are multiples of 64 samples).
    [[nodiscard]] static FDMTInputF32 chunk_input(const FDMTInputF32& in,
                                                  SizeType t0) {
        return {.data = in.data + t0, .nsamps = in.nsamps};
    }
    [[nodiscard]] static FDMTInputPacked chunk_input(const FDMTInputPacked& in,
                                                     SizeType t0) {
        return {.data = in.data + ((t0 * static_cast<SizeType>(in.nbits)) / 8),
                .row_bytes = in.row_bytes,
                .nbits     = in.nbits};
    }

    void execute(std::span<const uint8_t> waterfall_h,
                 SizeType nbits,
                 std::span<float> dmt_h) override {
        check_packed_inputs(waterfall_h.size(), nbits, dmt_h.size());
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        // Only the packed bytes cross PCIe (32/nbits times less than float).
        stage_to_device(m_stage_wf_packed_d, waterfall_h);
        execute_d(cuda::std::span<const uint8_t>(
                      thrust::raw_pointer_cast(m_stage_wf_packed_d.data()),
                      waterfall_h.size()),
                  nbits, stage_dmt_span(), m_host.stream());
        copy_result_to_host(dmt_h);
    }

    // Host-memory execute() staging: device copies of the input and of the
    // (nbeams * dmt_size) output, allocated on the first host call and
    // reused, so
    // steady-state host calls allocate nothing (device-memory callers never
    // pay for them). Transfers and the block's kernels run on
    // m_host.stream() (see gpu_host::ChunkedStager).
    template <typename T>
    void stage_to_device(thrust::device_vector<T>& dst,
                         std::span<const T> src) {
        if (dst.size() < src.size()) {
            dst.resize(src.size());
        }
        m_host.to_device(thrust::raw_pointer_cast(dst.data()), src.data(),
                         src.size_bytes());
    }

    cuda::std::span<float> stage_dmt_span() {
        const auto size = m_nbeams * m_plan.get_dmt_size();
        if (m_stage_dmt_d.size() < size) {
            m_stage_dmt_d.resize(size);
        }
        return {thrust::raw_pointer_cast(m_stage_dmt_d.data()), size};
    }

    // Copies the (nbeams * dmt_size) result back, after the work on
    // m_host.stream().
    void copy_result_to_host(std::span<float> dmt_h) {
        std::vector<gpu_host::ChunkedStager::Segment> segs;
        segs.push_back({dmt_h.data(),
                        thrust::raw_pointer_cast(m_stage_dmt_d.data()),
                        m_nbeams * m_plan.get_dmt_size() * sizeof(float)});
        m_host.to_host(segs);
    }

    void execute_d(cuda::std::span<const uint8_t> waterfall_d,
                   SizeType nbits,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        check_packed_inputs(waterfall_d.size(), nbits, dmt_d.size());
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
        const FDMTInputPacked input{.data      = waterfall_d.data(),
                                    .row_bytes = static_cast<int>(row_bytes),
                                    .nbits     = static_cast<int>(nbits)};
        run_block(input,
                  (m_int_tree && !(m_direct && m_direct->skewed))
                      ? m_int_levels[nbits]
                      : m_float_levels,
                  dmt_d, stream);
    }

    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        execute_d(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream));
        m_device_work.mark(to_cuda(stream));
    }

    void execute(DeviceSpan<const uint8_t> d_waterfall,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        execute_d(to_cuda(d_waterfall), nbits, to_cuda(d_dmt), to_cuda(stream));
        m_device_work.mark(to_cuda(stream));
    }

    void reset(DeviceSpan<const uint8_t> d_waterfall,
               SizeType nbits,
               DeviceSpan<float> d_dmt,
               Stream stream) override {
        reset_impl(to_cuda(d_waterfall), nbits, to_cuda(d_dmt), to_cuda(stream),
                   /*fuse=*/false);
        m_device_work.mark(to_cuda(stream));
    }

    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream) override {
        reset_impl(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream),
                   /*fuse=*/false);
        m_device_work.mark(to_cuda(stream));
    }

    // Host-memory stepper: the block is staged in the host-execute() device
    // buffers and stepped there; finalize() copies each beam's result into
    // `dmt_h`, and the view_* methods return host snapshots.
    void reset(std::span<const float> waterfall_h,
               std::span<float> dmt_h) override {
        check_inputs(waterfall_h.size(), dmt_h.size());
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        stage_to_device(m_stage_wf_f32_d, waterfall_h);
        reset_impl(cuda::std::span<const float>(
                       thrust::raw_pointer_cast(m_stage_wf_f32_d.data()),
                       waterfall_h.size()),
                   stage_dmt_span(), m_host.stream(), /*fuse=*/false);
        m_host_dmt = dmt_h;
    }

    void reset(std::span<const uint8_t> waterfall_h,
               SizeType nbits,
               std::span<float> dmt_h) override {
        check_packed_inputs(waterfall_h.size(), nbits, dmt_h.size());
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        stage_to_device(m_stage_wf_packed_d, waterfall_h);
        reset_impl(cuda::std::span<const uint8_t>(
                       thrust::raw_pointer_cast(m_stage_wf_packed_d.data()),
                       waterfall_h.size()),
                   nbits, stage_dmt_span(), m_host.stream(), /*fuse=*/false);
        m_host_dmt = dmt_h;
    }

    // The stepper (always unfused) runs whole blocks only.
    void check_stepper_block() const {
        if (m_chunk < m_plan.get_nsamps()) {
            throw std::logic_error(fmt::format(
                "FDMT (cuda): the stepper runs whole blocks, but this engine "
                "executes blocks of {} samples in chunks of {} "
                "(chunk_nsamps); construct with chunk_nsamps >= nsamps to "
                "step through levels.",
                m_plan.get_nsamps(), m_chunk));
        }
    }

    // `fuse`: execute() only -- replace level 0 and the first merges by the
    // fused kernel (see configure_fusion()); the stepper stays unfused.
    void reset_impl(cuda::std::span<const uint8_t> d_waterfall,
                    SizeType nbits,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse) {
        check_stepper_block();
        check_packed_inputs(d_waterfall.size(), nbits, d_dmt.size());
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
        const FDMTInputPacked input{.data      = d_waterfall.data(),
                                    .row_bytes = static_cast<int>(row_bytes),
                                    .nbits     = static_cast<int>(nbits)};
        start(input, m_int_tree ? m_int_levels[nbits] : m_float_levels,
              d_dmt.data(), m_plan.get_nsamps(), stream, fuse);
    }

    void reset_impl(cuda::std::span<const float> d_waterfall,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse) {
        check_stepper_block();
        check_inputs(d_waterfall.size(), d_dmt.size());
        const FDMTInputF32 input{
            .data   = d_waterfall.data(),
            .nsamps = static_cast<int>(m_plan.get_nsamps()),
        };
        start(input, m_float_levels, d_dmt.data(), m_plan.get_nsamps(), stream,
              fuse);
    }

    /// Levels below which execute() does not materialise the tree: the
    /// tile fusion's depth, or the direct levels' (fdmt_direct.hpp).
    [[nodiscard]] SizeType get_fuse_levels() const noexcept override {
        if (m_direct) {
            return m_direct->levels;
        }
        return m_fused ? static_cast<SizeType>(m_fused->args.fuse) : 0;
    }

    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept override {
        return m_chunk;
    }

    [[nodiscard]] std::string summary() const override {
        const auto mem = get_memory_usage();
        const auto mib = [](SizeType b) {
            return static_cast<double>(b) / 1048576.0;
        };
        const auto summary = fmt::format(
            "*** FDMT (cuda) ***\n"
            "Mode: {}, device {}, nbeams {}, chunk {} of {} samples, "
            "fuse_levels {}, int_tree {}\n"
            "Device memory: plan {:.1f} MiB, state {:.1f} MiB, history {:.1f} "
            "MiB, workspace {:.1f} MiB (total {:.1f} MiB); output buffer "
            "{:.1f} MiB "
            "per execute()\n",
            fdmt_mode_to_string(m_mode), m_device_id, m_nbeams, m_chunk,
            m_plan.get_nsamps(), get_fuse_levels(), m_int_tree, mib(mem.plan),
            mib(mem.state), mib(mem.history), mib(mem.workspace),
            mib(mem.plan + mem.state + mem.history + mem.workspace),
            mib(mem.output));
        return m_plan.summary() + summary;
    }

    [[nodiscard]] std::string search_path() const override {
        std::string path;
        if (m_direct) {
            path = fmt::format("{}direct levels 0-{}",
                               m_direct->skewed ? "skewed " : "",
                               m_direct->levels);
        }
        if (m_top) {
            path +=
                fmt::format("{}direct top {} -> {}", path.empty() ? "" : ", ",
                            m_top->level, m_top->target);
        }
        return path;
    }

    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept override {
        const auto coord_bytes = [](const plans::FDMTCoordD& c) {
            return (c.nsamps.size() + c.buf_offset.size() + c.offset.size() +
                    c.tail_buf_offset.size() + c.tail_nsamps.size() +
                    c.head_buf_offset.size() + c.head_nsamps.size() +
                    c.hist_offset.size()) *
                   sizeof(int);
        };
        const auto& sh = m_plan_d.state_shape;
        const auto& g0 = m_plan_d.grids0;
        const SizeType plan_bytes =
            coord_bytes(m_plan_d.coordinates_sum) +
            coord_bytes(m_plan_d.coordinates_copy) +
            ((sh.nchans.size() + sh.ncoords_sum.size() +
              sh.ncoords_copy.size() + sh.nsamps.size() + sh.dt_max.size() +
              g0.dt_grid.size() + g0.ndt.size() + g0.coord_offset.size()) *
             sizeof(int));
        SizeType fused_bytes = 0;
        if (m_fused) {
            fused_bytes = (m_fused->group_info.size() + m_fused->coords.size() +
                           m_fused->dt_grid0.size() + m_fused->ndt0.size() +
                           m_fused->coord_offset0.size()) *
                          sizeof(int);
        }
        return {
            .plan  = plan_bytes,
            .state = m_state_internal_d.size() * sizeof(float),
            .history =
                (m_history_a_d.size() + m_history_b_d.size() +
                 m_tree_fifo_d[0].size() + m_tree_fifo_d[1].size() +
                 m_tree_ring_d.size() + (m_top ? m_top->ring.size() : 0)) *
                    sizeof(float) +
                direct_window_bytes() +
                m_integer_ring.size() * sizeof(uint64_t) +
                m_integer32_ring.size() * sizeof(uint32_t),
            // Fused or direct tables, plus the host-execute() staging once
            // allocated.
            .workspace = fused_bytes + direct_workspace_bytes() +
                         top_workspace_bytes() +
                         (m_stage_wf_f32_d.size() * sizeof(float)) +
                         m_stage_wf_packed_d.size() +
                         (m_stage_dmt_d.size() * sizeof(float)),
            .output    = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
        };
    }

    void advance(SizeType levels, Stream stream_in) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        gpu_utils::set_device(m_device_id);
        m_host_level_valid         = false;
        cudaStream_t stream        = to_cuda(stream_in);
        cudaStream_t active_stream = stream ? stream : m_stream;
        const auto total_lvl       = total_levels();
        const auto target_level =
            std::min(m_current_level + levels, total_lvl - 1);

        while (m_current_level < target_level) {
            execute_iter_device(m_current_level + 1, active_stream);
            m_current_level++;
        }
        m_device_work.mark(active_stream);
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const auto target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] DeviceSpan<const float>
    view_level_data_device() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        const auto& shape = m_plan.get_container().state_shape[m_current_level];
        // The span covers through the last beam's valid elements.
        return {current_level_f32(),
                ((m_nbeams - 1) * level_beam_stride(m_current_level)) +
                    shape.nelements,
                device()};
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        const float* level = host_level_snapshot();
        return {level,
                m_plan.get_container().state_shape[m_current_level].nelements};
    }

    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        check_subband_idx(subband_idx);
        const auto& grid =
            m_plan.get_container().grids[m_current_level][subband_idx];
        const auto nsamps =
            m_plan.get_container().state_shape[m_current_level].nsamps;
        const auto offset = grid.coord_offset * nsamps;
        const auto count  = grid.ndt * nsamps;
        return cuda::std::span<const float>(current_level_f32() + offset,
                                            count);
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        const float* level = host_level_snapshot();
        check_subband_idx(subband_idx);
        const auto& grid =
            m_plan.get_container().grids[m_current_level][subband_idx];
        const auto nsamps =
            m_plan.get_container().state_shape[m_current_level].nsamps;
        return FDMTSubbandView{
            .data = std::span<const float>(level + (grid.coord_offset * nsamps),
                                           grid.ndt * nsamps),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
            .chan_begin  = grid.chan_begin,
            .chan_end    = grid.chan_end,
            .staggered   = grid.staggered,
        };
    }

    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        check_subband_idx(subband_idx);
        const auto& grid =
            m_plan.get_container().grids[m_current_level][subband_idx];
        const auto nsamps =
            m_plan.get_container().state_shape[m_current_level].nsamps;
        const auto data = view_subband_data(subband_idx);
        return FDMTSubbandDeviceView{
            .data = DeviceSpan<const float>(data.data(), data.size(), device()),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
            .chan_begin  = grid.chan_begin,
            .chan_end    = grid.chan_end,
            .staggered   = grid.staggered,
        };
    }

    SizeType current_level() const noexcept override { return m_current_level; }
    SizeType total_levels() const noexcept { return m_plan.get_niters() + 1; }
    SizeType num_subbands() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }
    bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream_in) override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        gpu_utils::set_device(m_device_id);
        cudaStream_t stream        = to_cuda(stream_in);
        cudaStream_t active_stream = stream ? stream : m_stream;
        advance_until_remaining(0, Stream{active_stream});

        // The root is written straight into dmt: nothing to copy.
        m_device_work.mark(active_stream);
        if (!m_host_dmt.empty()) {
            // Host-memory stepper: the result goes back to the caller's
            // buffer (through the staging stream).
            gpu_utils::check_gpu_call(cudaStreamSynchronize(active_stream));
            copy_result_to_host(m_host_dmt);
            m_host_dmt = {};
        }
        m_host_level_valid = false;
        m_is_initialized   = false;
    }

    /// Zeroes all cross-block history (level-0 and tree-level) so the next
    /// execute()/reset() call starts as if this were a brand-new instance.
    /// Mirrors the CPU engine's reset_history().
    void reset_history() noexcept override {
        m_chunk_pos = 0;
        m_next_pos  = 0;
        gpu_utils::set_device(m_device_id);
        for (auto* v : {&m_history_a_d, &m_history_b_d, &m_tree_fifo_d[0],
                        &m_tree_fifo_d[1], &m_tree_ring_d}) {
            if (!v->empty()) {
                v->assign(v->size(), 0.0F);
            }
        }
        if (!m_integer_ring.empty()) {
            m_integer_ring.assign(m_integer_ring.size(), 0);
        }
        if (!m_integer32_ring.empty()) {
            m_integer32_ring.assign(m_integer32_ring.size(), 0);
        }
        if (m_direct) {
            for (auto& w : m_direct->win) {
                w.assign(w.size(), 0.0F);
            }
            if (!m_direct->win_ring.empty()) {
                m_direct->win_ring.assign(m_direct->win_ring.size(), 0.0F);
            }
            if (!m_direct->win_ring_u8.empty()) {
                m_direct->win_ring_u8.assign(m_direct->win_ring_u8.size(), 0);
            }
            if (!m_direct->win_ring_u16.empty()) {
                m_direct->win_ring_u16.assign(m_direct->win_ring_u16.size(), 0);
            }
            // An all-zero window is consistent with all-zero tree histories.
            m_direct_filled = static_cast<SizeType>(m_direct->window);
        }
        m_direct_stale = false;
        if (m_top) {
            m_top->ring.assign(m_top->ring.size(), 0.0F);
            m_top_filled = static_cast<SizeType>(m_top->max_ring);
        }
        m_top_stale = false;
        m_parity    = false;
        // Abandons any unfinished stepper block along with its stream.
        m_is_initialized = false;
    }

    /// Size (floats) of this instance's "valid"-mode streaming history, as
    /// saved/restored by save_history()/load_history(): the level-0 input
    /// history, then the tree history of each beam in the plan's hist_offset
    /// layout, oldest sample first (the CPU engine's format). Zero for
    /// "full"/"roll" mode instances.
    ///
    /// Of each ping-pong pair only the buffer the next chunk reads is live
    /// (selected by m_parity); the ring is live as a whole.
    [[nodiscard]] SizeType history_state_size() const noexcept override {
        if (skew_history()) {
            SizeType n = kSkewHistHeader;
            for (const auto& b : skew_hist_buffers()) {
                n += b.size;
            }
            return n;
        }
        return m_mode == FDMTMode::kValid && m_search_nbits == 32
                   ? m_history_a_d.size() +
                         (m_plan.get_tree_history_size() == 0
                              ? 0
                              : m_nbeams * m_plan.get_tree_history_size())
                   : 0;
    }

    /// Copies the live streaming history out to a caller-owned device
    /// buffer, so one shared FDMT instance can multiplex several
    /// independent streams (each with its own history) instead of requiring
    /// one instance per stream. Enqueued on `stream`; synchronize before
    /// reading `out` elsewhere.
    void save_history(DeviceSpan<float> out_d,
                      Stream stream_in) const override {
        const cuda::std::span<float> out = to_cuda(out_d);
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT::save_history: Invalid output size. Expected {}, "
                "got {}",
                history_state_size(), out.size()));
        }
        gpu_utils::set_device(m_device_id);
        save_history_d(out.data(), to_cuda(stream_in));
        m_device_work.mark(to_cuda(stream_in));
    }

    /// Replaces the live streaming history with a buffer previously produced
    /// by save_history() (from an instance built with the same plan
    /// geometry), resuming that stream. Fully asynchronous on `stream` --
    /// use reset_history() instead to start a stream cold.
    void load_history(DeviceSpan<const float> in_d, Stream stream_in) override {
        const cuda::std::span<const float> in = to_cuda(in_d);
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT::load_history: Invalid input size. Expected {}, "
                "got {}",
                history_state_size(), in.size()));
        }
        gpu_utils::set_device(m_device_id);
        load_history_d(in.data(), to_cuda(stream_in));
        m_device_work.mark(to_cuda(stream_in));
    }

    // Host-memory history: the device format through a temporary device
    // buffer, after the stepper's queue drains.
    void save_history(std::span<float> out) const override {
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT::save_history: Invalid output size. Expected {}, "
                "got {}",
                history_state_size(), out.size()));
        }
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        thrust::device_vector<float> staged(out.size());
        save_history_d(thrust::raw_pointer_cast(staged.data()), nullptr);
        gpu_utils::check_gpu_call(
            cudaMemcpy(out.data(), thrust::raw_pointer_cast(staged.data()),
                       out.size_bytes(), cudaMemcpyDeviceToHost),
            "FDMT::save_history");
    }

    void load_history(std::span<const float> in) override {
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(fmt::format(
                "FDMT::load_history: Invalid input size. Expected {}, "
                "got {}",
                history_state_size(), in.size()));
        }
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        thrust::device_vector<float> staged(in.begin(), in.end());
        load_history_d(thrust::raw_pointer_cast(staged.data()), nullptr);
        gpu_utils::check_gpu_call(cudaStreamSynchronize(nullptr),
                                  "FDMT::load_history");
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    [[nodiscard]] Device device() const noexcept {
        return {.backend = detail::kGPUBackend, .id = m_device_id};
    }

    // Host copy of beam 0's current level for the host view_* methods,
    // refreshed only after the stepper moved (see m_host_level_valid).
    [[nodiscard]] const float* host_level_snapshot() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        const float* level_d = current_level_f32();
        if (!m_host_level_valid) {
            const auto nelements =
                m_plan.get_container().state_shape[m_current_level].nelements;
            m_host_level.resize(nelements);
            gpu_utils::set_device(m_device_id);
            m_device_work.wait();
            cudaMemcpy(m_host_level.data(), level_d, nelements * sizeof(float),
                       cudaMemcpyDeviceToHost);
            gpu_utils::check_last_gpu_error(
                "FDMT::view (host): D->H copy failed");
            m_host_level_valid = true;
        }
        return m_host_level.data();
    }

    int m_device_id;
    bool m_use_box_smearing;
    bool m_int_tree; // performance parameter (packed input)
    // Level-0 box means (fdmt_box_weights.hpp) by |dt| = s: dyadic float
    // weights, packed-input rounding recipes on the scale 2^-F and whether
    // their products need 64 bits (by nbits), and the value 2^-F of one
    // integer state unit (1 without box smearing).
    thrust::device_vector<float> m_box_w_d;
    thrust::device_vector<detail::FDMTBoxRound> m_box_round_d;
    std::array<bool, 17> m_box_round_u64{};
    float m_int_unit{1.0F};

    // The box means as kernel arguments, for packed input of `nbits` (0:
    // float input).
    [[nodiscard]] detail::FDMTBoxMeans box_means(SizeType nbits) const {
        return {
            .w_f32     = thrust::raw_pointer_cast(m_box_w_d.data()),
            .round     = thrust::raw_pointer_cast(m_box_round_d.data()),
            .unit      = m_int_unit,
            .round_u64 = m_box_round_u64[nbits] ? 1 : 0,
        };
    }

    template <typename Input>
    [[nodiscard]] detail::FDMTBoxMeans box_means_for(const Input& in) const {
        if constexpr (detail::kFDMTPackedInput<Input>) {
            return box_means(static_cast<SizeType>(in.nbits));
        } else {
            return box_means(0);
        }
    }
    FDMTMode m_mode;
    SizeType m_nbeams{1};
    const plans::FDMTPlan& m_plan; // owned by the FDMT facade
    // Chunked execution (fdmt_chunk.hpp): row length of the tree state
    // below the root, level 0's row stride, the elements of one beam's level
    // in a state half, and the current chunk's length.
    SizeType m_chunk{0};
    SizeType m_row_len0{0};
    SizeType m_half_elems{0};
    SizeType m_skew_nbatches{1};
    std::vector<int> m_skew_batch_top_grid;
    std::vector<std::vector<int>> m_skew_batch_sum_off;
    std::vector<std::vector<int>> m_skew_batch_copy_off;
    SizeType m_active{0};
    plans::FDMTPlanContainerD m_plan_d; // re-laid out for m_chunk
    // The two state halves (see init_device_structures()).
    thrust::device_vector<float> m_state_internal_d;
    // Level-0 history buffer for valid-mode streaming across FDMT blocks
    // (ping-pong pair)
    thrust::device_vector<float> m_history_a_d;
    thrust::device_vector<float> m_history_b_d;
    float* m_level0_hist_in_ptr{nullptr};
    float* m_level0_hist_out_ptr{nullptr};
    // Tree-level (level >= 1) cross-block history, per beam in the plan's
    // hist_offset layout of tree_history_size() elements:
    // - levels 1..m_fifo_levels (the fused levels, which merge and update
    //   their history inside one kernel across concurrent tiles) keep an
    //   oldest-first FIFO, ping-ponged between two buffers of m_fifo_size;
    // - the levels above keep one in-place ring of m_ring_size per beam
    //   (head sample of stream time u at u mod delay): each chunk writes
    //   only its newest samples, and no history is shifted or doubled.
    // The level-0 input history and the FIFOs flip on m_parity per chunk.
    std::array<thrust::device_vector<float>, 2> m_tree_fifo_d;
    mutable thrust::device_vector<float> m_tree_ring_d;
    SizeType m_fifo_levels{0};
    SizeType m_fifo_size{0};
    mutable SizeType m_ring_size{0};
    SizeType m_ring_low_base{0};
    SizeType m_ring_low_size{0};
    SizeType m_ring_high_orig_base{0};
    SizeType m_ring_high_size{0};
    mutable bool m_full_tree_ring{false};
    std::vector<SizeType> m_level_max_extra;
    bool m_parity{false};
    float* m_fifo_in_ptr{nullptr};
    float* m_fifo_out_ptr{nullptr};
    // Stream sample of the current chunk's start, and of the next chunk's.
    uint64_t m_chunk_pos{0};
    uint64_t m_next_pos{0};

    // Per-level storage types (all float, and per nbits for int_tree), fixed
    // at construction.
    using Elem = detail::FDMTLevelType;
    std::vector<Elem> m_float_levels;
    std::array<std::vector<Elem>, 17> m_int_levels; // indexed by nbits

    SizeType m_search_max_fuse{SIZE_MAX};
    SizeType m_search_nbits{32};
    bool m_search_u32{false};
    // The constructor's fuse_levels (the packed search runs no tile fusion,
    // but takes the direct levels when this is automatic).
    SizeType m_requested_fuse{kFDMTAutoFuse};
    /// 32-bit words per direct-top ring / trial element: 2 for the packed
    /// search's uint64_t state, else 1 (float or uint32_t).
    [[nodiscard]] SizeType top_words() const noexcept {
        return (m_search_nbits != 32 && !m_search_u32) ? 2 : 1;
    }
    // Packed search: exact integer tree history, a ring at every level.
    thrust::device_vector<uint64_t> m_integer_ring;
    thrust::device_vector<uint32_t> m_integer32_ring;

    // Level fusion (fuse_levels): the resolved depth's
    // device tables and launch geometry; empty when execute() is unfused.
    struct FusedLevelsD {
        DeviceVector<int> group_info;
        DeviceVector<int> coords;
        DeviceVector<int> dt_grid0;
        DeviceVector<int> ndt0;
        DeviceVector<int> coord_offset0;
        detail::FDMTFusedTileArgs args{};
        int tile{0};
        int ntiles{0}; ///< for a whole row (valid-mode chunks: per launch)
        int ngroups{0};
        SizeType smem_bytes{0};
    };
    std::unique_ptr<FusedLevelsD> m_fused;
    // Direct low levels (configure_direct()): the device tables, the input
    // window ping-pong (flipped with m_parity) and the per-chunk running
    // sums. Replaces m_fused when configured.
    static constexpr SizeType kDirectLevels = 3;
    struct DirectLevelsD {
        SizeType levels{0};
        bool skewed{false};
        int window{0};
        int nsubs{0};
        int max_rows{0};
        int max_extra{0};
        SizeType total_elems{0};
        int cpass{0};
        int pitch{0};
        SizeType smem{0};
        int64_t win_ring_size{0};
        std::vector<int32_t> row_off;
        std::vector<int32_t> batch_sub_off;
        thrust::device_vector<detail::FDMTDirectSub> subs;
        thrust::device_vector<int32_t> leaf_off;
        thrust::device_vector<detail::FDMTDirectLeaf> leaves;
        thrust::device_vector<detail::FDMTDirectLeaf> hist_leaves;
        std::vector<thrust::device_vector<detail::FDMTDirectHistCoord>>
            hist_coords;
        std::vector<int32_t> hist_max_delay;
        std::array<thrust::device_vector<float>, 2> win;
        thrust::device_vector<float> prefix;
        thrust::device_vector<int32_t> chan_ring_len;
        thrust::device_vector<int64_t> chan_ring_off;
        thrust::device_vector<float> win_ring;
        thrust::device_vector<uint8_t> win_ring_u8;
        thrust::device_vector<uint16_t> win_ring_u16;
    };
    std::unique_ptr<DirectLevelsD> m_direct;
    bool m_direct_eligible{false};
    const float* m_direct_win_in{nullptr};
    float* m_direct_win_out{nullptr};
    // Real input samples in the window (a load_history() empties it).
    SizeType m_direct_filled{0};
    // Levels 1..K tree histories lag the stream (direct chunks ran).
    mutable bool m_direct_stale{false};
    // Direct top (configure_top()): trials summed from level k's rows.
    struct DirectTopD {
        SizeType level{0};
        SizeType target{0};
        bool skewed{false};
        bool target_ring{false};
        int ntrials{0};
        int nrows{0};
        int max_ring{0};
        int64_t ring_size{0}; ///< per beam
        thrust::device_vector<int32_t> out_row;
        thrust::device_vector<int32_t> leaf_off;
        thrust::device_vector<detail::FDMTDirectTopLeaf> leaves;
        thrust::device_vector<int32_t> ring_len;
        thrust::device_vector<int64_t> ring_off;
        thrust::device_vector<float> ring;
        thrust::device_vector<detail::FDMTDirectTopLeaf> hist_leaves;
        std::vector<thrust::device_vector<detail::FDMTDirectHistCoord>>
            hist_coords;
        std::vector<int32_t> hist_max_delay;
        thrust::device_vector<float> trials; ///< search: one chunk's trials
    };
    std::unique_ptr<DirectTopD> m_top;
    bool m_search_upper{false};
    bool m_force_skewed{false}; // FDMTEngineConfig::force_skewed_direct
    bool m_top_chunk{false};
    // Samples of the rings' reach streamed since a load_history().
    SizeType m_top_filled{0};
    // Levels above k: tree histories lag the stream (top chunks ran).
    mutable bool m_top_stale{false};
    // Auto depth: the deepest fusion whose shared memory at a tile of at
    // least kMinAutoTile samples fits auto_smem_bytes() (kAutoBlocksPerSM
    // resident blocks per SM) with a level-0 halo of at most half a tile.
    // Four blocks (32 warps) hide the fused kernel's global-load latency
    // better than larger tiles at two blocks save halo work (L40S: 4.9 vs
    // 5.2 ms at 4096 channels, 16384 samples, 2049 DMs).
    static constexpr SizeType kDefaultSmemBytes = SizeType{48} * 1024;
    static constexpr int kAutoBlocksPerSM       = 4;
    static constexpr int kMinAutoTile           = 128;
    static constexpr int kMaxTile               = 1024;
    static constexpr int kFusedThreads          = 256;

    // One tree level's device state: beam 0 at `base`, beam b at element
    // offset b * level_beam_stride(level), stored as `type`.
    struct LevelBufD {
        std::byte* base{nullptr};
        Elem type{Elem::kF32};
    };

    // Stepper state
    bool m_is_initialized{false};
    SizeType m_current_level{0};
    std::vector<LevelBufD> m_levels_d;
    float* m_dmt_target_ptr{nullptr};
    cudaStream_t m_stream{nullptr};
    // End of the latest device-memory call; host-memory calls wait for it.
    mutable gpu_utils::DeviceWorkFence m_device_work;
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;
    // Host-memory execute() staging (see stage_to_device()).
    thrust::device_vector<float> m_stage_wf_f32_d;
    thrust::device_vector<uint8_t> m_stage_wf_packed_d;
    thrust::device_vector<float> m_stage_dmt_d;
    // Host-memory stepper: the caller's dmt buffer (empty after a
    // device-memory reset) and the host snapshot behind the view_* methods.
    std::span<float> m_host_dmt;
    gpu_host::ChunkedStager m_host; // pinned pipeline of the host overloads
    mutable std::vector<float> m_host_level;
    mutable bool m_host_level_valid{false};

    // The device plan stores per-beam coordinate offsets as 32-bit ints (see
    // plans_cuda.cuh), and the kernels index within one beam in 32-bit
    // arithmetic (only the beam-strided part is 64-bit). Reject plans whose
    // per-beam extents do not fit, instead of silently wrapping indices.
    void check_int32_extents() const {
        constexpr auto kMax =
            static_cast<SizeType>(std::numeric_limits<int32_t>::max());
        const auto check = [&](SizeType value, std::string_view what) {
            if (value > kMax) {
                throw std::invalid_argument(fmt::format(
                    "FDMT (cuda): per-beam {} ({} elements) exceeds the 32-bit "
                    "index range of the CUDA backend ({}); use a smaller "
                    "chunk_nsamps or block (nsamps), or fewer DM trials",
                    what, value, kMax));
            }
        };
        check(m_half_elems, "state level");
        check(m_plan.get_dmt_size(), "output");
        check(m_plan.get_nchans() * m_plan.get_nsamps(), "waterfall");
        check(m_plan.get_history_size(), "level-0 history");
        // Tree history offsets are unsigned 32-bit (plans_cuda.cuh).
        if (m_plan.get_tree_history_size() >
            static_cast<SizeType>(std::numeric_limits<uint32_t>::max())) {
            throw std::invalid_argument(fmt::format(
                "FDMT (cuda): per-beam tree history ({} elements) exceeds the "
                "unsigned 32-bit index range of the CUDA backend ({}); use "
                "fewer DM trials or a lower maximum DM",
                m_plan.get_tree_history_size(),
                std::numeric_limits<uint32_t>::max()));
        }
    }

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != m_nbeams * nchans * nsamps) {
            throw std::invalid_argument(
                fmt::format("FDMT (cuda): Invalid size of waterfall. "
                            "Expected {}, got {}",
                            m_nbeams * nchans * nsamps, waterfall_size));
        }
        if (dmt_size < m_nbeams * m_plan.get_dmt_size()) {
            throw std::invalid_argument(
                fmt::format("FDMT (cuda): Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_nbeams * m_plan.get_dmt_size(), dmt_size));
        }
    }

    void check_packed_inputs(SizeType waterfall_bytes,
                             SizeType nbits,
                             SizeType dmt_size) const {
        if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 &&
            nbits != 16) {
            throw std::invalid_argument(fmt::format(
                "FDMT (cuda): nbits={} must be one of 1, 2, 4, 8, 16", nbits));
        }
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
        const auto expected = m_nbeams * m_plan.get_nchans() * row_bytes;
        if (waterfall_bytes != expected) {
            throw std::invalid_argument(fmt::format(
                "FDMT (cuda): Invalid size of packed waterfall (nbits={}). "
                "Expected {} bytes, got {}",
                nbits, expected, waterfall_bytes));
        }
        if (dmt_size < m_nbeams * m_plan.get_dmt_size()) {
            throw std::invalid_argument(
                fmt::format("FDMT (cuda): Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_nbeams * m_plan.get_dmt_size(), dmt_size));
        }
    }

    void check_subband_idx(SizeType subband_idx) const {
        const auto nsubs = m_plan.get_container().grids[m_current_level].size();
        if (subband_idx >= nsubs) {
            throw std::out_of_range(
                fmt::format("FDMT (cuda): Subband index {} out of range "
                            "(current level has {} "
                            "subbands)",
                            subband_idx, nsubs));
        }
    }

    void ensure_full_tree_history(cudaStream_t stream) const {
        if (m_full_tree_ring || m_mode != FDMTMode::kValid ||
            m_search_nbits != 32) {
            return;
        }
        const SizeType full_ring = m_plan.get_tree_history_size() - m_fifo_size;
        thrust::device_vector<float> expanded(m_nbeams * full_ring, 0.0F);
        if (m_ring_low_size > 0 && !m_tree_ring_d.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(
                    thrust::raw_pointer_cast(expanded.data()) +
                        (m_ring_low_base - m_fifo_size),
                    full_ring * sizeof(float),
                    thrust::raw_pointer_cast(m_tree_ring_d.data()),
                    m_ring_size * sizeof(float),
                    m_ring_low_size * sizeof(float), m_nbeams,
                    cudaMemcpyDeviceToDevice, stream),
                "FDMT::ensure_full_tree_history (low)");
        }
        if (m_ring_high_size > 0 && !m_tree_ring_d.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(
                    thrust::raw_pointer_cast(expanded.data()) +
                        (m_ring_high_orig_base - m_fifo_size),
                    full_ring * sizeof(float),
                    thrust::raw_pointer_cast(m_tree_ring_d.data()) +
                        m_ring_low_size,
                    m_ring_size * sizeof(float),
                    m_ring_high_size * sizeof(float), m_nbeams,
                    cudaMemcpyDeviceToDevice, stream),
                "FDMT::ensure_full_tree_history (high)");
        }
        m_tree_ring_d.swap(expanded);
        m_ring_size      = full_ring;
        m_full_tree_ring = true;
    }

    // ---- skewed-window streaming history (m_direct->skewed) -------------
    //
    // The skewed path keeps no tree history: its state is the raw input
    // ring, the direct top's target rings and any active tree ring / FIFO /
    // level-0 history, all tied to the stream position. They are saved
    // verbatim after a header of 16-bit pieces held exactly in floats
    // (magic, version, layout hash, stream position). A buffer is accepted
    // only by an engine with the same skewed layout; anything else throws.
    static constexpr SizeType kSkewHistHeader  = 16;
    static constexpr uint64_t kSkewHistMagic   = 0x534BU; // "SK"
    static constexpr uint64_t kSkewHistVersion = 1;

    [[nodiscard]] bool skew_history() const noexcept {
        return m_mode == FDMTMode::kValid && m_search_nbits == 32 && m_direct &&
               m_direct->skewed;
    }

    struct SkewHistBuffer {
        float* data;
        SizeType size;
    };
    /// The live persistent buffers, in save order (the ping-pong buffers
    /// the next chunk reads).
    [[nodiscard]] std::vector<SkewHistBuffer> skew_hist_buffers() const {
        std::vector<SkewHistBuffer> v;
        const auto add = [&](const thrust::device_vector<float>& b) {
            if (!b.empty()) {
                v.push_back({const_cast<float*>( // NOLINT
                                 thrust::raw_pointer_cast(b.data())),
                             b.size()});
            }
        };
        add(m_direct->win_ring);
        if (m_top) {
            add(m_top->ring);
        }
        add(m_tree_ring_d);
        add(next_tree_fifo());
        add(next_level0_history());
        return v;
    }

    [[nodiscard]] uint64_t skew_layout_hash() const {
        uint64_t h     = 1469598103934665603ULL;
        const auto mix = [&](uint64_t x) {
            for (int i = 0; i < 8; ++i) {
                h = (h ^ ((x >> (8 * i)) & 0xFFU)) * 1099511628211ULL;
            }
        };
        mix(m_plan.get_nchans());
        mix(m_nbeams);
        mix(m_chunk);
        mix(m_direct->levels);
        mix(static_cast<uint64_t>(m_direct->win_ring_size));
        mix(m_top ? m_top->level : 0);
        mix(m_top ? static_cast<uint64_t>(m_top->ring_size) : 0);
        for (const auto& b : skew_hist_buffers()) {
            mix(b.size);
        }
        return h;
    }

    void save_skew_history_d(float* dst, cudaStream_t stream) const {
        std::array<float, kSkewHistHeader> hdr{};
        const auto put16 = [&](SizeType at, uint64_t x, SizeType words) {
            for (SizeType i = 0; i < words; ++i) {
                hdr[at + i] = static_cast<float>((x >> (16 * i)) & 0xFFFFU);
            }
        };
        put16(0, kSkewHistMagic, 1);
        put16(1, kSkewHistVersion, 1);
        put16(2, skew_layout_hash(), 4);
        put16(6, m_next_pos, 4);
        // Pageable source: returns once the header has been staged.
        gpu_utils::check_gpu_call(cudaMemcpyAsync(dst, hdr.data(), sizeof(hdr),
                                                  cudaMemcpyHostToDevice,
                                                  stream),
                                  "FDMT::save_history");
        SizeType off = kSkewHistHeader;
        for (const auto& b : skew_hist_buffers()) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(dst + off, b.data, b.size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "FDMT::save_history");
            off += b.size;
        }
    }

    void load_skew_history_d(const float* src, cudaStream_t stream) {
        std::array<float, kSkewHistHeader> hdr{};
        gpu_utils::check_gpu_call(cudaMemcpyAsync(hdr.data(), src, sizeof(hdr),
                                                  cudaMemcpyDeviceToHost,
                                                  stream),
                                  "FDMT::load_history");
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream),
                                  "FDMT::load_history");
        const auto get16 = [&](SizeType at, SizeType words) {
            uint64_t x = 0;
            for (SizeType i = 0; i < words; ++i) {
                const float f = hdr[at + i];
                if (!(f >= 0.0F && f <= 65535.0F) || f != std::floor(f)) {
                    throw std::invalid_argument(
                        "FDMT::load_history: not a skewed-window history");
                }
                x |= static_cast<uint64_t>(f) << (16 * i);
            }
            return x;
        };
        if (get16(0, 1) != kSkewHistMagic || get16(1, 1) != kSkewHistVersion) {
            throw std::invalid_argument(
                "FDMT::load_history: not a skewed-window history (it was "
                "saved by a different engine path)");
        }
        if (get16(2, 4) != skew_layout_hash()) {
            throw std::invalid_argument(
                "FDMT::load_history: the history was saved by an engine with "
                "a different skewed-window layout (plan, chunk_nsamps, beams "
                "or direct levels)");
        }
        const uint64_t pos = get16(6, 4);
        SizeType off       = kSkewHistHeader;
        for (const auto& b : skew_hist_buffers()) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(b.data, src + off, b.size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "FDMT::load_history");
            off += b.size;
        }
        m_chunk_pos      = pos;
        m_next_pos       = pos;
        m_is_initialized = false;
        m_direct_stale   = false;
        m_top_stale      = false;
        m_direct_filled  = static_cast<SizeType>(m_direct->window);
        if (m_top) {
            m_top_filled = static_cast<SizeType>(m_top->max_ring);
        }
    }

    // Saved history: level 0, then per beam the tree history in the plan
    // layout, oldest first (the FIFOs as stored, the ring linearized).
    void save_history_d(float* dst, cudaStream_t stream) const {
        if (skew_history()) {
            save_skew_history_d(dst, stream);
            return;
        }
        ensure_full_tree_history(stream);
        rebuild_direct_history(stream);
        rebuild_top_history(stream);
        const auto& level0 = next_level0_history();
        if (!level0.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(dst, thrust::raw_pointer_cast(level0.data()),
                                level0.size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "FDMT::save_history");
        }
        float* tree      = dst + level0.size();
        const auto total = m_plan.get_tree_history_size();
        if (m_fifo_size > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(
                    tree, total * sizeof(float),
                    thrust::raw_pointer_cast(next_tree_fifo().data()),
                    m_fifo_size * sizeof(float), m_fifo_size * sizeof(float),
                    m_nbeams, cudaMemcpyDeviceToDevice, stream),
                "FDMT::save_history");
        }
        for_each_ring_level(
            [&](plans::FDMTCoordDPtrs coords, int count, int max_delay) {
                const dim3 block(256);
                const dim3 grid((max_delay + 255) / 256,
                                std::min(count, kMaxGridY), m_nbeams);
                kernel_tree_ring_to_fifo<<<grid, block, 0, stream>>>(
                    thrust::raw_pointer_cast(m_tree_ring_d.data()), tree,
                    coords, count, static_cast<int64_t>(m_ring_size),
                    static_cast<int64_t>(total),
                    static_cast<int64_t>(m_fifo_size), m_next_pos);
            });
        gpu_utils::check_last_gpu_error("FDMT::save_history failed");
    }

    // Inverse of save_history_d(): an oldest-first history is the ring at
    // stream position 0.
    void load_history_d(const float* src, cudaStream_t stream) {
        if (skew_history()) {
            load_skew_history_d(src, stream);
            return;
        }
        ensure_full_tree_history(stream);
        auto& level0 = next_level0_history();
        if (!level0.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(thrust::raw_pointer_cast(level0.data()), src,
                                level0.size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream),
                "FDMT::load_history");
        }
        const float* tree = src + level0.size();
        const auto total  = m_plan.get_tree_history_size();
        const auto copy   = [&](float* dst, SizeType pitch, SizeType from,
                                SizeType width) {
            if (width > 0) {
                gpu_utils::check_gpu_call(
                    cudaMemcpy2DAsync(dst, pitch * sizeof(float), tree + from,
                                      total * sizeof(float),
                                      width * sizeof(float), m_nbeams,
                                      cudaMemcpyDeviceToDevice, stream),
                    "FDMT::load_history");
            }
        };
        if (!m_tree_fifo_d[0].empty() || !m_tree_ring_d.empty()) {
            copy(thrust::raw_pointer_cast(next_tree_fifo().data()), m_fifo_size,
                 0, m_fifo_size);
            copy(thrust::raw_pointer_cast(m_tree_ring_d.data()), m_ring_size,
                 m_fifo_size, m_ring_size);
        }
        m_chunk_pos      = 0;
        m_next_pos       = 0;
        m_is_initialized = false;
        // The loaded format holds no input window: execute() runs the tree
        // (from the loaded histories) until the window has refilled.
        m_direct_filled = 0;
        m_direct_stale  = false;
        m_top_filled    = 0;
        m_top_stale     = false;
    }

    // f(coords, count, max delay) over the ring levels' sum coordinates,
    // as one contiguous range of the device table.
    template <typename F> void for_each_ring_level(F&& f) const {
        if (m_tree_ring_d.empty()) {
            return;
        }
        const auto& pc   = m_plan.get_container();
        const auto first = m_fifo_levels + 1;
        int count        = 0;
        int max_delay    = 0;
        for (SizeType l = first; l <= m_plan.get_niters(); ++l) {
            count += static_cast<int>(pc.state_shape[l].ncoords_sum);
            max_delay =
                std::max(max_delay, static_cast<int>(pc.state_shape[l].dt_max));
        }
        if (count == 0 || max_delay == 0) {
            return;
        }
        auto coords = m_plan_d.coordinates_sum.get_raw_ptrs();
        coords.update_offsets(m_coords_sum_offsets[first]);
        f(coords, count, max_delay);
    }

    // The ping-pong buffers the next chunk's start() reads as `in` (see the
    // parity swap there): the only live ones between chunks.
    [[nodiscard]] thrust::device_vector<float>& next_level0_history() {
        return m_parity ? m_history_b_d : m_history_a_d;
    }
    [[nodiscard]] const thrust::device_vector<float>&
    next_level0_history() const {
        return m_parity ? m_history_b_d : m_history_a_d;
    }
    [[nodiscard]] thrust::device_vector<float>& next_tree_fifo() {
        return m_tree_fifo_d[m_parity ? 1 : 0];
    }
    [[nodiscard]] const thrust::device_vector<float>& next_tree_fifo() const {
        return m_tree_fifo_d[m_parity ? 1 : 0];
    }

    // Tree history of level l's merges in this chunk (kernel_execute_iter).
    /// kernel_advance_tree_history over `count` coordinates of up to
    /// `max_offset` samples each.
    template <typename TIn, typename THist>
    void launch_advance_tree_history(const TIn* in_ptr,
                                     const plans::FDMTCoordDPtrs& coords,
                                     const THist* hist_in,
                                     THist* hist_out,
                                     int count,
                                     int max_offset,
                                     int in_state_nelements,
                                     int64_t hist_stride,
                                     int64_t hist_base,
                                     uint64_t ring_pos,
                                     float int_unit,
                                     cudaStream_t stream) const {
        const dim3 block(kAdvanceWarps * 32);
        const dim3 grid((count + kAdvanceWarps - 1) / kAdvanceWarps,
                        (max_offset + kAdvanceSegment - 1) / kAdvanceSegment,
                        m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, block);
        kernel_advance_tree_history<TIn, THist><<<grid, block, 0, stream>>>(
            in_ptr, coords, hist_in, hist_out, count, in_state_nelements,
            hist_stride, hist_base, ring_pos, int_unit,
            static_cast<int>(m_active));
    }

    struct TreeHistory {
        float* in      = nullptr;
        float* out     = nullptr;
        int64_t stride = 0;
        int64_t base   = 0;
        uint64_t ring  = kFifoHistory;
    };
    [[nodiscard]] SizeType ring_base_for_level(SizeType level) const noexcept {
        if (m_full_tree_ring) {
            return m_fifo_size;
        }
        if (m_top && level > m_top->target) {
            return m_ring_high_orig_base - m_ring_low_size;
        }
        return m_ring_low_base;
    }

    [[nodiscard]] TreeHistory tree_history(SizeType level) const {
        if (m_mode != FDMTMode::kValid) {
            return {};
        }
        if (level <= m_fifo_levels) {
            return {m_fifo_in_ptr, m_fifo_out_ptr,
                    static_cast<int64_t>(m_fifo_size), 0, kFifoHistory};
        }
        if (m_direct && m_direct->skewed && m_top && level <= m_top->level) {
            return {nullptr, nullptr, 0, 0, m_chunk_pos};
        }
        auto* ring = const_cast<float*>( // NOLINT: device buffer, host const
            thrust::raw_pointer_cast(m_tree_ring_d.data()));
        return {ring, ring, static_cast<int64_t>(m_ring_size),
                static_cast<int64_t>(ring_base_for_level(level)), m_chunk_pos};
    }

    // Per-beam tree history of levels 1..level (the plan assigns hist
    // offsets level by level).
    [[nodiscard]] SizeType tree_history_through(SizeType level) const {
        SizeType total = 0;
        const auto& pc = m_plan.get_container();
        for (SizeType l = 1; l <= level && l <= m_plan.get_niters(); ++l) {
            for (const auto& c : pc.coordinates_sum[l]) {
                total += c.delay;
            }
        }
        return total;
    }

    // Elements between beams of level l: a state half's level, or the dmt
    // output for the root.
    [[nodiscard]] int level_beam_stride(SizeType l) const noexcept {
        return static_cast<int>(l == m_plan.get_niters() ? m_plan.get_dmt_size()
                                                         : m_half_elems);
    }

    // Calls f(T{}) with the storage type of `e` (tag dispatch: plain C++14
    // generic lambdas, the most portable form across nvcc versions).
    template <typename F> static void with_elem(Elem e, F&& f) {
        switch (e) {
        case Elem::kU8:
            f(uint8_t{});
            break;
        case Elem::kU16:
            f(uint16_t{});
            break;
        case Elem::kF32:
            f(float{});
            break;
        }
    }

    [[nodiscard]] const float* current_level_f32() const {
        if (m_levels_d[m_current_level].type != Elem::kF32) {
            throw std::logic_error(fmt::format(
                "FDMT (cuda): level {} is stored as an integer type "
                "(int_tree); construct with int_tree=false to inspect "
                "intermediate levels.",
                m_current_level));
        }
        return reinterpret_cast<const float*>(m_levels_d[m_current_level].base);
    }

    [[nodiscard]] SizeType max_optin_smem_bytes() const {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   m_device_id) != cudaSuccess ||
            bytes <= 0) {
            return kDefaultSmemBytes;
        }
        return static_cast<SizeType>(bytes);
    }

    /// The device's L2 size in bytes (0 if the query fails).
    [[nodiscard]] SizeType l2_bytes() const {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize,
                                   m_device_id) != cudaSuccess ||
            bytes <= 0) {
            return 0;
        }
        return static_cast<SizeType>(bytes);
    }

    /// Shared memory per fused block for kAutoBlocksPerSM resident blocks
    /// (less the ~1 KiB the runtime reserves per block), at most the
    /// no-opt-in limit.
    [[nodiscard]] SizeType auto_smem_bytes() const {
        constexpr SizeType kReserved = 1024;
        int per_sm                   = 0;
        if (cudaDeviceGetAttribute(&per_sm,
                                   cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                   m_device_id) != cudaSuccess ||
            per_sm <= 0) {
            return kDefaultSmemBytes;
        }
        const auto share = static_cast<SizeType>(per_sm) / kAutoBlocksPerSM;
        return std::min(kDefaultSmemBytes,
                        share > kReserved ? share - kReserved : share);
    }

    /**
     * Resolves the fuse_levels constructor argument for this plan and device
     * and uploads the fused kernel's tables. An explicit depth is clamped to
     * the plan's merge levels and kMaxFusedLevels, and reduced until a tile
     * fits the device's (opt-in) shared memory; kFDMTAutoFuse follows the rule
     * at auto_smem_bytes(). Leaves m_fused empty when no depth qualifies.
     */
    void configure_fusion(const plans::FDMTPlanContainer& pc,
                          SizeType requested) {
        m_fused.reset();
        const SizeType niters  = m_plan.get_niters();
        const SizeType deepest = std::min(
            {niters, SizeType{detail::kMaxFusedLevels}, m_search_max_fuse});
        if (requested == 0 || deepest == 0) {
            return;
        }
        const bool automatic  = requested == kFDMTAutoFuse;
        const SizeType budget = auto_smem_bytes();
        const SizeType optin  = automatic ? budget : max_optin_smem_bytes();
        for (SizeType f = automatic ? deepest : std::min(requested, deepest);
             f >= 1; --f) {
            const auto plan = detail::build_fused_tile_plan(pc, niters, f);
            // Never wider than the (32-rounded) level-F row itself.
            const int max_tile =
                std::min(kMaxTile, ((plan.ntiles_nsamps + 31) / 32) * 32);
            int tile = plan.max_tile_nsamps(budget, max_tile);
            if (automatic) {
                const int min_tile = std::min(kMinAutoTile, max_tile);
                if (tile < min_tile || 2 * plan.max_level0_halo > tile) {
                    continue;
                }
            } else if (tile < 64) {
                tile = std::max(tile, plan.max_tile_nsamps(optin, max_tile));
            }
            if (tile > 0) {
                adopt_fused_plan(plan, tile);
                return;
            }
        }
    }

    void adopt_fused_plan(const detail::FDMTFusedTilePlan& plan, int tile) {
        gpu_utils::set_device(m_device_id);
        auto fused           = std::make_unique<FusedLevelsD>();
        fused->group_info    = plan.group_info;
        fused->coords        = plan.coords;
        fused->dt_grid0      = plan.dt_grid0;
        fused->ndt0          = plan.ndt0;
        fused->coord_offset0 = plan.coord_offset0;
        const auto hist_len =
            static_cast<int>(m_plan.get_container().state_shape[0].dt_max);
        fused->args = detail::make_fused_tile_args(
            plan, tile, hist_len,
            thrust::raw_pointer_cast(fused->group_info.data()),
            thrust::raw_pointer_cast(fused->coords.data()),
            thrust::raw_pointer_cast(fused->dt_grid0.data()),
            thrust::raw_pointer_cast(fused->ndt0.data()),
            thrust::raw_pointer_cast(fused->coord_offset0.data()),
            box_means(0));
        const auto [cap_a, cap_b] = plan.smem_floats(tile);
        fused->smem_bytes         = (cap_a + cap_b) * sizeof(float);
        fused->tile               = tile;
        fused->ntiles             = (plan.ntiles_nsamps + tile - 1) / tile;
        fused->ngroups            = plan.ngroups;
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): fused plan upload failed");
        logging::debug("FDMT (cuda): fusing levels 0..{} (tile {} samples, {} "
                       "groups, {} B shared)",
                       plan.fuse, tile, plan.ngroups, fused->smem_bytes);
        m_fused = std::move(fused);
    }

    template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
    void
    launch_fused_typed(const Input& input, TOut* out, cudaStream_t stream) {
        const auto& f = *m_fused;
        if (f.smem_bytes > kDefaultSmemBytes) {
            cudaFuncSetAttribute(kernel_fused_levels<Mode, Smear, Input, TOut>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(f.smem_bytes));
        }
        const bool valid = Mode == FDMTMode::kValid;
        // Valid mode: this chunk's samples only.
        const int ntiles =
            valid ? static_cast<int>((m_active + f.tile - 1) / f.tile)
                  : f.ntiles;
        const dim3 grid_size(static_cast<unsigned>(ntiles),
                             static_cast<unsigned>(f.ngroups),
                             static_cast<unsigned>(m_nbeams));
        const dim3 block_size(kFusedThreads);
        gpu_utils::check_kernel_launch_params(grid_size, block_size);
        auto args   = f.args;
        args.box    = box_means_for(input);
        args.active = valid ? static_cast<int>(m_active) : 0;
        kernel_fused_levels<Mode, Smear, Input, TOut>
            <<<grid_size, block_size, f.smem_bytes, stream>>>(
                input, out, args,
                static_cast<int64_t>(level_beam_stride(f.args.fuse)),
                valid ? m_level0_hist_in_ptr : nullptr,
                valid ? m_fifo_in_ptr : nullptr,
                valid ? m_fifo_out_ptr : nullptr,
                static_cast<int64_t>(m_fifo_size));
        gpu_utils::check_last_gpu_error("kernel_fused_levels launch failed");
    }

    // Levels 0..F in one kernel, straight into level F's buffer (stored as
    // that level's type), then the level-0 input history as usual.
    template <typename Input>
    void launch_fused(const Input& input, cudaStream_t stream) {
        const SizeType fuse = get_fuse_levels();
        with_elem(m_levels_d[fuse].type, [&](auto tag) {
            using TOut        = decltype(tag);
            auto* out         = reinterpret_cast<TOut*>(m_levels_d[fuse].base);
            const auto launch = [&](auto mode_tag, auto smear_tag) {
                launch_fused_typed<decltype(mode_tag)::value,
                                   decltype(smear_tag)::value, Input, TOut>(
                    input, out, stream);
            };
            using Full  = std::integral_constant<FDMTMode, FDMTMode::kFull>;
            using Roll  = std::integral_constant<FDMTMode, FDMTMode::kRoll>;
            using Valid = std::integral_constant<FDMTMode, FDMTMode::kValid>;
            if (m_mode == FDMTMode::kFull) {
                m_use_box_smearing ? launch(Full{}, std::true_type{})
                                   : launch(Full{}, std::false_type{});
            } else if (m_mode == FDMTMode::kRoll) {
                m_use_box_smearing ? launch(Roll{}, std::true_type{})
                                   : launch(Roll{}, std::false_type{});
            } else {
                m_use_box_smearing ? launch(Valid{}, std::true_type{})
                                   : launch(Valid{}, std::false_type{});
            }
        });
        advance_level0_history(input, stream);
    }

    /**
     * Assigns each level's device buffer -- the same rule as the CPU FDMT:
     * level l < niters in state half l % 2 (a merge never reads and writes
     * the same half; integer levels use the halves' leading bytes), the root
     * in the caller's dmt buffer at `dmt` (the chunk's first sample).
     */
    void layout_levels(const std::vector<Elem>& types, float* dmt) {
        const SizeType niters = m_plan.get_niters();
        auto* internal        = reinterpret_cast<std::byte*>(
            thrust::raw_pointer_cast(m_state_internal_d.data()));
        const SizeType half = m_nbeams * m_half_elems * sizeof(float);
        for (SizeType l = 0; l < niters; ++l) {
            m_levels_d[l] = {.base = internal + ((l % 2) * half),
                             .type = types[l]};
        }
        m_levels_d[niters] = {.base = reinterpret_cast<std::byte*>(dmt),
                              .type = Elem::kF32};
    }

    // Level 0 of one chunk of `active` samples (the whole block for the
    // stepper), its root going to `dmt`.
    template <typename Input>
    void start(const Input& input,
               const std::vector<Elem>& types,
               float* dmt,
               SizeType active,
               cudaStream_t stream,
               bool fuse) {
        gpu_utils::set_device(m_device_id);
        // In "valid" mode every block must reach the root before the next
        // one starts: levels a stepper run never executed would skip this
        // block's history update, and the next block would read the
        // two-blocks-old ping-pong buffer.
        if (m_mode == FDMTMode::kValid && m_is_initialized && !is_finished()) {
            throw std::logic_error(fmt::format(
                "FDMT (cuda): the previous block's stepper run stopped at "
                "level "
                "{} of {}. In mode='valid' every block must be finalized "
                "before the next reset()/execute(), or its cross-block "
                "history is lost. Call finalize(), or reset_history() to "
                "abandon the stream.",
                m_current_level, total_levels() - 1));
        }
        m_stream           = stream;
        m_dmt_target_ptr   = dmt;
        m_active           = active;
        m_current_level    = 0;
        m_host_dmt         = {};
        m_host_level_valid = false;

        const bool direct = fuse && use_direct<Input>(types);
        m_top_chunk       = fuse && use_top(types);
        if (m_mode == FDMTMode::kValid) {
            if ((!direct && m_direct) || (!m_top_chunk && m_top)) {
                ensure_full_tree_history(stream);
            }
            // Tree levels after direct chunks lag behind: rebuild their
            // histories (at the current position) before the tree runs.
            if (!direct) {
                rebuild_direct_history(stream);
            }
            if (!m_top_chunk) {
                rebuild_top_history(stream);
            }
            // Ping-pong the level-0 history and the fused levels' FIFOs once
            // per chunk (not per level): all levels of one chunk must agree
            // on which buffer is "previous" vs "this". The ring stays put;
            // its slots follow the stream position.
            const int in = m_parity ? 1 : 0;
            if (m_direct && !m_direct->skewed) {
                m_direct_win_in =
                    thrust::raw_pointer_cast(m_direct->win[in].data());
                m_direct_win_out =
                    thrust::raw_pointer_cast(m_direct->win[1 - in].data());
            }
            m_fifo_in_ptr = thrust::raw_pointer_cast(m_tree_fifo_d[in].data());
            m_fifo_out_ptr =
                thrust::raw_pointer_cast(m_tree_fifo_d[1 - in].data());
            m_level0_hist_in_ptr = thrust::raw_pointer_cast(
                (m_parity ? m_history_b_d : m_history_a_d).data());
            m_level0_hist_out_ptr = thrust::raw_pointer_cast(
                (m_parity ? m_history_a_d : m_history_b_d).data());
            m_parity    = !m_parity;
            m_chunk_pos = m_next_pos;
            m_next_pos += active;
        }

        layout_levels(types, dmt);
        if (direct) {
            launch_direct(input, stream);
            if (!m_direct->skewed) {
                advance_level0_history(input, stream);
            }
            m_current_level = (m_skew_nbatches > 1 && m_top) ? m_top->level
                                                             : m_direct->levels;
            m_direct_stale  = !m_direct->skewed;
        } else if (fuse && m_fused) {
            launch_fused(input, stream);
            m_current_level = get_fuse_levels();
        } else {
            with_elem(m_levels_d[0].type, [&](auto tag0) {
                using T0 = decltype(tag0);
                initialise_device<Input, T0>(
                    input, reinterpret_cast<T0*>(m_levels_d[0].base), stream);
            });
        }
        if (m_direct && !m_direct->skewed) {
            advance_direct_window(input, stream);
        }
        m_is_initialized = true;
    }

    // Direct low levels (fdmt_direct.hpp) -----------------------------------

    /// Automatic fusion of a float valid-mode engine whose tile fusion stops
    /// short of kDirectLevels: compute level kDirectLevels from the input
    /// (replacing the tile fusion), if its shared-memory window fits.
    void configure_direct(SizeType requested) {
        m_direct.reset();
        m_direct_eligible     = false;
        const SizeType levels = kDirectLevels;
        // The packed search (exact integers) takes them too: its level-0
        // boxes use the plan's integer dyadic weights unrounded, so level K
        // is an exact integer sum of weighted input boxes (uint32_t running
        // sums, kernel_direct_level<uint32_t, TInt>).
        const bool packed = m_search_nbits != 32;
        if (packed) {
            requested = m_requested_fuse;
        }
        if (m_mode != FDMTMode::kValid || requested != kFDMTAutoFuse ||
            m_plan.get_niters() <= levels || m_search_max_fuse < levels ||
            get_fuse_levels() >= levels) {
            return;
        }
        m_direct_eligible = true;
        std::vector<float> box_w(m_box_w_d.size());
        if (packed) {
            // Integer weights (exact in float), so the leaves give the
            // integer search's level-0 values.
            const auto weights = m_plan.get_box_weights();
            for (SizeType i = 0; i < box_w.size(); ++i) {
                box_w[i] = static_cast<float>(weights[i]);
            }
        } else {
            thrust::copy(m_box_w_d.begin(), m_box_w_d.end(), box_w.begin());
        }
        auto plan = detail::build_fdmt_direct_plan(m_plan, levels,
                                                   m_use_box_smearing, box_w);
        int span  = 0;
        for (const auto& sub : plan.subs) {
            span = std::max(span, sub.span);
        }
        const auto per_channel =
            static_cast<SizeType>(span + detail::kDirectTile + 1) *
            sizeof(float);
        const SizeType optin = max_optin_smem_bytes();
        const auto rows      = m_nbeams * m_plan.get_nchans();
        if (per_channel > optin ||
            rows * static_cast<SizeType>(plan.window) * sizeof(float) >
                (SizeType{256} << 20)) {
            return;
        }
        if (packed) {
            // uint32_t running sums over the window and a chunk.
            const auto maxval = static_cast<double>(
                bit_pack_utils::max_sample_value(m_search_nbits));
            if (static_cast<double>(plan.window + m_chunk + 1) * maxval >
                4294967295.0) {
                return;
            }
        }
        auto d         = std::make_unique<DirectLevelsD>();
        d->levels      = levels;
        d->window      = plan.window;
        d->nsubs       = static_cast<int>(plan.subs.size());
        d->max_rows    = plan.max_rows;
        d->cpass       = static_cast<int>(std::min<SizeType>(
            static_cast<SizeType>(plan.max_nc), optin / per_channel));
        d->smem        = static_cast<SizeType>(d->cpass) * per_channel;
        d->subs        = plan.subs;
        d->leaf_off    = plan.leaf_off;
        d->leaves      = plan.leaves;
        d->hist_leaves = plan.hist_leaves;
        d->hist_coords.resize(levels + 1);
        d->hist_max_delay.assign(levels + 1, 0);
        for (SizeType l = 1; l <= levels; ++l) {
            d->hist_coords[l] = plan.hist_coords[l];
            for (const auto& hc : plan.hist_coords[l]) {
                d->hist_max_delay[l] = std::max(d->hist_max_delay[l], hc.delay);
            }
        }
        for (auto& w : d->win) {
            w.resize(rows * static_cast<SizeType>(d->window), 0.0F);
        }
        d->pitch = d->window + static_cast<int>(m_chunk) + 1;
        d->prefix.resize(rows * static_cast<SizeType>(d->pitch));
        const auto opt_in = [&](auto kernel) {
            gpu_utils::check_gpu_call(
                cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(d->smem)),
                "FDMT (cuda): direct levels shared-memory opt-in");
        };
        if (!packed) {
            opt_in(detail::kernel_direct_level<>);
        } else if (m_search_u32) {
            opt_in(detail::kernel_direct_level<uint32_t, uint32_t>);
        } else {
            opt_in(detail::kernel_direct_level<uint32_t, uint64_t>);
        }
        logging::debug("FDMT (cuda): levels 0..{} direct ({} subbands, {} "
                       "leaves, window {}, {} channels per pass, {} B shared)",
                       levels, d->nsubs, plan.leaves.size(), d->window,
                       d->cpass, d->smem);
        m_direct_filled = static_cast<SizeType>(d->window);
        m_direct        = std::move(d);
        m_fused.reset();
    }

    /// Whether this chunk computes the direct levels: configured, float
    /// levels, and an input window covering every row's reach.
    template <typename Input>
    [[nodiscard]] bool use_direct(const std::vector<Elem>& types) const {
        return m_direct && types[m_direct->levels] == Elem::kF32 &&
               (m_direct->skewed ||
                (std::is_same_v<Input, FDMTInputF32> &&
                 m_direct_filled >= static_cast<SizeType>(m_direct->window)));
    }

    template <typename T, typename PtrFn>
    void
    launch_skewed_batch_mid(SizeType b, PtrFn&& lvl_ptr, cudaStream_t stream) {
        const SizeType k     = m_direct->levels;
        const SizeType top_k = m_top->level;
        for (SizeType l = k + 1; l <= top_k; ++l) {
            const int sum_b0         = m_skew_batch_sum_off[l][b];
            const int sum_b1         = m_skew_batch_sum_off[l][b + 1];
            const int copy_b0        = m_skew_batch_copy_off[l][b];
            const int copy_b1        = m_skew_batch_copy_off[l][b + 1];
            const int ncoords_sum_b  = sum_b1 - sum_b0;
            const int ncoords_copy_b = copy_b1 - copy_b0;
            const int coords_max     = std::max(ncoords_sum_b, ncoords_copy_b);
            if (coords_max <= 0) {
                continue;
            }
            auto sums   = m_plan_d.coordinates_sum.get_raw_ptrs();
            auto copies = m_plan_d.coordinates_copy.get_raw_ptrs();
            sums.update_offsets(m_coords_sum_offsets[l] + sum_b0);
            copies.update_offsets(m_coords_copy_offsets[l] + copy_b0);
            const int nsamps_l = static_cast<int>(
                m_active +
                (l < m_level_max_extra.size() ? m_level_max_extra[l] : 0));
            constexpr int kBlockSamps = kIterThreads * kIterSampsPerTh;
            const dim3 block(kIterThreads, 1);
            const dim3 grid((nsamps_l + kBlockSamps - 1) / kBlockSamps,
                            std::min(coords_max, kMaxGridY), m_nbeams);
            gpu_utils::check_kernel_launch_params(grid, block);
            kernel_execute_iter<FDMTMode::kValid, T, T, T>
                <<<grid, block, 0, stream>>>(
                    lvl_ptr(l - 1), lvl_ptr(l), sums, copies, nullptr, nsamps_l,
                    ncoords_sum_b, ncoords_copy_b, level_beam_stride(l - 1),
                    level_beam_stride(l), 0, 0, m_chunk_pos, 1.0F,
                    static_cast<int>(m_active), false,
                    static_cast<int>(m_chunk));
        }
        if (m_top && m_top->target_ring) {
            auto& t              = *m_top;
            constexpr int kWarps = 8;
            const dim3 grid(t.ntrials, 1, m_nbeams);
            gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
            auto* rings =
                reinterpret_cast<T*>(thrust::raw_pointer_cast(t.ring.data()));
            detail::kernel_direct_top_target<T>
                <<<grid, kWarps * 32, 0, stream>>>(
                    lvl_ptr(top_k), static_cast<int64_t>(m_half_elems),
                    static_cast<int>(m_chunk), rings, t.ring_size,
                    thrust::raw_pointer_cast(t.ring_off.data()),
                    thrust::raw_pointer_cast(t.ring_len.data()),
                    thrust::raw_pointer_cast(t.leaf_off.data()),
                    thrust::raw_pointer_cast(t.leaves.data()),
                    thrust::raw_pointer_cast(t.out_row.data()), t.ntrials,
                    nullptr, 0, 0, m_chunk_pos, static_cast<int>(m_active),
                    m_skew_batch_top_grid[b], m_skew_batch_top_grid[b + 1],
                    false);
        }
    }

    template <typename Input>
    void launch_direct(const Input& input, cudaStream_t stream) {
        auto& d          = *m_direct;
        const int nchans = static_cast<int>(m_plan.get_nchans());
        const int active = static_cast<int>(m_active);
        if (d.skewed) {
            auto* wring = thrust::raw_pointer_cast(d.win_ring.data());
            const dim3 wgrid((active + 255) / 256, nchans, m_nbeams);
            detail::kernel_direct_write_ring<Input, float>
                <<<wgrid, 256, 0, stream>>>(
                    wring, d.win_ring_size,
                    thrust::raw_pointer_cast(d.chan_ring_off.data()),
                    thrust::raw_pointer_cast(d.chan_ring_len.data()), input,
                    nchans, active, m_chunk_pos);
            const int ntiles =
                (active + d.max_extra + detail::kDirectTile - 1) /
                detail::kDirectTile;
            const auto lvl_ptr = [&](SizeType l) {
                return reinterpret_cast<float*>(m_levels_d[l].base);
            };
            for (SizeType b = 0; b < m_skew_nbatches; ++b) {
                const int32_t sub0    = d.batch_sub_off[b];
                const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
                if (nsubs_b > 0) {
                    const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                    gpu_utils::check_kernel_launch_params(
                        lgrid, dim3(detail::kDirectWarps * 32));
                    const auto args = std::make_tuple(
                        static_cast<const float*>(wring), d.win_ring_size,
                        thrust::raw_pointer_cast(d.chan_ring_off.data()),
                        thrust::raw_pointer_cast(d.chan_ring_len.data()),
                        thrust::raw_pointer_cast(d.subs.data()) + sub0,
                        thrust::raw_pointer_cast(d.leaf_off.data()),
                        thrust::raw_pointer_cast(d.leaves.data()),
                        lvl_ptr(d.levels), static_cast<int>(m_chunk),
                        level_beam_stride(d.levels), d.cpass, m_chunk_pos,
                        active);
                    if constexpr (detail::kFDMTPackedInput<Input>) {
                        // Packed input keeps the tree's rounded box means
                        // (exact integer sums, see FDMTDirectRound).
                        const auto bm = box_means_for(input);
                        const detail::FDMTDirectRound rnd{
                            .round = m_use_box_smearing ? bm.round : nullptr,
                            .round_u64 = bm.round_u64,
                            .unit      = m_use_box_smearing ? m_int_unit : 1.0F,
                        };
                        std::apply(
                            [&](auto... a) {
                                detail::kernel_direct_level_skewed<
                                    float, uint32_t, uint32_t, true, float>
                                    <<<lgrid, detail::kDirectWarps * 32, d.smem,
                                       stream>>>(a..., rnd);
                            },
                            args);
                    } else {
                        std::apply(
                            [&](auto... a) {
                                detail::kernel_direct_level_skewed<float, float,
                                                                   float>
                                    <<<lgrid, detail::kDirectWarps * 32, d.smem,
                                       stream>>>(a...);
                            },
                            args);
                    }
                }
                if (m_skew_nbatches > 1 && m_top) {
                    launch_skewed_batch_mid<float>(b, lvl_ptr, stream);
                }
            }
            gpu_utils::check_last_gpu_error(
                "FDMT (cuda): skewed direct levels failed");
            return;
        }
        detail::kernel_direct_prefix<Input>
            <<<dim3(nchans, 1, m_nbeams), detail::kPrefixThreads, 0, stream>>>(
                input, m_direct_win_in, d.window, nchans, active,
                thrust::raw_pointer_cast(d.prefix.data()), d.pitch);
        constexpr int kRows = detail::kDirectRowsPerWarp * detail::kDirectWarps;
        const int ntiles =
            (active + detail::kDirectTile - 1) / detail::kDirectTile;
        const dim3 grid((d.max_rows + kRows - 1) / kRows, d.nsubs,
                        m_nbeams * ntiles);
        gpu_utils::check_kernel_launch_params(grid,
                                              dim3(detail::kDirectWarps * 32));
        detail::kernel_direct_level<>
            <<<grid, detail::kDirectWarps * 32, d.smem, stream>>>(
                thrust::raw_pointer_cast(d.prefix.data()), d.pitch, d.window,
                nchans, thrust::raw_pointer_cast(d.subs.data()),
                thrust::raw_pointer_cast(d.leaf_off.data()),
                thrust::raw_pointer_cast(d.leaves.data()),
                reinterpret_cast<float*>(m_levels_d[d.levels].base),
                static_cast<int>(m_chunk), level_beam_stride(d.levels), d.cpass,
                active, ntiles);
        gpu_utils::check_last_gpu_error("FDMT (cuda): direct levels failed");
    }

    /// Keeps the direct levels' input window (the last `window` samples of
    /// every channel) in step with the stream, whichever path ran.
    template <typename Input>
    void advance_direct_window(const Input& input, cudaStream_t stream) {
        if (!m_direct || m_direct->skewed || m_mode != FDMTMode::kValid) {
            return;
        }
        const auto& d   = *m_direct;
        const int nsubs = static_cast<int>(m_plan.get_nchans());
        const dim3 block(256);
        const dim3 grid((d.window + 255) / 256, nsubs, m_nbeams);
        kernel_advance_history_window<Input><<<grid, block, 0, stream>>>(
            m_direct_win_out, m_direct_win_in, input, nsubs,
            static_cast<int>(m_active), d.window);
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): direct window update failed");
        m_direct_filled = std::min<SizeType>(static_cast<SizeType>(d.window),
                                             m_direct_filled + m_active);
    }

    /// Rewrites the tree histories of levels 1..K (rings) from the input
    /// window at the current stream position, after direct chunks left them
    /// behind (fdmt_direct.hpp).
    void rebuild_direct_history(cudaStream_t stream) const {
        if (!m_direct || !m_direct_stale || m_direct->skewed) {
            return;
        }
        const auto& d = *m_direct;
        const auto* win =
            thrust::raw_pointer_cast(m_direct->win[m_parity ? 1 : 0].data());
        auto* ring = const_cast<float*>( // NOLINT: device buffer, host const
            thrust::raw_pointer_cast(m_tree_ring_d.data()));
        for (SizeType l = 1; l <= d.levels; ++l) {
            const int n = static_cast<int>(d.hist_coords[l].size());
            if (n == 0) {
                continue;
            }
            const dim3 grid((d.hist_max_delay[l] + 255) / 256,
                            std::min(n, kMaxGridY), m_nbeams);
            detail::kernel_direct_rebuild_history<<<grid, 256, 0, stream>>>(
                win, d.window, static_cast<int>(m_plan.get_nchans()),
                thrust::raw_pointer_cast(d.hist_coords[l].data()), n,
                thrust::raw_pointer_cast(d.hist_leaves.data()), ring,
                static_cast<int64_t>(m_ring_size),
                static_cast<int64_t>(m_fifo_size), m_next_pos);
        }
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): direct history rebuild failed");
        m_direct_stale = false;
    }

    void build_skewed_direct_from_windows(
        const std::vector<int32_t>& s_hi_k,
        const std::vector<int32_t>& sp_k,
        const std::vector<SizeType>& batch_grid_off = {}) {
        const SizeType k  = kDirectLevels;
        const bool packed = m_search_nbits != 32;
        std::vector<float> box_w(m_box_w_d.size());
        if (packed) {
            const auto weights = m_plan.get_box_weights();
            for (SizeType i = 0; i < box_w.size(); ++i) {
                box_w[i] = static_cast<float>(weights[i]);
            }
        } else {
            thrust::copy(m_box_w_d.begin(), m_box_w_d.end(), box_w.begin());
        }
        auto dplan = detail::build_fdmt_direct_skewed_plan(
            m_plan, k, m_use_box_smearing, box_w, m_chunk, s_hi_k, sp_k,
            batch_grid_off);
        const auto per_channel =
            static_cast<SizeType>(dplan.max_span + detail::kDirectTile + 1) *
            sizeof(float);
        const SizeType optin = max_optin_smem_bytes();
        if (per_channel > optin) {
            return;
        }
        auto d           = std::make_unique<DirectLevelsD>();
        d->levels        = k;
        d->skewed        = true;
        d->window        = dplan.window;
        d->nsubs         = static_cast<int>(dplan.subs.size());
        d->max_rows      = dplan.max_rows;
        d->max_extra     = dplan.max_extra;
        d->total_elems   = static_cast<SizeType>(dplan.total_elems);
        d->win_ring_size = dplan.win_ring_size;
        d->row_off       = std::move(dplan.row_off);
        d->batch_sub_off = std::move(dplan.batch_sub_off);
        d->cpass         = static_cast<int>(std::min<SizeType>(
            static_cast<SizeType>(dplan.max_nc), optin / per_channel));
        d->smem          = static_cast<SizeType>(d->cpass) * per_channel;
        d->subs          = dplan.subs;
        d->leaf_off      = dplan.leaf_off;
        d->leaves        = dplan.leaves;
        d->chan_ring_len = dplan.chan_ring_len;
        d->chan_ring_off = dplan.chan_ring_off;
        const auto total_ring =
            m_nbeams * static_cast<SizeType>(d->win_ring_size);
        if (!packed) {
            d->win_ring.resize(total_ring, 0.0F);
        } else if (m_search_nbits <= 8) {
            d->win_ring_u8.resize(total_ring, 0);
        } else {
            d->win_ring_u16.resize(total_ring, 0);
        }
        const auto opt_in = [&](auto kernel) {
            gpu_utils::check_gpu_call(
                cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(d->smem)),
                "FDMT (cuda): skewed direct levels shared-memory opt-in");
        };
        if (!packed) {
            opt_in(detail::kernel_direct_level_skewed<float, float, float>);
        } else if (m_search_nbits <= 8 && m_search_u32) {
            opt_in(detail::kernel_direct_level_skewed<uint8_t, uint32_t,
                                                      uint32_t>);
        } else if (m_search_nbits <= 8) {
            opt_in(detail::kernel_direct_level_skewed<uint8_t, uint32_t,
                                                      uint64_t>);
        } else if (m_search_u32) {
            opt_in(detail::kernel_direct_level_skewed<uint16_t, uint32_t,
                                                      uint32_t>);
        } else {
            opt_in(detail::kernel_direct_level_skewed<uint16_t, uint32_t,
                                                      uint64_t>);
        }
        logging::debug(
            "FDMT (cuda): levels 0..{} skewed direct ({} slices, {} leaves, "
            "win_ring {} MiB, max_span {}, max_extra {}, {} B shared)",
            k, d->nsubs, dplan.leaves.size(),
            (total_ring * (packed ? (m_search_nbits <= 8 ? 1 : 2) : 4)) >> 20,
            dplan.max_span, d->max_extra, d->smem);
        m_direct_filled = static_cast<SizeType>(d->window);
        m_direct        = std::move(d);
        m_fused.reset();
    }

    void configure_skewed_direct_and_mid(plans::FDMTPlanContainer& relaid,
                                         detail::FDMTDirectTopPlan& top_plan) {
        if (relaid.state_shape.empty()) {
            relaid = detail::fdmt_relayout_container(
                m_plan.get_container(), m_plan.get_niters(), m_chunk);
        }
        const auto& pc       = m_plan.get_container();
        const SizeType k     = kDirectLevels;
        const SizeType top_k = top_plan.level;
        std::vector<std::vector<int32_t>> s_lo(top_k + 1);
        std::vector<std::vector<int32_t>> s_hi(top_k + 1);
        for (SizeType l = k; l <= top_k; ++l) {
            s_lo[l].assign(pc.state_shape[l].ncoords, INT32_MAX);
            s_hi[l].assign(pc.state_shape[l].ncoords, -1);
        }
        for (SizeType r = 0; r < pc.state_shape[top_k].ncoords; ++r) {
            if (top_plan.row_s_hi[r] >= 0) {
                s_lo[top_k][r] = top_plan.row_s_lo[r];
                s_hi[top_k][r] = top_plan.row_s_lo[r];
            }
        }
        for (SizeType l = top_k; l > k; --l) {
            const auto in_stride  = pc.state_shape[l - 1].nsamps;
            const auto out_stride = pc.state_shape[l].nsamps;
            for (const auto& c : pc.coordinates_copy[l]) {
                const auto p = c.buf_offset / out_stride;
                const auto t = c.tail_buf_offset / in_stride;
                if (s_hi[l][p] < 0) {
                    continue;
                }
                s_lo[l - 1][t] = std::min(s_lo[l - 1][t], s_lo[l][p]);
                s_hi[l - 1][t] = std::max(s_hi[l - 1][t], s_hi[l][p]);
            }
            for (const auto& c : pc.coordinates_sum[l]) {
                const auto p = c.buf_offset / out_stride;
                const auto t = c.tail_buf_offset / in_stride;
                const auto h = c.head_buf_offset / in_stride;
                const auto d = static_cast<int32_t>(c.delay);
                if (s_hi[l][p] < 0) {
                    continue;
                }
                s_lo[l - 1][t] = std::min(s_lo[l - 1][t], s_lo[l][p]);
                s_hi[l - 1][t] = std::max(s_hi[l - 1][t], s_hi[l][p]);
                s_lo[l - 1][h] = std::min(s_lo[l - 1][h], s_lo[l][p] + d);
                s_hi[l - 1][h] = std::max(s_hi[l - 1][h], s_hi[l][p] + d);
            }
        }
        m_level_max_extra.assign(m_plan.get_niters() + 1, 0);
        std::vector<std::vector<int32_t>> sp(top_k + 1);
        for (SizeType l = k; l <= top_k; ++l) {
            sp[l].assign(pc.state_shape[l].ncoords, 0);
            for (SizeType r = 0; r < pc.state_shape[l].ncoords; ++r) {
                if (s_hi[l][r] >= 0) {
                    sp[l][r]             = s_hi[l][r] - s_lo[l][r];
                    m_level_max_extra[l] = std::max(
                        m_level_max_extra[l], static_cast<SizeType>(sp[l][r]));
                }
            }
        }
        const SizeType ngrids_top = pc.grids[top_k].size();
        for (SizeType f = 0; f < top_plan.out_row.size(); ++f) {
            std::ranges::stable_sort(
                top_plan.leaves.begin() + top_plan.leaf_off[f],
                top_plan.leaves.begin() + top_plan.leaf_off[f + 1], {},
                [](const detail::FDMTDirectTopLeaf& lf) { return lf.row; });
        }
        SizeType nbatches =
            (!m_search_upper && top_plan.target_ring && ngrids_top >= 24) ? 24
                                                                          : 1;
        std::vector<std::vector<SizeType>> batch_grid(top_k + 1);
        std::vector<std::vector<SizeType>> batch_coord(top_k + 1);
        const auto build_batch_bounds = [&](SizeType nb) {
            std::vector<SizeType> top_bounds(nb + 1, 0);
            top_bounds[nb] = ngrids_top;
            if (nb > 1) {
                const SizeType ngrids_k = pc.grids[k].size();
                std::vector<SizeType> cum_w(ngrids_top + 1, 0);
                for (SizeType g = 0; g < ngrids_top; ++g) {
                    const SizeType gk0 = std::min(ngrids_k, g << (top_k - k));
                    const SizeType gk1 =
                        std::min(ngrids_k, (g + 1) << (top_k - k));
                    const SizeType c0 = gk0 < ngrids_k
                                            ? pc.grids[k][gk0].coord_offset
                                            : pc.state_shape[k].ncoords;
                    const SizeType c1 = gk1 < ngrids_k
                                            ? pc.grids[k][gk1].coord_offset
                                            : pc.state_shape[k].ncoords;
                    SizeType w        = 1;
                    for (SizeType r = c0; r < c1; ++r) {
                        if (s_hi[k][r] >= 0) {
                            w += m_chunk + static_cast<SizeType>(sp[k][r]);
                        }
                    }
                    cum_w[g + 1] = cum_w[g] + w;
                }
                const SizeType total_w = cum_w[ngrids_top];
                for (SizeType b = 1; b < nb; ++b) {
                    const SizeType target_w = (b * total_w) / nb;
                    SizeType g              = top_bounds[b - 1] + 1;
                    const SizeType max_g    = ngrids_top - (nb - b);
                    while (g < max_g && cum_w[g] < target_w) {
                        ++g;
                    }
                    top_bounds[b] = g;
                }
            }
            for (SizeType l = k; l <= top_k; ++l) {
                batch_grid[l].resize(nb + 1, 0);
                batch_coord[l].resize(nb + 1, 0);
                const SizeType ngrids_l = pc.grids[l].size();
                for (SizeType b = 0; b <= nb; ++b) {
                    const SizeType g_top = top_bounds[b];
                    const SizeType g_l =
                        (b == nb) ? ngrids_l
                                  : std::min(ngrids_l, g_top << (top_k - l));
                    batch_grid[l][b]  = g_l;
                    batch_coord[l][b] = (g_l < ngrids_l)
                                            ? pc.grids[l][g_l].coord_offset
                                            : pc.state_shape[l].ncoords;
                }
            }
        };
        build_batch_bounds(nbatches);
        if (nbatches > 1) {
            bool valid_batches = true;
            for (SizeType f = 0; f < top_plan.out_row.size() && valid_batches;
                 ++f) {
                const auto f0 = static_cast<SizeType>(top_plan.leaf_off[f]);
                const auto f1 = static_cast<SizeType>(top_plan.leaf_off[f + 1]);
                if (f1 - f0 != ngrids_top) {
                    valid_batches = false;
                    break;
                }
                for (SizeType b = 0; b < nbatches && valid_batches; ++b) {
                    const auto c_lo = batch_coord[top_k][b];
                    const auto c_hi = batch_coord[top_k][b + 1];
                    for (SizeType m = batch_grid[top_k][b];
                         m < batch_grid[top_k][b + 1]; ++m) {
                        const auto r =
                            static_cast<SizeType>(top_plan.leaves[f0 + m].row);
                        if (r < c_lo || r >= c_hi) {
                            valid_batches = false;
                            break;
                        }
                    }
                }
            }
            for (SizeType l = k + 1; l <= top_k && valid_batches; ++l) {
                const SizeType orig_in  = pc.state_shape[l - 1].nsamps;
                const SizeType orig_out = pc.state_shape[l].nsamps;
                SizeType b_cur          = 0;
                for (const auto& o : pc.coordinates_copy[l]) {
                    const auto p = o.buf_offset / orig_out;
                    const auto t = o.tail_buf_offset / orig_in;
                    if (p < batch_coord[l][b_cur]) {
                        valid_batches = false;
                        break;
                    }
                    while (b_cur < nbatches && p >= batch_coord[l][b_cur + 1]) {
                        ++b_cur;
                    }
                    if (b_cur >= nbatches || t < batch_coord[l - 1][b_cur] ||
                        t >= batch_coord[l - 1][b_cur + 1]) {
                        valid_batches = false;
                        break;
                    }
                }
                b_cur = 0;
                for (const auto& o : pc.coordinates_sum[l]) {
                    const auto p = o.buf_offset / orig_out;
                    const auto t = o.tail_buf_offset / orig_in;
                    const auto h = o.head_buf_offset / orig_in;
                    if (p < batch_coord[l][b_cur]) {
                        valid_batches = false;
                        break;
                    }
                    while (b_cur < nbatches && p >= batch_coord[l][b_cur + 1]) {
                        ++b_cur;
                    }
                    if (b_cur >= nbatches || t < batch_coord[l - 1][b_cur] ||
                        t >= batch_coord[l - 1][b_cur + 1] ||
                        h < batch_coord[l - 1][b_cur] ||
                        h >= batch_coord[l - 1][b_cur + 1]) {
                        valid_batches = false;
                        break;
                    }
                }
            }
            if (!valid_batches) {
                nbatches = 1;
                build_batch_bounds(1);
            }
        }
        build_skewed_direct_from_windows(s_hi[k], sp[k], batch_grid[k]);
        if (!m_direct || !m_direct->skewed) {
            m_level_max_extra.clear();
            return;
        }
        m_skew_nbatches = nbatches;
        m_skew_batch_top_grid.resize(nbatches + 1, 0);
        for (SizeType b = 0; b <= nbatches; ++b) {
            m_skew_batch_top_grid[b] = static_cast<int>(batch_grid[top_k][b]);
        }
        m_skew_batch_sum_off.assign(top_k + 1, {});
        m_skew_batch_copy_off.assign(top_k + 1, {});
        relaid.state_shape[k].nsamps    = m_chunk + m_level_max_extra[k];
        relaid.state_shape[k].nelements = m_direct->total_elems;
        std::vector<std::vector<SizeType>> row_off(top_k + 1);
        row_off[k].resize(m_direct->row_off.size());
        for (SizeType r = 0; r < m_direct->row_off.size(); ++r) {
            row_off[k][r] = static_cast<SizeType>(m_direct->row_off[r]);
        }
        for (SizeType l = k + 1; l <= top_k; ++l) {
            const SizeType orig_in  = pc.state_shape[l - 1].nsamps;
            const SizeType orig_out = pc.state_shape[l].nsamps;
            const SizeType ncoords  = pc.state_shape[l].ncoords;
            row_off[l].resize(ncoords, 0);
            if (l == top_k) {
                SizeType max_used_rows = 0;
                std::vector<int32_t> remap(ncoords, -1);
                for (SizeType b = 0; b < nbatches; ++b) {
                    SizeType used_rows = 0;
                    for (SizeType p = batch_coord[l][b];
                         p < batch_coord[l][b + 1]; ++p) {
                        row_off[l][p] = used_rows * m_chunk;
                        if (s_hi[l][p] >= 0) {
                            remap[p] = static_cast<int32_t>(used_rows);
                            ++used_rows;
                        }
                    }
                    max_used_rows = std::max(max_used_rows, used_rows);
                }
                for (auto& lf : top_plan.leaves) {
                    lf.row = remap[static_cast<SizeType>(lf.row)];
                }
                if (!top_plan.target_ring) {
                    std::vector<int32_t> new_ring_len(max_used_rows, 0);
                    std::vector<int64_t> new_ring_off(max_used_rows, 0);
                    for (SizeType p = 0; p < ncoords; ++p) {
                        if (remap[p] >= 0) {
                            const auto u    = static_cast<SizeType>(remap[p]);
                            new_ring_len[u] = top_plan.ring_len[p];
                            new_ring_off[u] = top_plan.ring_off[p];
                        }
                    }
                    top_plan.ring_len = std::move(new_ring_len);
                    top_plan.ring_off = std::move(new_ring_off);
                }
                relaid.state_shape[l].nsamps    = m_chunk;
                relaid.state_shape[l].nelements = max_used_rows * m_chunk;
            } else {
                SizeType max_off = 0;
                for (SizeType b = 0; b < nbatches; ++b) {
                    SizeType off = 0;
                    for (SizeType p = batch_coord[l][b];
                         p < batch_coord[l][b + 1]; ++p) {
                        row_off[l][p] = off;
                        if (s_hi[l][p] >= 0) {
                            off += m_chunk + static_cast<SizeType>(sp[l][p]);
                        }
                    }
                    max_off = std::max(max_off, off);
                }
                relaid.state_shape[l].nsamps = m_chunk + m_level_max_extra[l];
                relaid.state_shape[l].nelements = max_off;
            }
            m_skew_batch_copy_off[l].resize(nbatches + 1, 0);
            m_skew_batch_sum_off[l].resize(nbatches + 1, 0);
            SizeType i_copy = 0;
            SizeType i_sum  = 0;
            for (SizeType b = 0; b <= nbatches; ++b) {
                const SizeType c_lim = batch_coord[l][b];
                while (i_copy < pc.coordinates_copy[l].size() &&
                       (pc.coordinates_copy[l][i_copy].buf_offset / orig_out) <
                           c_lim) {
                    ++i_copy;
                }
                m_skew_batch_copy_off[l][b] = static_cast<int>(i_copy);
                while (i_sum < pc.coordinates_sum[l].size() &&
                       (pc.coordinates_sum[l][i_sum].buf_offset / orig_out) <
                           c_lim) {
                    ++i_sum;
                }
                m_skew_batch_sum_off[l][b] = static_cast<int>(i_sum);
            }
            relaid.state_shape[l].dt_max = 0;
            for (SizeType i = 0; i < relaid.coordinates_copy[l].size(); ++i) {
                auto& c         = relaid.coordinates_copy[l][i];
                const auto& o   = pc.coordinates_copy[l][i];
                const auto p    = o.buf_offset / orig_out;
                const auto t    = o.tail_buf_offset / orig_in;
                const bool used = s_hi[l][p] >= 0;
                const SizeType extra_p =
                    used ? static_cast<SizeType>(sp[l][p]) : 0;
                const SizeType shift_t =
                    (used && s_hi[l - 1][t] >= s_hi[l][p])
                        ? static_cast<SizeType>(s_hi[l - 1][t] - s_hi[l][p])
                        : 0;
                c.nsamps          = used ? (m_chunk + extra_p) : 0;
                c.buf_offset      = row_off[l][p];
                c.tail_buf_offset = row_off[l - 1][t] + shift_t;
                c.tail_nsamps     = c.nsamps;
            }
            for (SizeType i = 0; i < relaid.coordinates_sum[l].size(); ++i) {
                auto& c         = relaid.coordinates_sum[l][i];
                const auto& o   = pc.coordinates_sum[l][i];
                const auto p    = o.buf_offset / orig_out;
                const auto t    = o.tail_buf_offset / orig_in;
                const auto h    = o.head_buf_offset / orig_in;
                const auto d    = static_cast<int32_t>(o.delay);
                const bool used = s_hi[l][p] >= 0;
                const SizeType extra_p =
                    used ? static_cast<SizeType>(sp[l][p]) : 0;
                const SizeType shift_t =
                    (used && s_hi[l - 1][t] >= s_hi[l][p])
                        ? static_cast<SizeType>(s_hi[l - 1][t] - s_hi[l][p])
                        : 0;
                const SizeType shift_h =
                    (used && s_hi[l - 1][h] >= s_hi[l][p] + d)
                        ? static_cast<SizeType>(s_hi[l - 1][h] -
                                                (s_hi[l][p] + d))
                        : 0;
                c.nsamps          = used ? (m_chunk + extra_p) : 0;
                c.buf_offset      = row_off[l][p];
                c.tail_buf_offset = row_off[l - 1][t] + shift_t;
                c.tail_nsamps     = c.nsamps;
                c.head_buf_offset = row_off[l - 1][h] + shift_h;
                c.head_nsamps     = c.nsamps;
                c.delay           = 0;
            }
        }
    }

    /// Direct top above the direct low levels (fdmt_direct.hpp) when no
    /// search reads the levels in between and its rings fit the budget.
    void configure_top(plans::FDMTPlanContainer& relaid) {
        m_top.reset();
        if (!m_direct && !m_direct_eligible) {
            return;
        }
        const auto& pc           = m_plan.get_container();
        const SizeType root      = m_plan.get_niters();
        const SizeType sub_depth = m_plan.get_subband_depth();
        const SizeType k_direct  = m_direct ? m_direct->levels : kDirectLevels;
        const auto fallback_skewed_direct = [&]() {
            if (!m_direct && m_direct_eligible) {
                const std::vector<int32_t> zeros(
                    pc.state_shape[k_direct].ncoords, 0);
                build_skewed_direct_from_windows(zeros, zeros);
            }
        };
        if (m_search_upper && sub_depth + 1 >= root) {
            fallback_skewed_direct();
            return;
        }
        const SizeType target  = m_search_upper ? (root - sub_depth - 1) : root;
        const SizeType ntrials = detail::fdmt_regular_coords(pc, target);
        const SizeType target_subs =
            std::max<SizeType>(1, pc.state_shape[target].nchans);
        SizeType level = k_direct + 1;
        while (level + 2 < target) {
            const SizeType leaves_saved =
                (ntrials * (pc.state_shape[level].nchans -
                            pc.state_shape[level + 1].nchans)) /
                target_subs;
            if (leaves_saved < 8 * pc.state_shape[level + 1].ncoords) {
                break;
            }
            ++level;
        }
        // When writing into m_levels_d[target], level and target must occupy
        // opposite ping-pong state halves (level % 2 != target % 2).
        if (target < root && (target - level) % 2 == 0) {
            ++level;
        }
        if (level + 1 >= target) {
            fallback_skewed_direct();
            return;
        }
        const SizeType orig_level        = level;
        constexpr SizeType kMaxRingBytes = SizeType{2} << 30;
        const SizeType step              = target < root ? 2 : 1;
        detail::FDMTDirectTopPlan plan;
        bool use_skewed = !m_direct || m_force_skewed;
        if (!use_skewed) {
            plan = detail::build_fdmt_direct_top(m_plan, level, target, false);
            while (m_nbeams * static_cast<SizeType>(plan.ring_size) *
                       sizeof(float) * top_words() >
                   kMaxRingBytes) {
                if (level + step + 1 >= target) {
                    use_skewed = true;
                    break;
                }
                level += step;
                plan =
                    detail::build_fdmt_direct_top(m_plan, level, target, false);
            }
        }
        if (use_skewed) {
            level = std::max<SizeType>(
                orig_level,
                k_direct + (pc.state_shape[0].nchans > 4096 ? 3 : 2));
            if (target < root && (target - level) % 2 == 0) {
                ++level;
            }
            if (level + 1 >= target) {
                level = orig_level;
            }
            plan = detail::build_fdmt_direct_top(m_plan, level, target, true,
                                                 m_chunk, true);
            if (m_nbeams * static_cast<SizeType>(plan.ring_size) *
                    sizeof(float) * top_words() >
                kMaxRingBytes) {
                plan =
                    detail::build_fdmt_direct_top(m_plan, level, target, true);
                while (m_nbeams * static_cast<SizeType>(plan.ring_size) *
                           sizeof(float) * top_words() >
                       kMaxRingBytes) {
                    if (level + step + 1 >= target) {
                        logging::debug(
                            "FDMT (cuda): skewed direct top rings over budget "
                            "at every level; top disabled");
                        fallback_skewed_direct();
                        return;
                    }
                    level += step;
                    plan = detail::build_fdmt_direct_top(m_plan, level, target,
                                                         true);
                }
            }
            configure_skewed_direct_and_mid(relaid, plan);
            if (!m_direct || !m_direct->skewed) {
                return;
            }
        }
        auto t         = std::make_unique<DirectTopD>();
        t->level       = level;
        t->target      = target;
        t->skewed      = plan.skewed;
        t->target_ring = plan.target_ring;
        t->ntrials     = static_cast<int>(plan.out_row.size());
        t->nrows       = static_cast<int>(plan.ring_len.size());
        t->max_ring    = plan.max_ring;
        t->ring_size   = plan.ring_size;
        t->out_row     = plan.out_row;
        t->leaf_off    = plan.leaf_off;
        t->leaves      = plan.leaves;
        t->ring_len    = plan.ring_len;
        t->ring_off    = plan.ring_off;
        t->hist_leaves = plan.hist_leaves;
        // uint64_t rings (packed search) take two floats' storage each.
        t->ring.resize(m_nbeams * static_cast<SizeType>(plan.ring_size) *
                           top_words(),
                       0.0F);
        t->hist_coords.resize(target + 1);
        t->hist_max_delay.assign(target + 1, 0);
        for (SizeType l = level + 1; l <= target; ++l) {
            if (l < plan.hist_coords.size()) {
                t->hist_coords[l] = plan.hist_coords[l];
                for (const auto& hc : plan.hist_coords[l]) {
                    t->hist_max_delay[l] =
                        std::max(t->hist_max_delay[l], hc.delay);
                }
            }
        }
        if (target == root) {
            t->trials.resize(m_nbeams * plan.out_row.size() * m_chunk *
                             top_words());
        }
        logging::debug("FDMT (cuda): level {} summed from level {} (skewed={}, "
                       "target_ring={}, {} leaves, rings {} floats per beam)",
                       target, level, t->skewed, t->target_ring,
                       plan.leaves.size(), plan.ring_size);
        m_top_filled = static_cast<SizeType>(t->max_ring);
        m_top        = std::move(t);
    }

    /// Whether this chunk's trials come from the direct top: configured,
    /// float level k, and rings covering every trial's reach.
    [[nodiscard]] bool use_top(const std::vector<Elem>& types) const {
        return m_top && m_mode == FDMTMode::kValid &&
               types[m_top->level] == Elem::kF32 &&
               (m_top->skewed ||
                m_top_filled >= static_cast<SizeType>(m_top->max_ring));
    }

    /// The chunk's trials into `out` (rows `out_row_stride` apart, beams
    /// `out_beam_stride` apart) from level k, then level k's rings.
    void run_top(float* out,
                 int64_t out_beam_stride,
                 int out_row_stride,
                 cudaStream_t stream) {
        auto& t              = *m_top;
        constexpr int kWarps = 8;
        constexpr int kSamps = 32 * detail::kTopSampsPerLane;
        const int active     = static_cast<int>(m_active);
        if (t.target_ring) {
            const dim3 grid(t.ntrials, 1, m_nbeams);
            gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
            detail::kernel_direct_top_target<>
                <<<grid, kWarps * 32, 0, stream>>>(
                    reinterpret_cast<const float*>(m_levels_d[t.level].base),
                    level_beam_stride(t.level), static_cast<int>(m_chunk),
                    thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
                    thrust::raw_pointer_cast(t.ring_off.data()),
                    thrust::raw_pointer_cast(t.ring_len.data()),
                    thrust::raw_pointer_cast(t.leaf_off.data()),
                    thrust::raw_pointer_cast(t.leaves.data()),
                    thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
                    out_beam_stride, out_row_stride, m_chunk_pos, active, 0,
                    m_skew_nbatches > 1 ? 0 : -1, true);
            gpu_utils::check_last_gpu_error(
                "FDMT (cuda): direct top target failed");
            m_top_stale     = false;
            m_current_level = t.target;
            return;
        }
        const dim3 grid((t.ntrials + kWarps - 1) / kWarps,
                        (active + kSamps - 1) / kSamps, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
        detail::kernel_direct_top<><<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<const float*>(m_levels_d[t.level].base),
            level_beam_stride(t.level), static_cast<int>(m_chunk),
            thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()),
            thrust::raw_pointer_cast(t.leaf_off.data()),
            thrust::raw_pointer_cast(t.leaves.data()),
            thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
            out_beam_stride, out_row_stride, m_chunk_pos, active);
        gpu_utils::check_last_gpu_error("FDMT (cuda): direct top failed");
        update_top_rings(stream);
        m_top_stale     = !t.skewed;
        m_current_level = t.target;
    }

    /// Level k's newest samples of this chunk into its rings.
    void update_top_rings(cudaStream_t stream) {
        if (m_top->target_ring) {
            return;
        }
        auto& t                = *m_top;
        constexpr int kWarps   = 8;
        constexpr int kSegment = 256;
        const int newest = std::min(t.max_ring, static_cast<int>(m_active));
        if (newest > 0) {
            const dim3 grid((t.nrows + kWarps - 1) / kWarps,
                            (newest + kSegment - 1) / kSegment, m_nbeams);
            detail::kernel_direct_top_ring<><<<grid, kWarps * 32, 0, stream>>>(
                reinterpret_cast<const float*>(m_levels_d[t.level].base),
                level_beam_stride(t.level), static_cast<int>(m_chunk),
                thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
                thrust::raw_pointer_cast(t.ring_off.data()),
                thrust::raw_pointer_cast(t.ring_len.data()), t.nrows,
                m_chunk_pos, static_cast<int>(m_active));
            gpu_utils::check_last_gpu_error(
                "FDMT (cuda): direct top rings failed");
        }
        m_top_filled = std::min<SizeType>(static_cast<SizeType>(t.max_ring),
                                          m_top_filled + m_active);
    }

    /// Rewrites the tree histories of levels k+1..root from the rings at the
    /// current stream position, after top chunks left them behind.
    void rebuild_top_history(cudaStream_t stream) const {
        if (!m_top || !m_top_stale || m_top->skewed) {
            return;
        }
        const auto& t = *m_top;
        auto* ring    = const_cast<float*>( // NOLINT: device buffer, host const
            thrust::raw_pointer_cast(m_tree_ring_d.data()));
        for (SizeType l = t.level + 1; l <= t.target; ++l) {
            const int n = static_cast<int>(t.hist_coords[l].size());
            if (n == 0) {
                continue;
            }
            const dim3 grid((t.hist_max_delay[l] + 255) / 256,
                            std::min(n, kMaxGridY), m_nbeams);
            detail::kernel_direct_top_rebuild_history<<<grid, 256, 0, stream>>>(
                thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
                thrust::raw_pointer_cast(t.ring_off.data()),
                thrust::raw_pointer_cast(t.ring_len.data()),
                thrust::raw_pointer_cast(t.hist_coords[l].data()), n,
                thrust::raw_pointer_cast(t.hist_leaves.data()), ring,
                static_cast<int64_t>(m_ring_size),
                static_cast<int64_t>(m_fifo_size), m_next_pos);
        }
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): direct top history rebuild failed");
        m_top_stale = false;
    }

    [[nodiscard]] SizeType top_workspace_bytes() const noexcept {
        if (!m_top) {
            return 0;
        }
        const auto& t = *m_top;
        SizeType hist = 0;
        for (const auto& hc : t.hist_coords) {
            hist += hc.size() * sizeof(detail::FDMTDirectHistCoord);
        }
        return (t.trials.size() * sizeof(float)) +
               ((t.out_row.size() + t.leaf_off.size() + t.ring_len.size()) *
                sizeof(int32_t)) +
               (t.ring_off.size() * sizeof(int64_t)) +
               ((t.leaves.size() + t.hist_leaves.size()) *
                sizeof(detail::FDMTDirectTopLeaf)) +
               hist;
    }

    [[nodiscard]] SizeType direct_window_bytes() const noexcept {
        if (!m_direct) {
            return 0;
        }
        return ((m_direct->win[0].size() + m_direct->win[1].size() +
                 m_direct->win_ring.size()) *
                sizeof(float)) +
               (m_direct->win_ring_u8.size() * sizeof(uint8_t)) +
               (m_direct->win_ring_u16.size() * sizeof(uint16_t));
    }

    [[nodiscard]] SizeType direct_workspace_bytes() const noexcept {
        if (!m_direct) {
            return 0;
        }
        const auto& d = *m_direct;
        SizeType hist = 0;
        for (const auto& hc : d.hist_coords) {
            hist += hc.size() * sizeof(detail::FDMTDirectHistCoord);
        }
        return (d.prefix.size() * sizeof(float)) +
               (d.subs.size() * sizeof(detail::FDMTDirectSub)) +
               ((d.leaf_off.size() + d.chan_ring_len.size()) *
                sizeof(int32_t)) +
               (d.chan_ring_off.size() * sizeof(int64_t)) +
               ((d.leaves.size() + d.hist_leaves.size()) *
                sizeof(detail::FDMTDirectLeaf)) +
               hist;
    }

    /// Returns the number of leading sum coordinates whose tree-history ring
    /// the merge kernel already advanced (0 when the advance was not fused).
    int execute_iter_device(SizeType next_level,
                            cudaStream_t stream,
                            bool regular_only = false,
                            bool advance_hist = true) {
        const int fused_coords =
            execute_iter_levels(next_level, stream, regular_only, advance_hist);
        // The tree produced the direct top's level: keep its rings current
        // (a top chunk stores them after its trials read them).
        if (m_top && next_level == m_top->level && !m_top_chunk &&
            m_mode == FDMTMode::kValid) {
            if (m_levels_d[next_level].type == Elem::kF32) {
                update_top_rings(stream);
            } else {
                m_top_filled = 0; // integer level: the rings must refill
            }
        }
        return fused_coords;
    }

    int execute_iter_levels(SizeType next_level,
                            cudaStream_t stream,
                            bool regular_only = false,
                            bool advance_hist = true) {
        int fused_coords = 0;
        with_elem(m_levels_d[next_level - 1].type, [&](auto tag_in) {
            using TIn = decltype(tag_in);
            with_elem(m_levels_d[next_level].type, [&](auto tag_out) {
                using TOut = decltype(tag_out);
                // Levels only ever widen (see detail::int_tree_level_types).
                if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                              !(std::is_floating_point_v<TIn> &&
                                !std::is_floating_point_v<TOut>)) {
                    fused_coords = execute_iter_typed<TIn, TOut>(
                        reinterpret_cast<const TIn*>(
                            m_levels_d[next_level - 1].base),
                        reinterpret_cast<TOut*>(m_levels_d[next_level].base),
                        next_level, stream, regular_only, advance_hist);
                } else {
                    throw std::logic_error(
                        "FDMT (cuda): invalid narrowing level transition");
                }
            });
        });
        return fused_coords;
    }

    template <typename TIn, typename TOut>
    int execute_iter_typed(const TIn* __restrict__ in_ptr,
                           TOut* __restrict__ out_ptr,
                           SizeType next_level,
                           cudaStream_t stream,
                           bool regular_only = false,
                           bool advance_hist = true) {
        const auto& pc    = m_plan.get_container();
        const auto& shape = pc.state_shape[next_level];
        int fused_coords  = 0;
        // Valid mode: this chunk's samples (rows keep their table strides).
        const bool valid = m_mode == FDMTMode::kValid;
        const int active = valid ? static_cast<int>(m_active) : 0;
        const int extra_l =
            (valid && next_level < m_level_max_extra.size())
                ? static_cast<int>(m_level_max_extra[next_level])
                : 0;
        const int nsamps =
            valid ? (active + extra_l) : static_cast<int>(shape.nsamps);
        const int ncoords_sum_cur = static_cast<int>(
            regular_only ? detail::fdmt_regular_sum_coords(pc, next_level)
                         : shape.ncoords_sum);
        const int ncoords_copy_cur = static_cast<int>(shape.ncoords_copy);

        auto coords_sum_cur  = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto coords_copy_cur = m_plan_d.coordinates_copy.get_raw_ptrs();
        coords_sum_cur.update_offsets(m_coords_sum_offsets[next_level]);
        coords_copy_cur.update_offsets(m_coords_copy_offsets[next_level]);

        const auto coords_max     = std::max(ncoords_sum_cur, ncoords_copy_cur);
        constexpr int kBlockSamps = kIterThreads * kIterSampsPerTh;
        const dim3 block_size     = dim3(kIterThreads, 1);
        // Coordinates beyond grid.y's limit are strided in the kernel.
        const dim3 grid_size = dim3((nsamps + kBlockSamps - 1) / kBlockSamps,
                                    std::min(coords_max, kMaxGridY), m_nbeams);
        gpu_utils::check_kernel_launch_params(grid_size, block_size);

        // Beams are level_beam_stride() elements apart in each level.
        const auto in_state_nelements  = level_beam_stride(next_level - 1);
        const auto out_state_nelements = level_beam_stride(next_level);
        const auto hist                = tree_history(next_level);

        if (m_mode == FDMTMode::kFull) {
            kernel_execute_iter<FDMTMode::kFull, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur, hist.in,
                    nsamps, ncoords_sum_cur, ncoords_copy_cur,
                    in_state_nelements, out_state_nelements, hist.stride,
                    hist.base, hist.ring, m_int_unit, active);
        } else if (m_mode == FDMTMode::kRoll) {
            kernel_execute_iter<FDMTMode::kRoll, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur, hist.in,
                    nsamps, ncoords_sum_cur, ncoords_copy_cur,
                    in_state_nelements, out_state_nelements, hist.stride,
                    hist.base, hist.ring, m_int_unit, active);
        } else {
            const bool skewed_mid = m_direct && m_direct->skewed && m_top &&
                                    next_level <= m_top->level;
            const bool fuse_ring  = valid && !skewed_mid &&
                                    hist.ring != kFifoHistory &&
                                    hist.out != nullptr && grid_size.x == 1;
            kernel_execute_iter<FDMTMode::kValid, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    skewed_mid ? nullptr : hist.in, nsamps, ncoords_sum_cur,
                    ncoords_copy_cur, in_state_nelements, out_state_nelements,
                    hist.stride, hist.base, hist.ring, m_int_unit, active,
                    fuse_ring, skewed_mid ? static_cast<int>(m_chunk) : 0);

            fused_coords = (skewed_mid || fuse_ring) ? ncoords_sum_cur : 0;
            if (advance_hist && !skewed_mid) {
                advance_tree_history<TIn>(in_ptr, next_level, stream,
                                          fused_coords);
            }
        }
        gpu_utils::check_last_gpu_error("kernel_execute_iter launch failed");
        return fused_coords;
    }

    /// Advances `level`'s tree history after its merges.
    template <typename TIn>
    void advance_tree_history(const TIn* in_ptr,
                              SizeType level,
                              cudaStream_t stream,
                              int coord_begin = 0) {
        const auto& shape = m_plan.get_container().state_shape[level];
        const int count   = static_cast<int>(shape.ncoords_sum) - coord_begin;
        const auto hist   = tree_history(level);
        // A ring takes only the chunk's newest min(delay, chunk) samples.
        const int max_offset =
            static_cast<int>(hist.ring == kFifoHistory
                                 ? shape.dt_max
                                 : std::min<SizeType>(shape.dt_max, m_active));
        if (hist.out == nullptr || count <= 0 || max_offset == 0) {
            return;
        }
        auto coords = m_plan_d.coordinates_sum.get_raw_ptrs();
        coords.update_offsets(m_coords_sum_offsets[level] + coord_begin);
        launch_advance_tree_history(in_ptr, coords, hist.in, hist.out, count,
                                    max_offset, level_beam_stride(level - 1),
                                    hist.stride, hist.base, hist.ring,
                                    m_int_unit, stream);
        gpu_utils::check_last_gpu_error(
            "kernel_advance_tree_history launch failed");
    }

    template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
    void launch_init(const Input& waterfall_d,
                     TOut* __restrict__ state_d,
                     dim3 grid_size,
                     dim3 block_size,
                     cudaStream_t stream) {
        const auto& plan_c = m_plan.get_container();
        kernel_init_fdmt<Mode, Smear, Input, TOut>
            <<<grid_size, block_size, 0, stream>>>(
                waterfall_d, state_d, m_plan_d.grids0.dt_grid.data().get(),
                m_plan_d.grids0.ndt.data().get(),
                m_plan_d.grids0.coord_offset.data().get(),
                static_cast<int>(plan_c.state_shape[0].nchans),
                static_cast<int>(m_active), static_cast<int>(m_row_len0),
                static_cast<int>(plan_c.state_shape[0].dt_max),
                level_beam_stride(0), m_level0_hist_in_ptr,
                box_means_for(waterfall_d));
    }

    template <typename Input, typename TOut>
    void initialise_device(const Input& waterfall_d,
                           TOut* __restrict__ state_d,
                           cudaStream_t stream) {
        const auto& plan_c = m_plan.get_container();
        const int nsubs    = static_cast<int>(plan_c.state_shape[0].nchans);
        const int nsamps   = static_cast<int>(m_active);
        const int threads  = nsamps <= 256 ? 256 : (nsamps <= 512 ? 512 : 1024);
        const dim3 block_size = dim3(threads, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, nsubs, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid_size, block_size);

        const auto launch = [&](auto mode_tag, auto smear_tag) {
            launch_init<decltype(mode_tag)::value, decltype(smear_tag)::value,
                        Input, TOut>(waterfall_d, state_d, grid_size,
                                     block_size, stream);
        };
        using Full  = std::integral_constant<FDMTMode, FDMTMode::kFull>;
        using Roll  = std::integral_constant<FDMTMode, FDMTMode::kRoll>;
        using Valid = std::integral_constant<FDMTMode, FDMTMode::kValid>;
        if (m_mode == FDMTMode::kFull) {
            if (m_use_box_smearing) {
                launch(Full{}, std::true_type{});
            } else {
                launch(Full{}, std::false_type{});
            }
        } else if (m_mode == FDMTMode::kRoll) {
            if (m_use_box_smearing) {
                launch(Roll{}, std::true_type{});
            } else {
                launch(Roll{}, std::false_type{});
            }
        } else {
            if (m_use_box_smearing) {
                launch(Valid{}, std::true_type{});
            } else {
                launch(Valid{}, std::false_type{});
            }
        }
        gpu_utils::check_last_gpu_error("kernel_init_fdmt launch failed");

        advance_level0_history(waterfall_d, stream);
    }

    // Valid mode: advance the per-channel level-0 input history by this
    // block (independent of how level 0 itself was computed).
    template <typename Input>
    void advance_level0_history(const Input& waterfall_d, cudaStream_t stream) {
        const auto& plan_c = m_plan.get_container();
        const int nsubs    = static_cast<int>(plan_c.state_shape[0].nchans);
        const int nsamps   = static_cast<int>(m_active);
        // The last dt0_max raw samples of every channel.
        const int dt_max_final = static_cast<int>(plan_c.state_shape[0].dt_max);
        if (m_mode == FDMTMode::kValid && m_level0_hist_out_ptr != nullptr &&
            dt_max_final > 0) {
            const dim3 block_hist = dim3(256, 1);
            const dim3 grid_hist =
                dim3((dt_max_final + block_hist.x - 1) / block_hist.x, nsubs,
                     m_nbeams);
            kernel_advance_history_window<Input>
                <<<grid_hist, block_hist, 0, stream>>>(
                    m_level0_hist_out_ptr, m_level0_hist_in_ptr, waterfall_d,
                    nsubs, nsamps, dt_max_final);
            gpu_utils::check_last_gpu_error(
                "kernel_advance_history_window launch failed");
        }
    }
}; // End FDMTCudaEngine definition

} // namespace

std::unique_ptr<detail::FDMTEngine>
detail::make_fdmt_gpu(const plans::FDMTPlan& plan,
                      const detail::FDMTEngineConfig& cfg) {
    return std::make_unique<FDMTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
