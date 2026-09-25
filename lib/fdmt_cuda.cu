#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/fdmt_fused_tile.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/modes.hpp"
#include "dmt/plans_cuda.cuh"

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
                 int dt_max_final,
                 int state0_nelements,
                 const float* __restrict__ hist) {
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
    const auto buffer_offset =
        (i_beam * state0_nelements) + (grids0_coord_offset_ptr[i_sub] * nsamps);
    const auto ndt_grid_sub = grids0_ndt_ptr[i_sub];
    const auto wf_row       = (i_beam * nsubs) + i_sub;
    const auto hist_offset =
        (i_beam * nsubs * dt_max_final) + (i_sub * dt_max_final);
    const auto dt_first = dt_grid_sub[0];
    const auto dt_last  = dt_grid_sub[ndt_grid_sub - 1];

    // Lambda to fetch sample at relative index (isamp - shift), shift >= 0.
    auto get_sample = [&](int t) -> float {
        if (t >= 0) {
            return waterfall.load(wf_row, t);
        }
        if constexpr (Mode == FDMTMode::kRoll) {
            return waterfall.load(wf_row, t + nsamps);
        } else if constexpr (Mode == FDMTMode::kValid) {
            return (hist != nullptr) ? hist[hist_offset + dt_max_final + t]
                                     : 0.0F;
        } else {
            return 0.0F; // kFull: zero-padded before t=0
        }
    };

    // Level-0 values are computed in float; for integer TOut (packed input,
    // int_tree) they are exact integers within the type's plan bound. The
    // column helper is shared with the fused kernel, so both agree bit for
    // bit.
    detail::fdmt_init_column<UseBoxSmearing>(
        dt_first, dt_last, isamp, get_sample, [&](int i_dt, float val) {
            state[buffer_offset + (static_cast<int64_t>(i_dt) * nsamps) +
                  isamp] = static_cast<TOut>(val);
        });
}

/**
 * @brief Advances the level-0 FIFO history window on the GPU.
 * Supports arbitrary block sizes (nsamps < dt_max_final and nsamps >=
 * dt_max_final).
 */
template <typename Input>
__global__ void kernel_advance_history_window(float* __restrict__ hist_out,
                                              const float* __restrict__ hist_in,
                                              const Input waterfall,
                                              int nsubs,
                                              int nsamps,
                                              int dt_max_final) {
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

/**
 * @brief Per-level tree merge kernel with multi-beam support.
 */
template <FDMTMode Mode, typename TIn = float, typename TOut = float>
__global__ void kernel_execute_iter(const TIn* __restrict__ state_in,
                                    TOut* __restrict__ state_out,
                                    const plans::FDMTCoordDPtrs coords_sum,
                                    const plans::FDMTCoordDPtrs coords_copy,
                                    const float* __restrict__ hist_in,
                                    int nsamps,
                                    int ncoords_sum_cur,
                                    int ncoords_copy_cur,
                                    int in_state_nelements,
                                    int out_state_nelements,
                                    int tree_hist_size) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_coord = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (isamp >= nsamps) {
        return;
    }

    if (i_coord < ncoords_sum_cur) {
        const auto nsamps_out  = coords_sum.nsamps[i_coord];
        const auto offset      = coords_sum.offset[i_coord];
        const auto nsamps_tail = coords_sum.tail_nsamps[i_coord];
        const auto out_idx_base =
            (i_beam * out_state_nelements) + coords_sum.buf_offset[i_coord];
        const auto tail_idx_base =
            (i_beam * in_state_nelements) + coords_sum.tail_buf_offset[i_coord];
        const auto head_idx_base =
            (i_beam * in_state_nelements) + coords_sum.head_buf_offset[i_coord];
        const auto hist_off =
            (i_beam * tree_hist_size) + coords_sum.hist_offset[i_coord];

        if (isamp < nsamps_out) {
            // Summed in the output storage type: float as before, or the
            // narrow integer type (int_tree), whose plan bound cannot wrap.
            TOut tail_val = TOut{0};
            TOut head_val = TOut{0};
            if constexpr (Mode == FDMTMode::kFull) {
                if (isamp < nsamps_tail) {
                    tail_val =
                        static_cast<TOut>(state_in[tail_idx_base + isamp]);
                }
                if (isamp >= offset && (isamp - offset) < nsamps_tail) {
                    head_val = static_cast<TOut>(
                        state_in[head_idx_base + (isamp - offset)]);
                }
            } else if constexpr (Mode == FDMTMode::kValid) {
                tail_val = static_cast<TOut>(state_in[tail_idx_base + isamp]);
                if (isamp >= offset) {
                    head_val = static_cast<TOut>(
                        state_in[head_idx_base + (isamp - offset)]);
                } else if (hist_in != nullptr) {
                    head_val = static_cast<TOut>(hist_in[hist_off + isamp]);
                }
            } else if constexpr (Mode == FDMTMode::kRoll) {
                tail_val = static_cast<TOut>(state_in[tail_idx_base + isamp]);
                const int head_samp = (isamp >= offset)
                                          ? (isamp - offset)
                                          : (nsamps_tail - offset + isamp);
                head_val =
                    static_cast<TOut>(state_in[head_idx_base + head_samp]);
            }
            state_out[out_idx_base + isamp] =
                static_cast<TOut>(tail_val + head_val);
        }
    }

    if (i_coord < ncoords_copy_cur) {
        const auto nsamps_out  = coords_copy.nsamps[i_coord];
        const auto nsamps_tail = coords_copy.tail_nsamps[i_coord];
        const auto out_idx_base =
            (i_beam * out_state_nelements) + coords_copy.buf_offset[i_coord];
        if (isamp < nsamps_tail) {
            const auto tail_idx_base = (i_beam * in_state_nelements) +
                                       coords_copy.tail_buf_offset[i_coord];
            state_out[out_idx_base + isamp] =
                static_cast<TOut>(state_in[tail_idx_base + isamp]);
        } else if (isamp < nsamps_out) {
            state_out[out_idx_base + isamp] = TOut{0};
        }
    }
}

/**
 * @brief Advances tree-level cross-block FIFO history on GPU.
 * Supports arbitrary block sizes and offsets.
 */
template <typename TIn = float>
__global__ void
kernel_advance_tree_history(const TIn* __restrict__ state_in,
                            const plans::FDMTCoordDPtrs coords_sum,
                            const float* __restrict__ hist_in,
                            float* __restrict__ hist_out,
                            int ncoords_sum_cur,
                            int in_state_nelements,
                            int tree_hist_size) {
    const auto k = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_coord = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (i_coord >= ncoords_sum_cur) {
        return;
    }
    const auto offset = coords_sum.offset[i_coord];
    if (offset <= 0 || k >= offset) {
        return;
    }

    const auto nsamps_head = coords_sum.head_nsamps[i_coord];
    const auto hist_off =
        (i_beam * tree_hist_size) + coords_sum.hist_offset[i_coord];
    const auto head_idx_base =
        (i_beam * in_state_nelements) + coords_sum.head_buf_offset[i_coord];

    if (nsamps_head >= offset) {
        hist_out[hist_off + k] = static_cast<float>(
            state_in[head_idx_base + (nsamps_head - offset + k)]);
    } else {
        if (k < offset - nsamps_head) {
            hist_out[hist_off + k] = (hist_in != nullptr)
                                         ? hist_in[hist_off + k + nsamps_head]
                                         : 0.0F;
        } else {
            hist_out[hist_off + k] = static_cast<float>(
                state_in[head_idx_base + (k - (offset - nsamps_head))]);
        }
    }
}

/// @brief Thread-block execution policy of detail::fdmt_fused_tile: work
/// items strided over the block's threads, __syncthreads() as the barrier.
struct DeviceBlock {
    template <typename F>
    __host__ __device__ void for_each(int n, F&& f) const {
#ifdef __CUDA_ARCH__
        for (int i = static_cast<int>(threadIdx.x); i < n;
             i += static_cast<int>(blockDim.x)) {
            f(i);
        }
#else
        (void)n;
        (void)f;
#endif
    }
    __host__ __device__ void sync() const {
#ifdef __CUDA_ARCH__
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
                                    int tree_hist_size) {
    extern __shared__ float smem[];
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    const auto hist0_stride =
        static_cast<int64_t>(args.nsubs) * args.dt_max_final;
    detail::fdmt_fused_tile<Mode, UseBoxSmearing>(
        DeviceBlock{}, waterfall, i_beam * args.nsubs,
        state_out + (i_beam * out_beam_stride), args,
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), smem,
        (hist0 != nullptr) ? hist0 + (i_beam * hist0_stride) : nullptr,
        (thist_in != nullptr) ? thist_in + (i_beam * tree_hist_size) : nullptr,
        (thist_out != nullptr) ? thist_out + (i_beam * tree_hist_size)
                               : nullptr);
}

} // namespace

class FDMTCUDA::Impl {
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
         int verbose,
         int device_id,
         SizeType nbeams,
         SizeType fuse_levels,
         bool int_tree)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_int_tree(int_tree),
          m_mode(parse_fdmt_mode(mode)),
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
                 verbose) {
        init_device_structures(fuse_levels);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         bool use_box_smearing,
         std::string_view mode,
         int verbose,
         int device_id,
         SizeType nbeams,
         SizeType fuse_levels,
         bool int_tree)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_int_tree(int_tree),
          m_mode(parse_fdmt_mode(mode)),
          m_nbeams(nbeams),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, verbose) {
        init_device_structures(fuse_levels);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing,
         std::string_view mode,
         int verbose,
         int device_id,
         SizeType nbeams,
         SizeType fuse_levels,
         bool int_tree)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_int_tree(int_tree),
          m_mode(parse_fdmt_mode(mode)),
          m_nbeams(nbeams),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, verbose) {
        init_device_structures(fuse_levels);
    }

    // Allocates every device buffer and resolves the fusion depth: nothing is
    // allocated after construction except the per-call staging of the
    // host-memory execute() overloads.
    void init_device_structures(SizeType fuse_levels) {
        check_int32_extents();
        cuda_utils::set_device(m_device_id);
        spdlog::debug("FDMTCUDA::Impl: Set device to {}", m_device_id);
        // Allocate internal state buffer on device for ping-pong
        m_state_internal_d.resize(m_nbeams * m_plan.get_buffer_size(), 0.0F);
        if (m_mode == FDMTMode::kValid) {
            const auto hist_size = m_nbeams * m_plan.get_history_size();
            m_history_a_d.resize(hist_size, 0.0F);
            m_history_b_d.resize(hist_size, 0.0F);
            // Two same-sized tree-history buffers, ping-ponged once per
            // block in reset() -- see kernel_execute_iter's doc comment for
            // why one buffer isn't safe here the way it is on the CPU.
            const auto tree_hist_size =
                m_nbeams * m_plan.get_tree_history_size();
            m_tree_history_a_d.resize(tree_hist_size, 0.0F);
            m_tree_history_b_d.resize(tree_hist_size, 0.0F);
        }
        plans::transfer_fdmt_plan_to_device(m_plan.get_container(), m_plan_d);

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
        configure_fusion(fuse_levels);
        cuda_utils::check_last_cuda_error("FDMTCUDA::Impl constructor failed");
    }

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    void execute_h(std::span<const float> waterfall_h, std::span<float> dmt_h) {
        check_inputs(waterfall_h.size(), dmt_h.size());

        cuda_utils::set_device(m_device_id);
        thrust::device_vector<float> waterfall_d(waterfall_h.size());
        thrust::device_vector<float> dmt_d(dmt_h.size());

        cudaStream_t stream = nullptr;
        // Copy H->D
        cudaMemcpyAsync(waterfall_d.data().get(), waterfall_h.data(),
                        waterfall_h.size_bytes(), cudaMemcpyHostToDevice,
                        stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaMemcpyAsync H->D waterfall failed");

        // Execute on device
        execute_d(cuda::std::span<const float>(
                      thrust::raw_pointer_cast(waterfall_d.data()),
                      waterfall_d.size()),
                  cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                         dmt_d.size()),
                  stream);

        // Copy D->H
        cudaMemcpyAsync(dmt_h.data(), dmt_d.data().get(), dmt_h.size_bytes(),
                        cudaMemcpyDeviceToHost, stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaMemcpyAsync D->H dmt failed");

        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaStreamSynchronize failed");

        spdlog::debug("FDMTCUDA::Impl: Host execution complete.");
    }

    void execute_d(cuda::std::span<const float> waterfall_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        reset_impl(waterfall_d, dmt_d, stream, /*fuse=*/true);
        advance_until_remaining(0, stream);
        finalize(stream);
        spdlog::debug("FDMTCUDA::Impl: Device execution complete on stream");
    }

    void execute_h(std::span<const uint8_t> waterfall_h,
                   SizeType nbits,
                   std::span<float> dmt_h) {
        check_packed_inputs(waterfall_h.size(), nbits, dmt_h.size());
        cuda_utils::set_device(m_device_id);
        // Only the packed bytes cross PCIe (32/nbits times less than float).
        thrust::device_vector<uint8_t> waterfall_d(waterfall_h.size());
        thrust::device_vector<float> dmt_d(dmt_h.size());
        cudaStream_t stream = nullptr;
        cudaMemcpyAsync(waterfall_d.data().get(), waterfall_h.data(),
                        waterfall_h.size_bytes(), cudaMemcpyHostToDevice,
                        stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaMemcpyAsync H->D packed waterfall failed");
        execute_d(cuda::std::span<const uint8_t>(
                      thrust::raw_pointer_cast(waterfall_d.data()),
                      waterfall_d.size()),
                  nbits,
                  cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                         dmt_d.size()),
                  stream);
        cudaMemcpyAsync(dmt_h.data(), dmt_d.data().get(), dmt_h.size_bytes(),
                        cudaMemcpyDeviceToHost, stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaMemcpyAsync D->H dmt failed");
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaStreamSynchronize failed");
    }

    void execute_d(cuda::std::span<const uint8_t> waterfall_d,
                   SizeType nbits,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        reset_impl(waterfall_d, nbits, dmt_d, stream, /*fuse=*/true);
        advance_until_remaining(0, stream);
        finalize(stream);
    }

    void reset(cuda::std::span<const uint8_t> d_waterfall,
               SizeType nbits,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr) {
        reset_impl(d_waterfall, nbits, d_dmt, stream, /*fuse=*/false);
    }

    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr) {
        reset_impl(d_waterfall, d_dmt, stream, /*fuse=*/false);
    }

    // `fuse`: execute() only -- replace level 0 and the first merges by the
    // fused kernel (see configure_fusion()); the stepper stays unfused.
    void reset_impl(cuda::std::span<const uint8_t> d_waterfall,
                    SizeType nbits,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse) {
        check_packed_inputs(d_waterfall.size(), nbits, d_dmt.size());
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
        const FDMTInputPacked input{.data      = d_waterfall.data(),
                                    .row_bytes = static_cast<int>(row_bytes),
                                    .nbits     = static_cast<int>(nbits)};
        start(input, m_int_tree ? m_int_levels[nbits] : m_float_levels, d_dmt,
              stream, fuse);
    }

    void reset_impl(cuda::std::span<const float> d_waterfall,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse) {
        check_inputs(d_waterfall.size(), d_dmt.size());
        const FDMTInputF32 input{
            .data   = d_waterfall.data(),
            .nsamps = static_cast<int>(m_plan.get_nsamps()),
        };
        start(input, m_float_levels, d_dmt, stream, fuse);
    }

    [[nodiscard]] SizeType get_fuse_levels() const noexcept {
        return m_fused ? static_cast<SizeType>(m_fused->args.fuse) : 0;
    }

    [[nodiscard]] bool get_int_tree() const noexcept { return m_int_tree; }

    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept {
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
            coord_bytes(m_plan_d.coordinates) +
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
            .plan    = plan_bytes,
            .state   = m_state_internal_d.size() * sizeof(float),
            .history = (m_history_a_d.size() + m_history_b_d.size() +
                        m_tree_history_a_d.size() + m_tree_history_b_d.size()) *
                       sizeof(float),
            .workspace = fused_bytes,
            .output    = m_nbeams * m_plan.get_buffer_size() * sizeof(float),
        };
    }

    void advance(SizeType levels = 1, cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        cuda_utils::set_device(m_device_id);
        cudaStream_t active_stream = stream ? stream : m_stream;
        const auto total_lvl       = total_levels();
        const auto target_level =
            std::min(m_current_level + levels, total_lvl - 1);

        while (m_current_level < target_level) {
            execute_iter_device(m_current_level + 1, active_stream);
            m_current_level++;
        }
        spdlog::debug("FDMTCUDA: Stepper advanced to level {}.",
                      m_current_level);
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
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

    cuda::std::span<const float> view_level_data() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& shape = m_plan.get_container().state_shape[m_current_level];
        // Beams are get_buffer_size() apart; the span covers through the
        // last beam's valid elements.
        return cuda::std::span<const float>(
            current_level_f32(),
            ((m_nbeams - 1) * m_plan.get_buffer_size()) + shape.nelements);
    }

    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& grid =
            m_plan.get_container().grids[m_current_level][subband_idx];
        const auto nsamps =
            m_plan.get_container().state_shape[m_current_level].nsamps;
        const auto offset = grid.coord_offset * nsamps;
        const auto count  = grid.ndt * nsamps;
        return cuda::std::span<const float>(current_level_f32() + offset,
                                            count);
    }

    FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& grid =
            m_plan.get_container().grids[m_current_level][subband_idx];
        const auto nsamps =
            m_plan.get_container().state_shape[m_current_level].nsamps;
        return FDMTSubbandViewCUDA{
            .data        = view_subband_data(subband_idx),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    SizeType current_level() const noexcept { return m_current_level; }
    SizeType total_levels() const noexcept { return m_plan.get_niters() + 1; }
    SizeType remaining_levels() const noexcept {
        return total_levels() - 1 - m_current_level;
    }
    SizeType num_subbands() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }
    bool is_finished() const noexcept {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        cuda_utils::set_device(m_device_id);
        cudaStream_t active_stream = stream ? stream : m_stream;
        advance_until_remaining(0, active_stream);

        const auto* final_ptr =
            reinterpret_cast<const float*>(m_levels_d[m_current_level].base);
        if (final_ptr != m_dmt_target_ptr) {
            spdlog::debug(
                "FDMTCUDA::finalize: Copying final state from internal "
                "scratch to dmt_target_ptr");
            const auto final_size =
                ((m_nbeams - 1) * m_plan.get_buffer_size()) +
                m_plan.get_container().state_shape.back().nelements;
            cudaMemcpyAsync(m_dmt_target_ptr, final_ptr,
                            final_size * sizeof(float),
                            cudaMemcpyDeviceToDevice, active_stream);
            cuda_utils::check_last_cuda_error(
                "finalize: cudaMemcpyAsync final state to dmt failed");
        }
        m_is_initialized = false;
    }

    // Analytical variance/sigma: pure plan-side math (see
    // plans::FDMTPlan::get_effective_variance), identical on CPU and CUDA --
    // provided here for API parity with FDMTCPU.
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

    [[nodiscard]] SizeType get_nbeams() const noexcept { return m_nbeams; }

    /// Zeroes all cross-block history (level-0 and tree-level) so the next
    /// execute()/reset() call starts as if this were a brand-new instance.
    /// Mirrors FDMTCPU::reset_history().
    void reset_history() noexcept {
        cuda_utils::set_device(m_device_id);
        if (!m_history_a_d.empty()) {
            m_history_a_d.assign(m_history_a_d.size(), 0.0F);
        }
        if (!m_history_b_d.empty()) {
            m_history_b_d.assign(m_history_b_d.size(), 0.0F);
        }
        if (!m_tree_history_a_d.empty()) {
            m_tree_history_a_d.assign(m_tree_history_a_d.size(), 0.0F);
        }
        if (!m_tree_history_b_d.empty()) {
            m_tree_history_b_d.assign(m_tree_history_b_d.size(), 0.0F);
        }
        m_tree_history_parity = false;
    }

    /// Total device-buffer size (floats) of this instance's "valid"-mode
    /// streaming history, as saved/restored by save_history()/load_history()
    /// -- the two level-0 buffers, the two tree-level buffers, and one extra
    /// float encoding m_tree_history_parity. Zero (well, the trailing flag
    /// aside) for "full"/"roll" mode instances.
    [[nodiscard]] SizeType history_state_size() const noexcept {
        return m_history_a_d.size() + m_history_b_d.size() +
               m_tree_history_a_d.size() + m_tree_history_b_d.size() + 1;
    }

    /// Copies this instance's current streaming history (both ping-pong
    /// buffer pairs plus the parity flag) out to a caller-owned device
    /// buffer, so one shared FDMTCUDA instance can multiplex several
    /// independent streams (each with its own history) instead of requiring
    /// one instance per stream. Enqueued on `stream`; synchronize before
    /// reading `out` elsewhere.
    void save_history(cuda::std::span<float> out, cudaStream_t stream) const {
        if (out.size() != history_state_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA::save_history: Invalid output size. Expected {}, "
                "got {}",
                history_state_size(), out.size()));
        }
        cuda_utils::set_device(m_device_id);
        float* dst    = out.data();
        auto copy_out = [&](const thrust::device_vector<float>& src) {
            if (!src.empty()) {
                cudaMemcpyAsync(dst, thrust::raw_pointer_cast(src.data()),
                                src.size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            }
            dst += src.size();
        };
        copy_out(m_history_a_d);
        copy_out(m_history_b_d);
        copy_out(m_tree_history_a_d);
        copy_out(m_tree_history_b_d);
        const float parity = m_tree_history_parity ? 1.0F : 0.0F;
        cudaMemcpyAsync(dst, &parity, sizeof(float), cudaMemcpyHostToDevice,
                        stream);
        cuda_utils::check_last_cuda_error("FDMTCUDA::save_history failed");
    }

    /// Replaces this instance's current streaming history with a buffer
    /// previously produced by save_history() (from an instance built with
    /// the same plan geometry), resuming that stream. Blocks briefly to read
    /// back the parity flag onto the host -- use reset_history() instead to
    /// start a stream cold.
    void load_history(cuda::std::span<const float> in, cudaStream_t stream) {
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA::load_history: Invalid input size. Expected {}, "
                "got {}",
                history_state_size(), in.size()));
        }
        cuda_utils::set_device(m_device_id);
        const float* src = in.data();
        auto copy_in     = [&](thrust::device_vector<float>& dst_vec) {
            if (!dst_vec.empty()) {
                cudaMemcpyAsync(thrust::raw_pointer_cast(dst_vec.data()), src,
                                    dst_vec.size() * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream);
            }
            src += dst_vec.size();
        };
        copy_in(m_history_a_d);
        copy_in(m_history_b_d);
        copy_in(m_tree_history_a_d);
        copy_in(m_tree_history_b_d);
        cudaStreamSynchronize(stream);
        float parity = 0.0F;
        cudaMemcpy(&parity, src, sizeof(float), cudaMemcpyDeviceToHost);
        m_tree_history_parity = parity != 0.0F;
        cuda_utils::check_last_cuda_error("FDMTCUDA::load_history failed");
    }

private:
    int m_device_id;
    bool m_use_box_smearing;
    bool m_int_tree; // performance parameter (packed input)
    FDMTMode m_mode;
    SizeType m_nbeams{1};
    plans::FDMTPlan m_plan;
    plans::FDMTPlanContainerD m_plan_d;
    // Internal state buffer on device (for ping-pong buffering)
    thrust::device_vector<float> m_state_internal_d;
    // Level-0 history buffer for valid-mode streaming across FDMT blocks
    // (ping-pong pair)
    thrust::device_vector<float> m_history_a_d;
    thrust::device_vector<float> m_history_b_d;
    float* m_level0_hist_in_ptr{nullptr};
    float* m_level0_hist_out_ptr{nullptr};
    // Tree-level (level >= 1) cross-block history: two same-sized buffers
    thrust::device_vector<float> m_tree_history_a_d;
    thrust::device_vector<float> m_tree_history_b_d;
    bool m_tree_history_parity{false};
    float* m_hist_in_ptr{nullptr};
    float* m_hist_out_ptr{nullptr};

    // Per-level storage types (all float, and per nbits for int_tree), fixed
    // at construction.
    using Elem = detail::FDMTLevelType;
    std::vector<Elem> m_float_levels;
    std::array<std::vector<Elem>, 17> m_int_levels; // indexed by nbits

    // Level fusion (fuse_levels): the resolved depth's
    // device tables and launch geometry; empty when execute() is unfused.
    struct FusedLevelsD {
        DeviceVector<int> group_info;
        DeviceVector<int> coords;
        DeviceVector<int> dt_grid0;
        DeviceVector<int> ndt0;
        DeviceVector<int> coord_offset0;
        detail::FDMTFusedTileArgs args{};
        int ntiles{0};
        int ngroups{0};
        SizeType smem_bytes{0};
    };
    std::unique_ptr<FusedLevelsD> m_fused;
    // Auto depth: the deepest fusion whose shared memory at a tile of at
    // least kMinAutoTile samples fits kAutoSmemBytes (two or more blocks per
    // SM on current GPUs) with a level-0 halo of at most half a tile.
    static constexpr SizeType kAutoSmemBytes = SizeType{48} * 1024;
    static constexpr int kMinAutoTile        = 256;
    static constexpr int kMaxTile            = 1024;
    static constexpr int kFusedThreads       = 256;

    // One tree level's device state: beam 0 at `base`, beam b at element
    // offset b * get_buffer_size(), stored as `type`.
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
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;

    // The device plan stores per-beam coordinate offsets as 32-bit ints (see
    // plans_cuda.cuh), and the kernels index within one beam in 32-bit
    // arithmetic (only the beam-strided part is 64-bit). Reject plans whose
    // per-beam extents do not fit, instead of silently wrapping indices.
    void check_int32_extents() const {
        constexpr auto kMax =
            static_cast<SizeType>(std::numeric_limits<int32_t>::max());
        const auto check = [&](SizeType value, std::string_view what) {
            if (value > kMax) {
                throw std::invalid_argument(std::format(
                    "FDMTCUDA: per-beam {} ({} elements) exceeds the 32-bit "
                    "index range of the CUDA backend ({}); use a smaller "
                    "block (nsamps) or fewer DM trials",
                    what, value, kMax));
            }
        };
        check(m_plan.get_buffer_size(), "state buffer");
        check(m_plan.get_nchans() * m_plan.get_nsamps(), "waterfall");
        check(m_plan.get_history_size(), "level-0 history");
        check(m_plan.get_tree_history_size(), "tree history");
    }

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != m_nbeams * nchans * nsamps) {
            throw std::invalid_argument(
                std::format("FDMTCUDA: Invalid size of waterfall. "
                            "Expected {}, got {}",
                            m_nbeams * nchans * nsamps, waterfall_size));
        }
        if (dmt_size < m_nbeams * m_plan.get_buffer_size()) {
            throw std::invalid_argument(
                std::format("FDMTCUDA: Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_nbeams * m_plan.get_buffer_size(), dmt_size));
        }
        spdlog::debug("FDMTCUDA: Input dimensions check passed: {}x{}x{}",
                      m_nbeams, nchans, nsamps);
    }

    void check_packed_inputs(SizeType waterfall_bytes,
                             SizeType nbits,
                             SizeType dmt_size) const {
        if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 &&
            nbits != 16) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA: nbits={} must be one of 1, 2, 4, 8, 16", nbits));
        }
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
        const auto expected = m_nbeams * m_plan.get_nchans() * row_bytes;
        if (waterfall_bytes != expected) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA: Invalid size of packed waterfall (nbits={}). "
                "Expected {} bytes, got {}",
                nbits, expected, waterfall_bytes));
        }
        if (dmt_size < m_nbeams * m_plan.get_buffer_size()) {
            throw std::invalid_argument(
                std::format("FDMTCUDA: Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_nbeams * m_plan.get_buffer_size(), dmt_size));
        }
    }

    [[nodiscard]] int beam_stride_elements() const noexcept {
        return static_cast<int>(m_plan.get_buffer_size());
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
            throw std::logic_error(std::format(
                "FDMTCUDA: level {} is stored as an integer type "
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
            return kAutoSmemBytes;
        }
        return static_cast<SizeType>(bytes);
    }

    /**
     * Resolves the fuse_levels constructor argument for this plan and device
     * and uploads the fused kernel's tables. An explicit depth is clamped to
     * the plan's merge levels and kMaxFusedLevels, and reduced until a tile
     * fits the device's (opt-in) shared memory; kFDMTAutoFuse follows the rule
     * at kAutoSmemBytes. Leaves m_fused empty when no depth qualifies.
     */
    void configure_fusion(SizeType requested) {
        m_fused.reset();
        const SizeType niters = m_plan.get_niters();
        const SizeType deepest =
            std::min<SizeType>(niters, detail::kMaxFusedLevels);
        if (requested == 0 || deepest == 0) {
            return;
        }
        const bool automatic = requested == kFDMTAutoFuse;
        const auto& pc       = m_plan.get_container();
        const SizeType optin =
            automatic ? kAutoSmemBytes : max_optin_smem_bytes();
        for (SizeType f = automatic ? deepest : std::min(requested, deepest);
             f >= 1; --f) {
            const auto plan = detail::build_fused_tile_plan(pc, niters, f);
            // Never wider than the (32-rounded) level-F row itself.
            const int max_tile =
                std::min(kMaxTile, ((plan.ntiles_nsamps + 31) / 32) * 32);
            int tile = plan.max_tile_nsamps(kAutoSmemBytes, max_tile);
            if (automatic) {
                const int min_tile = std::min(kMinAutoTile, max_tile);
                if (tile < min_tile || 2 * plan.max_level0_halo > tile) {
                    continue;
                }
            } else if (tile < 64) {
                tile = std::max(tile, plan.max_tile_nsamps(optin, max_tile));
            }
            if (tile > 0) {
                if (!automatic && f != requested) {
                    spdlog::warn("FDMTCUDA: fuse_levels={} reduced to {} (plan "
                                 "merge levels / shared memory)",
                                 requested, f);
                }
                adopt_fused_plan(plan, tile);
                return;
            }
        }
        if (!automatic) {
            spdlog::warn("FDMTCUDA: fuse_levels={} does not fit shared memory; "
                         "running unfused",
                         requested);
        }
    }

    void adopt_fused_plan(const detail::FDMTFusedTilePlan& plan, int tile) {
        cuda_utils::set_device(m_device_id);
        auto fused              = std::make_unique<FusedLevelsD>();
        fused->group_info       = plan.group_info;
        fused->coords           = plan.coords;
        fused->dt_grid0         = plan.dt_grid0;
        fused->ndt0             = plan.ndt0;
        fused->coord_offset0    = plan.coord_offset0;
        const auto dt_max_final = static_cast<int>(
            m_plan.get_container().state_shape[m_plan.get_niters()].dt_max);
        fused->args = detail::make_fused_tile_args(
            plan, tile, dt_max_final,
            thrust::raw_pointer_cast(fused->group_info.data()),
            thrust::raw_pointer_cast(fused->coords.data()),
            thrust::raw_pointer_cast(fused->dt_grid0.data()),
            thrust::raw_pointer_cast(fused->ndt0.data()),
            thrust::raw_pointer_cast(fused->coord_offset0.data()));
        const auto [cap_a, cap_b] = plan.smem_floats(tile);
        fused->smem_bytes         = (cap_a + cap_b) * sizeof(float);
        fused->ntiles             = (plan.ntiles_nsamps + tile - 1) / tile;
        fused->ngroups            = plan.ngroups;
        cuda_utils::check_last_cuda_error("FDMTCUDA: fused plan upload failed");
        spdlog::debug("FDMTCUDA: fusing levels 0..{} (tile {} samples, {} "
                      "groups, {} B shared)",
                      plan.fuse, tile, plan.ngroups, fused->smem_bytes);
        m_fused = std::move(fused);
    }

    template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
    void
    launch_fused_typed(const Input& input, TOut* out, cudaStream_t stream) {
        const auto& f = *m_fused;
        if (f.smem_bytes > kAutoSmemBytes) {
            cudaFuncSetAttribute(kernel_fused_levels<Mode, Smear, Input, TOut>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(f.smem_bytes));
        }
        const dim3 grid_size(static_cast<unsigned>(f.ntiles),
                             static_cast<unsigned>(f.ngroups),
                             static_cast<unsigned>(m_nbeams));
        const dim3 block_size(kFusedThreads);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);
        const bool valid = Mode == FDMTMode::kValid;
        kernel_fused_levels<Mode, Smear, Input, TOut>
            <<<grid_size, block_size, f.smem_bytes, stream>>>(
                input, out, f.args,
                static_cast<int64_t>(beam_stride_elements()),
                valid ? m_level0_hist_in_ptr : nullptr,
                valid ? m_hist_in_ptr : nullptr,
                valid ? m_hist_out_ptr : nullptr,
                static_cast<int>(m_plan.get_tree_history_size()));
        cuda_utils::check_last_cuda_error("kernel_fused_levels launch failed");
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
     * Assigns each level's device buffer -- the same rule as FDMTCPU: float
     * level l lives in the caller's dmt buffer iff niters - l is even (so
     * the root always lands there and finalize() never copies), otherwise in
     * m_state_internal_d; integer levels (int_tree) alternate between the
     * two halves of m_state_internal_d, each nbeams * buffer_size * 2 bytes
     * (every level's beams are buffer_size elements apart).
     * detail::int_tree_level_types() guarantees the first float level after
     * them is a dmt level.
     */
    void layout_levels(const std::vector<Elem>& types,
                       cuda::std::span<float> d_dmt) {
        const SizeType niters = m_plan.get_niters();
        auto* dmt_bytes       = reinterpret_cast<std::byte*>(d_dmt.data());
        auto* internal        = reinterpret_cast<std::byte*>(
            thrust::raw_pointer_cast(m_state_internal_d.data()));
        const SizeType half =
            m_nbeams * m_plan.get_buffer_size() * sizeof(uint16_t);
        for (SizeType l = 0; l <= niters; ++l) {
            if (types[l] == Elem::kF32) {
                m_levels_d[l] = {.base = ((niters - l) % 2 == 0) ? dmt_bytes
                                                                 : internal,
                                 .type = Elem::kF32};
            } else {
                m_levels_d[l] = {.base = internal + ((l % 2 == 0) ? 0 : half),
                                 .type = types[l]};
            }
        }
    }

    template <typename Input>
    void start(const Input& input,
               const std::vector<Elem>& types,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream,
               bool fuse) {
        cuda_utils::set_device(m_device_id);
        m_stream         = stream;
        m_dmt_target_ptr = d_dmt.data();
        m_current_level  = 0;

        if (m_mode == FDMTMode::kValid) {
            // Ping-pong once per block (not per level): every level's
            // history lives in the same pair of buffers at disjoint
            // offsets, and all levels of one block must agree on which
            // buffer is "previous block" vs "this block".
            if (m_tree_history_parity) {
                m_hist_in_ptr =
                    thrust::raw_pointer_cast(m_tree_history_b_d.data());
                m_hist_out_ptr =
                    thrust::raw_pointer_cast(m_tree_history_a_d.data());
                m_level0_hist_in_ptr =
                    thrust::raw_pointer_cast(m_history_b_d.data());
                m_level0_hist_out_ptr =
                    thrust::raw_pointer_cast(m_history_a_d.data());
            } else {
                m_hist_in_ptr =
                    thrust::raw_pointer_cast(m_tree_history_a_d.data());
                m_hist_out_ptr =
                    thrust::raw_pointer_cast(m_tree_history_b_d.data());
                m_level0_hist_in_ptr =
                    thrust::raw_pointer_cast(m_history_a_d.data());
                m_level0_hist_out_ptr =
                    thrust::raw_pointer_cast(m_history_b_d.data());
            }
            m_tree_history_parity = !m_tree_history_parity;
        }

        layout_levels(types, d_dmt);
        if (fuse && m_fused) {
            launch_fused(input, stream);
            m_current_level = get_fuse_levels();
        } else {
            with_elem(m_levels_d[0].type, [&](auto tag0) {
                using T0 = decltype(tag0);
                initialise_device<Input, T0>(
                    input, reinterpret_cast<T0*>(m_levels_d[0].base), stream);
            });
        }
        m_is_initialized = true;
        spdlog::debug("FDMTCUDA: Stepper initialized at level 0.");
    }

    void execute_iter_device(SizeType next_level, cudaStream_t stream) {
        with_elem(m_levels_d[next_level - 1].type, [&](auto tag_in) {
            using TIn = decltype(tag_in);
            with_elem(m_levels_d[next_level].type, [&](auto tag_out) {
                using TOut = decltype(tag_out);
                // Levels only ever widen (see detail::int_tree_level_types).
                if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                              !(std::is_floating_point_v<TIn> &&
                                !std::is_floating_point_v<TOut>)) {
                    execute_iter_typed<TIn, TOut>(
                        reinterpret_cast<const TIn*>(
                            m_levels_d[next_level - 1].base),
                        reinterpret_cast<TOut*>(m_levels_d[next_level].base),
                        next_level, stream);
                } else {
                    throw std::logic_error(
                        "FDMTCUDA: invalid narrowing level transition");
                }
            });
        });
    }

    template <typename TIn, typename TOut>
    void execute_iter_typed(const TIn* __restrict__ in_ptr,
                            TOut* __restrict__ out_ptr,
                            SizeType next_level,
                            cudaStream_t stream) {
        const auto& shape = m_plan.get_container().state_shape[next_level];
        const int nsamps  = static_cast<int>(shape.nsamps);
        const int ncoords_sum_cur  = static_cast<int>(shape.ncoords_sum);
        const int ncoords_copy_cur = static_cast<int>(shape.ncoords_copy);

        auto coords_sum_cur  = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto coords_copy_cur = m_plan_d.coordinates_copy.get_raw_ptrs();
        coords_sum_cur.update_offsets(m_coords_sum_offsets[next_level]);
        coords_copy_cur.update_offsets(m_coords_copy_offsets[next_level]);

        const auto coords_max = std::max(ncoords_sum_cur, ncoords_copy_cur);
        const dim3 block_size = dim3(256, 1);
        const dim3 grid_size  = dim3((nsamps + block_size.x - 1) / block_size.x,
                                     coords_max, m_nbeams);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);

        // Beam b of every level lives at element offset b * get_buffer_size()
        // -- the documented beam-major layout, shared with FDMTCPU. (Using
        // each level's own nelements as the stride put the root's beams at
        // b * root nelements instead, which only matched the documented
        // b * get_buffer_size() layout when the root was the largest level.)
        const auto beam_stride         = beam_stride_elements();
        const auto in_state_nelements  = beam_stride;
        const auto out_state_nelements = beam_stride;
        const auto tree_hist_size =
            static_cast<int>(m_plan.get_tree_history_size());

        if (m_mode == FDMTMode::kFull) {
            kernel_execute_iter<FDMTMode::kFull, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, nsamps, ncoords_sum_cur, ncoords_copy_cur,
                    in_state_nelements, out_state_nelements, tree_hist_size);
        } else if (m_mode == FDMTMode::kRoll) {
            kernel_execute_iter<FDMTMode::kRoll, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, nsamps, ncoords_sum_cur, ncoords_copy_cur,
                    in_state_nelements, out_state_nelements, tree_hist_size);
        } else {
            kernel_execute_iter<FDMTMode::kValid, TIn, TOut>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, nsamps, ncoords_sum_cur, ncoords_copy_cur,
                    in_state_nelements, out_state_nelements, tree_hist_size);

            if (m_hist_out_ptr != nullptr && ncoords_sum_cur > 0) {
                const auto max_offset = static_cast<int>(shape.dt_max);
                if (max_offset > 0) {
                    const dim3 block_thist = dim3(256, 1);
                    const dim3 grid_thist =
                        dim3((max_offset + block_thist.x - 1) / block_thist.x,
                             ncoords_sum_cur, m_nbeams);
                    kernel_advance_tree_history<TIn>
                        <<<grid_thist, block_thist, 0, stream>>>(
                            in_ptr, coords_sum_cur, m_hist_in_ptr,
                            m_hist_out_ptr, ncoords_sum_cur, in_state_nelements,
                            tree_hist_size);
                    cuda_utils::check_last_cuda_error(
                        "kernel_advance_tree_history launch failed");
                }
            }
        }
        cuda_utils::check_last_cuda_error("kernel_execute_iter launch failed");
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
                static_cast<int>(plan_c.state_shape[0].nsamps),
                static_cast<int>(
                    plan_c.state_shape[m_plan.get_niters()].dt_max),
                beam_stride_elements(), m_level0_hist_in_ptr);
    }

    template <typename Input, typename TOut>
    void initialise_device(const Input& waterfall_d,
                           TOut* __restrict__ state_d,
                           cudaStream_t stream) {
        const auto& plan_c    = m_plan.get_container();
        const int nsubs       = static_cast<int>(plan_c.state_shape[0].nchans);
        const int nsamps      = static_cast<int>(plan_c.state_shape[0].nsamps);
        const dim3 block_size = dim3(1024, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, nsubs, m_nbeams);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);

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
        cuda_utils::check_last_cuda_error("kernel_init_fdmt launch failed");

        advance_level0_history(waterfall_d, stream);
        spdlog::debug("FDMTCUDA::Impl: Initialise device submitted to stream.");
    }

    // Valid mode: advance the per-channel level-0 input history by this
    // block (independent of how level 0 itself was computed).
    template <typename Input>
    void advance_level0_history(const Input& waterfall_d, cudaStream_t stream) {
        const auto& plan_c = m_plan.get_container();
        const int nsubs    = static_cast<int>(plan_c.state_shape[0].nchans);
        const int nsamps   = static_cast<int>(plan_c.state_shape[0].nsamps);
        const int dt_max_final =
            static_cast<int>(plan_c.state_shape[m_plan.get_niters()].dt_max);
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
            cuda_utils::check_last_cuda_error(
                "kernel_advance_history_window launch failed");
        }
    }
}; // End FDMTCUDA::Impl definition

FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   IndexType dt_max,
                   IndexType dt_min,
                   SizeType dt_step,
                   bool use_box_smearing,
                   std::string_view mode,
                   int verbose,
                   int device_id,
                   SizeType nbeams,
                   SizeType fuse_levels,
                   bool int_tree)
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
                                    device_id,
                                    nbeams,
                                    fuse_levels,
                                    int_tree)) {}
FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<IndexType>& dt_grid,
                   bool use_box_smearing,
                   std::string_view mode,
                   int verbose,
                   int device_id,
                   SizeType nbeams,
                   SizeType fuse_levels,
                   bool int_tree)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id,
                                    nbeams,
                                    fuse_levels,
                                    int_tree)) {}

FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<float>& dm_grid,
                   bool use_box_smearing,
                   std::string_view mode,
                   int verbose,
                   int device_id,
                   SizeType nbeams,
                   SizeType fuse_levels,
                   bool int_tree)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dm_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id,
                                    nbeams,
                                    fuse_levels,
                                    int_tree)) {}

FDMTCUDA::~FDMTCUDA()                                    = default;
FDMTCUDA::FDMTCUDA(FDMTCUDA&& other) noexcept            = default;
FDMTCUDA& FDMTCUDA::operator=(FDMTCUDA&& other) noexcept = default;
const plans::FDMTPlan& FDMTCUDA::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FDMTCUDA::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute_h(waterfall, dmt);
}
void FDMTCUDA::execute(cuda::std::span<const float> d_waterfall,
                       cuda::std::span<float> d_dmt,
                       cudaStream_t stream) {
    m_impl->execute_d(d_waterfall, d_dmt, stream);
}
void FDMTCUDA::execute(std::span<const uint8_t> waterfall_packed,
                       SizeType nbits,
                       std::span<float> dmt) {
    m_impl->execute_h(waterfall_packed, nbits, dmt);
}
void FDMTCUDA::execute(cuda::std::span<const uint8_t> d_waterfall_packed,
                       SizeType nbits,
                       cuda::std::span<float> d_dmt,
                       cudaStream_t stream) {
    m_impl->execute_d(d_waterfall_packed, nbits, d_dmt, stream);
}
void FDMTCUDA::reset(cuda::std::span<const uint8_t> d_waterfall_packed,
                     SizeType nbits,
                     cuda::std::span<float> d_dmt,
                     cudaStream_t stream) {
    m_impl->reset(d_waterfall_packed, nbits, d_dmt, stream);
}
SizeType FDMTCUDA::get_fuse_levels() const noexcept {
    return m_impl->get_fuse_levels();
}
bool FDMTCUDA::get_int_tree() const noexcept { return m_impl->get_int_tree(); }
FDMTMemoryUsage FDMTCUDA::get_memory_usage() const noexcept {
    return m_impl->get_memory_usage();
}
void FDMTCUDA::reset(cuda::std::span<const float> d_waterfall,
                     cuda::std::span<float> d_dmt,
                     cudaStream_t stream) {
    m_impl->reset(d_waterfall, d_dmt, stream);
}
void FDMTCUDA::advance(SizeType levels, cudaStream_t stream) {
    m_impl->advance(levels, stream);
}
void FDMTCUDA::advance_until_remaining(SizeType remaining_levels,
                                       cudaStream_t stream) {
    m_impl->advance_until_remaining(remaining_levels, stream);
}
cuda::std::span<const float> FDMTCUDA::view_level_data() const {
    return m_impl->view_level_data();
}
cuda::std::span<const float>
FDMTCUDA::view_subband_data(SizeType subband_idx) const {
    return m_impl->view_subband_data(subband_idx);
}
FDMTSubbandViewCUDA FDMTCUDA::view_subband(SizeType subband_idx) const {
    return m_impl->view_subband(subband_idx);
}
SizeType FDMTCUDA::current_level() const noexcept {
    return m_impl->current_level();
}
SizeType FDMTCUDA::total_levels() const noexcept {
    return m_impl->total_levels();
}
SizeType FDMTCUDA::remaining_levels() const noexcept {
    return m_impl->remaining_levels();
}
SizeType FDMTCUDA::num_subbands() const { return m_impl->num_subbands(); }
bool FDMTCUDA::is_finished() const noexcept { return m_impl->is_finished(); }
void FDMTCUDA::finalize(cudaStream_t stream) { m_impl->finalize(stream); }
float FDMTCUDA::get_effective_variance(SizeType dm_idx,
                                       SizeType boxcar_width) const {
    return m_impl->get_effective_variance(dm_idx, boxcar_width);
}
float FDMTCUDA::get_effective_sigma(SizeType dm_idx,
                                    SizeType boxcar_width) const {
    return m_impl->get_effective_sigma(dm_idx, boxcar_width);
}
std::vector<float>
FDMTCUDA::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_variance_grid(boxcar_width);
}
std::vector<float>
FDMTCUDA::get_effective_sigma_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_sigma_grid(boxcar_width);
}
void FDMTCUDA::reset_history() noexcept { m_impl->reset_history(); }
SizeType FDMTCUDA::get_nbeams() const noexcept { return m_impl->get_nbeams(); }
SizeType FDMTCUDA::history_state_size() const noexcept {
    return m_impl->history_state_size();
}
void FDMTCUDA::save_history(cuda::std::span<float> out,
                            cudaStream_t stream) const {
    m_impl->save_history(out, stream);
}
void FDMTCUDA::load_history(cuda::std::span<const float> in,
                            cudaStream_t stream) {
    m_impl->load_history(in, stream);
}

[[nodiscard]] std::vector<float>
compute_fdmt_cuda(std::span<const float> waterfall,
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
                  int verbose,
                  int device_id,
                  SizeType nbeams) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                  use_box_smearing, mode, verbose, device_id, nbeams);
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
    return dmt_out;
}

[[nodiscard]] std::vector<float>
compute_fdmt_cuda(std::span<const float> waterfall,
                  float f_min,
                  float f_max,
                  SizeType nchans,
                  SizeType nsamps,
                  float tsamp,
                  const std::vector<IndexType>& dt_grid,
                  bool use_box_smearing,
                  std::string_view mode,
                  int verbose,
                  int device_id,
                  SizeType nbeams) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid,
                  use_box_smearing, mode, verbose, device_id, nbeams);
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
    return dmt_out;
}

[[nodiscard]] std::vector<float>
compute_fdmt_cuda(std::span<const float> waterfall,
                  float f_min,
                  float f_max,
                  SizeType nchans,
                  SizeType nsamps,
                  float tsamp,
                  const std::vector<float>& dm_grid,
                  bool use_box_smearing,
                  std::string_view mode,
                  int verbose,
                  int device_id,
                  SizeType nbeams) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid,
                  use_box_smearing, mode, verbose, device_id, nbeams);
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
    return dmt_out;
}

} // namespace dmt::algorithms
