#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms {

namespace {

enum class FDMTMode : uint8_t {
    kFull  = 0,
    kValid = 1,
    kRoll  = 2,
};

FDMTMode parse_mode(std::string_view mode) {
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
template <FDMTMode Mode, bool UseBoxSmearing>
__global__ void
kernel_init_fdmt(const float* __restrict__ waterfall,
                 float* __restrict__ state,
                 const int* __restrict__ grids0_dt_grid_ptr,
                 const int* __restrict__ grids0_ndt_ptr,
                 const int* __restrict__ grids0_coord_offset_ptr,
                 int nsubs,
                 int nsamps,
                 int dt_max_final,
                 const float* __restrict__ hist) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub = static_cast<int>(blockIdx.y);
    if (i_sub >= nsubs || isamp >= nsamps) {
        return;
    }

    const auto* dt_grid_sub =
        &grids0_dt_grid_ptr[grids0_coord_offset_ptr[i_sub]];
    const auto buffer_offset    = grids0_coord_offset_ptr[i_sub] * nsamps;
    const auto ndt_grid_sub     = grids0_ndt_ptr[i_sub];
    const auto waterfall_offset = i_sub * nsamps;
    const auto hist_offset      = i_sub * dt_max_final;
    const auto dt_first         = dt_grid_sub[0];
    const auto dt_last          = dt_grid_sub[ndt_grid_sub - 1];

    // Lambda to fetch sample at relative index (isamp - shift), shift >= 0.
    auto get_sample = [&](int t) -> float {
        if (t >= 0) {
            return waterfall[waterfall_offset + t];
        }
        if constexpr (Mode == FDMTMode::kRoll) {
            return waterfall[waterfall_offset + (t + nsamps)];
        } else if constexpr (Mode == FDMTMode::kValid) {
            return (hist != nullptr) ? hist[hist_offset + dt_max_final + t]
                                     : 0.0F;
        } else {
            return 0.0F; // kFull: zero-padded before t=0
        }
    };

    // Writes `val` into every dt row whose value has magnitude `s` (there
    // are up to two: +s and -s, whichever are actually present in this
    // dense sub-band grid).
    auto write_matches = [&](int s, float val) {
        if (s >= dt_first && s <= dt_last) {
            state[buffer_offset + ((s - dt_first) * nsamps) + isamp] = val;
        }
        const int neg = -s;
        if (s != 0 && neg >= dt_first && neg <= dt_last) {
            state[buffer_offset + ((neg - dt_first) * nsamps) + isamp] = val;
        }
    };

    // Smallest and largest |dt| actually present. A dense range either
    // straddles (or touches) zero -- smallest magnitude 0 -- or is purely
    // one-signed, in which case the smallest magnitude sits at whichever
    // end is closest to zero.
    const int s_lo = (dt_last >= 0) ? 0 : min(abs(dt_first), abs(dt_last));
    const int s_hi = max(abs(dt_first), abs(dt_last));

    if constexpr (UseBoxSmearing) {
        float prev_val;
        if (s_lo == 0) {
            // O(1) fast-path: zero delay across leaf channel (most common case)
            prev_val = get_sample(isamp);
        } else {
            // Unrolled coalesced summation loop seeding the box-sum at the
            // smallest magnitude present.
            float sum = 0.0F;
#pragma unroll 4
            for (int d = 0; d <= s_lo; ++d) {
                sum += get_sample(isamp - d);
            }
            prev_val = sum;
        }
        write_matches(s_lo, prev_val);
        for (int s = s_lo + 1; s <= s_hi; ++s) {
            const float cur_val = prev_val + get_sample(isamp - s);
            write_matches(s, cur_val);
            prev_val = cur_val;
        }
    } else {
        // No smearing: each row is an independent shift by |dt| samples, no
        // accumulation between rows.
        for (int s = s_lo; s <= s_hi; ++s) {
            write_matches(s, get_sample(isamp - s));
        }
    }
}

/**
 * @brief Per-level tree merge kernel. `kValid`'s cross-block history mirrors
 * `offset_add<kValid>` in fdmt.cpp (see its doc comment) with one CUDA-
 * specific twist: the CPU version reads its whole boundary region and then
 * overwrites the history buffer within one sequential per-coordinate thread,
 * so there's no hazard. Here, `isamp` is itself parallelized *within* one
 * coordinate, so a thread reading `hist_in[...]` (this coordinate's boundary,
 * from the previous block) and another thread of the same coordinate writing
 * `hist_out[...]` (capturing this block's own tail for the next block) would
 * race if `hist_in`/`hist_out` aliased the same buffer. They never do:
 * `FDMTCUDA::Impl::reset()` ping-pongs two same-sized device buffers once per
 * block, so within a single launch `hist_in` and `hist_out` are always
 * distinct arrays.
 */
template <FDMTMode Mode>
__global__ void kernel_execute_iter(const float* __restrict__ state_in,
                                    float* __restrict__ state_out,
                                    const plans::FDMTCoordDPtrs coords_sum,
                                    const plans::FDMTCoordDPtrs coords_copy,
                                    const float* __restrict__ hist_in,
                                    float* __restrict__ hist_out,
                                    int nsamps,
                                    int ncoords_sum_cur,
                                    int ncoords_copy_cur) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_coord = static_cast<int>(blockIdx.y);
    if (isamp >= nsamps) {
        return;
    }

    if (i_coord < ncoords_sum_cur) {
        const auto nsamps_out    = coords_sum.nsamps[i_coord];
        const auto offset        = coords_sum.offset[i_coord];
        const auto nsamps_tail   = coords_sum.tail_nsamps[i_coord];
        const auto out_idx_base  = coords_sum.buf_offset[i_coord];
        const auto tail_idx_base = coords_sum.tail_buf_offset[i_coord];
        const auto head_idx_base = coords_sum.head_buf_offset[i_coord];
        const auto hist_off      = coords_sum.hist_offset[i_coord];

        if (isamp < nsamps_out) {
            float tail_val = 0.0F;
            float head_val = 0.0F;
            if constexpr (Mode == FDMTMode::kFull) {
                if (isamp < nsamps_tail) {
                    tail_val = state_in[tail_idx_base + isamp];
                }
                if (isamp >= offset && (isamp - offset) < nsamps_tail) {
                    head_val = state_in[head_idx_base + (isamp - offset)];
                }
            } else if constexpr (Mode == FDMTMode::kValid) {
                tail_val = state_in[tail_idx_base + isamp];
                if (isamp >= offset) {
                    head_val = state_in[head_idx_base + (isamp - offset)];
                } else if (hist_in != nullptr) {
                    // Boundary sample: use the previous block's trailing
                    // head samples, captured below the last time this
                    // coordinate ran.
                    head_val = hist_in[hist_off + isamp];
                }
                if (hist_out != nullptr && offset > 0 &&
                    isamp >= nsamps_tail - offset) {
                    // Capture this block's own trailing `offset` samples of
                    // head for the next call to this coordinate.
                    hist_out[hist_off + (isamp - (nsamps_tail - offset))] =
                        state_in[head_idx_base + isamp];
                }
            } else if constexpr (Mode == FDMTMode::kRoll) {
                tail_val            = state_in[tail_idx_base + isamp];
                const int head_samp = (isamp >= offset)
                                          ? (isamp - offset)
                                          : (nsamps_tail - offset + isamp);
                head_val            = state_in[head_idx_base + head_samp];
            }
            state_out[out_idx_base + isamp] = tail_val + head_val;
        }
    }

    if (i_coord < ncoords_copy_cur) {
        const auto nsamps_out  = coords_copy.nsamps[i_coord];
        const auto nsamps_tail = coords_copy.tail_nsamps[i_coord];
        const int out_idx_base = coords_copy.buf_offset[i_coord];
        if (isamp < nsamps_tail) {
            const int tail_idx_base = coords_copy.tail_buf_offset[i_coord];
            state_out[out_idx_base + isamp] = state_in[tail_idx_base + isamp];
        } else if (isamp < nsamps_out) {
            state_out[out_idx_base + isamp] = 0.0F;
        }
    }
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
         bool verbose,
         int device_id)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
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
        init_device_structures();
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
         int device_id)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, verbose) {
        init_device_structures();
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
         int device_id)
        : m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, verbose) {
        init_device_structures();
    }

    void init_device_structures() {
        cuda_utils::set_device(m_device_id);
        spdlog::debug("FDMTCUDA::Impl: Set device to {}", m_device_id);
        // Allocate single internal state buffer on device for ping-pong
        m_state_internal_d.resize(m_plan.get_buffer_size(), 0.0F);
        if (m_mode == FDMTMode::kValid) {
            m_history_d.resize(m_plan.get_history_size(), 0.0F);
            // Two same-sized tree-history buffers, ping-ponged once per
            // block in reset() -- see kernel_execute_iter's doc comment for
            // why one buffer isn't safe here the way it is on the CPU.
            const auto tree_hist_size = m_plan.get_tree_history_size();
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
        reset(waterfall_d, dmt_d, stream);
        advance_until_remaining(0, stream);
        finalize(stream);
        spdlog::debug("FDMTCUDA::Impl: Device execution complete on stream");
    }

    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr) {
        check_inputs(d_waterfall.size(), d_dmt.size());
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
            } else {
                m_hist_in_ptr =
                    thrust::raw_pointer_cast(m_tree_history_a_d.data());
                m_hist_out_ptr =
                    thrust::raw_pointer_cast(m_tree_history_b_d.data());
            }
            m_tree_history_parity = !m_tree_history_parity;
        }

        const SizeType internal_iters = total_levels() - 2;
        const bool odd_swaps          = (internal_iters % 2) == 1;
        if (odd_swaps) {
            m_current_in_ptr = d_dmt.data();
            m_current_out_ptr =
                thrust::raw_pointer_cast(m_state_internal_d.data());
        } else {
            m_current_in_ptr =
                thrust::raw_pointer_cast(m_state_internal_d.data());
            m_current_out_ptr = d_dmt.data();
        }

        initialise_device(d_waterfall.data(), m_current_in_ptr, stream);
        m_is_initialized = true;
        spdlog::debug("FDMTCUDA: Stepper initialized at level 0.");
    }

    void advance(SizeType levels = 1, cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        cuda_utils::set_device(m_device_id);
        cudaStream_t active_stream = stream ? stream : m_stream;
        const auto total_lvl       = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            execute_iter_device(m_current_in_ptr, m_current_out_ptr, next_level,
                                active_stream);
            std::swap(m_current_in_ptr, m_current_out_ptr);
            m_current_level = next_level;
            --levels;
        }
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
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] cuda::std::span<const float> view_level_data() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& state_shape =
            m_plan.get_container().state_shape[m_current_level];
        return cuda::std::span<const float>(m_current_in_ptr,
                                            state_shape.nelements);
    }

    [[nodiscard]] cuda::std::span<const float>
    view_subband_data(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(
                std::format("FDMTCUDA: Subband index {} out of range (current "
                            "level has {} subbands)",
                            subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return cuda::std::span<const float>(m_current_in_ptr + offset, count);
    }

    [[nodiscard]] FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& plan_c      = m_plan.get_container();
        const auto& state_shape = plan_c.state_shape[m_current_level];
        if (subband_idx >= state_shape.nchans) {
            throw std::out_of_range(
                std::format("FDMTCUDA: Subband index {} out of range (current "
                            "level has {} subbands)",
                            subband_idx, state_shape.nchans));
        }
        const auto& grid  = plan_c.grids[m_current_level][subband_idx];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return FDMTSubbandViewCUDA{
            .data =
                cuda::std::span<const float>(m_current_in_ptr + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = state_shape.nsamps,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size())};
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
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().state_shape[m_current_level].nchans;
    }

    [[nodiscard]] bool is_finished() const noexcept {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        m_is_initialized = false;
    }

    // Analytical variance/sigma: pure plan-side math (see
    // plans::FDMTPlan::get_effective_variance), identical on CPU and CUDA --
    // provided here for API parity with FDMTCPU.
    [[nodiscard]] float
    get_effective_variance(SizeType dm_idx, SizeType boxcar_width) const {
        return m_plan.get_effective_variance(dm_idx, boxcar_width,
                                             m_use_box_smearing);
    }
    [[nodiscard]] float
    get_effective_sigma(SizeType dm_idx, SizeType boxcar_width) const {
        return m_plan.get_effective_sigma(dm_idx, boxcar_width,
                                          m_use_box_smearing);
    }
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width) const {
        return m_plan.get_effective_variance_grid(boxcar_width, m_use_box_smearing);
    }
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width) const {
        return m_plan.get_effective_sigma_grid(boxcar_width, m_use_box_smearing);
    }

    /// Zeroes all cross-block history (level-0 and tree-level) so the next
    /// execute()/reset() call starts as if this were a brand-new instance.
    /// Mirrors FDMTCPU::reset_history().
    void reset_history() noexcept {
        cuda_utils::set_device(m_device_id);
        if (!m_history_d.empty()) {
            m_history_d.assign(m_history_d.size(), 0.0F);
        }
        if (!m_tree_history_a_d.empty()) {
            m_tree_history_a_d.assign(m_tree_history_a_d.size(), 0.0F);
        }
        if (!m_tree_history_b_d.empty()) {
            m_tree_history_b_d.assign(m_tree_history_b_d.size(), 0.0F);
        }
        m_tree_history_parity = false;
    }

private:
    int m_device_id;
    bool m_use_box_smearing;
    FDMTMode m_mode;
    plans::FDMTPlan m_plan;
    plans::FDMTPlanContainerD m_plan_d;
    // Internal state buffer on device (for ping-pong buffering)
    thrust::device_vector<float> m_state_internal_d;
    // Level-0 history buffer for valid-mode streaming across FDMT blocks
    thrust::device_vector<float> m_history_d;
    // Tree-level (level >= 1) cross-block history: two same-sized buffers
    // ping-ponged once per block by reset(); see kernel_execute_iter's doc
    // comment for why a single buffer (as used on the CPU) isn't safe here.
    thrust::device_vector<float> m_tree_history_a_d;
    thrust::device_vector<float> m_tree_history_b_d;
    bool m_tree_history_parity{false};
    float* m_hist_in_ptr{nullptr};
    float* m_hist_out_ptr{nullptr};

    // Stepper state
    bool m_is_initialized{false};
    SizeType m_current_level{0};
    float* m_current_in_ptr{nullptr};
    float* m_current_out_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
    cudaStream_t m_stream{nullptr};
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != nchans * nsamps) {
            throw std::invalid_argument(
                std::format("FDMTCUDA: Invalid size of waterfall. "
                            "Expected {}, got {}",
                            nchans * nsamps, waterfall_size));
        }
        if (dmt_size < m_plan.get_buffer_size()) {
            throw std::invalid_argument(
                std::format("FDMTCUDA: Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_plan.get_buffer_size(), dmt_size));
        }
        spdlog::debug("FDMTCUDA: Input dimensions check passed: {}x{}", nchans,
                      nsamps);
    }

    void execute_iter_device(const float* __restrict__ in_ptr,
                             float* __restrict__ out_ptr,
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
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, coords_max);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);

        if (m_mode == FDMTMode::kFull) {
            kernel_execute_iter<FDMTMode::kFull>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, m_hist_out_ptr, nsamps, ncoords_sum_cur,
                    ncoords_copy_cur);
        } else if (m_mode == FDMTMode::kRoll) {
            kernel_execute_iter<FDMTMode::kRoll>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, m_hist_out_ptr, nsamps, ncoords_sum_cur,
                    ncoords_copy_cur);
        } else {
            kernel_execute_iter<FDMTMode::kValid>
                <<<grid_size, block_size, 0, stream>>>(
                    in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                    m_hist_in_ptr, m_hist_out_ptr, nsamps, ncoords_sum_cur,
                    ncoords_copy_cur);
        }
        cuda_utils::check_last_cuda_error("kernel_execute_iter launch failed");
    }

    void initialise_device(const float* __restrict__ waterfall_d,
                           float* __restrict__ state_d,
                           cudaStream_t stream) {
        const auto& plan_c = m_plan.get_container();
        const int nsubs    = static_cast<int>(plan_c.state_shape[0].nchans);
        const int nsamps   = static_cast<int>(plan_c.state_shape[0].nsamps);
        const int dt_max_final =
            static_cast<int>(plan_c.state_shape[m_plan.get_niters()].dt_max);

        const int* grids0_dt_grid_ptr = m_plan_d.grids0.dt_grid.data().get();
        const int* grids0_ndt_ptr     = m_plan_d.grids0.ndt.data().get();
        const int* grids0_coord_offset_ptr =
            m_plan_d.grids0.coord_offset.data().get();
        float* hist_ptr = (m_mode == FDMTMode::kValid && !m_history_d.empty())
                              ? thrust::raw_pointer_cast(m_history_d.data())
                              : nullptr;

        const dim3 block_size = dim3(1024, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, nsubs);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);

        if (m_mode == FDMTMode::kFull) {
            if (m_use_box_smearing) {
                kernel_init_fdmt<FDMTMode::kFull, true>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            } else {
                kernel_init_fdmt<FDMTMode::kFull, false>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            }
        } else if (m_mode == FDMTMode::kRoll) {
            if (m_use_box_smearing) {
                kernel_init_fdmt<FDMTMode::kRoll, true>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            } else {
                kernel_init_fdmt<FDMTMode::kRoll, false>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            }
        } else {
            if (m_use_box_smearing) {
                kernel_init_fdmt<FDMTMode::kValid, true>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            } else {
                kernel_init_fdmt<FDMTMode::kValid, false>
                    <<<grid_size, block_size, 0, stream>>>(
                        waterfall_d, state_d, grids0_dt_grid_ptr,
                        grids0_ndt_ptr, grids0_coord_offset_ptr, nsubs, nsamps,
                        dt_max_final, hist_ptr);
            }
        }
        cuda_utils::check_last_cuda_error("kernel_init_fdmt launch failed");

        if (m_mode == FDMTMode::kValid && hist_ptr != nullptr &&
            dt_max_final > 0) {
            cudaMemcpy2DAsync(hist_ptr, dt_max_final * sizeof(float),
                              waterfall_d + nsamps - dt_max_final,
                              nsamps * sizeof(float),
                              dt_max_final * sizeof(float), nsubs,
                              cudaMemcpyDeviceToDevice, stream);
            cuda_utils::check_last_cuda_error(
                "History update cudaMemcpy2DAsync failed");
        }
        spdlog::debug("FDMTCUDA::Impl: Initialise device submitted to stream.");
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
                   bool verbose,
                   int device_id)
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
                                    device_id)) {}
FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<IndexType>& dt_grid,
                   bool use_box_smearing,
                   std::string_view mode,
                   bool verbose,
                   int device_id)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id)) {}

FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<float>& dm_grid,
                   bool use_box_smearing,
                   std::string_view mode,
                   bool verbose,
                   int device_id)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dm_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id)) {}

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
                  bool verbose,
                  int device_id) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                  use_box_smearing, mode, verbose, device_id);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    std::vector<float> dmt(fdmt_plan.get_buffer_size(), 0.0F);
    fdmt.execute(waterfall, dmt);
    dmt.resize(fdmt_plan.get_dmt_size());
    return dmt;
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
                  bool verbose,
                  int device_id) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid,
                  use_box_smearing, mode, verbose, device_id);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    std::vector<float> dmt(fdmt_plan.get_buffer_size(), 0.0F);
    fdmt.execute(waterfall, dmt);
    dmt.resize(fdmt_plan.get_dmt_size());
    return dmt;
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
                  bool verbose,
                  int device_id) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid,
                  use_box_smearing, mode, verbose, device_id);
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    std::vector<float> dmt(fdmt_plan.get_buffer_size(), 0.0F);
    fdmt.execute(waterfall, dmt);
    dmt.resize(fdmt_plan.get_dmt_size());
    return dmt;
}

} // namespace dmt::algorithms
