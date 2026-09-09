#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms {

namespace {
__global__ void
kernel_init_fdmt(const float* __restrict__ waterfall,
                 float* __restrict__ state,
                 const int* __restrict__ grids0_dt_grid_ptr,
                 const int* __restrict__ grids0_ndt_ptr,
                 const int* __restrict__ grids0_coord_offset_ptr,
                 int nsubs,
                 int nsamps,
                 int dt_max,
                 const float* __restrict__ hist) {
    auto isamp = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    auto i_sub = static_cast<int>(blockIdx.y);
    if (i_sub >= nsubs || isamp >= nsamps) {
        return;
    }
    const auto* dt_grid_sub =
        &grids0_dt_grid_ptr[grids0_coord_offset_ptr[i_sub]];
    const auto buffer_offset    = grids0_coord_offset_ptr[i_sub] * nsamps;
    const auto ndt_grid_sub     = grids0_ndt_ptr[i_sub];
    const auto waterfall_offset = i_sub * nsamps;
    const auto hist_offset      = i_sub * dt_max;
    const auto dt_min           = dt_grid_sub[0];

    // Initialise state for [:, dt_init_min, dt_init_min:]
    if (isamp >= dt_min) {
        float sum = 0.0F;
        for (int i = isamp - dt_min; i <= isamp; ++i) {
            sum += waterfall[waterfall_offset + i];
        }
        state[buffer_offset + isamp] = sum / static_cast<float>(dt_min + 1);
    }
    for (int i_dt = 1; i_dt < ndt_grid_sub; ++i_dt) {
        const auto dt_cur            = dt_grid_sub[i_dt];
        const auto dt_prev           = dt_grid_sub[i_dt - 1];
        const auto state_offset_cur  = buffer_offset + (i_dt * nsamps);
        const auto state_offset_prev = buffer_offset + ((i_dt - 1) * nsamps);

        // Initialise state for [i_sub, i_dt, dt_cur:]
        if (isamp >= dt_cur) {
            float sum = 0.0F;
            for (int i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                sum += waterfall[waterfall_offset + i];
            }
            state[state_offset_cur + isamp] =
                (state[state_offset_prev + isamp] *
                     static_cast<float>(dt_prev + 1) +
                 sum) /
                static_cast<float>(dt_cur + 1);
        }
        // Initialise state for [i_sub, i_dt, 0:dt_cur]
        else {
            float sum       = 0.0F;
            int i_start_rel = isamp - dt_cur;
            int i_end_rel   = isamp - dt_prev;
            // Sum from history buffer if needed and available
            if (hist != nullptr) { // Check if history pointer is valid
                for (int i_rel = i_start_rel; i_rel < 0 && i_rel < i_end_rel;
                     ++i_rel) {
                    // Access history using relative index
                    sum += hist[hist_offset + (dt_max + i_rel)];
                }
            }
            // Sum from waterfall buffer for the remaining part of the window
            for (int i_rel = max(i_start_rel, 0); i_rel < i_end_rel; ++i_rel) {
                // Access waterfall using absolute index
                sum += waterfall[waterfall_offset + i_rel];
            }
            state[state_offset_cur + isamp] =
                (state[state_offset_prev + isamp] *
                     static_cast<float>(dt_prev + 1) +
                 sum) /
                static_cast<float>(dt_cur + 1);
        }
    }
}

__global__ void kernel_execute_iter(const float* __restrict__ state_in,
                                    float* __restrict__ state_out,
                                    const plans::FDMTCoordDPtrs coords_sum,
                                    const plans::FDMTCoordDPtrs coords_copy,
                                    int nsamps,
                                    int ncoords_sum_cur,
                                    int ncoords_copy_cur) {
    auto isamp   = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    auto i_coord = static_cast<int>(blockIdx.y);
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

        // Check if current sample contributes to the output for this coordinate
        if (isamp < nsamps_out) {
            float tail_val = 0.0F;
            float head_val = 0.0F;
            // Read from tail buffer if relevant
            if (isamp < nsamps_tail) {
                tail_val = state_in[tail_idx_base + isamp];
            }
            // Read from head buffer if relevant (apply offset)
            if (isamp >= offset && isamp < (nsamps_tail + offset)) {
                head_val = state_in[head_idx_base + (isamp - offset)];
            }
            // Write sum to output (handle boundary conditions implicitly via
            // reads)
            state_out[out_idx_base + isamp] = tail_val + head_val;
        }
    }
    //__syncthreads();

    if (i_coord < ncoords_copy_cur) {
        const auto nsamps_tail = coords_copy.tail_nsamps[i_coord];
        if (isamp < nsamps_tail) {
            const int out_idx_base  = coords_copy.buf_offset[i_coord];
            const int tail_idx_base = coords_copy.tail_buf_offset[i_coord];
            state_out[out_idx_base + isamp] = state_in[tail_idx_base + isamp];
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
         SizeType dt_max,
         SizeType dt_step,
         SizeType dt_min,
         bool use_history,
         bool verbose,
         int device_id)
        : m_device_id(device_id),
          m_use_history(use_history),
          m_plan(f_min,
                 f_max,
                 nchans,
                 nsamps,
                 tsamp,
                 dt_max,
                 dt_step,
                 dt_min,
                 verbose) {
        cuda_utils::set_device(m_device_id);
        spdlog::debug("FDMTCUDA::Impl: Set device to {}", m_device_id);
        // Allocate memory for the state buffers
        m_state_in_d.resize(m_plan.get_buffer_size(), 0.0F);
        m_state_out_d.resize(m_plan.get_buffer_size(), 0.0F);
        // Only allocate history buffer if needed
        if (m_use_history) {
            m_history_d.resize(m_plan.get_history_size(), 0.0F);
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

    // Host execute: handles HtoD copy, calls device execute, handles DtoH copy
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

        // Synchronize stream to ensure copies and kernel are complete
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "execute_h: cudaStreamSynchronize failed");

        spdlog::debug("FDMTCUDA::Impl: Host execution complete.");
    }

    // Device execute: calls reset, advances to completion, finalizes
    void execute_d(cuda::std::span<const float> waterfall_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        reset(waterfall_d, dmt_d, stream);
        advance_until_remaining(0, stream);
        finalize(stream);
        spdlog::debug("FDMTCUDA::Impl: Device execution complete on stream");
    }

    // Stepper API implementation
    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr) {
        check_inputs(d_waterfall.size(), d_dmt.size());
        cuda_utils::set_device(m_device_id);
        m_stream        = stream;
        m_dmt_user_ptr  = d_dmt.data();
        m_dmt_user_size = d_dmt.size();
        m_current_level = 0;

        const auto niters = m_plan.get_niters();
        if (niters == 0) {
            m_current_in_ptr  = m_dmt_user_ptr;
            m_current_out_ptr = m_dmt_user_ptr;
            initialise_device(d_waterfall.data(), m_current_in_ptr, stream);
        } else {
            m_current_in_ptr  = m_state_in_d.data().get();
            m_current_out_ptr = m_state_out_d.data().get();
            initialise_device(d_waterfall.data(), m_current_in_ptr, stream);
        }

        m_is_initialized = true;
        spdlog::debug("FDMTCUDA::Impl: Stepper initialized at level 0.");
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
            execute_iter_device(next_level, active_stream);
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
        const auto& grids = m_plan.get_container().grids[m_current_level];
        if (subband_idx >= grids.size()) {
            throw std::out_of_range(
                std::format("FDMTCUDA: Subband index {} out of range (current "
                            "level has {} subbands)",
                            subband_idx, grids.size()));
        }
        const auto& grid = grids[subband_idx];
        const auto& state_shape =
            m_plan.get_container().state_shape[m_current_level];
        const auto offset = grid.coord_offset * state_shape.nsamps;
        const auto count  = grid.ndt * state_shape.nsamps;
        return cuda::std::span<const float>(m_current_in_ptr + offset, count);
    }

    [[nodiscard]] FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        const auto& grids = m_plan.get_container().grids[m_current_level];
        if (subband_idx >= grids.size()) {
            throw std::out_of_range(
                std::format("FDMTCUDA: Subband index {} out of range (current "
                            "level has {} subbands)",
                            subband_idx, grids.size()));
        }
        const auto& grid = grids[subband_idx];
        const auto& state_shape =
            m_plan.get_container().state_shape[m_current_level];
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
            .dt_grid     = std::span<const SizeType>(grid.dt_grid.data(),
                                                     grid.dt_grid.size())};
    }

    [[nodiscard]] SizeType current_level() const noexcept {
        return m_current_level;
    }

    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan.get_niters() + 1;
    }

    [[nodiscard]] SizeType remaining_levels() const noexcept {
        const auto total_lvl = total_levels();
        return (m_current_level >= total_lvl - 1)
                   ? 0
                   : (total_lvl - 1 - m_current_level);
    }

    [[nodiscard]] SizeType num_subbands() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }

    [[nodiscard]] bool is_finished() const noexcept {
        return m_current_level >= total_levels() - 1;
    }

    void finalize(cudaStream_t stream = nullptr) {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTCUDA: Stepper is not initialized. Call reset() first.");
        }
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
    }

private:
    int m_device_id;
    bool m_use_history;
    plans::FDMTPlan m_plan;
    plans::FDMTPlanContainerD m_plan_d;
    // State buffers
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;
    thrust::device_vector<float> m_history_d;

    // Stepper state
    bool m_is_initialized{false};
    SizeType m_current_level{0};
    float* m_current_in_ptr{nullptr};
    float* m_current_out_ptr{nullptr};
    float* m_dmt_user_ptr{nullptr};
    SizeType m_dmt_user_size{0};
    cudaStream_t m_stream{nullptr};
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != nchans * nsamps) {
            throw std::invalid_argument(
                std::format("FDMTCUDA::Impl: Invalid size of waterfall. "
                            "Expected {}, got {}",
                            nchans * nsamps, waterfall_size));
        }
        if (dmt_size < m_plan.get_dmt_size()) {
            throw std::invalid_argument(
                std::format("FDMTCUDA::Impl: Invalid size of dmt. Expected at "
                            "least {}, got {}",
                            m_plan.get_dmt_size(), dmt_size));
        }
        spdlog::debug("FDMTCUDA::Impl: Input dimensions check passed: {}x{}",
                      nchans, nsamps);
    }

    void execute_iter_device(SizeType next_level, cudaStream_t stream) {
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

        const auto niters = m_plan.get_niters();
        float* current_out_ptr =
            (next_level == niters) ? m_dmt_user_ptr : m_current_out_ptr;

        kernel_execute_iter<<<grid_size, block_size, 0, stream>>>(
            m_current_in_ptr, current_out_ptr, coords_sum_cur, coords_copy_cur,
            nsamps, ncoords_sum_cur, ncoords_copy_cur);
        cuda_utils::check_last_cuda_error("kernel_execute_iter launch failed");

        if (next_level == niters) {
            m_current_in_ptr = m_dmt_user_ptr;
        } else {
            std::swap(m_current_in_ptr, m_current_out_ptr);
        }
        m_current_level = next_level;
    }

    void initialise_device(const float* __restrict__ waterfall_d,
                           float* __restrict__ state_d,
                           cudaStream_t stream) {
        const int nsubs  = m_plan_d.state_shape.nchans[0];
        const int nsamps = m_plan_d.state_shape.nsamps[0];
        const int dt_max = m_plan_d.state_shape.dt_max[0];

        // Get raw pointers to device plan data needed by the kernel
        // Ensure these pointers within m_plan_d are valid device pointers
        const int* grids0_dt_grid_ptr = m_plan_d.grids0.dt_grid.data().get();
        const int* grids0_ndt_ptr     = m_plan_d.grids0.ndt.data().get();
        const int* grids0_coord_offset_ptr =
            m_plan_d.grids0.coord_offset.data().get();
        float* hist_ptr = (m_use_history && !m_history_d.empty())
                              ? m_history_d.data().get()
                              : nullptr;

        const dim3 block_size = dim3(1024, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, nsubs);
        cuda_utils::check_kernel_launch_params(grid_size, block_size);

        // Launch kernel for initialisation
        kernel_init_fdmt<<<grid_size, block_size, 0, stream>>>(
            waterfall_d, state_d, grids0_dt_grid_ptr, grids0_ndt_ptr,
            grids0_coord_offset_ptr, nsubs, nsamps, dt_max, hist_ptr);
        cuda_utils::check_last_cuda_error("kernel_init_fdmt launch failed");

        // Update history buffer if enabled
        if (m_use_history) {
            // Copy the last nchans x dt_max elements from waterfall to hist
            for (int i_sub = 0; i_sub < nsubs; ++i_sub) {
                thrust::copy_n(
                    &waterfall_d[(i_sub * nsamps) + nsamps - dt_max], dt_max,
                    &hist_ptr[static_cast<ptrdiff_t>(i_sub * dt_max)]);
            }
            cuda_utils::check_last_cuda_error(
                "History update cudaMemcpyAsync failed");
        }
        spdlog::debug("FDMTCUDA::Impl: Initialise device submitted to stream.");
    }
}; // End FDMTCUDA::Impl definition

FDMTCUDA::FDMTCUDA(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   SizeType dt_max,
                   SizeType dt_step,
                   SizeType dt_min,
                   bool use_history,
                   bool verbose,
                   int device_id)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_step,
                                    dt_min,
                                    use_history,
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

std::vector<float> compute_fdmt_cuda(std::span<const float> waterfall,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     SizeType nsamps,
                                     float tsamp,
                                     SizeType dt_max,
                                     SizeType dt_step,
                                     SizeType dt_min,
                                     bool verbose,
                                     int device_id) {
    FDMTCUDA fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min,
                  false, verbose, device_id);
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size(), 0.0F);
    fdmt.execute(waterfall, dmt);
    return dmt;
}
} // namespace dmt::algorithms
