#include "dmt/fdmt.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <utility>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/common/plans_cuda.hpp"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"

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
                                    const FDMTCoordDPtrs coords_sum,
                                    const FDMTCoordDPtrs coords_copy,
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

namespace dmt {

template <>
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
        set_device(m_device_id);
        spdlog::debug("FDMTCUDA::Impl: Set device to {}", m_device_id);
        // Allocate memory for the state buffers
        m_state_in_d.resize(m_plan.get_buffer_size(), 0.0F);
        m_state_out_d.resize(m_plan.get_buffer_size(), 0.0F);
        // Only allocate history buffer if needed
        if (m_use_history) {
            m_history_d.resize(m_plan.get_history_size(), 0.0F);
        }
        transfer_fdmt_plan_to_device(m_plan.get_container(), m_plan_d);
        DMT_CHECK_LAST_CUDA_ERROR("FDMT<CUDA>::Impl constructor failed");
    }
    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const FDMTPlan& get_plan() const { return m_plan; }

    // Host execute: handles HtoD copy, calls device execute, handles DtoH copy
    void execute_h(std::span<const float> waterfall_h, std::span<float> dmt_h) {
        check_inputs(waterfall_h.size(), dmt_h.size());

        // Allocate temporary device buffers for input/output (consider reusing
        // or caching)
        thrust::device_vector<float> waterfall_d(waterfall_h.size());
        thrust::device_vector<float> dmt_d(dmt_h.size());

        // Use default stream 0 for simplicity here, could use a member stream
        cudaStream_t stream = nullptr;
        // Copy H->D
        cudaMemcpyAsync(waterfall_d.data().get(), waterfall_h.data(),
                        waterfall_h.size_bytes(), cudaMemcpyHostToDevice,
                        stream);
        DMT_CHECK_LAST_CUDA_ERROR(
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
        DMT_CHECK_LAST_CUDA_ERROR("execute_h: cudaMemcpyAsync D->H dmt failed");

        // Synchronize stream to ensure copies and kernel are complete
        cudaStreamSynchronize(stream);
        DMT_CHECK_LAST_CUDA_ERROR("execute_h: cudaStreamSynchronize failed");

        spdlog::debug("FDMT<CUDA>::Impl: Host execution complete.");
    }
    // Device execute: directly calls internal device logic
    void execute_d(cuda::std::span<const float> waterfall_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        // Get raw pointers and sizes from cuda::std::span
        const float* wf_ptr = waterfall_d.data();
        SizeType wf_size    = waterfall_d.size();
        float* dmt_ptr      = dmt_d.data();
        SizeType dmt_size   = dmt_d.size();

        check_inputs(wf_size, dmt_size);
        execute_device(wf_ptr, wf_size, dmt_ptr, dmt_size, stream);
        spdlog::debug("FDMT<CUDA>::Impl: Device execution complete on stream");
    }

private:
    int m_device_id;
    bool m_use_history;
    FDMTPlan m_plan;
    FDMTPlanContainerD m_plan_d;
    // State buffers
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;
    thrust::device_vector<float> m_history_d;

    static void set_device(int device_id) {
        if (device_id < 0) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA::Impl: Invalid device_id: {}", device_id));
        }
        cudaSetDevice(device_id);
        DMT_CHECK_LAST_CUDA_ERROR(std::format(
            "FDMTCUDA::Impl: cudaSetDevice failed for device_id: {}",
            device_id));
    }

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != nchans * nsamps) {
            throw std::invalid_argument(
                std::format("FDMTCUDA::Impl: Invalid size of waterfall. "
                            "Expected {}, got {}",
                            nchans * nsamps, waterfall_size));
        }
        if (dmt_size != m_plan.get_dmt_size()) {
            throw std::invalid_argument(std::format(
                "FDMTCUDA::Impl: Invalid size of dmt. Expected {}, got {}",
                m_plan.get_dmt_size(), dmt_size));
        }
        spdlog::debug("FDMTCUDA::Impl: Input dimensions check passed: {}x{}",
                      nchans, nsamps);
    }

    void execute_device(const float* __restrict__ waterfall_d,
                        SizeType /*waterfall_size*/,
                        float* __restrict__ dmt_d,
                        SizeType /*dmt_size*/,
                        cudaStream_t stream) {
        float* state_in_ptr  = m_state_in_d.data().get();
        float* state_out_ptr = m_state_out_d.data().get();

        initialise_device(waterfall_d, state_in_ptr, stream);

        auto coords_sum_cur  = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto coords_copy_cur = m_plan_d.coordinates_copy.get_raw_ptrs();
        // auto coords_prev     = m_plan_d.coordinates.get_raw_ptrs();
        coords_sum_cur.update_offsets(m_plan_d.state_shape.ncoords_sum[0]);
        coords_copy_cur.update_offsets(m_plan_d.state_shape.ncoords_copy[0]);
        DMT_CHECK_LAST_CUDA_ERROR("thrust::raw_pointer_cast failed");

        const auto niters = static_cast<int>(m_plan.get_niters());
        for (int i_iter = 1; i_iter < niters + 1; ++i_iter) {
            const int nsamps = m_plan_d.state_shape.nsamps[i_iter];
            const int ncoords_sum_cur =
                m_plan_d.state_shape.ncoords_sum[i_iter];
            const int ncoords_copy_cur =
                m_plan_d.state_shape.ncoords_copy[i_iter];

            const auto coords_max = std::max(ncoords_sum_cur, ncoords_copy_cur);
            const dim3 block_size = dim3(256, 1);
            const dim3 grid_size =
                dim3((nsamps + block_size.x - 1) / block_size.x, coords_max);
            DMT_CHECK_KERNEL_PARAMS(grid_size, block_size);

            // Determine output buffer: final iteration writes to device_dmt
            float* current_out_ptr = (i_iter == niters) ? dmt_d : state_out_ptr;

            // Launch kernel for this iteration
            kernel_execute_iter<<<grid_size, block_size, 0, stream>>>(
                state_in_ptr, current_out_ptr, coords_sum_cur, coords_copy_cur,
                nsamps, ncoords_sum_cur, ncoords_copy_cur);
            DMT_CHECK_LAST_CUDA_ERROR("kernel_execute_iter launch failed");

            coords_sum_cur.update_offsets(ncoords_sum_cur);
            coords_copy_cur.update_offsets(ncoords_copy_cur);

            // Ping-pong buffers (unless it's the final iteration)
            if (i_iter < niters) {
                std::swap(state_in_ptr, state_out_ptr);
            }
        }
        spdlog::debug("FDMT<CUDA>::Impl: Iterations submitted to stream.");
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
        DMT_CHECK_KERNEL_PARAMS(grid_size, block_size);

        // Launch kernel for initialisation
        kernel_init_fdmt<<<grid_size, block_size, 0, stream>>>(
            waterfall_d, state_d, grids0_dt_grid_ptr, grids0_ndt_ptr,
            grids0_coord_offset_ptr, nsubs, nsamps, dt_max, hist_ptr);
        DMT_CHECK_LAST_CUDA_ERROR("kernel_init_fdmt launch failed");

        // Update history buffer if enabled
        if (m_use_history) {
            // Copy the last nchans x dt_max elements from waterfall to hist
            for (int i_sub = 0; i_sub < nsubs; ++i_sub) {
                thrust::copy_n(
                    &waterfall_d[(i_sub * nsamps) + nsamps - dt_max], dt_max,
                    &hist_ptr[static_cast<ptrdiff_t>(i_sub * dt_max)]);
            }
            DMT_CHECK_LAST_CUDA_ERROR("History update cudaMemcpyAsync failed");
        }
        spdlog::debug(
            "FDMT<CUDA>::Impl: Initialise device submitted to stream.");
    }
}; // End FDMTCUDA::Impl definition

// CUDA-specific constructor implementation
template <>
template <std::same_as<backend::CUDA> P>
FDMT<backend::CUDA>::FDMT(float f_min,
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
                                    device_id)) {
    spdlog::debug("FDMT<CUDA> object created for device {}", device_id);
}

template <>
FDMT<backend::CUDA>::~FDMT() = default;
template <>
FDMT<backend::CUDA>::FDMT(FDMT&& other) noexcept = default;
template <>
FDMT<backend::CUDA>&
FDMT<backend::CUDA>::operator=(FDMT&& other) noexcept = default;
template <>
const FDMTPlan& FDMT<backend::CUDA>::get_plan() const {
    return m_impl->get_plan();
}
template <>
void FDMT<backend::CUDA>::execute(std::span<const float> waterfall,
                                  std::span<float> dmt) {
    m_impl->execute_h(waterfall, dmt);
}

template <>
template <typename B> // Need template parameter from header declaration
auto FDMT<backend::CUDA>::execute(cuda::std::span<const float> d_waterfall,
                                  cuda::std::span<float> d_dmt,
                                  cudaStream_t stream)
    requires std::is_same_v<B,
                            backend::CUDA> // Match requires clause from header
{
    // Forward the call to the implementation object's device handler
    m_impl->execute_d(d_waterfall, d_dmt, stream);
}

} // namespace dmt
