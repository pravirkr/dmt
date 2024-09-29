#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include <dmt/fdmt/fdmt_cuda.hpp>

#include "dmt/cuda_utils.cuh"

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
    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    int i_sub = blockIdx.y;
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
        const auto state_offset_cur  = buffer_offset + i_dt * nsamps;
        const auto state_offset_prev = buffer_offset + (i_dt - 1) * nsamps;

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
        if (isamp < dt_cur) {
            float sum = 0.0F;
            int i     = isamp - dt_cur;
            int i_end = isamp - dt_prev;
            // Sum from history
            for (; i < 0 && i < i_end; ++i) {
                sum += hist[hist_offset + (dt_max + i)];
            }
            // Sum from waterfall
            for (; i < i_end; ++i) {
                sum += waterfall[waterfall_offset + i];
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
    int isamp   = blockIdx.x * blockDim.x + threadIdx.x;
    int i_coord = blockIdx.y;
    if (isamp >= nsamps) {
        return;
    }

    if (i_coord < ncoords_sum_cur) {
        const auto nsamps_out            = coords_sum.nsamps[i_coord];
        const auto coord_buf_offset      = coords_sum.buf_offset[i_coord];
        const auto offset                = coords_sum.offset[i_coord];
        const auto nsamps_tail           = coords_sum.tail_nsamps[i_coord];
        const auto coord_tail_buf_offset = coords_sum.tail_buf_offset[i_coord];
        const auto coord_head_buf_offset = coords_sum.head_buf_offset[i_coord];

        const float* __restrict tail = &state_in[coord_tail_buf_offset];
        const float* __restrict head = &state_in[coord_head_buf_offset];
        float* __restrict out        = &state_out[coord_buf_offset];
        if (isamp < offset) {
            out[isamp] = tail[isamp];
        } else if (isamp >= offset && isamp < nsamps_tail) {
            out[isamp] = tail[isamp] + head[isamp - offset];
        } else if (isamp >= nsamps_tail &&
                   isamp < min(nsamps_tail + offset, nsamps_out)) {
            out[isamp] = head[isamp - offset];
        } else if (isamp >= min(nsamps_tail + offset, nsamps_out) &&
                   isamp < nsamps_out) {
            out[isamp] = 0.0F;
        }
    }
    __syncthreads();

    if (i_coord < ncoords_copy_cur) {
        const auto coord_buf_offset      = coords_copy.buf_offset[i_coord];
        const auto nsamps_tail           = coords_copy.tail_nsamps[i_coord];
        const auto coord_tail_buf_offset = coords_copy.tail_buf_offset[i_coord];
        if (isamp < nsamps_tail) {
            state_out[coord_buf_offset + isamp] =
                state_in[coord_tail_buf_offset + isamp];
        }
    }
}

FDMTGPU::FDMTGPU(float f_min,
                 float f_max,
                 size_t nchans,
                 size_t nsamps,
                 float tsamp,
                 size_t dt_max,
                 size_t dt_step,
                 size_t dt_min,
                 bool use_history,
                 int device_id)
    : m_use_history(use_history),
      m_device_id(device_id),
      m_plan(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    if (m_device_id < 0) {
        throw std::invalid_argument("Invalid device_id");
    }
    set_device(m_device_id);
    // Allocate memory for the state buffers
    const auto state_size = m_plan.get_buffer_size();
    const auto hist_size  = m_plan.get_history_size();
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
    m_history_d.resize(hist_size, 0.0F);
    transfer_fdmt_plan_to_device(m_plan.get_container(), m_plan_d);
}

void FDMTGPU::execute(const float* __restrict waterfall,
                      size_t waterfall_size,
                      float* __restrict dmt,
                      size_t dmt_size) {
    execute(waterfall, waterfall_size, dmt, dmt_size, false);
}

void FDMTGPU::set_log_level(int level) { FDMTPlan::set_log_level(level); }

const FDMTPlan& FDMTGPU::get_plan() const { return m_plan; }

void FDMTGPU::set_device(int device_id) {
    cudaSetDevice(device_id);
    error_checker::check_cuda("cudaSetDevice failed");
}

void FDMTGPU::execute(const float* __restrict waterfall,
                      size_t waterfall_size,
                      float* __restrict dmt,
                      size_t dmt_size,
                      bool device_flags) {
    if (device_flags) {
        execute_device(waterfall, waterfall_size, dmt, dmt_size);
    } else {
        thrust::device_vector<float> waterfall_d(waterfall,
                                                 waterfall + waterfall_size);
        thrust::device_vector<float> dmt_d(dmt, dmt + dmt_size);
        execute_device(thrust::raw_pointer_cast(waterfall_d.data()),
                       waterfall_size, thrust::raw_pointer_cast(dmt_d.data()),
                       dmt_size);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt);
        error_checker::check_cuda("thrust::copy failed");
    }
}

void FDMTGPU::initialise(const float* __restrict waterfall,
                         size_t waterfall_size,
                         float* __restrict state,
                         size_t state_size) {
    initialise(waterfall, waterfall_size, state, state_size, false);
}

void FDMTGPU::initialise(const float* __restrict waterfall,
                         size_t waterfall_size,
                         float* __restrict state,
                         size_t state_size,
                         bool device_flags) {
    if (device_flags) {
        initialise_device(waterfall, state);
    } else {
        thrust::device_vector<float> waterfall_d(waterfall,
                                                 waterfall + waterfall_size);
        thrust::device_vector<float> state_d(state, state + state_size);
        initialise_device(thrust::raw_pointer_cast(waterfall_d.data()),
                          thrust::raw_pointer_cast(state_d.data()));
        thrust::copy(state_d.begin(), state_d.end(), state);
        error_checker::check_cuda("thrust::copy failed");
    }
}

void FDMTGPU::execute_device(const float* __restrict waterfall,
                             size_t waterfall_size,
                             float* __restrict dmt,
                             size_t dmt_size) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = thrust::raw_pointer_cast(m_state_in_d.data());
    float* state_out_ptr = thrust::raw_pointer_cast(m_state_out_d.data());

    initialise_device(waterfall, state_in_ptr);

    auto coords_sum_cur  = m_plan_d.coordinates_sum.get_raw_ptrs();
    auto coords_copy_cur = m_plan_d.coordinates_copy.get_raw_ptrs();
    // auto coords_prev     = m_plan_d.coordinates.get_raw_ptrs();
    coords_sum_cur.update_offsets(m_plan_d.state_shape.ncoords_sum[0]);
    coords_copy_cur.update_offsets(m_plan_d.state_shape.ncoords_copy[0]);
    error_checker::check_cuda("thrust::raw_pointer_cast failed");

    const auto niters = static_cast<int>(m_plan.get_niters());
    for (int i_iter = 1; i_iter < niters + 1; ++i_iter) {
        const int nsamps           = m_plan_d.state_shape.nsamps[i_iter];
        const int ncoords_sum_cur  = m_plan_d.state_shape.ncoords_sum[i_iter];
        const int ncoords_copy_cur = m_plan_d.state_shape.ncoords_copy[i_iter];

        const auto coords_max = std::max(ncoords_sum_cur, ncoords_copy_cur);
        const dim3 block_size = dim3(256, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, coords_max);
        kernel_execute_iter<<<grid_size, block_size>>>(
            state_in_ptr, state_out_ptr, coords_sum_cur, coords_copy_cur,
            nsamps, ncoords_sum_cur, ncoords_copy_cur);
        error_checker::check_cuda("kernel_execute_iter failed");

        coords_sum_cur.update_offsets(ncoords_sum_cur);
        coords_copy_cur.update_offsets(ncoords_copy_cur);

        std::swap(state_in_ptr, state_out_ptr);
        if (i_iter == (niters - 1)) {
            state_out_ptr = dmt;
        }
    }
}

void FDMTGPU::initialise_device(const float* __restrict waterfall,
                                float* __restrict state) {
    const int nsubs  = m_plan_d.state_shape.nchans[0];
    const int nsamps = m_plan_d.state_shape.nsamps[0];
    const int dt_max = m_plan_d.state_shape.dt_max[0];
    const int* grids0_dt_grid_ptr =
        thrust::raw_pointer_cast(m_plan_d.grids0.dt_grid.data());
    const int* grids0_ndt_ptr =
        thrust::raw_pointer_cast(m_plan_d.grids0.ndt.data());
    const int* grids0_coord_offset_ptr =
        thrust::raw_pointer_cast(m_plan_d.grids0.coord_offset.data());
    auto* hist = thrust::raw_pointer_cast(m_history_d.data());

    const dim3 block_size = dim3(1024, 1);
    const dim3 grid_size =
        dim3((nsamps + block_size.x - 1) / block_size.x, nsubs);
    kernel_init_fdmt<<<grid_size, block_size>>>(
        waterfall, state, grids0_dt_grid_ptr, grids0_ndt_ptr,
        grids0_coord_offset_ptr, nsubs, nsamps, dt_max, hist);
    error_checker::check_cuda("kernel_init_fdmt failed");
    if (m_use_history) {
        // Copy the last nchans x dt_max elements from waterfall to hist
        for (int i_sub = 0; i_sub < nsubs; ++i_sub) {
            thrust::copy_n(&waterfall[i_sub * nsamps + nsamps - dt_max], dt_max,
                           &hist[i_sub * dt_max]);
        }
    }
}

void FDMTGPU::check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
    const auto nchans = m_plan.get_nchans();
    const auto nsamps = m_plan.get_nsamps();
    if (waterfall_size != nchans * nsamps) {
        throw std::invalid_argument("Invalid size of waterfall");
    }
    if (dmt_size != m_plan.get_dmt_size()) {
        throw std::invalid_argument("Invalid size of dmt");
    }
    // spdlog::debug("FDMT: Input dimensions: {}x{}", nchans, nsamps);
}