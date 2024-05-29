#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <dmt/fdmt_gpu.hpp>

namespace error_checker {

inline void
check_cuda_error(const char* file, int line, const std::string& msg = "") {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream error_msg;
        error_msg << "CUDA failed with error: " << cudaGetErrorString(error)
                  << " (" << file << ":" << line << ")";
        if (!msg.empty()) {
            error_msg << " - " << msg;
        }
        throw std::runtime_error(error_msg.str());
    }
}

inline void
check_cuda_error_sync(const char* file, int line, const std::string& msg = "") {
    cudaDeviceSynchronize();
    check_cuda_error(file, line, msg);
}

void check_cuda(const std::string& msg = "") {
    check_cuda_error(__FILE__, __LINE__, msg);
}

void check_cuda_sync(const std::string& msg = "") {
    check_cuda_error_sync(__FILE__, __LINE__, msg);
}

} // namespace error_checker

__global__ void
kernel_init_fdmt(const float* __restrict__ waterfall,
                 float* __restrict__ state,
                 const int* __restrict__ state_sub_idx_d_ptr,
                 const int* __restrict__ dt_grid_init_d_ptr,
                 const int* __restrict__ ndt_grid_init_d_ptr,
                 const int* __restrict__ dt_grid_init_sub_idx_d_ptr,
                 int nsubs,
                 int nsamps) {
    int isamp = blockIdx.x * blockDim.x + threadIdx.x;
    int i_sub = blockIdx.y;
    if (i_sub >= nsubs || isamp >= nsamps) {
        return;
    }
    const int* dt_grid_sub =
        &dt_grid_init_d_ptr[dt_grid_init_sub_idx_d_ptr[i_sub]];
    const auto& state_sub_idx    = state_sub_idx_d_ptr[i_sub];
    const auto& dt_grid_sub_size = ndt_grid_init_d_ptr[i_sub];
    const auto& dt_grid_sub_min  = dt_grid_sub[0];

    // Initialise state for [:, dt_init_min, dt_init_min:]
    if (isamp >= dt_grid_sub_min) {
        float sum = 0.0F;
        for (int i = isamp - dt_grid_sub_min; i <= isamp; ++i) {
            sum += waterfall[i_sub * nsamps + i];
        }
        state[state_sub_idx + isamp] =
            sum / static_cast<float>(dt_grid_sub_min + 1);
    }
    // Initialise state for [:, dt_grid_init[i_dt], dt_grid_init[i_dt]:]
    for (int i_dt = 1; i_dt < dt_grid_sub_size; ++i_dt) {
        const auto dt_cur  = dt_grid_sub[i_dt];
        const auto dt_prev = dt_grid_sub[i_dt - 1];
        if (isamp >= dt_cur) {
            float sum = 0.0F;
            for (int i = isamp - dt_cur; i < isamp - dt_prev; ++i) {
                sum += waterfall[i_sub * nsamps + i];
            }
            state[state_sub_idx + i_dt * nsamps + isamp] =
                (state[state_sub_idx + (i_dt - 1) * nsamps + isamp] *
                     (static_cast<float>(dt_prev) + 1.0F) +
                 sum) /
                (static_cast<float>(dt_cur) + 1.0F);
        }
    }
}

__global__ void
kernel_execute_iter(const float* __restrict__ state_in,
                    float* __restrict__ state_out,
                    const int* __restrict__ coords_d_cur,
                    const int* __restrict__ mappings_d_cur,
                    const int* __restrict__ coords_copy_d_cur,
                    const int* __restrict__ mappings_copy_d_cur,
                    const int* __restrict__ state_sub_idx_d_cur,
                    const int* __restrict__ state_sub_idx_d_prev,
                    int nsamps,
                    int coords_cur_size,
                    int coords_copy_cur_size) {
    int isamp   = blockIdx.x * blockDim.x + threadIdx.x;
    int i_coord = blockIdx.y;
    if (isamp >= nsamps) {
        return;
    }

    if (i_coord < coords_cur_size) {
        const auto& i_sub              = coords_d_cur[i_coord * 2];
        const auto& i_dt               = coords_d_cur[i_coord * 2 + 1];
        const auto& i_sub_tail         = mappings_d_cur[i_coord * 5];
        const auto& i_dt_tail          = mappings_d_cur[i_coord * 5 + 1];
        const auto& i_sub_head         = mappings_d_cur[i_coord * 5 + 2];
        const auto& i_dt_head          = mappings_d_cur[i_coord * 5 + 3];
        const auto& offset             = mappings_d_cur[i_coord * 5 + 4];
        const auto& state_sub_idx      = state_sub_idx_d_cur[i_sub];
        const auto& state_sub_idx_tail = state_sub_idx_d_prev[i_sub_tail];
        const auto& state_sub_idx_head = state_sub_idx_d_prev[i_sub_head];

        const float* __restrict tail =
            &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        const float* __restrict head =
            &state_in[state_sub_idx_head + i_dt_head * nsamps];
        float* __restrict out = &state_out[state_sub_idx + i_dt * nsamps];
        if (isamp < offset) {
            out[isamp] = tail[isamp];
        } else {
            out[isamp] = tail[isamp] + head[isamp - offset];
        }
    }
    __syncthreads();

    if (i_coord < coords_copy_cur_size) {
        const auto& i_sub              = coords_copy_d_cur[i_coord * 2];
        const auto& i_dt               = coords_copy_d_cur[i_coord * 2 + 1];
        const auto& i_sub_tail         = mappings_copy_d_cur[i_coord * 5];
        const auto& i_dt_tail          = mappings_copy_d_cur[i_coord * 5 + 1];
        const auto& state_sub_idx      = state_sub_idx_d_cur[i_sub];
        const auto& state_sub_idx_tail = state_sub_idx_d_prev[i_sub_tail];

        const float* __restrict tail =
            &state_in[state_sub_idx_tail + i_dt_tail * nsamps];
        float* __restrict out = &state_out[state_sub_idx + i_dt * nsamps];
        out[isamp]            = tail[isamp];
    }
}

FDMTGPU::FDMTGPU(float f_min,
                 float f_max,
                 size_t nchans,
                 size_t nsamps,
                 float tsamp,
                 size_t dt_max,
                 size_t dt_step,
                 size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
    transfer_plan_to_device(plan, m_fdmt_plan_d);
}

void FDMTGPU::execute(const float* __restrict waterfall,
                      size_t waterfall_size,
                      float* __restrict dmt,
                      size_t dmt_size) {
    execute(waterfall, waterfall_size, dmt, dmt_size, false);
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

void FDMTGPU::transfer_plan_to_device(const FDMTPlan& plan, FDMTPlanD& plan_d) {
    // Transfer the plan to the device
    const auto niter_size = static_cast<int>(plan.state_shape.size());

    // Temp vectors to store the flattened plan on the host
    std::vector<int> nsubs_h, ncoords_h, nsamps_h, ncoords_to_copy_h;
    std::vector<int> subs_iter_idx_h, coords_iter_idx_h,
        coords_to_copy_iter_idx_h, mappings_iter_idx_h,
        mappings_to_copy_iter_idx_h;
    std::vector<int> coordinates_h, coordinates_to_copy_h, mappings_h,
        mappings_to_copy_h, state_sub_idx_h;
    nsubs_h.reserve(niter_size);
    ncoords_h.reserve(niter_size);
    nsamps_h.reserve(niter_size);
    ncoords_to_copy_h.reserve(niter_size);
    subs_iter_idx_h.reserve(niter_size);
    coords_iter_idx_h.reserve(niter_size);
    coords_to_copy_iter_idx_h.reserve(niter_size);
    mappings_iter_idx_h.reserve(niter_size);
    mappings_to_copy_iter_idx_h.reserve(niter_size);
    for (int i = 0; i < niter_size; ++i) {
        nsubs_h.emplace_back(plan.state_shape[i][0]);
        nsamps_h.emplace_back(plan.state_shape[i][4]);
        ncoords_h.emplace_back(plan.coordinates[i].size());
        ncoords_to_copy_h.emplace_back(plan.coordinates_to_copy[i].size());
    }

    // Cumulative sum
    subs_iter_idx_h.emplace_back(0);
    coords_iter_idx_h.emplace_back(0);
    coords_to_copy_iter_idx_h.emplace_back(0);
    mappings_iter_idx_h.emplace_back(0);
    mappings_to_copy_iter_idx_h.emplace_back(0);
    for (int i = 1; i < niter_size; ++i) {
        subs_iter_idx_h.emplace_back(subs_iter_idx_h[i - 1] + nsubs_h[i - 1]);
        coords_iter_idx_h.emplace_back(coords_iter_idx_h[i - 1] +
                                       ncoords_h[i - 1] * 2);
        coords_to_copy_iter_idx_h.emplace_back(
            coords_to_copy_iter_idx_h[i - 1] + ncoords_to_copy_h[i - 1] * 2);
        mappings_iter_idx_h.emplace_back(mappings_iter_idx_h[i - 1] +
                                         ncoords_h[i - 1] * 5);
        mappings_to_copy_iter_idx_h.emplace_back(
            mappings_to_copy_iter_idx_h[i - 1] + ncoords_to_copy_h[i - 1] * 5);
    }

    // Resize the vectors
    coordinates_h.reserve(coords_iter_idx_h.back() + ncoords_h.back() * 2);
    coordinates_to_copy_h.reserve(coords_to_copy_iter_idx_h.back() +
                                  ncoords_to_copy_h.back() * 2);
    mappings_h.reserve(mappings_iter_idx_h.back() + ncoords_h.back() * 5);
    mappings_to_copy_h.reserve(mappings_to_copy_iter_idx_h.back() +
                               ncoords_to_copy_h.back() * 5);
    state_sub_idx_h.reserve(subs_iter_idx_h.back() + nsubs_h.back());
    // Flatten the coordinates and mappings
    for (int i = 0; i < niter_size; ++i) {
        for (int j = 0; j < ncoords_h[i]; ++j) {
            const auto& coord   = plan.coordinates[i][j];
            const auto& mapping = plan.mappings[i][j];
            coordinates_h.emplace_back(coord.first);
            coordinates_h.emplace_back(coord.second);
            mappings_h.emplace_back(mapping.tail.first);
            mappings_h.emplace_back(mapping.tail.second);
            mappings_h.emplace_back(mapping.head.first);
            mappings_h.emplace_back(mapping.head.second);
            mappings_h.emplace_back(mapping.offset);
        }
        for (int j = 0; j < ncoords_to_copy_h[i]; ++j) {
            const auto& coord   = plan.coordinates_to_copy[i][j];
            const auto& mapping = plan.mappings_to_copy[i][j];
            coordinates_to_copy_h.emplace_back(coord.first);
            coordinates_to_copy_h.emplace_back(coord.second);
            mappings_to_copy_h.emplace_back(mapping.tail.first);
            mappings_to_copy_h.emplace_back(mapping.tail.second);
            mappings_to_copy_h.emplace_back(mapping.head.first);
            mappings_to_copy_h.emplace_back(mapping.head.second);
            mappings_to_copy_h.emplace_back(mapping.offset);
        }
        for (int j = 0; j < nsubs_h[i]; ++j) {
            state_sub_idx_h.emplace_back(plan.state_sub_idx[i][j]);
        }
    }

    // Copy to device
    plan_d.nsubs_d                     = nsubs_h;
    plan_d.nsamps_d                    = nsamps_h;
    plan_d.ncoords_d                   = ncoords_h;
    plan_d.ncoords_to_copy_d           = ncoords_to_copy_h;
    plan_d.subs_iter_idx_d             = subs_iter_idx_h;
    plan_d.coords_iter_idx_d           = coords_iter_idx_h;
    plan_d.coords_to_copy_iter_idx_d   = coords_to_copy_iter_idx_h;
    plan_d.mappings_iter_idx_d         = mappings_iter_idx_h;
    plan_d.mappings_to_copy_iter_idx_d = mappings_to_copy_iter_idx_h;
    plan_d.coordinates_d               = coordinates_h;
    plan_d.coordinates_to_copy_d       = coordinates_to_copy_h;
    plan_d.mappings_d                  = mappings_h;
    plan_d.mappings_to_copy_d          = mappings_to_copy_h;
    plan_d.state_sub_idx_d             = state_sub_idx_h;

    // dt_grid allocation for initialisation
    const auto& dt_grid_init = plan.dt_grid[0];

    std::vector<int> ndt_grid_init_h, dt_grid_init_sub_idx_h, dt_grid_init_h;
    ndt_grid_init_h.reserve(dt_grid_init.size());
    for (const auto& dt_grid : dt_grid_init) {
        ndt_grid_init_h.emplace_back(dt_grid.size());
    }
    dt_grid_init_sub_idx_h.emplace_back(0);
    for (int i = 1; i < ndt_grid_init_h.size(); ++i) {
        dt_grid_init_sub_idx_h.emplace_back(dt_grid_init_sub_idx_h[i - 1] +
                                            ndt_grid_init_h[i - 1]);
    }
    dt_grid_init_h.reserve(dt_grid_init_sub_idx_h.back() +
                           ndt_grid_init_h.back());
    for (const auto& dt_grid : dt_grid_init) {
        for (const auto& dt : dt_grid) {
            dt_grid_init_h.emplace_back(dt);
        }
    }
    plan_d.ndt_grid_init_d        = ndt_grid_init_h;
    plan_d.dt_grid_init_sub_idx_d = dt_grid_init_sub_idx_h;
    plan_d.dt_grid_init_d         = dt_grid_init_h;
}

void FDMTGPU::execute_device(const float* __restrict waterfall,
                             size_t waterfall_size,
                             float* __restrict dmt,
                             size_t dmt_size) {
    check_inputs(waterfall_size, dmt_size);
    float* state_in_ptr  = thrust::raw_pointer_cast(m_state_in_d.data());
    float* state_out_ptr = thrust::raw_pointer_cast(m_state_out_d.data());

    initialise_device(waterfall, state_in_ptr);

    const auto* state_sub_idx_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.state_sub_idx_d.data());
    const auto* coords_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.coordinates_d.data());
    const auto* coords_copy_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.coordinates_to_copy_d.data());
    const auto* mappings_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.mappings_d.data());
    const auto* mappings_copy_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.mappings_to_copy_d.data());
    error_checker::check_cuda("thrust::raw_pointer_cast failed");

    const auto niters = static_cast<int>(get_niters());
    for (int i_iter = 1; i_iter < niters + 1; ++i_iter) {
        const auto& nsamps          = m_fdmt_plan_d.nsamps_d[i_iter];
        const auto& ncoords         = m_fdmt_plan_d.ncoords_d[i_iter];
        const auto& ncoords_to_copy = m_fdmt_plan_d.ncoords_to_copy_d[i_iter];

        const auto* coords_d_cur =
            &coords_d_ptr[m_fdmt_plan_d.coords_iter_idx_d[i_iter]];
        const auto* mappings_d_cur =
            &mappings_d_ptr[m_fdmt_plan_d.mappings_iter_idx_d[i_iter]];
        const auto* coords_copy_d_cur =
            &coords_copy_d_ptr[m_fdmt_plan_d.coords_to_copy_iter_idx_d[i_iter]];
        const auto* mappings_copy_d_cur =
            &mappings_copy_d_ptr[m_fdmt_plan_d
                                     .mappings_to_copy_iter_idx_d[i_iter]];
        const auto* state_sub_idx_d_cur =
            &state_sub_idx_d_ptr[m_fdmt_plan_d.subs_iter_idx_d[i_iter]];
        const auto* state_sub_idx_d_prev =
            &state_sub_idx_d_ptr[m_fdmt_plan_d.subs_iter_idx_d[i_iter - 1]];

        const auto coords_max = std::max(ncoords, ncoords_to_copy);
        const dim3 block_size = dim3(1024, 1);
        const dim3 grid_size =
            dim3((nsamps + block_size.x - 1) / block_size.x, coords_max);
        kernel_execute_iter<<<grid_size, block_size>>>(
            state_in_ptr, state_out_ptr, coords_d_cur, mappings_d_cur,
            coords_copy_d_cur, mappings_copy_d_cur, state_sub_idx_d_cur,
            state_sub_idx_d_prev, nsamps, ncoords, ncoords_to_copy);
        error_checker::check_cuda("kernel_execute_iter failed");

        std::swap(state_in_ptr, state_out_ptr);
        if (i_iter == (niters - 1)) {
            state_out_ptr = dmt;
        }
    }
}

void FDMTGPU::initialise_device(const float* __restrict waterfall,
                                float* __restrict state) {
    const int nsubs  = m_fdmt_plan_d.nsubs_d[0];
    const int nsamps = m_fdmt_plan_d.nsamps_d[0];
    const int* state_sub_idx_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.state_sub_idx_d.data());
    const int* dt_grid_init_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.dt_grid_init_d.data());
    const int* ndt_grid_init_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.ndt_grid_init_d.data());
    const int* dt_grid_init_sub_idx_d_ptr =
        thrust::raw_pointer_cast(m_fdmt_plan_d.dt_grid_init_sub_idx_d.data());

    const dim3 block_size = dim3(256, 1);
    const dim3 grid_size =
        dim3((nsamps + block_size.x - 1) / block_size.x, nsubs);
    kernel_init_fdmt<<<grid_size, block_size>>>(
        waterfall, state, state_sub_idx_d_ptr, dt_grid_init_d_ptr,
        ndt_grid_init_d_ptr, dt_grid_init_sub_idx_d_ptr, nsubs, nsamps);
    error_checker::check_cuda("kernel_init_fdmt failed");
}
