#pragma once

#include <cstddef>
#include <thrust/device_vector.h>

#include <dmt/fdmt_base.hpp>

struct FDMTPlanD {
    // i = i_iter
    thrust::device_vector<int> nsubs_d;
    thrust::device_vector<int> nsamps_d;
    thrust::device_vector<int> ncoords_d;
    thrust::device_vector<int> ncoords_to_copy_d;
    thrust::device_vector<int> subs_iter_idx_d;
    thrust::device_vector<int> coords_iter_idx_d;
    thrust::device_vector<int> coords_to_copy_iter_idx_d;
    thrust::device_vector<int> mappings_iter_idx_d;
    thrust::device_vector<int> mappings_to_copy_iter_idx_d;
    // i, i+1 = coords_iter_idx_d[i_iter] + i_coord
    thrust::device_vector<int> coordinates_d;
    thrust::device_vector<int> coordinates_to_copy_d;
    // i, i+1, ... i+4 = mappings_iter_idx_d[i_iter] + i_coord
    thrust::device_vector<int> mappings_d;
    thrust::device_vector<int> mappings_to_copy_d;
    // i = subs_iter_idx_d[i_iter] + isub
    thrust::device_vector<int> state_sub_idx_d;

    // i = i_sub (only for i_iter = 0)
    thrust::device_vector<int> ndt_grid_init_d;
    thrust::device_vector<int> dt_grid_init_sub_idx_d;
    // i = dt_grid_init_sub_idx_d[i_sub] + i_dt
    thrust::device_vector<int> dt_grid_init_d;
};

class FDMTGPU : public FDMT {
public:
    FDMTGPU(float f_min,
            float f_max,
            size_t nchans,
            size_t nsamps,
            float tsamp,
            size_t dt_max,
            size_t dt_step = 1,
            size_t dt_min  = 0);
    void execute(const float* __restrict waterfall,
                 size_t waterfall_size,
                 float* __restrict dmt,
                 size_t dmt_size) override;

    void initialise(const float* __restrict waterfall,
                    size_t waterfall_size,
                    float* __restrict state,
                    size_t state_size) override;

    void execute(const float* __restrict waterfall,
                 size_t waterfall_size,
                 float* __restrict dmt,
                 size_t dmt_size,
                 bool device_flags);

    void initialise(const float* __restrict waterfall,
                    size_t waterfall_size,
                    float* __restrict state,
                    size_t state_size,
                    bool device_flags);

private:
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;

    FDMTPlanD m_fdmt_plan_d;

    static void transfer_plan_to_device(const FDMTPlan& plan,
                                        FDMTPlanD& plan_d);
    void initialise_device(const float* __restrict waterfall,
                           float* __restrict state);

    void execute_device(const float* __restrict waterfall,
                        size_t waterfall_size,
                        float* __restrict dmt,
                        size_t dmt_size);
};

__global__ void
kernel_init_fdmt(const float* __restrict__ waterfall,
                 float* __restrict__ state,
                 const int* __restrict__ state_sub_idx_d_ptr,
                 const int* __restrict__ dt_grid_init_d_ptr,
                 const int* __restrict__ ndt_grid_init_d_ptr,
                 const int* __restrict__ dt_grid_init_sub_idx_d_ptr,
                 int nsubs,
                 int nsamps);

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
                    int coords_copy_cur_size);
