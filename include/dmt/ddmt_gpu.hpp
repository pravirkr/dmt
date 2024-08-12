#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <dmt/dmt_plans.hpp>
#include <dmt/dmt_plans_gpu.hpp>


class DDMTGPU{
public:
    DDMTGPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);
    
    DDMTGPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<float>& dm_arr);
    
    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size);

    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size,
                 bool device_flags);

private:
    DDMTPlanD m_ddmt_plan_d;

    void execute_device(const float* __restrict waterfall,
                        SizeType waterfall_size,
                        float* __restrict dmt,
                        SizeType dmt_size);
};