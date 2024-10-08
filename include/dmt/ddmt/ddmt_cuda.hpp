#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/plans_cuda.hpp"
#include "dmt/common/types.hpp"

class DDMTCUDA {
public:
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min = 0.0F);

    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_arr);

    void execute(const float* __restrict__ waterfall,
                 SizeType waterfall_size,
                 float* __restrict__ dmt,
                 SizeType dmt_size);

    void execute(const float* __restrict__ waterfall,
                 SizeType waterfall_size,
                 float* __restrict__ dmt,
                 SizeType dmt_size,
                 bool device_flags);

private:
    int m_device_id;
    DDMTPlan m_plan;
    DDMTPlanD m_plan_d;

    void execute_device(const float* __restrict__ waterfall,
                        SizeType waterfall_size,
                        float* __restrict__ dmt,
                        SizeType dmt_size);
};