#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <dmt/ddmt_base.hpp>

struct DDMTPlanD {
    thrust::device_vector<float> dm_arr_d;
    thrust::device_vector<int> delay_arr_d;
    thrust::device_vector<int> kill_mask_d;
};

class DDMTGPU : public DDMT {
public:
    DDMTGPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);
    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size) override;

    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size,
                 bool device_flags);

private:
    DDMTPlanD m_ddmt_plan_d;

    static void transfer_plan_to_device(const DDMTPlan& plan,
                                        DDMTPlanD& plan_d);
    void execute_device(const float* __restrict waterfall,
                        SizeType waterfall_size,
                        float* __restrict dmt,
                        SizeType dmt_size);
};