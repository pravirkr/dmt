#pragma once

#include <dmt/ddmt_base.hpp>

class DDMTCPU : public DDMT {
public:
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);
    static void set_num_threads(int nthreads);
    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size) override;
};
