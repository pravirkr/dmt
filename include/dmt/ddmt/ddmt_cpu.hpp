#pragma once

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

class DDMTCPU {
public:
    DDMTCPU(const DDMTCPU&)            = delete;
    DDMTCPU(DDMTCPU&&)                 = default;
    DDMTCPU& operator=(const DDMTCPU&) = delete;
    DDMTCPU& operator=(DDMTCPU&&)      = default;
    ~DDMTCPU()                         = default;
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);

    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            const std::vector<float>& dm_arr);

    const DDMTPlan& get_plan() const;

    static void set_num_threads(int nthreads);
    void execute(const float* __restrict__ waterfall,
                 SizeType waterfall_size,
                 float* __restrict__ dmt,
                 SizeType dmt_size);

private:
    DDMTPlan m_plan;
    static void execute_dedisp(const float* __restrict__ d_in,
                               size_t in_chan_stride,
                               size_t in_samp_stride,
                               float* __restrict__ d_out,
                               size_t out_dm_stride,
                               size_t out_samp_stride,
                               const size_t* __restrict__ delay_table,
                               size_t dm_count,
                               size_t nchans,
                               size_t nsamps_reduced);
};
