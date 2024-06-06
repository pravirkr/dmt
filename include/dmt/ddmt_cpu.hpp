#pragma once

#include <dmt/ddmt_base.hpp>

class DDMTCPU : public DDMT {
public:
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
            const float* dm_arr,
            SizeType dm_count);

    static void set_num_threads(int nthreads);
    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size) override;

private:
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
