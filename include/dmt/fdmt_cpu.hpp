#pragma once

#include <vector>

#include <dmt/fdmt_base.hpp>

class FDMTCPU : public FDMT {
public:
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            SizeType dt_max,
            SizeType dt_step = 1,
            SizeType dt_min  = 0);
    static void set_num_threads(int nthreads);
    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size) override;
    void initialise(const float* __restrict waterfall,
                    SizeType waterfall_size,
                    float* __restrict state,
                    SizeType state_size) override;

private:
    // Buffers
    std::vector<float> m_state_in;
    std::vector<float> m_state_out;

    void execute_iter(const float* __restrict state_in,
                      float* __restrict state_out,
                      SizeType i_iter);
};
