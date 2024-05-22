#pragma once

#include <fdmt_base.hpp>

class FDMTCPU : public FDMT {
public:
    FDMTCPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    static void set_num_threads(int nthreads);
    void execute(const float* __restrict waterfall, size_t waterfall_size, float* __restrict dmt,
                 size_t dmt_size) override;
    void initialise(const float* __restrict waterfall, float* __restrict state) override;

private:
    // Buffers
    std::vector<float> m_state_in;
    std::vector<float> m_state_out;

    void execute_iter(const float* __restrict state_in, float* __restrict state_out, SizeType i_iter);
};
