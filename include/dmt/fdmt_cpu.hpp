#pragma once

#include <vector>

#include <dmt/dmt_plans.hpp>

class FDMTCPU {
public:
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            SizeType dt_max,
            SizeType dt_step = 1,
            SizeType dt_min  = 0);

    FDMTCPU(const FDMTCPU&)            = delete;
    FDMTCPU& operator=(const FDMTCPU&) = delete;
    FDMTCPU(FDMTCPU&&)                 = delete;
    FDMTCPU& operator=(FDMTCPU&&)      = delete;
    ~FDMTCPU()                         = default;

    static void set_num_threads(int nthreads);
    static void set_log_level(int level);

    FDMTPlan& get_plan() const;

    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size);
    void initialise(const float* __restrict waterfall,
                    SizeType waterfall_size,
                    float* __restrict state,
                    SizeType state_size);
    void initialise2(const float* __restrict waterfall,
                     SizeType waterfall_size,
                     float* __restrict state,
                     SizeType state_size);

private:
    FDMTPlan m_plan;
    // State buffers
    std::vector<float> m_state_in;
    std::vector<float> m_state_out;

    void execute_iter(const float* __restrict state_in,
                      float* __restrict state_out,
                      SizeType i_iter);
    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const;
};
