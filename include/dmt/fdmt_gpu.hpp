#pragma once

#include <thrust/device_vector.h>

#include <dmt/dmt_plans.hpp>
#include <dmt/dmt_plans_gpu.hpp>

class FDMTGPU {
public:
    FDMTGPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            SizeType dt_max,
            SizeType dt_step = 1,
            SizeType dt_min  = 0,
            bool use_history = false,
            int device_id    = 0);

    FDMTGPU(const FDMTGPU&)            = delete;
    FDMTGPU& operator=(const FDMTGPU&) = delete;
    FDMTGPU(FDMTGPU&&)                 = delete;
    FDMTGPU& operator=(FDMTGPU&&)      = delete;
    ~FDMTGPU()                         = default;

    static void set_log_level(int level);
    const FDMTPlan& get_plan() const;

    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size);

    void initialise(const float* __restrict waterfall,
                    SizeType waterfall_size,
                    float* __restrict state,
                    SizeType state_size);

    void execute(const float* __restrict waterfall,
                 SizeType waterfall_size,
                 float* __restrict dmt,
                 SizeType dmt_size,
                 bool device_flags);

    void initialise(const float* __restrict waterfall,
                    SizeType waterfall_size,
                    float* __restrict state,
                    SizeType state_size,
                    bool device_flags);

private:
    bool m_use_history;
    int m_device_id;
    FDMTPlan m_plan;
    FDMTPlanContainerD m_plan_d;
    // State buffers
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;
    thrust::device_vector<float> m_history_d;
    static void set_device(int device_id);

    void initialise_device(const float* __restrict waterfall,
                           float* __restrict state);

    void execute_device(const float* __restrict waterfall,
                        SizeType waterfall_size,
                        float* __restrict dmt,
                        SizeType dmt_size);
    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const;
};
