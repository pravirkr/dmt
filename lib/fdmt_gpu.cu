#include <cuda_runtime.h>

#include <dmt/fdmt_gpu.hpp>

FDMTGPU::FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps,
                 float tsamp, size_t dt_max, size_t dt_step, size_t dt_min)
    : FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_step, dt_min) {
    // Allocate memory for the state buffers
    const auto& plan      = get_plan();
    const auto state_size = plan.state_shape[0][3] * plan.state_shape[0][4];
    m_state_in_d.resize(state_size, 0.0F);
    m_state_out_d.resize(state_size, 0.0F);
    transfer_plan_to_device(plan, m_fdmt_plan_d);
}

void FDMTGPU::transfer_plan_to_device(const FDMTPlan& plan, FDMTPlanD& plan_d) {
    // Transfer the plan to the device
    const auto niter = plan.state_shape.size();
    plan_d.state_shape_d.resize(niter);
    for (size_t i = 0; i < niter; ++i) {
        const auto& shape = plan.state_shape[i];
        plan_d.state_shape_d[i] = make_int4(shape[0], shape[1], shape[2], shape[3]);
    }
}