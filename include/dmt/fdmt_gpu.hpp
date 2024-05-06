#pragma once

#include <thrust/device_vector.h>

#include <dmt/fdmt_base.hpp>

struct FDMTPlanD {
    thrust::device_vector<SizeType> state_shape_d;
    thrust::device_vector<SizeType> state_idx_d;
    thrust::device_vector<SizeType> dt_grid_d;
    thrust::device_vector<SizeType> dt_plan_d;
};

class FDMTGPU : public FDMT {
public:
    FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    void execute(const float* waterfall, size_t waterfall_size, float* dmt,
                 size_t dmt_size) override;
    void initialise(const float* waterfall, float* state) override;

private:
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;

    FDMTPlanD m_fdmt_plan_d;

    FDMTPlanD transfer_plan_to_device() {
        const auto& plan = get_plan();
        FDMTPlanD plan_d;
        for (const auto& state_shape_iter : plan.state_shape) {
            for (const auto& shape : state_shape_iter) {
                plan_d.state_shape_d.push_back(shape);
            }
        }
        // flatten sub_plan and transfer to device
        for (const auto& sub_plan_iter : plan.sub_plan) {
            for (const auto& sub_plan : sub_plan_iter) {
                plan_d.state_idx_d.push_back(sub_plan.state_idx);
                for (const auto& dt : sub_plan.dt_grid) {
                    plan_d.dt_grid_d.push_back(dt);
                }
                for (const auto& dt_tuple : sub_plan.dt_plan) {
                    for (const auto& idt : dt_tuple) {
                        plan_d.dt_plan_d.push_back(idt);
                    }
                }
            }
        }
        return plan_d;
    };
};