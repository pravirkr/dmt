#pragma once

#include <thrust/device_vector.h>

#include <dmt/fdmt_base.hpp>

using DtGridTypeD  = thrust::device_vector<SizeType>;
using DtPlanTypeD  = thrust::tuple<SizeType, SizeType, SizeType, SizeType>;
using StShapeTypeD = thrust::tuple<SizeType, SizeType, SizeType>;

struct SubbandPlanD {
    DtGridTypeD dt_grid_d;
    thrust::device_vector<DtPlanTypeD> dt_plan_d;
};

struct FDMTPlanD {
    thrust::device_vector<float> df_top_d;
    thrust::device_vector<float> df_bot_d;
    thrust::device_vector<DtGridTypeD> dt_grid_sub_top_d;
    thrust::device_vector<StShapeTypeD> state_shape_d;
    thrust::device_vector<thrust::device_vector<SubbandPlanD>> sub_plan_d;
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

    FDMTPlanD fdmt_plan_d_;

    FDMTPlan_d transferPlanToDevice(const FDMTPlan& plan) {
        FDMTPlan_d plan_d;

        plan_d.df_top          = plan.df_top;
        plan_d.df_bot          = plan.df_bot;
        plan_d.dt_grid_sub_top = plan.dt_grid_sub_top;
        plan_d.state_shape     = plan.state_shape;

        for (const auto& subPlanVector : plan.sub_plan) {
            thrust::device_vector<SubbandPlanD> subPlanVector_d;

            for (const auto& subPlan : subPlanVector) {
                SubbandPlanD subPlan_d;

                subPlan_d.f_start = subPlan.f_start;
                subPlan_d.f_end   = subPlan.f_end;
                subPlan_d.f_mid1  = subPlan.f_mid1;
                subPlan_d.f_mid2  = subPlan.f_mid2;
                subPlan_d.dt_grid = subPlan.dt_grid;
                subPlan_d.dt_plan = subPlan.dt_plan;

                subPlanVector_d.push_back(subPlan_d);
            }

            plan_d.sub_plan.push_back(subPlanVector_d);
        }

        return plan_d;
    }
};
