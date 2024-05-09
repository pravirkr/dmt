#pragma once

#include <thrust/device_vector.h>

#include <dmt/fdmt_base.hpp>

using StShapeTypeD   = int4;
using FDMTCoordTypeD = int2;

struct FDMTCoordMappingD {
    FDMTCoordTypeD head;
    FDMTCoordTypeD tail;
    SizeType offset;
};

struct FDMTPlanD {
    thrust::device_vector<SizeType> nsubs_d;
    thrust::device_vector<SizeType> ncoords_d;
    thrust::device_vector<SizeType> ncoords_to_copy_d;
    thrust::device_vector<SizeType> nsubs_cumul_d;
    thrust::device_vector<SizeType> ncoords_cumul_d;
    thrust::device_vector<SizeType> ncoords_to_copy_cumul_d;
    // i = i_iter
    thrust::device_vector<StShapeTypeD> state_shape_d;
    // i = i_iter * ncoords_cumul_iter + i_coord
    thrust::device_vector<FDMTCoordTypeD> coordinates_d;
    thrust::device_vector<FDMTCoordMappingD> mappings_d;
    // i = i_iter * ncoords_to_copy_cumul_iter + i_coord_to_copy
    thrust::device_vector<FDMTCoordTypeD> coordinates_to_copy_d;
    thrust::device_vector<FDMTCoordMappingD> mappings_to_copy_d;
    // i = i_iter * nsubs_cumul_iter + isub
    thrust::device_vector<SizeType> state_sub_idx_d;
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

    static void transfer_plan_to_device(const FDMTPlan& plan,
                                        FDMTPlanD& plan_d);
};