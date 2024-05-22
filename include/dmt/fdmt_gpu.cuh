#pragma once

#include <thrust/device_vector.h>
#include <fdmt_base.hpp>

using StShapeTypeD   = int4;
using FDMTCoordTypeD = int2;

struct FDMTCoordMappingD {
    FDMTCoordTypeD head;
    FDMTCoordTypeD tail;
    int offset;
};

struct FDMTPlanD
{  
    // is analogue of std::vector<std::vector<FDMTCoordType>> coordinates;
    // "coordinates_d" is flattened vector "coordinates" 
    // in vector "len_inner_vects_coordinates_cumsum" we store cummulative sums of elements  inner vectors of "coordinates" 
    // So, 
    // len_inner_vects_coordinates_cumsum[0] = 0
    // len_inner_vects_coordinates_cumsum[1] = len_inner_vects_coordinates_cumsum[0] + coordinates[0].size() *2
    // ...
    // len_inner_vects_coordinates_cumsum[n] = len_inner_vects_coordinates_cumsum[n-1] + coordinates[n-1].size() *2
    // ...
    // Remember that always:  len_inner_vects_coordinates_cumsum.size() = NUmIter +1
    thrust::device_vector<int> coordinates_d;
    std::vector<int> lenof_innerVects_coords_cumsum_h;

    // Is an analogues as previous
    thrust::device_vector<int> coordinates_to_copy_d;
    std::vector<int> lenof_innerVects_coords_to_copy_cumsum_h;


    // It is analogue of:   std::vector<std::vector<FDMTCoordMapping>> mappings;
    // each FDMTCoordMapping consists of 5 elements
    // "mappings_d" is flattened vector "mappings" 
    // in vector "len_mappings_cumsum" we store cummulative sums of elements  inner vectors of "mappings" 
    // So, 
    // len_mappings_cumsum[0] = 0
    // len_mappings_cumsum[1] = len_mappings_cumsum[0] + mappings[0].size() *5
    // ...
    // len_mappings_cumsum[n] = len_mappings_cumsum[n-1] + mappings[n-1].size() *5
    // ...
    // Remember that always:  len_mappings_cumsum.size() = NUmIter +1
    thrust::device_vector<int> mappings_d;
    std::vector<int> len_mappings_cumsum_h;


    // Is an analogues as previous
    thrust::device_vector<int> mappings_to_copy_d;
    std::vector<int> len_mappings_to_copy_cumsum_h;


    // It is analogue of state_sub_idx
    // Has size: state_sub_idx_d.size() = m_niters +1
    thrust::device_vector<int>state_sub_idx_d;
    std::vector<int> len_state_sub_idx_cumsum_h;


    // It is analogue of dt_grid
    // Has size: dt_grid_d.size() = m_niters +1
    thrust::device_vector<int>dt_grid_d;
    thrust::device_vector<int> pos_gridInnerVects_d;
    std::vector<int> pos_gridSubVects_h;
    
};
class FDMTGPU : public FDMT {
public:
    FDMTGPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    void execute(const float* __restrict  waterfall, size_t waterfall_size, float* __restrict  dmt,
                 size_t dmt_size) override;
   

    

private:
    thrust::device_vector<float> m_state_in_d;
    thrust::device_vector<float> m_state_out_d;
    thrust::device_vector<int> m_nsamps_d;

    FDMTPlanD m_fdmt_plan_d;

    static void transfer_plan_to_device(const FDMTPlan& plan,
                                        FDMTPlanD& plan_d);
    void initialise(const float* __restrict waterfall, float* __restrict  state) override;

};

std::vector<SizeType>  flatten_mappings(const std::vector<std::vector<FDMTCoordMapping>>& mappings);

__global__
void  kernel_init_fdmt(const float* __restrict waterfall, int* __restrict p_state_sub_idx
    , int* __restrict p_dt_grid, int* __restrict p_pos_gridInnerVects, float* __restrict state, const int nsamps);

__global__
void  kernel_init_fdmt_v1(const float* __restrict waterfall, int* __restrict p_state_sub_idx
    , int* __restrict p_dt_grid, int* __restrict p_pos_gridInnerVects, float* __restrict state, const int nsamps);

__global__
void kernel_execute_iter(const float* __restrict state_in, float* __restrict state_out
    , int* __restrict pcoords_cur//+   
    , int* __restrict pmappings_cur //+
    , int* __restrict pcoords_copy_cur //+
    , int* __restrict pmappings_copy_cur //+
    , int* __restrict pstate_sub_idx_cur //+
    , int* __restrict pstate_sub_idx_prev//+
    , int nsamps //+
    , int  coords_cur_size//+
    , int  coords_copy_cur_size//+
);


__global__
void kernel_execute_iter_v1(const float* __restrict state_in, float* __restrict state_out
    , int* __restrict pcoords_cur//+   
    , int* __restrict pmappings_cur //+
    , int* __restrict pcoords_copy_cur //+
    , int* __restrict pmappings_copy_cur //+
    , int* __restrict pstate_sub_idx_cur //+
    , int* __restrict pstate_sub_idx_prev//+
    , int nsamps //+
    , int  coords_cur_size//+
    , int  coords_copy_cur_size//+
);
