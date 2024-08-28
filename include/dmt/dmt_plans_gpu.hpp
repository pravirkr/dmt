#pragma once

#include <numeric>
#include <vector>

#include <thrust/device_vector.h>

#include <dmt/dmt_plans.hpp>

template <typename T> using DeviceVector = thrust::device_vector<T>;

// Static helper functions
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec_2d) {
    std::vector<T> flattened;
    for (const auto& vec : vec_2d) {
        flattened.insert(flattened.end(), vec.begin(), vec.end());
    }
    return flattened;
}

template <typename T> std::vector<T> cumulative_sum(const std::vector<T>& vec) {
    std::vector<T> result(vec.size() + 1, 0);
    std::partial_sum(vec.begin(), vec.end(), result.begin() + 1);
    return result;
}

struct FDMTShapeD {
    DeviceVector<int> nchans;
    DeviceVector<int> ncoords_sum;
    DeviceVector<int> ncoords_copy;
    DeviceVector<int> nsamps;
    DeviceVector<int> dt_max;
};

struct FDMTCoordDPtrs {
    const int* nsamps;
    const int* buf_offset;
    const int* offset;
    const int* tail_buf_offset;
    const int* tail_nsamps;
    const int* head_buf_offset;
    const int* head_nsamps;

    __host__ __device__ void update_offsets(int offset_value);
};

struct FDMTCoordD {
    DeviceVector<int> nsamps;
    DeviceVector<int> buf_offset;
    DeviceVector<int> offset;
    DeviceVector<int> tail_buf_offset;
    DeviceVector<int> tail_nsamps;
    DeviceVector<int> head_buf_offset;
    DeviceVector<int> head_nsamps;

    FDMTCoordDPtrs get_raw_ptrs() const;
};

struct FDMTCoordGridD {
    DeviceVector<int> dt_grid;
    DeviceVector<int> ndt;
    DeviceVector<int> coord_offset;
};

struct FDMTPlanContainerD {
    FDMTShapeD state_shape;
    FDMTCoordGridD grids0;
    FDMTCoordD coordinates;
    FDMTCoordD coordinates_sum;
    FDMTCoordD coordinates_copy;
};

void transfer_fdmt_plan_to_device(const FDMTPlanContainer& plan,
                                  FDMTPlanContainerD& plan_d);

struct DDMTPlanD {
    DeviceVector<float> dm_arr_d;
    DeviceVector<int> delay_arr_d;
    DeviceVector<int> kill_mask_d;
};