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

struct FDMTPlanD {
    // i = i_iter
    DeviceVector<int> nsubs_d;
    DeviceVector<int> nsamps_d;
    DeviceVector<int> ncoords_d;
    DeviceVector<int> ncoords_to_copy_d;
    DeviceVector<int> subs_iter_idx_d;
    DeviceVector<int> coords_iter_idx_d;
    DeviceVector<int> coords_to_copy_iter_idx_d;
    DeviceVector<int> mappings_iter_idx_d;
    DeviceVector<int> mappings_to_copy_iter_idx_d;
    // i, i+1 = coords_iter_idx_d[i_iter] + i_coord
    DeviceVector<int> coordinates_d;
    DeviceVector<int> coordinates_to_copy_d;
    // i, i+1, ... i+4 = mappings_iter_idx_d[i_iter] + i_coord
    DeviceVector<int> mappings_d;
    DeviceVector<int> mappings_to_copy_d;
    // i = subs_iter_idx_d[i_iter] + isub
    DeviceVector<int> state_sub_idx_d;

    // i = i_sub (only for i_iter = 0)
    DeviceVector<int> ndt_grid_init_d;
    DeviceVector<int> dt_grid_init_sub_idx_d;
    // i = dt_grid_init_sub_idx_d[i_sub] + i_dt
    DeviceVector<int> dt_grid_init_d;

    static FDMTPlanD create_from_plan(const FDMTPlan& plan);
    static FDMTPlanD create_from_plan2(const FDMTPlan& plan);
};

struct DDMTPlanD {
    DeviceVector<float> dm_arr_d;
    DeviceVector<int> delay_arr_d;
    DeviceVector<int> kill_mask_d;
};
