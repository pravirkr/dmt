#pragma once

#include <numeric>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::plans {

// Static helper functions
template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& vec_2d) {
    std::vector<T> flattened;
    for (const auto& vec : vec_2d) {
        flattened.insert(flattened.end(), vec.begin(), vec.end());
    }
    return flattened;
}

template <typename T>
std::vector<T> cumulative_sum(const std::vector<T>& vec) {
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

    __host__ __device__ void update_offsets(int offset_value) {
        nsamps += offset_value;
        buf_offset += offset_value;
        offset += offset_value;
        tail_buf_offset += offset_value;
        tail_nsamps += offset_value;
        head_buf_offset += offset_value;
        head_nsamps += offset_value;
    }
};

struct FDMTCoordD {
    DeviceVector<int> nsamps;
    DeviceVector<int> buf_offset;
    DeviceVector<int> offset;
    DeviceVector<int> tail_buf_offset;
    DeviceVector<int> tail_nsamps;
    DeviceVector<int> head_buf_offset;
    DeviceVector<int> head_nsamps;

    __host__ FDMTCoordDPtrs get_raw_ptrs() const {
        return {
            .nsamps          = thrust::raw_pointer_cast(nsamps.data()),
            .buf_offset      = thrust::raw_pointer_cast(buf_offset.data()),
            .offset          = thrust::raw_pointer_cast(offset.data()),
            .tail_buf_offset = thrust::raw_pointer_cast(tail_buf_offset.data()),
            .tail_nsamps     = thrust::raw_pointer_cast(tail_nsamps.data()),
            .head_buf_offset = thrust::raw_pointer_cast(head_buf_offset.data()),
            .head_nsamps     = thrust::raw_pointer_cast(head_nsamps.data())};
    }
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

namespace detail {
__host__ inline void
transfer_coords_impl(const std::vector<std::vector<FDMTCoord>>& host_coords,
                     FDMTCoordD& device_coords) {
    std::vector<int> nsamps, buf_offset, i_coord_tail, i_coord_head, offset,
        tail_buf_offset, tail_nsamps, head_buf_offset, head_nsamps;

    // Calculate total size for efficient allocation
    size_t total_size = 0;
    for (const auto& host_coords_iter : host_coords) {
        total_size += host_coords_iter.size();
    }

    // Reserve space once
    nsamps.reserve(total_size);
    buf_offset.reserve(total_size);
    i_coord_tail.reserve(total_size);
    i_coord_head.reserve(total_size);
    offset.reserve(total_size);
    tail_buf_offset.reserve(total_size);
    tail_nsamps.reserve(total_size);
    head_buf_offset.reserve(total_size);
    head_nsamps.reserve(total_size);

    // Fill vectors
    for (const auto& host_coords_iter : host_coords) {
        for (const auto& coord : host_coords_iter) {
            nsamps.emplace_back(coord.nsamps);
            buf_offset.emplace_back(coord.buf_offset);
            i_coord_tail.emplace_back(coord.i_coord_tail);
            i_coord_head.emplace_back(coord.i_coord_head);
            offset.emplace_back(coord.offset);
            tail_buf_offset.emplace_back(coord.tail_buf_offset);
            tail_nsamps.emplace_back(coord.tail_nsamps);
            head_buf_offset.emplace_back(coord.head_buf_offset);
            head_nsamps.emplace_back(coord.head_nsamps);
        }
    }

    // Transfer to device vectors
    device_coords.nsamps          = nsamps;
    device_coords.buf_offset      = buf_offset;
    device_coords.offset          = offset;
    device_coords.tail_buf_offset = tail_buf_offset;
    device_coords.tail_nsamps     = tail_nsamps;
    device_coords.head_buf_offset = head_buf_offset;
    device_coords.head_nsamps     = head_nsamps;
}
} // namespace detail

__host__ inline void transfer_fdmt_plan_to_device(const FDMTPlanContainer& plan,
                                                  FDMTPlanContainerD& plan_d) {
    const auto niter_size = static_cast<int>(plan.state_shape.size());
    // Transfer shape to device
    plan_d.state_shape.nchans.resize(niter_size);
    plan_d.state_shape.ncoords_sum.resize(niter_size);
    plan_d.state_shape.ncoords_copy.resize(niter_size);
    plan_d.state_shape.nsamps.resize(niter_size);
    plan_d.state_shape.dt_max.resize(niter_size);
    for (int i = 0; i < niter_size; ++i) {
        plan_d.state_shape.nchans[i] =
            static_cast<int>(plan.state_shape[i].nchans);
        plan_d.state_shape.ncoords_sum[i] =
            static_cast<int>(plan.state_shape[i].ncoords_sum);
        plan_d.state_shape.ncoords_copy[i] =
            static_cast<int>(plan.state_shape[i].ncoords_copy);
        plan_d.state_shape.nsamps[i] =
            static_cast<int>(plan.state_shape[i].nsamps);
        plan_d.state_shape.dt_max[i] =
            static_cast<int>(plan.state_shape[i].dt_max);
    }
    // Transfer grids to device only for the first iteration
    const auto& grid_init = plan.grids[0];
    const auto nsubs      = static_cast<int>(grid_init.size());
    plan_d.grids0.ndt.resize(nsubs);
    plan_d.grids0.coord_offset.resize(nsubs);
    plan_d.grids0.dt_grid.reserve(plan.state_shape[0].ncoords);
    for (int i = 0; i < nsubs; ++i) {
        plan_d.grids0.ndt[i] = static_cast<int>(grid_init[i].ndt);
        plan_d.grids0.coord_offset[i] =
            static_cast<int>(grid_init[i].coord_offset);
        plan_d.grids0.dt_grid.insert(plan_d.grids0.dt_grid.end(),
                                     grid_init[i].dt_grid.begin(),
                                     grid_init[i].dt_grid.end());
    }

    // Transfer coordinates to device
    detail::transfer_coords_impl(plan.coordinates, plan_d.coordinates);
    detail::transfer_coords_impl(plan.coordinates_sum, plan_d.coordinates_sum);
    detail::transfer_coords_impl(plan.coordinates_copy,
                                 plan_d.coordinates_copy);
}

struct DDMTPlanD {
    DeviceVector<float> dm_arr_d;
    DeviceVector<int> delay_arr_d;
    DeviceVector<int> kill_mask_d;
};

} // namespace dmt::plans