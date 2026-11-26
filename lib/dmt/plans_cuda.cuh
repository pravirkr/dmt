#pragma once

#include <algorithm>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/gpu_planned_vector.cuh"
#include "dmt/gpu_utils.cuh"

namespace dmt::plans {

struct FDMTShapeD {
    gpu_utils::PlannedVector<int> nchans;
    gpu_utils::PlannedVector<int> ncoords_sum;
    gpu_utils::PlannedVector<int> ncoords_copy;
    gpu_utils::PlannedVector<int> nsamps;
    gpu_utils::PlannedVector<int> dt_max;
};

struct FDMTCoordDPtrs {
    const int* nsamps;
    const int* buf_offset;
    const int* offset;
    const int* tail_buf_offset;
    const int* tail_nsamps;
    const int* head_buf_offset;
    const int* head_nsamps;
    // Valid-mode cross-block tree history slot (coordinates_sum only):
    // unsigned, so one beam's tree history may hold up to 2^32 - 1 elements.
    const uint32_t* hist_offset;

    __host__ __device__ void update_offsets(int offset_value) {
        nsamps += offset_value;
        buf_offset += offset_value;
        offset += offset_value;
        tail_buf_offset += offset_value;
        tail_nsamps += offset_value;
        head_buf_offset += offset_value;
        head_nsamps += offset_value;
        hist_offset += offset_value;
    }
};

struct FDMTCoordD {
    gpu_utils::PlannedVector<int> nsamps;
    gpu_utils::PlannedVector<int> buf_offset;
    gpu_utils::PlannedVector<int> offset;
    gpu_utils::PlannedVector<int> tail_buf_offset;
    gpu_utils::PlannedVector<int> tail_nsamps;
    gpu_utils::PlannedVector<int> head_buf_offset;
    gpu_utils::PlannedVector<int> head_nsamps;
    gpu_utils::PlannedVector<uint32_t> hist_offset;

    __host__ FDMTCoordDPtrs get_raw_ptrs() const {
        return {
            .nsamps          = thrust::raw_pointer_cast(nsamps.data()),
            .buf_offset      = thrust::raw_pointer_cast(buf_offset.data()),
            .offset          = thrust::raw_pointer_cast(offset.data()),
            .tail_buf_offset = thrust::raw_pointer_cast(tail_buf_offset.data()),
            .tail_nsamps     = thrust::raw_pointer_cast(tail_nsamps.data()),
            .head_buf_offset = thrust::raw_pointer_cast(head_buf_offset.data()),
            .head_nsamps     = thrust::raw_pointer_cast(head_nsamps.data()),
            .hist_offset     = thrust::raw_pointer_cast(hist_offset.data())};
    }
};

struct FDMTCoordCopyDPtrs {
    const int* nsamps;
    const int* buf_offset;
    const int* tail_buf_offset;
    const int* tail_nsamps;

    __host__ __device__ void update_offsets(int offset_value) {
        nsamps += offset_value;
        buf_offset += offset_value;
        tail_buf_offset += offset_value;
        tail_nsamps += offset_value;
    }
};

struct FDMTCoordCopyD {
    gpu_utils::PlannedVector<int> nsamps;
    gpu_utils::PlannedVector<int> buf_offset;
    gpu_utils::PlannedVector<int> tail_buf_offset;
    gpu_utils::PlannedVector<int> tail_nsamps;

    __host__ FDMTCoordCopyDPtrs get_raw_ptrs() const {
        return {.nsamps     = thrust::raw_pointer_cast(nsamps.data()),
                .buf_offset = thrust::raw_pointer_cast(buf_offset.data()),
                .tail_buf_offset =
                    thrust::raw_pointer_cast(tail_buf_offset.data()),
                .tail_nsamps = thrust::raw_pointer_cast(tail_nsamps.data())};
    }
};

struct FDMTCoordGridD {
    gpu_utils::PlannedVector<int> dt_grid;
    gpu_utils::PlannedVector<int> ndt;
    gpu_utils::PlannedVector<int> coord_offset;
};

struct FDMTPlanContainerD {
    FDMTShapeD state_shape;
    FDMTCoordGridD grids0;
    FDMTCoordD coordinates_sum;
    FDMTCoordCopyD coordinates_copy;
};

namespace detail {
__host__ inline void
transfer_coords_impl(const std::vector<std::vector<FDMTCoord>>& host_coords,
                     FDMTCoordD& device_coords) {
    std::vector<int> nsamps, buf_offset, offset, tail_buf_offset, tail_nsamps,
        head_buf_offset, head_nsamps;
    std::vector<uint32_t> hist_offset;

    // Calculate total size for efficient allocation
    size_t total_size = 0;
    for (const auto& host_coords_iter : host_coords) {
        total_size += host_coords_iter.size();
    }

    // Reserve space once
    nsamps.reserve(total_size);
    buf_offset.reserve(total_size);
    offset.reserve(total_size);
    tail_buf_offset.reserve(total_size);
    tail_nsamps.reserve(total_size);
    head_buf_offset.reserve(total_size);
    head_nsamps.reserve(total_size);
    hist_offset.reserve(total_size);

    // Fill vectors
    for (const auto& host_coords_iter : host_coords) {
        for (const auto& coord : host_coords_iter) {
            nsamps.emplace_back(coord.nsamps);
            buf_offset.emplace_back(coord.buf_offset);
            offset.emplace_back(coord.delay);
            tail_buf_offset.emplace_back(coord.tail_buf_offset);
            tail_nsamps.emplace_back(coord.tail_nsamps);
            head_buf_offset.emplace_back(coord.head_buf_offset);
            head_nsamps.emplace_back(coord.head_nsamps);
            hist_offset.emplace_back(static_cast<uint32_t>(coord.hist_offset));
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
    device_coords.hist_offset     = hist_offset;
}

__host__ inline void transfer_copy_coords_impl(
    const std::vector<std::vector<FDMTCoord>>& host_coords,
    FDMTCoordCopyD& device_coords) {
    std::vector<int> nsamps, buf_offset, tail_buf_offset, tail_nsamps;

    size_t total_size = 0;
    for (const auto& host_coords_iter : host_coords) {
        total_size += host_coords_iter.size();
    }

    nsamps.reserve(total_size);
    buf_offset.reserve(total_size);
    tail_buf_offset.reserve(total_size);
    tail_nsamps.reserve(total_size);

    for (const auto& host_coords_iter : host_coords) {
        for (const auto& coord : host_coords_iter) {
            nsamps.emplace_back(coord.nsamps);
            buf_offset.emplace_back(coord.buf_offset);
            tail_buf_offset.emplace_back(coord.tail_buf_offset);
            tail_nsamps.emplace_back(coord.tail_nsamps);
        }
    }

    device_coords.nsamps          = nsamps;
    device_coords.buf_offset      = buf_offset;
    device_coords.tail_buf_offset = tail_buf_offset;
    device_coords.tail_nsamps     = tail_nsamps;
}
} // namespace detail

__host__ inline void transfer_fdmt_plan_to_device(const FDMTPlanContainer& plan,
                                                  FDMTPlanContainerD& plan_d) {
    // Built on the host and uploaded once per table (deferrable: see
    // gpu_planned_vector.cuh).
    const auto niter_size = plan.state_shape.size();
    std::vector<int> nchans(niter_size);
    std::vector<int> ncoords_sum(niter_size);
    std::vector<int> ncoords_copy(niter_size);
    std::vector<int> nsamps(niter_size);
    std::vector<int> dt_max(niter_size);
    for (SizeType i = 0; i < niter_size; ++i) {
        nchans[i]       = static_cast<int>(plan.state_shape[i].nchans);
        ncoords_sum[i]  = static_cast<int>(plan.state_shape[i].ncoords_sum);
        ncoords_copy[i] = static_cast<int>(plan.state_shape[i].ncoords_copy);
        nsamps[i]       = static_cast<int>(plan.state_shape[i].nsamps);
        dt_max[i]       = static_cast<int>(plan.state_shape[i].dt_max);
    }
    plan_d.state_shape.nchans       = nchans;
    plan_d.state_shape.ncoords_sum  = ncoords_sum;
    plan_d.state_shape.ncoords_copy = ncoords_copy;
    plan_d.state_shape.nsamps       = nsamps;
    plan_d.state_shape.dt_max       = dt_max;
    // Grids of the first iteration only.
    const auto& grid_init = plan.grids[0];
    const auto nsubs      = grid_init.size();
    std::vector<int> ndt(nsubs);
    std::vector<int> coord_offset(nsubs);
    std::vector<int> dt_grid;
    dt_grid.reserve(plan.state_shape[0].ncoords);
    for (SizeType i = 0; i < nsubs; ++i) {
        ndt[i]          = static_cast<int>(grid_init[i].ndt);
        coord_offset[i] = static_cast<int>(grid_init[i].coord_offset);
        for (const auto dt : grid_init[i].dt_grid) {
            dt_grid.push_back(static_cast<int>(dt));
        }
    }
    plan_d.grids0.ndt          = ndt;
    plan_d.grids0.coord_offset = coord_offset;
    plan_d.grids0.dt_grid      = dt_grid;

    // Merge coordinates.
    detail::transfer_coords_impl(plan.coordinates_sum, plan_d.coordinates_sum);
    detail::transfer_copy_coords_impl(plan.coordinates_copy,
                                      plan_d.coordinates_copy);
}

} // namespace dmt::plans
