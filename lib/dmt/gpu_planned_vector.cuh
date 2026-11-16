#pragma once

// Deferred device allocation for memory estimates: planned_vector.hpp with
// thrust::device_vector storage.

#include <thrust/device_vector.h>

#include "dmt/gpu_utils.cuh"
#include "dmt/planned_vector.hpp"

namespace dmt::planned {

template <typename T> struct ZeroBytes<thrust::device_vector<T>> {
    static void apply(thrust::device_vector<T>& v) {
        if (!v.empty()) {
            gpu_utils::check_gpu_call(
                cudaMemset(thrust::raw_pointer_cast(v.data()), 0,
                           v.size() * sizeof(T)),
                "Planned: zero fill");
        }
    }
};

} // namespace dmt::planned

namespace dmt::gpu_utils {

using planned::DeferScope;
using planned::PendingList;

/// A thrust::device_vector<T> whose allocation a DeferScope can defer.
template <typename T>
using PlannedVector = planned::Planned<thrust::device_vector<T>>;

} // namespace dmt::gpu_utils
