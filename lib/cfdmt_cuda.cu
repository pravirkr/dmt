#include <string>

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <dmt/cfdmt/cfdmt_cuda.hpp>

#include "dmt/cuda_utils.cuh"

CohFDMTCUDA::CohFDMTCUDA(float f_center,
                       float sub_bw,
                       SizeType nsub,
                       float tbin,
                       SizeType nbin,
                       SizeType nfft,
                       float tp,
                       float dm_max,
                       float dm_min,
                       SizeType noverlap)
    : m_plan(f_center,
             sub_bw,
             nsub,
             tbin,
             nbin,
             nfft,
             tp,
             dm_max,
             dm_min,
             noverlap) {
    initialise();
}

void CohFDMTCUDA::execute(const uint8_t* __restrict__ data_in,
                         SizeType in_size,
                         const std::string& in_order,
                         float* __restrict__ dmt,
                         SizeType dmt_size) {
    execute(data_in, in_size, in_order, dmt, dmt_size, false);
}

void CohFDMTCUDA::execute(const uint8_t* __restrict__ data_in,
                         SizeType in_size,
                         const std::string& in_order,
                         float* __restrict__ dmt,
                         SizeType dmt_size,
                         bool device_flags) {
    if (device_flags) {
        execute_device(data_in, in_size, in_order, dmt, dmt_size);
    } else {
        thrust::device_vector<uint8_t> data_in_d(data_in, data_in + in_size);
        thrust::device_vector<float> dmt_d(dmt, dmt + dmt_size);
        execute_device(thrust::raw_pointer_cast(data_in_d.data()), in_size,
                       in_order, thrust::raw_pointer_cast(dmt_d.data()),
                       dmt_size);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt);
        error_checker::check_cuda("thrust::copy failed");
    }
}

void CohFDMTCUDA::initialise() {}

void CohFDMTCUDA::execute_device(const uint8_t* __restrict__ data_in,
                                SizeType in_size,
                                const std::string& in_order,
                                float* __restrict__ dmt,
                                SizeType dmt_size) {}
