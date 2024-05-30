
#include "dmt/ddmt_base.hpp"
#include <curand_mtgp32_kernel.h>
#include <dmt/cuda_utils.cuh>
#include <dmt/ddmt_gpu.hpp>

// Kernel tuning parameters
constexpr DDMTSize kDDMTBlockSize      = 256;
constexpr DDMTSize kDDMTBlockSamps     = 8;
constexpr DDMTSize kDDMTSampsPerThread = 2; // 4 is better for Fermi?
// Constant memory lookup table (64 KB)
constexpr DDMTSize kConstMemorySize = 64L * 1024L;
constexpr DDMTSize kDdmtMaxNchans   = kConstMemorySize / 2 / sizeof(DDMTFloat);

__constant__ DDMTFloat c_delay_arr[kDdmtMaxNchans];
__constant__ DDMTFloat c_killmask[kDdmtMaxNchans];

// Summation type metafunction
template <int InNbits>
using DDMTSumType = std::conditional_t<InNbits == 32, DDMTSize, DDMTWord>;

template <int NBITS, typename T>
inline __host__ __device__ T extract_subword(T value, int idx) {
    constexpr T kMask = (1 << NBITS) - 1;
    return (value >> (idx * NBITS)) & kMask;
}

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
// InNbits: Possible values - 8, 16, 32
template <int InNbits, int SampsPerThread>
__global__ void dedisperse_kernel(const DDMTWord* d_in,
                                  DDMTSize nsamps,
                                  DDMTSize nsamps_reduced,
                                  DDMTSize nsamp_blocks,
                                  DDMTSize stride,
                                  DDMTSize dm_count,
                                  DDMTSize dm_stride,
                                  DDMTSize ndm_blocks,
                                  DDMTSize nchans,
                                  DDMTSize chan_stride,
                                  DDMTFloat* d_out,
                                  DDMTSize out_stride,
                                  const DDMTFloat* d_dm_list) {
    constexpr uint64_t kChansPerWord = sizeof(DDMTWord) * 8 / InNbits;
    // Compute the block and thread indices
    DDMTSize isamp = blockIdx.x * blockDim.x + threadIdx.x;
    DDMTSize idm   = (blockIdx.y % ndm_blocks) * blockDim.y + threadIdx.y;
    const DDMTSize nsamp_threads = nsamp_blocks * blockDim.x;
    const DDMTSize ndm_threads   = ndm_blocks * blockDim.y;

    // Iterate over grids of DMs
    for (; idm < dm_count; idm += ndm_threads) {
        DDMTFloat dm = d_dm_list[idm * dm_stride];
        for (; isamp < nsamps_reduced; isamp += nsamp_threads) {
            DDMTSumType<InNbits> sum[SampsPerThread];
#pragma unroll
            for (DDMTSize s = 0; s < SampsPerThread; ++s) {
                sum[s] = 0;
            }
            // Loop over channel words
            for (DDMTSize ichan_word = 0; ichan_word < nchans;
                 ichan_word += kChansPerWord) {
                // Pre-compute the memory offset
                DDMTSize offset = isamp * SampsPerThread +
                                  ichan_word / kChansPerWord * stride;

                // Loop over channel subwords
                for (DDMTSize ichan_sub = 0; ichan_sub < kChansPerWord;
                     ++ichan_sub) {
                    DDMTSize ichan = (ichan_word + ichan_sub) * chan_stride;
                    DDMTSize delay = __float2uint_rn(dm * c_delay_arr[ichan]);
#pragma unroll
                    for (DDMTSize s = 0; s < SampsPerThread; ++s) {
                        DDMTWord sample = d_in[offset + s + delay];
                        sum[s] += c_killmask[ichan] *
                                  extract_subword<InNbits>(sample, ichan_sub);
                    }
                }
            }
            DDMTSize out_idx = (isamp * SampsPerThread + idm * out_stride);
#pragma unroll
            for (DDMTSize s = 0; s < SampsPerThread; ++s) {
                if (isamp * SampsPerThread + s < nsamps) {
                    d_out[out_idx + s] = static_cast<DDMTFloat>(sum[s]);
                }
            }
        }
    }
}

void dedisperse(const DDMTWord* d_in,
                DDMTSize in_stride,
                DDMTSize nsamps,
                DDMTSize in_nbits,
                DDMTSize nchans,
                DDMTSize chan_stride,
                const DDMTFloat* d_dm_list,
                DDMTSize dm_count,
                DDMTSize dm_stride,
                DDMTFloat* d_out,
                DDMTSize out_stride,
                cudaStream_t stream) {
    // Block/grid dimensions x and y represent time samples and DMs respectively
    dim3 block(kDDMTBlockSamps, kDDMTBlockSize / kDDMTBlockSamps);
    DDMTSize nsamp_blocks = (nsamps + kDDMTSampsPerThread * block.x - 1) /
                            (kDDMTSampsPerThread * block.x);
    DDMTSize ndm_blocks = (dm_count + block.y - 1) / block.y;
    dim3 grid(nsamp_blocks, std::min(ndm_blocks, static_cast<DDMTSize>(65535)));
    DDMTSize nsamps_reduced =
        (nsamps + kDDMTSampsPerThread - 1) / kDDMTSampsPerThread;

    // Execute the kernel
#define DEDISP_CALL_KERNEL(NBITS)
    dedisperse_kernel<NBITS, kDDMTSampsPerThread>
        <<<grid, block, 0, stream>>>(d_in, nsamps, nsamps_reduced, nsamp_blocks,
                                     in_stride, dm_count, dm_stride, ndm_blocks,
                                     nchans, chan_stride, d_out, out_stride,
                                     d_dm_list)
        // Note: Here we dispatch dynamically on nbits for supported values
        switch (in_nbits) {
    case 8:
        DEDISP_CALL_KERNEL(8);
        break;
    case 16:
        DEDISP_CALL_KERNEL(16);
        break;
    case 32:
        DEDISP_CALL_KERNEL(32);
        break;
    default: /* should never be reached */
        break;
    }
#undef DEDISP_CALL_KERNEL
}

DDMTGPU::DDMTGPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min)
    : DDMT(f_min, f_max, nchans, nsamps, tsamp, dm_max, dm_step, dm_min) {
    transfer_plan_to_device(m_ddmt_plan, m_ddmt_plan_d);

    // Compute the problem decomposition
    DDMTSize nsamps_computed = nsamps - plan->max_delay;
}

void DDMTGPU::execute(const float* __restrict waterfall,
                      size_t waterfall_size,
                      float* __restrict dmt,
                      size_t dmt_size) {
    execute(waterfall, waterfall_size, dmt, dmt_size, false);
}

void DDMTGPU::execute(const float* __restrict waterfall,
                      size_t waterfall_size,
                      float* __restrict dmt,
                      size_t dmt_size,
                      bool device_flags) {
    if (device_flags) {
        execute_device(waterfall, waterfall_size, dmt, dmt_size);
    } else {
        thrust::device_vector<float> waterfall_d(waterfall,
                                                 waterfall + waterfall_size);
        thrust::device_vector<float> dmt_d(dmt, dmt + dmt_size);
        execute_device(thrust::raw_pointer_cast(waterfall_d.data()),
                       waterfall_size, thrust::raw_pointer_cast(dmt_d.data()),
                       dmt_size);
        thrust::copy(dmt_d.begin(), dmt_d.end(), dmt);
        error_checker::check_cuda("thrust::copy failed");
    }
}

void DDMTGPU::transfer_plan_to_device(const DDMTPlan& plan, DDMTPlanD& plan_d) {
    plan_d.delay_arr_d = plan.delay_arr;
    plan_d.kill_mask_d = plan.kill_mask;
    cudaMemcpyToSymbolAsync(
        delay_arr_c, thrust::raw_pointer_cast(plan_d.delay_arr_d.data()),
        nchans * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
    error_checker::check_cuda(
        "Failed to copy delay table to device constant memory");
    cudaMemcpyToSymbolAsync(
        kill_mask_c, thrust::raw_pointer_cast(plan_d.kill_mask_d.data()),
        nchans * sizeof(bool), 0, cudaMemcpyDeviceToDevice, stream);
    error_checker::check_cuda(
        "Failed to copy kill mask to device constant memory");
}
