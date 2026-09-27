#pragma once

#include <cstdint>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Direct dedispersion kernel, float input, with SAMPS_PER_THREAD
 * register unrolling.
 * @details
 * Reads the per-(dm,chan) delay table and kill mask from global memory
 * (device-resident DDMTPlanD arrays, one copy per DDMTCUDA instance).
 * Deliberately *not* backed by __constant__ memory: constant memory is one
 * resource shared by the whole device/process, and DDMTCUDA's device-span
 * execute() overloads are meant to let independent instances pipeline
 * concurrently on different streams -- a shared __constant__ table would
 * let one instance's upload race another's in-flight kernel.
 *
 * `d_in`/`d_out` are beam-major: blockIdx.z selects the beam, offsetting
 * into (nbeams, nchans, in_chan_stride) and (nbeams, dm_count,
 * out_dm_stride) respectively. At nbeams == 1 (gridDim.z == 1,
 * in_beam_stride/out_beam_stride == 0 in practice since blockIdx.z is
 * always 0) this is identical to the pre-multi-beam kernel.
 */
template <int SAMPS_PER_THREAD = 2>
__global__ void ddmt_kernel_float(const float* __restrict__ d_in,
                                  int in_chan_stride,
                                  int in_beam_stride,
                                  float* __restrict__ d_out,
                                  int out_dm_stride,
                                  int out_beam_stride,
                                  const int* __restrict__ delay_table,
                                  const int* __restrict__ kill_mask,
                                  int nchans,
                                  int dm_count,
                                  int nsamps_reduced) {
    const auto isamp_base =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x) *
        SAMPS_PER_THREAD;
    if (isamp_base >= nsamps_reduced) {
        return;
    }
    const auto i_beam  = static_cast<int>(blockIdx.z);
    const auto* d_in_b = d_in + (i_beam * in_beam_stride);
    auto* d_out_b      = d_out + (i_beam * out_beam_stride);

    for (auto idm = static_cast<int>(blockIdx.y); idm < dm_count;
         idm += static_cast<int>(gridDim.y)) {
        const auto* delays = &delay_table[idm * nchans];

        float sum[SAMPS_PER_THREAD] = {0.0F};

        for (int ichan = 0; ichan < nchans; ++ichan) {
            if (!kill_mask[ichan]) {
                continue;
            }
            const int delay = delays[ichan];
            const int base  = (ichan * in_chan_stride) + isamp_base + delay;

#pragma unroll
            for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
                if (isamp_base + s < nsamps_reduced) {
                    sum[s] += d_in_b[base + s];
                }
            }
        }

        const auto out_offset = idm * out_dm_stride;
#pragma unroll
        for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
            if (isamp_base + s < nsamps_reduced) {
                d_out_b[out_offset + isamp_base + s] = sum[s];
            }
        }
    }
}

/**
 * @brief Direct dedispersion kernel, packed-integer input, with
 * SAMPS_PER_THREAD register unrolling. See ddmt_kernel_float's doc comment
 * for why delay/kill-mask lookups are global memory, not __constant__, and
 * for the beam-major (blockIdx.z) layout.
 * @param sample_offset Added to every read index before applying delay;
 * see execute()'s byte-alignment comment in lib/ddmt_cuda.cu for NBITS < 8.
 */
template <unsigned NBITS, int SAMPS_PER_THREAD = 2>
__global__ void ddmt_kernel_packed(const uint8_t* __restrict__ d_in,
                                   SizeType row_bytes,
                                   SizeType in_beam_stride,
                                   SizeType sample_offset,
                                   int32_t* __restrict__ d_out,
                                   int out_dm_stride,
                                   int out_beam_stride,
                                   const int* __restrict__ delay_table,
                                   const int* __restrict__ kill_mask,
                                   int nchans,
                                   int dm_count,
                                   int nsamps_reduced) {
    const auto isamp_base =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x) *
        SAMPS_PER_THREAD;
    if (isamp_base >= nsamps_reduced) {
        return;
    }
    const auto i_beam  = static_cast<SizeType>(blockIdx.z);
    const auto* d_in_b = d_in + (i_beam * in_beam_stride);
    auto* d_out_b      = d_out + (static_cast<int>(i_beam) * out_beam_stride);

    for (auto idm = static_cast<int>(blockIdx.y); idm < dm_count;
         idm += static_cast<int>(gridDim.y)) {
        const auto* delays = &delay_table[idm * nchans];

        int32_t sum[SAMPS_PER_THREAD] = {0};

        for (int ichan = 0; ichan < nchans; ++ichan) {
            if (!kill_mask[ichan]) {
                continue;
            }
            const int delay = delays[ichan];
            const auto* row =
                d_in_b + (static_cast<SizeType>(ichan) * row_bytes);

#pragma unroll
            for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
                if (isamp_base + s < nsamps_reduced) {
                    const auto sample_idx =
                        sample_offset +
                        static_cast<SizeType>(isamp_base + s + delay);
                    const auto sample =
                        bit_pack_utils::read_packed_sample<NBITS>(row,
                                                                  sample_idx);
                    sum[s] += static_cast<int32_t>(sample);
                }
            }
        }

        const auto out_offset = idm * out_dm_stride;
#pragma unroll
        for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
            if (isamp_base + s < nsamps_reduced) {
                d_out_b[out_offset + isamp_base + s] = sum[s];
            }
        }
    }
}

/**
 * @brief Direct dedispersion kernel for time-major packed inputs (nsamps,
 * nchans), where at each time sample, channels are packed together
 * consecutively. See ddmt_kernel_float's doc comment for the beam-major
 * (blockIdx.z) layout.
 */
template <unsigned NBITS, int SAMPS_PER_THREAD = 2>
__global__ void ddmt_kernel_time_major(const uint8_t* __restrict__ d_in,
                                       SizeType samp_bytes,
                                       SizeType in_beam_stride,
                                       int32_t* __restrict__ d_out,
                                       int out_dm_stride,
                                       int out_beam_stride,
                                       const int* __restrict__ delay_table,
                                       const int* __restrict__ kill_mask,
                                       int nchans,
                                       int dm_count,
                                       int nsamps_reduced) {
    const auto isamp_base =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x) *
        SAMPS_PER_THREAD;
    if (isamp_base >= nsamps_reduced) {
        return;
    }
    const auto i_beam  = static_cast<SizeType>(blockIdx.z);
    const auto* d_in_b = d_in + (i_beam * in_beam_stride);
    auto* d_out_b      = d_out + (static_cast<int>(i_beam) * out_beam_stride);

    for (auto idm = static_cast<int>(blockIdx.y); idm < dm_count;
         idm += static_cast<int>(gridDim.y)) {
        const auto* delays            = &delay_table[idm * nchans];
        int32_t sum[SAMPS_PER_THREAD] = {0};

        for (int ichan = 0; ichan < nchans; ++ichan) {
            if (!kill_mask[ichan]) {
                continue;
            }
            const int delay = delays[ichan];

#pragma unroll
            for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
                if (isamp_base + s < nsamps_reduced) {
                    const auto samp_idx =
                        static_cast<SizeType>(isamp_base + s + delay);
                    const auto* samp_ptr = d_in_b + (samp_idx * samp_bytes);
                    const auto sample =
                        bit_pack_utils::read_packed_sample<NBITS>(
                            samp_ptr, static_cast<SizeType>(ichan));
                    sum[s] += static_cast<int32_t>(sample);
                }
            }
        }

        const auto out_offset = idm * out_dm_stride;
#pragma unroll
        for (int s = 0; s < SAMPS_PER_THREAD; ++s) {
            if (isamp_base + s < nsamps_reduced) {
                d_out_b[out_offset + isamp_base + s] = sum[s];
            }
        }
    }
}

} // namespace dmt::algorithms
