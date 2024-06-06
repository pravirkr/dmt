#include "dmt/ddmt_base.hpp"
#include <cmath>
#include <cstddef>
#include <spdlog/spdlog.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <dmt/ddmt_cpu.hpp>

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min)
    : DDMT(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min) {}

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 const float* dm_arr,
                 SizeType dm_count)
    : DDMT(f_min, f_max, nchans, tsamp, dm_arr, dm_count) {}

void DDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void DDMTCPU::execute(const float* __restrict waterfall,
                      SizeType waterfall_size,
                      float* __restrict dmt,
                      SizeType dmt_size) {
    const auto& plan           = get_plan();
    const auto nchans          = plan.nchans;
    const auto nsamps          = waterfall_size / nchans;
    const auto max_delay       = plan.delay_table.back();
    const auto nsamps_reduced  = nsamps - max_delay;
    const auto out_dm_stride   = nsamps_reduced;
    const auto out_samp_stride = 1;
    const auto in_chan_stride  = nsamps;
    const auto in_samp_stride  = 1;
    const auto* delay_table    = plan.delay_table.data();
    const auto dm_count        = plan.dm_arr.size();

    if (dmt_size != dm_count * nsamps_reduced) {
        spdlog::error("Output buffer size mismatch: expected {}, got {}",
                      dm_count * nsamps_reduced, dmt_size);
        return;
    }

    execute_dedisp(waterfall, in_chan_stride, in_samp_stride, dmt,
                   out_dm_stride, out_samp_stride, delay_table, dm_count,
                   nchans, nsamps_reduced);
}

void DDMTCPU::execute_dedisp(const float* __restrict__ d_in,
                             size_t in_chan_stride,
                             size_t in_samp_stride,
                             float* __restrict__ d_out,
                             size_t out_dm_stride,
                             size_t out_samp_stride,
                             const size_t* __restrict__ delay_table,
                             size_t dm_count,
                             size_t nchans,
                             size_t nsamps_reduced) {
#pragma omp parallel for default(none)                                         \
    shared(d_in, d_out, delay_table, dm_count, nchans, nsamps_reduced,         \
               in_chan_stride, in_samp_stride, out_dm_stride, out_samp_stride)
    for (size_t i_dm = 0; i_dm < dm_count; ++i_dm) {
        const auto& delays = &delay_table[i_dm * nchans];
        const auto out_idx = i_dm * out_dm_stride;
        for (size_t i_samp = 0; i_samp < nsamps_reduced; ++i_samp) {
            float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
            for (size_t i_chan = 0; i_chan < nchans; ++i_chan) {
                const auto& delay = delays[i_chan];
                sum += d_in[i_chan * in_chan_stride +
                            (i_samp + delay) * in_samp_stride];
            }
            d_out[out_idx + i_samp * out_samp_stride] = sum;
        }
    }
}
