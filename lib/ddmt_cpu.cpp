#include <spdlog/spdlog.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <dmt/ddmt_cpu.hpp>

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min)
    : DDMT(f_min, f_max, nchans, nsamps, tsamp, dm_max, dm_step, dm_min) {}

void DDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}