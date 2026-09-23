#pragma once

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

/**
 * @brief Configures the number of OpenMP worker threads for CPU parallel
 * execution.
 *
 * If DMT was compiled without OpenMP support (`DMT_ENABLE_OPENMP` undefined),
 * this logs a warning if `nthreads > 1` and defaults to 1 thread.
 *
 * @param nthreads Target number of OpenMP threads (<= 0 sets to
 * omp_get_max_threads()).
 * @return Actual number of threads configured.
 */
inline int set_dmt_openmp_threads(int nthreads) {
#ifdef DMT_ENABLE_OPENMP
    if (nthreads <= 0) {
        nthreads = omp_get_max_threads();
    }
    omp_set_num_threads(nthreads);
    spdlog::debug("set_openmp_threads: Using {} OpenMP threads", nthreads);
#else
    // Warn if nthreads > 1 but OpenMP is not enabled
    if (nthreads > 1) {
        spdlog::warn(
            "set_openmp_threads: Warning - nthreads > 1 specified, but "
            "OpenMP is not enabled (DMT_ENABLE_OPENMP not defined).");
    }
    nthreads = 1;
#endif
    return nthreads;
}