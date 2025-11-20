#pragma once

#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

class DDMTCPU {
public:
    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F,
            int nthreads = 1);

    DDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            float tsamp,
            std::span<const float> dm_arr,
            int nthreads = 1);

    ~DDMTCPU();
    DDMTCPU(DDMTCPU&&) noexcept;
    DDMTCPU& operator=(DDMTCPU&&) noexcept;
    DDMTCPU(const DDMTCPU&)            = delete;
    DDMTCPU& operator=(const DDMTCPU&) = delete;

    const plans::DDMTPlan& get_plan() const noexcept;
    void execute(std::span<const float> waterfall, std::span<float> dmt);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#ifdef DMT_ENABLE_CUDA
class DDMTCUDA {
public:
    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min  = 0.0F,
             int device_id = 0);

    DDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const std::vector<float>& dm_arr,
             int device_id = 0);

    ~DDMTCUDA();
    DDMTCUDA(DDMTCUDA&&) noexcept;
    DDMTCUDA& operator=(DDMTCUDA&&) noexcept;
    DDMTCUDA(const DDMTCUDA&)            = delete;
    DDMTCUDA& operator=(const DDMTCUDA&) = delete;

    const plans::DDMTPlan& get_plan() const noexcept;
    void execute(std::span<const float> waterfall, std::span<float> dmt);
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

#endif // DMT_ENABLE_CUDA
} // namespace dmt::algorithms
