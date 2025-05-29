#pragma once

#include <memory>
#include <span>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <type_traits>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

template <backend::ExecutionBackend Backend = backend::CPU>
class DDMT {
public:
    template <std::same_as<backend::CPU> P = Backend>
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min = 0.0F,
         int nthreads = 1);

    template <std::same_as<backend::CPU> P = Backend>
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const std::vector<float>& dm_arr,
         int nthreads = 1);

#ifdef DMT_ENABLE_CUDA
    template <std::same_as<backend::CUDA> P = Backend>
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min  = 0.0F,
         int device_id = 0);

    template <std::same_as<backend::CUDA> P = Backend>
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const std::vector<float>& dm_arr,
         int device_id = 0);
#endif // DMT_ENABLE_CUDA

    ~DDMT();
    DDMT(DDMT&&) noexcept;
    DDMT& operator=(DDMT&&) noexcept;
    DDMT(const DDMT&)            = delete;
    DDMT& operator=(const DDMT&) = delete;

    const plans::DDMTPlan& get_plan() const;

    void execute(std::span<const float> waterfall, std::span<float> dmt);

#ifdef DMT_ENABLE_CUDA
    template <typename B = Backend>
    auto execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr)
        requires std::is_same_v<B, backend::CUDA>;

#endif // DMT_ENABLE_CUDA

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

// Type aliases for convenience
using DDMTCPU = DDMT<backend::CPU>;
#ifdef DMT_ENABLE_CUDA
using DDMTCUDA = DDMT<backend::CUDA>;
#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
