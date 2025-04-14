#include "dmt/cfdmt.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/bb_utils_cuda.cuh"
#include "dmt/common/types.hpp"
#include "dmt/common/unpacker.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/fdmt.hpp"
#include "dmt/fft.hpp"

namespace dmt {

template <>
class CohFDMT<backend::CPU>::Impl {
public:
    Impl(float f_center,
         float bw_sub,
         SizeType nsub,
         float tbin,
         SizeType nbin,
         SizeType nfft,
         float t_p,
         float dm_max,
         float dm_min,
         SizeType noverlap,
         std::string_view data_order,
         bool verbose,
         int device_id)
        : m_device_id(device_id),
          m_plan(f_center,
                 bw_sub,
                 nsub,
                 tbin,
                 nbin,
                 nfft,
                 t_p,
                 dm_max,
                 dm_min,
                 noverlap,
                 data_order,
                 verbose) {
        initialise();
    }

    const CohFDMTPlan& get_plan() const { return m_plan; }

    void execute(const uint8_t* __restrict__ data_in,
                 SizeType in_size,
                 const std::string& in_order,
                 float* __restrict__ dmt,
                 SizeType dmt_size) {
        execute(data_in, in_size, in_order, dmt, dmt_size, false);
    }

    void execute(const uint8_t* __restrict__ data_in,
                 SizeType in_size,
                 const std::string& in_order,
                 float* __restrict__ dmt,
                 SizeType dmt_size,
                 bool device_flags) {
        if (device_flags) {
            execute_device(data_in, in_size, in_order, dmt, dmt_size);
        } else {
            thrust::device_vector<uint8_t> data_in_d(data_in,
                                                     data_in + in_size);
            thrust::device_vector<float> dmt_d(dmt, dmt + dmt_size);
            execute_device(thrust::raw_pointer_cast(data_in_d.data()), in_size,
                           in_order, thrust::raw_pointer_cast(dmt_d.data()),
                           dmt_size);
            thrust::copy(dmt_d.begin(), dmt_d.end(), dmt);
            error_checker::check_cuda("thrust::copy failed");
        }
    }

private:
    int m_device_id;
    CohFDMTPlan m_plan;
    std::unique_ptr<FFTManagerCUDA> m_thefft;
    std::unique_ptr<FDMTCUDA> m_thefdmt;
    std::unique_ptr<DataUnpackerCUDA> m_theunpacker;

    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p2;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p2;
    thrust::device_vector<float> m_intensity_buf;
    thrust::device_vector<ComplexTypeCUDA> m_chirp_table;

    void initialise();

    void apply_chirp(cuda::std::span<const cufftComplex> data1_in,
                     cuda::std::span<const cufftComplex> data2_in,
                     cuda::std::span<cufftComplex> data1_out,
                     cuda::std::span<cufftComplex> data2_out,
                     SizeType idm,
                     cudaStream_t stream) {
        const auto scale = m_plan.get_chirp_scale();
        bb_utils_cu::apply_chirp(data1_in, data2_in, m_chirp_table, data1_out,
                                 data2_out, m_plan.get_nsub(),
                                 m_plan.get_nbin(), m_plan.get_nfft(), idm,
                                 scale, stream);
    }

    void unpad_detect(cuda::std::span<const cufftComplex> fft_p1,
                      cuda::std::span<const cufftComplex> fft_p2,
                      cuda::std::span<float> intensity,
                      cudaStream_t stream) {
        bb_utils_cu::unpad_detect(fft_p1, fft_p2, intensity, m_plan.get_nchan(),
                                  m_plan.get_nfft(), m_plan.get_nsub(),
                                  m_plan.get_mbin(), m_plan.get_noverlap(),
                                  stream);
    }
}; // End CohFDMT<backend::CUDA>::Impl definition

// CUDA-specific constructor implementation
template <>
template <std::same_as<backend::CUDA> P>
CohFDMT<backend::CUDA>::CohFDMT(float f_center,
                                float bw_sub,
                                SizeType nsub,
                                float tbin,
                                SizeType nbin,
                                SizeType nfft,
                                float t_p,
                                float dm_max,
                                float dm_min,
                                SizeType noverlap,
                                std::string_view data_order,
                                bool verbose,
                                int device_id)
    : m_impl(std::make_unique<Impl>(f_center,
                                    bw_sub,
                                    nsub,
                                    tbin,
                                    nbin,
                                    nfft,
                                    t_p,
                                    dm_max,
                                    dm_min,
                                    noverlap,
                                    data_order,
                                    verbose,
                                    device_id)) {
    spdlog::debug("CohFDMT<CPU> object created.");
}
template <>
CohFDMT<backend::CUDA>::~CohFDMT() {
    spdlog::debug("CohFDMT<CUDA> object destroyed.");
}
template <>
CohFDMT<backend::CUDA>::CohFDMT(CohFDMT&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("CohFDMT<CUDA> object moved.");
}
template <>
CohFDMT<backend::CUDA>&
CohFDMT<backend::CUDA>::operator=(CohFDMT&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
const CohFDMTPlan& CohFDMT<backend::CUDA>::get_plan() const {
    return m_impl->get_plan();
}
template <>
void CohFDMT<backend::CUDA>::execute(std::span<const uint8_t> data_in,
                                     std::span<float> dmt) {
    m_impl->execute(data_in, dmt);
}

// Explicit instantiation (for linking)
template CohFDMT<backend::CUDA>::CohFDMT(float,
                                         float,
                                         SizeType,
                                         float,
                                         SizeType,
                                         SizeType,
                                         float,
                                         float,
                                         float,
                                         SizeType,
                                         std::string_view,
                                         bool,
                                         int);
} // namespace dmt
