#include "dmt/algorithms/cfdmt.hpp"

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bb_utils_cuda.cuh"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/utils/fft.hpp"
#include "dmt/utils/unpacker.hpp"

namespace dmt::algorithms {

template <>
class CohFDMT<backend::CUDA>::Impl {
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
        : m_plan(f_center,
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
                 verbose),
          m_device_id(device_id) {
        cuda_utils::set_device(m_device_id);
        initialise();
    }

    const plans::CohFDMTPlan& get_plan() const { return m_plan; }

    template <IntegralDataType DataType>
    void execute_d(cuda::std::span<const DataType> data_in_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        if (dmt_d.size() != m_plan.get_dmt_size()) {
            throw std::runtime_error("Invalid DMT size");
        }
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        thrust::device_vector<float> dm_grid_coh_d(dm_grid_coh.size());
        thrust::copy(dm_grid_coh.begin(), dm_grid_coh.end(),
                     dm_grid_coh_d.begin());
        auto dm_grid_coh_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(dm_grid_coh_d.data()),
            dm_grid_coh_d.size());
        auto m_unpack_buf_p1_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p1.data()),
            m_unpack_buf_p1.size());
        auto m_unpack_buf_p2_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p2.data()),
            m_unpack_buf_p2.size());
        auto m_delay_buf_p1_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_delay_buf_p1.data()),
            m_delay_buf_p1.size());
        auto m_delay_buf_p2_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_delay_buf_p2.data()),
            m_delay_buf_p2.size());
        auto m_intensity_buf_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(m_intensity_buf.data()),
            m_intensity_buf.size());
        m_theunpacker->execute<DataType>(data_in_d, m_unpack_buf_p1_span,
                                         m_unpack_buf_p2_span, stream);
        // Forward FFT
        m_thefft->forward_fft(m_unpack_buf_p1_span, m_unpack_buf_p2_span,
                              stream);
        for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
            // Apply chirp
            apply_chirp(m_unpack_buf_p1_span, m_unpack_buf_p2_span,
                        m_delay_buf_p1_span, m_delay_buf_p2_span, idm, stream);
            // Backward FFT
            m_thefft->backward_fft(m_delay_buf_p1_span, m_delay_buf_p2_span,
                                   stream);
            // Detect and unpad
            unpad_detect(m_delay_buf_p1_span, m_delay_buf_p2_span,
                         m_intensity_buf_span, stream);
            // Perform inter-channel dedispersion at the current coherent DM
            // TODO
            // Perform the FDMT
            const auto dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
            auto dmt_cur_span = dmt_d.subspan(idm * dmt_cur_size, dmt_cur_size);
            m_thefdmt->execute(m_intensity_buf_span, dmt_cur_span, stream);
        }
    }

private:
    plans::CohFDMTPlan m_plan;
    int m_device_id;
    std::unique_ptr<utils::FFTManagerCUDA> m_thefft;
    std::unique_ptr<algorithms::FDMT<backend::CUDA>> m_thefdmt;
    std::unique_ptr<utils::DataUnpacker<backend::CUDA>> m_theunpacker;

    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p2;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p2;
    thrust::device_vector<float> m_intensity_buf;
    thrust::device_vector<ComplexTypeCUDA> m_chirp_table;

    void initialise() {
        // Initialise the FFT manager, FDMT and data unpacker
        m_thefft = std::make_unique<utils::FFTManagerCUDA>(
            m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
            m_plan.get_mbin(), m_plan.get_nchan(), m_device_id);
        m_thefdmt = std::make_unique<algorithms::FDMTCUDA>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max(),
            m_device_id);
        m_theunpacker = std::make_unique<utils::DataUnpackerCUDA>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order(), m_device_id);

        // Allocate buffers
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());

        // Compute the chirp table
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        thrust::device_vector<float> dm_grid_coh_d(dm_grid_coh.size());
        thrust::copy(dm_grid_coh.begin(), dm_grid_coh.end(),
                     dm_grid_coh_d.begin());
        auto dm_grid_coh_span = cuda::std::span<const float>(
            thrust::raw_pointer_cast(dm_grid_coh_d.data()),
            dm_grid_coh_d.size());
        auto m_chirp_table_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_chirp_table.data()),
            m_chirp_table.size());
        bb_utils::compute_chirp(dm_grid_coh_span, m_chirp_table_span,
                                m_plan.get_f_center(), m_plan.get_bw(),
                                m_plan.get_nbin(), m_plan.get_nsub(),
                                m_plan.get_nchan());
    }

    void apply_chirp(cuda::std::span<const ComplexTypeCUDA> data1_in,
                     cuda::std::span<const ComplexTypeCUDA> data2_in,
                     cuda::std::span<ComplexTypeCUDA> data1_out,
                     cuda::std::span<ComplexTypeCUDA> data2_out,
                     SizeType idm,
                     cudaStream_t stream) {
        const auto scale        = m_plan.get_chirp_scale();
        auto m_chirp_table_span = cuda::std::span<const ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_chirp_table.data()),
            m_chirp_table.size());
        bb_utils::apply_chirp(data1_in, data2_in, m_chirp_table_span, data1_out,
                              data2_out, static_cast<int>(m_plan.get_nsub()),
                              static_cast<int>(m_plan.get_nbin()),
                              static_cast<int>(m_plan.get_nfft()),
                              static_cast<int>(idm), scale, stream);
    }

    void unpad_detect(cuda::std::span<const ComplexTypeCUDA> fft_p1,
                      cuda::std::span<const ComplexTypeCUDA> fft_p2,
                      cuda::std::span<float> intensity,
                      cudaStream_t stream) {
        bb_utils::unpad_detect(fft_p1, fft_p2, intensity,
                               static_cast<int>(m_plan.get_nchan()),
                               static_cast<int>(m_plan.get_nfft()),
                               static_cast<int>(m_plan.get_nsub()),
                               static_cast<int>(m_plan.get_mbin()),
                               static_cast<int>(m_plan.get_noverlap()), stream);
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
    spdlog::debug("CohFDMT<CUDA> object created.");
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
const plans::CohFDMTPlan& CohFDMT<backend::CUDA>::get_plan() const {
    return m_impl->get_plan();
}
// template <>
// template <IntegralDataType DataType>
// void CohFDMT<backend::CUDA>::execute(std::span<const DataType> data_in,
//                                      std::span<float> dmt) const {
//     m_impl->execute_h<DataType>(data_in, dmt);
// }
template <>
template <IntegralDataType DataType, std::same_as<backend::CUDA> P>
void CohFDMT<backend::CUDA>::execute(cuda::std::span<const DataType> d_data_in,
                                     cuda::std::span<float> d_dmt,
                                     cudaStream_t stream) const {
    m_impl->execute_d<DataType>(d_data_in, d_dmt, stream);
}

} // namespace dmt::algorithms
