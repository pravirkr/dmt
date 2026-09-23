#include "dmt/algorithms/cfdmt.hpp"

#include <vector>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <spdlog/spdlog.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bb_utils_cuda.cuh"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/dm_utils.hpp"
#include "dmt/utils/fft.hpp"
#include "dmt/utils/unpacker.hpp"

namespace dmt::algorithms {

class CohFDMTCUDA::Impl {
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

    void execute_pipeline(cuda::std::span<float> dmt_d, cudaStream_t stream) {
        if (dmt_d.size() != m_plan.get_dmt_size()) {
            throw std::runtime_error("Invalid DMT size");
        }
        const auto& dm_grid_coh   = m_plan.get_dm_grid_coh();
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
        auto m_aligned_buf_span = cuda::std::span<float>(
            thrust::raw_pointer_cast(m_aligned_buf.data()),
            m_aligned_buf.size());
        auto shift_table_span = cuda::std::span<const int>(
            thrust::raw_pointer_cast(m_dedisperse_shift_table_d.data()),
            m_dedisperse_shift_table_d.size());
        auto offset_table_span = cuda::std::span<const SizeType>(
            thrust::raw_pointer_cast(m_dedisperse_offset_table_d.data()),
            m_dedisperse_offset_table_d.size());

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
            // Detect and unpad (produces ascending frequency channels)
            unpad_detect(m_delay_buf_p1_span, m_delay_buf_p2_span,
                         m_intensity_buf_span, stream);

            // Apply causal streaming inter-channel delay line for this coarse
            // trial
            auto delay_hist_span = cuda::std::span<float>(
                thrust::raw_pointer_cast(m_channel_delay_histories[idm].data()),
                m_channel_delay_histories[idm].size());
            bb_utils::channel_delay_line(
                m_intensity_buf_span, m_aligned_buf_span, delay_hist_span,
                shift_table_span, offset_table_span, static_cast<int>(idm),
                static_cast<int>(m_plan.get_mchan()),
                static_cast<int>(m_plan.get_msamp()), stream);

            // Perform the FDMT for this coarse-DM trial's fine residual
            // search on the aligned waterfall. One shared FDMTCUDA instance
            // (same geometry for every coarse trial); only the small per-trial
            // streaming history differs.
            auto hist_span = cuda::std::span<float>(
                thrust::raw_pointer_cast(m_dm_histories[idm].data()),
                m_dm_histories[idm].size());
            auto fdmt_scratch_span = cuda::std::span<float>(
                thrust::raw_pointer_cast(m_fdmt_scratch.data()),
                m_fdmt_scratch.size());
            m_thefdmt->load_history(hist_span, stream);
            m_thefdmt->execute(m_aligned_buf_span, fdmt_scratch_span, stream);
            m_thefdmt->save_history(hist_span, stream);
            const auto dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
            auto dmt_cur_span = dmt_d.subspan(idm * dmt_cur_size, dmt_cur_size);
            cudaMemcpyAsync(dmt_cur_span.data(), fdmt_scratch_span.data(),
                            dmt_cur_size * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Host entry point: feeds input host span directly into DataUnpackerCUDA
    template <IntegralDataType DataType>
    void execute_h(std::span<const DataType> data_in_h,
                   std::span<float> dmt_h) {
        cuda_utils::set_device(m_device_id);
        thrust::device_vector<float> dmt_d(dmt_h.size());
        cudaStream_t stream = nullptr;

        auto p1_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p1.data()),
            m_unpack_buf_p1.size());
        auto p2_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p2.data()),
            m_unpack_buf_p2.size());
        m_theunpacker->execute<DataType>(data_in_h, p1_span, p2_span, stream);

        execute_pipeline(
            cuda::std::span<float>(thrust::raw_pointer_cast(dmt_d.data()),
                                   dmt_d.size()),
            stream);

        cudaMemcpyAsync(dmt_h.data(), thrust::raw_pointer_cast(dmt_d.data()),
                        dmt_h.size_bytes(), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cuda_utils::check_last_cuda_error(
            "CohFDMTCUDA::execute (host): D2H copy failed");
    }

    // Device entry point: input is already device-resident, unpacks directly on
    // GPU
    template <IntegralDataType DataType>
    void execute_d(cuda::std::span<const DataType> data_in_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream) {
        cuda_utils::set_device(m_device_id);
        auto p1_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p1.data()),
            m_unpack_buf_p1.size());
        auto p2_span = cuda::std::span<ComplexTypeCUDA>(
            thrust::raw_pointer_cast(m_unpack_buf_p2.data()),
            m_unpack_buf_p2.size());
        m_theunpacker->execute<DataType>(data_in_d, p1_span, p2_span, stream);

        execute_pipeline(dmt_d, stream);
    }

    void reset_history() noexcept {
        for (auto& hist : m_channel_delay_histories) {
            if (!hist.empty()) {
                thrust::fill(hist.begin(), hist.end(), 0.0F);
            }
        }
        for (auto& hist : m_dm_histories) {
            if (!hist.empty()) {
                thrust::fill(hist.begin(), hist.end(), 0.0F);
            }
        }
    }

private:
    plans::CohFDMTPlan m_plan;
    int m_device_id;
    std::unique_ptr<utils::FFTManagerCUDA> m_thefft;
    std::unique_ptr<algorithms::FDMTCUDA> m_thefdmt;
    std::unique_ptr<utils::DataUnpackerCUDA> m_theunpacker;

    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_unpack_buf_p2;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p1;
    thrust::device_vector<ComplexTypeCUDA> m_delay_buf_p2;
    thrust::device_vector<float> m_intensity_buf;
    thrust::device_vector<float> m_aligned_buf;
    thrust::device_vector<ComplexTypeCUDA> m_chirp_table;
    thrust::device_vector<int> m_dedisperse_shift_table_d;
    thrust::device_vector<SizeType> m_dedisperse_offset_table_d;
    // FDMT ping-pong scratch, reused across coarse-DM trials -- sized to
    // get_buffer_size() (>= get_dmt_size()); see execute_core()'s comment.
    thrust::device_vector<float> m_fdmt_scratch;
    // Small per-coarse-DM-trial "valid"-mode streaming history, swapped
    // into the one shared m_thefdmt instance around each trial's execute()
    // call -- mirrors CohFDMTCPU::Impl::m_dm_histories.
    std::vector<thrust::device_vector<float>> m_dm_histories;
    std::vector<thrust::device_vector<float>> m_channel_delay_histories;

    void initialise() {
        // Initialise the FFT manager, FDMT and data unpacker
        m_thefft = std::make_unique<utils::FFTManagerCUDA>(
            m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
            m_plan.get_mbin(), m_plan.get_nchan(), m_device_id);
        // One shared FDMTCUDA instance for every coarse-DM trial
        m_thefdmt = std::make_unique<algorithms::FDMTCUDA>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max(),
            m_plan.get_dt_min(), 1, true, "valid", false, m_device_id);
        m_theunpacker = std::make_unique<utils::DataUnpackerCUDA>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order(), m_device_id);

        // Allocate buffers
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_aligned_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());
        m_fdmt_scratch.resize(m_thefdmt->get_plan().get_buffer_size());

        // Precompute the inter-channel delay shift and offset tables once
        const auto& dm_grid_coh_h = m_plan.get_dm_grid_coh();
        const auto shift_table_h  = dmt::utils::generate_dedisperse_shift_table(
            dm_grid_coh_h, m_plan.get_f_min(), m_plan.get_f_max(),
            m_plan.get_mchan(), m_plan.get_tsamp());
        m_dedisperse_shift_table_d.resize(shift_table_h.size());
        thrust::copy(shift_table_h.begin(), shift_table_h.end(),
                     m_dedisperse_shift_table_d.begin());

        const auto offset_table_h =
            dmt::utils::generate_dedisperse_offset_table(
                dm_grid_coh_h, m_plan.get_f_min(), m_plan.get_f_max(),
                m_plan.get_mchan(), m_plan.get_tsamp());
        m_dedisperse_offset_table_d.resize(offset_table_h.size());
        thrust::copy(offset_table_h.begin(), offset_table_h.end(),
                     m_dedisperse_offset_table_d.begin());

        // Allocate per-channel delay histories for each coarse-DM trial
        m_channel_delay_histories.resize(dm_grid_coh_h.size());
        for (SizeType idm = 0; idm < dm_grid_coh_h.size(); ++idm) {
            SizeType total_h = 0;
            for (SizeType c = 0; c < m_plan.get_mchan(); ++c) {
                int s = shift_table_h[(idm * m_plan.get_mchan()) + c];
                if (s > 0) {
                    total_h += static_cast<SizeType>(s);
                }
            }
            m_channel_delay_histories[idm].resize(total_h, 0.0F);
        }

        // One small streaming-history slot per coarse-DM trial for FDMT
        m_dm_histories.assign(dm_grid_coh_h.size(),
                              thrust::device_vector<float>(
                                  m_thefdmt->history_state_size(), 0.0F));

        // Compute the chirp table
        thrust::device_vector<float> dm_grid_coh_d(dm_grid_coh_h.size());
        thrust::copy(dm_grid_coh_h.begin(), dm_grid_coh_h.end(),
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
}; // End CohFDMTCUDA::Impl definition

CohFDMTCUDA::CohFDMTCUDA(float f_center,
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
                                    device_id)) {}
CohFDMTCUDA::~CohFDMTCUDA()                                       = default;
CohFDMTCUDA::CohFDMTCUDA(CohFDMTCUDA&& other) noexcept            = default;
CohFDMTCUDA& CohFDMTCUDA::operator=(CohFDMTCUDA&& other) noexcept = default;
const plans::CohFDMTPlan& CohFDMTCUDA::get_plan() const noexcept {
    return m_impl->get_plan();
}
SizeType CohFDMTCUDA::get_dmt_size() const noexcept {
    return m_impl->get_plan().get_dmt_size();
}
template <IntegralDataType DataType>
void CohFDMTCUDA::execute(cuda::std::span<const DataType> data_in,
                          cuda::std::span<float> dmt,
                          cudaStream_t stream) const {
    m_impl->execute_d<DataType>(data_in, dmt, stream);
}
template <IntegralDataType DataType>
void CohFDMTCUDA::execute(std::span<const DataType> data_in,
                          std::span<float> dmt) const {
    m_impl->execute_h<DataType>(data_in, dmt);
}
void CohFDMTCUDA::reset_history() noexcept { m_impl->reset_history(); }

// Instantiate the public execute methods for each supported DataType
template void CohFDMTCUDA::execute<int8_t>(std::span<const int8_t>,
                                           std::span<float>) const;
template void CohFDMTCUDA::execute<uint8_t>(std::span<const uint8_t>,
                                            std::span<float>) const;
template void CohFDMTCUDA::execute<int8_t>(cuda::std::span<const int8_t>,
                                           cuda::std::span<float>,
                                           cudaStream_t) const;
template void CohFDMTCUDA::execute<uint8_t>(cuda::std::span<const uint8_t>,
                                            cuda::std::span<float>,
                                            cudaStream_t) const;

} // namespace dmt::algorithms
