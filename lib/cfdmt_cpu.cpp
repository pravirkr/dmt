#include "dmt/algorithms/cfdmt.hpp"

#include <span>
#include <stdexcept>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <spdlog/spdlog.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bb_utils_cpu.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/utils/fft.hpp"
#include "dmt/utils/unpacker.hpp"

namespace dmt::algorithms {

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
         int nthreads)
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
          m_nthreads(set_dmt_openmp_threads(nthreads)) {
        initialise();
    }

    const plans::CohFDMTPlan& get_plan() const { return m_plan; }

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) {
        if (dmt.size() != m_plan.get_dmt_size()) {
            throw std::runtime_error("Invalid DMT size");
        }
        m_theunpacker->execute<DataType>(data_in, m_unpack_buf_p1,
                                         m_unpack_buf_p2);
        // Forward FFT
        m_thefft->forward_fft(m_unpack_buf_p1, m_unpack_buf_p2);
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
            // Apply chirp
            apply_chirp(m_unpack_buf_p1, m_unpack_buf_p2, m_delay_buf_p1,
                        m_delay_buf_p2, idm);
            // Backward FFT
            m_thefft->backward_fft(m_delay_buf_p1, m_delay_buf_p2);
            // Detect and unpad
            unpad_detect(m_delay_buf_p1, m_delay_buf_p2, m_intensity_buf);
            // Perform inter-channel dedispersion at the current coherent DM
            dmt::utils::dedisperse(
                m_intensity_buf.data(), m_intensity_buf.size(),
                dm_grid_coh[idm], m_plan.get_f_min(), m_plan.get_f_max(),
                m_plan.get_mchan(), m_plan.get_msamp(), m_plan.get_tsamp());

            // Perform the FDMT
            const auto dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
            auto dmt_cur_span = dmt.subspan(idm * dmt_cur_size, dmt_cur_size);
            m_thefdmt->execute(m_intensity_buf, dmt_cur_span);
        }
    }

private:
    plans::CohFDMTPlan m_plan;
    int m_nthreads;
    std::unique_ptr<utils::FFTManagerCPU> m_thefft;
    std::unique_ptr<algorithms::FDMT<backend::CPU>> m_thefdmt;
    std::unique_ptr<utils::DataUnpacker<backend::CPU>> m_theunpacker;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<ComplexType> m_chirp_table;

    void initialise() {
        // Initialise the FFT manager, FDMT and data unpacker
        m_thefft = std::make_unique<utils::FFTManagerCPU>(
            m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
            m_plan.get_mbin(), m_plan.get_nchan(), m_nthreads);
        m_thefft->initialize_plans(std::span<ComplexType>(m_unpack_buf_p1),
                                   std::span<ComplexType>(m_delay_buf_p1));
        m_thefdmt = std::make_unique<algorithms::FDMT<backend::CPU>>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max());
        m_theunpacker = std::make_unique<utils::DataUnpacker<backend::CPU>>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order(), m_nthreads);

        // Allocate buffers
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());

        // Compute the chirp table
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        bb_utils::compute_chirp(
            dm_grid_coh, m_chirp_table, m_plan.get_f_center(), m_plan.get_bw(),
            m_plan.get_nbin(), m_plan.get_nsub(), m_plan.get_nchan());
    }

    void apply_chirp(std::span<const ComplexType> data1_in,
                     std::span<const ComplexType> data2_in,
                     std::span<ComplexType> data1_out,
                     std::span<ComplexType> data2_out,
                     SizeType idm) {
        const auto scale = m_plan.get_chirp_scale();
        bb_utils::apply_chirp(data1_in, data2_in, m_chirp_table, data1_out,
                              data2_out, m_plan.get_nsub(), m_plan.get_nbin(),
                              m_plan.get_nfft(), idm, scale);
    }

    void unpad_detect(std::span<const ComplexType> fft_p1,
                      std::span<const ComplexType> fft_p2,
                      std::span<float> intensity) const {
        bb_utils::unpad_detect(fft_p1, fft_p2, intensity, m_plan.get_nchan(),
                               m_plan.get_nfft(), m_plan.get_nsub(),
                               m_plan.get_mbin(), m_plan.get_noverlap());
    }

}; // End CohFDMT<backend::CPU>::Impl definition

// CPU-specific constructor implementation
template <>
template <std::same_as<backend::CPU> P>
CohFDMT<backend::CPU>::CohFDMT(float f_center,
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
                               int nthreads)
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
                                    nthreads)) {
    spdlog::debug("CohFDMT<CPU> object created.");
}
template <>
CohFDMT<backend::CPU>::~CohFDMT() {
    spdlog::debug("CohFDMT<CPU> object destroyed.");
}
template <>
CohFDMT<backend::CPU>::CohFDMT(CohFDMT&& other) noexcept
    : m_impl(std::move(other.m_impl)) {
    spdlog::debug("CohFDMT<CPU> object moved.");
}
template <>
CohFDMT<backend::CPU>&
CohFDMT<backend::CPU>::operator=(CohFDMT&& other) noexcept {
    if (this != &other) {
        m_impl = std::move(other.m_impl);
    }
    return *this;
}
template <>
const plans::CohFDMTPlan& CohFDMT<backend::CPU>::get_plan() const {
    return m_impl->get_plan();
}
template <>
template <IntegralDataType DataType>
void CohFDMT<backend::CPU>::execute(std::span<const DataType> data_in,
                                    std::span<float> dmt) const {
    m_impl->execute<DataType>(data_in, dmt);
}

// Explicit instantiation (for linking)
template CohFDMT<backend::CPU>::CohFDMT(float,
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

// Instantiate the public execute method for each supported DataType
template void CohFDMT<backend::CPU>::execute<int8_t>(std::span<const int8_t>,
                                                     std::span<float>) const;
template void CohFDMT<backend::CPU>::execute<uint8_t>(std::span<const uint8_t>,
                                                      std::span<float>) const;

} // namespace dmt::algorithms
