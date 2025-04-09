#include "dmt/cfdmt.hpp"

#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <spdlog/spdlog.h>

#include "dmt/bb_utils_cpu.hpp"
#include "dmt/common/types.hpp"
#include "dmt/common/unpacker.hpp"
#include "dmt/dm_utils.hpp"
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
         int nthreads)
        : m_nthreads(nthreads),
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
#ifdef DMT_ENABLE_OPENMP
        omp_set_num_threads(nthreads);
#endif
        initialise();
    }

    const CohFDMTPlan& get_plan() const { return m_plan; }

    void execute(std::span<const uint8_t> data_in, std::span<float> dmt) {
        const uint8_t* data_in_ptr = data_in.data();
        if (dmt.size() != m_plan.get_dmt_size()) {
            throw std::runtime_error("Invalid DMT size");
        }
        m_theunpacker->unpack_and_padd<uint8_t>(
            data_in_ptr, data_in.size(), m_unpack_buf_p1.data(),
            m_unpack_buf_p2.data(), m_unpack_buf_p1.size());
        // Forward FFT
        m_thefft->forward_fft(std::span<ComplexType>(m_unpack_buf_p1));
        m_thefft->forward_fft(std::span<ComplexType>(m_unpack_buf_p2));

        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
            // Apply chirp
            apply_chirp(m_unpack_buf_p1.data(), m_delay_buf_p1.data(), idm);
            apply_chirp(m_unpack_buf_p2.data(), m_delay_buf_p2.data(), idm);
            // Backward FFT
            m_thefft->backward_fft(std::span<ComplexType>(m_delay_buf_p1));
            m_thefft->backward_fft(std::span<ComplexType>(m_delay_buf_p2));
            // Detect and unpad
            unpad_detect(m_delay_buf_p1.data(), m_delay_buf_p2.data(),
                         m_delay_buf_p1.size(), m_intensity_buf.data(),
                         m_intensity_buf.size());
            // Perform inter-channel dedispersion at the current coherent DM
            dmt::utils::dedisperse(
                m_intensity_buf.data(), m_intensity_buf.size(),
                dm_grid_coh[idm], m_plan.get_f_min(), m_plan.get_f_max(),
                m_plan.get_mchan(), m_plan.get_msamp(), m_plan.get_tsamp());

            // Perform the FDMT
            SizeType dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
            float* dmt_cur        = &dmt[idm * dmt_cur_size];
            m_thefdmt->execute(std::span<const float>(m_intensity_buf.data(),
                                                      m_intensity_buf.size()),
                               std::span<float>(dmt_cur, dmt_cur_size));
        }
    }

private:
    int m_nthreads;
    CohFDMTPlan m_plan;
    std::unique_ptr<FFTManagerCPU> m_thefft;
    std::unique_ptr<FDMTCPU> m_thefdmt;
    std::unique_ptr<DataUnpacker> m_theunpacker;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<ComplexType> m_chirp_table;

    void initialise() {
        // Initialise the FFT manager, FDMT and data unpacker
        m_thefft = std::make_unique<FFTManagerCPU>(
            m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
            m_plan.get_mbin(), m_plan.get_nchan(), m_nthreads);
        m_thefft->initialize_plans(std::span<ComplexType>(m_unpack_buf_p1),
                                   std::span<ComplexType>(m_delay_buf_p1));
        m_thefdmt = std::make_unique<dmt::FDMTCPU>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max());
        m_theunpacker = std::make_unique<DataUnpacker>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order());

        // Allocate buffers
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());

        // Compute the chirp table
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        dmt::utils::compute_chirp(
            m_chirp_table.data(), m_chirp_table.size(), dm_grid_coh.data(),
            dm_grid_coh.size(), m_plan.get_f_center(), m_plan.get_bw(),
            m_plan.get_nbin(), m_plan.get_nsub(), m_plan.get_nchan());
    }

    void unpad_detect(const ComplexType* __restrict__ fft_p1,
                      const ComplexType* __restrict__ fft_p2,
                      SizeType in_size,
                      float* __restrict__ intensity,
                      SizeType out_size) const {
        if (in_size != m_plan.get_delay_buf_size()) {
            throw std::runtime_error("Invalid input size");
        }
        if (out_size != m_plan.get_intensity_buf_size()) {
            throw std::runtime_error("Invalid output size");
        }

        const SizeType noverlap_per_channel =
            m_plan.get_noverlap() / m_plan.get_nchan();
        const SizeType mbin_adjusted =
            m_plan.get_mbin() - (2 * noverlap_per_channel);
        const SizeType msamp = m_plan.get_nfft() * mbin_adjusted;

#ifdef USE_OPENMP
#pragma omp parallel for collapse(4) num_threads(m_nthreads)
#endif
        for (SizeType ibin = 0; ibin < mbin_adjusted; ++ibin) {
            for (SizeType ichan = 0; ichan < m_plan.get_nchan(); ++ichan) {
                for (SizeType ifft = 0; ifft < m_plan.get_nfft(); ++ifft) {
                    for (SizeType isub = 0; isub < m_plan.get_nsub(); ++isub) {
                        const SizeType isamp = ibin + (mbin_adjusted * ifft);
                        const SizeType ibin_adjusted =
                            ibin + noverlap_per_channel;
                        const SizeType src_idx =
                            (ifft * m_plan.get_nsub() * m_plan.get_nchan() *
                             m_plan.get_mbin()) +
                            ((m_plan.get_nsub() - isub - 1) *
                             m_plan.get_nchan() * m_plan.get_mbin()) +
                            (ichan * m_plan.get_mbin()) + ibin_adjusted;
                        const SizeType dst_idx =
                            (isub * m_plan.get_nchan() * msamp) +
                            ((m_plan.get_nchan() - ichan - 1) * msamp) + isamp;
                        intensity[dst_idx] = std::norm(fft_p1[src_idx]) +
                                             std::norm(fft_p2[src_idx]);
                    }
                }
            }
        }
    }

    void apply_chirp(const ComplexType* __restrict__ data_in,
                     ComplexType* __restrict__ data_out,
                     SizeType idm) {
        const auto scale = m_plan.get_chirp_scale();
        pointwise_complex_multiply(data_in, m_chirp_table.data(), data_out,
                                   m_plan.get_nsub() * m_plan.get_nbin(),
                                   m_plan.get_nfft(), idm, scale);
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
const CohFDMTPlan& CohFDMT<backend::CPU>::get_plan() const {
    return m_impl->get_plan();
}
template <>
void CohFDMT<backend::CPU>::execute(std::span<const uint8_t> data_in,
                                    std::span<float> dmt) {
    m_impl->execute(data_in, dmt);
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
} // namespace dmt
