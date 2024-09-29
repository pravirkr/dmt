#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <dmt/cfdmt/cfdmt_cpu.hpp>
#include <dmt/common/types.hpp>

#include "dmt/bb_utils_cpu.hpp"
#include "dmt/dm_utils.hpp"

FFTManager::FFTManager(SizeType nfft,
                       SizeType nsub,
                       SizeType nbin,
                       SizeType mbin,
                       SizeType nchan,
                       SizeType nthreads)
    : m_nfft(nfft),
      m_nsub(nsub),
      m_nbin(nbin),
      m_mbin(mbin),
      m_nchan(nchan),
      m_nthreads(nthreads) {}

FFTManager::~FFTManager() {
    if (m_forward_plan != nullptr) {
        fftwf_destroy_plan(m_forward_plan);
    }
    if (m_backward_plan != nullptr) {
        fftwf_destroy_plan(m_backward_plan);
    }
    fftwf_cleanup_threads();
}

void FFTManager::initialize_plans(ComplexType* __restrict__ unpack_buffer,
                                  ComplexType* __restrict__ delay_buffer) {
    // Set the number of threads
    auto error = fftwf_init_threads();
    if (error == 0) {
        throw std::runtime_error("Failed to initialise FFTW threads");
    }
    fftwf_plan_with_nthreads(static_cast<int>(m_nthreads));
    // Generate FFT plan (batch in-place forward and backward FFT)
    const std::array<int, 1> fft_size_fw = {static_cast<int>(m_nbin)};
    const std::array<int, 1> fft_size_bw = {static_cast<int>(m_mbin)};

    m_forward_plan = fftwf_plan_many_dft(
        1, fft_size_fw.data(), static_cast<int>(m_nfft * m_nsub),
        reinterpret_cast<fftwf_complex*>(unpack_buffer), nullptr, 1,
        static_cast<int>(m_nbin),
        reinterpret_cast<fftwf_complex*>(unpack_buffer), nullptr, 1,
        static_cast<int>(m_nbin), FFTW_FORWARD, FFTW_MEASURE);

    m_backward_plan = fftwf_plan_many_dft(
        1, fft_size_bw.data(), static_cast<int>(m_nfft * m_nsub * m_nchan),
        reinterpret_cast<fftwf_complex*>(delay_buffer), nullptr, 1,
        static_cast<int>(m_mbin),
        reinterpret_cast<fftwf_complex*>(delay_buffer), nullptr, 1,
        static_cast<int>(m_mbin), FFTW_BACKWARD, FFTW_MEASURE);

    if (m_forward_plan == nullptr || m_backward_plan == nullptr) {
        throw std::runtime_error("Failed to create FFTW plans");
    }
}

void FFTManager::forward_fft(ComplexType* __restrict__ data) const {
    fftwf_execute_dft(m_forward_plan, reinterpret_cast<fftwf_complex*>(data),
                      reinterpret_cast<fftwf_complex*>(data));
    swap_spectrum(data, m_nfft, m_nsub, m_nbin);
}

void FFTManager::backward_fft(ComplexType* __restrict__ data) const {
    swap_spectrum(data, m_nfft, m_nsub, m_nbin);
    fftwf_execute_dft(m_backward_plan, reinterpret_cast<fftwf_complex*>(data),
                      reinterpret_cast<fftwf_complex*>(data));
}

void FFTManager::swap_spectrum(ComplexType* __restrict__ data_in,
                               SizeType nz,
                               SizeType ny,
                               SizeType nx) {
    // Swap the halves along the last dimension
    const SizeType mid_bin = nx / 2;
    for (SizeType iz = 0; iz < nz; ++iz) {
        for (SizeType iy = 0; iy < ny; ++iy) {
            const SizeType offset = (iz * ny + iy) * nx;
            std::rotate(data_in + offset, data_in + offset + mid_bin,
                        data_in + offset + nx);
        }
    }
}

CohFDMTCPU::CohFDMTCPU(float f_center,
                       float sub_bw,
                       SizeType nsub,
                       float tbin,
                       SizeType nbin,
                       SizeType nfft,
                       float tp,
                       float dm_max,
                       float dm_min,
                       SizeType noverlap,
                       SizeType nthreads)
    : m_plan(f_center,
             sub_bw,
             nsub,
             tbin,
             nbin,
             nfft,
             tp,
             dm_max,
             dm_min,
             noverlap) {
#ifdef USE_OPENMP
    omp_set_num_threads(static_cast<int>(nthreads));
#endif
    initialise();
}

const CohFDMTPlan& CohFDMTCPU::get_plan() const { return m_plan; }

SizeType CohFDMTCPU::get_dmt_size() const {
    const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
    return dm_grid_coh.size() * m_thefdmt->get_plan().get_dmt_size();
}

void CohFDMTCPU::execute(const uint8_t* __restrict__ data_in,
                         SizeType in_size,
                         const std::string& in_order,
                         float* __restrict__ dmt,
                         SizeType dmt_size) {
    if (dmt_size != get_dmt_size()) {
        throw std::runtime_error("Invalid DMT size");
    }
    unpack_init(data_in, in_size, in_order, m_unpack_buf_p1.data(),
                m_unpack_buf_p2.data(), m_unpack_buf_p1.size());
    // Forward FFT
    m_thefft->forward_fft(m_unpack_buf_p1.data());
    m_thefft->forward_fft(m_unpack_buf_p2.data());

    const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
    for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
        // Apply chirp
        apply_chirp(m_unpack_buf_p1.data(), m_delay_buf_p1.data(), idm);
        apply_chirp(m_unpack_buf_p2.data(), m_delay_buf_p2.data(), idm);
        // Backward FFT
        m_thefft->backward_fft(m_delay_buf_p1.data());
        m_thefft->backward_fft(m_delay_buf_p2.data());
        // Detect and unpad
        unpad_detect(m_delay_buf_p1.data(), m_delay_buf_p2.data(),
                     m_delay_buf_p1.size(), m_intensity_buf.data(),
                     m_intensity_buf.size());
        // Perform inter-channel dedispersion at the current coherent DM
        dm_utils::dedisperse(m_intensity_buf.data(), m_intensity_buf.size(),
                             dm_grid_coh[idm], m_plan.get_f_min(),
                             m_plan.get_f_max(), m_plan.get_mchan(),
                             m_plan.get_msamp(), m_plan.get_tsamp());

        // Perform the FDMT
        SizeType dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
        float* dmt_cur        = &dmt[idm * dmt_cur_size];
        m_thefdmt->execute(m_intensity_buf.data(), m_intensity_buf.size(),
                           dmt_cur, dmt_cur_size);
    }
}

void CohFDMTCPU::initialise() {
    // Initialise the FFT manager and FDMT
    m_thefft = std::make_unique<FFTManager>(
        m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
        m_plan.get_mbin(), m_plan.get_nchan());
    m_thefft->initialize_plans(m_unpack_buf_p1.data(),
                                    m_delay_buf_p1.data());
    m_thefdmt = std::make_unique<FDMTCPU>(
        m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
        m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max());

    // Allocate buffers
    m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
    m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
    m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
    m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
    m_intensity_buf.resize(m_plan.get_intensity_buf_size());

    // Compute the chirp table
    const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
    m_chirp_table.resize(dm_grid_coh.size() * m_plan.get_nsub() *
                         m_plan.get_nbin());
    dm_utils::compute_chirp(
        m_chirp_table.data(), m_chirp_table.size(), dm_grid_coh.data(),
        dm_grid_coh.size(), m_plan.get_fcenter(), m_plan.get_bw(),
        m_plan.get_nbin(), m_plan.get_nsub(), m_plan.get_nchan());
}

void CohFDMTCPU::unpack_init(const uint8_t* __restrict__ data_in,
                             SizeType in_size,
                             const std::string& in_order,
                             ComplexType* __restrict__ data_p1,
                             ComplexType* __restrict__ data_p2,
                             SizeType out_size) const {
    DataOrder order = string_to_data_order(in_order);
    if (order == DataOrder::kFTPRI) {
        DataUnpacker<DataOrder::kFTPRI> unpacker(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft());
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else if (order == DataOrder::kPRITF) {
        DataUnpacker<DataOrder::kPRITF> unpacker(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft());
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else if (order == DataOrder::kRITFP) {
        DataUnpacker<DataOrder::kRITFP> unpacker(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft());
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else {
        throw std::runtime_error("Invalid order: " + in_order);
    }
}

void CohFDMTCPU::unpad_detect(const ComplexType* __restrict__ fft_p1,
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
#pragma omp parallel for collapse(4)
#endif
    for (SizeType ibin = 0; ibin < mbin_adjusted; ++ibin) {
        for (SizeType ichan = 0; ichan < m_plan.get_nchan(); ++ichan) {
            for (SizeType ifft = 0; ifft < m_plan.get_nfft(); ++ifft) {
                for (SizeType isub = 0; isub < m_plan.get_nsub(); ++isub) {
                    const SizeType isamp = ibin + (mbin_adjusted * ifft);
                    const SizeType ibin_adjusted = ibin + noverlap_per_channel;
                    const SizeType src_idx =
                        (ifft * m_plan.get_nsub() * m_plan.get_nchan() *
                         m_plan.get_mbin()) +
                        ((m_plan.get_nsub() - isub - 1) * m_plan.get_nchan() *
                         m_plan.get_mbin()) +
                        (ichan * m_plan.get_mbin()) + ibin_adjusted;
                    const SizeType dst_idx =
                        (isub * m_plan.get_nchan() * msamp) +
                        ((m_plan.get_nchan() - ichan - 1) * msamp) + isamp;
                    intensity[dst_idx] =
                        std::norm(fft_p1[src_idx]) + std::norm(fft_p2[src_idx]);
                }
            }
        }
    }
}

void CohFDMTCPU::apply_chirp(const ComplexType* __restrict__ data_in,
                             ComplexType* __restrict__ data_out,
                             SizeType idm) {
    const auto scale = m_plan.get_chirp_scale();
    pointwise_complex_multiply(data_in, m_chirp_table.data(), data_out,
                               m_plan.get_nsub() * m_plan.get_nbin(),
                               m_plan.get_nfft(), idm, scale);
}
