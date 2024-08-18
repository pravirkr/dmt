#include <algorithm>
#include <complex>
#include <cstdint>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include "dmt/baseband_utils.hpp"
#include "dmt/dm_utils.hpp"
#include <dmt/cfdmt_cpu.hpp>

CohFDMTCPU::CohFDMTCPU(float f_center,
                       float sub_bw,
                       SizeType nsub,
                       float tbin,
                       SizeType nbin,
                       SizeType nfft,
                       float tp,
                       float dm_max,
                       float dm_min,
                       SizeType noverlap)
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
    initialise();
}

CohFDMTCPU::~CohFDMTCPU() {
    fftwf_destroy_plan(m_fft_plan_fw);
    fftwf_destroy_plan(m_fft_plan_bw);
}

const CohFDMTPlan& CohFDMTCPU::get_plan() const { return m_plan; }

SizeType CohFDMTCPU::get_dmt_size() const {
    const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
    return dm_grid_coh.size() * m_plan.dt_max * m_plan.msamp;
}

void CohFDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void CohFDMTCPU::execute(const uint8_t* __restrict data_in,
                         SizeType in_size,
                         std::string in_order,
                         float* __restrict dmt,
                         SizeType dmt_size) {
    if (dmt_size != get_dmt_size()) {
        throw std::runtime_error("Invalid DMT size");
    }
    unpack_init(data_in, in_size, in_order, m_unpacked_buffer_p1.data(),
                m_unpacked_buffer_p2.data(), m_unpacked_buffer_p1.size());
    // Perform the forward FFT
    fftwf_execute_dft(
        m_fft_plan_fw,
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p1.data()),
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p1.data()));
    fftwf_execute_dft(
        m_fft_plan_fw,
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p2.data()),
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p2.data()));
    // Swap the halves of the spectrum
    swap_spectrum_halves(m_unpacked_buffer_p1.data(), m_plan.nfft, m_plan.nsub,
                         m_plan.nbin);
    swap_spectrum_halves(m_unpacked_buffer_p2.data(), m_plan.nfft, m_plan.nsub,
                         m_plan.nbin);
    const auto scale = 1.0F / static_cast<float>(m_plan.nbin);
    for (SizeType idm = 0; idm < m_plan.dm_grid_coh.size(); ++idm) {
        // Apply the chirp to the data
        pointwise_complex_multiply(
            m_unpacked_buffer_p1.data(), m_chirp_table.data(),
            m_fftdelay_buffer_p1.data(), m_plan.nsub * m_plan.nbin, m_plan.nfft,
            idm, scale);
        pointwise_complex_multiply(
            m_unpacked_buffer_p2.data(), m_chirp_table.data(),
            m_fftdelay_buffer_p2.data(), m_plan.nsub * m_plan.nbin, m_plan.nfft,
            idm, scale);
        // Swap the halves of the spectrum back
        swap_spectrum_halves(m_fftdelay_buffer_p1.data(), m_plan.nfft,
                             m_plan.nsub * m_plan.nchan, m_plan.mbin);
        swap_spectrum_halves(m_fftdelay_buffer_p2.data(), m_plan.nfft,
                             m_plan.nsub * m_plan.nchan, m_plan.mbin);
        // Perform the backward FFT
        fftwf_execute_dft(
            m_fft_plan_bw,
            reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p1.data()),
            reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p1.data()));
        fftwf_execute_dft(
            m_fft_plan_bw,
            reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p2.data()),
            reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p2.data()));
        // Detect the intensity
        unpad_detect(m_fftdelay_buffer_p1.data(), m_fftdelay_buffer_p2.data(),
                     m_fftdelay_buffer_p1.size(), m_intensity_buffer.data(),
                     m_intensity_buffer.size());

        // Perform inter-channel dedispersion at the current coherent DM
        dm_utils::dedisperse(m_intensity_buffer.data(),
                             m_intensity_buffer.size(), m_plan.dm_grid_coh[idm],
                             m_plan.f_min, m_plan.f_max, m_plan.nchan,
                             m_plan.msamp, m_plan.tsamp);

        // Perform the FDMT
        float* dmt_cur        = &dmt[idm * m_plan.dt_max * m_plan.msamp];
        SizeType dmt_cur_size = m_plan.dt_max * m_plan.msamp;
        m_thefdmt->execute(m_intensity_buffer.data(), m_intensity_buffer.size(),
                           dmt_cur, dmt_cur_size);
    }
}

void CohFDMTCPU::initialise() {
    m_unpacked_buffer_p1.resize(m_plan.nfft * m_plan.nsub * m_plan.nbin);
    m_unpacked_buffer_p2.resize(m_plan.nfft * m_plan.nsub * m_plan.nbin);
    m_fftdelay_buffer_p1.resize(m_plan.nfft * m_plan.nsub * m_plan.nbin);
    m_fftdelay_buffer_p2.resize(m_plan.nfft * m_plan.nsub * m_plan.nbin);
    m_intensity_buffer.resize(m_plan.nsub * m_plan.nchan * m_plan.msamp);
    m_chirp_table.resize(m_plan.dm_grid_coh.size() * m_plan.nsub * m_plan.nbin);
    // Compute the chirp table
    dm_utils::compute_chirp(m_chirp_table.data(), m_chirp_table.size(),
                            m_plan.dm_grid_coh.data(),
                            m_plan.dm_grid_coh.size(), m_plan.fcenter,
                            m_plan.bw, m_plan.nbin, m_plan.nsub, m_plan.nchan);

    // Generate FFT plan (batch in-place forward and backward FFT)
    const std::array<int, 1> fft_size_fw = {static_cast<int>(m_plan.nbin)};
    const std::array<int, 1> fft_size_bw = {static_cast<int>(m_plan.mbin)};

    m_fft_plan_fw = fftwf_plan_many_dft(
        1, fft_size_fw.data(), static_cast<int>(m_plan.nfft * m_plan.nsub),
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p1.data()), nullptr,
        1, static_cast<int>(m_plan.nbin),
        reinterpret_cast<fftwf_complex*>(m_unpacked_buffer_p1.data()), nullptr,
        1, static_cast<int>(m_plan.nbin), FFTW_FORWARD, FFTW_ESTIMATE);
    m_fft_plan_bw = fftwf_plan_many_dft(
        1, fft_size_bw.data(),
        static_cast<int>(m_plan.nfft * m_plan.nsub * m_plan.nchan),
        reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p1.data()), nullptr,
        1, static_cast<int>(m_plan.mbin),
        reinterpret_cast<fftwf_complex*>(m_fftdelay_buffer_p1.data()), nullptr,
        1, static_cast<int>(m_plan.mbin), FFTW_BACKWARD, FFTW_ESTIMATE);
    if (m_fft_plan_fw == nullptr || m_fft_plan_bw == nullptr) {
        throw std::runtime_error("Failed to create FFTW plan");
    }
    m_thefdmt =
        std::make_unique<FDMTCPU>(m_plan.f_min, m_plan.f_max, m_plan.mchan,
                                  m_plan.msamp, m_plan.tsamp, m_plan.dt_max);
}

void CohFDMTCPU::unpack_init(const uint8_t* __restrict data_in,
                             SizeType in_size,
                             std::string& in_order,
                             ComplexType* __restrict data_p1,
                             ComplexType* __restrict data_p2,
                             SizeType out_size) const {
    DataOrder order = string_to_data_order(in_order);
    if (order == DataOrder::kFTPRI) {
        DataUnpacker<DataOrder::kFTPRI> unpacker(m_plan.nsub, m_plan.nbin,
                                                 m_plan.noverlap, m_plan.nfft);
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else if (order == DataOrder::kPRITF) {
        DataUnpacker<DataOrder::kPRITF> unpacker(m_plan.nsub, m_plan.nbin,
                                                 m_plan.noverlap, m_plan.nfft);
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else if (order == DataOrder::kRITFP) {
        DataUnpacker<DataOrder::kRITFP> unpacker(m_plan.nsub, m_plan.nbin,
                                                 m_plan.noverlap, m_plan.nfft);
        unpacker.unpack_and_padd(data_in, in_size, data_p1, data_p2, out_size);
    } else {
        throw std::runtime_error("Invalid order: " + in_order);
    }
}

void CohFDMTCPU::unpad_detect(const ComplexType* __restrict fft_p1,
                              const ComplexType* __restrict fft_p2,
                              SizeType in_size,
                              float* __restrict intensity,
                              SizeType out_size) const {
    if (in_size != m_plan.nfft * m_plan.nsub * m_plan.nbin) {
        throw std::runtime_error("Invalid input size");
    }
    if (out_size != m_plan.nsub * m_plan.nchan * m_plan.msamp) {
        throw std::runtime_error("Invalid output size");
    }

    const SizeType noverlap_per_channel = m_plan.noverlap / m_plan.nchan;
    const SizeType mbin_adjusted = m_plan.mbin - 2 * noverlap_per_channel;
    const SizeType msamp         = m_plan.nfft * mbin_adjusted;

#ifdef USE_OPENMP
#pragma omp parallel for collapse(4)
#endif
    for (SizeType ibin = 0; ibin < mbin_adjusted; ++ibin) {
        for (SizeType ichan = 0; ichan < m_plan.nchan; ++ichan) {
            for (SizeType ifft = 0; ifft < m_plan.nfft; ++ifft) {
                for (SizeType isub = 0; isub < m_plan.nsub; ++isub) {
                    const SizeType isamp         = ibin + mbin_adjusted * ifft;
                    const SizeType ibin_adjusted = ibin + noverlap_per_channel;
                    const SizeType src_idx =
                        ifft * m_plan.nsub * m_plan.nchan * m_plan.mbin +
                        (m_plan.nsub - isub - 1) * m_plan.nchan * m_plan.mbin +
                        ichan * m_plan.mbin + ibin_adjusted;
                    const SizeType dst_idx =
                        isub * m_plan.nchan * msamp +
                        (m_plan.nchan - ichan - 1) * msamp + isamp;
                    intensity[dst_idx] =
                        std::norm(fft_p1[src_idx]) + std::norm(fft_p2[src_idx]);
                }
            }
        }
    }
}

void CohFDMTCPU::swap_spectrum_halves(ComplexType* __restrict data_in,
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

void CohFDMTCPU::pointwise_complex_multiply(const ComplexType* __restrict a,
                                            const ComplexType* __restrict b,
                                            ComplexType* __restrict c,
                                            SizeType nx,
                                            SizeType ny,
                                            SizeType idm,
                                            float scale) {
    for (SizeType i = 0; i < nx; ++i) {
        for (SizeType j = 0; j < ny; ++j) {
            c[i + nx * j] = a[i + nx * j] * b[i + nx * idm] * scale;
        }
    }
}
