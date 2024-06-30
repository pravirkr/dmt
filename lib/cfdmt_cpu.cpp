#include <complex>
#include <stdexcept>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <dmt/cfdmt_base.hpp>
#include <dmt/cfdmt_cpu.hpp>

// Unpack and pad the input data to generate complex timeseries
// ordered as polarisation-Real/Imag-time-frequency (PTF) - LOFAR
// data_in shape: (npol=2, 2, nsamp, nsub)
// noverlap = n_d * n_c // 2
// nsamp = nfft * (nbin - 2 * noverlap)
// data_out shape: (2, nfft, nsub, nbin)
void unpack_and_padd_ptf(const uint8_t* __restrict data_in,
                         SizeType in_size,
                         std::complex<float>* __restrict data_out,
                         SizeType out_size,
                         SizeType nsamp,
                         SizeType nsub,
                         SizeType nbin,
                         SizeType noverlap,
                         SizeType nfft) {
    if (in_size != 4 * nsamp * nsub) {
        throw std::runtime_error("Invalid input size");
    }
    if (out_size != 2 * nfft * nsub * nbin) {
        throw std::runtime_error("Invalid output size");
    }
    for (SizeType ipol = 0; ipol < 2; ++ipol) {
        auto pol_offset = ipol * 2 * nsamp * nsub;
        for (SizeType ifft = 0; ifft < nfft; ++ifft) {
            for (SizeType isub = 0; isub < nsub; ++isub) {
                for (SizeType ibin = 0; ibin < nbin; ++ibin) {
                    auto isamp = ibin + (nbin - 2 * noverlap) * ifft - noverlap;
                    if (isamp >= 0 && isamp < nsamp) {
                        auto idx_in_re = pol_offset + (0 * nsub * nsamp) +
                                         (nsub * isamp) + isub;
                        auto idx_in_im = pol_offset + (1 * nsub * nsamp) +
                                         (nsub * isamp) + isub;
                        auto idx_out = ipol * nfft * nsub * nbin +
                                       ifft * nsub * nbin + isub * nbin + ibin;
                        data_out[idx_out] = std::complex<float>(
                            data_in[idx_in_re], data_in[idx_in_im]);
                    }
                }
            }
        }
    }
}

CohFDMTCPU::CohFDMTCPU(float f_center,
                       float sub_bw,
                       SizeType nsub,
                       float tbin,
                       SizeType nbin,
                       SizeType nfft,
                       SizeType nchan,
                       float dm_max,
                       float dm_step,
                       float dm_min)
    : CohFDMT(f_center,
              sub_bw,
              nsub,
              tbin,
              nbin,
              nfft,
              nchan,
              dm_max,
              dm_step,
              dm_min) {}

void CohFDMTCPU::set_num_threads(int nthreads) {
#ifdef USE_OPENMP
    omp_set_num_threads(nthreads);
#endif
}

void CohFDMTCPU::execute(const uint8_t* __restrict data_in,
                         SizeType in_size,
                         std::string in_order) {
    if (in_order == "ftp") {
        execute_ftp(data_in, in_size);
    } else if (in_order == "ptf") {
        execute_ptf(data_in, in_size);
    } else if (in_order == "tfp") {
        execute_tfp(data_in, in_size);
    } else {
        throw std::runtime_error("Invalid order: " + in_order);
    }
}

void CohFDMTCPU::execute_ptf(const uint8_t* __restrict data_in,
                             SizeType in_size) {
    //unpack_and_padd_ptf(data_in, in_size, m_data, m_data.size(), m_nsamp,
    //                    m_nsub, m_nbin, m_noverlap, m_nfft);
}