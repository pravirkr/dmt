#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <fftw3.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/fdmt/fdmt_cpu.hpp"

class FFTManager {
public:
    FFTManager(SizeType nfft,
               SizeType nsub,
               SizeType nbin,
               SizeType mbin,
               SizeType nchan,
               SizeType nthreads = 1);
    FFTManager(const FFTManager&)            = delete;
    FFTManager& operator=(const FFTManager&) = delete;
    FFTManager(FFTManager&&)                 = delete;
    FFTManager& operator=(FFTManager&&)      = delete;
    ~FFTManager();

    void initialize_plans(ComplexType* __restrict__ unpack_buffer,
                          ComplexType* __restrict__ delay_buffer);
    void forward_fft(ComplexType* __restrict__ data) const;
    void backward_fft(ComplexType* __restrict__ data) const;
    static void swap_spectrum(ComplexType* __restrict__ data,
                              SizeType nz,
                              SizeType ny,
                              SizeType nx);

private:
    SizeType m_nfft;
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_mbin;
    SizeType m_nchan;
    SizeType m_nthreads;

    fftwf_plan m_forward_plan  = nullptr;
    fftwf_plan m_backward_plan = nullptr;
};

class CohFDMTCPU {
public:
    CohFDMTCPU(float f_center,
               float sub_bw,
               SizeType nsub,
               float tbin,
               SizeType nbin,
               SizeType nfft,
               float tp,
               float dm_max,
               float dm_min      = 0.0F,
               SizeType noverlap = 8192,
               SizeType nthreads = 1);

    CohFDMTCPU(const CohFDMTCPU&)            = delete;
    CohFDMTCPU& operator=(const CohFDMTCPU&) = delete;
    CohFDMTCPU(CohFDMTCPU&&)                 = delete;
    CohFDMTCPU& operator=(CohFDMTCPU&&)      = delete;
    ~CohFDMTCPU()                            = default;

    const CohFDMTPlan& get_plan() const;
    SizeType get_dmt_size() const;
    void execute(const uint8_t* __restrict__ data_in,
                 SizeType in_size,
                 const std::string& in_order,
                 float* __restrict__ dmt,
                 SizeType dmt_size);

private:
    CohFDMTPlan m_plan;
    std::unique_ptr<FFTManager> m_thefft;
    std::unique_ptr<FDMTCPU> m_thefdmt;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<ComplexType> m_chirp_table;

    void initialise();

    void unpack_init(const uint8_t* __restrict__ data_in,
                     SizeType in_size,
                     const std::string& in_order,
                     ComplexType* __restrict__ data_p1,
                     ComplexType* __restrict__ data_p2,
                     SizeType out_size) const;
    void unpad_detect(const ComplexType* __restrict__ fft_p1,
                      const ComplexType* __restrict__ fft_p2,
                      SizeType in_size,
                      float* __restrict__ intensity,
                      SizeType out_size) const;
    void apply_chirp(const ComplexType* __restrict__ data_in,
                     ComplexType* __restrict__ data_out,
                     SizeType idm);
};
