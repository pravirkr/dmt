#pragma once

#include <memory>
#include <string>
#include <vector>

#include <fftw3.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/common/unpacker.hpp"
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
    static void
    swap_spectrum(ComplexType* __restrict__ data, SizeType nx, SizeType ny);

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
               float bw_sub,
               SizeType nsub,
               float tbin,
               SizeType nbin,
               SizeType nfft,
               float t_p,
               float dm_max,
               float dm_min                  = 0.0F,
               SizeType noverlap             = 8192,
               const std::string& data_order = "PRITF",
               int nthreads                  = 1,
               bool verbose                  = false);

    CohFDMTCPU(const CohFDMTCPU&)            = delete;
    CohFDMTCPU& operator=(const CohFDMTCPU&) = delete;
    CohFDMTCPU(CohFDMTCPU&&)                 = delete;
    CohFDMTCPU& operator=(CohFDMTCPU&&)      = delete;
    ~CohFDMTCPU()                            = default;

    const CohFDMTPlan& get_plan() const;

    template <typename DataType>
    void execute(const DataType* __restrict__ data_in,
                 SizeType in_size,
                 float* __restrict__ dmt,
                 SizeType dmt_size);

private:
    CohFDMTPlan m_plan;
    std::unique_ptr<FFTManager> m_thefft;
    std::unique_ptr<FDMTCPU> m_thefdmt;
    std::unique_ptr<DataUnpacker> m_theunpacker;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<ComplexType> m_chirp_table;

    void initialise();

    void unpad_detect(const ComplexType* __restrict__ fft_p1,
                      const ComplexType* __restrict__ fft_p2,
                      SizeType in_size,
                      float* __restrict__ intensity,
                      SizeType out_size) const;
    void apply_chirp(const ComplexType* __restrict__ data_in,
                     ComplexType* __restrict__ data_out,
                     SizeType idm);
};
