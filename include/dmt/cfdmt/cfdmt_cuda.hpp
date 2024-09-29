#pragma once

#include <string>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/fdmt/fdmt_cuda.hpp"

class CUDAFFTManager {
public:
    CUDAFFTManager(SizeType nfft,
                   SizeType nsub,
                   SizeType nbin,
                   SizeType mbin,
                   SizeType nchan);
    CUDAFFTManager(const CUDAFFTManager&)            = delete;
    CUDAFFTManager& operator=(const CUDAFFTManager&) = delete;
    CUDAFFTManager(CUDAFFTManager&&)                 = delete;
    CUDAFFTManager& operator=(CUDAFFTManager&&)      = delete;
    ~CUDAFFTManager();

    void initialize_plans(ComplexTypeCUDA* __restrict__ unpack_buffer,
                          ComplexTypeCUDA* __restrict__ delay_buffer);
    void forward_fft(ComplexTypeCUDA* __restrict__ data) const;
    void backward_fft(ComplexTypeCUDA* __restrict__ data) const;
    static void swap_spectrum(ComplexTypeCUDA* __restrict__ data,
                              SizeType nz,
                              SizeType ny,
                              SizeType nx);

private:
    SizeType m_nfft;
    SizeType m_nsub;
    SizeType m_nbin;
    SizeType m_mbin;
    SizeType m_nchan;

    cufftHandle m_forward_plan  = nullptr;
    cufftHandle m_backward_plan = nullptr;
};

class CohFDMTCUDA {
public:
    CohFDMTCUDA(float f_center,
                float sub_bw,
                SizeType nsub,
                float tbin,
                SizeType nbin,
                SizeType nfft,
                float tp,
                float dm_max,
                float dm_min      = 0.0F,
                SizeType noverlap = 8192);

    CohFDMTCUDA(const CohFDMTCUDA&)            = delete;
    CohFDMTCUDA& operator=(const CohFDMTCUDA&) = delete;
    CohFDMTCUDA(CohFDMTCUDA&&)                 = delete;
    CohFDMTCUDA& operator=(CohFDMTCUDA&&)      = delete;
    ~CohFDMTCUDA();

    const CohFDMTPlan& get_plan() const;
    SizeType get_dmt_size() const;
    void execute(const uint8_t* __restrict__ data_in,
                 SizeType in_size,
                 const std::string& in_order,
                 float* __restrict__ dmt,
                 SizeType dmt_size);

    void execute(const uint8_t* __restrict__ data_in,
                 SizeType in_size,
                 const std::string& in_order,
                 float* __restrict__ dmt,
                 SizeType dmt_size,
                 bool device_flags);

private:
    CohFDMTPlan m_plan;
    std::unique_ptr<CUDAFFTManager> m_thefft;
    std::unique_ptr<FDMTCUDA> m_thefdmt;

    DeviceVector<ComplexTypeCUDA> m_unpack_buf_p1;
    DeviceVector<ComplexTypeCUDA> m_unpack_buf_p2;
    DeviceVector<ComplexTypeCUDA> m_delay_buf_p1;
    DeviceVector<ComplexTypeCUDA> m_delay_buf_p2;
    DeviceVector<float> m_intensity_buf;
    DeviceVector<ComplexTypeCUDA> m_chirp_table;

    void initialise();

    void execute_device(const uint8_t* __restrict__ data_in,
                        SizeType in_size,
                        const std::string& in_order,
                        float* __restrict__ dmt,
                        SizeType dmt_size);

    void unpack_init(const uint8_t* __restrict__ data_in,
                     SizeType in_size,
                     const std::string& in_order,
                     ComplexTypeCUDA* __restrict__ data_p1,
                     ComplexTypeCUDA* __restrict__ data_p2,
                     SizeType out_size) const;
    void unpad_detect(const ComplexTypeCUDA* __restrict__ fft_p1,
                      const ComplexTypeCUDA* __restrict__ fft_p2,
                      SizeType in_size,
                      float* __restrict__ intensity,
                      SizeType out_size) const;
    void apply_chirp(const ComplexTypeCUDA* __restrict__ data_in,
                     ComplexTypeCUDA* __restrict__ data_out,
                     SizeType idm);
};
