#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <fftw3.h>

#include <dmt/dmt_plans.hpp>
#include <dmt/dmt_types.hpp>
#include <dmt/fdmt_cpu.hpp>

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
               SizeType noverlap = 8192);

    CohFDMTCPU(const CohFDMTCPU&)            = delete;
    CohFDMTCPU& operator=(const CohFDMTCPU&) = delete;
    CohFDMTCPU(CohFDMTCPU&&)                 = delete;
    CohFDMTCPU& operator=(CohFDMTCPU&&)      = delete;
    ~CohFDMTCPU();

    const CohFDMTPlan& get_plan() const;
    SizeType get_dmt_size() const;
    static void set_num_threads(int nthreads);
    void execute(const uint8_t* __restrict data_in,
                 SizeType in_size,
                 std::string in_order,
                 float* __restrict dmt,
                 SizeType dmt_size);

private:
    CohFDMTPlan m_plan;

    std::vector<ComplexType> m_unpacked_buffer_p1;
    std::vector<ComplexType> m_unpacked_buffer_p2;
    std::vector<ComplexType> m_fftdelay_buffer_p1;
    std::vector<ComplexType> m_fftdelay_buffer_p2;
    std::vector<float> m_intensity_buffer;
    std::vector<ComplexType> m_chirp_table;

    fftwf_plan m_fft_plan_fw = nullptr;
    fftwf_plan m_fft_plan_bw = nullptr;
    std::unique_ptr<FDMTCPU> m_thefdmt;

    void initialise();

    void unpack_init(const uint8_t* __restrict data_in,
                     SizeType in_size,
                     std::string& in_order,
                     ComplexType* __restrict data_p1,
                     ComplexType* __restrict data_p2,
                     SizeType out_size) const;
    void unpad_detect(const ComplexType* __restrict fft_p1,
                      const ComplexType* __restrict fft_p2,
                      SizeType in_size,
                      float* __restrict intensity,
                      SizeType out_size) const;

    static void swap_spectrum_halves(ComplexType* __restrict data_in,
                                     SizeType nz,
                                     SizeType ny,
                                     SizeType nx);
    static void pointwise_complex_multiply(const ComplexType* __restrict a,
                                           const ComplexType* __restrict b,
                                           ComplexType* __restrict c,
                                           size_t nx,
                                           size_t ny,
                                           size_t idm,
                                           float scale);
};
