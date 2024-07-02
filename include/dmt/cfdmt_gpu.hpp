#pragma once

#include <string>
#include <thrust/device_vector.h>

#include <dmt/cfdmt_base.hpp>
#include <dmt/dmt_types.hpp>
#include <dmt/fdmt_gpu.hpp>

using ComplexTypeGPU = thrust::complex<float>;

class CohFDMTGPU {
public:
    CohFDMTGPU(float f_center,
               float sub_bw,
               SizeType nsub,
               float tbin,
               SizeType nbin,
               SizeType nfft,
               float tp,
               float dm_max,
               float dm_min      = 0.0F,
               SizeType noverlap = 8192);

    CohFDMTGPU(const CohFDMTGPU&)            = delete;
    CohFDMTGPU& operator=(const CohFDMTGPU&) = delete;
    CohFDMTGPU(CohFDMTGPU&&)                 = delete;
    CohFDMTGPU& operator=(CohFDMTGPU&&)      = delete;
    ~CohFDMTGPU();

    CohFDMTPlan get_plan() const;
    SizeType get_dmt_size() const;
    void execute(const uint8_t* __restrict data_in,
                 SizeType in_size,
                 std::string in_order,
                 float* __restrict dmt,
                 SizeType dmt_size);

    void execute(const uint8_t* __restrict data_in,
                 SizeType in_size,
                 std::string in_order,
                 float* __restrict dmt,
                 SizeType dmt_size,
                 bool device_flags);

private:
    CohFDMTPlan m_plan;

    thrust::device_vector<ComplexTypeGPU> m_unpacked_buffer_p1;
    thrust::device_vector<ComplexTypeGPU> m_unpacked_buffer_p2;
    thrust::device_vector<ComplexTypeGPU> m_fftdelay_buffer_p1;
    thrust::device_vector<ComplexTypeGPU> m_fftdelay_buffer_p2;
    thrust::device_vector<float> m_intensity_buffer;
    thrust::device_vector<ComplexTypeGPU> m_chirp_table;

    std::unique_ptr<FDMTGPU> m_thefdmt;

    void initialise();

    void execute_device(const uint8_t* __restrict data_in,
                        SizeType in_size,
                        std::string in_order,
                        float* __restrict dmt,
                        SizeType dmt_size);

    void unpack_init(const uint8_t* __restrict data_in,
                     SizeType in_size,
                     std::string& in_order,
                     ComplexTypeGPU* __restrict data_p1,
                     ComplexTypeGPU* __restrict data_p2,
                     SizeType out_size);
    void unpad_detect(const ComplexTypeGPU* __restrict fft_p1,
                      const ComplexTypeGPU* __restrict fft_p2,
                      SizeType in_size,
                      float* __restrict intensity,
                      SizeType out_size);

    static void swap_spectrum_halves(ComplexTypeGPU* __restrict data_in,
                                     SizeType nz,
                                     SizeType ny,
                                     SizeType nx);
    static void pointwise_complex_multiply(const ComplexTypeGPU* __restrict a,
                                           const ComplexTypeGPU* __restrict b,
                                           ComplexTypeGPU* __restrict c,
                                           size_t nx,
                                           size_t ny,
                                           size_t idm);
};
