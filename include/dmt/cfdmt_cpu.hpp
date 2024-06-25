#pragma once

#include <vector>

#include <dmt/cfdmt_base.hpp>

class CohFDMTCPU : public CohFDMT {
public:
    CohFDMTCPU(float f_center,
               float sub_bw,
               SizeType nsub,
               float tbin,
               SizeType nbin,
               SizeType nfft,
               SizeType nchan,
               float dm_max,
               float dm_step,
               float dm_min = 0.0F);
    static void set_num_threads(int nthreads);
    void execute(const uint8_t* __restrict data_in,
                 SizeType in_size,
                 std::string in_order) override;

private:
    void execute_ftp(const uint8_t* __restrict data_in,
                     SizeType in_size) override;
    void execute_ptf(const uint8_t* __restrict data_in,
                     SizeType in_size) override;
    void execute_tfp(const uint8_t* __restrict data_in,
                     SizeType in_size) override;
};
