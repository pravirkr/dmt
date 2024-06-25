#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using SizeType = std::size_t;

class CohFDMT {
public:
    CohFDMT(float f_center,
            float sub_bw,
            SizeType nsub,
            float tbin,
            SizeType nbin,
            SizeType nfft,
            SizeType nchan,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);
    CohFDMT(const CohFDMT&)            = delete;
    CohFDMT& operator=(const CohFDMT&) = delete;
    CohFDMT(CohFDMT&&)                 = delete;
    CohFDMT& operator=(CohFDMT&&)      = delete;
    virtual ~CohFDMT()                 = default;

    virtual void execute(const uint8_t* __restrict data_in,
                         SizeType in_size,
                         std::string in_order) = 0;

protected:
    float m_fcenter;
    float m_sub_bw;
    SizeType m_nsub;
    float m_tbin;
    SizeType m_nbin;
    SizeType m_nfft;
    SizeType m_nchan;

private:
    std::vector<float> m_dm_grid;

    virtual void execute_ftp(const uint8_t* __restrict data_in,
                             SizeType in_size) = 0;
    virtual void execute_ptf(const uint8_t* __restrict data_in,
                             SizeType in_size) = 0;
    virtual void execute_tfp(const uint8_t* __restrict data_in,
                             SizeType in_size) = 0;
};