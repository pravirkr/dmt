#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

using SizeType = std::size_t;

class CohFDMT {
public:
    CohFDMT(float f_center,
            float subband_bw,
            SizeType nsubbands,
            SizeType nchans,
            float tbin,
            SizeType nbin,
            SizeType nfft,
            float dm_max,
            float dm_step,
            float dm_min = 0.0F);
    CohFDMT(const CohFDMT&)            = delete;
    CohFDMT& operator=(const CohFDMT&) = delete;
    CohFDMT(CohFDMT&&)                 = delete;
    CohFDMT& operator=(CohFDMT&&)      = delete;
    virtual ~CohFDMT()                 = default;


private:
    std::vector<float> m_dm_grid;

};