#pragma once

#include <cstddef>
#include <vector>

#include <dmt/dmt_types.hpp>

struct CohFDMTPlan {
    // Input parameters
    float fcenter;
    float bwsub;
    SizeType nsub;
    float tbin;
    SizeType nbin;
    SizeType nfft;
    float t_p;
    float dm_max;
    float dm_min;
    SizeType noverlap_inp;

    // Derived parameters
    float bw;
    float f_min;
    float f_max;
    SizeType n_p;
    SizeType nchan;
    std::vector<float> dm_grid_coh;
    std::vector<float> dm_grid_final;
    SizeType noverlap;
    SizeType nsamp;
    SizeType mbin;
    SizeType mchan;
    SizeType msamp;
    float tsamp;
    SizeType dt_max;

    CohFDMTPlan(float fcenter,
                float bwsub,
                SizeType nsub,
                float tbin,
                SizeType nbin,
                SizeType nfft,
                float t_p,
                float dm_max,
                float dm_min,
                SizeType noverlap_inp);
};