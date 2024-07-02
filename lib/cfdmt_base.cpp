#include <algorithm>
#include <cmath>
#include <cstddef>

#include "dmt/dm_utils.hpp"
#include <dmt/cfdmt_base.hpp>

CohFDMTPlan::CohFDMTPlan(float fcenter,
                         float bwsub,
                         SizeType nsub,
                         float tbin,
                         SizeType nbin,
                         SizeType nfft,
                         float t_p,
                         float dm_max,
                         float dm_min,
                         SizeType noverlap_inp)
    : fcenter(fcenter),
      bwsub(bwsub),
      nsub(nsub),
      tbin(tbin),
      nbin(nbin),
      nfft(nfft),
      t_p(t_p),
      dm_max(dm_max),
      dm_min(dm_min),
      noverlap_inp(noverlap_inp) {
    if (nsub < 1) {
        throw std::invalid_argument("nsub must be greater than 0");
    }
    bw    = bwsub * static_cast<float>(nsub);
    f_min = fcenter - bw / 2;
    f_max = fcenter + bw / 2;
    n_p   = static_cast<SizeType>(std::ceil(t_p / tbin));
    nchan = n_p;

    dm_grid =
        dm_utils::generate_coherent_dms(dm_min, dm_max, fcenter, bw, tbin, t_p);
    if (dm_grid.empty()) {
        throw std::runtime_error("Empty DM grid");
    }

    auto noverlap_optimal = dm_utils::minimum_overlap(
        *std::max_element(dm_grid.begin(), dm_grid.end()), fcenter, bw, tbin,
        nsub, nchan);
    auto noverlap_optimal_pow2 = static_cast<SizeType>(
        std::pow(2, std::round(std::log2(noverlap_optimal))));
    noverlap = std::max(noverlap_inp, noverlap_optimal_pow2);
    if (nbin < 2 * noverlap) {
        throw std::invalid_argument("nbin must be greater than 2 * noverlap");
    }
    nsamp = nfft * (nbin - 2 * noverlap);
    mbin  = nbin / nchan;
    mchan = nsub * nchan;
    msamp = nsamp / nchan;
    tsamp = tbin * static_cast<float>(nchan);
    dt_max = n_p;
}
