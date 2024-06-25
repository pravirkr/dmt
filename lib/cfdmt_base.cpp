#include <dmt/cfdmt_base.hpp>

CohFDMT::CohFDMT(float f_center,
                 float sub_bw,
                 SizeType nsub,
                 float tbin,
                 SizeType nbin,
                 SizeType nfft,
                 SizeType nchan,
                 float dm_max,
                 float dm_step,
                 float dm_min)
    : m_fcenter(f_center),
      m_sub_bw(sub_bw),
      m_nsub(nsub),
      m_tbin(tbin),
      m_nbin(nbin),
      m_nfft(nfft),
      m_nchan(nchan) {
    // Generate the DM grid
    const auto ndm = static_cast<SizeType>((dm_max - dm_min) / dm_step) + 1;
    m_dm_grid.resize(ndm);
    for (SizeType i = 0; i < ndm; ++i) {
        m_dm_grid[i] = dm_min + i * dm_step;
    }
}