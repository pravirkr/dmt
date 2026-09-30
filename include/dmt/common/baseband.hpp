#pragma once

/**
 * @file baseband.hpp
 * @brief Baseband voltage format descriptor and the CohFDMT search
 * configuration.
 */

#include <array>
#include <string>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt {

/**
 * @brief Memory layout and sample encoding of a block of complex, dual-
 * polarisation baseband voltages.
 *
 * A block is a 4-D array over the axes
 * - `P`  polarisation (2),
 * - `RI` real/imaginary component (2, real first),
 * - `T`  time (the block's samples per subband),
 * - `F`  subband (the group's subbands, ascending frequency),
 *
 * stored in the order given by @ref order, outermost axis first. Every
 * permutation of the four tokens is accepted, for example:
 *
 * | order     | used by                                           |
 * |-----------|---------------------------------------------------|
 * | `"FTPRI"` | GUPPI raw (GBT, Parkes, Breakthrough Listen)      |
 * | `"PRITF"` | LOFAR (one Xr/Xi/Yr/Yi stream per pol component)  |
 * | `"TFPRI"` | PSRDADA time-major (e.g. MeerKAT beamformer)      |
 *
 * Each element is one real number of @ref nbits bits. Elements are counted
 * in memory order; for nbits < 8 consecutive elements share a byte (see
 * @ref msb_first).
 *
 * Each subband is complex baseband centred on DC at its centre frequency
 * and critically sampled (sampling interval 1 / bw_sub), upper sideband.
 */
struct BasebandFormat {
    /// Axis order, outermost first: a permutation of "P", "RI", "T", "F".
    std::string order{"FTPRI"};
    /// Bits per real element: 2, 4 or 8.
    SizeType nbits{8};
    /// 4/8-bit: two's complement (true) or offset binary (false; the value
    /// is code - 2^(nbits-1), e.g. uint8 - 128). Ignored for 2-bit.
    bool is_signed{true};
    /// Sub-byte packing: the first element of a byte sits in its most
    /// significant bits (true; GUPPI 4-bit, real in the high nibble) or its
    /// least significant bits (false; VDIF 2-bit).
    bool msb_first{true};
    /// 2-bit: value of each code 0..3 (default: VDIF offset binary).
    std::array<float, 4> levels_2bit{-3.3359F, -1.0F, 1.0F, 3.3359F};
};

/**
 * @brief Parameters of a CohFDMT (hybrid coherent + FDMT) search.
 *
 * Only the first six fields are required; the rest default to automatic
 * choices. Use C++20 designated initialisers, e.g.
 * @code
 * dmt::CohFDMTConfig cfg{.f_center = 1406.25F, .bw_sub = 2.9296875F,
 *                        .nsub = 64, .t_p = 10.0E-6F,
 *                        .dm_min = 50.0F, .dm_max = 60.0F};
 * @endcode
 */
struct CohFDMTConfig {
    /// Centre frequency of the whole band (all subbands), MHz.
    float f_center{};
    /// Bandwidth of one subband, MHz (> 0). The subband sampling interval
    /// is 1 / bw_sub.
    float bw_sub{};
    /// Number of subbands (all groups together).
    SizeType nsub{};
    /// Target detected time resolution, s. Rounded to n_p subband samples
    /// with n_p 2,3,5,7-smooth (see CohFDMTPlan::get_tsamp()).
    float t_p{};
    /// Lowest DM searched, pc cm^-3 (>= 0).
    float dm_min{};
    /// Highest DM searched, pc cm^-3 (>= dm_min).
    float dm_max{};
    /// Raw samples per subband the caller reads per block; 0 picks one.
    /// Rounded down to whole FFT blocks (see CohFDMTPlan::get_block_nsamps()).
    SizeType block_nsamps{0};
    /// Forward FFT length (a multiple of n_p); 0 picks one.
    SizeType nbin{0};
    /// Largest intra-channel dispersion smearing left by the coarse
    /// coherent grid, in output samples (tsamp). 1 matches Zackay & Ofek.
    float smear_tol{1.0F};
    /// Stride between fine FDMT delay trials; > 1 trades DM resolution for
    /// fewer output rows (for pulses wider than dt_step samples).
    SizeType dt_step{1};
    /// Normalise every channel to zero mean and unit variance before the
    /// FDMT (then get_effective_variance_grid() is the output variance).
    bool normalize{true};
    /// Input layout and encoding.
    BasebandFormat format{};
    /// Subbands per input group (e.g. one group per GUPPI node file), in
    /// ascending frequency; must sum to nsub. Empty = one group.
    std::vector<SizeType> subband_groups;
};

} // namespace dmt
