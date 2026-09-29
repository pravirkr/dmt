#pragma once

#include <cstdint>
#include <span>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

/**
 * @file sdmt.hpp
 * @brief Subband-shared Dispersion Measure Transform (SDMT): exact
 * delay-and-sum dedispersion that shares partial sums between DM trials.
 */

namespace dmt::algorithms {

/**
 * @brief Subband-shared Dispersion Measure Transform (SDMT).
 *
 * Computes exactly the DDMT sums,
 * @f[
 * D[\text{DM}, t] = \sum_{\nu} I[\nu, t + \Delta t(\nu, \text{DM})],
 * @f]
 * with the same delay table (any DM grid, kill mask), but with fewer
 * additions. Within subbands of 16 channels, trials whose integer delays
 * agree (relative to a per-trial base) have identical partial sums, which
 * are computed once and reused. Nothing is approximated: integer results
 * are bit-identical to DDMT, and float results differ only by the order of
 * additions (rounding).
 *
 * The API, streaming model and history are those of DDMT (SDMT is-a DDMT),
 * on every backend. On a GPU, the sharing is organised in a fixed two-level
 * hierarchy per DM tile (unique 4-channel sums, then unique 16-channel sums
 * built from them); when a DM grid shares too little for that to pay
 * (sparse or coarse grids), the GPU runs the DDMT kernel instead, with the
 * same result. Constructing an SDMT builds the sharing plan, which takes
 * longer than a DDMT.
 */
class SDMT final : public DDMT {
public:
    /// @brief Linear DM grid; see the matching DDMT constructor.
    SDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min                       = 0.0F,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /// @brief Explicit DM trial array; see the matching DDMT constructor.
    SDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         std::span<const float> dm_arr,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /// @brief Lina Levin DM grid; see the matching DDMT constructor.
    SDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const plans::LevinConfig& levin,
         Exec exec                          = {},
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         SizeType nbeams                    = 1);

    /// @brief From a pre-configured DDMTPlan.
    explicit SDMT(const plans::DDMTPlan& plan,
                  Exec exec       = {},
                  SizeType nbeams = 1);
};

} // namespace dmt::algorithms
