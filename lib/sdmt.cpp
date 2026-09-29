#include "dmt/algorithms/sdmt.hpp"

#include <cstdint>
#include <span>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

SDMT::SDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           float dm_max,
           float dm_step,
           float dm_min,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : DDMT(plans::DDMTPlan(f_min,
                           f_max,
                           nchans,
                           tsamp,
                           dm_max,
                           dm_step,
                           dm_min,
                           nbits,
                           kill_mask),
           exec,
           nbeams,
           EngineKind::kSharedSums) {}

SDMT::SDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           std::span<const float> dm_arr,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : DDMT(plans::DDMTPlan(
               f_min, f_max, nchans, tsamp, dm_arr, nbits, kill_mask),
           exec,
           nbeams,
           EngineKind::kSharedSums) {}

SDMT::SDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           const plans::LevinConfig& levin,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : DDMT(
          plans::DDMTPlan(f_min, f_max, nchans, tsamp, levin, nbits, kill_mask),
          exec,
          nbeams,
          EngineKind::kSharedSums) {}

SDMT::SDMT(const plans::DDMTPlan& plan, Exec exec, SizeType nbeams)
    : DDMT(plan, exec, nbeams, EngineKind::kSharedSums) {}

} // namespace dmt::algorithms
