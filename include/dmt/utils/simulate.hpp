#pragma once

#include <tuple>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::utils {

std::tuple<std::vector<float>, SizeType>
generate_pure_frb(SizeType nchans,
                  SizeType nsamps,
                  float f_min,
                  float f_max,
                  SizeType dt,
                  float pulse_toa,
                  float amplitude = 1.0F);

} // namespace dmt::utils