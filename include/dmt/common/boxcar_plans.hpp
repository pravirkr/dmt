#pragma once

#include <cstdint>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::plans {

/// Widths begin at 1; each next width is max(w + 1, ceil(w * wtsp)) while it
/// does not exceed max_width, and max_width ends the bank when the steps stop
/// below it (9 at wtsp 2: 1, 2, 4, 8, 9).
[[nodiscard]] std::vector<uint32_t>
generate_box_width_trials(SizeType max_width, float wtsp);

/// Read-only through FDMTBoxcar::get_search_plans(); local DM-major tables.
struct BoxcarPlanePlan {
    SizeType level         = 0; ///< Tree level (0 is channel initialization).
    uint32_t subband_level = 0; ///< Remaining merges; 0 is full band.
    uint32_t subband_idx   = 0; ///< Tree subbands, then staggered ones.
    bool staggered         = false;
    uint32_t chan_begin    = 0;
    uint32_t chan_end      = 0;
    float f_low_mhz        = 0.0F;
    float f_high_mhz       = 0.0F;
    SizeType coord_offset  = 0;
    std::vector<uint32_t> widths;
    std::vector<float> dm_grid;
    std::vector<uint64_t> warmup_samples; ///< FDMT support before width-1.
    std::vector<float> inv_sigma_lut;
    std::vector<float> raw_thresh_lut;
    std::vector<double> integer_mean_lut; ///< Baseline in dyadic integer units.
    std::vector<uint64_t> integer_thresh_lut; ///< Inclusive integer cutoffs.
};

} // namespace dmt::plans
