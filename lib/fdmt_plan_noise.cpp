#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_variance.hpp"

#include "fdmt_plan_impl.hpp"

// FDMTPlan: the noise model (variance of every row, box weights).

namespace dmt::plans {

float FDMTPlan::Impl::variance_from_smearing_row(
    const float* smearing_row,
    float w,
    std::span<const uint8_t> kill_mask) const noexcept {
    // Each channel's level-0 box of L samples carries the box-mean
    // weight round(2^b / L) * 2^-b (see get_box_fraction_bits()).
    // Killed channels read as zero and add nothing.
    const auto bits  = get_box_fraction_bits();
    const double inv = std::ldexp(1.0, -static_cast<int>(bits));
    double total_var = 0.0;
    for (SizeType c = 0; c < m_nchans; ++c) {
        if (!is_active(kill_mask, c)) {
            continue;
        }
        const auto box_len = static_cast<SizeType>(smearing_row[c]) + 1;
        const double weight =
            static_cast<double>(algorithms::detail::box_weight(box_len, bits)) *
            inv;
        total_var += algorithms::detail::fdmt_box_variance(box_len, w, weight);
    }
    return static_cast<float>(total_var);
}

std::vector<uint32_t> FDMTPlan::Impl::get_box_weights() const {
    const auto bits    = get_box_fraction_bits();
    const auto box_max = m_container.state_shape[0].dt_max + 1;
    std::vector<uint32_t> weights(box_max);
    for (SizeType len = 1; len <= box_max; ++len) {
        weights[len - 1] = algorithms::detail::box_weight(len, bits);
    }
    return weights;
}

void FDMTPlan::Impl::check_kill_mask(std::span<const uint8_t> kill_mask) const {
    if (!kill_mask.empty() && kill_mask.size() != m_nchans) {
        throw std::invalid_argument(
            fmt::format("FDMTPlan: kill_mask size={} must be 0 or "
                        "nchans={}",
                        kill_mask.size(), m_nchans));
    }
}

SizeType FDMTPlan::Impl::count_active(std::span<const uint8_t> kill_mask,
                                      SizeType begin,
                                      SizeType end) noexcept {
    SizeType n = 0;
    for (SizeType c = begin; c < end; ++c) {
        n += is_active(kill_mask, c) ? 1 : 0;
    }
    return n;
}

float FDMTPlan::Impl::get_effective_variance(
    SizeType dm_idx,
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    if (boxcar_width == 0) {
        throw std::invalid_argument("boxcar_width must be greater than 0");
    }
    const auto ndms = get_dmt_ndms();
    if (dm_idx >= ndms) {
        throw std::out_of_range("dm_idx " + std::to_string(dm_idx) +
                                " out of range [0, " + std::to_string(ndms) +
                                ")");
    }
    check_kill_mask(kill_mask);
    const float w = static_cast<float>(boxcar_width);
    if (!use_box_smearing) {
        return static_cast<float>(count_active(kill_mask, 0, m_nchans)) * w;
    }
    const auto smearing_grid = get_smearing_grid_final();
    return variance_from_smearing_row(&smearing_grid[dm_idx * m_nchans], w,
                                      kill_mask);
}

float FDMTPlan::Impl::get_effective_sigma(
    SizeType dm_idx,
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    return std::sqrt(get_effective_variance(dm_idx, boxcar_width,
                                            use_box_smearing, kill_mask));
}

std::vector<float> FDMTPlan::Impl::get_effective_variance_grid(
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    if (boxcar_width == 0) {
        throw std::invalid_argument("boxcar_width must be greater than 0");
    }
    check_kill_mask(kill_mask);
    const auto ndms = get_dmt_ndms();
    std::vector<float> grid(ndms, 0.0F);
    const auto w = static_cast<float>(boxcar_width);
    if (!use_box_smearing) {
        std::ranges::fill(
            grid, static_cast<float>(count_active(kill_mask, 0, m_nchans)) * w);
        return grid;
    }
    // Compute the O(ndms * nchans) smearing-grid tree trace exactly
    // once and reuse it for every DM index below. Calling
    // get_effective_variance() in this loop instead would recompute
    // get_smearing_grid_final() from scratch on every iteration, making
    // this O(ndms^2 * nchans) for no reason.
    const auto smearing_grid = get_smearing_grid_final();
    for (SizeType i = 0; i < ndms; ++i) {
        grid[i] = variance_from_smearing_row(&smearing_grid[i * m_nchans], w,
                                             kill_mask);
    }
    return grid;
}

std::vector<float> FDMTPlan::Impl::get_effective_sigma_grid(
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    auto grid =
        get_effective_variance_grid(boxcar_width, use_box_smearing, kill_mask);
    for (auto& v : grid) {
        v = std::sqrt(v);
    }
    return grid;
}

std::vector<float> FDMTPlan::Impl::get_subband_variance_grid(
    SizeType level,
    SizeType subband,
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    if (boxcar_width == 0) {
        throw std::invalid_argument("boxcar_width must be greater than 0");
    }
    check_kill_mask(kill_mask);
    if (level > m_niters) {
        throw std::out_of_range(fmt::format(
            "FDMTPlan: level {} out of range [0, {}]", level, m_niters));
    }
    const auto& grids = m_container.grids[level];
    if (subband >= grids.size()) {
        throw std::out_of_range(
            fmt::format("FDMTPlan: subband {} out of range [0, {}) at "
                        "level {}",
                        subband, grids.size(), level));
    }
    const auto& g = grids[subband];
    const auto w  = static_cast<float>(boxcar_width);
    std::vector<float> out(g.ndt, 0.0F);
    if (!use_box_smearing) {
        std::ranges::fill(out, static_cast<float>(count_active(
                                   kill_mask, g.chan_begin, g.chan_end)) *
                                   w);
        return out;
    }
    // The level-0 box of every channel a row reaches (all of the
    // subband's channels, once each), as in get_smearing_grid_final().
    std::vector<float> smear(m_nchans, 0.0F);
    const auto trace = [&](auto& self, SizeType l, SizeType idx) -> void {
        const auto& c = m_container.coordinates[l][idx];
        if (l == 0) {
            smear[c.i_sub] = std::abs(static_cast<float>(
                m_container.grids[0][c.i_sub].dt_grid[c.i_dt]));
            return;
        }
        self(self, l - 1, c.i_coord_tail);
        if (c.i_coord_head != SIZE_MAX) {
            self(self, l - 1, c.i_coord_head);
        }
    };
    const auto bits  = get_box_fraction_bits();
    const double inv = std::ldexp(1.0, -static_cast<int>(bits));
    for (SizeType d = 0; d < g.ndt; ++d) {
        if (level == 0) {
            smear[subband] = std::abs(static_cast<float>(g.dt_grid[d]));
        } else {
            trace(trace, level, g.coord_offset + d);
        }
        double total = 0.0;
        for (SizeType ch = g.chan_begin; ch < g.chan_end; ++ch) {
            if (!is_active(kill_mask, ch)) {
                continue;
            }
            const auto box_len = static_cast<SizeType>(smear[ch]) + 1;
            const double weight =
                static_cast<double>(
                    algorithms::detail::box_weight(box_len, bits)) *
                inv;
            total += algorithms::detail::fdmt_box_variance(box_len, w, weight);
        }
        out[d] = static_cast<float>(total);
    }
    return out;
}

} // namespace dmt::plans
