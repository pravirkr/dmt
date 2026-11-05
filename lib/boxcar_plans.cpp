#include "dmt/common/boxcar_plans.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::plans {

namespace {
// Keep the integer load opaque: -ffast-math may otherwise fold even
// bitwise NaN guards when it infers that floating arguments are finite.
bool finite(float x) {
    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
    return (bits & 0x7F800000U) != 0x7F800000U;
}
} // namespace

std::vector<uint32_t> generate_box_width_trials(SizeType max_width,
                                                float wtsp) {
    if (max_width == 0 || max_width > 4096 || !finite(wtsp) || wtsp < 1.0F) {
        throw std::invalid_argument(
            "Boxcar: max_width must be 1..4096 and wtsp finite >= 1");
    }
    std::vector<uint32_t> widths;
    for (SizeType w = 1; w <= max_width;) {
        widths.push_back(static_cast<uint32_t>(w));
        const double next = static_cast<double>(w) * wtsp;
        if (next > static_cast<double>(max_width)) {
            break;
        }
        w = std::max(w + 1, static_cast<SizeType>(std::ceil(next)));
    }
    return widths;
}

} // namespace dmt::plans

namespace dmt::algorithms::detail {

std::vector<plans::BoxcarPlanePlan>
make_boxcar_plans(const plans::FDMTPlan& plan,
                  const BoxcarConfig& config,
                  bool smear,
                  std::span<const uint8_t> kill_mask,
                  uint64_t extra_warmup) {
    const auto finite = [](float x) {
        const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
        return (bits & 0x7F800000U) != 0x7F800000U;
    };
    if (plan.get_mode() != "valid" || !finite(config.detect_thresh) ||
        config.detect_thresh <= 0 || !finite(config.noise_sigma) ||
        config.noise_sigma <= 0 || config.max_candidates == 0 ||
        static_cast<unsigned>(config.peak_mode) >
            static_cast<unsigned>(PeakFindingMode::kAllAboveThreshold) ||
        plan.get_nchans() > UINT32_MAX ||
        (!kill_mask.empty() && kill_mask.size() != plan.get_nchans())) {
        throw std::invalid_argument(
            "FDMTBoxcar: invalid configuration (requires valid mode, positive "
            "finite threshold/sigma and capacity, kill mask of nchans "
            "entries)");
    }
    const bool packed = config.input_nbits != 32;
    if ((!packed && config.input_mean != 0) || !finite(config.input_mean) ||
        (packed && config.input_nbits != 1 && config.input_nbits != 2 &&
         config.input_nbits != 4 && config.input_nbits != 8 &&
         config.input_nbits != 16) ||
        config.input_mean < 0 ||
        (packed && config.input_mean > ((1U << config.input_nbits) - 1))) {
        throw std::invalid_argument(
            "FDMTBoxcar: input_nbits must be 32 or 1/2/4/8/16; packed mean "
            "must be finite and within code range; float mean must be zero");
    }
    // Bound every intermediate and the rebased boxcar prefix, not just an
    // emitted window. The common dyadic scale is exact; there is no extra
    // level-0 mean rounding in the search path.
    if (packed) {
        const uint64_t maxval = (uint64_t{1} << config.input_nbits) - 1;
        uint64_t leaf         = 1;
        if (smear) {
            for (SizeType len = 1; len <= plan.get_box_weights().size();
                 ++len) {
                leaf = std::max(
                    leaf, static_cast<uint64_t>(len) *
                              box_weight(len, plan.get_box_fraction_bits()));
            }
        }
        if (leaf > UINT64_MAX / maxval / plan.get_nchans() / 8192) {
            throw std::invalid_argument(
                "FDMTBoxcar: dyadic integer prefix would overflow uint64");
        }
    }
    const auto widths =
        plans::generate_box_width_trials(config.max_boxcar, config.wtsp);
    const auto& pc       = plan.get_container();
    const SizeType root  = plan.get_niters();
    const SizeType depth = plan.get_subband_depth();
    const auto root_dm   = plan.get_dm_grid_final();
    const double unit =
        std::ldexp(1.0, -static_cast<int>(plan.get_box_fraction_bits()));
    std::vector<plans::BoxcarPlanePlan> planes;
    // Every subband of the searched levels: the tree subbands and, below
    // the root, the staggered ones that straddle their boundaries.
    for (SizeType l = root - depth; l <= root; ++l) {
        for (SizeType band = 0; band < pc.grids[l].size(); ++band) {
            const auto& grid = pc.grids[l][band];
            const auto begin = grid.chan_begin;
            const auto end   = grid.chan_end;
            plans::BoxcarPlanePlan out;
            out.level         = l;
            out.subband_level = static_cast<uint32_t>(root - l);
            out.subband_idx   = static_cast<uint32_t>(band);
            out.staggered     = grid.staggered;
            out.chan_begin    = static_cast<uint32_t>(begin);
            out.chan_end      = static_cast<uint32_t>(end);
            out.f_low_mhz =
                plan.get_f_min() + static_cast<float>(begin) * plan.get_df();
            out.f_high_mhz =
                plan.get_f_min() + static_cast<float>(end) * plan.get_df();
            out.coord_offset = grid.coord_offset;
            out.widths       = widths;
            if (grid.ndt > UINT32_MAX || grid.ndt > SIZE_MAX / widths.size()) {
                throw std::invalid_argument(
                    "FDMTBoxcar: trial table is too large");
            }
            out.dm_grid.resize(grid.ndt);
            out.warmup_samples.resize(grid.ndt);
            out.inv_sigma_lut.resize(grid.ndt * widths.size());
            out.raw_thresh_lut.resize(grid.ndt * widths.size());
            if (packed) {
                out.integer_mean_lut.resize(grid.ndt * widths.size());
                out.integer_thresh_lut.resize(grid.ndt * widths.size());
            }
            // Trace each actual intermediate coordinate, including copy
            // nodes and plan-specific pruning. Shift does not change energy
            // between independent channels, but does determine cold support.
            // Leaf lengths of a row, as a histogram: a row has one leaf per
            // channel but only a few distinct box lengths, and the noise
            // terms below depend on the length alone.
            std::vector<SizeType> lengths;
            std::vector<std::pair<SizeType, SizeType>> length_counts;
            lengths.reserve(end - begin);
            for (SizeType d = 0; d < grid.ndt; ++d) {
                lengths.clear();
                uint64_t support = 0;
                const auto trace = [&](auto& self, SizeType level, SizeType idx,
                                       uint64_t shift) -> void {
                    const auto& c = pc.coordinates[level][idx];
                    if (level == 0) {
                        const auto dt    = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
                        const SizeType s = static_cast<SizeType>(std::abs(dt));
                        // A killed channel adds no noise or signal (it reads as
                        // zero).
                        if (kill_mask.empty() || kill_mask[c.i_sub] != 0) {
                            lengths.push_back(smear ? s + 1 : 1);
                            support = std::max(support, shift + s);
                        }
                    } else {
                        self(self, level - 1,
                             c.tail_buf_offset /
                                 pc.state_shape[level - 1].nsamps,
                             shift);
                        if (c.i_coord_head != SIZE_MAX) {
                            self(self, level - 1,
                                 c.head_buf_offset /
                                     pc.state_shape[level - 1].nsamps,
                                 shift + c.delay);
                        }
                    }
                };
                trace(trace, l, grid.coord_offset + d, 0);
                std::ranges::sort(lengths);
                length_counts.clear();
                for (const auto len : lengths) {
                    if (length_counts.empty() ||
                        length_counts.back().first != len) {
                        length_counts.emplace_back(len, 0);
                    }
                    ++length_counts.back().second;
                }
                out.warmup_samples[d] = support + extra_warmup;
                const double f0       = out.f_low_mhz;
                const double f1       = out.f_high_mhz;
                const double delay_per_dm =
                    kDispConst * (1.0 / (f0 * f0) - 1.0 / (f1 * f1));
                out.dm_grid[d] = static_cast<float>(
                    grid.dt_grid[d] * plan.get_tsamp() / delay_per_dm);
                if (l == root) {
                    // Preserve the exact DM values of custom physical grids.
                    out.dm_grid[d] = root_dm[d];
                }
                for (SizeType f = 0; f < widths.size(); ++f) {
                    const SizeType width = widths[f];
                    const SizeType w_edge =
                        config.use_trapezoid ? trapezoid_edge_width(widths[f])
                                             : 0;
                    double variance    = 0;
                    double mean_weight = 0;
                    for (const auto& [len, count] : length_counts) {
                        const double weight =
                            smear ? box_weight(len,
                                               plan.get_box_fraction_bits()) *
                                        unit
                                  : 1.0;
                        const auto n = static_cast<double>(count);
                        variance += n * fdmt_trapezoid_variance(len, width,
                                                                w_edge, weight);
                        mean_weight += n * static_cast<double>(len) * weight;
                    }
                    if (variance == 0) {
                        // Every channel of the row is killed: never detect.
                        out.inv_sigma_lut[d * widths.size() + f] = 0;
                        out.raw_thresh_lut[d * widths.size() + f] =
                            std::numeric_limits<float>::max();
                        if (packed) {
                            out.integer_mean_lut[d * widths.size() + f] = 0;
                            out.integer_thresh_lut[d * widths.size() + f] =
                                UINT64_MAX;
                        }
                        continue;
                    }
                    const double sigma =
                        config.noise_sigma * std::sqrt(variance);
                    const float inv_sigma = static_cast<float>(1.0 / sigma);
                    const float threshold =
                        static_cast<float>(config.detect_thresh * sigma);
                    if (!finite(inv_sigma) || !finite(threshold) ||
                        inv_sigma <= 0 || threshold <= 0) {
                        throw std::invalid_argument(
                            "FDMTBoxcar: sigma/threshold is outside float32 "
                            "range");
                    }
                    out.inv_sigma_lut[d * widths.size() + f]  = inv_sigma;
                    out.raw_thresh_lut[d * widths.size() + f] = threshold;
                    if (packed) {
                        const double scale = smear ? 1.0 / unit : 1.0;
                        const double filter_weight =
                            w_edge == 0
                                ? static_cast<double>(width)
                                : static_cast<double>(2 * (width - w_edge));
                        const double mean   = config.input_mean * mean_weight *
                                              filter_weight * scale;
                        const double cutoff = std::ceil(
                            mean + config.detect_thresh * sigma * scale);
                        if (cutoff >= std::ldexp(1.0, 64)) {
                            throw std::invalid_argument(
                                "FDMTBoxcar: integer threshold exceeds uint64");
                        }
                        out.integer_mean_lut[d * widths.size() + f] = mean;
                        out.integer_thresh_lut[d * widths.size() + f] =
                            static_cast<uint64_t>(cutoff);
                    }
                }
            }
            planes.push_back(std::move(out));
        }
    }
    return planes;
}

std::vector<BoxcarRow>
make_boxcar_rows(const plans::FDMTPlan& plan,
                 std::span<const plans::BoxcarPlanePlan> planes) {
    std::vector<BoxcarRow> rows;
    const auto& pc      = plan.get_container();
    const SizeType root = plan.get_niters();
    std::vector<std::vector<SizeType>> level_history(root + 1);
    for (SizeType l = 1; l <= root; ++l) {
        level_history[l].resize(pc.coordinates[l].size(), 0);
        for (const auto& c : pc.coordinates_sum[l]) {
            level_history[l][c.buf_offset / pc.state_shape[l].nsamps] =
                c.hist_offset;
        }
    }
    for (const auto& plane : planes) {
        const bool unmat = (plane.level == root) || plane.staggered;
        for (SizeType d = 0; d < plane.dm_grid.size(); ++d) {
            BoxcarRow row;
            row.row                     = plane.coord_offset + d;
            row.warmup                  = plane.warmup_samples[d];
            row.unmaterialized          = unmat;
            row.candidate.dm            = plane.dm_grid[d];
            row.candidate.dm_idx        = static_cast<uint32_t>(d);
            row.candidate.subband_level = plane.subband_level;
            row.candidate.subband_idx   = plane.subband_idx;
            row.candidate.staggered     = plane.staggered ? 1U : 0U;
            row.candidate.chan_begin    = plane.chan_begin;
            row.candidate.chan_end      = plane.chan_end;
            row.candidate.f_low_mhz     = plane.f_low_mhz;
            row.candidate.f_high_mhz    = plane.f_high_mhz;
            if (unmat) {
                const SizeType lvl  = plane.level;
                const SizeType prev = pc.state_shape[lvl - 1].nsamps;
                const auto& c       = pc.coordinates[lvl][row.row];
                row.tail            = c.tail_buf_offset / prev;
                row.head            = c.i_coord_head == SIZE_MAX
                                          ? SIZE_MAX
                                          : c.head_buf_offset / prev;
                row.delay           = row.head == SIZE_MAX ? 0 : c.delay;
                // Only coordinates_sum stores the allocated history slot.
                if (row.head != SIZE_MAX) {
                    row.hist = level_history[lvl][row.row];
                }
            }
            rows.push_back(row);
        }
    }
    return rows;
}

} // namespace dmt::algorithms::detail
