// make_boxcar_plans() noise tables and warm-up against a per-row trace of
// each row's FDMT subtree (the original construction), bit for bit.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_variance.hpp"

#include "test_sized.hpp"

using namespace dmt;
using namespace dmt::algorithms::detail;

namespace {

struct RowLeaves {
    std::vector<std::pair<SizeType, SizeType>> length_counts;
    uint64_t support = 0;
};

/// The row's leaves by walking its subtree.
RowLeaves trace_row(const plans::FDMTPlan& plan,
                    bool smear,
                    std::span<const uint8_t> kill_mask,
                    SizeType level,
                    SizeType node) {
    const auto& pc = plan.get_container();
    std::vector<SizeType> lengths;
    RowLeaves out;
    const auto trace = [&](auto& self, SizeType l, SizeType idx,
                           uint64_t shift) -> void {
        const auto& c = pc.coordinates[l][idx];
        if (l == 0) {
            const auto dt  = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
            SizeType s     = static_cast<SizeType>(std::abs(dt));
            SizeType reach = s;
            if (!pc.level0_boxes.empty()) {
                const auto& box = pc.level0_boxes[idx];
                s               = static_cast<SizeType>(box.s);
                reach           = smear ? static_cast<SizeType>(box.lag) + s
                                        : static_cast<SizeType>(box.point);
            }
            if (kill_mask.empty() || kill_mask[c.i_sub] != 0) {
                lengths.push_back(smear ? s + 1 : 1);
                out.support = std::max(out.support, shift + reach);
            }
            return;
        }
        self(self, l - 1, c.tail_buf_offset / pc.state_shape[l - 1].nsamps,
             shift);
        if (c.i_coord_head != SIZE_MAX) {
            self(self, l - 1, c.head_buf_offset / pc.state_shape[l - 1].nsamps,
                 shift + c.delay);
        }
    };
    trace(trace, level, node, 0);
    std::ranges::sort(lengths);
    for (const auto len : lengths) {
        if (out.length_counts.empty() ||
            out.length_counts.back().first != len) {
            out.length_counts.emplace_back(len, 0);
        }
        ++out.length_counts.back().second;
    }
    out.support = std::max<uint64_t>(out.support, pc.level0_causal_delay);
    return out;
}

void check_plans(const plans::FDMTPlan& plan,
                 const BoxcarConfig& config,
                 bool smear,
                 std::span<const uint8_t> kill_mask) {
    constexpr uint64_t kExtra = 5;
    const auto planes         = algorithms::detail::make_boxcar_plans(
        plan, config, smear, kill_mask, kExtra);
    const bool packed = config.input_nbits != 32;
    const double unit =
        std::ldexp(1.0, -static_cast<int>(plan.get_box_fraction_bits()));
    SizeType rows = 0;
    for (const auto& p : planes) {
        const auto nw = p.widths.size();
        for (SizeType k = 0; k < p.coord_indices.size(); ++k, ++rows) {
            INFO("level " << p.level << " band " << p.subband_idx << " row "
                          << k);
            const auto row = trace_row(plan, smear, kill_mask, p.level,
                                       p.coord_offset + p.coord_indices[k]);
            REQUIRE(p.warmup_samples[k] == row.support + kExtra);
            for (SizeType f = 0; f < nw; ++f) {
                const SizeType width  = p.widths[f];
                const SizeType w_edge = config.use_trapezoid
                                            ? trapezoid_edge_width(p.widths[f])
                                            : 0;
                double variance       = 0;
                double mean_weight    = 0;
                for (const auto& [len, count] : row.length_counts) {
                    const double weight =
                        smear ? box_weight(len, plan.get_box_fraction_bits()) *
                                    unit
                              : 1.0;
                    const auto n = static_cast<double>(count);
                    variance +=
                        n * fdmt_trapezoid_variance(len, width, w_edge, weight);
                    mean_weight += n * static_cast<double>(len) * weight;
                }
                const auto at = (k * nw) + f;
                if (variance == 0) {
                    REQUIRE(p.inv_sigma_lut[at] == 0);
                    REQUIRE(p.raw_thresh_lut[at] ==
                            std::numeric_limits<float>::max());
                    continue;
                }
                const double sigma = config.noise_sigma * std::sqrt(variance);
                REQUIRE(
                    std::bit_cast<uint32_t>(p.inv_sigma_lut[at]) ==
                    std::bit_cast<uint32_t>(static_cast<float>(1.0 / sigma)));
                REQUIRE(std::bit_cast<uint32_t>(p.raw_thresh_lut[at]) ==
                        std::bit_cast<uint32_t>(
                            static_cast<float>(config.detect_thresh * sigma)));
                if (packed) {
                    const double scale = smear ? 1.0 / unit : 1.0;
                    const double filter_weight =
                        w_edge == 0 ? static_cast<double>(width)
                                    : static_cast<double>(2 * (width - w_edge));
                    const double mean =
                        config.input_mean * mean_weight * filter_weight * scale;
                    REQUIRE(std::bit_cast<uint64_t>(p.integer_mean_lut[at]) ==
                            std::bit_cast<uint64_t>(mean));
                    REQUIRE(p.integer_thresh_lut[at] ==
                            static_cast<uint64_t>(std::ceil(
                                mean + config.detect_thresh * sigma * scale)));
                }
            }
        }
    }
    REQUIRE(rows > 0);
}

std::vector<std::vector<uint8_t>> trace_masks(SizeType nchans) {
    std::vector<std::vector<uint8_t>> masks(1); // none
    std::mt19937 gen(7);
    std::bernoulli_distribution kill(0.3);
    auto& random = masks.emplace_back(nchans);
    for (auto& v : random) {
        v = kill(gen) ? 0 : 1;
    }
    // Lower and upper halves killed (whole tail / head subtrees), a single
    // live channel, everything killed.
    masks.emplace_back(nchans, 1);
    std::fill_n(masks.back().begin(), nchans / 2, 0);
    masks.emplace_back(nchans, 1);
    std::fill(masks.back().begin() + static_cast<std::ptrdiff_t>(nchans / 2),
              masks.back().end(), 0);
    masks.emplace_back(nchans, 0);
    masks.back()[nchans / 3] = 1;
    masks.emplace_back(nchans, 0);
    return masks;
}

} // namespace

DMT_SIZED_TEST_CASE("make_boxcar_plans equals a per-row trace of the FDMT tree",
                    "[boxcar][plans][cpu]") {
    struct Case {
        SizeType nchans;
        SizeType depth;
        double chan_dm;
        float sub_loss;
    };
    const std::vector<IndexType> dts = {-73, -12, -5, 0, 4, 9, 30, 60, 97};
    const std::vector<Case> cases =
        full
            ? std::vector<Case>{Case{7, 2, 0.0, 0.0F},  Case{7, 2, 20.0, 0.0F},
                                Case{7, 2, 0.0, 0.3F},  Case{32, 0, 0.0, 0.0F},
                                Case{45, 3, 0.0, 0.0F}, Case{45, 3, 20.0, 0.0F},
                                Case{45, 3, 0.0, 0.3F}}
            : std::vector<Case>{Case{7, 2, 0.0, 0.0F}, Case{7, 2, 20.0, 0.0F},
                                Case{7, 2, 0.0, 0.3F}, Case{32, 0, 0.0, 0.0F},
                                Case{45, 3, 20.0, 0.0F}};
    for (const auto& c : cases) {
        const plans::FDMTPlan plan(700, 1200, c.nchans, 311, 0.001F, dts,
                                   "valid", c.depth, c.chan_dm, c.sub_loss);
        const auto masks = trace_masks(c.nchans);
        for (const bool smear : {true, false}) {
            for (const bool trap : {false, true}) {
                for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
                    DYNAMIC_SECTION("nchans "
                                    << c.nchans << " depth " << c.depth
                                    << " chan_dm " << c.chan_dm << " loss "
                                    << c.sub_loss << " smear " << smear
                                    << " trap " << trap << " nbits " << nbits) {
                        BoxcarConfig cfg;
                        cfg.max_boxcar    = 37;
                        cfg.wtsp          = 1.5F;
                        cfg.detect_thresh = 2.7F;
                        cfg.use_trapezoid = trap;
                        cfg.input_nbits   = nbits;
                        cfg.input_mean    = nbits == 8 ? 3.0F : 0.0F;
                        for (SizeType m = 0; m < masks.size(); ++m) {
                            INFO("mask " << m);
                            check_plans(plan, cfg, smear, masks[m]);
                        }
                    }
                }
            }
        }
    }
}
