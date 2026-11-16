#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/dm_grids.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/dm_utils.hpp"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"

#include "fdmt_plan_impl.hpp"

// FDMTPlan: the subband tree, its DM grids and merge plan.

namespace dmt::plans {

namespace {

/// Delay trials (samples across [f_start, f_end]) of generate_dm_grid() on
/// a subband of nsub channels, over the delays [dt_lo, dt_hi] (both kept):
/// one-sample pulses at the given loss. DMs convert to delays as
/// BoxcarPlanePlan reports them (kDispConst).
std::vector<IndexType> subband_dt_grid(float f_start,
                                       float f_end,
                                       SizeType nsub,
                                       float tsamp,
                                       double chan_dm,
                                       float loss,
                                       IndexType dt_lo,
                                       IndexType dt_hi) {
    const double f0     = f_start;
    const double f1     = f_end;
    const double per_dm = static_cast<double>(kDispConst) *
                          ((1.0 / (f0 * f0)) - (1.0 / (f1 * f1))) /
                          static_cast<double>(tsamp);
    DMGridConfig c;
    c.f_min        = f0;
    c.f_max        = f1;
    c.nchans       = nsub;
    c.tsamp        = tsamp;
    c.pulse_width  = tsamp;
    c.chan_dm      = chan_dm;
    c.max_snr_loss = loss;
    const auto dms = generate_dm_grid(c, static_cast<double>(dt_lo) / per_dm,
                                      static_cast<double>(dt_hi) / per_dm);
    std::vector<IndexType> dts{dt_lo, dt_hi};
    dts.reserve(dms.size() + 2);
    for (const double dm : dms) {
        const auto dt = static_cast<IndexType>(std::nearbyint(dm * per_dm));
        dts.push_back(std::clamp(dt, dt_lo, dt_hi));
    }
    std::ranges::sort(dts);
    const auto [first, last] = std::ranges::unique(dts);
    dts.erase(first, last);
    return dts;
}

} // namespace

FDMTPlan::Impl::Impl(float f_min,
                     float f_max,
                     SizeType nchans,
                     SizeType nsamps,
                     float tsamp,
                     IndexType dt_max,
                     IndexType dt_min,
                     SizeType dt_step,
                     std::string_view mode,
                     SizeType subband_depth,
                     double chan_dm,
                     float subband_max_snr_loss)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dt_max(dt_max),
      m_dt_min(dt_min),
      m_dt_step(dt_step),
      m_mode(parse_fdmt_mode(mode)),
      m_subband_depth(subband_depth),
      m_chan_dm(chan_dm),
      m_subband_max_snr_loss(subband_max_snr_loss) {
    validate_inputs();
    configure_plan();
}

FDMTPlan::Impl::Impl(float f_min,
                     float f_max,
                     SizeType nchans,
                     SizeType nsamps,
                     float tsamp,
                     const std::vector<IndexType>& dt_grid,
                     std::string_view mode,
                     SizeType subband_depth,
                     double chan_dm,
                     float subband_max_snr_loss)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dt_max(0),
      m_dt_min(0),
      m_dt_step(0),
      m_mode(parse_fdmt_mode(mode)),
      m_subband_depth(subband_depth),
      m_chan_dm(chan_dm),
      m_subband_max_snr_loss(subband_max_snr_loss),
      m_is_custom_grid(true),
      m_dt_grid_target(dt_grid) {
    if (m_dt_grid_target.empty()) {
        throw std::invalid_argument("FDMT: dt_grid must not be empty");
    }
    std::ranges::sort(m_dt_grid_target);
    const auto [first, last] = std::ranges::unique(m_dt_grid_target);
    m_dt_grid_target.erase(first, last);

    m_dt_min = m_dt_grid_target.front();
    m_dt_max = m_dt_grid_target.back();
    validate_inputs();
    configure_plan();
}

FDMTPlan::Impl::Impl(float f_min,
                     float f_max,
                     SizeType nchans,
                     SizeType nsamps,
                     float tsamp,
                     const std::vector<float>& dm_grid,
                     std::string_view mode,
                     SizeType subband_depth,
                     double chan_dm,
                     float subband_max_snr_loss)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dt_max(0),
      m_dt_min(0),
      m_dt_step(0),
      m_mode(parse_fdmt_mode(mode)),
      m_subband_depth(subband_depth),
      m_chan_dm(chan_dm),
      m_subband_max_snr_loss(subband_max_snr_loss),
      m_is_custom_grid(true) {
    if (dm_grid.empty()) {
        throw std::invalid_argument("FDMT: dm_grid must not be empty");
    }
    for (const auto dm : dm_grid) {
        if (!utils::is_finite_bits(dm)) {
            throw std::invalid_argument("FDMT: dm_grid values must be finite");
        }
    }
    validate_band();
    auto dm_sorted = dm_grid;
    std::ranges::sort(dm_sorted);
    const auto [dm_first, dm_last] = std::ranges::unique(dm_sorted);
    dm_sorted.erase(dm_first, dm_last);

    const float dm_conv = utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    m_dt_grid_target.reserve(dm_sorted.size());
    for (const auto dm : dm_sorted) {
        const double scaled = std::nearbyint(static_cast<double>(dm) /
                                             static_cast<double>(dm_conv));
        if (!utils::is_finite_bits(scaled) ||
            scaled <
                static_cast<double>(std::numeric_limits<std::int32_t>::min()) ||
            scaled >
                static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument(
                "FDMT: dm_grid delay trial exceeds representable range");
        }
        m_dt_grid_target.push_back(static_cast<IndexType>(scaled));
    }
    std::ranges::sort(m_dt_grid_target);
    const auto [first, last] = std::ranges::unique(m_dt_grid_target);
    m_dt_grid_target.erase(first, last);

    m_dt_min = m_dt_grid_target.front();
    m_dt_max = m_dt_grid_target.back();
    validate_inputs();
    configure_plan();
}

void FDMTPlan::Impl::validate_band() const {
    if (!utils::is_finite_bits(m_f_min) || !utils::is_finite_bits(m_f_max) ||
        !utils::is_finite_bits(m_tsamp)) {
        throw std::invalid_argument(
            "FDMT: f_min, f_max and tsamp must be finite");
    }
    if (!utils::is_finite_bits(m_chan_dm)) {
        throw std::invalid_argument("FDMT: chan_dm must be finite");
    }
    if (!utils::is_finite_bits(m_subband_max_snr_loss) ||
        !(m_subband_max_snr_loss >= 0.0F && m_subband_max_snr_loss < 1.0F)) {
        throw std::invalid_argument(
            fmt::format("FDMT: subband_max_snr_loss must be in [0, 1), got {}",
                        m_subband_max_snr_loss));
    }
    if (!(m_f_min > 0.0F)) {
        throw std::invalid_argument(
            fmt::format("FDMT: f_min={} must be positive", m_f_min));
    }
    if (m_f_min >= m_f_max) {
        throw std::invalid_argument(fmt::format(
            "FDMT: f_min={} must be less than f_max={}", m_f_min, m_f_max));
    }
    if (m_nchans < 2) {
        throw std::invalid_argument(
            fmt::format("FDMT: nchans={} must be at least 2", m_nchans));
    }
    if (m_nsamps == 0) {
        throw std::invalid_argument(
            fmt::format("FDMT: nsamps={} must be greater than 0", m_nsamps));
    }
    if (m_tsamp <= 0) {
        throw std::invalid_argument(
            fmt::format("FDMT: tsamp={} must be greater than 0", m_tsamp));
    }
}

void FDMTPlan::Impl::validate_inputs() const {
    validate_band();
    if (m_dt_min < std::numeric_limits<std::int32_t>::min() ||
        m_dt_max > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(
            "FDMT: delay trial exceeds representable range");
    }
    if (m_is_custom_grid) {
        if (m_dt_grid_target.empty()) {
            throw std::invalid_argument("FDMT: dt_grid must not be empty");
        }
        bool all_zero = true;
        for (const auto dt : m_dt_grid_target) {
            if (dt != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            throw std::invalid_argument("FDMT: dt_grid must contain at "
                                        "least one non-zero delay trial");
        }
    } else {
        if (m_dt_min == 0 && m_dt_max == 0) {
            throw std::invalid_argument(
                "FDMT: dt_max and dt_min cannot both be 0");
        }
        if (m_dt_min >= m_dt_max) {
            throw std::invalid_argument(
                fmt::format("FDMT: dt_min={} must be less than dt_max={}",
                            m_dt_min, m_dt_max));
        }
        if (m_dt_step == 0) {
            throw std::invalid_argument(fmt::format(
                "FDMT: dt_step={} must be greater than 0", m_dt_step));
        }
        if (std::cmp_greater(m_dt_step, m_dt_max - m_dt_min)) {
            throw std::invalid_argument(
                fmt::format("FDMT: dt_step={} cannot be greater than "
                            "(dt_max - dt_min)={}",
                            m_dt_step, m_dt_max - m_dt_min));
        }
    }
}

std::vector<std::vector<FDMTPlan::Impl::TreeNode>>
FDMTPlan::Impl::build_tree() const {
    std::vector<std::vector<TreeNode>> tree(m_niters + 1);
    tree[m_niters].push_back({0, m_nchans, SIZE_MAX, SIZE_MAX, false});
    for (SizeType l = m_niters; l >= 1; --l) {
        auto& cur  = tree[l];
        auto& prev = tree[l - 1];
        for (auto& node : cur) {
            const auto width = node.chan_end - node.chan_begin;
            node.tail        = prev.size();
            if (width >= 2) {
                const auto mid = node.chan_begin + ((width + 1) / 2);
                node.head      = prev.size() + 1;
                prev.push_back(
                    {node.chan_begin, mid, SIZE_MAX, SIZE_MAX, false});
                prev.push_back({mid, node.chan_end, SIZE_MAX, SIZE_MAX, false});
            } else {
                node.head = SIZE_MAX;
                prev.push_back({node.chan_begin, node.chan_end, SIZE_MAX,
                                SIZE_MAX, false});
            }
        }
        const auto depth = m_niters - l;
        if (depth >= 1 && depth <= m_subband_depth) {
            const auto nregular = cur.size();
            for (SizeType i = 0; i + 1 < nregular; ++i) {
                const auto tail =
                    cur[i].head == SIZE_MAX ? cur[i].tail : cur[i].head;
                const auto head = cur[i + 1].tail;
                cur.push_back({prev[tail].chan_begin, prev[head].chan_end, tail,
                               head, true});
            }
        }
    }
    return tree;
}

std::array<float, 3>
FDMTPlan::Impl::node_freqs(const TreeNode& node,
                           const std::vector<TreeNode>& prev) const {
    const float f_start = chan_freq(node.chan_begin);
    const auto& tail    = prev[node.tail];
    const float f_mid =
        f_start + (m_df * static_cast<float>(tail.chan_end - tail.chan_begin));
    const float f_end =
        f_start + (m_df * static_cast<float>(node.chan_end - node.chan_begin));
    return {f_start, f_mid, f_end};
}

std::vector<std::vector<std::vector<IndexType>>>
FDMTPlan::Impl::determine_active_grids(
    const std::vector<std::vector<TreeNode>>& tree) const {
    std::vector<std::vector<std::vector<IndexType>>> active_dts(m_niters + 1);
    for (SizeType l = 0; l <= m_niters; ++l) {
        active_dts[l].resize(tree[l].size());
    }

    // Target sparse grid at root level (m_niters, subband 0)
    auto& root = active_dts[m_niters][0];
    if (m_is_custom_grid) {
        root = m_dt_grid_target;
    } else {
        for (IndexType dt = m_dt_min; dt <= m_dt_max;
             dt += static_cast<IndexType>(m_dt_step)) {
            root.push_back(dt);
        }
    }
    // A staggered subband searches the root's DM trials scaled to its
    // own delay span.
    for (SizeType l = 1; l < m_niters; ++l) {
        for (SizeType i_sub = 0; i_sub < tree[l].size(); ++i_sub) {
            const auto& node = tree[l][i_sub];
            if (!node.staggered) {
                continue;
            }
            const float phi =
                utils::cff(chan_freq(node.chan_begin), chan_freq(node.chan_end),
                           m_f_min, m_f_max);
            auto& dts = active_dts[l][i_sub];
            if (m_subband_max_snr_loss > 0.0F && !root.empty()) {
                // Its own trials at the subband loss over the same span.
                const auto [lo, hi] = std::ranges::minmax(root);
                dts                 = subband_dt_grid(
                    chan_freq(node.chan_begin), chan_freq(node.chan_end),
                    node.chan_end - node.chan_begin, m_tsamp, m_chan_dm,
                    m_subband_max_snr_loss,
                    static_cast<IndexType>(
                        std::nearbyint(static_cast<float>(lo) * phi)),
                    static_cast<IndexType>(
                        std::nearbyint(static_cast<float>(hi) * phi)));
                continue;
            }
            for (const auto dt : root) {
                dts.push_back(static_cast<IndexType>(
                    std::nearbyint(static_cast<float>(dt) * phi)));
            }
            std::ranges::sort(dts);
            const auto [first, last] = std::ranges::unique(dts);
            dts.erase(first, last);
        }
    }

    // Top-down traversal from root to level 1
    for (SizeType l = m_niters; l >= 1; --l) {
        const auto& prev = tree[l - 1];
        for (SizeType i_sub = 0; i_sub < tree[l].size(); ++i_sub) {
            const auto& node    = tree[l][i_sub];
            const auto& cur_dts = active_dts[l][i_sub];
            if (cur_dts.empty()) {
                continue;
            }
            if (node.head == SIZE_MAX) {
                auto& tail_dts = active_dts[l - 1][node.tail];
                tail_dts.insert(tail_dts.end(), cur_dts.begin(), cur_dts.end());
                continue;
            }
            const auto [f_start, f_mid, f_end] = node_freqs(node, prev);
            const float tail_phi = utils::cff(f_start, f_mid, f_start, f_end);
            for (const auto dt : cur_dts) {
                const auto dt_tail = static_cast<IndexType>(
                    std::nearbyint(static_cast<float>(dt) * tail_phi));
                const auto dt_head = dt - dt_tail;
                active_dts[l - 1][node.tail].push_back(dt_tail);
                active_dts[l - 1][node.head].push_back(dt_head);
            }
        }

        // Deduplicate and sort active_dts[l - 1]
        for (auto& vec : active_dts[l - 1]) {
            std::ranges::sort(vec);
            const auto [first, last] = std::ranges::unique(vec);
            vec.erase(first, last);
        }
    }

    return active_dts;
}

void FDMTPlan::Impl::configure_plan() {
    m_df     = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
    m_niters = static_cast<SizeType>(std::bit_width(m_nchans - 1));
    if (m_subband_depth > 0 && m_subband_depth >= m_niters) {
        throw std::invalid_argument(
            fmt::format("FDMT: subband_depth={} must be less than niters={}",
                        m_subband_depth, m_niters));
    }
    m_container = FDMTPlanContainer(m_niters);

    // Step 1: the merge tree (channel extents and children per level)
    const auto tree = build_tree();

    // Step 2: Top-down reachability traversal to identify active (i_sub,
    // dt) nodes
    auto active_dts = determine_active_grids(tree);

    // Step 3: Populate grids and coordinates bottom-up using active sets
    make_plan_iter0(active_dts[0]);
    make_level0_boxes();
    for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
        make_plan(i_iter, tree, active_dts[i_iter]);
    }
    select_search_rows();

    // Step 4: Assign tree history offsets for valid-mode cross-block
    // streaming
    SizeType total_tree_hist = 0;
    for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
        for (auto& coord : m_container.coordinates_sum[i_iter]) {
            if (coord.delay > 0) {
                coord.hist_offset = total_tree_hist;
                total_tree_hist += coord.delay;
            } else {
                coord.hist_offset = 0;
            }
        }
    }
    m_container.tree_history_size = total_tree_hist;

    m_buffer_size = m_container.get_buffer_size();
    m_fft_support = compute_fft_support();
    m_fft_size    = compute_fft_size();
    logging::debug("FDMTPlan: df={}, dt_max={}, dt_min={}, dt_step={}, "
                   "niters={}, subband_depth={}",
                   m_df, m_dt_max, m_dt_min, m_dt_step, m_niters,
                   m_subband_depth);
}

void FDMTPlan::Impl::select_search_rows() {
    m_container.subband_max_snr_loss = m_subband_max_snr_loss;
    if (m_subband_max_snr_loss == 0.0F || m_subband_depth == 0) {
        return;
    }
    // Tree subbands below the root keep every trial their parents need;
    // the search scores only those nearest the subband's own grid. (The
    // root is never pruned: its rows are dm_grid_final.)
    for (SizeType l = m_niters - m_subband_depth; l < m_niters; ++l) {
        for (auto& grid : m_container.grids[l]) {
            const auto& g = grid.dt_grid;
            if (grid.staggered || g.size() < 3 ||
                grid.chan_end - grid.chan_begin < 2) {
                continue;
            }
            const auto want = subband_dt_grid(
                grid.f_start, grid.f_end, grid.chan_end - grid.chan_begin,
                m_tsamp, m_chan_dm, m_subband_max_snr_loss, g.front(),
                g.back());
            std::vector<uint32_t> rows;
            rows.reserve(want.size());
            for (const auto dt : want) {
                auto it = std::ranges::lower_bound(g, dt);
                if (it == g.end()) {
                    --it;
                } else if (it != g.begin() && (dt - *(it - 1)) <= (*it - dt)) {
                    --it;
                }
                rows.push_back(static_cast<uint32_t>(it - g.begin()));
            }
            std::ranges::sort(rows);
            const auto [first, last] = std::ranges::unique(rows);
            rows.erase(first, last);
            if (rows.size() < g.size()) {
                grid.search_rows = std::move(rows);
            }
        }
    }
}

void FDMTPlan::Impl::make_plan_iter0(
    const std::vector<std::vector<IndexType>>& active_level0) {
    const SizeType i_iter = 0;
    SizeType buf_offset   = 0;
    SizeType ncoords      = 0;
    m_container.grids[i_iter].resize(m_nchans);
    SizeType max_dt_overall = 0;

    for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
        const auto f_start = (m_df * static_cast<float>(i_sub)) + m_f_min;
        const auto f_end   = f_start + m_df;

        // Level 0 grid: dense from min_dt_chan to max_dt_chan required by
        // top-down pruning
        IndexType min_dt_chan = 0;
        IndexType max_dt_chan = 0;
        if (!active_level0[i_sub].empty()) {
            min_dt_chan = active_level0[i_sub].front();
            max_dt_chan = active_level0[i_sub].back(); // already sorted
        }
        max_dt_overall = std::max({
            max_dt_overall,
            static_cast<SizeType>(std::abs(min_dt_chan)),
            static_cast<SizeType>(std::abs(max_dt_chan)),
        });

        std::vector<IndexType> dt_sub;
        dt_sub.reserve(max_dt_chan - min_dt_chan + 1);
        for (IndexType dt = min_dt_chan; dt <= max_dt_chan; ++dt) {
            dt_sub.push_back(dt);
        }
        const auto ndt_sub = dt_sub.size();

        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto coord_cur = FDMTCoord{
                .i_sub           = i_sub,
                .i_dt            = i_dt,
                .nsamps          = m_nsamps,
                .buf_offset      = buf_offset,
                .i_coord_tail    = SIZE_MAX,
                .i_coord_head    = SIZE_MAX,
                .delay           = SIZE_MAX,
                .tail_buf_offset = SIZE_MAX,
                .tail_nsamps     = SIZE_MAX,
                .head_buf_offset = SIZE_MAX,
                .head_nsamps     = SIZE_MAX,
            };
            m_container.coordinates[i_iter].emplace_back(coord_cur);
            buf_offset += m_nsamps;
        }
        m_container.grids[i_iter][i_sub] = FDMTCoordGrid{
            .dt_grid      = std::move(dt_sub),
            .ndt          = ndt_sub,
            .coord_offset = ncoords,
            .f_start      = f_start,
            .f_end        = f_end,
            .chan_begin   = i_sub,
            .chan_end     = i_sub + 1,
            .search_rows  = {},
        };
        ncoords += ndt_sub;
    }

    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_container.grids[i_iter].begin(), m_container.grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    m_container.state_shape[i_iter] = {
        .nchans       = m_nchans,
        .nstaggered   = 0,
        .ndt_min      = ndt_min_it->ndt,
        .ndt_max      = ndt_max_it->ndt,
        .ncoords      = ncoords,
        .ncoords_sum  = 0,
        .ncoords_copy = 0,
        .nsamps       = m_nsamps,
        .nelements    = ncoords * m_nsamps,
        .dt_max       = max_dt_overall,
    };
    m_container.dt_grid_sub_top[i_iter] =
        m_container.grids[i_iter].back().dt_grid;
}

void FDMTPlan::Impl::make_level0_boxes() {
    m_container.chan_dm = m_chan_dm;
    m_container.level0_boxes.clear();
    m_container.level0_order.clear();
    m_container.level0_causal_delay = 0;
    m_container.level0_max_box_s    = m_container.state_shape[0].dt_max;
    if (m_chan_dm == 0.0) {
        return;
    }
    if (m_mode != FDMTMode::kValid) {
        throw std::invalid_argument(
            "FDMT: chan_dm != 0 requires mode \"valid\"");
    }
    // Channel c = [f_lo, f_hi] was coherently dedispersed to chan_dm about
    // its centre f_c, so a pulse of dispersion DM reaches frequency f of the
    // channel at lag(f) = k/tsamp [DM (f_ref^-2 - f^-2) + chan_dm (f^-2 -
    // f_c^-2)] behind the tree's reference: the bottom edge f_lo for dt >=
    // 0 and the top edge f_hi for dt < 0 (the upper merges align it there).
    // Its edges are, with l_lo = k chan_dm (f_lo^-2 - f_c^-2) / tsamp and
    // l_hi = k chan_dm (f_c^-2 - f_hi^-2) / tsamp,
    //   dt >= 0: e1 = l_lo,  e2 = dt - l_hi
    //   dt <  0: e1 = -l_hi, e2 = |dt| + l_lo.
    // The box spans s = round(|e2 - e1|) + 1 samples centred on (e1 + e2) /
    // 2; rounding the span first keeps a residual sweep under half a sample
    // to one sample. The centre lead is bounded by min(l_lo, -l_hi) - 1/4
    // whatever dt, so the causal delay D below depends on the channels and
    // chan_dm only, not on the delay grid.
    const auto& grids0 = m_container.grids[0];
    // k = 1 / 2.41e-4 (kDispConst, in double).
    const double k_ts = (1.0 / 2.41e-4) / static_cast<double>(m_tsamp);
    std::vector<double> l_lo(m_nchans);
    std::vector<double> l_hi(m_nchans);
    IndexType lead = 0;
    for (SizeType c = 0; c < m_nchans; ++c) {
        const auto f_lo = static_cast<double>(grids0[c].f_start);
        const auto f_hi = static_cast<double>(grids0[c].f_end);
        const auto f_c  = 0.5 * (f_lo + f_hi);
        const auto g_c  = 1.0 / (f_c * f_c);
        l_lo[c]         = k_ts * m_chan_dm * ((1.0 / (f_lo * f_lo)) - g_c);
        l_hi[c]         = k_ts * m_chan_dm * (g_c - (1.0 / (f_hi * f_hi)));
        const auto lo   = std::nearbyint(std::min(l_lo[c], -l_hi[c]) - 0.25);
        lead            = std::max(lead, static_cast<IndexType>(-lo));
    }
    const auto causal   = static_cast<int64_t>(std::max<IndexType>(lead, 0));
    const auto ncoords0 = m_container.state_shape[0].ncoords;
    m_container.level0_boxes.resize(ncoords0);
    m_container.level0_order.resize(ncoords0);
    SizeType extent    = 0;
    SizeType max_box_s = 0;
    for (SizeType c = 0; c < m_nchans; ++c) {
        const auto& g = grids0[c];
        for (SizeType i = 0; i < g.ndt; ++i) {
            const auto dt     = static_cast<double>(g.dt_grid[i]);
            const double e1   = dt >= 0.0 ? l_lo[c] : -l_hi[c];
            const double e2   = dt >= 0.0 ? dt - l_hi[c] : -dt + l_lo[c];
            const double span = std::nearbyint(std::abs(e2 - e1));
            const double mid  = 0.5 * (e1 + e2);
            const auto lo =
                static_cast<int64_t>(std::nearbyint(mid - (0.5 * span)));
            const auto pt = static_cast<int64_t>(std::nearbyint(mid));
            const auto sp = static_cast<int64_t>(span);
            if (lo + causal < 0 || pt + causal < 0 ||
                sp >= (int64_t{1} << 29) ||
                lo + causal + sp > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(fmt::format(
                    "FDMT: level-0 box of channel {} dt {} out of range "
                    "(lag {}, span {}, causal delay {})",
                    c, g.dt_grid[i], lo, sp, causal));
            }
            m_container.level0_boxes[g.coord_offset + i] = FDMTLevel0Box{
                .lag   = static_cast<int32_t>(lo + causal),
                .s     = static_cast<int32_t>(sp),
                .point = static_cast<int32_t>(pt + causal),
            };
            extent = std::max({extent, static_cast<SizeType>(lo + causal + sp),
                               static_cast<SizeType>(pt + causal)});
            max_box_s = std::max(max_box_s, static_cast<SizeType>(sp));
        }
        auto* order = m_container.level0_order.data() + g.coord_offset;
        for (SizeType i = 0; i < g.ndt; ++i) {
            order[i] = static_cast<int32_t>(i);
        }
        const auto* box = m_container.level0_boxes.data() + g.coord_offset;
        std::stable_sort(order, order + g.ndt, [box](int32_t a, int32_t b) {
            return box[a].lag != box[b].lag ? box[a].lag < box[b].lag
                                            : box[a].s < box[b].s;
        });
    }
    m_container.level0_causal_delay = static_cast<SizeType>(causal);
    m_container.level0_max_box_s    = max_box_s;
    // Raw input history, Fourier support and level-0 window extent need the
    // farthest read (`state_shape[0].dt_max`), while box-weight tables and
    // integer box bounds use the widest box (`level0_max_box_s`).
    auto& shape0  = m_container.state_shape[0];
    shape0.dt_max = std::max(shape0.dt_max, extent);
}

void FDMTPlan::Impl::make_plan(
    SizeType i_iter,
    const std::vector<std::vector<TreeNode>>& tree,
    const std::vector<std::vector<IndexType>>& active_level) {
    if (i_iter < 1 || i_iter > m_niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const auto& nsamps_prev = m_container.state_shape[i_iter - 1].nsamps;
    const auto& grids_prev  = m_container.grids[i_iter - 1];
    const auto& coords_prev = m_container.coordinates[i_iter - 1];
    auto& coords_cur        = m_container.coordinates[i_iter];
    auto& coords_sum_cur    = m_container.coordinates_sum[i_iter];
    auto& coords_copy_cur   = m_container.coordinates_copy[i_iter];

    const auto& nodes         = tree[i_iter];
    const SizeType nsubs_cur  = nodes.size();
    const SizeType nstaggered = static_cast<SizeType>(
        std::ranges::count_if(nodes, &TreeNode::staggered));
    const SizeType nchans_cur = nsubs_cur - nstaggered;

    // Determine max absolute dt trial at this iteration
    SizeType dt_max_iter = 0;
    for (SizeType i_sub = 0; i_sub < nsubs_cur; ++i_sub) {
        if (!active_level[i_sub].empty()) {
            const auto abs_front =
                static_cast<SizeType>(std::abs(active_level[i_sub].front()));
            const auto abs_back =
                static_cast<SizeType>(std::abs(active_level[i_sub].back()));
            dt_max_iter = std::max({dt_max_iter, abs_front, abs_back});
        }
    }

    const auto abs_dt_max =
        static_cast<SizeType>(std::max(std::abs(m_dt_min), std::abs(m_dt_max)));
    if (dt_max_iter > abs_dt_max) {
        throw std::runtime_error(
            fmt::format("dt_max_iter={} is greater than abs_dt_max={}",
                        dt_max_iter, abs_dt_max));
    }
    // Note: mode="valid" intentionally allows dt_max_iter > m_nsamps
    // (block size smaller than this level's max delay) -- the
    // cross-block history FIFO in offset_add<kValid> (see
    // FDMTCoord::hist_offset) accumulates real history across multiple
    // blocks in that case rather than requiring one block to cover the
    // whole delay.
    const auto nsamps_iter =
        m_nsamps + ((m_mode == FDMTMode::kFull) ? dt_max_iter : 0);

    SizeType buf_offset = 0;
    SizeType ncoords    = 0;
    m_container.grids[i_iter].resize(nsubs_cur);

    for (SizeType i_sub = 0; i_sub < nsubs_cur; ++i_sub) {
        const auto& node                   = nodes[i_sub];
        const bool is_copy                 = node.head == SIZE_MAX;
        const auto& grids_tail             = grids_prev[node.tail];
        const auto [f_start, f_mid, f_end] = node_freqs(node, tree[i_iter - 1]);

        const auto& dt_sub   = active_level[i_sub];
        const auto ndt_sub   = dt_sub.size();
        const float tail_phi = utils::cff(f_start, f_mid, f_start, f_end);

        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const IndexType dt = dt_sub[i_dt];
            if (is_copy) {
                const auto i_dt_tail =
                    utils::find_nearest_sorted_idx(grids_tail.dt_grid, dt);
                const auto i_coord_tail = grids_tail.coord_offset + i_dt_tail;
                const auto coord_cur    = FDMTCoord{
                    .i_sub           = i_sub,
                    .i_dt            = i_dt,
                    .nsamps          = nsamps_iter,
                    .buf_offset      = buf_offset,
                    .i_coord_tail    = i_coord_tail,
                    .i_coord_head    = SIZE_MAX,
                    .delay           = 0,
                    .tail_buf_offset = coords_prev[i_coord_tail].buf_offset,
                    .tail_nsamps     = coords_prev[i_coord_tail].nsamps,
                    .head_buf_offset = SIZE_MAX,
                    .head_nsamps     = SIZE_MAX,
                };

                coords_cur.emplace_back(coord_cur);
                coords_copy_cur.emplace_back(coord_cur);
            } else {
                const auto& grids_head = grids_prev[node.head];
                const auto dt_tail     = static_cast<IndexType>(
                    std::nearbyint(static_cast<float>(dt) * tail_phi));
                const auto dt_head = dt - dt_tail;

                // Sanity check on the fractional-delay split itself,
                // independent of sign handling below: dt_tail must never
                // overshoot dt in magnitude or point the opposite way
                // (tail_phi is a ratio in [0, 1], so dt_tail should lie
                // between 0 and dt, inclusive, on the same side of zero
                // as dt). This is the signed generalization of the
                // dt_tail > dt guard that existed before negative-DM
                // support; it only runs at plan-build time, never in the
                // per-sample hot path.
                if ((dt >= 0 && (dt_tail < 0 || dt_tail > dt)) ||
                    (dt < 0 && (dt_tail > 0 || dt_tail < dt))) {
                    throw std::runtime_error(fmt::format(
                        "Invalid dt_tail (sign/magnitude mismatch with "
                        "dt): dt_tail={}, dt={}",
                        dt_tail, dt));
                }

                const auto i_dt_tail =
                    utils::find_nearest_sorted_idx(grids_tail.dt_grid, dt_tail);
                const auto i_dt_head =
                    utils::find_nearest_sorted_idx(grids_head.dt_grid, dt_head);
                const auto i_coord_tail = grids_tail.coord_offset + i_dt_tail;
                const auto i_coord_head = grids_head.coord_offset + i_dt_head;

                // Sign-aware reference/shift assignment. `offset_add`
                // (the hot loop) only ever computes
                // out[k] = tail[k] + head[k - delay_shift] with an
                // unsigned delay_shift; it has no notion of DM sign at
                // all. Both branches below feed it exactly that shape —
                // the sign of dt only changes *which* child subband
                // plays the unshifted "tail" (reference) role in that
                // formula, decided once here at plan-build time, not
                // per sample:
                //
                //  - dt >= 0 (standard): the low-frequency child is the
                //    reference (delay_shift = 0 relative to it); the
                //    high-frequency child is shifted forward in time by
                //    dt_tail samples, since a positive DM means the
                //    high-frequency signal arrives *after* the low
                //    frequency signal by that amount.
                //  - dt < 0: the roles swap. The high-frequency child
                //    becomes the reference, and the low-frequency child
                //    (which now arrives *earlier*) is instead shifted
                //    forward by |dt_head| samples so that it lines up
                //    causally with the high-frequency reference.
                //
                // This is deliberately different from — but equivalent
                // in intent to — the shift_input/shift_output axis-offset
                // sketched (never implemented) in Zackay & Ofek's
                // reference code: instead of reindexing the whole dt-axis
                // by a fixed +delta_t offset so every trial has a
                // non-negative array index, we pick whichever child is
                // already causally "later" to be the shifted operand, so
                // delay_shift here is always >= 0 without needing any
                // axis reindexing or extra buffer space.
                SizeType delay_shift = 0;
                SizeType tail_offset = 0;
                SizeType tail_n      = 0;
                SizeType head_offset = 0;
                SizeType head_n      = 0;

                if (dt >= 0) {
                    delay_shift = static_cast<SizeType>(dt_tail);
                    tail_offset = coords_prev[i_coord_tail].buf_offset;
                    tail_n      = coords_prev[i_coord_tail].nsamps;
                    head_offset = coords_prev[i_coord_head].buf_offset;
                    head_n      = coords_prev[i_coord_head].nsamps;
                } else {
                    // Negative DM: lower freq (tail) arrived earlier than
                    // higher freq (head). Reference is head (top edge), so
                    // tail is shifted by |dt_head| relative to head!
                    delay_shift = static_cast<SizeType>(std::abs(dt_head));
                    tail_offset =
                        coords_prev[i_coord_head]
                            .buf_offset; // reference (head, unshifted)
                    tail_n      = coords_prev[i_coord_head].nsamps;
                    head_offset = coords_prev[i_coord_tail]
                                      .buf_offset; // shifted by |dt_head|
                    head_n      = coords_prev[i_coord_tail].nsamps;
                }

                // "full"/"roll" have no cross-block history mechanism,
                // so a delay that doesn't fit within one block can never
                // be satisfied and must fail fast at plan-build time.
                // "valid" mode's cross-block history FIFO (see
                // FDMTCoord::hist_offset, offset_add<kValid> in
                // fdmt.cpp) is explicitly designed to support
                // delay_shift >= nsamps_prev by accumulating real
                // history across multiple blocks.
                if (m_mode != FDMTMode::kValid && delay_shift >= nsamps_prev) {
                    throw std::runtime_error(
                        fmt::format("DM delay is greater than input size: "
                                    "delay_shift={}, nsamps_prev={}",
                                    delay_shift, nsamps_prev));
                }

                const auto coord_cur = FDMTCoord{
                    .i_sub           = i_sub,
                    .i_dt            = i_dt,
                    .nsamps          = nsamps_iter,
                    .buf_offset      = buf_offset,
                    .i_coord_tail    = i_coord_tail,
                    .i_coord_head    = i_coord_head,
                    .delay           = delay_shift,
                    .tail_buf_offset = tail_offset,
                    .tail_nsamps     = tail_n,
                    .head_buf_offset = head_offset,
                    .head_nsamps     = head_n,
                };
                coords_cur.emplace_back(coord_cur);
                coords_sum_cur.emplace_back(coord_cur);
            }
            buf_offset += nsamps_iter;
        }
        m_container.grids[i_iter][i_sub] = FDMTCoordGrid{
            .dt_grid      = dt_sub,
            .ndt          = ndt_sub,
            .coord_offset = ncoords,
            .f_start      = f_start,
            .f_end        = f_end,
            .chan_begin   = node.chan_begin,
            .chan_end     = node.chan_end,
            .i_sub_tail   = node.tail,
            .i_sub_head   = node.head,
            .staggered    = node.staggered,
            .search_rows  = {},
        };
        ncoords += ndt_sub;
    }

    const auto ncoords_sum              = coords_sum_cur.size();
    const auto ncoords_copy             = coords_copy_cur.size();
    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_container.grids[i_iter].begin(), m_container.grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    m_container.state_shape[i_iter] = {
        .nchans       = nchans_cur,
        .nstaggered   = nstaggered,
        .ndt_min      = ndt_min_it->ndt,
        .ndt_max      = ndt_max_it->ndt,
        .ncoords      = ncoords,
        .ncoords_sum  = ncoords_sum,
        .ncoords_copy = ncoords_copy,
        .nsamps       = nsamps_iter,
        .nelements    = ncoords * nsamps_iter,
        .dt_max       = dt_max_iter,
    };
    m_container.dt_grid_sub_top[i_iter] =
        m_container.grids[i_iter][nchans_cur - 1].dt_grid;
}

} // namespace dmt::plans
