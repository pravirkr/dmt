#include "dmt/common/plans.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"

namespace dmt::plans {

constexpr std::string_view FDMTShape::header_fmt() {
    return "{ncoords} ({nchans}x[{ndt_min}..{ndt_max}]) x "
           "{nsamps}, {nelements}";
}

std::string FDMTShape::to_string() const {
    return std::format("{:5d} ({:5d}x[{:5d}..{:5d}]) x {:6d}, {:10d}", ncoords,
                       nchans, ndt_min, ndt_max, nsamps, nelements);
}

std::string FDMTComplexity::to_string() const {
    std::string result = std::format(
        "FDMT Complexity:\n"
        "  Output DM trials (N_dt)      : {}\n"
        "  Input channels   (N_chans)   : {}\n"
        "  Brute-force ops / time samp  : {} (N_dt x N_chans)\n"
        "  FDMT tree nodes  (Levels 1..M): {}\n"
        "  FDMT additions   (Levels 1..M): {} (offset_add operations)\n"
        "  FDMT copy nodes  (Levels 1..M): {}\n"
        "  Theoretical Speedup Factor   : {:.2f}x (brute_force / "
        "fdmt_additions)",
        n_dt, n_chans, brute_force_ops, total_tree_nodes, sum_additions,
        copy_nodes, ops_ratio);
    if (ops_ratio > 0.0F && ops_ratio < 2.0F) {
        result += "\n  Note: Speedup factor is < 2.0x (coarse dt spacing "
                  "approaches brute-force complexity)";
    }
    return result;
}

FDMTPlanContainer::FDMTPlanContainer(SizeType niters) {
    state_shape.resize(niters + 1);
    grids.resize(niters + 1);
    coordinates.resize(niters + 1);
    coordinates_sum.resize(niters + 1);
    coordinates_copy.resize(niters + 1);
    dt_grid_sub_top.resize(niters + 1);
    df_top.resize(niters + 1);
    df_bot.resize(niters + 1);
}

SizeType FDMTPlanContainer::get_memory_usage() const noexcept {
    SizeType mem_use = 0;
    mem_use += state_shape.size() * sizeof(FDMTShape);
    for (const auto& grid_iter : grids) {
        mem_use += grid_iter.size() * sizeof(FDMTCoordGrid);
    }
    for (const auto& coord : coordinates) {
        mem_use += 2 * coord.size() * sizeof(FDMTCoord);
    }
    for (const auto& dt_grid : dt_grid_sub_top) {
        mem_use += dt_grid.size() * sizeof(SizeType);
    }
    mem_use += df_top.size() * sizeof(float);
    mem_use += df_bot.size() * sizeof(float);
    return mem_use;
}

SizeType FDMTPlanContainer::get_buffer_size() const noexcept {
    auto max_elements = std::ranges::max_element(
        state_shape, [](const FDMTShape& a, const FDMTShape& b) {
            return a.nelements < b.nelements;
        });
    return max_elements->nelements;
}

class FDMTPlan::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         SizeType dt_max,
         SizeType dt_min,
         SizeType dt_step,
         std::string_view mode,
         bool verbose)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_nchans(nchans),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_dt_max(dt_max),
          m_dt_min(dt_min),
          m_dt_step(dt_step),
          m_mode(mode),
          m_is_custom_grid(false) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        validate_inputs();
        configure_plan();
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<SizeType>& dt_grid,
         std::string_view mode,
         bool verbose)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_nchans(nchans),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_dt_max(0),
          m_dt_min(0),
          m_dt_step(0),
          m_mode(mode),
          m_is_custom_grid(true),
          m_dt_grid_target(dt_grid) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
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

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         std::string_view mode,
         bool verbose)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_nchans(nchans),
          m_nsamps(nsamps),
          m_tsamp(tsamp),
          m_dt_max(0),
          m_dt_min(0),
          m_dt_step(0),
          m_mode(mode),
          m_is_custom_grid(true) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        if (dm_grid.empty()) {
            throw std::invalid_argument("FDMT: dm_grid must not be empty");
        }
        for (const auto dm : dm_grid) {
            if (dm < 0.0F) {
                throw std::invalid_argument(
                    "FDMT: dm_grid values must be non-negative");
            }
        }
        auto dm_sorted = dm_grid;
        std::ranges::sort(dm_sorted);
        const auto [dm_first, dm_last] = std::ranges::unique(dm_sorted);
        dm_sorted.erase(dm_first, dm_last);

        const float dm_conv = utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
        m_dt_grid_target.reserve(dm_sorted.size());
        for (const auto dm : dm_sorted) {
            m_dt_grid_target.push_back(
                static_cast<SizeType>(std::nearbyint(dm / dm_conv)));
        }
        std::ranges::sort(m_dt_grid_target);
        const auto [first, last] = std::ranges::unique(m_dt_grid_target);
        m_dt_grid_target.erase(first, last);

        m_dt_min = m_dt_grid_target.front();
        m_dt_max = m_dt_grid_target.back();
        validate_inputs();
        configure_plan();
    }

    ~Impl()                                = default;
    Impl(Impl&& other) noexcept            = default;
    Impl& operator=(Impl&& other) noexcept = default;
    Impl(const Impl& other)                = default;
    Impl& operator=(const Impl& other)     = default;

    float get_f_min() const noexcept { return m_f_min; }
    float get_f_max() const noexcept { return m_f_max; }
    SizeType get_nchans() const noexcept { return m_nchans; }
    SizeType get_nsamps() const noexcept { return m_nsamps; }
    float get_tsamp() const noexcept { return m_tsamp; }
    SizeType get_dt_max() const noexcept { return m_dt_max; }
    SizeType get_dt_min() const noexcept { return m_dt_min; }
    SizeType get_dt_step() const noexcept { return m_dt_step; }
    bool is_custom_grid() const noexcept { return m_is_custom_grid; }
    float get_df() const noexcept { return m_df; }
    SizeType get_niters() const noexcept { return m_niters; }
    const FDMTPlanContainer& get_container() const noexcept {
        return m_container;
    }
    FDMTComplexity get_complexity() const noexcept {
        const auto n_dt            = get_dt_grid_final().size();
        const auto n_chans         = m_nchans;
        const auto brute_force_ops = n_dt * n_chans;
        SizeType total_tree_nodes  = 0;
        SizeType sum_additions     = 0;
        SizeType copy_nodes        = 0;
        for (SizeType l = 1; l <= m_niters; ++l) {
            total_tree_nodes += m_container.coordinates[l].size();
            sum_additions += m_container.coordinates_sum[l].size();
            copy_nodes += m_container.coordinates_copy[l].size();
        }
        const float ops_ratio = sum_additions > 0
                                    ? static_cast<float>(brute_force_ops) /
                                          static_cast<float>(sum_additions)
                                    : 0.0F;
        return FDMTComplexity{
            .n_dt             = n_dt,
            .n_chans          = n_chans,
            .brute_force_ops  = brute_force_ops,
            .total_tree_nodes = total_tree_nodes,
            .sum_additions    = sum_additions,
            .copy_nodes       = copy_nodes,
            .ops_ratio        = ops_ratio,
        };
    }
    void print_complexity_summary() const {
        const auto comp = get_complexity();
        spdlog::info("{}", comp.to_string());
    }
    std::vector<SizeType> get_dt_grid_final() const noexcept {
        return m_container.grids[m_niters][0].dt_grid;
    }
    std::vector<float> get_dm_grid_final() const noexcept {
        const float dm_conv = utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
        const auto& dt_grid_final = get_dt_grid_final();
        std::vector<float> dm_grid_final(dt_grid_final.size());
        std::ranges::transform(
            dt_grid_final, dm_grid_final.begin(),
            [dm_conv](auto& dt) { return static_cast<float>(dt) * dm_conv; });
        return dm_grid_final;
    }
    std::vector<float> get_smearing_grid_final() const noexcept {
        const auto ndms = get_dmt_ndms();
        std::vector<float> smearing_grid(ndms * m_nchans, 0.0F);
        if (m_niters == 0 || m_nchans == 0) {
            return smearing_grid;
        }

        // Helper lambda to recursively trace coordinates down to level 0
        auto trace_coord = [&](auto& self, SizeType level, SizeType coord_idx,
                               SizeType dm_idx) -> void {
            if (level == 0) {
                if (coord_idx < m_container.coordinates[0].size()) {
                    const auto& coord0 = m_container.coordinates[0][coord_idx];
                    const auto chan    = coord0.i_sub;
                    if (chan < m_nchans &&
                        coord0.i_dt <
                            m_container.grids[0][chan].dt_grid.size()) {
                        const auto dt =
                            m_container.grids[0][chan].dt_grid[coord0.i_dt];
                        smearing_grid[(dm_idx * m_nchans) + chan] =
                            static_cast<float>(dt);
                    }
                }
                return;
            }

            if (level < m_container.coordinates.size() &&
                coord_idx < m_container.coordinates[level].size()) {
                const auto& coord = m_container.coordinates[level][coord_idx];
                if (coord.i_coord_tail != SIZE_MAX) {
                    self(self, level - 1, coord.i_coord_tail, dm_idx);
                }
                if (coord.i_coord_head != SIZE_MAX) {
                    self(self, level - 1, coord.i_coord_head, dm_idx);
                }
            }
        };

        for (SizeType i_dm = 0; i_dm < ndms; ++i_dm) {
            trace_coord(trace_coord, m_niters, i_dm, i_dm);
        }
        return smearing_grid;
    }
    SizeType get_dmt_ndms() const noexcept {
        return m_container.state_shape[m_niters].ncoords;
    }
    SizeType get_dmt_nsamps() const noexcept {
        return m_container.state_shape[m_niters].nsamps;
    }
    SizeType get_dmt_size() const noexcept {
        return m_container.state_shape[m_niters].nelements;
    }
    SizeType get_buffer_size() const noexcept {
        return m_container.get_buffer_size();
    }
    SizeType get_history_size() const noexcept {
        return m_nchans * m_container.state_shape[m_niters].dt_max;
    }
    SizeType get_history_init_size() const noexcept {
        return m_nchans * m_container.state_shape[0].dt_max;
    }

    void print_summary(std::string_view prefix) const {
        auto size_in_mb = [](SizeType count, SizeType size) {
            return static_cast<float>(count * size) / 1024.0F / 1024.0F;
        };
        const auto& state_shape = m_container.state_shape;
        const auto waterfall_size =
            state_shape[0].nchans * state_shape[0].nsamps;
        const auto dmt_size     = get_dmt_size();
        const auto history_size = get_history_size();
        const auto buffer_size  = 2 * m_buffer_size;
        const auto niters       = state_shape.size() - 1;

        const auto waterfall_size_mb =
            size_in_mb(waterfall_size, sizeof(float));
        const auto dmt_size_mb     = size_in_mb(dmt_size, sizeof(float));
        const auto history_size_mb = size_in_mb(history_size, sizeof(float));
        const auto buffer_size_mb  = size_in_mb(buffer_size, sizeof(float));
        const auto plan_use_mb = size_in_mb(m_container.get_memory_usage(), 1);

        std::array<std::string, 5> lines = {
            std::format("*** FDMT Plan Summary ***"),
            std::format("Input: Waterfall Size: ({} x {}; {:.1f} MB ), DMT "
                        "Size: ({} x {}; {:.1f} MB )",
                        state_shape[0].nchans, state_shape[0].nsamps,
                        waterfall_size_mb, state_shape[niters].ncoords,
                        state_shape[niters].nsamps, dmt_size_mb),
            std::format("Plan Memory Usage: {:.1f} MB", plan_use_mb),
            std::format("Plan Buffer Size: ({}; {:.1f} MB ), History Size: "
                        "({}; {:.1f} MB )",
                        buffer_size, buffer_size_mb, history_size,
                        history_size_mb),
            std::format("Plan Details: {}", FDMTShape::header_fmt())};
        std::cout << '\n';
        for (const auto& line : lines) {
            std::cout << prefix << line << '\n';
        }
        for (SizeType i_iter = 0; i_iter < niters + 1; ++i_iter) {
            std::cout << prefix
                      << std::format("Iteration {:2d}: {}\n", i_iter,
                                     state_shape[i_iter].to_string());
        }
        std::cout << prefix << std::format("{:*>80}\n", "");
    }

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    SizeType m_nsamps;
    float m_tsamp;
    SizeType m_dt_max;
    SizeType m_dt_min;
    SizeType m_dt_step;
    bool m_is_custom_grid{false};
    std::vector<SizeType> m_dt_grid_target{};
    std::string_view m_mode;

    float m_df{};
    SizeType m_niters{};
    FDMTPlanContainer m_container;
    SizeType m_buffer_size{};

    void validate_inputs() const {
        if (m_f_min >= m_f_max) {
            throw std::invalid_argument(std::format(
                "FDMT: f_min={} must be less than f_max={}", m_f_min, m_f_max));
        }
        if (m_nchans < 2) {
            throw std::invalid_argument(
                std::format("FDMT: nchans={} must be at least 2", m_nchans));
        }
        if (m_nsamps == 0) {
            throw std::invalid_argument(std::format(
                "FDMT: nsamps={} must be greater than 0", m_nsamps));
        }
        if (m_tsamp <= 0) {
            throw std::invalid_argument(
                std::format("FDMT: tsamp={} must be greater than 0", m_tsamp));
        }
        if (m_is_custom_grid) {
            if (m_dt_grid_target.empty()) {
                throw std::invalid_argument("FDMT: dt_grid must not be empty");
            }
            if (m_dt_max == 0) {
                throw std::invalid_argument(
                    "FDMT: dt_max in dt_grid must be greater than 0");
            }
        } else {
            if (m_dt_max == 0) {
                throw std::invalid_argument(std::format(
                    "FDMT: dt_max={} must be greater than 0", m_dt_max));
            }
            if (m_dt_min >= m_dt_max) {
                throw std::invalid_argument(
                    std::format("FDMT: dt_min={} must be less than dt_max={}",
                                m_dt_min, m_dt_max));
            }
            if (m_dt_step == 0) {
                throw std::invalid_argument(std::format(
                    "FDMT: dt_step={} must be greater than 0", m_dt_step));
            }
            if (m_dt_step > (m_dt_max - m_dt_min)) {
                throw std::invalid_argument(
                    std::format("FDMT: dt_step={} cannot be greater than "
                                "(dt_max - dt_min)={}",
                                m_dt_step, m_dt_max - m_dt_min));
            }
        }
        if (m_mode != "full" && m_mode != "valid" && m_mode != "roll") {
            throw std::invalid_argument(std::format(
                "FDMT: mode={} must be 'full' or 'valid' or 'roll'", m_mode));
        }
    }

    std::vector<std::vector<std::vector<SizeType>>>
    determine_active_grids(const std::vector<SizeType>& nchans_level) const {
        std::vector<std::vector<std::vector<SizeType>>> active_dts(m_niters +
                                                                   1);
        for (SizeType l = 0; l <= m_niters; ++l) {
            active_dts[l].resize(nchans_level[l]);
        }

        // Target sparse grid at root level (m_niters, subband 0)
        if (m_is_custom_grid) {
            active_dts[m_niters][0] = m_dt_grid_target;
        } else {
            for (SizeType dt = m_dt_min; dt <= m_dt_max; dt += m_dt_step) {
                active_dts[m_niters][0].push_back(dt);
            }
        }

        // Top-down traversal from root to level 1
        for (SizeType l = m_niters; l >= 1; --l) {
            const auto nchans_cur  = nchans_level[l];
            const auto nchans_prev = nchans_level[l - 1];
            const bool do_copy     = (nchans_prev % 2 == 1);
            const float df_top     = m_container.df_top[l];
            const float df_bot     = m_container.df_bot[l];

            for (SizeType i_sub = 0; i_sub < nchans_cur; ++i_sub) {
                const auto& cur_dts = active_dts[l][i_sub];
                if (cur_dts.empty()) {
                    continue;
                }

                const auto f_start =
                    (df_bot * static_cast<float>(i_sub)) + m_f_min;
                float f_end = 0.0F;
                float f_mid = 0.0F;
                if (i_sub == nchans_cur - 1) {
                    if (do_copy) {
                        f_end = f_start + (df_top * 2);
                        f_mid = f_start + df_top;
                    } else {
                        f_end = f_start + df_top;
                        f_mid = f_start + (df_bot / 2);
                    }
                } else {
                    f_end = f_start + df_bot;
                    f_mid = f_start + (df_bot / 2);
                }
                const float tail_phi =
                    utils::cff(f_start, f_mid, f_start, f_end);

                for (const auto dt : cur_dts) {
                    if (i_sub == nchans_cur - 1 && do_copy) {
                        active_dts[l - 1][2 * i_sub].push_back(dt);
                    } else {
                        const auto dt_tail = static_cast<SizeType>(
                            std::nearbyint(static_cast<float>(dt) * tail_phi));
                        const auto dt_head = dt - dt_tail;
                        active_dts[l - 1][2 * i_sub].push_back(dt_tail);
                        active_dts[l - 1][(2 * i_sub) + 1].push_back(dt_head);
                    }
                }
            }

            // Deduplicate and sort active_dts[l - 1]
            for (SizeType i_sub = 0; i_sub < nchans_prev; ++i_sub) {
                auto& vec = active_dts[l - 1][i_sub];
                std::ranges::sort(vec);
                const auto [first, last] = std::ranges::unique(vec);
                vec.erase(first, last);
            }
        }

        return active_dts;
    }

    void configure_plan() {
        m_df        = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
        m_niters    = static_cast<SizeType>(std::ceil(std::log2(m_nchans)));
        m_container = FDMTPlanContainer(m_niters);

        // Step 1: Compute hierarchy structure and frequency bandwidths
        std::vector<SizeType> nchans_level(m_niters + 1);
        nchans_level[0]       = m_nchans;
        m_container.df_top[0] = m_df;
        m_container.df_bot[0] = m_df;
        for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
            const auto nchans_prev = nchans_level[i_iter - 1];
            nchans_level[i_iter]   = (nchans_prev / 2) + (nchans_prev % 2);
            const bool do_copy     = (nchans_prev % 2 == 1);
            m_container.df_top[i_iter] =
                do_copy ? m_container.df_top[i_iter - 1]
                        : m_container.df_top[i_iter - 1] +
                              m_container.df_bot[i_iter - 1];
            m_container.df_bot[i_iter] = m_container.df_bot[i_iter - 1] * 2;
        }

        // Step 2: Top-down reachability traversal to identify active (i_sub,
        // dt) nodes
        auto active_dts = determine_active_grids(nchans_level);

        // Step 3: Populate grids and coordinates bottom-up using active sets
        make_plan_iter0(active_dts[0]);
        for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
            make_plan(i_iter, active_dts[i_iter]);
        }

        m_buffer_size = m_container.get_buffer_size();
        spdlog::debug("FDMT: configured fdmt plan");
        spdlog::debug(
            "FDMT: df={}, dt_max={}, dt_min={}, dt_step={}, niters={}", m_df,
            m_dt_max, m_dt_min, m_dt_step, m_niters);
    }

    void
    make_plan_iter0(const std::vector<std::vector<SizeType>>& active_level0) {
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
            SizeType min_dt_chan = 0;
            SizeType max_dt_chan = 0;
            if (!active_level0[i_sub].empty()) {
                min_dt_chan = active_level0[i_sub].front();
                max_dt_chan = active_level0[i_sub].back(); // already sorted
            }
            max_dt_overall = std::max(max_dt_overall, max_dt_chan);

            std::vector<SizeType> dt_sub;
            dt_sub.reserve(max_dt_chan - min_dt_chan + 1);
            for (SizeType dt = min_dt_chan; dt <= max_dt_chan; ++dt) {
                dt_sub.push_back(dt);
            }
            const auto ndt_sub = dt_sub.size();

            for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
                const auto coord_cur = FDMTCoord{.i_sub           = i_sub,
                                                 .i_dt            = i_dt,
                                                 .nsamps          = m_nsamps,
                                                 .buf_offset      = buf_offset,
                                                 .i_coord_tail    = SIZE_MAX,
                                                 .i_coord_head    = SIZE_MAX,
                                                 .delay           = SIZE_MAX,
                                                 .tail_buf_offset = SIZE_MAX,
                                                 .tail_nsamps     = SIZE_MAX,
                                                 .head_buf_offset = SIZE_MAX,
                                                 .head_nsamps     = SIZE_MAX};
                m_container.coordinates[i_iter].emplace_back(coord_cur);
                buf_offset += m_nsamps;
            }
            m_container.grids[i_iter][i_sub] =
                FDMTCoordGrid{.dt_grid      = std::move(dt_sub),
                              .ndt          = ndt_sub,
                              .coord_offset = ncoords,
                              .f_start      = f_start,
                              .f_end        = f_end};
            ncoords += ndt_sub;
        }

        const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
            m_container.grids[i_iter].begin(), m_container.grids[i_iter].end(),
            [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
        m_container.state_shape[i_iter] = {.nchans       = m_nchans,
                                           .ndt_min      = ndt_min_it->ndt,
                                           .ndt_max      = ndt_max_it->ndt,
                                           .ncoords      = ncoords,
                                           .ncoords_sum  = 0,
                                           .ncoords_copy = 0,
                                           .nsamps       = m_nsamps,
                                           .nelements    = ncoords * m_nsamps,
                                           .dt_max       = max_dt_overall};
        m_container.dt_grid_sub_top[i_iter] =
            m_container.grids[i_iter].back().dt_grid;
    }

    void make_plan(SizeType i_iter,
                   const std::vector<std::vector<SizeType>>& active_level) {
        if (i_iter < 1 || i_iter > m_niters) {
            throw std::invalid_argument("Invalid iteration number");
        }
        const auto& df_bot_prev = m_container.df_bot[i_iter - 1];
        const auto& df_top_prev = m_container.df_top[i_iter - 1];
        const auto& nchans_prev = m_container.state_shape[i_iter - 1].nchans;
        const auto& nsamps_prev = m_container.state_shape[i_iter - 1].nsamps;
        const auto& grids_prev  = m_container.grids[i_iter - 1];
        const auto& coords_prev = m_container.coordinates[i_iter - 1];
        auto& coords_cur        = m_container.coordinates[i_iter];
        auto& coords_sum_cur    = m_container.coordinates_sum[i_iter];
        auto& coords_copy_cur   = m_container.coordinates_copy[i_iter];

        const SizeType nchans_cur = (nchans_prev / 2) + (nchans_prev % 2);
        const bool do_copy        = (nchans_prev % 2 == 1);
        const float df_top        = m_container.df_top[i_iter];
        const float df_bot        = m_container.df_bot[i_iter];

        // Determine max dt trial at this iteration
        SizeType dt_max_iter = 0;
        for (SizeType i_sub = 0; i_sub < nchans_cur; ++i_sub) {
            if (!active_level[i_sub].empty()) {
                dt_max_iter = std::max(dt_max_iter, active_level[i_sub].back());
            }
        }

        if (dt_max_iter > m_dt_max) {
            throw std::runtime_error(
                std::format("dt_max_iter={} is greater than dt_max={}",
                            dt_max_iter, m_dt_max));
        }
        if (m_mode == "valid" && dt_max_iter > m_nsamps) {
            throw std::runtime_error(std::format(
                "dt_max_iter={} is greater than nsamps={} for mode='{}'",
                dt_max_iter, m_nsamps, m_mode));
        }
        const auto nsamps_iter =
            m_nsamps + ((m_mode == "full") ? dt_max_iter : 0);

        SizeType buf_offset = 0;
        SizeType ncoords    = 0;
        m_container.grids[i_iter].resize(nchans_cur);

        for (SizeType i_sub = 0; i_sub < nchans_cur; ++i_sub) {
            const auto& grids_tail = grids_prev[2 * i_sub];
            const auto f_start = (df_bot * static_cast<float>(i_sub)) + m_f_min;
            float f_end        = 0.0F;
            float f_mid        = 0.0F;

            if (i_sub == nchans_cur - 1) {
                if (do_copy) {
                    f_end = f_start + (df_top * 2);
                    f_mid = f_start + df_top;
                } else {
                    f_end = f_start + df_top;
                    f_mid = f_start + (df_bot / 2);
                }
            } else {
                f_end = f_start + df_bot;
                f_mid = f_start + (df_bot / 2);
            }

            const auto& dt_sub   = active_level[i_sub];
            const auto ndt_sub   = dt_sub.size();
            const float tail_phi = utils::cff(f_start, f_mid, f_start, f_end);

            for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
                const SizeType dt = dt_sub[i_dt];
                if (i_sub == nchans_cur - 1 && do_copy) {
                    const auto i_dt_tail =
                        utils::find_nearest_sorted_idx(grids_tail.dt_grid, dt);
                    const auto i_coord_tail =
                        grids_tail.coord_offset + i_dt_tail;
                    const auto coord_cur = FDMTCoord{
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
                        .head_nsamps     = SIZE_MAX};

                    coords_cur.emplace_back(coord_cur);
                    coords_copy_cur.emplace_back(coord_cur);
                } else {
                    const auto& grids_head = grids_prev[(2 * i_sub) + 1];
                    const auto dt_tail     = static_cast<SizeType>(
                        std::nearbyint(static_cast<float>(dt) * tail_phi));
                    if (dt_tail > dt) {
                        throw std::runtime_error(std::format(
                            "Invalid dt_tail (> dt) values: dt_tail={}, dt={}",
                            dt_tail, dt));
                    }
                    if (dt_tail >= nsamps_prev) {
                        throw std::runtime_error(std::format(
                            "DM delay is greater than input size (dt_tail "
                            ">= nsamps_prev): dt_tail={}, nsamps_prev={}",
                            dt_tail, nsamps_prev));
                    }
                    const auto dt_head   = dt - dt_tail;
                    const auto i_dt_tail = utils::find_nearest_sorted_idx(
                        grids_tail.dt_grid, dt_tail);
                    const auto i_dt_head = utils::find_nearest_sorted_idx(
                        grids_head.dt_grid, dt_head);
                    const auto i_coord_tail =
                        grids_tail.coord_offset + i_dt_tail;
                    const auto i_coord_head =
                        grids_head.coord_offset + i_dt_head;
                    const auto coord_cur = FDMTCoord{
                        .i_sub           = i_sub,
                        .i_dt            = i_dt,
                        .nsamps          = nsamps_iter,
                        .buf_offset      = buf_offset,
                        .i_coord_tail    = i_coord_tail,
                        .i_coord_head    = i_coord_head,
                        .delay           = dt_tail,
                        .tail_buf_offset = coords_prev[i_coord_tail].buf_offset,
                        .tail_nsamps     = coords_prev[i_coord_tail].nsamps,
                        .head_buf_offset = coords_prev[i_coord_head].buf_offset,
                        .head_nsamps     = coords_prev[i_coord_head].nsamps};
                    coords_cur.emplace_back(coord_cur);
                    coords_sum_cur.emplace_back(coord_cur);
                }
                buf_offset += nsamps_iter;
            }
            m_container.grids[i_iter][i_sub] =
                FDMTCoordGrid{.dt_grid      = dt_sub,
                              .ndt          = ndt_sub,
                              .coord_offset = ncoords,
                              .f_start      = f_start,
                              .f_end        = f_end};
            ncoords += ndt_sub;
        }

        const auto ncoords_sum              = coords_sum_cur.size();
        const auto ncoords_copy             = coords_copy_cur.size();
        const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
            m_container.grids[i_iter].begin(), m_container.grids[i_iter].end(),
            [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
        m_container.state_shape[i_iter] = {.nchans       = nchans_cur,
                                           .ndt_min      = ndt_min_it->ndt,
                                           .ndt_max      = ndt_max_it->ndt,
                                           .ncoords      = ncoords,
                                           .ncoords_sum  = ncoords_sum,
                                           .ncoords_copy = ncoords_copy,
                                           .nsamps       = nsamps_iter,
                                           .nelements = ncoords * nsamps_iter,
                                           .dt_max    = dt_max_iter};
        m_container.dt_grid_sub_top[i_iter] =
            m_container.grids[i_iter].back().dt_grid;
    }
};

class CohFDMTPlan::Impl {
public:
    Impl(float f_center,
         float bw_sub,
         SizeType nsub,
         float tbin,
         SizeType nbin,
         SizeType nfft,
         float t_p,
         float dm_max,
         float dm_min                = 0.0F,
         SizeType noverlap           = 8192,
         std::string_view data_order = "PRITF",
         bool verbose                = false)
        : m_f_center(f_center),
          m_bw_sub(bw_sub),
          m_nsub(nsub),
          m_tbin(tbin),
          m_nbin(nbin),
          m_nfft(nfft),
          m_t_p(t_p),
          m_dm_max(dm_max),
          m_dm_min(dm_min),
          m_noverlap(noverlap),
          m_data_order(data_order),
          m_bw(m_bw_sub * static_cast<float>(m_nsub)),
          m_f_min(m_f_center - (m_bw / 2)),
          m_f_max(m_f_center + (m_bw / 2)) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        validate_inputs();
        configure_plan();
    }

    ~Impl() = default;
    Impl(const Impl& other)
        : m_f_center(other.m_f_center),
          m_bw_sub(other.m_bw_sub),
          m_nsub(other.m_nsub),
          m_tbin(other.m_tbin),
          m_nbin(other.m_nbin),
          m_nfft(other.m_nfft),
          m_t_p(other.m_t_p),
          m_dm_max(other.m_dm_max),
          m_dm_min(other.m_dm_min),
          m_noverlap(other.m_noverlap),
          m_data_order(other.m_data_order),
          m_bw(other.m_bw),
          m_f_min(other.m_f_min),
          m_f_max(other.m_f_max),
          m_n_p(other.m_n_p),
          m_nchan(other.m_nchan),
          m_dm_grid_coh(other.m_dm_grid_coh),
          m_dm_grid_final(other.m_dm_grid_final),
          m_nsamp(other.m_nsamp),
          m_mbin(other.m_mbin),
          m_mchan(other.m_mchan),
          m_msamp(other.m_msamp),
          m_tsamp(other.m_tsamp),
          m_dt_max(other.m_dt_max),
          m_fdmt_plan(other.m_fdmt_plan
                          ? std::make_unique<FDMTPlan>(*other.m_fdmt_plan)
                          : nullptr) {}
    Impl& operator=(const Impl& other) {
        if (this != &other) {
            m_f_center      = other.m_f_center;
            m_bw_sub        = other.m_bw_sub;
            m_nsub          = other.m_nsub;
            m_tbin          = other.m_tbin;
            m_nbin          = other.m_nbin;
            m_nfft          = other.m_nfft;
            m_t_p           = other.m_t_p;
            m_dm_max        = other.m_dm_max;
            m_dm_min        = other.m_dm_min;
            m_noverlap      = other.m_noverlap;
            m_data_order    = other.m_data_order;
            m_bw            = other.m_bw;
            m_f_min         = other.m_f_min;
            m_f_max         = other.m_f_max;
            m_n_p           = other.m_n_p;
            m_nchan         = other.m_nchan;
            m_dm_grid_coh   = other.m_dm_grid_coh;
            m_dm_grid_final = other.m_dm_grid_final;
            m_nsamp         = other.m_nsamp;
            m_mbin          = other.m_mbin;
            m_mchan         = other.m_mchan;
            m_msamp         = other.m_msamp;
            m_tsamp         = other.m_tsamp;
            m_dt_max        = other.m_dt_max;
            m_fdmt_plan = other.m_fdmt_plan
                              ? std::make_unique<FDMTPlan>(*other.m_fdmt_plan)
                              : nullptr;
        }
        return *this;
    }
    Impl(Impl&&)            = delete;
    Impl& operator=(Impl&&) = delete;

    // Getters
    float get_f_center() const noexcept { return m_f_center; }
    float get_bw_sub() const noexcept { return m_bw_sub; }
    SizeType get_nsub() const noexcept { return m_nsub; }
    float get_tbin() const noexcept { return m_tbin; }
    SizeType get_nbin() const noexcept { return m_nbin; }
    SizeType get_nfft() const noexcept { return m_nfft; }
    float get_t_p() const noexcept { return m_t_p; }
    float get_dm_max() const noexcept { return m_dm_max; }
    float get_dm_min() const noexcept { return m_dm_min; }
    SizeType get_noverlap() const noexcept { return m_noverlap; }
    std::string_view get_data_order() const noexcept { return m_data_order; }

    float get_bw() const noexcept { return m_bw; }
    float get_f_min() const noexcept { return m_f_min; }
    float get_f_max() const noexcept { return m_f_max; }
    SizeType get_n_p() const noexcept { return m_n_p; }
    SizeType get_nchan() const noexcept { return m_nchan; }
    [[nodiscard]] std::vector<float> get_dm_grid_coh() const noexcept {
        return m_dm_grid_coh;
    }
    [[nodiscard]] std::vector<float> get_dm_grid_final() const noexcept {
        return m_dm_grid_final;
    }
    SizeType get_nsamp() const noexcept { return m_nsamp; }
    SizeType get_mbin() const noexcept { return m_mbin; }
    SizeType get_mchan() const noexcept { return m_mchan; }
    SizeType get_msamp() const noexcept { return m_msamp; }
    float get_tsamp() const noexcept { return m_tsamp; }
    SizeType get_dt_max() const noexcept { return m_dt_max; }

    SizeType get_chirp_table_size() const noexcept {
        return m_dm_grid_coh.size() * m_nsub * m_nbin;
    }
    SizeType get_unpack_buf_size() const noexcept {
        return m_nfft * m_nsub * m_nbin;
    }
    SizeType get_delay_buf_size() const noexcept {
        return m_nfft * m_nsub * m_nbin;
    }
    SizeType get_intensity_buf_size() const noexcept {
        return m_nsub * m_nchan * m_msamp;
    }
    SizeType get_dmt_size() const {
        return m_dm_grid_coh.size() * m_fdmt_plan->get_dmt_size();
    }
    float get_chirp_scale() const noexcept {
        return 1.0F / static_cast<float>(m_nbin);
    }

    const FDMTPlan& get_fdmt_plan() const { return *m_fdmt_plan; }

    void print_summary() const {
        auto size_in_mb = [](SizeType count, SizeType size) {
            return static_cast<float>(count * size) / 1024.0F / 1024.0F;
        };
        const auto baseband_size =
            static_cast<SizeType>(2 * 2) * m_nsub * m_nsamp;
        const auto dmt_size = get_dmt_size();
        const auto buffer_size =
            2 * (get_unpack_buf_size() + get_delay_buf_size());
        const auto chirp_size     = get_chirp_table_size();
        const auto waterfall_size = get_intensity_buf_size();

        const auto baseband_size_mb =
            size_in_mb(baseband_size, sizeof(uint8_t));
        const auto dmt_size_mb = size_in_mb(dmt_size, sizeof(float));
        const auto buffer_size_mb =
            size_in_mb(buffer_size, sizeof(ComplexType));
        const auto chirp_size_mb = size_in_mb(chirp_size, sizeof(float));
        const auto waterfall_size_mb =
            size_in_mb(waterfall_size, sizeof(float));

        std::cout << std::format("\n*** CohFDMT Plan Summary ***\n");
        std::cout << std::format(
            "Input: Baseband Size: (2 x 2 x {} x {}; {:.1f} MB ), DMT "
            "Size: ({} x {}; {:.1f} MB )\n",
            m_nsub, m_nsamp, baseband_size_mb, m_dm_grid_final.size(),
            m_fdmt_plan->get_dmt_nsamps(), dmt_size_mb);
        std::cout << std::format(
            "Plan Buffer Size: ({}; {:.1f} MB ), Chirp: "
            "({}; {:.1f} MB ), Waterfall: ({}; {:.1f} MB )\n",
            buffer_size, buffer_size_mb, chirp_size, chirp_size_mb,
            waterfall_size, waterfall_size_mb);
        std::cout << std::format("Forward FFT-1D calls: {}(n={})\n",
                                 m_nfft * m_nsub, m_nbin);
        std::cout << std::format("Coherent DMs: {}\n", m_dm_grid_coh.size());
        std::cout << std::format("Per coherent call details ...\n");
        std::cout << std::format("\tBackward FFT-1D calls: {}(n={})\n",
                                 m_nfft * m_nsub * m_nchan, m_mbin);
        m_fdmt_plan->print_summary("\t");
        std::cout << std::format("{:*>80}\n", "");
    }

private:
    float m_f_center;
    float m_bw_sub;
    SizeType m_nsub;
    float m_tbin;
    SizeType m_nbin;
    SizeType m_nfft;
    float m_t_p;
    float m_dm_max;
    float m_dm_min;
    SizeType m_noverlap;
    std::string_view m_data_order;

    float m_bw;
    float m_f_min;
    float m_f_max;
    SizeType m_n_p{};
    SizeType m_nchan{};
    std::vector<float> m_dm_grid_coh;
    std::vector<float> m_dm_grid_final;
    SizeType m_nsamp{};
    SizeType m_mbin{};
    SizeType m_mchan{};
    SizeType m_msamp{};
    float m_tsamp{};
    SizeType m_dt_max{};

    std::unique_ptr<FDMTPlan> m_fdmt_plan;

    void validate_inputs() const {
        if (m_nsub < 1) {
            throw std::invalid_argument("nsub must be greater than 0");
        }
        if (m_bw_sub <= 0.0F) {
            throw std::invalid_argument("bwsub must be positive");
        }
        if (m_tbin <= 0.0F) {
            throw std::invalid_argument("tbin must be positive");
        }
        if (m_nbin == 0) {
            throw std::invalid_argument("nbin must be greater than 0");
        }
        if (m_nfft == 0) {
            throw std::invalid_argument("nfft must be greater than 0");
        }
        if (m_t_p <= 0.0F) {
            throw std::invalid_argument("t_p must be positive");
        }
        if (m_dm_max < m_dm_min) {
            throw std::invalid_argument(
                "dm_max must be greater than or equal to dm_min");
        }
        if (!kBasebandDataOrderMap.contains(m_data_order)) {
            auto kv = std::ranges::views::keys(kBasebandDataOrderMap);
            std::vector<std::string_view> keys(kv.begin(), kv.end());
            throw std::invalid_argument(
                fmt::format("Invalid data order: {}. Supported values are: {}",
                            m_data_order, fmt::join(keys, ", ")));
        }
    }

    void configure_plan() {
        // Calculate the number of samples corresponding to pulse width
        m_n_p   = static_cast<SizeType>(std::ceil(m_t_p / m_tbin));
        m_nchan = m_n_p;

        // Generate coherent DM grid
        m_dm_grid_coh = utils::generate_coherent_dms(
            m_dm_min, m_dm_max, m_f_center, m_bw, m_tbin, m_t_p);
        if (m_dm_grid_coh.empty()) {
            throw std::runtime_error("Empty DM grid");
        }

        // Compute optimal overlap size and adjust to the nearest power of two
        const auto max_dm           = *std::ranges::max_element(m_dm_grid_coh);
        const auto noverlap_optimal = utils::minimum_overlap(
            max_dm, m_f_center, m_bw, m_tbin, m_nsub, m_nchan);
        const auto noverlap_optimal_pow2 = static_cast<SizeType>(
            std::pow(2, std::nearbyint(std::log2(noverlap_optimal))));
        m_noverlap = std::max(m_noverlap, noverlap_optimal_pow2);

        // Validate nbin and noverlap
        if (m_nbin < 2 * m_noverlap) {
            throw std::invalid_argument(
                std::format("nbin must be greater than 2 * noverlap: "
                            "nbin={}, noverlap={}",
                            m_nbin, m_noverlap));
        }
        m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);

        // Calculate bins per channel
        if (m_nbin % m_nchan != 0) {
            throw std::runtime_error(
                std::format("nbin must be divisible by nchan: "
                            "nbin={}, nchan={}",
                            m_nbin, m_nchan));
        }
        m_mbin  = m_nbin / m_nchan;
        m_mchan = m_nsub * m_nchan;

        if (m_nsamp % m_nchan != 0) {
            throw std::runtime_error(
                std::format("nsamp must be divisible by nchan: "
                            "nsamp={}, nchan={}",
                            m_nsamp, m_nchan));
        }
        m_msamp  = m_nsamp / m_nchan;
        m_tsamp  = m_tbin * static_cast<float>(m_nchan);
        m_dt_max = m_n_p - 1;

        m_fdmt_plan =
            std::make_unique<FDMTPlan>(m_f_min, m_f_max, m_mchan, m_msamp,
                                       m_tsamp, m_dt_max, 0, 1, "full", false);

        // Generate final DM grid
        const auto& fdmt_dm_grid = m_fdmt_plan->get_dm_grid_final();
        m_dm_grid_final.resize(m_dm_grid_coh.size() * fdmt_dm_grid.size());
        for (SizeType i = 0; i < m_dm_grid_coh.size(); ++i) {
            const auto offset = static_cast<IndexType>(i * fdmt_dm_grid.size());
            std::ranges::transform(
                fdmt_dm_grid, m_dm_grid_final.begin() + offset,
                [dm_coh = m_dm_grid_coh[i]](auto& dm) { return dm + dm_coh; });
        }
    }
};

class DDMTPlan::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min = 0.0F,
         bool verbose = false)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_nchans(nchans),
          m_tsamp(tsamp),
          m_dm_arr(generate_dm_arr(dm_max, dm_step, dm_min)) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        validate_inputs();
        configure_plan();
        spdlog::debug("DDMT: dm_max={}, dm_min={}, dm_step={}", dm_max, dm_min,
                      dm_step);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         std::span<const float> dm_arr,
         bool verbose = false)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_nchans(nchans),
          m_tsamp(tsamp),
          m_dm_arr(dm_arr.begin(), dm_arr.end()) {
        if (verbose) {
            spdlog::set_level(spdlog::level::trace);
        } else {
            spdlog::set_level(spdlog::level::info);
        }
        validate_inputs();
        configure_plan();
        spdlog::debug("DDMT: dm_count={}", m_dm_arr.size());
    }

    ~Impl()                                = default;
    Impl(Impl&& other) noexcept            = default;
    Impl& operator=(Impl&& other) noexcept = default;
    Impl(const Impl& other)                = default;
    Impl& operator=(const Impl& other)     = default;

    float get_f_min() const noexcept { return m_f_min; }
    float get_f_max() const noexcept { return m_f_max; }
    SizeType get_nchans() const noexcept { return m_nchans; }
    float get_tsamp() const noexcept { return m_tsamp; }
    std::vector<float> get_dm_arr() const noexcept { return m_dm_arr; }
    const DDMTPlanContainer& get_container() const noexcept {
        return m_container;
    }
    [[nodiscard]] std::vector<float> get_dm_grid() const noexcept {
        return m_container.dm_arr;
    }

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    float m_tsamp;
    std::vector<float> m_dm_arr;

    DDMTPlanContainer m_container;

    void validate_inputs() const {
        if (m_f_min >= m_f_max) {
            throw std::invalid_argument("f_min must be less than f_max");
        }
        if (m_tsamp <= 0) {
            throw std::invalid_argument("tsamp must be greater than 0");
        }
        if (m_nchans <= 0) {
            throw std::invalid_argument("nchans must be greater than 0");
        }
        if (m_dm_arr.empty()) {
            throw std::invalid_argument("dm_arr must not be empty");
        }
    }
    void configure_plan() {
        m_container.nchans = m_nchans;
        m_container.dm_arr = m_dm_arr;
        const auto df      = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
        m_container.delay_table = utils::generate_delay_table(
            m_dm_arr, m_nchans, m_f_min, df, m_tsamp);
    }

    static std::vector<float>
    generate_dm_arr(float dm_max, float dm_step, float dm_min) {
        std::vector<float> dm_arr;
        for (float dm = dm_min; dm <= dm_max; dm += dm_step) {
            dm_arr.push_back(dm);
        }
        return dm_arr;
    }
};

// --- Definitions for FDMTPlan ---
FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   SizeType dt_max,
                   SizeType dt_min,
                   SizeType dt_step,
                   std::string_view mode,
                   bool verbose)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_min,
                                    dt_step,
                                    mode,
                                    verbose)) {}
FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<SizeType>& dt_grid,
                   std::string_view mode,
                   bool verbose)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, verbose)) {}

FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<float>& dm_grid,
                   std::string_view mode,
                   bool verbose)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, verbose)) {}

FDMTPlan::~FDMTPlan()                              = default;
FDMTPlan::FDMTPlan(FDMTPlan&&) noexcept            = default;
FDMTPlan& FDMTPlan::operator=(FDMTPlan&&) noexcept = default;
FDMTPlan::FDMTPlan(const FDMTPlan& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

FDMTPlan& FDMTPlan::operator=(const FDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
float FDMTPlan::get_f_min() const noexcept { return m_impl->get_f_min(); }
float FDMTPlan::get_f_max() const noexcept { return m_impl->get_f_max(); }
SizeType FDMTPlan::get_nchans() const noexcept { return m_impl->get_nchans(); }
SizeType FDMTPlan::get_nsamps() const noexcept { return m_impl->get_nsamps(); }
float FDMTPlan::get_tsamp() const noexcept { return m_impl->get_tsamp(); }
SizeType FDMTPlan::get_dt_max() const noexcept { return m_impl->get_dt_max(); }
SizeType FDMTPlan::get_dt_min() const noexcept { return m_impl->get_dt_min(); }
SizeType FDMTPlan::get_dt_step() const noexcept {
    return m_impl->get_dt_step();
}
bool FDMTPlan::is_custom_grid() const noexcept {
    return m_impl->is_custom_grid();
}
float FDMTPlan::get_df() const noexcept { return m_impl->get_df(); }
SizeType FDMTPlan::get_niters() const noexcept { return m_impl->get_niters(); }
const FDMTPlanContainer& FDMTPlan::get_container() const noexcept {
    return m_impl->get_container();
}
FDMTComplexity FDMTPlan::get_complexity() const noexcept {
    return m_impl->get_complexity();
}
void FDMTPlan::print_complexity_summary() const {
    m_impl->print_complexity_summary();
}
std::vector<SizeType> FDMTPlan::get_dt_grid_final() const noexcept {
    return m_impl->get_dt_grid_final();
}
std::vector<float> FDMTPlan::get_dm_grid_final() const noexcept {
    return m_impl->get_dm_grid_final();
}
std::vector<float> FDMTPlan::get_smearing_grid_final() const noexcept {
    return m_impl->get_smearing_grid_final();
}
SizeType FDMTPlan::get_dmt_ndms() const noexcept {
    return m_impl->get_dmt_ndms();
}
SizeType FDMTPlan::get_dmt_nsamps() const noexcept {
    return m_impl->get_dmt_nsamps();
}
SizeType FDMTPlan::get_dmt_size() const noexcept {
    return m_impl->get_dmt_size();
}
SizeType FDMTPlan::get_buffer_size() const noexcept {
    return m_impl->get_buffer_size();
}
SizeType FDMTPlan::get_history_size() const noexcept {
    return m_impl->get_history_size();
}
SizeType FDMTPlan::get_history_init_size() const noexcept {
    return m_impl->get_history_init_size();
}
void FDMTPlan::print_summary(std::string_view prefix) const {
    m_impl->print_summary(prefix);
}

// --- Definitions for CohFDMTPlan ---
CohFDMTPlan::CohFDMTPlan(float f_center,
                         float bw_sub,
                         SizeType nsub,
                         float tbin,
                         SizeType nbin,
                         SizeType nfft,
                         float t_p,
                         float dm_max,
                         float dm_min,
                         SizeType noverlap,
                         std::string_view data_order,
                         bool verbose)
    : m_impl(std::make_unique<Impl>(f_center,
                                    bw_sub,
                                    nsub,
                                    tbin,
                                    nbin,
                                    nfft,
                                    t_p,
                                    dm_max,
                                    dm_min,
                                    noverlap,
                                    data_order,
                                    verbose)) {}
CohFDMTPlan::~CohFDMTPlan()                                 = default;
CohFDMTPlan::CohFDMTPlan(CohFDMTPlan&&) noexcept            = default;
CohFDMTPlan& CohFDMTPlan::operator=(CohFDMTPlan&&) noexcept = default;
CohFDMTPlan::CohFDMTPlan(const CohFDMTPlan& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}
CohFDMTPlan& CohFDMTPlan::operator=(const CohFDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
float CohFDMTPlan::get_f_center() const noexcept {
    return m_impl->get_f_center();
}
float CohFDMTPlan::get_bw_sub() const noexcept { return m_impl->get_bw_sub(); }
SizeType CohFDMTPlan::get_nsub() const noexcept { return m_impl->get_nsub(); }
float CohFDMTPlan::get_tbin() const noexcept { return m_impl->get_tbin(); }
SizeType CohFDMTPlan::get_nbin() const noexcept { return m_impl->get_nbin(); }
SizeType CohFDMTPlan::get_nfft() const noexcept { return m_impl->get_nfft(); }
float CohFDMTPlan::get_t_p() const noexcept { return m_impl->get_t_p(); }
float CohFDMTPlan::get_dm_max() const noexcept { return m_impl->get_dm_max(); }
float CohFDMTPlan::get_dm_min() const noexcept { return m_impl->get_dm_min(); }
SizeType CohFDMTPlan::get_noverlap() const noexcept {
    return m_impl->get_noverlap();
}
std::string_view CohFDMTPlan::get_data_order() const noexcept {
    return m_impl->get_data_order();
}

float CohFDMTPlan::get_bw() const noexcept { return m_impl->get_bw(); }
float CohFDMTPlan::get_f_min() const noexcept { return m_impl->get_f_min(); }
float CohFDMTPlan::get_f_max() const noexcept { return m_impl->get_f_max(); }
SizeType CohFDMTPlan::get_n_p() const noexcept { return m_impl->get_n_p(); }
SizeType CohFDMTPlan::get_nchan() const noexcept { return m_impl->get_nchan(); }
std::vector<float> CohFDMTPlan::get_dm_grid_coh() const noexcept {
    return m_impl->get_dm_grid_coh();
}
std::vector<float> CohFDMTPlan::get_dm_grid_final() const noexcept {
    return m_impl->get_dm_grid_final();
}
SizeType CohFDMTPlan::get_nsamp() const noexcept { return m_impl->get_nsamp(); }
SizeType CohFDMTPlan::get_mbin() const noexcept { return m_impl->get_mbin(); }
SizeType CohFDMTPlan::get_mchan() const noexcept { return m_impl->get_mchan(); }
SizeType CohFDMTPlan::get_msamp() const noexcept { return m_impl->get_msamp(); }
float CohFDMTPlan::get_tsamp() const noexcept { return m_impl->get_tsamp(); }
SizeType CohFDMTPlan::get_dt_max() const noexcept {
    return m_impl->get_dt_max();
}
SizeType CohFDMTPlan::get_chirp_table_size() const noexcept {
    return m_impl->get_chirp_table_size();
}
SizeType CohFDMTPlan::get_unpack_buf_size() const noexcept {
    return m_impl->get_unpack_buf_size();
}
SizeType CohFDMTPlan::get_delay_buf_size() const noexcept {
    return m_impl->get_delay_buf_size();
}
SizeType CohFDMTPlan::get_intensity_buf_size() const noexcept {
    return m_impl->get_intensity_buf_size();
}
SizeType CohFDMTPlan::get_dmt_size() const { return m_impl->get_dmt_size(); }
float CohFDMTPlan::get_chirp_scale() const noexcept {
    return m_impl->get_chirp_scale();
}
const FDMTPlan& CohFDMTPlan::get_fdmt_plan() const {
    return m_impl->get_fdmt_plan();
}
void CohFDMTPlan::print_summary() const { m_impl->print_summary(); }

// --- Definitions for DDMTPlan ---
DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   float dm_max,
                   float dm_step,
                   float dm_min,
                   bool verbose)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min, verbose)) {}

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   std::span<const float> dm_arr,
                   bool verbose)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, dm_arr, verbose)) {}

DDMTPlan::~DDMTPlan()                              = default;
DDMTPlan::DDMTPlan(DDMTPlan&&) noexcept            = default;
DDMTPlan& DDMTPlan::operator=(DDMTPlan&&) noexcept = default;
DDMTPlan::DDMTPlan(const DDMTPlan& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

DDMTPlan& DDMTPlan::operator=(const DDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
float DDMTPlan::get_f_min() const noexcept { return m_impl->get_f_min(); }
float DDMTPlan::get_f_max() const noexcept { return m_impl->get_f_max(); }
SizeType DDMTPlan::get_nchans() const noexcept { return m_impl->get_nchans(); }
float DDMTPlan::get_tsamp() const noexcept { return m_impl->get_tsamp(); }
std::vector<float> DDMTPlan::get_dm_arr() const noexcept {
    return m_impl->get_dm_arr();
}
const DDMTPlanContainer& DDMTPlan::get_container() const noexcept {
    return m_impl->get_container();
}
std::vector<float> DDMTPlan::get_dm_grid() const noexcept {
    return m_impl->get_dm_grid();
}
} // namespace dmt::plans