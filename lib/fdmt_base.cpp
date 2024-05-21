#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
//#include <spdlog/spdlog.h>
#include <stdexcept>  // For std::invalid_argument
#include "fdmt_utils.hpp"
#include <fdmt_base.hpp>

size_t FDMTPlan::calculate_memory_usage() const {
    size_t mem_use = 0;
    mem_use += df_top.size() * sizeof(float);
    mem_use += df_bot.size() * sizeof(float);
    mem_use += state_shape.size() * sizeof(StShapeType);
    for (const auto& coord : coordinates) {
        mem_use += coord.size() * sizeof(FDMTCoordType);
    }
    for (const auto& mapping : mappings) {
        mem_use += mapping.size() * sizeof(FDMTCoordMapping);
    }
    for (const auto& state_sub_idx : state_sub_idx) {
        mem_use += state_sub_idx.size() * sizeof(SizeType);
    }
    for (const auto& dt_grid_iter : dt_grid) {
        for (const auto& dt_grid : dt_grid_iter) {
            mem_use += dt_grid.size() * sizeof(SizeType);
        }
    }
    for (const auto& dt_grid : dt_grid_sub_top) {
        mem_use += dt_grid.size() * sizeof(SizeType);
    }

    return mem_use;
}

FDMT::FDMT(float f_min, float f_max, SizeType nchans, SizeType nsamps,
           float tsamp, SizeType dt_max, SizeType dt_step, SizeType dt_min)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dt_max(dt_max),
      m_dt_step(dt_step),
      m_dt_min(dt_min),
      m_df(calculate_df(m_f_min, m_f_max, m_nchans)),
      m_correction(m_df / 2),
      m_niters(calculate_niters(m_nchans)) {
    configure_fdmt_plan();
   /* spdlog::debug("FDMT: df={}, dt_max={}, dt_min={}, dt_step={}, niters={}",
                  m_df, m_dt_max, m_dt_min, m_dt_step, m_niters);*/
}

// Getters
float FDMT::get_df() const { return m_df; }
float FDMT::get_correction() const { return m_correction; }
int FDMT::get_m_nsamps() const { return static_cast<int>(m_nsamps); };

SizeType FDMT::get_niters() const { return m_niters; }
const FDMTPlan& FDMT::get_plan() const { return m_fdmt_plan; }
const DtGridType& FDMT::get_dt_grid_final() const {
    return m_fdmt_plan.dt_grid[m_niters][0];
}
std::vector<float> FDMT::get_dm_grid_final() const {
    const float dm_conv       = kDispConst * (std::pow(m_f_min, kDispCoeff) -
                                        std::pow(m_f_max, kDispCoeff));
    const float dm_step       = m_tsamp / dm_conv;
    const auto& dt_grid_final = get_dt_grid_final();
    std::vector<float> dm_grid_final(dt_grid_final.size());
    std::transform(
        dt_grid_final.begin(), dt_grid_final.end(), dm_grid_final.begin(),
        [dm_step](auto& dt) { return static_cast<float>(dt) * dm_step; });
    return dm_grid_final;
}

// Setters
//void FDMT::set_log_level(int level) {
//    if (level < static_cast<int>(spdlog::level::trace) ||
//        level > static_cast<int>(spdlog::level::off)) {
//        spdlog::set_level(spdlog::level::info);
//    }
//    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
//}

// Private methods
SizeType FDMT::calculate_niters(SizeType nchans) {
    return static_cast<SizeType>(std::ceil(std::log2(nchans)));
}

float FDMT::calculate_df(float f_min, float f_max, SizeType nchans) {
    return (f_max - f_min) / static_cast<float>(nchans);
}

DtGridType FDMT::calculate_dt_grid_sub(float f_start, float f_end) const {
    const auto dt_max_sub = static_cast<SizeType>(
        fdmt::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_max));
    const auto dt_min_sub = static_cast<SizeType>(
        fdmt::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_min));
    DtGridType dt_grid;
    for (SizeType dt = dt_min_sub; dt <= dt_max_sub; dt += m_dt_step) {
        dt_grid.push_back(dt);
    }
    return dt_grid;
}

void FDMT::check_inputs(size_t waterfall_size, size_t dmt_size) const {
    if (waterfall_size != m_nchans * m_nsamps) {
        throw std::invalid_argument("Invalid size of waterfall");
    }
    const auto& plan = get_plan();
    if (dmt_size !=
        plan.state_shape[m_niters][3] * plan.state_shape[m_niters][4]) {
        throw std::invalid_argument("Invalid size of dmt");
    }
   // spdlog::debug("FDMT: Input dimensions: {}x{}", m_nchans, m_nsamps);
}

void FDMT::configure_fdmt_plan() {
    // Allocate memory/size for plan members
    m_fdmt_plan.df_top.resize(m_niters + 1);
    m_fdmt_plan.df_bot.resize(m_niters + 1);
    m_fdmt_plan.state_shape.resize(m_niters + 1);
    m_fdmt_plan.coordinates.resize(m_niters + 1);
    m_fdmt_plan.coordinates_to_copy.resize(m_niters + 1);
    m_fdmt_plan.mappings.resize(m_niters + 1);
    m_fdmt_plan.mappings_to_copy.resize(m_niters + 1);
    m_fdmt_plan.state_sub_idx.resize(m_niters + 1);
    m_fdmt_plan.dt_grid.resize(m_niters + 1);
    m_fdmt_plan.dt_grid_sub_top.resize(m_niters + 1);
    // For iteration 0
    make_fdmt_plan_iter0();
    // For iterations 1 to niters
    for (SizeType i_iter = 1; i_iter < m_niters + 1; ++i_iter) {
        make_fdmt_plan(i_iter);
    }
   // spdlog::debug("FDMT: configured fdmt plan");
}

void FDMT::make_fdmt_plan_iter0() {
    // For iteration 0
    float f_start, f_end;
    DtGridType dt_grid_sub, nchans_ndt_size(m_nchans);
    SizeType state_idx = 0;

    m_fdmt_plan.state_sub_idx[0].resize(m_nchans);
    m_fdmt_plan.dt_grid[0].resize(m_nchans);
    for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
        f_start     = m_df * static_cast<float>(i_sub) + m_f_min;
        f_end       = f_start + m_df;
        dt_grid_sub = calculate_dt_grid_sub(f_start, f_end);

        for (SizeType i_dt = 0; i_dt < dt_grid_sub.size(); ++i_dt) {
            m_fdmt_plan.coordinates[0].emplace_back(i_sub, i_dt);
        }

        m_fdmt_plan.state_sub_idx[0][i_sub] = state_idx;
        m_fdmt_plan.dt_grid[0][i_sub]       = dt_grid_sub;
        nchans_ndt_size[i_sub]              = dt_grid_sub.size();
        state_idx += dt_grid_sub.size() * m_nsamps;
    }
    m_fdmt_plan.df_top[0]      = m_df;
    m_fdmt_plan.df_bot[0]      = m_df;
    m_fdmt_plan.state_shape[0] = {
        m_nchans,
        *std::min_element(nchans_ndt_size.begin(), nchans_ndt_size.end()),
        *std::max_element(nchans_ndt_size.begin(), nchans_ndt_size.end()),
        std::accumulate(nchans_ndt_size.begin(), nchans_ndt_size.end(),
                        static_cast<size_t>(0)),
        m_nsamps};
    // 0th iteration has no mappings
    m_fdmt_plan.dt_grid_sub_top[0] = m_fdmt_plan.dt_grid[0][m_nchans - 1];
}

void FDMT::make_fdmt_plan(SizeType i_iter) {
    if (i_iter < 1 || i_iter > m_niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const auto& df_bot_prev          = m_fdmt_plan.df_bot[i_iter - 1];
    const auto& df_top_prev          = m_fdmt_plan.df_top[i_iter - 1];
    const auto& nchans_prev          = m_fdmt_plan.state_shape[i_iter - 1][0];
    const auto& dt_grid_sub_top_prev = m_fdmt_plan.dt_grid_sub_top[i_iter - 1];
    const auto& dt_grid_prev         = m_fdmt_plan.dt_grid[i_iter - 1];

    const SizeType nchans_cur = nchans_prev / 2 + nchans_prev % 2;
    const bool do_copy = nchans_prev % 2 == 1; // true if nchans_prev is odd
    const float df_top = (do_copy) ? df_top_prev : df_top_prev + df_bot_prev;
    const float df_bot = df_bot_prev * 2;
    DtGridType dt_grid_sub_top = dt_grid_sub_top_prev;

    float f_start, f_end, f_mid, f_mid1, f_mid2;
    DtGridType dt_grid_sub, nchans_ndt_size(nchans_cur);
    SizeType i_sub_head, i_sub_tail, state_idx{0};

    m_fdmt_plan.state_sub_idx[i_iter].resize(nchans_cur);
    m_fdmt_plan.dt_grid[i_iter].resize(nchans_cur);
    for (size_t i_sub = 0; i_sub < nchans_cur; ++i_sub) {
        i_sub_tail = 2 * i_sub;
        i_sub_head = 2 * i_sub + 1;
        f_start    = df_bot * static_cast<float>(i_sub) + m_f_min;
        if (i_sub == nchans_cur - 1) {
            // For the top sub-band
            if (do_copy) {
                f_end       = f_start + df_top * 2;
                f_mid       = f_start + df_top;
                dt_grid_sub = dt_grid_sub_top;
            } else {
                f_end           = f_start + df_top;
                f_mid           = f_start + df_bot / 2;
                dt_grid_sub     = calculate_dt_grid_sub(f_start, f_end);
                dt_grid_sub_top = dt_grid_sub;
            }
        } else {
            // For the bottom sub-bands
            f_end       = f_start + df_bot;
            f_mid       = f_start + df_bot / 2;
            dt_grid_sub = calculate_dt_grid_sub(f_start, f_end);
        }
        f_mid1 = f_mid - m_correction;
        f_mid2 = f_mid + m_correction;

        // Populate the dt_plan mapping current dt grid to the previous dt grid
        size_t dt, dt_mid1, dt_mid2, dt_head, i_dt_head, i_dt_tail, offset;
        FDMTCoordMapping coord_mapping;
        for (size_t i_dt = 0; i_dt < dt_grid_sub.size(); ++i_dt) {
            // dt ~= dt_tail (dt_mid) + dt_head
            dt      = dt_grid_sub[i_dt];
            dt_mid1 = static_cast<size_t>(
                std::round(static_cast<float>(dt) *
                           fdmt::cff(f_start, f_mid1, f_start, f_end)));
            dt_mid2 = static_cast<size_t>(
                std::round(static_cast<float>(dt) *
                           fdmt::cff(f_start, f_mid2, f_start, f_end)));
            // check dt_head is always >= 0, otherwise throw error
            if (dt_mid1 > dt || dt_mid2 > dt) {
                throw std::runtime_error("Invalid dt_mid values");
            }
            dt_head = dt - dt_mid2;
            if (i_sub == nchans_cur - 1 && do_copy) {
                i_dt_tail =
                    fdmt::find_closest_index(dt_grid_prev[i_sub_tail], dt);
                m_fdmt_plan.coordinates_to_copy[i_iter].emplace_back(i_sub,
                                                                     i_dt);
                coord_mapping = {FDMTCoordType{SIZE_MAX, SIZE_MAX},
                                 FDMTCoordType{i_sub_tail, i_dt_tail}, 0};
                m_fdmt_plan.mappings_to_copy[i_iter].emplace_back(
                    coord_mapping);
            } else {
                i_dt_head =
                    fdmt::find_closest_index(dt_grid_prev[i_sub_head], dt_head);
                i_dt_tail =
                    fdmt::find_closest_index(dt_grid_prev[i_sub_tail], dt_mid1);
                offset = dt_mid2;
                if (offset >= m_nsamps) {
                    throw std::runtime_error(
                        "Offset is greater than input size");
                }
                m_fdmt_plan.coordinates[i_iter].emplace_back(i_sub, i_dt);
                coord_mapping = {FDMTCoordType{i_sub_head, i_dt_head},
                                 FDMTCoordType{i_sub_tail, i_dt_tail}, offset};
                m_fdmt_plan.mappings[i_iter].emplace_back(coord_mapping);
            }
        }
        m_fdmt_plan.state_sub_idx[i_iter][i_sub] = state_idx;
        m_fdmt_plan.dt_grid[i_iter][i_sub]       = dt_grid_sub;
        nchans_ndt_size[i_sub]                   = dt_grid_sub.size();
        state_idx += dt_grid_sub.size() * m_nsamps;
    }
    m_fdmt_plan.df_top[i_iter]      = df_top;
    m_fdmt_plan.df_bot[i_iter]      = df_bot;
    m_fdmt_plan.state_shape[i_iter] = {
        nchans_cur,
        *std::min_element(nchans_ndt_size.begin(), nchans_ndt_size.end()),
        *std::max_element(nchans_ndt_size.begin(), nchans_ndt_size.end()),
        std::accumulate(nchans_ndt_size.begin(), nchans_ndt_size.end(),
                        static_cast<size_t>(0)),
        m_nsamps};
    m_fdmt_plan.dt_grid_sub_top[i_iter] = dt_grid_sub_top;
}
