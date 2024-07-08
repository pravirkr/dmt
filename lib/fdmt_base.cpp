#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include "dmt/dm_utils.hpp"
#include "dmt/dmt_types.hpp"
#include <dmt/fdmt_base.hpp>

SizeType FDMTPlan::get_memory_usage() const {
    SizeType mem_use = 0;
    mem_use += df_top.size() * sizeof(float);
    mem_use += df_bot.size() * sizeof(float);
    mem_use += state_shape.size() * sizeof(StShapeType);
    for (const auto& coord : coordinates) {
        mem_use += 2 * coord.size() * sizeof(FDMTCoord);
    }
    for (const auto& dt_grid_iter : dt_grids) {
        mem_use += dt_grid_iter.size() * sizeof(FDMTSubDTGrid);
    }
    for (const auto& dt_grid : dt_grid_sub_top) {
        mem_use += dt_grid.size() * sizeof(SizeType);
    }

    return mem_use;
}

SizeType FDMTPlan::get_buffer_size() const {
    auto max_elements = std::max_element(
        state_shape.begin(), state_shape.end(),
        [](const auto& a, const auto& b) { return a[5] < b[5]; });
    return (*max_elements)[5];
}

void FDMTPlan::print_summary() const {
    const auto mem_use_mb =
        static_cast<float>(get_memory_usage()) / 1024.0F / 1024.0F;
    const auto buffer_size = 2 * get_buffer_size();
    const auto buffer_size_mb =
        static_cast<float>(buffer_size) * sizeof(float) / 1024.0F / 1024.0F;
    const auto niters = state_shape.size() - 1;
    spdlog::info("FDMT: Plan memory usage: {:.3F} MB", mem_use_mb);
    spdlog::info("FDMT: Plan buffer size: {} elements, {:.3F} MB", buffer_size,
                 buffer_size_mb);
    spdlog::info("FDMT: waterfall_size: ({}x{}), dmt_size: ({}x{})",
                 state_shape[0][0], state_shape[0][4], state_shape[niters][3],
                 state_shape[niters][4]);
    spdlog::info("FDMT: Plan details: {ncoords} ({nchans}x"
                 "[{ndt_min}..{ndt_max}]) x {nsamps}, nelements: {nelements}");
    for (SizeType i_iter = 0; i_iter < niters + 1; ++i_iter) {
        const auto& [nchans, ndt_min, ndt_max, ncoords, nsamps, nelements] =
            state_shape[i_iter];
        spdlog::info(
            "FDMT: Iteration {}, dimensions: {} ({}x[{}..{}]) x {}, {}", i_iter,
            ncoords, nchans, ndt_min, ndt_max, nsamps, nelements);
    }
}

FDMT::FDMT(float f_min,
           float f_max,
           SizeType nchans,
           SizeType nsamps,
           float tsamp,
           SizeType dt_max,
           SizeType dt_step,
           SizeType dt_min)
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
    spdlog::debug("FDMT: df={}, dt_max={}, dt_min={}, dt_step={}, niters={}",
                  m_df, m_dt_max, m_dt_min, m_dt_step, m_niters);
}

// Getters
float FDMT::get_df() const { return m_df; }
float FDMT::get_correction() const { return m_correction; }
SizeType FDMT::get_niters() const { return m_niters; }
const FDMTPlan& FDMT::get_plan() const { return m_fdmt_plan; }
const DtGridType& FDMT::get_dt_grid_final() const {
    return m_fdmt_plan.dt_grids[m_niters][0].dt_grid;
}
std::vector<float> FDMT::get_dm_grid_final() const {
    const float dm_conv       = dm_utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    const auto& dt_grid_final = get_dt_grid_final();
    std::vector<float> dm_grid_final(dt_grid_final.size());
    std::transform(
        dt_grid_final.begin(), dt_grid_final.end(), dm_grid_final.begin(),
        [dm_conv](auto& dt) { return static_cast<float>(dt) * dm_conv; });
    return dm_grid_final;
}
SizeType FDMT::get_dmt_size() const {
    const auto& plan = get_plan();
    return plan.state_shape[m_niters][5];
}

// Setters
void FDMT::set_log_level(int level) {
    if (level < static_cast<int>(spdlog::level::trace) ||
        level > static_cast<int>(spdlog::level::off)) {
        spdlog::set_level(spdlog::level::info);
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

// Private methods
SizeType FDMT::calculate_niters(SizeType nchans) {
    return static_cast<SizeType>(std::ceil(std::log2(nchans)));
}

float FDMT::calculate_df(float f_min, float f_max, SizeType nchans) {
    return (f_max - f_min) / static_cast<float>(nchans);
}

DtGridType FDMT::calculate_dt_grid_sub(float f_start, float f_end) const {
    const auto dt_max_sub = static_cast<SizeType>(
        dm_utils::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_max));
    const auto dt_min_sub = static_cast<SizeType>(
        dm_utils::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_min));
    DtGridType dt_grid;
    for (SizeType dt = dt_min_sub; dt <= dt_max_sub; dt += m_dt_step) {
        dt_grid.push_back(dt);
    }
    return dt_grid;
}

void FDMT::check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
    if (waterfall_size != m_nchans * m_nsamps) {
        throw std::invalid_argument("Invalid size of waterfall");
    }
    const auto& plan = get_plan();
    if (dmt_size !=
        plan.state_shape[m_niters][3] * plan.state_shape[m_niters][4]) {
        throw std::invalid_argument("Invalid size of dmt");
    }
    spdlog::debug("FDMT: Input dimensions: {}x{}", m_nchans, m_nsamps);
}

void FDMT::configure_fdmt_plan() {
    // Allocate memory/size for plan members
    m_fdmt_plan.df_top.resize(m_niters + 1);
    m_fdmt_plan.df_bot.resize(m_niters + 1);
    m_fdmt_plan.state_shape.resize(m_niters + 1);
    m_fdmt_plan.coordinates.resize(m_niters + 1);
    m_fdmt_plan.coordinates_to_sum.resize(m_niters + 1);
    m_fdmt_plan.coordinates_to_copy.resize(m_niters + 1);
    m_fdmt_plan.dt_grids.resize(m_niters + 1);
    m_fdmt_plan.dt_grid_sub_top.resize(m_niters + 1);
    // For iteration 0
    make_fdmt_plan_iter0();
    // For iterations 1 to niters
    for (SizeType i_iter = 1; i_iter < m_niters + 1; ++i_iter) {
        make_fdmt_plan(i_iter);
    }
    spdlog::debug("FDMT: configured fdmt plan");
}

void FDMT::make_fdmt_plan_iter0() {
    // For iteration 0
    const SizeType i_iter  = 0;
    SizeType buffer_offset = 0;
    SizeType sub_offset    = 0;
    m_fdmt_plan.dt_grids[i_iter].resize(m_nchans);
    for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
        const auto f_start = m_df * static_cast<float>(i_sub) + m_f_min;
        const auto f_end   = f_start + m_df;
        const auto dt_sub  = calculate_dt_grid_sub(f_start, f_end);
        const auto ndt_sub = dt_sub.size();

        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto coord_cur =
                FDMTCoord{i_sub,    i_dt,     m_nsamps, buffer_offset,
                          SIZE_MAX, SIZE_MAX, SIZE_MAX};
            m_fdmt_plan.coordinates[i_iter].emplace_back(coord_cur);
            buffer_offset += m_nsamps;
        }
        m_fdmt_plan.dt_grids[i_iter][i_sub] =
            FDMTSubDTGrid{dt_sub, ndt_sub, sub_offset};
        sub_offset += ndt_sub;
    }
    m_fdmt_plan.df_top[i_iter] = m_df;
    m_fdmt_plan.df_bot[i_iter] = m_df;

    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_fdmt_plan.dt_grids[i_iter].begin(),
        m_fdmt_plan.dt_grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    const auto ncoords = m_fdmt_plan.dt_grids[i_iter][m_nchans - 1].sub_offset +
                         m_fdmt_plan.dt_grids[i_iter][m_nchans - 1].ndt;
    m_fdmt_plan.state_shape[i_iter] = {m_nchans,        ndt_min_it->ndt,
                                       ndt_max_it->ndt, ncoords,
                                       m_nsamps,        ncoords * m_nsamps};
    // 0th iteration has no mappings
    m_fdmt_plan.dt_grid_sub_top[i_iter] =
        m_fdmt_plan.dt_grids[i_iter][m_nchans - 1].dt_grid;
}

void FDMT::make_fdmt_plan(SizeType i_iter) {
    if (i_iter < 1 || i_iter > m_niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const auto& df_bot_prev          = m_fdmt_plan.df_bot[i_iter - 1];
    const auto& df_top_prev          = m_fdmt_plan.df_top[i_iter - 1];
    const auto& nchans_prev          = m_fdmt_plan.state_shape[i_iter - 1][0];
    const auto& nsamps_prev          = m_fdmt_plan.state_shape[i_iter - 1][4];
    const auto& dt_grid_sub_top_prev = m_fdmt_plan.dt_grid_sub_top[i_iter - 1];
    const auto& dt_grids_prev        = m_fdmt_plan.dt_grids[i_iter - 1];

    const SizeType nchans_cur = nchans_prev / 2 + nchans_prev % 2;
    const bool do_copy = nchans_prev % 2 == 1; // true if nchans_prev is odd
    const float df_top = (do_copy) ? df_top_prev : df_top_prev + df_bot_prev;
    const float df_bot = df_bot_prev * 2;

    // Calculate nsamps for the current iteration
    const auto df_tmp = (nchans_cur == 1) ? df_top : df_bot;
    const auto dt_max_iter =
        calculate_dt_grid_sub(m_f_min, m_f_min + df_tmp).back();
    if (dt_max_iter > m_dt_max) {
        throw std::runtime_error("dt_max_iter is greater than dt_max");
    }
    const auto nsamps_iter = m_nsamps + dt_max_iter;

    DtGridType dt_grid_sub_top = dt_grid_sub_top_prev;
    float f_end, f_mid;
    DtGridType dt_sub;
    SizeType buffer_offset = 0;
    SizeType sub_offset    = 0;
    m_fdmt_plan.dt_grids[i_iter].resize(nchans_cur);
    for (SizeType i_sub = 0; i_sub < nchans_cur; ++i_sub) {
        const auto& dt_grids_tail = dt_grids_prev[2 * i_sub];
        const auto& dt_grids_head = dt_grids_prev[2 * i_sub + 1];
        const auto f_start = df_bot * static_cast<float>(i_sub) + m_f_min;
        if (i_sub == nchans_cur - 1) {
            // For the top sub-band
            if (do_copy) {
                f_end  = f_start + df_top * 2;
                f_mid  = f_start + df_top;
                dt_sub = dt_grid_sub_top;
            } else {
                f_end           = f_start + df_top;
                f_mid           = f_start + df_bot / 2;
                dt_sub          = calculate_dt_grid_sub(f_start, f_end);
                dt_grid_sub_top = dt_sub;
            }
        } else {
            // For the bottom sub-bands
            f_end  = f_start + df_bot;
            f_mid  = f_start + df_bot / 2;
            dt_sub = calculate_dt_grid_sub(f_start, f_end);
        }
        const auto f_mid1  = f_mid - m_correction;
        const auto f_mid2  = f_mid + m_correction;
        const auto ndt_sub = dt_sub.size();

        // Populate the dt_plan mapping current dt grid to the previous dt grid
        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            // dt ~= dt_tail (dt_mid) + dt_head
            const auto dt      = dt_sub[i_dt];
            const auto dt_mid1 = static_cast<SizeType>(
                std::round(static_cast<float>(dt) *
                           dm_utils::cff(f_start, f_mid1, f_start, f_end)));
            const auto dt_mid2 = static_cast<SizeType>(
                std::round(static_cast<float>(dt) *
                           dm_utils::cff(f_start, f_mid2, f_start, f_end)));
            // check dt_head is always >= 0, otherwise throw error
            if (dt_mid1 > dt || dt_mid2 > dt) {
                throw std::runtime_error("Invalid dt_mid values");
            }
            if (dt_mid2 >= nsamps_prev) {
                throw std::runtime_error("Offset is greater than input size");
            }
            if (i_sub == nchans_cur - 1 && do_copy) {
                const auto i_dt_tail =
                    dm_utils::find_closest_index(dt_grids_tail.dt_grid, dt);
                const auto coord_cur =
                    FDMTCoord{i_sub,
                              i_dt,
                              nsamps_iter,
                              buffer_offset,
                              dt_grids_tail.sub_offset + i_dt_tail,
                              SIZE_MAX,
                              0};
                m_fdmt_plan.coordinates[i_iter].emplace_back(coord_cur);
                m_fdmt_plan.coordinates_to_copy[i_iter].emplace_back(coord_cur);
            } else {
                const auto dt_head   = dt - dt_mid2;
                const auto i_dt_tail = dm_utils::find_closest_index(
                    dt_grids_tail.dt_grid, dt_mid1);
                const auto i_dt_head = dm_utils::find_closest_index(
                    dt_grids_head.dt_grid, dt_head);
                const auto coord_cur =
                    FDMTCoord{i_sub,
                              i_dt,
                              nsamps_iter,
                              buffer_offset,
                              dt_grids_tail.sub_offset + i_dt_tail,
                              dt_grids_head.sub_offset + i_dt_head,
                              dt_mid2};
                m_fdmt_plan.coordinates[i_iter].emplace_back(coord_cur);
                m_fdmt_plan.coordinates_to_sum[i_iter].emplace_back(coord_cur);
            }
            buffer_offset += nsamps_iter;
        }
        m_fdmt_plan.dt_grids[i_iter][i_sub] =
            FDMTSubDTGrid{dt_sub, ndt_sub, sub_offset};
        sub_offset += ndt_sub;
    }
    m_fdmt_plan.df_top[i_iter] = df_top;
    m_fdmt_plan.df_bot[i_iter] = df_bot;

    // state shape summary
    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_fdmt_plan.dt_grids[i_iter].begin(),
        m_fdmt_plan.dt_grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    const auto ncoords =
        m_fdmt_plan.dt_grids[i_iter][nchans_cur - 1].sub_offset +
        m_fdmt_plan.dt_grids[i_iter][nchans_cur - 1].ndt;
    m_fdmt_plan.state_shape[i_iter]     = {nchans_cur,      ndt_min_it->ndt,
                                           ndt_max_it->ndt, ncoords,
                                           nsamps_iter,     ncoords * nsamps_iter};
    m_fdmt_plan.dt_grid_sub_top[i_iter] = dt_grid_sub_top;
}
