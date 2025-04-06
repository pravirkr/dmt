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

constexpr std::string_view FDMTShape::header_fmt() {
    return "{ncoords} ({nchans}x[{ndt_min}..{ndt_max}]) x "
           "{nsamps}, {nelements}";
}

std::string FDMTShape::to_string() const {
    return std::format("{:5d} ({:5d}x[{:5d}..{:5d}]) x {:6d}, {:10d}", ncoords,
                       nchans, ndt_min, ndt_max, nsamps, nelements);
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

FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   SizeType dt_max,
                   SizeType dt_step,
                   SizeType dt_min,
                   bool verbose)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dt_max(dt_max),
      m_dt_step(dt_step),
      m_dt_min(dt_min) {
    if (verbose) {
        spdlog::set_level(spdlog::level::trace);
    } else {
        spdlog::set_level(spdlog::level::info);
    }
    validate_inputs();
    configure_plan();
}

// Getters
float FDMTPlan::get_f_min() const noexcept { return m_f_min; }
float FDMTPlan::get_f_max() const noexcept { return m_f_max; }
SizeType FDMTPlan::get_nchans() const noexcept { return m_nchans; }
SizeType FDMTPlan::get_nsamps() const noexcept { return m_nsamps; }
float FDMTPlan::get_tsamp() const noexcept { return m_tsamp; }
SizeType FDMTPlan::get_dt_max() const noexcept { return m_dt_max; }
SizeType FDMTPlan::get_dt_step() const noexcept { return m_dt_step; }
SizeType FDMTPlan::get_dt_min() const noexcept { return m_dt_min; }
float FDMTPlan::get_df() const noexcept { return m_df; }
float FDMTPlan::get_correction() const noexcept { return m_correction; }
SizeType FDMTPlan::get_niters() const noexcept { return m_niters; }
const FDMTPlanContainer& FDMTPlan::get_container() const noexcept {
    return m_container;
}
const DtGridType& FDMTPlan::get_dt_grid_final() const noexcept {
    return m_container.grids[m_niters][0].dt_grid;
}
std::vector<float> FDMTPlan::get_dm_grid_final() const noexcept {
    const float dm_conv       = dm_utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    const auto& dt_grid_final = get_dt_grid_final();
    std::vector<float> dm_grid_final(dt_grid_final.size());
    std::ranges::transform(
        dt_grid_final, dm_grid_final.begin(),
        [dm_conv](auto& dt) { return static_cast<float>(dt) * dm_conv; });
    return dm_grid_final;
}
SizeType FDMTPlan::get_dmt_ndms() const noexcept {
    return m_container.state_shape[m_niters].ncoords;
}
SizeType FDMTPlan::get_dmt_nsamps() const noexcept {
    return m_container.state_shape[m_niters].nsamps;
}

SizeType FDMTPlan::get_dmt_size() const noexcept {
    return m_container.state_shape[m_niters].nelements;
}
SizeType FDMTPlan::get_buffer_size() const noexcept { return m_buffer_size; }

SizeType FDMTPlan::get_history_size() const noexcept {
    return m_nchans * m_container.state_shape[0].dt_max;
}

void FDMTPlan::print_summary(std::string_view prefix) const {
    auto size_in_mb = [](SizeType count, SizeType size) {
        return static_cast<float>(count * size) / 1024.0F / 1024.0F;
    };
    const auto& state_shape   = m_container.state_shape;
    const auto waterfall_size = state_shape[0].nchans * state_shape[0].nsamps;
    const auto dmt_size       = get_dmt_size();
    const auto history_size   = get_history_size();
    const auto buffer_size    = 2 * m_buffer_size;
    const auto niters         = state_shape.size() - 1;

    const auto waterfall_size_mb = size_in_mb(waterfall_size, sizeof(float));
    const auto dmt_size_mb       = size_in_mb(dmt_size, sizeof(float));
    const auto history_size_mb   = size_in_mb(history_size, sizeof(float));
    const auto buffer_size_mb    = size_in_mb(buffer_size, sizeof(float));
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
                    buffer_size, buffer_size_mb, history_size, history_size_mb),
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

// Private methods
void FDMTPlan::validate_inputs() const {
    if (m_f_min >= m_f_max) {
        throw std::invalid_argument("f_min must be less than f_max");
    }
    if (m_nchans == 0) {
        throw std::invalid_argument("nchans must be greater than 0");
    }
    if (m_nsamps == 0) {
        throw std::invalid_argument("nsamps must be greater than 0");
    }
    if (m_tsamp <= 0) {
        throw std::invalid_argument("tsamp must be greater than 0");
    }
    if (m_dt_max == 0) {
        throw std::invalid_argument("dt_max must be greater than 0");
    }
    if (m_dt_min >= m_dt_max) {
        throw std::invalid_argument("dt_min must be less than dt_max");
    }
    if (m_dt_step == 0) {
        throw std::invalid_argument("dt_step must be greater than 0");
    }
}

DtGridType FDMTPlan::calculate_dt_grid_sub(float f_start, float f_end) const {
    const auto dt_max_sub =
        dm_utils::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_max);
    const auto dt_min_sub =
        dm_utils::calculate_dt_sub(f_start, f_end, m_f_min, m_f_max, m_dt_min);
    DtGridType dt_grid;
    for (SizeType dt = dt_min_sub; dt <= dt_max_sub; dt += m_dt_step) {
        dt_grid.push_back(dt);
    }
    return dt_grid;
}

void FDMTPlan::configure_plan() {
    m_df         = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
    m_correction = m_df / 2;
    m_niters     = static_cast<SizeType>(std::ceil(std::log2(m_nchans)));
    m_container  = FDMTPlanContainer(m_niters);
    make_plan_iter0();
    // For iterations 1 to niters
    for (SizeType i_iter = 1; i_iter < m_niters + 1; ++i_iter) {
        make_plan(i_iter);
    }
    m_buffer_size = m_container.get_buffer_size();
    spdlog::debug("FDMT: configured fdmt plan");
    spdlog::debug("FDMT: df={}, dt_max={}, dt_min={}, dt_step={}, niters={}",
                  m_df, m_dt_max, m_dt_min, m_dt_step, m_niters);
}

void FDMTPlan::make_plan_iter0() {
    // For iteration 0
    const SizeType i_iter = 0;
    SizeType buf_offset   = 0;
    SizeType ncoords      = 0;
    m_container.grids[i_iter].resize(m_nchans);
    for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
        const auto f_start = (m_df * static_cast<float>(i_sub)) + m_f_min;
        const auto f_end   = f_start + m_df;
        const auto dt_sub  = calculate_dt_grid_sub(f_start, f_end);
        const auto ndt_sub = dt_sub.size();
        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto coord_cur = FDMTCoord{.i_sub           = i_sub,
                                             .i_dt            = i_dt,
                                             .nsamps          = m_nsamps,
                                             .buf_offset      = buf_offset,
                                             .i_coord_tail    = SIZE_MAX,
                                             .i_coord_head    = SIZE_MAX,
                                             .offset          = SIZE_MAX,
                                             .tail_buf_offset = SIZE_MAX,
                                             .tail_nsamps     = SIZE_MAX,
                                             .head_buf_offset = SIZE_MAX,
                                             .head_nsamps     = SIZE_MAX};
            m_container.coordinates[i_iter].emplace_back(coord_cur);
            buf_offset += m_nsamps;
        }
        m_container.grids[i_iter][i_sub] =
            FDMTCoordGrid{.dt_grid      = dt_sub,
                          .ndt          = ndt_sub,
                          .coord_offset = ncoords,
                          .f_start      = f_start,
                          .f_end        = f_end};
        ncoords += ndt_sub;
    }

    const auto dt_max = calculate_dt_grid_sub(m_f_min, m_f_min + m_df).back();
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
                                       .dt_max       = dt_max};
    m_container.dt_grid_sub_top[i_iter] =
        m_container.grids[i_iter].back().dt_grid;
    m_container.df_top[i_iter] = m_df;
    m_container.df_bot[i_iter] = m_df;
}

void FDMTPlan::make_plan(SizeType i_iter) {
    if (i_iter < 1 || i_iter > m_niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const auto& df_bot_prev = m_container.df_bot[i_iter - 1];
    const auto& df_top_prev = m_container.df_top[i_iter - 1];
    const auto& nchans_prev = m_container.state_shape[i_iter - 1].nchans;
    const auto& nsamps_prev = m_container.state_shape[i_iter - 1].nsamps;
    const auto& grids_prev  = m_container.grids[i_iter - 1];
    const auto& dt_grid_sub_top_prev = m_container.dt_grid_sub_top[i_iter - 1];

    const SizeType nchans_cur = (nchans_prev / 2) + (nchans_prev % 2);
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
    SizeType buf_offset = 0;
    SizeType ncoords    = 0;
    m_container.grids[i_iter].resize(nchans_cur);
    for (SizeType i_sub = 0; i_sub < nchans_cur; ++i_sub) {
        const auto& grids_tail = grids_prev[2 * i_sub];
        const auto& grids_head = grids_prev[(2 * i_sub) + 1];
        const auto f_start     = (df_bot * static_cast<float>(i_sub)) + m_f_min;
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
            const auto dt = dt_sub[i_dt];
            const auto dt_mid1 =
                dm_utils::calculate_dt_sub(f_start, f_mid1, f_start, f_end, dt);
            const auto dt_mid2 =
                dm_utils::calculate_dt_sub(f_start, f_mid2, f_start, f_end, dt);
            // check dt_head is always >= 0, otherwise throw error
            if (dt_mid1 > dt || dt_mid2 > dt) {
                throw std::runtime_error("Invalid dt_mid values");
            }
            if (dt_mid2 >= nsamps_prev) {
                throw std::runtime_error("Offset is greater than input size");
            }
            if (i_sub == nchans_cur - 1 && do_copy) {
                const auto i_dt_tail =
                    dm_utils::find_closest_index(grids_tail.dt_grid, dt);
                const auto i_coord_tail = grids_tail.coord_offset + i_dt_tail;
                const auto coord_cur    = FDMTCoord{
                       .i_sub        = i_sub,
                       .i_dt         = i_dt,
                       .nsamps       = nsamps_iter,
                       .buf_offset   = buf_offset,
                       .i_coord_tail = i_coord_tail,
                       .i_coord_head = SIZE_MAX,
                       .offset       = 0,
                       .tail_buf_offset =
                        m_container.coordinates[i_iter - 1][i_coord_tail]
                            .buf_offset,
                       .tail_nsamps =
                        m_container.coordinates[i_iter - 1][i_coord_tail]
                            .nsamps,
                       .head_buf_offset = SIZE_MAX,
                       .head_nsamps     = SIZE_MAX};

                m_container.coordinates[i_iter].emplace_back(coord_cur);
                m_container.coordinates_copy[i_iter].emplace_back(coord_cur);
            } else {
                const auto dt_head = dt - dt_mid2;
                const auto i_dt_tail =
                    dm_utils::find_closest_index(grids_tail.dt_grid, dt_mid1);
                const auto i_dt_head =
                    dm_utils::find_closest_index(grids_head.dt_grid, dt_head);
                const auto i_coord_tail = grids_tail.coord_offset + i_dt_tail;
                const auto i_coord_head = grids_head.coord_offset + i_dt_head;
                const auto coord_cur    = FDMTCoord{
                       .i_sub        = i_sub,
                       .i_dt         = i_dt,
                       .nsamps       = nsamps_iter,
                       .buf_offset   = buf_offset,
                       .i_coord_tail = i_coord_tail,
                       .i_coord_head = i_coord_head,
                       .offset       = dt_mid2,
                       .tail_buf_offset =
                        m_container.coordinates[i_iter - 1][i_coord_tail]
                            .buf_offset,
                       .tail_nsamps =
                        m_container.coordinates[i_iter - 1][i_coord_tail]
                            .nsamps,
                       .head_buf_offset =
                        m_container.coordinates[i_iter - 1][i_coord_head]
                            .buf_offset,
                       .head_nsamps =
                        m_container.coordinates[i_iter - 1][i_coord_head]
                            .nsamps};
                m_container.coordinates[i_iter].emplace_back(coord_cur);
                m_container.coordinates_sum[i_iter].emplace_back(coord_cur);
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
    // state shape summary
    const auto ncoords_sum  = m_container.coordinates_sum[i_iter].size();
    const auto ncoords_copy = m_container.coordinates_copy[i_iter].size();
    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_container.grids[i_iter].begin(), m_container.grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    m_container.state_shape[i_iter]     = {.nchans       = nchans_cur,
                                           .ndt_min      = ndt_min_it->ndt,
                                           .ndt_max      = ndt_max_it->ndt,
                                           .ncoords      = ncoords,
                                           .ncoords_sum  = ncoords_sum,
                                           .ncoords_copy = ncoords_copy,
                                           .nsamps       = nsamps_iter,
                                           .nelements    = ncoords * nsamps_iter,
                                           .dt_max       = dt_max_iter};
    m_container.dt_grid_sub_top[i_iter] = dt_grid_sub_top;
    m_container.df_top[i_iter]          = df_top;
    m_container.df_bot[i_iter]          = df_bot;
}

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

// Getters
float CohFDMTPlan::get_f_center() const noexcept { return m_f_center; }
float CohFDMTPlan::get_bw_sub() const noexcept { return m_bw_sub; }
SizeType CohFDMTPlan::get_nsub() const noexcept { return m_nsub; }
float CohFDMTPlan::get_tbin() const noexcept { return m_tbin; }
SizeType CohFDMTPlan::get_nbin() const noexcept { return m_nbin; }
SizeType CohFDMTPlan::get_nfft() const noexcept { return m_nfft; }
float CohFDMTPlan::get_t_p() const noexcept { return m_t_p; }
float CohFDMTPlan::get_dm_max() const noexcept { return m_dm_max; }
float CohFDMTPlan::get_dm_min() const noexcept { return m_dm_min; }
SizeType CohFDMTPlan::get_noverlap() const noexcept { return m_noverlap; }
std::string_view CohFDMTPlan::get_data_order() const noexcept {
    return m_data_order;
}

float CohFDMTPlan::get_bw() const noexcept { return m_bw; }
float CohFDMTPlan::get_f_min() const noexcept { return m_f_min; }
float CohFDMTPlan::get_f_max() const noexcept { return m_f_max; }
SizeType CohFDMTPlan::get_n_p() const noexcept { return m_n_p; }
SizeType CohFDMTPlan::get_nchan() const noexcept { return m_nchan; }
const std::vector<float>& CohFDMTPlan::get_dm_grid_coh() const noexcept {
    return m_dm_grid_coh;
}
const std::vector<float>& CohFDMTPlan::get_dm_grid_final() const noexcept {
    return m_dm_grid_final;
}
SizeType CohFDMTPlan::get_nsamp() const noexcept { return m_nsamp; }
SizeType CohFDMTPlan::get_mbin() const noexcept { return m_mbin; }
SizeType CohFDMTPlan::get_mchan() const noexcept { return m_mchan; }
SizeType CohFDMTPlan::get_msamp() const noexcept { return m_msamp; }
float CohFDMTPlan::get_tsamp() const noexcept { return m_tsamp; }
SizeType CohFDMTPlan::get_dt_max() const noexcept { return m_dt_max; }

SizeType CohFDMTPlan::get_chirp_table_size() const noexcept {
    return m_dm_grid_coh.size() * m_nsub * m_nbin;
}
SizeType CohFDMTPlan::get_unpack_buf_size() const noexcept {
    return m_nfft * m_nsub * m_nbin;
}
SizeType CohFDMTPlan::get_delay_buf_size() const noexcept {
    return m_nfft * m_nsub * m_nbin;
}
SizeType CohFDMTPlan::get_intensity_buf_size() const noexcept {
    return m_nsub * m_nchan * m_msamp;
}
SizeType CohFDMTPlan::get_dmt_size() const {
    return m_dm_grid_coh.size() * m_fdmt_plan->get_dmt_size();
}
float CohFDMTPlan::get_chirp_scale() const noexcept {
    return 1.0F / static_cast<float>(m_nbin);
}

const FDMTPlan& CohFDMTPlan::get_fdmt_plan() const { return *m_fdmt_plan; }

void CohFDMTPlan::validate_inputs() const {
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

void CohFDMTPlan::print_summary() const {
    auto size_in_mb = [](SizeType count, SizeType size) {
        return static_cast<float>(count * size) / 1024.0F / 1024.0F;
    };
    const auto baseband_size = static_cast<SizeType>(2 * 2) * m_nsub * m_nsamp;
    const auto dmt_size      = get_dmt_size();
    const auto buffer_size = 2 * (get_unpack_buf_size() + get_delay_buf_size());
    const auto chirp_size  = get_chirp_table_size();
    const auto waterfall_size = get_intensity_buf_size();

    const auto baseband_size_mb  = size_in_mb(baseband_size, sizeof(uint8_t));
    const auto dmt_size_mb       = size_in_mb(dmt_size, sizeof(float));
    const auto buffer_size_mb    = size_in_mb(buffer_size, sizeof(ComplexType));
    const auto chirp_size_mb     = size_in_mb(chirp_size, sizeof(float));
    const auto waterfall_size_mb = size_in_mb(waterfall_size, sizeof(float));

    std::cout << std::format("\n*** CohFDMT Plan Summary ***\n");
    std::cout << std::format(
        "Input: Baseband Size: (2 x 2 x {} x {}; {:.1f} MB ), DMT "
        "Size: ({} x {}; {:.1f} MB )\n",
        m_nsub, m_nsamp, baseband_size_mb, m_dm_grid_final.size(),
        m_fdmt_plan->get_dmt_nsamps(), dmt_size_mb);
    std::cout << std::format("Plan Buffer Size: ({}; {:.1f} MB ), Chirp: "
                             "({}; {:.1f} MB ), Waterfall: ({}; {:.1f} MB )\n",
                             buffer_size, buffer_size_mb, chirp_size,
                             chirp_size_mb, waterfall_size, waterfall_size_mb);
    std::cout << std::format("Forward FFT-1D calls: {}(n={})\n",
                             m_nfft * m_nsub, m_nbin);
    std::cout << std::format("Coherent DMs: {}\n", m_dm_grid_coh.size());
    std::cout << std::format("Per coherent call details ...\n");
    std::cout << std::format("\tBackward FFT-1D calls: {}(n={})\n",
                             m_nfft * m_nsub * m_nchan, m_mbin);
    m_fdmt_plan->print_summary("\t");
    std::cout << std::format("{:*>80}\n", "");
}

void CohFDMTPlan::configure_plan() {
    // Calculate the number of samples corresponding to pulse width
    m_n_p   = static_cast<SizeType>(std::ceil(m_t_p / m_tbin));
    m_nchan = m_n_p;

    // Generate coherent DM grid
    m_dm_grid_coh = dm_utils::generate_coherent_dms(
        m_dm_min, m_dm_max, m_f_center, m_bw, m_tbin, m_t_p);
    if (m_dm_grid_coh.empty()) {
        throw std::runtime_error("Empty DM grid");
    }

    // Compute optimal overlap size and adjust to the nearest power of two
    const auto max_dm           = *std::ranges::max_element(m_dm_grid_coh);
    const auto noverlap_optimal = dm_utils::minimum_overlap(
        max_dm, m_f_center, m_bw, m_tbin, m_nsub, m_nchan);
    const auto noverlap_optimal_pow2 = static_cast<SizeType>(
        std::pow(2, std::round(std::log2(noverlap_optimal))));
    m_noverlap = std::max(m_noverlap, noverlap_optimal_pow2);

    // Validate nbin and noverlap
    if (m_nbin < 2 * m_noverlap) {
        throw std::invalid_argument("nbin must be greater than 2 * noverlap");
    }
    m_nsamp = m_nfft * (m_nbin - 2 * m_noverlap);

    // Calculate bins per channel
    if (m_nbin % m_nchan != 0) {
        throw std::runtime_error("nbin must be divisible by nchan");
    }
    m_mbin  = m_nbin / m_nchan;
    m_mchan = m_nsub * m_nchan;

    if (m_nsamp % m_nchan != 0) {
        throw std::runtime_error("nsamp must be divisible by nchan");
    }
    m_msamp  = m_nsamp / m_nchan;
    m_tsamp  = m_tbin * static_cast<float>(m_nchan);
    m_dt_max = m_n_p - 1;

    m_fdmt_plan = std::make_unique<FDMTPlan>(m_f_min, m_f_max, m_mchan, m_msamp,
                                             m_tsamp, m_dt_max, 1, 0, false);

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

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   float dm_max,
                   float dm_step,
                   float dm_min)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_tsamp(tsamp),
      m_dm_arr(generate_dm_arr(dm_max, dm_step, dm_min)) {
    validate_inputs();
    configure_plan();
    spdlog::debug("DDMT: dm_max={}, dm_min={}, dm_step={}", dm_max, dm_min,
                  dm_step);
}

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   const std::vector<float>& dm_arr)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_tsamp(tsamp),
      m_dm_arr(dm_arr) {
    validate_inputs();
    configure_plan();
    spdlog::debug("DDMT: dm_count={}", m_dm_arr.size());
}

// Getters
float DDMTPlan::get_f_min() const noexcept { return m_f_min; }
float DDMTPlan::get_f_max() const noexcept { return m_f_max; }
SizeType DDMTPlan::get_nchans() const noexcept { return m_nchans; }
float DDMTPlan::get_tsamp() const noexcept { return m_tsamp; }
const std::vector<float>& DDMTPlan::get_dm_arr() const noexcept {
    return m_dm_arr;
}
const DDMTPlanContainer& DDMTPlan::get_container() const noexcept {
    return m_container;
}

std::vector<float> DDMTPlan::get_dm_grid() const noexcept {
    return m_container.dm_arr;
}

void DDMTPlan::set_log_level(int level) {
    if (level < static_cast<int>(spdlog::level::trace) ||
        level > static_cast<int>(spdlog::level::off)) {
        spdlog::set_level(spdlog::level::info);
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

void DDMTPlan::configure_plan() {
    m_container.nchans = m_nchans;
    m_container.dm_arr = m_dm_arr;
    const auto df      = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
    m_container.delay_table = dm_utils::generate_delay_table(
        m_dm_arr.data(), m_dm_arr.size(), m_f_min, df, m_nchans, m_tsamp);
}

void DDMTPlan::validate_inputs() const {
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

std::vector<float>
DDMTPlan::generate_dm_arr(float dm_max, float dm_step, float dm_min) {
    std::vector<float> dm_arr;
    for (float dm = dm_min; dm <= dm_max; dm += dm_step) {
        dm_arr.push_back(dm);
    }
    return dm_arr;
}