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
#include <dmt/dmt_plans.hpp>

SizeType FDMTPlanContainer::get_memory_usage() const noexcept {
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

FDMTPlan::FDMTPlan(float f_min,
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
      m_dt_min(dt_min) {
    validate_inputs();
    init_plan();
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
    return m_container.dt_grids[m_niters][0].dt_grid;
}
std::vector<float> FDMTPlan::get_dm_grid_final() const noexcept {
    const float dm_conv       = dm_utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    const auto& dt_grid_final = get_dt_grid_final();
    std::vector<float> dm_grid_final(dt_grid_final.size());
    std::transform(
        dt_grid_final.begin(), dt_grid_final.end(), dm_grid_final.begin(),
        [dm_conv](auto& dt) { return static_cast<float>(dt) * dm_conv; });
    return dm_grid_final;
}

SizeType FDMTPlan::get_dmt_size() const noexcept {
    return m_container.state_shape[m_niters][5];
}
SizeType FDMTPlan::get_buffer_size() const noexcept { return m_buffer_size; }

void FDMTPlan::print_summary() const {
    const auto& state_shape = m_container.state_shape;
    const auto mem_use_mb =
        static_cast<float>(m_container.get_memory_usage()) / 1024.0F / 1024.0F;
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

void FDMTPlan::set_log_level(int level) {
    if (level < static_cast<int>(spdlog::level::trace) ||
        level > static_cast<int>(spdlog::level::off)) {
        spdlog::set_level(spdlog::level::info);
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
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
}

void FDMTPlan::init_plan() {
    m_df         = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
    m_correction = m_df / 2;
    m_niters     = static_cast<SizeType>(std::ceil(std::log2(m_nchans)));
    configure_plan();

    m_buffer_size = compute_buffer_size();
    spdlog::debug("FDMT: df={}, dt_max={}, dt_min={}, dt_step={}, niters={}",
                  m_df, m_dt_max, m_dt_min, m_dt_step, m_niters);
}

DtGridType FDMTPlan::calculate_dt_grid_sub(float f_start, float f_end) const {
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

SizeType FDMTPlan::compute_buffer_size() const {
    const auto& state_shape = m_container.state_shape;
    auto max_elements       = std::max_element(
        state_shape.begin(), state_shape.end(),
        [](const auto& a, const auto& b) { return a[5] < b[5]; });
    return (*max_elements)[5];
}

void FDMTPlan::configure_plan() {
    // Allocate memory/size for plan members
    m_container.df_top.resize(m_niters + 1);
    m_container.df_bot.resize(m_niters + 1);
    m_container.state_shape.resize(m_niters + 1);
    m_container.coordinates.resize(m_niters + 1);
    m_container.coordinates_to_sum.resize(m_niters + 1);
    m_container.coordinates_to_copy.resize(m_niters + 1);
    m_container.dt_grids.resize(m_niters + 1);
    m_container.dt_grid_sub_top.resize(m_niters + 1);
    // For iteration 0
    make_plan_iter0();
    // For iterations 1 to niters
    for (SizeType i_iter = 1; i_iter < m_niters + 1; ++i_iter) {
        make_plan(i_iter);
    }
    spdlog::debug("FDMT: configured fdmt plan");
}

void FDMTPlan::make_plan_iter0() {
    // For iteration 0
    const SizeType i_iter  = 0;
    SizeType buffer_offset = 0;
    SizeType sub_offset    = 0;
    m_container.dt_grids[i_iter].resize(m_nchans);
    for (SizeType i_sub = 0; i_sub < m_nchans; ++i_sub) {
        const auto f_start = m_df * static_cast<float>(i_sub) + m_f_min;
        const auto f_end   = f_start + m_df;
        const auto dt_sub  = calculate_dt_grid_sub(f_start, f_end);
        const auto ndt_sub = dt_sub.size();

        for (SizeType i_dt = 0; i_dt < ndt_sub; ++i_dt) {
            const auto coord_cur =
                FDMTCoord{i_sub,    i_dt,     m_nsamps, buffer_offset,
                          SIZE_MAX, SIZE_MAX, SIZE_MAX};
            m_container.coordinates[i_iter].emplace_back(coord_cur);
            buffer_offset += m_nsamps;
        }
        m_container.dt_grids[i_iter][i_sub] =
            FDMTSubDTGrid{dt_sub, ndt_sub, sub_offset};
        sub_offset += ndt_sub;
    }
    m_container.df_top[i_iter] = m_df;
    m_container.df_bot[i_iter] = m_df;

    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_container.dt_grids[i_iter].begin(),
        m_container.dt_grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    const auto ncoords = m_container.dt_grids[i_iter][m_nchans - 1].sub_offset +
                         m_container.dt_grids[i_iter][m_nchans - 1].ndt;
    m_container.state_shape[i_iter] = {m_nchans,        ndt_min_it->ndt,
                                       ndt_max_it->ndt, ncoords,
                                       m_nsamps,        ncoords * m_nsamps};
    // 0th iteration has no mappings
    m_container.dt_grid_sub_top[i_iter] =
        m_container.dt_grids[i_iter][m_nchans - 1].dt_grid;
}

void FDMTPlan::make_plan(SizeType i_iter) {
    if (i_iter < 1 || i_iter > m_niters) {
        throw std::invalid_argument("Invalid iteration number");
    }
    const auto& df_bot_prev          = m_container.df_bot[i_iter - 1];
    const auto& df_top_prev          = m_container.df_top[i_iter - 1];
    const auto& nchans_prev          = m_container.state_shape[i_iter - 1][0];
    const auto& nsamps_prev          = m_container.state_shape[i_iter - 1][4];
    const auto& dt_grid_sub_top_prev = m_container.dt_grid_sub_top[i_iter - 1];
    const auto& dt_grids_prev        = m_container.dt_grids[i_iter - 1];

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
    m_container.dt_grids[i_iter].resize(nchans_cur);
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
                m_container.coordinates[i_iter].emplace_back(coord_cur);
                m_container.coordinates_to_copy[i_iter].emplace_back(coord_cur);
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
                m_container.coordinates[i_iter].emplace_back(coord_cur);
                m_container.coordinates_to_sum[i_iter].emplace_back(coord_cur);
            }
            buffer_offset += nsamps_iter;
        }
        m_container.dt_grids[i_iter][i_sub] =
            FDMTSubDTGrid{dt_sub, ndt_sub, sub_offset};
        sub_offset += ndt_sub;
    }
    m_container.df_top[i_iter] = df_top;
    m_container.df_bot[i_iter] = df_bot;

    // state shape summary
    const auto [ndt_min_it, ndt_max_it] = std::minmax_element(
        m_container.dt_grids[i_iter].begin(),
        m_container.dt_grids[i_iter].end(),
        [](const auto& a, const auto& b) { return a.ndt < b.ndt; });
    const auto ncoords =
        m_container.dt_grids[i_iter][nchans_cur - 1].sub_offset +
        m_container.dt_grids[i_iter][nchans_cur - 1].ndt;
    m_container.state_shape[i_iter]     = {nchans_cur,      ndt_min_it->ndt,
                                           ndt_max_it->ndt, ncoords,
                                           nsamps_iter,     ncoords * nsamps_iter};
    m_container.dt_grid_sub_top[i_iter] = dt_grid_sub_top;
}

CohFDMTPlan::CohFDMTPlan(float fcenter,
                         float bwsub,
                         SizeType nsub,
                         float tbin,
                         SizeType nbin,
                         SizeType nfft,
                         float t_p,
                         float dm_max,
                         float dm_min,
                         SizeType noverlap_inp)
    : m_fcenter(fcenter),
      m_bwsub(bwsub),
      m_nsub(nsub),
      m_tbin(tbin),
      m_nbin(nbin),
      m_nfft(nfft),
      m_t_p(t_p),
      m_dm_max(dm_max),
      m_dm_min(dm_min),
      m_noverlap_inp(noverlap_inp) {
    validate_inputs();
    configure_plan();
}

// Getters
float CohFDMTPlan::get_fcenter() const noexcept { return m_fcenter; }
float CohFDMTPlan::get_bwsub() const noexcept { return m_bwsub; }
SizeType CohFDMTPlan::get_nsub() const noexcept { return m_nsub; }
float CohFDMTPlan::get_tbin() const noexcept { return m_tbin; }
SizeType CohFDMTPlan::get_nbin() const noexcept { return m_nbin; }
SizeType CohFDMTPlan::get_nfft() const noexcept { return m_nfft; }
float CohFDMTPlan::get_t_p() const noexcept { return m_t_p; }
float CohFDMTPlan::get_dm_max() const noexcept { return m_dm_max; }
float CohFDMTPlan::get_dm_min() const noexcept { return m_dm_min; }
SizeType CohFDMTPlan::get_noverlap_inp() const noexcept {
    return m_noverlap_inp;
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
SizeType CohFDMTPlan::get_noverlap() const noexcept { return m_noverlap; }
SizeType CohFDMTPlan::get_nsamp() const noexcept { return m_nsamp; }
SizeType CohFDMTPlan::get_mbin() const noexcept { return m_mbin; }
SizeType CohFDMTPlan::get_mchan() const noexcept { return m_mchan; }
SizeType CohFDMTPlan::get_msamp() const noexcept { return m_msamp; }
float CohFDMTPlan::get_tsamp() const noexcept { return m_tsamp; }
SizeType CohFDMTPlan::get_dt_max() const noexcept { return m_dt_max; }

void CohFDMTPlan::validate_inputs() const {
    if (m_nsub < 1) {
        throw std::invalid_argument("nsub must be greater than 0");
    }
}

void CohFDMTPlan::configure_plan() {
    m_bw    = m_bwsub * static_cast<float>(m_nsub);
    m_f_min = m_fcenter - m_bw / 2;
    m_f_max = m_fcenter + m_bw / 2;
    m_n_p   = static_cast<SizeType>(std::ceil(m_t_p / m_tbin));
    m_nchan = m_n_p;

    m_dm_grid_coh = dm_utils::generate_coherent_dms(
        m_dm_min, m_dm_max, m_fcenter, m_bw, m_tbin, m_t_p);
    if (m_dm_grid_coh.empty()) {
        throw std::runtime_error("Empty DM grid");
    }

    auto noverlap_optimal = dm_utils::minimum_overlap(
        *std::max_element(m_dm_grid_coh.begin(), m_dm_grid_coh.end()),
        m_fcenter, m_bw, m_tbin, m_nsub, m_nchan);
    auto noverlap_optimal_pow2 = static_cast<SizeType>(
        std::pow(2, std::round(std::log2(noverlap_optimal))));
    m_noverlap = std::max(m_noverlap_inp, noverlap_optimal_pow2);
    if (m_nbin < 2 * m_noverlap) {
        throw std::invalid_argument("nbin must be greater than 2 * noverlap");
    }
    m_nsamp  = m_nfft * (m_nbin - 2 * m_noverlap);
    m_mbin   = m_nbin / m_nchan;
    m_mchan  = m_nsub * m_nchan;
    m_msamp  = m_nsamp / m_nchan;
    m_tsamp  = m_tbin * static_cast<float>(m_nchan);
    m_dt_max = m_n_p;

    // Generate final DM grid
    m_dm_grid_final.resize(m_dm_grid_coh.size() * m_dt_max);
    const float dm_conv = dm_utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    for (SizeType i = 0; i < m_dm_grid_coh.size(); ++i) {
        for (SizeType j = 0; j < m_dt_max; ++j) {
            m_dm_grid_final[i * m_dt_max + j] =
                m_dm_grid_coh[i] + static_cast<float>(j) * dm_conv;
        }
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