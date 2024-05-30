#include <spdlog/spdlog.h>

#include <dmt/ddmt_base.hpp>


DDMT::DDMT(float f_min,
           float f_max,
           SizeType nchans,
           SizeType nsamps,
           float tsamp,
           float dm_max,
           float dm_step,
           float dm_min)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_nsamps(nsamps),
      m_tsamp(tsamp),
      m_dm_max(dm_max),
      m_dm_step(dm_step),
      m_dm_min(dm_min) {
    configure_ddmt_plan();
    spdlog::debug("DDMT: dt_max={}, dt_min={}, dt_step={}", m_dm_max, m_dm_min,
                  m_dm_step);
}

std::vector<SizeType> DDMT::get_dt_grid() const {
    // TODO(abc): Implement get_dt_grid
}
std::vector<float> DDMT::get_dm_grid() const { return m_ddmt_plan.dm_arr; }

void DDMT::set_dm_arr(const std::vector<float>& dm_arr) {
    m_ddmt_plan.dm_arr = dm_arr;
}

void DDMT::set_kill_mask(const std::vector<int>& kill_mask) {
    m_ddmt_plan.kill_mask = kill_mask;
}

void DDMT::set_log_level(int level) {
    if (level < static_cast<int>(spdlog::level::trace) ||
        level > static_cast<int>(spdlog::level::off)) {
        spdlog::set_level(spdlog::level::info);
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

void DDMT::configure_ddmt_plan() {
    m_ddmt_plan.dm_arr.resize(10);
    m_ddmt_plan.delay_arr.resize(m_nchans);
    m_ddmt_plan.kill_mask.resize(m_nchans);
}