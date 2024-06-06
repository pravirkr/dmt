#include <stdexcept>

#include <spdlog/spdlog.h>

#include <dmt/ddmt_base.hpp>
#include <dmt/fdmt_utils.hpp>

DDMT::DDMT(float f_min,
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
    configure_ddmt_plan();
    spdlog::debug("DDMT: dm_max={}, dm_min={}, dm_step={}", dm_max, dm_min,
                  dm_step);
}

DDMT::DDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           const float* dm_arr,
           SizeType dm_count)
    : m_f_min(f_min),
      m_f_max(f_max),
      m_nchans(nchans),
      m_tsamp(tsamp),
      m_dm_arr(generate_dm_arr(dm_arr, dm_count)) {
    validate_inputs();
    configure_ddmt_plan();
    spdlog::debug("DDMT: dm_count={}", dm_count);
}

const DDMTPlan& DDMT::get_plan() const { return m_ddmt_plan; }

std::vector<float> DDMT::get_dm_grid() const { return m_ddmt_plan.dm_arr; }

void DDMT::set_log_level(int level) {
    if (level < static_cast<int>(spdlog::level::trace) ||
        level > static_cast<int>(spdlog::level::off)) {
        spdlog::set_level(spdlog::level::info);
    }
    spdlog::set_level(static_cast<spdlog::level::level_enum>(level));
}

void DDMT::configure_ddmt_plan() {
    m_ddmt_plan.nchans = m_nchans;
    m_ddmt_plan.dm_arr = m_dm_arr;
    const auto df      = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
    m_ddmt_plan.delay_table = ddmt::generate_delay_table(
        m_dm_arr.data(), m_dm_arr.size(), m_f_min, df, m_nchans, m_tsamp);
}

void DDMT::validate_inputs() const {
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
DDMT::generate_dm_arr(float dm_max, float dm_step, float dm_min) {
    std::vector<float> dm_arr;
    for (float dm = dm_min; dm <= dm_max; dm += dm_step) {
        dm_arr.push_back(dm);
    }
    return dm_arr;
}

std::vector<float> DDMT::generate_dm_arr(const float* dm_arr,
                                         SizeType dm_count) {
    std::vector<float> dm_vec(dm_arr, dm_arr + dm_count);
    return dm_vec;
}