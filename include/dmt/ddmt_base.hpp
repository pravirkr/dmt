#pragma once

#include <cstdint>
#include <vector>

using DDMTSize  = uint64_t;
using DDMTFloat = float;
using DDMTWord  = uint32_t;

struct DDMTPlan {
    std::vector<float> dm_arr;
    std::vector<int> delay_arr;
    std::vector<int> kill_mask;
};

class DDMT {
public:
    DDMT(float f_min,
         float f_max,
         DDMTSize nchans,
         DDMTSize nsamps,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min = 0.0F);
    DDMT(const DDMT&)            = delete;
    DDMT& operator=(const DDMT&) = delete;
    DDMT(DDMT&&)                 = delete;
    DDMT& operator=(DDMT&&)      = delete;
    virtual ~DDMT()              = default;

    std::vector<DDMTSize> get_dt_grid() const;
    std::vector<float> get_dm_grid() const;
    static void set_log_level(int level);
    void set_dm_arr(const std::vector<float>& dm_arr);
    void set_kill_mask(const std::vector<int>& kill_mask);
    virtual void execute(const float* __restrict waterfall,
                         DDMTSize waterfall_size,
                         float* __restrict dmt,
                         DDMTSize dmt_size) = 0;

private:
    float m_f_min;
    float m_f_max;
    DDMTSize m_nchans;
    DDMTSize m_nsamps;
    float m_tsamp;
    float m_dm_max;
    float m_dm_step;
    float m_dm_min;

    DDMTPlan m_ddmt_plan;

    void configure_ddmt_plan();
};