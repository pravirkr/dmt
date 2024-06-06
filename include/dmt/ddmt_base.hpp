#pragma once

#include <vector>

using SizeType = std::size_t;

struct DDMTPlan {
    std::vector<float> dm_arr;
    // ndm x nchan
    std::vector<size_t> delay_table;
    size_t nchans;
};

class DDMT {
public:
    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min = 0.0F);

    DDMT(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const float* dm_arr,
         SizeType dm_count);

    DDMT(const DDMT&)            = delete;
    DDMT& operator=(const DDMT&) = delete;
    DDMT(DDMT&&)                 = delete;
    DDMT& operator=(DDMT&&)      = delete;
    virtual ~DDMT()              = default;

    const DDMTPlan& get_plan() const;
    std::vector<float> get_dm_grid() const;
    static void set_log_level(int level);
    virtual void execute(const float* __restrict waterfall,
                         SizeType waterfall_size,
                         float* __restrict dmt,
                         SizeType dmt_size) = 0;

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    float m_tsamp;
    std::vector<float> m_dm_arr;

    DDMTPlan m_ddmt_plan;

    void validate_inputs() const;
    void configure_ddmt_plan();
    static std::vector<float>
    generate_dm_arr(float dm_max, float dm_step, float dm_min);
    static std::vector<float> generate_dm_arr(const float* dm_arr,
                                              SizeType dm_count);
};