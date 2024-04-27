#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

using DtPlan = std::tuple<size_t, size_t, size_t, size_t>;
using DtGrid = std::vector<size_t>;

struct SubbandPlan {
    float f_start;
    float f_end;
    float f_mid1;
    float f_mid2;
    // dt grid
    DtGrid dt_grid;
    // index plan mapping current dt grid to the previous dt grid
    // i_dt_out, offset, i_dt_tail, i_dt_head
    std::vector<DtPlan> dt_plan;
};
struct FDMTPlan {
    std::vector<float> df_top;
    std::vector<float> df_bot;
    // Temp array to remember the top subband dt grid
    std::vector<DtGrid> dt_grid_sub_top;
    std::vector<std::tuple<size_t, size_t, size_t>> state_shape;
    std::vector<std::vector<SubbandPlan>> sub_plan;
};

class FDMT {
public:
    FDMT(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
         size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    float get_df() const;
    float get_correction() const;
    DtGrid get_dt_grid_init() const;
    DtGrid get_dt_grid_final() const;
    size_t get_niters() const;
    FDMTPlan get_plan() const;
    std::vector<float> get_dm_arr() const;
    void set_log_level(int level);
    void execute(const float* waterfall, size_t waterfall_size, float* dmt,
                 size_t dmt_size);
    void initialise(const float* waterfall, float* state);

private:
    float f_min;
    float f_max;
    size_t nchans;
    size_t nsamps;
    float tsamp;
    size_t dt_max;
    size_t dt_step;
    size_t dt_min;
    float df;
    float correction;
    size_t niters;
    DtGrid dt_grid_final;

    FDMTPlan fdmt_plan;
    std::vector<float> dm_arr;
    // Buffers
    std::vector<float> state_in;
    std::vector<float> state_out;

    static size_t calculate_niters(size_t nchans);
    static float calculate_df(float f_min, float f_max, size_t nchans);
    size_t calculate_dt_sub(float f_start, float f_end, size_t dt) const;
    std::vector<size_t> calculate_dt_grid_sub(float f_start, float f_end) const;
    void check_inputs(size_t waterfall_size, size_t dmt_size) const;
    void execute_iter(const float* state_in, float* state_out, size_t i_iter);
    void configure_fdmt_plan();
    void make_fdmt_plan_iter0();
    void make_fdmt_plan(size_t i_iter);
};
