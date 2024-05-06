#pragma once

#include <array>
#include <cstddef>
#include <vector>

using SizeType    = size_t;
using DtGridType  = std::vector<SizeType>;
using DtPlanType  = std::vector<std::array<SizeType, 4>>;
using StShapeType = std::array<SizeType, 5>;

struct SubbandPlan {
    float f_start;
    float f_end;
    float f_mid1;
    float f_mid2;
    // index in the state buffer
    SizeType state_idx;
    // dt grid
    DtGridType dt_grid;
    // index plan mapping current dt grid to the previous dt grid
    // i_dt_out, offset, i_dt_tail, i_dt_head
    DtPlanType dt_plan;
};
struct FDMTPlan {
    std::vector<float> df_top;
    std::vector<float> df_bot;
    // Temp array to remember the top subband dt grid
    std::vector<DtGridType> dt_grid_sub_top;
    // state shape: nchans, ndt_min, ndt_max, nchansxndt, nsamps
    std::vector<StShapeType> state_shape;
    std::vector<std::vector<SubbandPlan>> sub_plan;

    size_t calculate_memory_usage() const;
};

class FDMT {
public:
    FDMT(float f_min, float f_max, SizeType nchans, SizeType nsamps,
         float tsamp, SizeType dt_max, SizeType dt_step = 1,
         SizeType dt_min = 0);
    FDMT(const FDMT&)            = delete;
    FDMT& operator=(const FDMT&) = delete;
    FDMT(FDMT&&)                 = delete;
    FDMT& operator=(FDMT&&)      = delete;
    virtual ~FDMT()              = default;
    float get_df() const;
    float get_correction() const;
    SizeType get_niters() const;
    const FDMTPlan& get_plan() const;
    const DtGridType& get_dt_grid_final() const;
    std::vector<float> get_dm_grid_final() const;
    static void set_log_level(int level);
    virtual void execute(const float* waterfall, size_t waterfall_size,
                         float* dmt, size_t dmt_size)
        = 0;
    virtual void initialise(const float* waterfall, float* state) = 0;

protected:
    void check_inputs(size_t waterfall_size, size_t dmt_size) const;

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    SizeType m_nsamps;
    float m_tsamp;
    SizeType m_dt_max;
    SizeType m_dt_step;
    SizeType m_dt_min;

    float m_df;
    float m_correction;
    SizeType m_niters;
    FDMTPlan m_fdmt_plan;

    static SizeType calculate_niters(SizeType nchans);
    static float calculate_df(float f_min, float f_max, SizeType nchans);
    DtGridType calculate_dt_grid_sub(float f_start, float f_end) const;
    void configure_fdmt_plan();
    void make_fdmt_plan_iter0();
    void make_fdmt_plan(SizeType i_iter);
};
