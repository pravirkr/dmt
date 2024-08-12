#pragma once

#include <array>
#include <vector>

#include <dmt/dmt_types.hpp>

using DtGridType = std::vector<SizeType>;
// state shape: nchans, ndt_min, ndt_max, ncoords, nsamps, nelements
using StShapeType = std::array<SizeType, 6>;

struct FDMTCoord {
    SizeType i_sub;
    SizeType i_dt;
    SizeType nsamps;
    SizeType buffer_offset;
    SizeType i_coord_tail;
    SizeType i_coord_head;
    SizeType offset;
};

struct FDMTSubDTGrid {
    DtGridType dt_grid;
    SizeType ndt;
    SizeType sub_offset;
};

struct FDMTPlanContainer {
    std::vector<float> df_top;
    std::vector<float> df_bot;
    std::vector<StShapeType> state_shape;

    std::vector<std::vector<FDMTCoord>> coordinates;
    std::vector<std::vector<FDMTCoord>> coordinates_to_sum;
    std::vector<std::vector<FDMTCoord>> coordinates_to_copy;
    std::vector<std::vector<FDMTSubDTGrid>> dt_grids;
    // Temp array to remember the top subband dt grid
    std::vector<DtGridType> dt_grid_sub_top;

    SizeType get_memory_usage() const noexcept;
};

class FDMTPlan {
public:
    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_step = 1,
             SizeType dt_min  = 0);

    FDMTPlan(const FDMTPlan&)            = delete;
    FDMTPlan& operator=(const FDMTPlan&) = delete;
    FDMTPlan(FDMTPlan&&)                 = delete;
    FDMTPlan& operator=(FDMTPlan&&)      = delete;
    ~FDMTPlan()                          = default;

    // Getters
    float get_f_min() const noexcept;
    float get_f_max() const noexcept;
    SizeType get_nchans() const noexcept;
    SizeType get_nsamps() const noexcept;
    float get_tsamp() const noexcept;
    SizeType get_dt_max() const noexcept;
    SizeType get_dt_step() const noexcept;
    SizeType get_dt_min() const noexcept;
    float get_df() const noexcept;
    float get_correction() const noexcept;
    SizeType get_niters() const noexcept;
    const FDMTPlanContainer& get_container() const noexcept;
    const DtGridType& get_dt_grid_final() const noexcept;
    std::vector<float> get_dm_grid_final() const noexcept;
    SizeType get_dmt_size() const noexcept;
    SizeType get_buffer_size() const noexcept;

    void print_summary() const;
    static void set_log_level(int level);

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    SizeType m_nsamps;
    float m_tsamp;
    SizeType m_dt_max;
    SizeType m_dt_step;
    SizeType m_dt_min;

    float m_df{};
    float m_correction{};
    SizeType m_niters{};
    FDMTPlanContainer m_container;
    SizeType m_buffer_size{};

    void validate_inputs() const;
    void init_plan();
    DtGridType calculate_dt_grid_sub(float f_start, float f_end) const;
    SizeType compute_buffer_size() const;
    void configure_plan();
    void make_plan_iter0();
    void make_plan(SizeType i_iter);
};

class CohFDMTPlan {
public:
    CohFDMTPlan(float fcenter,
                float bwsub,
                SizeType nsub,
                float tbin,
                SizeType nbin,
                SizeType nfft,
                float t_p,
                float dm_max,
                float dm_min          = 0.0F,
                SizeType noverlap_inp = 8192);

    CohFDMTPlan(const CohFDMTPlan&)            = delete;
    CohFDMTPlan& operator=(const CohFDMTPlan&) = delete;
    CohFDMTPlan(CohFDMTPlan&&)                 = delete;
    CohFDMTPlan& operator=(CohFDMTPlan&&)      = delete;
    ~CohFDMTPlan()                             = default;

    // Getters
    float get_fcenter() const noexcept;
    float get_bwsub() const noexcept;
    SizeType get_nsub() const noexcept;
    float get_tbin() const noexcept;
    SizeType get_nbin() const noexcept;
    SizeType get_nfft() const noexcept;
    float get_t_p() const noexcept;
    float get_dm_max() const noexcept;
    float get_dm_min() const noexcept;
    SizeType get_noverlap_inp() const noexcept;
    float get_bw() const noexcept;
    float get_f_min() const noexcept;
    float get_f_max() const noexcept;
    SizeType get_n_p() const noexcept;
    SizeType get_nchan() const noexcept;
    const std::vector<float>& get_dm_grid_coh() const noexcept;
    const std::vector<float>& get_dm_grid_final() const noexcept;
    SizeType get_noverlap() const noexcept;
    SizeType get_nsamp() const noexcept;
    SizeType get_mbin() const noexcept;
    SizeType get_mchan() const noexcept;
    SizeType get_msamp() const noexcept;
    float get_tsamp() const noexcept;
    SizeType get_dt_max() const noexcept;

private:
    float m_fcenter;
    float m_bwsub;
    SizeType m_nsub;
    float m_tbin;
    SizeType m_nbin;
    SizeType m_nfft;
    float m_t_p;
    float m_dm_max;
    float m_dm_min;
    SizeType m_noverlap_inp;

    float m_bw{};
    float m_f_min{};
    float m_f_max{};
    SizeType m_n_p{};
    SizeType m_nchan{};
    std::vector<float> m_dm_grid_coh;
    std::vector<float> m_dm_grid_final;
    SizeType m_noverlap{};
    SizeType m_nsamp{};
    SizeType m_mbin{};
    SizeType m_mchan{};
    SizeType m_msamp{};
    float m_tsamp{};
    SizeType m_dt_max{};

    void validate_inputs() const;
    void configure_plan();
};

struct DDMTPlanContainer {
    std::vector<float> dm_arr;
    // ndm x nchan
    std::vector<SizeType> delay_table;
    size_t nchans;
};

class DDMTPlan {
public:
    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min = 0.0F);

    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const std::vector<float>& dm_arr);

    DDMTPlan(const DDMTPlan&)            = delete;
    DDMTPlan& operator=(const DDMTPlan&) = delete;
    DDMTPlan(DDMTPlan&&)                 = delete;
    DDMTPlan& operator=(DDMTPlan&&)      = delete;
    ~DDMTPlan()                          = default;

    float get_f_min() const noexcept;
    float get_f_max() const noexcept;
    SizeType get_nchans() const noexcept;
    float get_tsamp() const noexcept;
    const std::vector<float>& get_dm_arr() const noexcept;
    const DDMTPlanContainer& get_container() const noexcept;
    std::vector<float> get_dm_grid() const noexcept;
    static void set_log_level(int level);

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    float m_tsamp;
    std::vector<float> m_dm_arr;

    DDMTPlanContainer m_container;

    void validate_inputs() const;
    void configure_plan();
    static std::vector<float>
    generate_dm_arr(float dm_max, float dm_step, float dm_min);
    static std::vector<float> generate_dm_arr(const float* dm_arr,
                                              SizeType dm_count);
};