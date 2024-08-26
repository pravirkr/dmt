#pragma once

#include <vector>

#include <dmt/dmt_types.hpp>

using DtGridType = std::vector<SizeType>;

// Shape of the FDMT state buffer in a single iteration
struct FDMTShape {
    SizeType nchans;    // Number of channels
    SizeType ndt_min;   // Minimum number of delays
    SizeType ndt_max;   // Maximum number of delays
    SizeType ncoords;   // Number of coordinates (nchans * ndt)
    SizeType nsamps;    // Number of samples
    SizeType nelements; // Number of elements (ncoords * nsamps)
};

// Coordinate of the FDMT plan in a single iteration
struct FDMTCoord {
    SizeType i_sub;        // Subband index
    SizeType i_dt;         // Delay index
    SizeType nsamps;       // Number of samples in the state buffer
    SizeType buf_offset;   // Offset (starting point) in the state buffer
    SizeType i_coord_tail; // Tail coordinate index in the previous iteration
    SizeType i_coord_head; // Head coordinate index in the previous iteration
    SizeType offset;       // Offset between the tail and head coordinates
};

// Coordinates grid for the FDMT plan for each subband in a single iteration
struct FDMTCoordGrid {
    DtGridType dt_grid;    // Delay grid for the subband
    SizeType ndt;          // Number of delays
    SizeType coord_offset; // Offset (starting point) in the coordinates array
    float f_start;         // Start frequency of the subband
    float f_end;           // End frequency of the subband
};

struct FDMTPlanContainer {
    std::vector<FDMTShape> state_shape;
    std::vector<std::vector<FDMTCoordGrid>> grids;
    std::vector<std::vector<FDMTCoord>> coordinates;
    std::vector<std::vector<FDMTCoord>> coordinates_to_sum;
    std::vector<std::vector<FDMTCoord>> coordinates_to_copy;
    // Temp arrays to compute the plan
    std::vector<DtGridType> dt_grid_sub_top;
    std::vector<float> df_top;
    std::vector<float> df_bot;
    std::vector<SizeType> dt_max;

    FDMTPlanContainer() = default;
    explicit FDMTPlanContainer(SizeType niters);

    SizeType get_memory_usage() const noexcept;
    SizeType get_buffer_size() const noexcept;
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
    DtGridType calculate_dt_grid_sub(float f_start, float f_end) const;
    void configure_plan();
    void make_plan_iter0();
    void make_plan(SizeType i_iter);
};

struct CohFDMTPlan {
    float fcenter;
    float bwsub;
    SizeType nsub;
    float tbin;
    SizeType nbin;
    SizeType nfft;
    float t_p;
    float dm_max;
    float dm_min;
    SizeType noverlap_inp;

    float bw;
    float f_min;
    float f_max;
    SizeType n_p{};
    SizeType nchan{};
    std::vector<float> dm_grid_coh;
    std::vector<float> dm_grid_final;
    SizeType noverlap{};
    SizeType nsamp{};
    SizeType mbin{};
    SizeType mchan{};
    SizeType msamp{};
    float tsamp{};
    SizeType dt_max{};

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

    const std::vector<float>& get_dm_grid_coh() const noexcept;
    const std::vector<float>& get_dm_grid_final() const noexcept;

private:
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
    DDMTPlan(DDMTPlan&&)                 = default;
    DDMTPlan& operator=(DDMTPlan&&)      = default;
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