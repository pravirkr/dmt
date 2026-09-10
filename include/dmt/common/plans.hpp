#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::plans {

// Shape parameters of the FDMT state buffer in a single iteration
struct FDMTShape {
    SizeType nchans;       // Number of frequency channels
    SizeType ndt_min;      // Minimum number of delays (at highest frequency)
    SizeType ndt_max;      // Maximum number of delays (at lowest frequency)
    SizeType ncoords;      // Number of coordinates (nchans * \sum ndt)
    SizeType ncoords_sum;  // Number of coordinates to sum
    SizeType ncoords_copy; // Number of coordinates to copy
    SizeType nsamps;       // Number of samples
    SizeType nelements;    // Number of elements (ncoords * nsamps)
    SizeType dt_max;       // Maximum subband delay (dt) value

    static constexpr std::string_view header_fmt();
    std::string to_string() const;
};

// Coordinate of the FDMT plan in a single iteration
struct FDMTCoord {
    SizeType i_sub;        // Frequency subband index
    SizeType i_dt;         // Delay index
    SizeType nsamps;       // Number of samples in the state buffer
    SizeType buf_offset;   // Offset (starting point) in the state buffer
    SizeType i_coord_tail; // Tail coordinate index in the previous iteration
    SizeType i_coord_head; // Head coordinate index in the previous iteration
    SizeType delay;        // Delay offset between the tail and head coordinates
    SizeType tail_buf_offset; // Offset (starting point) in the tail buffer
    SizeType tail_nsamps;     // Number of samples in the tail buffer
    SizeType head_buf_offset; // Offset (starting point) in the head buffer
    SizeType head_nsamps;     // Number of samples in the head buffer
};

// Coordinates grid for the FDMT plan for each subband in a single iteration
struct FDMTCoordGrid {
    std::vector<SizeType> dt_grid; // Delay grid for the subband
    SizeType ndt;                  // Number of delays
    SizeType coord_offset; // Offset (starting point) in the coordinates array
    float f_start;         // Start frequency of the subband
    float f_end;           // End frequency of the subband
};

struct FDMTPlanContainer {
    std::vector<FDMTShape> state_shape;
    std::vector<std::vector<FDMTCoordGrid>> grids;
    std::vector<std::vector<FDMTCoord>> coordinates;
    std::vector<std::vector<FDMTCoord>> coordinates_sum;
    std::vector<std::vector<FDMTCoord>> coordinates_copy;
    // Temp arrays to compute the plan
    std::vector<std::vector<SizeType>> dt_grid_sub_top;
    std::vector<float> df_top;
    std::vector<float> df_bot;

    FDMTPlanContainer() = default;
    explicit FDMTPlanContainer(SizeType niters);

    SizeType get_memory_usage() const noexcept;
    SizeType get_buffer_size() const noexcept;
};

struct DDMTPlanContainer {
    std::vector<float> dm_arr;
    // ndm x nchans
    std::vector<SizeType> delay_table;
    SizeType nchans;
};

/**
 * @brief Performance and complexity statistics comparing FDMT to brute-force
 * dedispersion.
 */
struct FDMTComplexity {
    SizeType n_dt;             ///< Number of DM delay trials
    SizeType n_chans;          ///< Number of frequency channels
    SizeType brute_force_ops;  ///< Brute force operations per time sample (n_dt
                               ///< * n_chans)
    SizeType total_tree_nodes; ///< Total active coordinates across levels 1..M
    SizeType sum_additions; ///< Total offset additions per time sample across
                            ///< levels 1..M
    SizeType copy_nodes;    ///< Total copy/forwarding nodes across levels 1..M
    float ops_ratio;        ///< Theoretical speedup factor (brute_force_ops /
                            ///< sum_additions)

    std::string to_string() const;
};

/**
 * @brief Fast Dispersion Measure Transform (FDMT) plan class.
 * @details
 * This class holds all data and logic for an FDMT plan.
 * This includes parameter grids, coordinate mappings, and buffer sizes.
 */
class FDMTPlan {
public:
    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             SizeType dt_max,
             SizeType dt_min       = 0,
             SizeType dt_step      = 1,
             std::string_view mode = "full",
             bool verbose          = false);

    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<SizeType>& dt_grid,
             std::string_view mode = "full",
             bool verbose          = false);

    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             std::string_view mode = "full",
             bool verbose          = false);

    // --- Rule of five: PIMPL ---
    ~FDMTPlan();
    FDMTPlan(FDMTPlan&&) noexcept;
    FDMTPlan& operator=(FDMTPlan&&) noexcept;
    FDMTPlan(const FDMTPlan&);
    FDMTPlan& operator=(const FDMTPlan&);

    // --- Getters ---
    /// @brief Minimum frequency (MHz)
    float get_f_min() const noexcept;
    /// @brief Maximum frequency (MHz)
    float get_f_max() const noexcept;
    /// @brief Number of frequency channels
    SizeType get_nchans() const noexcept;
    /// @brief Number of time samples
    SizeType get_nsamps() const noexcept;
    /// @brief Time sample interval (seconds)
    float get_tsamp() const noexcept;
    /// @brief Maximum delay in time bins
    SizeType get_dt_max() const noexcept;
    /// @brief Minimum delay in time bins
    SizeType get_dt_min() const noexcept;
    /// @brief Delay step in time bins
    SizeType get_dt_step() const noexcept;
    /// @brief Whether a custom arbitrary delay or DM grid was provided
    bool is_custom_grid() const noexcept;
    /// @brief Frequency resolution (MHz)
    float get_df() const noexcept;
    /// @brief Number of iterations
    SizeType get_niters() const noexcept;
    /// @brief Container for the FDMT plan
    const FDMTPlanContainer& get_container() const noexcept;

    // --- Methods ---
    /// @brief Final delay grid in time bins
    [[nodiscard]] std::vector<SizeType> get_dt_grid_final() const noexcept;
    /// @brief Final DM grid (pc/cm^3)
    [[nodiscard]] std::vector<float> get_dm_grid_final() const noexcept;
    /// @brief Final smearing grid (samples per channel)
    [[nodiscard]] std::vector<float> get_smearing_grid_final() const noexcept;
    /// @brief Number of DMs in the final DMT transform
    SizeType get_dmt_ndms() const noexcept;
    /// @brief Number of time samples in the final DMT transform
    SizeType get_dmt_nsamps() const noexcept;
    /// @brief Number of elements in the final DMT transform
    SizeType get_dmt_size() const noexcept;
    /// @brief Size of the buffer for the FDMT plan
    SizeType get_buffer_size() const noexcept;
    /// @brief Size of the Overlap-Save history for the FDMT plan
    SizeType get_history_size() const noexcept;
    /// @brief Size of the Boxcar smearing history for the FDMT plan
    SizeType get_history_init_size() const noexcept;

    /// @brief Computes complexity comparison between FDMT and brute-force
    /// dedispersion
    [[nodiscard]] FDMTComplexity get_complexity() const noexcept;
    /// @brief Prints a formatted comparison of FDMT vs brute-force operations
    void print_complexity_summary() const;

    /// @brief Print a summary of the FDMT plan
    void print_summary(std::string_view prefix = "") const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Coherent Fast Dispersion Measure Transform (CohFDMT) plan class.
 * @details
 * This class holds all data and logic for a CohFDMT plan.
 * This includes parameter grids, coordinate mappings, and buffer sizes.
 */
class CohFDMTPlan {
public:
    CohFDMTPlan(float f_center,
                float bw_sub,
                SizeType nsub,
                float tbin,
                SizeType nbin,
                SizeType nfft,
                float t_p,
                float dm_max,
                float dm_min                = 0.0F,
                SizeType noverlap           = 8192,
                std::string_view data_order = "PRITF",
                bool verbose                = false);

    // --- Rule of five: PIMPL ---
    ~CohFDMTPlan();
    CohFDMTPlan(CohFDMTPlan&&) noexcept;
    CohFDMTPlan& operator=(CohFDMTPlan&&) noexcept;
    CohFDMTPlan(const CohFDMTPlan&);
    CohFDMTPlan& operator=(const CohFDMTPlan&);

    // --- Getters ---
    /// @brief Center frequency (MHz)
    float get_f_center() const noexcept;
    /// @brief Subband bandwidth (MHz)
    float get_bw_sub() const noexcept;
    /// @brief Number of subbands
    SizeType get_nsub() const noexcept;
    /// @brief Time bin size (seconds)
    float get_tbin() const noexcept;
    /// @brief Number of time bins
    SizeType get_nbin() const noexcept;
    /// @brief Number of 1D FFT calls
    SizeType get_nfft() const noexcept;
    /// @brief Pulse width (seconds)
    float get_t_p() const noexcept;
    /// @brief Maximum DM (pc/cm^3)
    float get_dm_max() const noexcept;
    /// @brief Minimum DM (pc/cm^3)
    float get_dm_min() const noexcept;
    /// @brief Number of overlap samples
    SizeType get_noverlap() const noexcept;
    /// @brief Data order
    std::string_view get_data_order() const noexcept;

    // --- Methods ---
    /// @brief Subband bandwidth (MHz)
    float get_bw() const noexcept;
    /// @brief Minimum frequency (MHz)
    float get_f_min() const noexcept;
    /// @brief Maximum frequency (MHz)
    float get_f_max() const noexcept;
    /// @brief Pulse width in time bins
    SizeType get_n_p() const noexcept;
    /// @brief Number of channels
    SizeType get_nchan() const noexcept;
    /// @brief Coherent DM grid (pc/cm^3)
    [[nodiscard]] std::vector<float> get_dm_grid_coh() const noexcept;
    /// @brief Final DM grid (pc/cm^3)
    [[nodiscard]] std::vector<float> get_dm_grid_final() const noexcept;
    /// @brief Number of samples
    SizeType get_nsamp() const noexcept;
    /// @brief Number of bins per channel
    SizeType get_mbin() const noexcept;
    /// @brief Number of channels per subband
    SizeType get_mchan() const noexcept;
    /// @brief Number of samples per channel
    SizeType get_msamp() const noexcept;
    /// @brief Time sample interval (seconds)
    float get_tsamp() const noexcept;
    /// @brief Maximum delay in time bins
    SizeType get_dt_max() const noexcept;

    /// @brief Size of the chirp table
    SizeType get_chirp_table_size() const noexcept;
    /// @brief Size of the unpack buffer
    SizeType get_unpack_buf_size() const noexcept;
    /// @brief Size of the delay buffer
    SizeType get_delay_buf_size() const noexcept;
    /// @brief Size of the intensity buffer
    SizeType get_intensity_buf_size() const noexcept;
    /// @brief Number of elements in the final DMT transform
    SizeType get_dmt_size() const;
    /// @brief Chirp scale
    float get_chirp_scale() const noexcept;

    /// @brief FDMT plan
    const FDMTPlan& get_fdmt_plan() const;
    /// @brief Print a summary of the CohFDMT plan
    void print_summary() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Direct Dispersion Measure Transform (DDMT) plan class.
 * @details
 * This class holds all data and logic for a DDMT plan.
 * This includes parameter grids, coordinate mappings, and buffer sizes.
 */
class DDMTPlan {
public:
    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min = 0.0F,
             bool verbose = false);

    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             std::span<const float> dm_arr,
             bool verbose = false);

    // --- Rule of five: PIMPL ---
    ~DDMTPlan();
    DDMTPlan(DDMTPlan&&) noexcept;
    DDMTPlan& operator=(DDMTPlan&&) noexcept;
    DDMTPlan(const DDMTPlan&);
    DDMTPlan& operator=(const DDMTPlan&);

    // --- Getters ---
    /// @brief Minimum frequency (MHz)
    float get_f_min() const noexcept;
    /// @brief Maximum frequency (MHz)
    float get_f_max() const noexcept;
    /// @brief Number of frequency channels
    SizeType get_nchans() const noexcept;
    /// @brief Time sample interval (seconds)
    float get_tsamp() const noexcept;
    /// @brief DM array (pc/cm^3)
    [[nodiscard]] std::vector<float> get_dm_arr() const noexcept;
    /// @brief Container for the DDMT plan
    const DDMTPlanContainer& get_container() const noexcept;
    /// @brief DM grid (pc/cm^3)
    [[nodiscard]] std::vector<float> get_dm_grid() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::plans