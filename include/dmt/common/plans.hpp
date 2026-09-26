#pragma once

/**
 * @file plans.hpp
 * @brief Dedispersion execution plans, coordinate graphs, complexity
 * calculators, and parameter containers.
 */

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::plans {

/** @brief FDMT state-buffer shape at one tree iteration level. */
struct FDMTShape {
    SizeType nchans;       ///< Frequency subbands at this level
    SizeType ndt_min;      ///< Min delay trials (highest-frequency subband)
    SizeType ndt_max;      ///< Max delay trials (lowest-frequency subband)
    SizeType ncoords;      ///< Active coordinates (nchans * sum of ndt)
    SizeType ncoords_sum;  ///< Coordinates that add tail + head operands
    SizeType ncoords_copy; ///< Coordinates that copy from the previous level
    SizeType nsamps;       ///< Time samples allocated per coordinate
    SizeType nelements;    ///< Buffer length in elements (ncoords * nsamps)
    SizeType dt_max;       ///< Max subband delay (dt) at this level

    static constexpr std::string_view header_fmt();
    std::string to_string() const;
};

/**
 * @brief Execution coordinate and dependency edge of the FDMT tree DAG in a
 * single iteration.
 *
 * Encapsulates the binary merge node:
 * @code
 * state[coord.buf_offset + t] = state_prev[tail_buf_offset + t] +
 * state_prev[head_buf_offset + t - delay]
 * @endcode
 */
struct FDMTCoord {
    SizeType i_sub;        ///< Subband index
    SizeType i_dt;         ///< Delay-trial index within the subband
    SizeType nsamps;       ///< Valid output time samples
    SizeType buf_offset;   ///< Start offset in the destination state buffer
    SizeType i_coord_tail; ///< Prev-level linear index of tail (lower subband)
    SizeType i_coord_head; ///< Prev-level linear index of head (upper subband)
    SizeType delay;        ///< Head dispersive shift in samples
    SizeType tail_buf_offset; ///< Tail start offset in source buffer
    SizeType tail_nsamps;     ///< Valid samples in tail operand
    SizeType head_buf_offset; ///< Head start offset in source buffer
    SizeType head_nsamps;     ///< Valid samples in the head operand
    SizeType hist_offset{0};  ///< Overlap-save history offset (mode="valid")
};

/** @brief Subband delay grid and MHz edges for one FDMT iteration. */
struct FDMTCoordGrid {
    std::vector<IndexType> dt_grid; ///< Delay trials in sample units
    SizeType ndt;                   ///< Number of delay trials
    SizeType coord_offset;          ///< Offset into the global coordinate list
    float f_start;                  ///< Subband lower edge frequency (MHz)
    float f_end;                    ///< Subband upper edge frequency (MHz)
};

/** @brief Precomputed FDMT execution graph and memory layout. */
struct FDMTPlanContainer {
    std::vector<FDMTShape> state_shape; ///< State shape per level (0..niters)
    std::vector<std::vector<FDMTCoordGrid>> grids;   ///< Subband grids per iter
    std::vector<std::vector<FDMTCoord>> coordinates; ///< Coords per iteration
    std::vector<std::vector<FDMTCoord>> coordinates_sum;  ///< Add ops per iter
    std::vector<std::vector<FDMTCoord>> coordinates_copy; ///< Copy ops per iter
    std::vector<std::vector<IndexType>> dt_grid_sub_top;  ///< Scratch dt grids
    std::vector<float> df_top;     ///< Channel bandwidths (top half)
    std::vector<float> df_bot;     ///< Channel bandwidths (bottom half)
    SizeType tree_history_size{0}; ///< Floats needed for streaming tree history

    FDMTPlanContainer() = default;
    explicit FDMTPlanContainer(SizeType niters);

    [[nodiscard]] SizeType get_memory_usage() const noexcept;
    [[nodiscard]] SizeType get_buffer_size() const noexcept;
};

/** @brief Precomputed DDMT delay tables and channel masks. */
struct DDMTPlanContainer {
    std::vector<float> dm_arr;         ///< DM trials (pc/cm^3), length ndm
    std::vector<SizeType> delay_table; ///< Row-major delays (ndm x nchans)
    std::vector<float> fractional_delay_table; ///< d(delay)/d(DM) per channel
    std::vector<uint8_t> kill_mask; ///< Channel mask: 1=active, 0=RFI masked
    SizeType nchans;                ///< Number of frequency channels
    SizeType nbits = 32;            ///< Bits per sample (32, 16, 8, 4, 2, 1)
};

/** @brief FDMT vs brute-force dedispersion complexity metrics. */
struct FDMTComplexity {
    SizeType n_dt;             ///< DM delay trials
    SizeType n_chans;          ///< Frequency channels
    SizeType brute_force_ops;  ///< Ops per sample (n_dt * n_chans)
    SizeType total_tree_nodes; ///< Active coordinates on levels 1..M
    SizeType sum_additions;    ///< Additions per sample on levels 1..M
    SizeType copy_nodes;       ///< Copy nodes on levels 1..M
    float ops_ratio;           ///< Theoretical speedup (excludes level 0)
    std::vector<SizeType> ops_by_iter;   ///< Adds per sample per level (0..M)
    std::vector<SizeType> nodes_by_iter; ///< Active coords per level (0..M)

    std::string to_string() const;
};

/**
 * @brief Fast Dispersion Measure Transform (FDMT) execution plan.
 *
 * Precomputes the hierarchical dyadic subband merge tree, coordinate dependency
 * DAG, and memory requirements for incoherent dedispersion.
 *
 * The algorithm recursively merges subbands:
 * @f[
 * N_{\text{subband}} \to \frac{N_{\text{subband}}}{2} \to \dots \to 1
 * @f]
 * reducing computational complexity from brute-force @f$ O(N_{\text{chans}}
 * \cdot N_{\text{samps}} \cdot N_{\text{DM}}) @f$ to @f$ O(N_{\text{chans}}
 * \cdot N_{\text{samps}} \log_2 N_{\text{chans}}) @f$.
 */
class FDMTPlan {
public:
    /**
     * @brief Constructs an FDMT plan with a linear delay grid.
     *
     * @param f_min Bottom edge frequency of the band in MHz.
     * @param f_max Top edge frequency of the band in MHz.
     * @param nchans Number of frequency channels. Must be a power of two.
     * @param nsamps Number of time samples per processing block.
     * @param tsamp Sampling interval in seconds.
     * @param dt_max Maximum dispersive delay trial in samples across the full
     * band.
     * @param dt_min Minimum dispersive delay trial in samples across the band
     * (supports negative delays, default: 0).
     * @param dt_step Stride between delay trials (default: 1).
     * @param mode Output time alignment mode: "valid" (overlap-save streaming
     * history), "full" (zero-padded), or "roll" (cyclic) (default: "valid").
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     */
    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             IndexType dt_max,
             IndexType dt_min      = 0,
             SizeType dt_step      = 1,
             std::string_view mode = "valid",
             int verbose           = 0);

    /**
     * @brief Constructs an FDMT plan with a custom delay trial grid.
     *
     * Top-down pruning derives optimal intermediate subband delay sets that
     * guarantee only the requested final delay trials are materialized at the
     * tree root.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels (must be a power of two).
     * @param nsamps Number of time samples per block.
     * @param tsamp Sampling interval in seconds.
     * @param dt_grid Explicit list of delay trials in samples (e.g. from
     * generate_optimal_dt_grid).
     * @param mode Transform mode: "valid", "full", or "roll".
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     */
    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<IndexType>& dt_grid,
             std::string_view mode = "valid",
             int verbose           = 0);

    /**
     * @brief Constructs an FDMT plan with a custom physical DM trial grid.
     *
     * Converts trial DMs (in pc/cm^3) to delay samples using the dispersion
     * constant:
     * @f$ \Delta t = k_{\text{DM}} \cdot \text{DM} \cdot (f_{\text{min}}^{-2} -
     * f_{\text{max}}^{-2}) / t_{\text{samp}} @f$.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels (must be a power of two).
     * @param nsamps Number of time samples per block.
     * @param tsamp Sampling interval in seconds.
     * @param dm_grid Explicit list of DM trials in pc/cm^3 (supports
     * non-uniform spacing and negative DMs).
     * @param mode Transform mode: "valid", "full", or "roll".
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     */
    FDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             std::string_view mode = "valid",
             int verbose           = 0);

    // --- Rule of five: PIMPL ---
    ~FDMTPlan();
    FDMTPlan(FDMTPlan&&) noexcept;
    FDMTPlan& operator=(FDMTPlan&&) noexcept;
    FDMTPlan(const FDMTPlan&);
    FDMTPlan& operator=(const FDMTPlan&);

    // --- Getters ---
    /// @brief Minimum frequency in MHz
    [[nodiscard]] float get_f_min() const noexcept;
    /// @brief Maximum frequency in MHz
    [[nodiscard]] float get_f_max() const noexcept;
    /// @brief Number of frequency channels
    [[nodiscard]] SizeType get_nchans() const noexcept;
    /// @brief Number of time samples per block
    [[nodiscard]] SizeType get_nsamps() const noexcept;
    /// @brief Time sampling interval in seconds
    [[nodiscard]] float get_tsamp() const noexcept;
    /// @brief Maximum delay trial in samples
    [[nodiscard]] IndexType get_dt_max() const noexcept;
    /// @brief Minimum delay trial in samples
    [[nodiscard]] IndexType get_dt_min() const noexcept;
    /// @brief Delay trial stride in samples
    [[nodiscard]] SizeType get_dt_step() const noexcept;
    /// @brief Transform mode ("valid", "full", or "roll")
    [[nodiscard]] std::string_view get_mode() const noexcept;
    /// @brief True if initialized from a custom non-uniform dt or DM grid
    [[nodiscard]] bool is_custom_grid() const noexcept;
    /// @brief Channel frequency resolution in MHz: (f_max - f_min) / nchans
    [[nodiscard]] float get_df() const noexcept;
    /// @brief Number of tree iterations: log2(nchans)
    [[nodiscard]] SizeType get_niters() const noexcept;
    /// @brief Read-only reference to precomputed coordinate graphs and shapes
    [[nodiscard]] const FDMTPlanContainer& get_container() const noexcept;

    // --- Methods ---
    /// @brief Final delay grid in time samples at the root of the tree
    [[nodiscard]] std::vector<IndexType> get_dt_grid_final() const noexcept;
    /// @brief Final DM grid in pc/cm^3
    [[nodiscard]] std::vector<float> get_dm_grid_final() const noexcept;
    /// @brief Final intra-channel smearing grid in samples per channel
    [[nodiscard]] std::vector<float> get_smearing_grid_final() const noexcept;

    /**
     * @brief Decompiles a final DM trial's tree lineage into per-channel time
     * shifts.
     *
     * Used by @ref dmt::algorithms::add_frb_track to inject a synthetic pulse
     * landing exactly on trial @p dm_idx at the tree output.
     *
     * @param dm_idx Index in the final DM trial grid (0 <= dm_idx <
     * get_dmt_ndms()).
     * @return Vector of length nchans containing sample shifts relative to
     * reference channel.
     * @throws std::out_of_range if dm_idx is invalid.
     */
    [[nodiscard]] std::vector<IndexType> trace_dm(SizeType dm_idx) const;

    /**
     * @brief Computes theoretical noise variance for a DM trial and boxcar
     * width.
     * @param dm_idx DM trial index.
     * @param boxcar_width Downsampling / boxcar filter width in time samples
     * (default: 1).
     * @param use_box_smearing Whether intra-channel boxcar smearing was applied
     * in the tree.
     * @return Theoretical noise variance for unit input Gaussian noise.
     */
    [[nodiscard]] float
    get_effective_variance(SizeType dm_idx,
                           SizeType boxcar_width,
                           bool use_box_smearing = true) const;

    /**
     * @brief Computes theoretical noise standard deviation for a DM trial and
     * boxcar width.
     * @param dm_idx DM trial index.
     * @param boxcar_width Boxcar filter width in samples.
     * @param use_box_smearing Whether boxcar smearing was enabled.
     * @return Theoretical noise standard deviation (sqrt of effective
     * variance).
     */
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width,
                                            bool use_box_smearing = true) const;

    /**
     * @brief Evaluates theoretical noise variance across all DM trials.
     * @param boxcar_width Boxcar filter width in samples.
     * @param use_box_smearing Whether boxcar smearing was enabled.
     * @return Vector of length ndm containing noise variance profile.
     */
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width,
                                bool use_box_smearing = true) const;

    /**
     * @brief Evaluates theoretical noise sigma profile across all DM trials.
     *
     * Dividing the raw output DMT matrix by this profile yields exact
     * matched-filter S/N.
     *
     * @param boxcar_width Boxcar filter width in samples.
     * @param use_box_smearing Whether boxcar smearing was enabled.
     * @return Vector of length ndm containing noise standard deviations.
     */
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width,
                             bool use_box_smearing = true) const;

    /// @brief Number of DM trials in the final transform
    [[nodiscard]] SizeType get_dmt_ndms() const noexcept;
    /// @brief Number of time samples per DM trial in the output transform
    [[nodiscard]] SizeType get_dmt_nsamps() const noexcept;
    /// @brief Total elements in the final DMT output buffer (get_dmt_ndms() *
    /// get_dmt_nsamps())
    [[nodiscard]] SizeType get_dmt_size() const noexcept;
    /// @brief Scratch buffer size in floats needed for internal ping-pong state
    /// buffers
    [[nodiscard]] SizeType get_buffer_size() const noexcept;
    /// @brief Overlap-save history size in floats
    [[nodiscard]] SizeType get_history_size() const noexcept;
    /// @brief Boxcar smearing initialization history size in floats
    [[nodiscard]] SizeType get_history_init_size() const noexcept;
    /// @brief Cross-block tree streaming history buffer size in floats
    /// (mode="valid")
    [[nodiscard]] SizeType get_tree_history_size() const noexcept;

    /// @brief Overlap length L = max(|dt_min|, |dt_max|) for FDMT-FFT
    /// full/valid modes
    [[nodiscard]] SizeType get_fft_overlap() const noexcept;
    /// @brief FFT length for FDMT-FFT convolution
    [[nodiscard]] SizeType get_fft_size() const noexcept;
    /// @brief Number of complex bins in R2C Fourier domain (fft_size / 2 + 1)
    [[nodiscard]] SizeType get_fft_n_bins() const noexcept;
    /// @brief Buffer size in complex elements for FDMT-FFT ping-pong buffers
    [[nodiscard]] SizeType get_fft_buffer_size() const noexcept;
    /// @brief Maximum shift in samples across all tree coordinates
    [[nodiscard]] SizeType get_max_shift() const noexcept;
    /// @brief Precomputed complex phasor rotation table of shape (max_shift +
    /// 1, n_bins)
    [[nodiscard]] std::vector<ComplexType> get_fft_phasor_table() const;

    /**
     * @brief Computes complexity metrics comparing FDMT to direct brute-force
     * dedispersion.
     * @param use_box_smearing Whether boxcar smearing is enabled.
     * @return FDMTComplexity struct containing node counts, operations, and
     * speedup ratio.
     */
    [[nodiscard]] FDMTComplexity
    get_complexity(bool use_box_smearing = true) const noexcept;

    /// @brief Prints a formatted comparison of FDMT vs brute-force operations
    void print_complexity_summary() const;

    /// @brief Number of additions per time sample for each tree level (0..M)
    [[nodiscard]] std::vector<SizeType>
    get_operations_by_iteration(bool use_box_smearing = true) const;

    /// @brief Number of active tree coordinates for each tree level (0..M)
    [[nodiscard]] std::vector<SizeType> get_nodes_by_iteration() const;

    /// @brief State shape for each tree level (0..M)
    [[nodiscard]] const std::vector<FDMTShape>&
    get_state_shapes() const noexcept;

    /// @brief Total operations (additions) per time sample across all levels
    /// (0..M)
    [[nodiscard]] SizeType
    get_total_operations(bool use_box_smearing = true) const noexcept;

    /// @brief Total operations (additions) per block across all levels (0..M)
    [[nodiscard]] SizeType
    get_total_flops(bool use_box_smearing = true) const noexcept;

    /// @brief Theoretical compute throughput (GFLOPs/sec) required for
    /// real-time processing
    [[nodiscard]] float
    get_theoretical_gflops(bool use_box_smearing = true) const noexcept;

    /// @brief Prints human-readable summary of plan dimensions, memory, and
    /// parameters
    void print_summary(std::string_view prefix = "") const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Coherent Fast Dispersion Measure Transform (CohFDMT) execution plan.
 *
 * Implements the hybrid coherent/incoherent search algorithm (Zackay et al.).
 * Combines coherent dedispersion over sparse coarse DM trials on raw baseband
 * voltages with a fine-resolution FDMT tree covering residual subband delays.
 */
class CohFDMTPlan {
public:
    /**
     * @brief Constructs a coherent hybrid search plan.
     *
     * @param f_center Central radio frequency of the observation in MHz.
     * @param bw_sub Subband bandwidth in MHz.
     * @param nsub Number of subbands.
     * @param tbin Raw baseband voltage time resolution in seconds (1 /
     * total_bandwidth).
     * @param nbin FFT block size for coherent dedispersion.
     * @param nfft Number of contiguous FFT blocks processed per coherent
     * segment.
     * @param t_p Desired output time resolution of detected Stokes I samples in
     * seconds.
     * @param dm_max Maximum coherent DM trial in pc/cm^3.
     * @param dm_min Minimum coherent DM trial in pc/cm^3 (default: 0).
     * @param noverlap Overlap length in samples for overlap-save baseband
     * convolution (default: 8192).
     * @param data_order Voltage memory order: "PRITF", "FTPRI", or "RITFP"
     * (default: "PRITF").
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     */
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
                int verbose                 = 0);

    // --- Rule of five: PIMPL ---
    ~CohFDMTPlan();
    CohFDMTPlan(CohFDMTPlan&&) noexcept;
    CohFDMTPlan& operator=(CohFDMTPlan&&) noexcept;
    CohFDMTPlan(const CohFDMTPlan&);
    CohFDMTPlan& operator=(const CohFDMTPlan&);

    // --- Getters ---
    /// @brief Center frequency in MHz
    [[nodiscard]] float get_f_center() const noexcept;
    /// @brief Bandwidth per subband in MHz
    [[nodiscard]] float get_bw_sub() const noexcept;
    /// @brief Number of synthesized subbands
    [[nodiscard]] SizeType get_nsub() const noexcept;
    /// @brief Baseband voltage time sample interval in seconds
    [[nodiscard]] float get_tbin() const noexcept;
    /// @brief Number of voltage time bins per FFT block
    [[nodiscard]] SizeType get_nbin() const noexcept;
    /// @brief Number of FFT segments
    [[nodiscard]] SizeType get_nfft() const noexcept;
    /// @brief Output detected pulse resolution in seconds
    [[nodiscard]] float get_t_p() const noexcept;
    /// @brief Maximum coherent search DM in pc/cm^3
    [[nodiscard]] float get_dm_max() const noexcept;
    /// @brief Minimum coherent search DM in pc/cm^3
    [[nodiscard]] float get_dm_min() const noexcept;
    /// @brief Overlap sample count for linear convolution
    [[nodiscard]] SizeType get_noverlap() const noexcept;
    /// @brief Baseband memory layout identifier
    [[nodiscard]] std::string_view get_data_order() const noexcept;

    // --- Methods ---
    /// @brief Total processed RF bandwidth in MHz (nsub * bw_sub)
    [[nodiscard]] float get_bw() const noexcept;
    /// @brief Lowest RF frequency in MHz
    [[nodiscard]] float get_f_min() const noexcept;
    /// @brief Highest RF frequency in MHz
    [[nodiscard]] float get_f_max() const noexcept;
    /// @brief Pulse width in voltage time bins
    [[nodiscard]] SizeType get_n_p() const noexcept;
    /// @brief Total synthesized channels
    [[nodiscard]] SizeType get_nchan() const noexcept;
    /// @brief Coarse coherent DM grid in pc/cm^3
    [[nodiscard]] std::vector<float> get_dm_grid_coh() const noexcept;
    /// @brief Final fine DM grid across all trials in pc/cm^3
    [[nodiscard]] std::vector<float> get_dm_grid_final() const noexcept;
    /// @brief Total voltage samples per segment
    [[nodiscard]] SizeType get_nsamp() const noexcept;
    /// @brief Voltage bins per channel
    [[nodiscard]] SizeType get_mbin() const noexcept;
    /// @brief Channels per subband
    [[nodiscard]] SizeType get_mchan() const noexcept;
    /// @brief Detected time samples per channel
    [[nodiscard]] SizeType get_msamp() const noexcept;
    /// @brief Detected sampling interval in seconds
    [[nodiscard]] float get_tsamp() const noexcept;
    /// @brief Maximum residual delay in time samples for the fine FDMT tree
    [[nodiscard]] SizeType get_dt_max() const noexcept;
    /// @brief Minimum (symmetric negative) delay in samples for the fine FDMT
    /// tree
    [[nodiscard]] IndexType get_dt_min() const noexcept;

    /// @brief Size of the complex chirp phase lookup table
    [[nodiscard]] SizeType get_chirp_table_size() const noexcept;
    /// @brief Size of the baseband unpacking scratch buffer
    [[nodiscard]] SizeType get_unpack_buf_size() const noexcept;
    /// @brief Size of the inter-channel delay buffer
    [[nodiscard]] SizeType get_delay_buf_size() const noexcept;
    /// @brief Size of the detected Stokes I intensity buffer
    [[nodiscard]] SizeType get_intensity_buf_size() const noexcept;
    /// @brief Total number of final DM trials across all coarse and fine steps
    [[nodiscard]] SizeType get_ndm() const noexcept;
    /// @brief Total number of final DM trials (alias for get_ndm())
    [[nodiscard]] SizeType get_dmt_ndms() const noexcept;
    /// @brief Number of detected time samples per DM trial in output matrix
    [[nodiscard]] SizeType get_dmt_nsamps() const noexcept;
    /// @brief Theoretical noise variance grid across all DM trials
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1,
                                bool use_box_smearing = true) const;
    /// @brief Theoretical noise standard deviation profile across all DM trials
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1,
                             bool use_box_smearing = true) const;
    /// @brief Cumulative sample accumulation count per channel-integrated DM
    /// trial
    [[nodiscard]] std::vector<float> get_cumulative_count_grid() const;
    /// @brief Total float elements in the final output DMT transform (ndm *
    /// dmt_nsamps)
    [[nodiscard]] SizeType get_dmt_size() const;
    /**
     * @brief Output buffer length execute() needs: (N - 1) * D + B, with N
     * coarse-DM trials, D = the fine FDMT's get_dmt_size() and B = its
     * get_buffer_size(). Trial i's fine FDMT runs in place at offset i * D
     * (its B-sized ping-pong span overlaps the next trials' slots, which are
     * written afterwards), so only the leading get_dmt_size() = N * D values
     * are the result; the tail is scratch.
     */
    [[nodiscard]] SizeType get_buffer_size() const;
    /// @brief Scaling factor applied to phase chirps
    [[nodiscard]] float get_chirp_scale() const noexcept;

    /// @brief Read-only reference to internal fine FDMT plan
    [[nodiscard]] const FDMTPlan& get_fdmt_plan() const;
    /// @brief Prints human-readable summary of the CohFDMT search parameters
    void print_summary() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Configuration parameters for generating an optimal DM trial grid using
 * Lina Levin's formulation.
 *
 * Implements the pulse-broadening tolerance criterion from Levin (2012):
 * @f$ W_{\text{eff}} \le \text{tol} \cdot W_0 @f$.
 */
struct LevinConfig {
    float dm_start;    ///< Lowest DM trial in pc/cm^3 (must be >= 0)
    float dm_end;      ///< Upper DM search bound in pc/cm^3
    float pulse_width; ///< Intrinsic pulse width in seconds
    float tol; ///< Pulse broadening tolerance factor (e.g. 1.15 to 1.25, must
               ///< be > 1.0)
};

/**
 * @brief Direct Dispersion Measure Transform (DDMT) execution plan.
 *
 * Precomputes per-channel delay tables, fractional delays, and channel kill
 * masks for brute-force delay-and-sum dedispersion on float or packed-integer
 * filterbanks.
 */
class DDMTPlan {
public:
    /**
     * @brief Constructs a DDMT plan with a linear DM trial grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_max Maximum trial DM in pc/cm^3.
     * @param dm_step Linear spacing between DM trials in pc/cm^3.
     * @param dm_min Minimum trial DM in pc/cm^3 (default: 0).
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nbits Precision per sample: 32 (float), or 1, 2, 4, 8, 16 (packed
     * integers).
     * @param kill_mask Optional per-channel mask (size nchans, 1=keep, 0=mask
     * out).
     */
    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             float dm_max,
             float dm_step,
             float dm_min                       = 0.0F,
             int verbose                        = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {});

    /**
     * @brief Constructs a DDMT plan with an explicit list of DM trials.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tsamp Sampling interval in seconds.
     * @param dm_arr Explicit span of DM trials in pc/cm^3.
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nbits Precision per sample (default: 32).
     * @param kill_mask Optional per-channel mask.
     */
    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             std::span<const float> dm_arr,
             int verbose                        = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {});

    /**
     * @brief Constructs a DDMT plan with an optimal Lina Levin DM grid.
     *
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of channels.
     * @param tsamp Sampling interval in seconds.
     * @param levin LevinConfig specifying dm_start, dm_end, pulse_width, and
     * tol.
     * @param verbose 0 = warnings, 1 = info, 2 = debug (process-wide).
     * @param nbits Precision per sample (default: 32).
     * @param kill_mask Optional per-channel mask.
     */
    DDMTPlan(float f_min,
             float f_max,
             SizeType nchans,
             float tsamp,
             const LevinConfig& levin,
             int verbose                        = 0,
             SizeType nbits                     = 32,
             std::span<const uint8_t> kill_mask = {});

    // --- Rule of five: PIMPL ---
    ~DDMTPlan();
    DDMTPlan(DDMTPlan&&) noexcept;
    DDMTPlan& operator=(DDMTPlan&&) noexcept;
    DDMTPlan(const DDMTPlan&);
    DDMTPlan& operator=(const DDMTPlan&);

    // --- Getters ---
    /// @brief Minimum frequency in MHz
    [[nodiscard]] float get_f_min() const noexcept;
    /// @brief Maximum frequency in MHz
    [[nodiscard]] float get_f_max() const noexcept;
    /// @brief Number of frequency channels
    [[nodiscard]] SizeType get_nchans() const noexcept;
    /// @brief Time sample interval in seconds
    [[nodiscard]] float get_tsamp() const noexcept;
    /// @brief DM array in pc/cm^3
    [[nodiscard]] std::vector<float> get_dm_arr() const noexcept;
    /// @brief Read-only reference to precomputed delay tables and masks
    [[nodiscard]] const DDMTPlanContainer& get_container() const noexcept;
    /// @brief DM grid in pc/cm^3
    [[nodiscard]] std::vector<float> get_dm_grid() const noexcept;
    /// @brief Per-channel fractional delay rate array (size nchans) in bins /
    /// (pc cm^-3)
    [[nodiscard]] std::vector<float>
    get_fractional_delay_table() const noexcept;
    /// @brief Bit precision per input sample (32 => float, 1/2/4/8/16 => packed
    /// integer)
    [[nodiscard]] SizeType get_nbits() const noexcept;
    /// @brief Channel kill mask (1 = keep, 0 = masked out); size == nchans
    [[nodiscard]] std::vector<uint8_t> get_kill_mask() const noexcept;

    /**
     * @brief Replaces the channel kill mask without recomputing the delay
     * table.
     * @param kill_mask New per-channel mask (size must equal nchans).
     * @throws std::invalid_argument if kill_mask.size() != nchans.
     */
    void set_kill_mask(std::span<const uint8_t> kill_mask);

    /// @brief Theoretical noise variance for a DM trial (shared by all trials)
    [[nodiscard]] float get_effective_variance() const noexcept;
    /// @brief Theoretical noise standard deviation for a DM trial
    [[nodiscard]] float get_effective_sigma() const noexcept;
    /// @brief Theoretical noise variance grid, one value per DM trial
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid() const noexcept;
    /// @brief Theoretical noise standard deviation grid, one value per DM trial
    [[nodiscard]] std::vector<float> get_effective_sigma_grid() const noexcept;

    /**
     * @brief Generates an optimal DM trial grid using Lina Levin's tolerance
     * rule.
     *
     * @param dm_start Lowest trial DM in pc/cm^3.
     * @param dm_end Highest search DM in pc/cm^3.
     * @param tsamp Sampling interval in seconds.
     * @param pulse_width Intrinsic pulse width in seconds.
     * @param f_min Bottom edge frequency in MHz.
     * @param f_max Top edge frequency in MHz.
     * @param nchans Number of frequency channels.
     * @param tol Broadening tolerance factor (> 1.0).
     * @return Sorted vector of trial DMs in pc/cm^3.
     */
    [[nodiscard]] static std::vector<float>
    generate_levin_dm_grid(float dm_start,
                           float dm_end,
                           float tsamp,
                           float pulse_width,
                           float f_min,
                           float f_max,
                           SizeType nchans,
                           float tol);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::plans
