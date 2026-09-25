#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Zero-copy read-only view of a sub-band's intermediate DM-time
 * transform.
 */
struct FDMTSubbandView {
    std::span<const float> data; ///< Contiguous DM-time data (ndt * nsamps)
    SizeType subband_idx;        ///< (0 <= subband_idx < num_subbands)
    SizeType ndt;                ///< Number of DM delay trials
    SizeType nsamps;             ///< Number of time samples per delay trial
    float f_start;               ///< Start frequency of this sub-band (MHz)
    float f_end;                 ///< End frequency of this sub-band (MHz)
    std::span<const IndexType> dt_grid; ///< Delay trials in samples
};

/**
 * @brief `fuse_levels` value selecting the fusion depth from the plan (the
 * default). See the `fuse_levels` constructor parameter.
 */
inline constexpr SizeType kFDMTAutoFuse = static_cast<SizeType>(-1);

/**
 * @brief Memory an FDMT engine allocates at construction, in bytes (host
 * memory for FDMTCPU, device memory for FDMTCUDA). execute() allocates
 * nothing further; the caller provides the input and the `output` bytes of
 * output buffer per call.
 */
struct FDMTMemoryUsage {
    SizeType plan;      ///< Plan tables (coordinate DAG, sub-band grids)
    SizeType state;     ///< Internal ping-pong tree state
    SizeType history;   ///< "valid"-mode streaming history
    SizeType workspace; ///< CPU: per-thread fusion/unpack scratch;
                        ///< CUDA: fused-kernel index tables
    SizeType output;    ///< Caller-provided dmt buffer per execute() (not
                        ///< owned by the engine)

    /// Total allocated by the engine (excludes `output`).
    [[nodiscard]] SizeType total() const noexcept {
        return plan + state + history + workspace;
    }
};

/**
 * @brief Fast Dispersion Measure Transform (FDMT) for CPU execution.
 *
 * Performs the FDMT algorithm using CPU cores with optional OpenMP
 * parallelization. Supports both single-shot transformation and stepped
 * execution for intermediate sub-band inspection.
 */
class FDMTCPU {
public:
    /**
     * @brief Constructs an FDMT object for the CPU execution.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels.
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_max Maximum delay trial (in samples).
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param dt_step Stride between delay trials (default: 1).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * using boxcar summation (default: true).
     * @param mode Mode of the FDMT transform. Available modes are:
     * - "full": Full FDMT transform of every sample in the input waterfall.
     * - "valid": Valid FDMT transform for samples upto dt_max. Every merge
     *   node that needs one keeps a small cross-block history slot (see
     *   FDMTCoord::hist_offset), so repeated calls to execute() (or the
     *   reset()/advance()/finalize() stepper) on consecutive, non-overlapping
     *   blocks reproduce monolithic full-mode execution bit-exactly for
     *   every trial, not just dt=0 -- an overlap-save scheme applied inside
     *   the tree rather than to the raw input. Call reset_history() to
     *   restart streaming from a cold state (e.g. for a new observation).
     * - "roll": Roll FDMT transform using rotation of the input waterfall.
     * (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param nthreads Number of OpenMP threads to use (default: 1).
     * @param nbeams Number of independent beams to process together
     * (default: 1). The plan/coordinate DAG is shared across all beams
     * (dedispersion delays don't depend on beam), so this only scales the
     * state/history buffers and adds an outer beam loop around the existing
     * per-coordinate execution -- at nbeams=1 this is the same code path and
     * layout as before this parameter existed. `waterfall`/`dmt` become
     * beam-major: shape (nbeams, nchans, nsamps) / (nbeams, ndms, nsamps)
     * flattened, beam b at offset b*nchans*nsamps / b*get_buffer_size().
     * @param fuse_levels Performance parameter (most users keep the
     * default): execute() fuses level-0 initialisation with the first
     * `fuse_levels` tree merges, processing channels in groups of
     * 2^fuse_levels whose intermediate levels stay in a per-thread,
     * cache-resident scratch buffer instead of travelling to and from main
     * memory. Bit-identical output. 0 = the original level-by-level path;
     * values above the plan's merge levels are clamped. kFDMTAutoFuse
     * (default) picks the deepest depth whose two scratch buffers fit in
     * max(36 MiB / nthreads, 5 MiB) per thread -- a rule on the plan alone,
     * with no hardware detection (see get_fuse_levels()). The stepper
     * (reset()/advance()) always runs level by level.
     * @param int_tree Performance parameter: for packed low-bit input, store
     * tree levels whose exact value bound fits as uint8/uint16 instead of
     * float (exact; default true). Integer levels cannot be inspected through
     * the stepper's view_* methods; pass false to inspect every level of a
     * packed stepper run. Ignored for float input.
     *
     * All working memory is allocated here (see get_memory_usage()); execute()
     * performs no allocation.
     */
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            IndexType dt_max,
            IndexType dt_min      = 0,
            SizeType dt_step      = 1,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            int verbose           = 0,
            int nthreads          = 1,
            SizeType nbeams       = 1,
            SizeType fuse_levels  = kFDMTAutoFuse,
            bool int_tree         = true);

    /**
     * @brief Constructs an FDMTCPU instance with a custom delay trial grid.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels (power of 2).
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_grid Explicit list of delay trials in samples (e.g. from
     * generate_optimal_dt_grid).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * (default: true).
     * @param mode Mode: "valid", "full", or "roll" (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param nthreads Number of OpenMP threads to use (default: 1).
     * @param nbeams Number of independent beams to process together (default:
     * 1).
     * @param fuse_levels, int_tree Performance parameters (see the first
     * constructor).
     */
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<IndexType>& dt_grid,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            int verbose           = 0,
            int nthreads          = 1,
            SizeType nbeams       = 1,
            SizeType fuse_levels  = kFDMTAutoFuse,
            bool int_tree         = true);

    /**
     * @brief Constructs an FDMTCPU instance with a custom physical DM trial
     * grid.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels (power of 2).
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dm_grid Explicit list of DM trials in pc/cm^3 (supports
     * non-uniform spacing and negative DMs).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * (default: true).
     * @param mode Mode: "valid", "full", or "roll" (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param nthreads Number of OpenMP threads to use (default: 1).
     * @param nbeams Number of independent beams to process together (default:
     * 1).
     * @param fuse_levels, int_tree Performance parameters (see the first
     * constructor).
     */
    FDMTCPU(float f_min,
            float f_max,
            SizeType nchans,
            SizeType nsamps,
            float tsamp,
            const std::vector<float>& dm_grid,
            bool use_box_smearing = true,
            std::string_view mode = "valid",
            int verbose           = 0,
            int nthreads          = 1,
            SizeType nbeams       = 1,
            SizeType fuse_levels  = kFDMTAutoFuse,
            bool int_tree         = true);

    ~FDMTCPU();
    FDMTCPU(FDMTCPU&&) noexcept;
    FDMTCPU& operator=(FDMTCPU&&) noexcept;
    FDMTCPU(const FDMTCPU&)            = delete;
    FDMTCPU& operator=(const FDMTCPU&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    const plans::FDMTPlan& get_plan() const noexcept;

    /**
     * @brief Number of beams processed together (see the `nbeams`
     * constructor parameter). 1 unless constructed otherwise.
     */
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Executes the full FDMT transform in a single shot.
     *
     * @param waterfall Input waterfall data, beam-major flat
     * (nbeams*nchans*nsamps); nbeams=1 (the default) is just (nchans,
     * nsamps).
     * @param dmt Output DM-time array, beam-major flat
     * (nbeams*get_buffer_size()). Only the leading plan.get_dmt_size()
     * values of each beam are the transform; the rest of each beam's slice
     * is ping-pong scratch whose contents depend on the execution path
     * (e.g. the fusion depth) and are unspecified.
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Executes the FDMT transform on packed low-bit integer input.
     *
     * Each channel row holds nsamps unsigned samples of `nbits` bits,
     * LSB-first within a byte for nbits < 8 (same convention as DDMTCPU),
     * padded to a whole byte: layout (nbeams, nchans,
     * packed_row_bytes(nsamps, nbits)). The output is float and identical to
     * execute() on the same values converted to float.
     *
     * @param waterfall_packed Packed input, beam-major flat.
     * @param nbits Sample width: 1, 2, 4, 8 or 16.
     * @param dmt Output DM-time array (size >= nbeams*get_buffer_size()).
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt);

    /**
     * @brief Fusion depth execute() uses: the `fuse_levels` constructor
     * argument clamped to the plan's merge levels, or the depth chosen for
     * kFDMTAutoFuse (0 = original level-by-level path).
     */
    [[nodiscard]] SizeType get_fuse_levels() const noexcept;

    /// @brief Whether packed input uses the narrow-integer tree.
    [[nodiscard]] bool get_int_tree() const noexcept;

    /**
     * @brief Memory allocated at construction (host bytes), plus the output
     * buffer size each execute() call needs.
     */
    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept;

    // =========================================================================
    // Stepper / Hierarchical DP Engine API
    // =========================================================================

    /**
     * @brief Resets and initializes the stepper with a new waterfall block and
     * caller-provided dmt buffer for zero-allocation ping-pong storage (Level
     * 0).
     *
     * @param waterfall Input waterfall data, beam-major flat
     * (nbeams*nchans*nsamps).
     * @param dmt Output DM-time array (size >= nbeams*plan.get_buffer_size()).
     *            Used as one of the two ping-pong scratch buffers and receives
     *            the final transform at finalize().
     *
     * @note When nbeams() > 1, the stepper inspection methods
     * (view_level_data/view_subband_data/view_subband) only expose beam 0's
     * slice -- advance()/advance_until_remaining()/finalize() still process
     * every beam correctly, but per-beam intermediate-level inspection isn't
     * exposed by this API yet.
     */
    void reset(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Packed low-bit analogue of reset() (see the packed execute()).
     *
     * @note With int_tree enabled (the default), levels stored as integers
     * cannot be inspected: the view_* methods throw std::logic_error at such
     * a level.
     */
    void reset(std::span<const uint8_t> waterfall_packed,
               SizeType nbits,
               std::span<float> dmt);

    /**
     * @brief Steps forward by a given number of levels.
     * @param levels Number of levels to advance (default: 1).
     */
    void advance(SizeType levels = 1);

    /**
     * @brief Advances execution until N levels remain before the root.
     *
     * Examples:
     * - remaining_levels = 1: Stops 1 level before root (2 children sub-bands
     * remaining).
     * - remaining_levels = 2: Stops 2 levels before root (4 children sub-bands
     * remaining).
     * - remaining_levels = 0: Advances to completion (root level).
     *
     * @param remaining_levels Target number of levels remaining before root.
     */
    void advance_until_remaining(SizeType remaining_levels);

    /**
     * @brief Read-only view of all intermediate data across all sub-bands at
     * current level.
     */
    [[nodiscard]] std::span<const float> view_level_data() const;

    /**
     * @brief Read-only view of the data slice for a specific sub-band at
     * current level.
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;

    /**
     * @brief Detailed read-only view and metadata for a specific sub-band at
     * current level.
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;

    /**
     * @brief Current level index (0 = initialised waterfall, total_levels() - 1
     * = root).
     */
    [[nodiscard]] SizeType current_level() const noexcept;

    /**
     * @brief Total number of levels (niters + 1).
     */
    [[nodiscard]] SizeType total_levels() const noexcept;

    /**
     * @brief Number of levels remaining before root (total_levels() - 1 -
     * current_level()).
     */
    [[nodiscard]] SizeType remaining_levels() const noexcept;

    /**
     * @brief Number of active sub-bands at the current level.
     */
    [[nodiscard]] SizeType num_subbands() const;

    /**
     * @brief Check if execution has reached the final root level.
     */
    [[nodiscard]] bool is_finished() const noexcept;

    /**
     * @brief Finalizes execution to root level.
     *
     * Advances all remaining levels to the root. The final root transform is
     * guaranteed to reside in the dmt buffer provided during reset().
     */
    void finalize();

    /**
     * @brief Theoretical noise variance for a given DM trial and boxcar width.
     */
    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise standard deviation for a given DM trial and
     * boxcar width.
     */
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise variance grid across all DM trials for a boxcar
     * width.
     */
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise standard deviation grid across all DM trials for
     * a boxcar width.
     */
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    /**
     * @brief Resets the internal history buffer for valid-mode streaming across
     * FDMT blocks.
     */
    void reset_history() noexcept;

    /**
     * @brief Size (in floats) of this instance's "valid"-mode streaming
     * history state, as used by save_history()/load_history(). Zero for
     * "full"/"roll" mode instances.
     */
    [[nodiscard]] SizeType history_state_size() const noexcept;

    /**
     * @brief Copies this instance's current "valid"-mode streaming history
     * out to caller-owned storage, so it can be swapped out and later
     * restored with load_history() -- e.g. to let one shared FDMTCPU
     * instance multiplex several independent streams (each with its own
     * history) rather than requiring one instance per stream.
     *
     * @param out Destination span, size must equal history_state_size().
     * A "full"/"roll" mode instance has history_state_size() == 0, so this
     * is a no-op for those.
     * @throws std::invalid_argument if out.size() != history_state_size().
     */
    void save_history(std::span<float> out) const;

    /**
     * @brief Replaces this instance's current "valid"-mode streaming history
     * with a buffer previously produced by save_history() (from an instance
     * built with the same plan geometry), resuming that stream. Use
     * reset_history() instead to start a stream cold.
     *
     * @param in Source span, size must equal history_state_size().
     * @throws std::invalid_argument if in.size() != history_state_size().
     */
    void load_history(std::span<const float> in);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief High-level convenience function to execute FDMT on a waterfall block
 * with a linear delay grid.
 *
 * @param waterfall Input waterfall data, beam-major flat
 * (nbeams*nchans*nsamps).
 * @param f_min Bottom edge frequency in MHz.
 * @param f_max Top edge frequency in MHz.
 * @param nchans Number of frequency channels (power of 2).
 * @param nsamps Number of time samples per block.
 * @param tsamp Sampling interval in seconds.
 * @param dt_max Maximum delay trial in samples.
 * @param dt_min Minimum delay trial in samples (default: 0).
 * @param dt_step Stride between delay trials (default: 1).
 * @param use_box_smearing Whether to account for intra-channel smearing
 * (default: true).
 * @param mode Mode: "valid", "full", or "roll" (default: "valid").
 * @param verbose 0 = silent, 1 = info, 2 = debug.
 * @param nthreads Number of OpenMP threads (default: 1).
 * @param nbeams Number of batched beams (default: 1).
 * @return Tuple of (transformed DMT buffer as std::vector<float>, FDMTPlan
 * object).
 */
[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             IndexType dt_max,
             IndexType dt_min      = 0,
             SizeType dt_step      = 1,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int nthreads          = 1,
             SizeType nbeams       = 1);

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<IndexType>& dt_grid,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int nthreads          = 1,
             SizeType nbeams       = 1);

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int nthreads          = 1,
             SizeType nbeams       = 1);

/**
 * @brief Injects a synthetic dispersed pulse ("FRB track") into a waterfall
 * buffer so that it lands, after running the FDMT transform, exactly on a
 * given final DM/dt trial at a chosen time sample.
 *
 * Built on plans::FDMTPlan::trace_dm(), which decompiles trial @p dm_idx's
 * coordinate lineage into a per-channel sample shift. Adds @p amplitude
 * (does not overwrite) to @p width consecutive samples per channel, so this
 * can be used to inject a test pulse into real or simulated noise.
 *
 * @param waterfall Waterfall buffer to inject into, shape (nchans, nsamps)
 *                  matching plan.get_nchans()/get_nsamps(); modified in
 *                  place.
 * @param plan The FDMT plan whose trial grid and channel layout to target.
 * @param dm_idx Index into the plan's final DM/dt trial grid.
 * @param amplitude Amplitude added per channel, per injected sample.
 * @param toffset Reference-channel time sample at which the pulse should
 *                peak after running FDMT (default: 0).
 * @param width Number of consecutive samples per channel to inject, for a
 *              simple top-hat pulse shape instead of a single impulse
 *              (default: 1).
 * @throws std::invalid_argument if width == 0 or waterfall has the wrong
 *         size.
 * @throws std::out_of_range if dm_idx is out of range, or if the injected
 *         samples would fall outside [0, nsamps) for any channel.
 */
void add_frb_track(std::span<float> waterfall,
                   const plans::FDMTPlan& plan,
                   SizeType dm_idx,
                   float amplitude   = 1.0F,
                   IndexType toffset = 0,
                   SizeType width    = 1);

#ifdef DMT_ENABLE_CUDA
/**
 * @brief Zero-copy read-only view of a sub-band's intermediate DM-time
 * transform residing in CUDA device memory.
 */
struct FDMTSubbandViewCUDA {
    cuda::std::span<const float> data; ///< Contiguous DM-time (ndt * nsamps)
    SizeType subband_idx;              ///< (0 <= subband_idx < num_subbands)
    SizeType ndt;                      ///< Number of DM delay trials
    SizeType nsamps; ///< Number of time samples per delay trial
    float f_start;   ///< Start frequency of this sub-band (MHz)
    float f_end;     ///< End frequency of this sub-band (MHz)
    std::span<const IndexType> dt_grid; ///< Delay trials in samples
};

using FDMTCUDASubbandView = FDMTSubbandViewCUDA;

/**
 * @brief Fast Dispersion Measure Transform (FDMT) for CUDA execution.
 *
 * Performs the FDMT algorithm using CUDA-enabled GPUs for high-performance
 * processing. Supports single-shot execution (host/device memory) as well as
 * interactive hierarchical stepped execution on device memory for intermediate
 * sub-band matched-filtering pipelines.
 */
class FDMTCUDA {
public:
    /**
     * @brief Constructs an FDMT object for the CUDA execution.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels.
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_max Maximum delay trial (in samples).
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param dt_step Delay trial step (in samples, default: 1).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * using boxcar summation (default: true).
     * @param mode Mode of the FDMT transform. Available modes are:
     * - "full": Full FDMT transform of every sample in the input waterfall.
     * - "valid": Valid FDMT transform for samples upto dt_max. Every merge
     *   node that needs one keeps a small cross-block history slot (see
     *   FDMTCoord::hist_offset), so repeated calls to execute() (or the
     *   reset()/advance()/finalize() stepper) on consecutive, non-overlapping
     *   blocks reproduce monolithic full-mode execution bit-exactly for
     *   every trial, not just dt=0 -- an overlap-save scheme applied inside
     *   the tree rather than to the raw input. Mirrors FDMTCPU's mechanism;
     *   see kernel_execute_iter's doc comment in fdmt_cuda.cu for the one
     *   CUDA-specific difference (a ping-ponged pair of history buffers,
     *   needed because samples within a coordinate are processed in
     *   parallel here, unlike the CPU's one-thread-per-coordinate loop).
     *   Call reset_history() to restart streaming from a cold state.
     * - "roll": Roll FDMT transform using rotation of the input waterfall.
     * (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param device_id CUDA device ID to use (default: 0).
     * @param nbeams Number of independent beams to process together
     * (default: 1).
     * @param fuse_levels Performance parameter (most users keep the
     * default): the device execute() overloads fuse level-0 initialisation
     * with the first `fuse_levels` merges in one kernel, one thread block per
     * (channel group, time tile) with the intermediate levels in shared
     * memory. Bit-identical output. 0 = level-by-level kernels. An explicit
     * depth is clamped to the plan's merge levels and to 8, and reduced
     * until a tile fits the device's opt-in shared memory. kFDMTAutoFuse
     * (default) picks the deepest depth whose tile of >= 256 samples fits
     * the portable 48 KiB of shared memory with a level-0 halo of at most
     * half a tile (see get_fuse_levels()).
     * @param int_tree Performance parameter: narrow-integer tree state for
     * packed input, as on the CPU (default true).
     */
    FDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             IndexType dt_max,
             IndexType dt_min      = 0,
             SizeType dt_step      = 1,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int device_id         = 0,
             SizeType nbeams       = 1,
             SizeType fuse_levels  = kFDMTAutoFuse,
             bool int_tree         = true);

    /**
     * @brief Constructs an FDMTCUDA object with a custom delay trial grid.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels (power of 2).
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_grid Explicit list of delay trials in samples.
     * @param use_box_smearing Whether to account for intra-channel smearing
     * (default: true).
     * @param mode Mode: "valid", "full", or "roll" (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param device_id CUDA device ID to use (default: 0).
     * @param nbeams Number of independent beams to process together (default:
     * 1).
     * @param fuse_levels, int_tree Performance parameters (see the first
     * constructor).
     */
    FDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<IndexType>& dt_grid,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int device_id         = 0,
             SizeType nbeams       = 1,
             SizeType fuse_levels  = kFDMTAutoFuse,
             bool int_tree         = true);

    /**
     * @brief Constructs an FDMTCUDA object with a custom DM trial grid.
     *
     * @param f_min Frequency of the lowest channel (MHz).
     * @param f_max Frequency of the highest channel (MHz).
     * @param nchans Number of frequency channels (power of 2).
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dm_grid Explicit list of DM trials in pc/cm^3 (supports
     * non-uniform and negative DMs).
     * @param use_box_smearing Whether to account for intra-channel smearing
     * (default: true).
     * @param mode Mode: "valid", "full", or "roll" (default: "valid").
     * @param verbose 0 = silent, 1 = info, 2 = debug.
     * @param device_id CUDA device ID to use (default: 0).
     * @param nbeams Number of independent beams to process together (default:
     * 1).
     * @param fuse_levels, int_tree Performance parameters (see the first
     * constructor).
     */
    FDMTCUDA(float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             bool use_box_smearing = true,
             std::string_view mode = "valid",
             int verbose           = 0,
             int device_id         = 0,
             SizeType nbeams       = 1,
             SizeType fuse_levels  = kFDMTAutoFuse,
             bool int_tree         = true);

    ~FDMTCUDA();
    FDMTCUDA(FDMTCUDA&&) noexcept;
    FDMTCUDA& operator=(FDMTCUDA&&) noexcept;
    FDMTCUDA(const FDMTCUDA&)            = delete;
    FDMTCUDA& operator=(const FDMTCUDA&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    const plans::FDMTPlan& get_plan() const noexcept;

    /**
     * @brief Executes the FDMT transform using host memory.
     *
     * Input and output data reside on the host. This involves transferring data
     * to/from the device internally.
     *
     * @param waterfall Input waterfall data view (host memory).
     * @param dmt Output DM-time array view (host memory).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Executes the FDMT transform using device memory.
     *
     * Input and output data pointers refer to memory allocated on the CUDA
     * device. Uses libcudacxx span (`cuda::std::span`) for device memory views.
     * Allows specifying a CUDA stream for asynchronous execution.
     *
     * @param d_waterfall Input waterfall data view (device memory)
     * @param d_dmt Output DM-time array view (device memory). As on the
     * CPU, only the leading plan.get_dmt_size() values of each beam are the
     * transform; the remainder is unspecified scratch.
     * @param stream The CUDA stream to execute the transform on.
     *
     * @note Work is submitted asynchronously to @p stream. Call
     *       cudaStreamSynchronize(stream) (or an equivalent ordering guarantee)
     *       before reading @p d_dmt on the host or reusing the buffers.
     */
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    /**
     * @brief Executes the FDMT transform on packed low-bit input in host
     * memory (see FDMTCPU::execute(std::span<const uint8_t>, ...) for the
     * layout). Only the packed bytes are copied to the device -- 32 / nbits
     * times less PCIe traffic than float input. The output is float and
     * matches the float execute() on the same values.
     *
     * @param waterfall_packed Packed input (host), (nbeams, nchans,
     * packed_row_bytes(nsamps, nbits)).
     * @param nbits Sample width: 1, 2, 4, 8 or 16.
     * @param dmt Output DM-time array (host).
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt);

    /**
     * @brief Packed low-bit analogue of the device-memory execute().
     * @note Asynchronous on @p stream, like the float overload.
     */
    void execute(cuda::std::span<const uint8_t> d_waterfall_packed,
                 SizeType nbits,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    template <typename Alloc1 = std::allocator<float>,
              typename Alloc2 = std::allocator<float>>
    void execute(const std::vector<float, Alloc1>& waterfall,
                 std::vector<float, Alloc2>& dmt) {
        execute(std::span<const float>(waterfall), std::span<float>(dmt));
    }

    template <typename Alloc1 = std::allocator<uint8_t>,
              typename Alloc2 = std::allocator<float>>
    void execute(const std::vector<uint8_t, Alloc1>& waterfall_packed,
                 SizeType nbits,
                 std::vector<float, Alloc2>& dmt) {
        execute(std::span<const uint8_t>(waterfall_packed), nbits,
                std::span<float>(dmt));
    }

    /**
     * @brief Packed low-bit analogue of reset() (device memory).
     *
     * @note With int_tree enabled (the default), levels stored as integers
     * cannot be inspected: the view_* methods throw std::logic_error at such
     * a level.
     */
    void reset(cuda::std::span<const uint8_t> d_waterfall_packed,
               SizeType nbits,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr);

    /**
     * @brief Fusion depth the device execute() overloads use (0 = unfused);
     * see the `fuse_levels` constructor parameter.
     */
    [[nodiscard]] SizeType get_fuse_levels() const noexcept;

    /// @brief Whether packed input uses the narrow-integer tree.
    [[nodiscard]] bool get_int_tree() const noexcept;

    /**
     * @brief Device memory allocated at construction, plus the output buffer
     * size each execute() call needs. The host-memory execute() overloads
     * additionally stage the input and output on the device per call.
     */
    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept;

    /**
     * @brief Resets and initializes the stepped execution state using device
     * memory.
     *
     * Prepares the internal pipeline buffers for hierarchical stepping,
     * binding the input waterfall and output DM-time buffers directly on the
     * device.
     *
     * @param d_waterfall 2D input array on device (nchans * nsamps).
     * @param d_dmt 2D output array on device (capacity at least
     * get_dmt_size()).
     * @param stream CUDA stream for asynchronous execution (default: nullptr).
     * @throws std::invalid_argument If the input or output buffers are too
     * small.
     *
     * @note Kernels and memory copies are queued on @p stream. Synchronize
     *       before consuming @p d_dmt or calling reset() again on another
     *       stream unless ordering is guaranteed externally.
     */
    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr);

    /**
     * @brief Advances execution by the specified number of levels on the CUDA
     * device.
     *
     * @param levels Number of hierarchical levels to advance (default: 1).
     *               If levels exceeds the remaining levels, advances to the
     * root.
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       results.
     */
    void advance(SizeType levels = 1, cudaStream_t stream = nullptr);

    /**
     * @brief Advances execution until a given number of levels remain before
     * the root.
     *
     * Commonly used to pause 1 level before root (2 sub-bands) or 2 levels
     * before root (4 sub-bands) for sub-band search / matched filtering.
     *
     * @param remaining_levels Target remaining levels before the root.
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       results.
     */
    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream = nullptr);

    /**
     * @brief Obtains a read-only device span of the entire buffer at the
     * current level.
     *
     * @return Read-only device view of the active state buffer.
     * @throws std::logic_error If reset() was not called first.
     */
    cuda::std::span<const float> view_level_data() const;

    /**
     * @brief Obtains a read-only device span of a specific sub-band at the
     * current level.
     *
     * @param subband_idx Sub-band index in [0, num_subbands()).
     * @return Read-only device view of the sub-band's DM-time array (ndt *
     * nsamps floats).
     * @throws std::logic_error If reset() was not called first.
     * @throws std::out_of_range If subband_idx >= num_subbands().
     */
    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const;

    /**
     * @brief Obtains a rich descriptor view of a specific sub-band on the
     * device.
     *
     * Provides frequency boundaries, DM trial counts, and a device span.
     *
     * @param subband_idx Sub-band index in [0, num_subbands()).
     * @return FDMTSubbandViewCUDA containing sub-band metadata and device span.
     * @throws std::logic_error If reset() was not called first.
     * @throws std::out_of_range If subband_idx >= num_subbands().
     */
    FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const;

    /**
     * @brief Current hierarchical level of the transform (0 = initial
     * waterfall).
     */
    SizeType current_level() const noexcept;

    /**
     * @brief Total number of levels in the hierarchy (niters + 1).
     */
    SizeType total_levels() const noexcept;

    /**
     * @brief Number of levels remaining before reaching the root level.
     */
    SizeType remaining_levels() const noexcept;

    /**
     * @brief Number of sub-bands present at the current level.
     * @throws std::logic_error If reset() was not called first.
     */
    SizeType num_subbands() const;

    /**
     * @brief Checks if execution has reached the root level.
     */
    bool is_finished() const noexcept;

    /**
     * @brief Advances through any remaining levels to the root level.
     *
     * @param stream CUDA stream to use (default: nullptr, uses stream from
     * reset if nullptr).
     * @throws std::logic_error If reset() was not called first.
     *
     * @note Asynchronous on the active CUDA stream; synchronize before reading
     *       the final transform in @p d_dmt passed to reset().
     */
    void finalize(cudaStream_t stream = nullptr);

    /**
     * @brief Theoretical noise variance for a given DM trial and boxcar
     * width. Identical formula to FDMTCPU::get_effective_variance (pure
     * plan-side math, independent of which backend executed the transform).
     */
    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise standard deviation for a given DM trial and
     * boxcar width.
     */
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise variance grid across all DM trials for a boxcar
     * width.
     */
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;

    /**
     * @brief Theoretical noise standard deviation grid across all DM trials for
     * a boxcar width.
     */
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    /**
     * @brief Number of beams processed concurrently by this instance.
     */
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Resets the internal history buffers for valid-mode streaming
     * across FDMT blocks. Mirrors FDMTCPU::reset_history().
     */
    void reset_history() noexcept;

    /**
     * @brief Size (in floats) of this instance's "valid"-mode streaming
     * history state (device-resident), as used by
     * save_history()/load_history(). Mirrors FDMTCPU::history_state_size().
     */
    [[nodiscard]] SizeType history_state_size() const noexcept;

    /**
     * @brief Copies this instance's current streaming history out to a
     * caller-owned device buffer, so one shared FDMTCUDA instance can
     * multiplex several independent streams (each with its own history)
     * rather than requiring one instance per stream. Mirrors
     * FDMTCPU::save_history().
     *
     * @param out Destination device span, size must equal
     * history_state_size().
     * @param stream CUDA stream to enqueue the copy on; synchronize before
     * reading @p out elsewhere.
     * @throws std::invalid_argument if out.size() != history_state_size().
     */
    void save_history(cuda::std::span<float> out,
                      cudaStream_t stream = nullptr) const;

    /**
     * @brief Replaces this instance's current streaming history with a
     * device buffer previously produced by save_history() (from an instance
     * built with the same plan geometry), resuming that stream. Mirrors
     * FDMTCPU::load_history(). Use reset_history() instead to start a stream
     * cold.
     *
     * @param in Source device span, size must equal history_state_size().
     * @param stream CUDA stream to enqueue the copy on. Briefly synchronizes
     * internally to read back a small parity flag onto the host.
     * @throws std::invalid_argument if in.size() != history_state_size().
     */
    void load_history(cuda::std::span<const float> in,
                      cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Convenience function to run FDMT on GPU device using host memory
 * views.
 *
 * @param waterfall Input waterfall data on host.
 * @param f_min Bottom edge frequency in MHz.
 * @param f_max Top edge frequency in MHz.
 * @param nchans Number of frequency channels.
 * @param nsamps Number of time samples per block.
 * @param tsamp Sampling interval in seconds.
 * @param dt_max Maximum delay trial in samples.
 * @param dt_min Minimum delay trial in samples (default: 0).
 * @param dt_step Stride between delay trials (default: 1).
 * @param use_box_smearing Whether to account for intra-channel smearing
 * (default: true).
 * @param mode Mode: "valid", "full", or "roll" (default: "valid").
 * @param verbose 0 = silent, 1 = info, 2 = debug.
 * @param device_id CUDA device ID.
 * @param nbeams Number of batched beams.
 * @return Transformed DMT buffer on host.
 */
std::vector<float> compute_fdmt_cuda(std::span<const float> waterfall,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     SizeType nsamps,
                                     float tsamp,
                                     IndexType dt_max,
                                     IndexType dt_min      = 0,
                                     SizeType dt_step      = 1,
                                     bool use_box_smearing = true,
                                     std::string_view mode = "valid",
                                     int verbose           = 0,
                                     int device_id         = 0,
                                     SizeType nbeams       = 1);

std::vector<float> compute_fdmt_cuda(std::span<const float> waterfall,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     SizeType nsamps,
                                     float tsamp,
                                     const std::vector<IndexType>& dt_grid,
                                     bool use_box_smearing = true,
                                     std::string_view mode = "valid",
                                     int verbose           = 0,
                                     int device_id         = 0,
                                     SizeType nbeams       = 1);

std::vector<float> compute_fdmt_cuda(std::span<const float> waterfall,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     SizeType nsamps,
                                     float tsamp,
                                     const std::vector<float>& dm_grid,
                                     bool use_box_smearing = true,
                                     std::string_view mode = "valid",
                                     int verbose           = 0,
                                     int device_id         = 0,
                                     SizeType nbeams       = 1);

#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
