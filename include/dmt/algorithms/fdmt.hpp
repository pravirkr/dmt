#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Read-only view of a sub-band's intermediate DM-time transform in
 * host memory (see FDMT::view_subband()).
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
 * @brief Zero-copy view of a sub-band's intermediate DM-time transform in
 * device memory (see FDMT::view_subband_device()). The metadata is host
 * memory (plan data).
 */
struct FDMTSubbandDeviceView {
    DeviceSpan<const float> data; ///< Contiguous DM-time data (ndt * nsamps)
    SizeType subband_idx;         ///< (0 <= subband_idx < num_subbands)
    SizeType ndt;                 ///< Number of DM delay trials
    SizeType nsamps;              ///< Number of time samples per delay trial
    float f_start;                ///< Start frequency of this sub-band (MHz)
    float f_end;                  ///< End frequency of this sub-band (MHz)
    std::span<const IndexType> dt_grid; ///< Delay trials in samples
};

/**
 * @brief `fuse_levels` value selecting the fusion depth from the plan (the
 * default). See the `fuse_levels` constructor parameter.
 */
inline constexpr SizeType kFDMTAutoFuse = static_cast<SizeType>(-1);

/**
 * @brief Memory an FDMT engine allocates at construction, in bytes (host
 * memory on the CPU backend, device memory on a GPU backend). execute()
 * allocates nothing further; the caller provides the input and the `output`
 * bytes of output buffer per call.
 */
struct FDMTMemoryUsage {
    SizeType plan;      ///< Plan tables (coordinate DAG, sub-band grids)
    SizeType state;     ///< Internal ping-pong tree state
    SizeType history;   ///< "valid"-mode streaming history
    SizeType workspace; ///< CPU: per-thread fusion/unpack scratch;
                        ///< GPU: fused-kernel index tables, plus the
                        ///< host-memory call staging once allocated
    SizeType output;    ///< Caller-provided dmt buffer per execute() (not
                        ///< owned by the engine)

    /// Total allocated by the engine (excludes `output`).
    [[nodiscard]] SizeType total() const noexcept {
        return plan + state + history + workspace;
    }
};

/**
 * @brief Fast Dispersion Measure Transform (FDMT).
 *
 * Runs on the backend chosen by the last constructor argument (`Exec`):
 * CPU cores with OpenMP, or a GPU. Supports both single-shot transformation
 * and stepped execution for intermediate sub-band inspection. Every backend
 * produces bit-identical output for the same plan.
 *
 * In short:
 * - Input: float32 (nbeams, nchans, nsamps), or packed unsigned 1/2/4/8/16-bit
 *   samples (see the packed execute()).
 * - Output: always float32. Pass nbeams * plan.get_buffer_size() floats; each
 *   beam's leading plan.get_dmt_size() values are the (ndms, dmt_nsamps)
 *   result, and the rest is ping-pong scratch.
 * - Construct once per plan: the constructor allocates everything, and
 *   execute() allocates nothing. In mode "valid", call execute() on
 *   contiguous, non-overlapping blocks to get a contiguous output stream.
 * - Host memory (`std::span`) works on every backend; a GPU backend stages
 *   it through device buffers allocated on the first host call and blocks
 *   until the result is on the host. Device memory (`DeviceSpan`) works on
 *   GPU backends only (the CPU backend throws std::invalid_argument) and is
 *   asynchronous on the given Stream.
 */
class FDMT {
public:
    /**
     * @brief Constructs an FDMT object.
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
     * @param exec Backend and its resources (default: CPU, 1 thread), e.g.
     * `Exec::cpu(8)` or `Exec::cuda(0)`. It takes the place of the old
     * `nthreads` / `device_id` argument; an `int` does not convert to it.
     * @throws std::invalid_argument if `exec.backend` is not in this build
     * (see available_backends()).
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
     * `fuse_levels` tree merges, so the intermediate levels of a group of
     * 2^fuse_levels channels stay in fast memory. Bit-identical output.
     * 0 = the level-by-level path; values above the plan's merge levels are
     * clamped. The stepper (reset()/advance()) always runs level by level.
     * kFDMTAutoFuse (default) picks a depth from the plan alone, with no
     * hardware detection (see get_fuse_levels()):
     * - CPU: groups live in a per-thread, cache-resident scratch buffer; the
     *   deepest depth whose two scratch buffers fit in
     *   max(36 MiB / nthreads, 5 MiB) per thread.
     * - GPU: one thread block per (channel group, time tile) with the
     *   intermediate levels in shared memory; the deepest depth whose tile of
     *   >= 256 samples fits the portable 48 KiB of shared memory with a
     *   level-0 halo of at most half a tile. An explicit depth is also
     *   clamped to 8 and reduced until a tile fits the device's opt-in shared
     *   memory.
     * @param int_tree Performance parameter: for packed low-bit input, store
     * tree levels whose exact value bound fits as uint8/uint16 instead of
     * float (exact; default true). Integer levels cannot be inspected through
     * the stepper's view_* methods; pass false to inspect every level of a
     * packed stepper run. Ignored for float input.
     *
     * All working memory is allocated here (see get_memory_usage()); execute()
     * performs no allocation.
     */
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         IndexType dt_max,
         IndexType dt_min      = 0,
         SizeType dt_step      = 1,
         bool use_box_smearing = true,
         std::string_view mode = "valid",
         Exec exec             = {},
         SizeType nbeams       = 1,
         SizeType fuse_levels  = kFDMTAutoFuse,
         bool int_tree         = true);

    /**
     * @brief Constructs an FDMT instance with a custom delay trial grid.
     *
     * @param dt_grid Explicit list of delay trials in samples (e.g. from
     * generate_optimal_dt_grid).
     * @param f_min, f_max, nchans, nsamps, tsamp, use_box_smearing, mode,
     * exec, nbeams, fuse_levels, int_tree See the first constructor.
     */
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         bool use_box_smearing = true,
         std::string_view mode = "valid",
         Exec exec             = {},
         SizeType nbeams       = 1,
         SizeType fuse_levels  = kFDMTAutoFuse,
         bool int_tree         = true);

    /**
     * @brief Constructs an FDMT instance with a custom physical DM trial
     * grid.
     *
     * @param dm_grid Explicit list of DM trials in pc/cm^3 (supports
     * non-uniform spacing and negative DMs).
     * @param f_min, f_max, nchans, nsamps, tsamp, use_box_smearing, mode,
     * exec, nbeams, fuse_levels, int_tree See the first constructor.
     */
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing = true,
         std::string_view mode = "valid",
         Exec exec             = {},
         SizeType nbeams       = 1,
         SizeType fuse_levels  = kFDMTAutoFuse,
         bool int_tree         = true);

    ~FDMT();
    FDMT(FDMT&&) noexcept;
    FDMT& operator=(FDMT&&) noexcept;
    FDMT(const FDMT&)            = delete;
    FDMT& operator=(const FDMT&) = delete;

    /**
     * @brief Gets the FDMT plan details.
     * @return Constant reference to the FDMTPlan object.
     */
    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;

    /**
     * @brief Number of beams processed together (see the `nbeams`
     * constructor parameter). 1 unless constructed otherwise.
     */
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /// @brief Backend this instance runs on.
    [[nodiscard]] Backend backend() const noexcept;

    /// @brief OpenMP threads used (CPU backend); 1 on a GPU backend.
    [[nodiscard]] int nthreads() const noexcept;

    /// @brief Device ordinal used (GPU backend); -1 on the CPU backend.
    [[nodiscard]] int device() const noexcept;

    /**
     * @brief Executes the full FDMT transform in a single shot, on host
     * memory.
     *
     * @param waterfall Input waterfall data, beam-major flat
     * (nbeams*nchans*nsamps); nbeams=1 (the default) is just (nchans,
     * nsamps).
     * @param dmt Output DM-time array, beam-major flat
     * (nbeams*get_buffer_size()). Only the leading plan.get_dmt_size()
     * values of each beam are the transform; the rest of each beam's slice
     * is ping-pong scratch whose contents depend on the execution path
     * (e.g. the fusion depth) and are unspecified. A GPU backend writes only
     * the leading values and leaves the scratch tail untouched.
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    /**
     * @brief Executes the FDMT transform on packed low-bit integer input in
     * host memory.
     *
     * Each channel row holds nsamps unsigned samples of `nbits` bits,
     * LSB-first within a byte for nbits < 8 (same convention as DDMT),
     * little-endian for 16, padded to a whole byte: layout (nbeams, nchans,
     * ceil(nsamps * nbits / 8)) bytes. There is no 32-bit integer input;
     * use the float overload for 32-bit float data. The output is float32
     * and identical to execute() on the same values converted to float,
     * laid out as for the float overload (only the leading get_dmt_size()
     * values per beam are the result). On a GPU backend only the packed
     * bytes are copied to the device.
     *
     * @param waterfall_packed Packed input, beam-major flat.
     * @param nbits Sample width: 1, 2, 4, 8 or 16.
     * @param dmt Output DM-time array (size >= nbeams*get_buffer_size()).
     * @throws std::invalid_argument for another nbits or a size mismatch.
     */
    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt);

    /**
     * @brief Executes the FDMT transform on device memory (GPU backends).
     *
     * @param d_waterfall Input waterfall data (device memory), laid out as
     * for the host overload.
     * @param d_dmt Output DM-time array (device memory). Only the leading
     * plan.get_dmt_size() values of each beam are the transform; the
     * remainder is unspecified scratch.
     * @param stream Queue to run on (default: the backend's default stream).
     *
     * @note Work is submitted asynchronously to @p stream. Synchronize it
     *       before reading @p d_dmt on the host or reusing the buffers.
     * @throws std::invalid_argument on the CPU backend.
     */
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

    /**
     * @brief Packed low-bit analogue of the device-memory execute().
     * @note Asynchronous on @p stream, like the float overload.
     */
    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream = {});

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
     * @brief Fusion depth execute() uses: the `fuse_levels` constructor
     * argument clamped as described there, or the depth chosen for
     * kFDMTAutoFuse (0 = original level-by-level path).
     */
    [[nodiscard]] SizeType get_fuse_levels() const noexcept;

    /// @brief Whether packed input uses the narrow-integer tree.
    [[nodiscard]] bool get_int_tree() const noexcept;

    /**
     * @brief Memory allocated at construction (host bytes on the CPU
     * backend, device bytes on a GPU backend), plus the output buffer size
     * each execute() call needs.
     */
    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept;

    /**
     * @brief Human-readable summary: the plan (see FDMTPlan::summary()),
     * then the engine configuration (backend, mode, threads or device,
     * beams, fusion depth, integer tree) and its memory breakdown.
     */
    [[nodiscard]] std::string summary() const;

    // =========================================================================
    // Stepper / Hierarchical DP Engine API
    // =========================================================================

    /**
     * @brief Resets and initializes the stepper with a new waterfall block and
     * caller-provided dmt buffer for zero-allocation ping-pong storage (Level
     * 0), both in host memory.
     *
     * @param waterfall Input waterfall data, beam-major flat
     * (nbeams*nchans*nsamps).
     * @param dmt Output DM-time array (size >= nbeams*plan.get_buffer_size()).
     *            Receives the final transform at finalize(). On the CPU
     *            backend it is also one of the two ping-pong scratch buffers;
     *            a GPU backend steps in device staging buffers instead and
     *            writes only each beam's leading get_dmt_size() values.
     *
     * @note When nbeams() > 1, the stepper inspection methods
     * (view_level_data/view_subband_data/view_subband) only expose beam 0's
     * slice -- advance()/advance_until_remaining()/finalize() still process
     * every beam correctly, but per-beam intermediate-level inspection isn't
     * exposed by this API yet.
     * @note Each reset() (like each execute()) consumes one input block. In
     * mode "valid" a block must be advanced to the root (finalize()) before
     * the next reset()/execute(): stopping early would skip the upper
     * levels' cross-block history update, so this throws std::logic_error
     * instead. reset_history() abandons the unfinished block and the stream.
     * The stepper always runs level by level (never fused).
     * @throws std::logic_error if a "valid"-mode block is still unfinished.
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
     * @brief Device-memory analogue of reset() (GPU backends). Binds the
     * input waterfall and output DM-time buffers directly on the device.
     *
     * @note Kernels and memory copies are queued on @p stream, which the
     *       stepper keeps using when advance()/finalize() get no stream.
     *       Synchronize before consuming @p d_dmt or calling reset() again on
     *       another stream unless ordering is guaranteed externally.
     * @throws std::invalid_argument on the CPU backend.
     */
    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream = {});

    /// @brief Packed low-bit analogue of the device-memory reset().
    void reset(DeviceSpan<const uint8_t> d_waterfall_packed,
               SizeType nbits,
               DeviceSpan<float> d_dmt,
               Stream stream = {});

    /**
     * @brief Steps forward by a given number of levels.
     * @param levels Number of levels to advance (default: 1).
     * @param stream GPU queue (default: the one given to reset()). Must be
     * empty on the CPU backend.
     */
    void advance(SizeType levels = 1, Stream stream = {});

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
     * @param stream As for advance().
     */
    void advance_until_remaining(SizeType remaining_levels, Stream stream = {});

    /**
     * @brief Read-only host view of beam 0's intermediate data across all
     * sub-bands at the current level.
     *
     * On the CPU backend this is zero-copy. On a GPU backend it is a host
     * snapshot of the current level (the stepper's queue is synchronized
     * first), shared by all view_* calls until the stepper moves. Either way
     * it is overwritten as later levels are computed.
     */
    [[nodiscard]] std::span<const float> view_level_data() const;

    /**
     * @brief Read-only host view of the data slice for a specific sub-band
     * at current level (see view_level_data()).
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;

    /**
     * @brief Detailed read-only host view and metadata for a specific
     * sub-band at current level (see view_level_data()).
     * @param subband_idx Index of the sub-band (0 <= subband_idx <
     * num_subbands()).
     */
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;

    /**
     * @brief Zero-copy device view of the whole state buffer at the current
     * level (GPU backends): beams are get_buffer_size() apart, and the view
     * covers through the last beam's valid elements.
     * @throws std::invalid_argument on the CPU backend.
     */
    [[nodiscard]] DeviceSpan<const float> view_level_data_device() const;

    /**
     * @brief Zero-copy device view and metadata of a specific sub-band of
     * beam 0 at the current level (GPU backends).
     * @throws std::invalid_argument on the CPU backend.
     */
    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const;

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
     * guaranteed to reside in the dmt buffer provided during reset(). After a
     * host-memory reset() on a GPU backend this blocks until it is there.
     *
     * @param stream As for advance().
     */
    void finalize(Stream stream = {});

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
     * FDMT blocks (a cold start, e.g. for a new observation). Also abandons an
     * unfinished stepper block.
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
     * out to caller-owned host storage, so it can be swapped out and later
     * restored with load_history() -- e.g. to let one shared FDMT instance
     * multiplex several independent streams (each with its own history)
     * rather than requiring one instance per stream.
     *
     * @param out Destination span, size must equal history_state_size().
     * A "full"/"roll" mode instance has history_state_size() == 0, so this
     * is a no-op for those.
     * @throws std::invalid_argument if out.size() != history_state_size().
     */
    void save_history(std::span<float> out) const;

    /**
     * @brief Replaces this instance's current "valid"-mode streaming history
     * with a host buffer previously produced by save_history() (from an
     * instance with the same backend and plan geometry; the layout differs
     * between backends), resuming that stream. Use reset_history() instead to
     * start a stream cold.
     *
     * @param in Source span, size must equal history_state_size().
     * @throws std::invalid_argument if in.size() != history_state_size().
     */
    void load_history(std::span<const float> in);

    /**
     * @brief Device-memory analogue of save_history() (GPU backends).
     * Enqueued on @p stream; synchronize before reading @p out elsewhere.
     * @throws std::invalid_argument on the CPU backend.
     */
    void save_history(DeviceSpan<float> out, Stream stream = {}) const;

    /**
     * @brief Device-memory analogue of load_history() (GPU backends). Fully
     * asynchronous on @p stream; later work on the same stream sees the
     * loaded history.
     * @throws std::invalid_argument on the CPU backend.
     */
    void load_history(DeviceSpan<const float> in, Stream stream = {});

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
 * @param exec Backend and its resources (default: CPU, 1 thread).
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
             Exec exec             = {},
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
             Exec exec             = {},
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
             Exec exec             = {},
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

} // namespace dmt::algorithms
