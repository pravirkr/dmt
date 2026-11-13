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
    SizeType chan_begin;                ///< First input channel
    SizeType chan_end;                  ///< One past the last input channel
    bool staggered; ///< Overlap subband straddling a tree boundary
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
    SizeType chan_begin;                ///< First input channel
    SizeType chan_end;                  ///< One past the last input channel
    bool staggered; ///< Overlap subband straddling a tree boundary
};

/**
 * @brief `fuse_levels` value selecting the fusion depth from the plan (the
 * default). See the `fuse_levels` constructor parameter.
 */
inline constexpr SizeType kFDMTAutoFuse = static_cast<SizeType>(-1);

/**
 * Default memory budget of FDMT and FDMTBoxcar (10 GiB): everything the
 * instance allocates (FDMTMemoryUsage::total(); for FDMTBoxcar
 * FDMTBoxcarMemoryUsage::budgeted(), device memory only on a GPU), not the
 * caller's input and output buffers. Within the budget
 * the fastest streaming path is taken; a plan whose tree history does not
 * fit (e.g. 6656 channels at 64 us to DM 2000) takes the skewed-window
 * path with the longest automatic chunk that fits. Construction throws
 * std::invalid_argument when no path fits.
 */
inline constexpr SizeType kFDMTDefaultMaxMemory = SizeType{10} << 30;

/**
 * @brief Memory an FDMT engine allocates at construction, in bytes (host
 * memory on the CPU backend, device memory on a GPU backend). execute()
 * allocates nothing further; the caller provides the input and the `output`
 * bytes of output buffer per call.
 */
struct FDMTMemoryUsage {
    SizeType plan;      ///< Plan tables (coordinate DAG, sub-band grids)
    SizeType state;     ///< Internal tree state (two levels of a chunk)
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
 * and stepped execution for intermediate sub-band inspection. Packed paths,
 * integer-valued input, explicit fusion depths and the CPU tree are
 * bit-identical for the same plan. Automatic GPU direct levels agree with
 * that tree within float rounding (see fuse_levels) and are not bit-identical
 * across chunks.
 *
 * In short:
 * - Input: float32 (nbeams, nchans, nsamps), or packed unsigned 1/2/4/8/16-bit
 *   samples (see the packed execute()).
 * - Output: always float32, (nbeams, ndms, dmt_nsamps): pass at least
 *   nbeams * plan.get_dmt_size() floats. The engine owns all its working
 *   state; the output buffer is only written.
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
     * @param f_min Bottom edge of the band (MHz); also the reference
     * frequency of output times (FDMTPlan::get_f_ref()).
     * @param f_max Top edge of the band (MHz).
     * @param nchans Number of frequency channels.
     * @param nsamps Number of time samples per incoming block.
     * @param tsamp Sampling time (s).
     * @param dt_max Maximum delay trial (in samples).
     * @param dt_min Minimum delay trial (in samples, default: 0).
     * @param dt_step Stride between delay trials (default: 1).
     * @param use_box_smearing true (default): each channel contributes the
     * mean of its level-0 box (|dt|+1 samples, upper- to lower-edge
     * arrival), a matched filter for intra-channel smearing that also
     * low-pass filters (colours) the noise. false: the channel's upper-edge
     * sample only, so white input gives white output (as DDMT), for
     * periodicity / power-spectrum searches.
     * @param mode Mode of the FDMT transform. Available modes are:
     * - "full": Full FDMT transform of every sample in the input waterfall
     *   (nsamps + dt_max outputs).
     * - "valid": nsamps outputs per block. The first
     *   FDMTPlan::get_startup_samples() outputs after construction or
     *   reset_history() are partial sums over zero-filled history (they are
     *   written, so discard or mask them); later blocks are complete. Every
     * merge node that needs one keeps a small cross-block history slot (see
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
     * flattened, beam b at offset b*nchans*nsamps / b*get_dmt_size().
     * @param fuse_levels Performance parameter (most users keep the
     * default): execute() fuses level-0 initialisation with the first
     * `fuse_levels` tree merges, so the intermediate levels of a group of
     * 2^fuse_levels channels stay in fast memory.
     * 0 = the level-by-level path; values above the plan's merge levels are
     * clamped. The stepper (reset()/advance()) always runs level by level.
     * Output is bit-identical, except for the automatic GPU direct levels
     * described below.
     * kFDMTAutoFuse (default) picks a depth from the plan alone, with no
     * hardware detection (see get_fuse_levels()):
     * - CPU, mode "valid": each thread walks one group through the chunk in
     *   time tiles of 512 or 256 samples, with the group's levels in two
     *   per-thread scratch buffers of a tile and its fused merges' tree
     *   history in cache; the deepest depth (then the longer tile) whose
     *   mean per-group working set is at most 1.75 MiB, with at least two
     *   groups per thread. summary() reports the tile.
     * - CPU, modes "full" and "roll": each group runs the whole block level
     *   by level in two per-thread scratch buffers; the deepest depth whose
     *   two buffers fit in max(36 MiB / nthreads, 8 MiB) per thread.
     * - GPU: one thread block per (channel group, time tile) with the
     *   intermediate levels in shared memory; the deepest depth whose tile of
     *   >= 128 samples fits a quarter of the SM's shared memory (four
     *   resident blocks, at most 48 KiB) with a level-0 halo of at most half
     *   a tile. An explicit depth is also
     *   clamped to 8 and reduced until a tile fits the device's opt-in shared
     *   memory.
     * - GPU, mode "valid", float input, when that tile depth stays below 3
     *   (many delays per channel): levels 0..3 are computed directly from
     *   per-channel running sums, and the trials are summed directly from
     *   level 4 when nothing reads the levels in between (see
     *   docs/pipeline_guide/performance.md). Output then equals the tree
     *   within float rounding (exactly for integer-valued input); the
     *   explicit depths and every other path stay bit-identical.
     * @param int_tree Performance parameter: for packed low-bit input, store
     * tree levels whose exact value bound fits as uint8/uint16 instead of
     * float (exact; default true). Integer levels cannot be inspected through
     * the stepper's view_* methods; pass false to inspect every level of a
     * packed stepper run. Ignored for float input.
     * @param chunk_nsamps Memory parameter (mode "valid" only): execute()
     * runs each block in chunks of this many samples, so the tree state is
     * two levels of rows x chunk instead of rows x nsamps -- at high DM tens
     * of GB for a block whose input and output are a few hundred MB.
     * Packed paths, integer-valued input, explicit fusion depths and the CPU
     * tree are bit-identical for any chunk (valid mode is overlap-save inside
     * the tree). Automatic GPU direct levels (float input, valid mode, many
     * delays per channel) agree within float rounding across chunks and are
     * not bit-identical. Fractional-delay FDMTFFT has a separate segmentation
     * and guard error. Candidate lists are threshold-aware: a small numerical
     * difference can split or join an excursion, and parallel overflow does
     * not promise the same survivor order. Each chunk costs a little
     * cross-block history traffic (a few per cent at a few thousand samples).
     * 0 (default) = automatic: the
     * longest power of two >= 2048 whose state fits 2 GiB, capped at nsamps
     * (GPU: the longest power of two >= 256 whose state fits the device L2,
     * when one does; see get_chunk_nsamps()); other values are rounded up to
     * a multiple of 64. The stepper (reset()/advance()) needs chunk >= nsamps.
     * @param max_memory_bytes Budget for everything the instance allocates
     * (get_memory_usage().total(); default kFDMTDefaultMaxMemory, 10 GiB).
     * The fastest streaming path that fits is taken. A valid-mode plan whose
     * tree history does not fit (e.g. 6656 channels at 64 us to DM 2000)
     * runs the skewed-window path instead, with the longest automatic chunk
     * that fits (on parkes_uwl faster than the tree path on both backends).
     * Throws std::invalid_argument when no path fits.
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
         IndexType dt_min          = 0,
         SizeType dt_step          = 1,
         bool use_box_smearing     = true,
         std::string_view mode     = "valid",
         Exec exec                 = {},
         SizeType nbeams           = 1,
         SizeType fuse_levels      = kFDMTAutoFuse,
         bool int_tree             = true,
         SizeType chunk_nsamps     = 0,
         SizeType max_memory_bytes = kFDMTDefaultMaxMemory);

    /**
     * @brief Constructs an FDMT instance with a custom delay trial grid.
     *
     * @param dt_grid Explicit list of delay trials in samples (e.g. from
     * generate_optimal_dt_grid).
     * @param f_min, f_max, nchans, nsamps, tsamp, use_box_smearing, mode,
     * exec, nbeams, fuse_levels, int_tree, chunk_nsamps, max_memory_bytes See
     * the first constructor.
     */
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         bool use_box_smearing     = true,
         std::string_view mode     = "valid",
         Exec exec                 = {},
         SizeType nbeams           = 1,
         SizeType fuse_levels      = kFDMTAutoFuse,
         bool int_tree             = true,
         SizeType chunk_nsamps     = 0,
         SizeType max_memory_bytes = kFDMTDefaultMaxMemory);

    /**
     * @brief Constructs an FDMT instance with a custom physical DM trial
     * grid.
     *
     * @param dm_grid Explicit list of DM trials in pc/cm^3 (supports
     * non-uniform spacing and negative DMs).
     * @param f_min, f_max, nchans, nsamps, tsamp, use_box_smearing, mode,
     * exec, nbeams, fuse_levels, int_tree, chunk_nsamps, max_memory_bytes See
     * the first constructor.
     */
    FDMT(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing     = true,
         std::string_view mode     = "valid",
         Exec exec                 = {},
         SizeType nbeams           = 1,
         SizeType fuse_levels      = kFDMTAutoFuse,
         bool int_tree             = true,
         SizeType chunk_nsamps     = 0,
         SizeType max_memory_bytes = kFDMTDefaultMaxMemory);

    /**
     * @brief Constructs an FDMT instance from a prepared plan, e.g. one with
     * a subband search depth (plans::FDMTPlan subband_depth), whose
     * staggered subbands the stepper then exposes. The mode is the plan's.
     *
     * @param plan The plan, copied into the instance.
     * @param use_box_smearing, exec, nbeams, fuse_levels, int_tree,
     * chunk_nsamps, max_memory_bytes See the first constructor.
     */
    explicit FDMT(plans::FDMTPlan plan,
                  bool use_box_smearing     = true,
                  Exec exec                 = {},
                  SizeType nbeams           = 1,
                  SizeType fuse_levels      = kFDMTAutoFuse,
                  bool int_tree             = true,
                  SizeType chunk_nsamps     = 0,
                  SizeType max_memory_bytes = kFDMTDefaultMaxMemory);

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
     * @param dmt Output DM-time array, beam-major flat (nbeams, ndms,
     * dmt_nsamps): at least nbeams*plan.get_dmt_size() floats. Only those
     * leading values are written.
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
     * use the float overload for 32-bit float data. The output is float32,
     * laid out as for the float overload. It equals execute() on the same
     * values converted to float, except that with box smearing each level-0
     * box mean is rounded to a few fractional bits (see
     * FDMTPlan::get_box_mean_bits(): at most 2^-(f + 1) per channel), which
     * keeps the integer tree narrow. On a GPU backend only the packed bytes
     * are copied to the device.
     *
     * @param waterfall_packed Packed input, beam-major flat.
     * @param nbits Sample width: 1, 2, 4, 8 or 16.
     * @param dmt Output DM-time array (size >= nbeams*get_dmt_size()).
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
     * @param d_dmt Output DM-time array (device memory), laid out as for the
     * host overload (size >= nbeams*get_dmt_size()).
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
     * @brief Chunk length execute() runs each block in: the `chunk_nsamps`
     * constructor argument (rounded up to a multiple of 64), or the
     * automatic choice, capped at nsamps (= nsamps: not chunked).
     */
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept;

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
     * @brief Resets and initializes the stepper (level 0) with a new
     * waterfall block and the caller's dmt buffer, both in host memory.
     *
     * @param waterfall Input waterfall data, beam-major flat
     * (nbeams*nchans*nsamps).
     * @param dmt Output DM-time array (size >= nbeams*plan.get_dmt_size()).
     *            Receives the root level, i.e. the final transform, at
     *            finalize().
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
     * The stepper always runs level by level (never fused), on whole
     * blocks: it needs get_chunk_nsamps() == nsamps.
     * @throws std::logic_error if a "valid"-mode block is still unfinished,
     * or if the engine runs blocks in chunks.
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
     * @brief Zero-copy device view of beam 0's state at the current level
     * (GPU backends), laid out as for view_level_data().
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
     * @brief Number of sub-bands at the current level: the tree sub-bands
     * in ascending frequency, then (for a plan with subband_depth > 0, at
     * levels niters - subband_depth .. niters - 1) the staggered sub-bands
     * that straddle their boundaries (FDMTSubbandView::staggered). With
     * remaining_levels() == k this is 2^k tree sub-bands (fewer near the
     * channel level of a non-power-of-two band) and 2^k - 1 staggered ones.
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
     *
     * Off the skewed-window path the layout is the same on every backend: a
     * 16-word header (magic, version, layout hash, stream position), then
     * the level-0 input history and the tree history (each merge's slot,
     * oldest sample first).
     *
     * On the CPU skewed-window path (plans whose tree history exceeds the
     * memory budget) the size follows the input seen so far: while only
     * packed 8- or 16-bit blocks have arrived the raw input ring is saved at
     * that width (about a third of the float size for 8-bit input); after a
     * float block, or before any input, it is the float size. Query it when
     * saving; load_history() accepts either layout.
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
     * with a host buffer previously produced by save_history(), resuming
     * that stream. Off the skewed-window path any backend's buffer is
     * accepted from an instance with the same plan tree (DM grid), nbeams
     * and use_box_smearing; chunk length, fusion and threads may differ. A
     * skewed-window history is accepted only by the same backend and
     * skewed layout. Use reset_history() instead to start a stream cold (a
     * zeroed buffer is not a history).
     *
     * @param in Source span, size must equal history_state_size() (on the
     * CPU skewed-window path: the size of the saved layout, packed or float).
     * @throws std::invalid_argument on a size mismatch, or when the header
     * shows another configuration, another path or no header (nothing is
     * replaced then).
     */
    void load_history(std::span<const float> in);

    /**
     * @brief Device-memory analogue of save_history() (GPU backends).
     * Enqueued on @p stream; synchronize before reading @p out elsewhere.
     * @throws std::invalid_argument on the CPU backend.
     */
    void save_history(DeviceSpan<float> out, Stream stream = {}) const;

    /**
     * @brief Device-memory analogue of load_history() (GPU backends). The
     * header is copied to the host and checked first, which synchronizes
     * @p stream; the copy of the history is then asynchronous on it, and
     * later work on the same stream sees the loaded history.
     * @throws std::invalid_argument on the CPU backend, and as
     * load_history().
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
