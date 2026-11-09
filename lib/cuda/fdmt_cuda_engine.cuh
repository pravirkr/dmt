#pragma once

// Private: the CUDA FDMT engine (detail::make_fdmt_gpu). The class is
// declared here and defined in fdmt_cuda.cu (construction, path and
// memory planning), fdmt_cuda_direct.cu (the direct levels and the
// direct top), fdmt_cuda_execute.cu (execute(), the search, the tree
// levels and their kernels) and fdmt_cuda_stream.cu (the stepper and
// the stream history).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_direct.hpp"
#include "dmt/fdmt_fused_tile.hpp"
#include "dmt/fdmt_history.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_skew_windows.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"
#include "dmt/plans_cuda.cuh"

#include "fdmt_cuda_kernels.cuh"

namespace dmt::algorithms::detail::fdmt_cuda {

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

inline cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

class FDMTCudaEngine final : public detail::FDMTEngine {
public:
    // Keep the host-span overloads visible next to the device ones below.
    using detail::FDMTEngine::execute_search;

    FDMTCudaEngine(const plans::FDMTPlan& plan,
                   const detail::FDMTEngineConfig& cfg);

    void execute_search(DeviceSpan<const float> waterfall,
                        detail::FDMTSearchSink& sink,
                        Stream queue) override;

    void execute_search(DeviceSpan<const uint8_t> waterfall,
                        SizeType nbits,
                        detail::FDMTSearchSink& sink,
                        Stream queue) override;

    template <typename TInt>
    void execute_search_int_typed(const FDMTInputPacked& full,
                                  TInt* ring,
                                  detail::FDMTSearchSink& sink,
                                  Stream queue,
                                  cudaStream_t stream);

    /**
     * The packed search with the direct levels (and the direct top when
     * configured), as the float search runs them: level K from the input's
     * uint32_t running sums, the tree from K + 1, and either the trials (a
     * full-band search: materialised, as the sink's root) or level
     * root - subband_depth - 1 summed from level k's rings. Both start at
     * stream position 0 with all-zero windows and rings, so the tree
     * histories of the levels they replace are never read (the integer
     * search keeps no stepper, history export or tree fallback).
     */
    template <typename TInt>
    void execute_search_int_direct(const FDMTInputPacked& full,
                                   TInt* ring,
                                   detail::FDMTSearchSink& sink,
                                   Stream queue,
                                   cudaStream_t stream);

    /// Level K of the packed search from the input (uint32_t running sums).
    template <typename TInt>
    void launch_direct_int(const FDMTInputPacked& input,
                           TInt* out,
                           cudaStream_t stream);

    /// The direct top of the packed search: trials (rows m_chunk apart,
    /// beams `out_beam_stride` apart) from level k's state and rings, then
    /// level k's rings (run_top's integer analogue).
    template <typename TInt>
    void run_top_int(const TInt* level_state,
                     TInt* out,
                     int64_t out_beam_stride,
                     cudaStream_t stream);

    void reset_search_history(Stream queue) override;

    // Allocates every device buffer and resolves the fusion depth: nothing is
    // allocated after construction except the per-call staging of the
    // host-memory execute() overloads.
    void init_device_structures(SizeType fuse_levels);

    ~FDMTCudaEngine() override                       = default;
    FDMTCudaEngine(const FDMTCudaEngine&)            = delete;
    FDMTCudaEngine& operator=(const FDMTCudaEngine&) = delete;
    FDMTCudaEngine(FDMTCudaEngine&&)                 = delete;
    FDMTCudaEngine& operator=(FDMTCudaEngine&&)      = delete;

    void execute(std::span<const float> waterfall_h,
                 std::span<float> dmt_h) override;

    void execute_d(cuda::std::span<const float> waterfall_d,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream);

    // execute(): the block's chunks, each from level 0 (fused when
    // configured) to the root; the root writes the chunk's columns of dmt.
    template <typename Input>
    void run_block(const Input& input,
                   const std::vector<detail::FDMTLevelType>& types,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream);

    // The input from sample t0 on (rows keep the block's stride; packed
    // chunks start on a whole byte: chunks are multiples of 64 samples).
    [[nodiscard]] static FDMTInputF32 chunk_input(const FDMTInputF32& in,
                                                  SizeType t0) {
        return {.data = in.data + t0, .nsamps = in.nsamps};
    }
    [[nodiscard]] static FDMTInputPacked chunk_input(const FDMTInputPacked& in,
                                                     SizeType t0) {
        return {.data = in.data + ((t0 * static_cast<SizeType>(in.nbits)) / 8),
                .row_bytes = in.row_bytes,
                .nbits     = in.nbits};
    }

    void execute(std::span<const uint8_t> waterfall_h,
                 SizeType nbits,
                 std::span<float> dmt_h) override;

    // Host-memory execute() staging: device copies of the input and of the
    // (nbeams * dmt_size) output, allocated on the first host call and
    // reused, so
    // steady-state host calls allocate nothing (device-memory callers never
    // pay for them). Transfers and the block's kernels run on
    // m_host.stream() (see gpu_host::ChunkedStager).
    template <typename T>
    void stage_to_device(thrust::device_vector<T>& dst, std::span<const T> src);

    cuda::std::span<float> stage_dmt_span() {
        const auto size = m_nbeams * m_plan.get_dmt_size();
        if (m_stage_dmt_d.size() < size) {
            m_stage_dmt_d.resize(size);
        }
        return {thrust::raw_pointer_cast(m_stage_dmt_d.data()), size};
    }

    // Copies the (nbeams * dmt_size) result back, after the work on
    // m_host.stream().
    void copy_result_to_host(std::span<float> dmt_h);

    void execute_d(cuda::std::span<const uint8_t> waterfall_d,
                   SizeType nbits,
                   cuda::std::span<float> dmt_d,
                   cudaStream_t stream);

    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override;

    void execute(DeviceSpan<const uint8_t> d_waterfall,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override;

    void reset(DeviceSpan<const uint8_t> d_waterfall,
               SizeType nbits,
               DeviceSpan<float> d_dmt,
               Stream stream) override;

    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream) override;

    // Host-memory stepper: the block is staged in the host-execute() device
    // buffers and stepped there; finalize() copies each beam's result into
    // `dmt_h`, and the view_* methods return host snapshots.
    void reset(std::span<const float> waterfall_h,
               std::span<float> dmt_h) override;

    void reset(std::span<const uint8_t> waterfall_h,
               SizeType nbits,
               std::span<float> dmt_h) override;

    // The stepper (always unfused) runs whole blocks only.
    void check_stepper_block() const;

    // `fuse`: execute() only -- replace level 0 and the first merges by the
    // fused kernel (see configure_fusion()); the stepper stays unfused.
    void reset_impl(cuda::std::span<const uint8_t> d_waterfall,
                    SizeType nbits,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse);

    void reset_impl(cuda::std::span<const float> d_waterfall,
                    cuda::std::span<float> d_dmt,
                    cudaStream_t stream,
                    bool fuse);

    /// Levels below which execute() does not materialise the tree: the
    /// tile fusion's depth, or the direct levels' (fdmt_direct.hpp).
    [[nodiscard]] SizeType get_fuse_levels() const noexcept override {
        if (m_direct) {
            return m_direct->levels;
        }
        return m_fused ? static_cast<SizeType>(m_fused->args.fuse) : 0;
    }

    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept override {
        return m_chunk;
    }

    [[nodiscard]] std::string summary() const override;

    [[nodiscard]] uint64_t search_latency() const noexcept override {
        return m_top && m_top->windowed ? m_top->max_lag : 0;
    }
    [[nodiscard]] std::string search_path() const override;

    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept override;

    void advance(SizeType levels, Stream stream_in) override;

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override;

    [[nodiscard]] DeviceSpan<const float>
    view_level_data_device() const override;

    [[nodiscard]] std::span<const float> view_level_data() const override;

    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const;

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override;

    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const override;

    SizeType current_level() const noexcept override { return m_current_level; }
    SizeType total_levels() const noexcept { return m_plan.get_niters() + 1; }
    SizeType num_subbands() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT (cuda): Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }
    bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream_in) override;

    /// Zeroes all cross-block history (level-0 and tree-level) so the next
    /// execute()/reset() call starts as if this were a brand-new instance.
    /// Mirrors the CPU engine's reset_history().
    void reset_history() noexcept override;

    /// Size (floats) of this instance's "valid"-mode streaming history, as
    /// saved/restored by save_history()/load_history(): the level-0 input
    /// history, then the tree history of each beam in the plan's hist_offset
    /// layout, oldest sample first (the CPU engine's format). Zero for
    /// "full"/"roll" mode instances.
    ///
    /// Of each ping-pong pair only the buffer the next chunk reads is live
    /// (selected by m_parity); the ring is live as a whole.
    [[nodiscard]] SizeType history_state_size() const noexcept override;

    /// Copies the live streaming history out to a caller-owned device
    /// buffer, so one shared FDMT instance can multiplex several
    /// independent streams (each with its own history) instead of requiring
    /// one instance per stream. Enqueued on `stream`; synchronize before
    /// reading `out` elsewhere.
    void save_history(DeviceSpan<float> out_d, Stream stream_in) const override;

    /// Replaces the live streaming history with a buffer previously produced
    /// by save_history() (from an instance built with the same plan
    /// geometry), resuming that stream. Fully asynchronous on `stream` --
    /// use reset_history() instead to start a stream cold.
    void load_history(DeviceSpan<const float> in_d, Stream stream_in) override;

    // Host-memory history: the device format through a temporary device
    // buffer, after the stepper's queue drains.
    void save_history(std::span<float> out) const override;

    void load_history(std::span<const float> in) override;

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    [[nodiscard]] Device device() const noexcept {
        return {.backend = detail::kGPUBackend, .id = m_device_id};
    }

    // Host copy of beam 0's current level for the host view_* methods,
    // refreshed only after the stepper moved (see m_host_level_valid).
    [[nodiscard]] const float* host_level_snapshot() const;

    int m_device_id;
    bool m_use_box_smearing;
    bool m_int_tree; // performance parameter (packed input)
    // Level-0 box means (fdmt_box_weights.hpp) by |dt| = s: dyadic float
    // weights, packed-input rounding recipes on the scale 2^-F and whether
    // their products need 64 bits (by nbits), and the value 2^-F of one
    // integer state unit (1 without box smearing).
    thrust::device_vector<float> m_box_w_d;
    thrust::device_vector<detail::FDMTBoxRound> m_box_round_d;
    std::array<bool, 17> m_box_round_u64{};
    float m_int_unit{1.0F};

    // The box means as kernel arguments, for packed input of `nbits` (0:
    // float input).
    [[nodiscard]] detail::FDMTBoxMeans box_means(SizeType nbits) const {
        return {
            .w_f32     = thrust::raw_pointer_cast(m_box_w_d.data()),
            .round     = thrust::raw_pointer_cast(m_box_round_d.data()),
            .unit      = m_int_unit,
            .round_u64 = m_box_round_u64[nbits] ? 1 : 0,
        };
    }

    template <typename Input>
    [[nodiscard]] detail::FDMTBoxMeans box_means_for(const Input& in) const {
        if constexpr (detail::kFDMTPackedInput<Input>) {
            return box_means(static_cast<SizeType>(in.nbits));
        } else {
            return box_means(0);
        }
    }
    FDMTMode m_mode;
    SizeType m_nbeams{1};
    const plans::FDMTPlan& m_plan; // owned by the FDMT facade
    // Chunked execution (fdmt_chunk.hpp): row length of the tree state
    // below the root, level 0's row stride, the elements of one beam's level
    // in a state half, and the current chunk's length.
    SizeType m_chunk{0};
    SizeType m_row_len0{0};
    SizeType m_half_elems{0};
    SizeType m_skew_nbatches{1};
    std::vector<int> m_skew_batch_top_grid;
    std::vector<std::vector<int>> m_skew_batch_sum_off;
    std::vector<std::vector<int>> m_skew_batch_copy_off;
    SizeType m_active{0};
    plans::FDMTPlanContainerD m_plan_d; // re-laid out for m_chunk
    // The two state halves (see init_device_structures()).
    thrust::device_vector<float> m_state_internal_d;
    // Level-0 history buffer for valid-mode streaming across FDMT blocks
    // (ping-pong pair)
    thrust::device_vector<float> m_history_a_d;
    thrust::device_vector<float> m_history_b_d;
    float* m_level0_hist_in_ptr{nullptr};
    float* m_level0_hist_out_ptr{nullptr};
    // Tree-level (level >= 1) cross-block history, per beam in the plan's
    // hist_offset layout of tree_history_size() elements:
    // - levels 1..m_fifo_levels (the fused levels, which merge and update
    //   their history inside one kernel across concurrent tiles) keep an
    //   oldest-first FIFO, ping-ponged between two buffers of m_fifo_size;
    // - the levels above keep one in-place ring of m_ring_size per beam
    //   (head sample of stream time u at u mod delay): each chunk writes
    //   only its newest samples, and no history is shifted or doubled.
    // The level-0 input history and the FIFOs flip on m_parity per chunk.
    std::array<thrust::device_vector<float>, 2> m_tree_fifo_d;
    mutable thrust::device_vector<float> m_tree_ring_d;
    SizeType m_fifo_levels{0};
    SizeType m_fifo_size{0};
    mutable SizeType m_ring_size{0};
    SizeType m_ring_low_base{0};
    SizeType m_ring_low_size{0};
    SizeType m_ring_high_orig_base{0};
    SizeType m_ring_high_size{0};
    mutable bool m_full_tree_ring{false};
    std::vector<SizeType> m_level_max_extra;
    bool m_parity{false};
    float* m_fifo_in_ptr{nullptr};
    float* m_fifo_out_ptr{nullptr};
    // Stream sample of the current chunk's start, and of the next chunk's.
    uint64_t m_chunk_pos{0};
    uint64_t m_next_pos{0};
    /// Plain-tree history layout (fdmt_tree_history_hash), saved in and
    /// checked against the history header.
    uint64_t m_hist_hash{0};
    /// Stream position of the last loaded plain-tree history (0 after
    /// reset_history()); the saved position is this plus m_next_pos.
    uint64_t m_hist_pos_base{0};

    // Per-level storage types (all float, and per nbits for int_tree), fixed
    // at construction.
    using Elem = detail::FDMTLevelType;
    std::vector<Elem> m_float_levels;
    std::array<std::vector<Elem>, 17> m_int_levels; // indexed by nbits

    SizeType m_search_max_fuse{SIZE_MAX};
    SizeType m_search_nbits{32};
    bool m_search_u32{false};
    // The constructor's fuse_levels (the packed search runs no tile fusion,
    // but takes the direct levels when this is automatic).
    SizeType m_requested_fuse{kFDMTAutoFuse};
    /// 32-bit words per direct-top ring / trial element: 2 for the packed
    /// search's uint64_t state, else 1 (float or uint32_t).
    [[nodiscard]] SizeType top_words() const noexcept {
        return (m_search_nbits != 32 && !m_search_u32) ? 2 : 1;
    }
    // Packed search: exact integer tree history, a ring at every level.
    thrust::device_vector<uint64_t> m_integer_ring;
    thrust::device_vector<uint32_t> m_integer32_ring;

    // Level fusion (fuse_levels): the resolved depth's
    // device tables and launch geometry; empty when execute() is unfused.
    struct FusedLevelsD {
        DeviceVector<int> group_info;
        DeviceVector<int> coords;
        DeviceVector<int> dt_grid0;
        DeviceVector<int> ndt0;
        DeviceVector<int> coord_offset0;
        detail::FDMTFusedTileArgs args{};
        int tile{0};
        int ntiles{0}; ///< for a whole row (valid-mode chunks: per launch)
        int ngroups{0};
        SizeType smem_bytes{0};
    };
    std::unique_ptr<FusedLevelsD> m_fused;
    // Direct low levels (configure_direct()): the device tables, the input
    // window ping-pong (flipped with m_parity) and the per-chunk running
    // sums. Replaces m_fused when configured.
    static constexpr SizeType kDirectLevels = 3;
    struct DirectLevelsD {
        SizeType levels{0};
        bool skewed{false};
        int window{0};
        int nsubs{0};
        int max_rows{0};
        int max_extra{0};
        SizeType total_elems{0};
        int cpass{0};
        int pitch{0};
        SizeType smem{0};
        int64_t win_ring_size{0};
        std::vector<int32_t> row_off;
        std::vector<int32_t> batch_sub_off;
        thrust::device_vector<detail::FDMTDirectSub> subs;
        thrust::device_vector<int32_t> leaf_off;
        thrust::device_vector<detail::FDMTDirectLeaf> leaves;
        thrust::device_vector<detail::FDMTDirectLeaf> hist_leaves;
        std::vector<thrust::device_vector<detail::FDMTDirectHistCoord>>
            hist_coords;
        std::vector<int32_t> hist_max_delay;
        std::array<thrust::device_vector<float>, 2> win;
        thrust::device_vector<float> prefix;
        thrust::device_vector<int32_t> chan_ring_len;
        thrust::device_vector<int64_t> chan_ring_off;
        thrust::device_vector<float> win_ring;
        thrust::device_vector<uint8_t> win_ring_u8;
        thrust::device_vector<uint16_t> win_ring_u16;
    };
    std::unique_ptr<DirectLevelsD> m_direct;
    bool m_direct_eligible{false};
    const float* m_direct_win_in{nullptr};
    float* m_direct_win_out{nullptr};
    // Real input samples in the window (a load_history() empties it).
    SizeType m_direct_filled{0};
    // Levels 1..K tree histories lag the stream (direct chunks ran).
    mutable bool m_direct_stale{false};
    // Direct top (configure_top()): trials summed from level k's rows.
    struct DirectTopD {
        SizeType level{0};
        SizeType target{0};
        bool skewed{false};
        bool target_ring{false};
        int ntrials{0};
        int nrows{0};
        int max_ring{0};
        int64_t ring_size{0}; ///< per beam
        thrust::device_vector<int32_t> out_row;
        thrust::device_vector<int32_t> leaf_off;
        thrust::device_vector<detail::FDMTDirectTopLeaf> leaves;
        thrust::device_vector<int32_t> ring_len;
        thrust::device_vector<int64_t> ring_off;
        thrust::device_vector<float> ring;
        thrust::device_vector<detail::FDMTDirectTopLeaf> hist_leaves;
        std::vector<thrust::device_vector<detail::FDMTDirectHistCoord>>
            hist_coords;
        std::vector<int32_t> hist_max_delay;
        thrust::device_vector<float> trials; ///< search: one chunk's trials
        /// Windowed top (subband search, configure_windowed_top): level
        /// `level` whole (words of top_words()), and per searched level the
        /// rows' emission offsets and lags (FDMTSearchInput::row_offset).
        bool windowed{false};
        thrust::device_vector<float> win_top;
        /// Rows of win_top with a carried spread: window start, spread.
        thrust::device_vector<SizeType> carry_off;
        thrust::device_vector<int32_t> carry_len;
        std::vector<thrust::device_vector<SizeType>> emit_off;
        std::vector<thrust::device_vector<uint64_t>> emit_lag;
        uint64_t max_lag{0}; ///< largest emit_lag (search_latency())
    };
    std::unique_ptr<DirectTopD> m_top;
    bool m_search_upper{false};
    SizeType m_max_memory{kFDMTDefaultMaxMemory}; // FDMTEngineConfig
    bool m_force_skewed{false}; // FDMTEngineConfig::force_skewed_direct
    bool m_top_chunk{false};
    // Samples of the rings' reach streamed since a load_history().
    SizeType m_top_filled{0};
    // Levels above k: tree histories lag the stream (top chunks ran).
    mutable bool m_top_stale{false};
    // Auto depth: the deepest fusion whose shared memory at a tile of at
    // least kMinAutoTile samples fits auto_smem_bytes() (kAutoBlocksPerSM
    // resident blocks per SM) with a level-0 halo of at most half a tile.
    // Four blocks (32 warps) hide the fused kernel's global-load latency
    // better than larger tiles at two blocks save halo work (L40S: 4.9 vs
    // 5.2 ms at 4096 channels, 16384 samples, 2049 DMs).
    static constexpr SizeType kDefaultSmemBytes = SizeType{48} * 1024;
    static constexpr int kAutoBlocksPerSM       = 4;
    static constexpr int kMinAutoTile           = 128;
    static constexpr int kMaxTile               = 1024;
    static constexpr int kFusedThreads          = 256;

    // One tree level's device state: beam 0 at `base`, beam b at element
    // offset b * level_beam_stride(level), stored as `type`.
    struct LevelBufD {
        std::byte* base{nullptr};
        Elem type{Elem::kF32};
    };

    // Stepper state
    bool m_is_initialized{false};
    SizeType m_current_level{0};
    std::vector<LevelBufD> m_levels_d;
    float* m_dmt_target_ptr{nullptr};
    cudaStream_t m_stream{nullptr};
    // End of the latest device-memory call; host-memory calls wait for it.
    mutable gpu_utils::DeviceWorkFence m_device_work;
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;
    // Host-memory execute() staging (see stage_to_device()).
    thrust::device_vector<float> m_stage_wf_f32_d;
    thrust::device_vector<uint8_t> m_stage_wf_packed_d;
    thrust::device_vector<float> m_stage_dmt_d;
    // Host-memory stepper: the caller's dmt buffer (empty after a
    // device-memory reset) and the host snapshot behind the view_* methods.
    std::span<float> m_host_dmt;
    gpu_host::ChunkedStager m_host; // pinned pipeline of the host overloads
    mutable std::vector<float> m_host_level;
    mutable bool m_host_level_valid{false};

    // The device plan stores per-beam coordinate offsets as 32-bit ints (see
    // plans_cuda.cuh), and the kernels index within one beam in 32-bit
    // arithmetic (only the beam-strided part is 64-bit). Reject plans whose
    // per-beam extents do not fit, instead of silently wrapping indices.
    void check_int32_extents() const;

    /// Tree history offsets are unsigned 32-bit (plans_cuda.cuh), taken from
    /// the full tree layout. Checked against the highest offset an active
    /// FIFO or ring level reads (`ring_elems` per beam): levels covered by
    /// the direct levels or the direct top are never indexed, so a plan
    /// whose full tree exceeds the range still runs when they cover it.
    /// Checked again when ensure_full_tree_history() expands to the full
    /// tree.
    static void check_tree_ring_extent(SizeType ring_elems);

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const;

    void check_packed_inputs(SizeType waterfall_bytes,
                             SizeType nbits,
                             SizeType dmt_size) const;

    void check_subband_idx(SizeType subband_idx) const;

    void ensure_full_tree_history(cudaStream_t stream) const;

    // ---- skewed-window streaming history (m_direct->skewed) -------------
    //
    // The skewed path keeps no tree history: its state is the raw input
    // ring, the direct top's target rings and any active tree ring / FIFO /
    // level-0 history, all tied to the stream position. They are saved
    // verbatim after a header of 16-bit pieces held exactly in floats
    // (magic, version, layout hash, stream position). A buffer is accepted
    // only by an engine with the same skewed layout; anything else throws.
    static constexpr SizeType kSkewHistHeader  = detail::kFDMTHistHeader;
    static constexpr uint64_t kSkewHistMagic   = detail::kFDMTSkewHistMagic;
    static constexpr uint64_t kSkewHistVersion = 1;

    [[nodiscard]] bool skew_history() const noexcept {
        return m_mode == FDMTMode::kValid && m_search_nbits == 32 && m_direct &&
               m_direct->skewed;
    }

    struct SkewHistBuffer {
        float* data;
        SizeType size;
    };
    /// The live persistent buffers, in save order (the ping-pong buffers
    /// the next chunk reads).
    [[nodiscard]] std::vector<SkewHistBuffer> skew_hist_buffers() const;

    [[nodiscard]] uint64_t skew_layout_hash() const;

    void save_skew_history_d(float* dst, cudaStream_t stream) const;

    void load_skew_history_d(const float* src, cudaStream_t stream);

    // Saved history: level 0, then per beam the tree history in the plan
    // layout, oldest first (the FIFOs as stored, the ring linearized).
    void save_history_d(float* dst, cudaStream_t stream) const;

    // Inverse of save_history_d(): an oldest-first history is the ring at
    // stream position 0.
    void load_history_d(const float* src, cudaStream_t stream);

    // f(coords, count, max delay) over the ring levels' sum coordinates,
    // as one contiguous range of the device table.
    template <typename F> void for_each_ring_level(F&& f) const;

    // The ping-pong buffers the next chunk's start() reads as `in` (see the
    // parity swap there): the only live ones between chunks.
    [[nodiscard]] thrust::device_vector<float>& next_level0_history() {
        return m_parity ? m_history_b_d : m_history_a_d;
    }
    [[nodiscard]] const thrust::device_vector<float>&
    next_level0_history() const {
        return m_parity ? m_history_b_d : m_history_a_d;
    }
    [[nodiscard]] thrust::device_vector<float>& next_tree_fifo() {
        return m_tree_fifo_d[m_parity ? 1 : 0];
    }
    [[nodiscard]] const thrust::device_vector<float>& next_tree_fifo() const {
        return m_tree_fifo_d[m_parity ? 1 : 0];
    }

    // Tree history of level l's merges in this chunk (kernel_execute_iter).
    /// kernel_advance_tree_history over `count` coordinates of up to
    /// `max_offset` samples each.
    template <typename TIn, typename THist>
    void launch_advance_tree_history(const TIn* in_ptr,
                                     const plans::FDMTCoordDPtrs& coords,
                                     const THist* hist_in,
                                     THist* hist_out,
                                     int count,
                                     int max_offset,
                                     int in_state_nelements,
                                     int64_t hist_stride,
                                     int64_t hist_base,
                                     uint64_t ring_pos,
                                     float int_unit,
                                     cudaStream_t stream) const;

    struct TreeHistory {
        float* in      = nullptr;
        float* out     = nullptr;
        int64_t stride = 0;
        int64_t base   = 0;
        uint64_t ring  = kFifoHistory;
    };
    [[nodiscard]] SizeType ring_base_for_level(SizeType level) const noexcept {
        if (m_full_tree_ring) {
            return m_fifo_size;
        }
        if (m_top && level > m_top->target) {
            return m_ring_high_orig_base - m_ring_low_size;
        }
        return m_ring_low_base;
    }

    [[nodiscard]] TreeHistory tree_history(SizeType level) const {
        if (m_mode != FDMTMode::kValid) {
            return {};
        }
        if (level <= m_fifo_levels) {
            return {m_fifo_in_ptr, m_fifo_out_ptr,
                    static_cast<int64_t>(m_fifo_size), 0, kFifoHistory};
        }
        if (m_direct && m_direct->skewed && m_top && level <= m_top->level) {
            return {nullptr, nullptr, 0, 0, m_chunk_pos};
        }
        auto* ring = const_cast<float*>( // NOLINT: device buffer, host const
            thrust::raw_pointer_cast(m_tree_ring_d.data()));
        return {ring, ring, static_cast<int64_t>(m_ring_size),
                static_cast<int64_t>(ring_base_for_level(level)), m_chunk_pos};
    }

    // Per-beam tree history of levels 1..level (the plan assigns hist
    // offsets level by level).
    [[nodiscard]] SizeType tree_history_through(SizeType level) const {
        SizeType total = 0;
        const auto& pc = m_plan.get_container();
        for (SizeType l = 1; l <= level && l <= m_plan.get_niters(); ++l) {
            for (const auto& c : pc.coordinates_sum[l]) {
                total += c.delay;
            }
        }
        return total;
    }

    // Elements between beams of level l: a state half's level, or the dmt
    // output for the root.
    [[nodiscard]] int level_beam_stride(SizeType l) const noexcept {
        return static_cast<int>(l == m_plan.get_niters() ? m_plan.get_dmt_size()
                                                         : m_half_elems);
    }

    // Calls f(T{}) with the storage type of `e` (tag dispatch: plain C++14
    // generic lambdas, the most portable form across nvcc versions).
    template <typename F> static void with_elem(Elem e, F&& f) {
        switch (e) {
        case Elem::kU8:
            f(uint8_t{});
            break;
        case Elem::kU16:
            f(uint16_t{});
            break;
        case Elem::kF32:
            f(float{});
            break;
        }
    }

    [[nodiscard]] const float* current_level_f32() const {
        if (m_levels_d[m_current_level].type != Elem::kF32) {
            throw std::logic_error(fmt::format(
                "FDMT (cuda): level {} is stored as an integer type "
                "(int_tree); construct with int_tree=false to inspect "
                "intermediate levels.",
                m_current_level));
        }
        return reinterpret_cast<const float*>(m_levels_d[m_current_level].base);
    }

    [[nodiscard]] SizeType max_optin_smem_bytes() const;

    /// The device's L2 size in bytes (0 if the query fails).
    [[nodiscard]] SizeType l2_bytes() const;

    /// Shared memory per fused block for kAutoBlocksPerSM resident blocks
    /// (less the ~1 KiB the runtime reserves per block), at most the
    /// no-opt-in limit.
    [[nodiscard]] SizeType auto_smem_bytes() const;

    /**
     * Resolves the fuse_levels constructor argument for this plan and device
     * and uploads the fused kernel's tables. An explicit depth is clamped to
     * the plan's merge levels and kMaxFusedLevels, and reduced until a tile
     * fits the device's (opt-in) shared memory; kFDMTAutoFuse follows the rule
     * at auto_smem_bytes(). Leaves m_fused empty when no depth qualifies.
     */
    void configure_fusion(const plans::FDMTPlanContainer& pc,
                          SizeType requested);

    void adopt_fused_plan(const detail::FDMTFusedTilePlan& plan, int tile);

    template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
    void launch_fused_typed(const Input& input, TOut* out, cudaStream_t stream);

    // Levels 0..F in one kernel, straight into level F's buffer (stored as
    // that level's type), then the level-0 input history as usual.
    template <typename Input>
    void launch_fused(const Input& input, cudaStream_t stream);

    /**
     * Assigns each level's device buffer -- the same rule as the CPU FDMT:
     * level l < niters in state half l % 2 (a merge never reads and writes
     * the same half; integer levels use the halves' leading bytes), the root
     * in the caller's dmt buffer at `dmt` (the chunk's first sample).
     */
    void layout_levels(const std::vector<Elem>& types, float* dmt);

    // Level 0 of one chunk of `active` samples (the whole block for the
    // stepper), its root going to `dmt`.
    template <typename Input>
    void start(const Input& input,
               const std::vector<Elem>& types,
               float* dmt,
               SizeType active,
               cudaStream_t stream,
               bool fuse);

    // Direct low levels (fdmt_direct.hpp) -----------------------------------

    /// Automatic fusion of a float valid-mode engine whose tile fusion stops
    /// short of kDirectLevels: compute level kDirectLevels from the input
    /// (replacing the tile fusion), if its shared-memory window fits.
    void configure_direct(SizeType requested);

    /// Whether this chunk computes the direct levels: configured, float
    /// levels, and an input window covering every row's reach.
    template <typename Input>
    [[nodiscard]] bool use_direct(const std::vector<Elem>& types) const {
        return m_direct && types[m_direct->levels] == Elem::kF32 &&
               (m_direct->skewed ||
                (std::is_same_v<Input, FDMTInputF32> &&
                 m_direct_filled >= static_cast<SizeType>(m_direct->window)));
    }

    template <typename T, typename PtrFn>
    void
    launch_skewed_batch_mid(SizeType b, PtrFn&& lvl_ptr, cudaStream_t stream);

    template <typename Input>
    void launch_direct(const Input& input, cudaStream_t stream);

    /// Keeps the direct levels' input window (the last `window` samples of
    /// every channel) in step with the stream, whichever path ran.
    template <typename Input>
    void advance_direct_window(const Input& input, cudaStream_t stream);

    /// Rewrites the tree histories of levels 1..K (rings) from the input
    /// window at the current stream position, after direct chunks left them
    /// behind (fdmt_direct.hpp).
    void rebuild_direct_history(cudaStream_t stream) const;

    void build_skewed_direct_from_windows(
        const std::vector<int32_t>& s_hi_k,
        const std::vector<int32_t>& sp_k,
        const std::vector<SizeType>& batch_grid_off = {});

    /// `win` (windowed top, see configure_windowed_top): the levels' windows
    /// come from compute_skew_windows() instead of `top_plan`'s rows, and
    /// level top_k is laid out whole (not per batch) for the global merges
    /// above it.
    void configure_skewed_direct_and_mid(
        plans::FDMTPlanContainer& relaid,
        detail::FDMTDirectTopPlan& top_plan,
        const detail::FDMTSkewWindows* win     = nullptr,
        const detail::FDMTSkewWindows* top_win = nullptr);

    /**
     * Subband search on the skewed direct levels without tree history (the
     * GPU analogue of FDMTSkewCpu::execute_search): the windows of every
     * level come from compute_skew_windows() with the plan's subband depth
     * (each searched row emitted with its own lag). Levels k..top_k run in
     * subband batches as for the target-ring top, level top_k is kept
     * whole in DirectTopD::win_top, and levels top_k + 1 .. root are merged
     * over their full windows in the state halves (no history), the root's
     * rows (zero spread) into DirectTopD::trials. False when the skewed
     * direct levels do not fit.
     */
    bool configure_windowed_top(plans::FDMTPlanContainer& relaid,
                                SizeType top_k);

    /// The windowed top's whole level, trials and emission tables.
    [[nodiscard]] SizeType windowed_top_bytes() const noexcept;

    /// Level l's state: a state half, or DirectTopD::win_top for the
    /// windowed top's level (elements of T).
    template <typename T> [[nodiscard]] T* state_level_ptr(SizeType l) {
        if (m_top && m_top->windowed && l == m_top->level) {
            return reinterpret_cast<T*>(
                thrust::raw_pointer_cast(m_top->win_top.data()));
        }
        return reinterpret_cast<T*>(
                   thrust::raw_pointer_cast(m_state_internal_d.data())) +
               ((l % 2) * m_nbeams * m_half_elems);
    }

    /// The windowed top's levels top_k + 1 .. root (merges over the full
    /// skewed windows), calling consume(l, state) after each searched level
    /// (root - subband_depth .. root - 1) and the root (DirectTopD::trials).
    template <typename T, typename Consume>
    void run_windowed_top(cudaStream_t stream, Consume&& consume);

    /// The search input of windowed-top level l (lagged rows below the
    /// root, the root's trials m_chunk apart).
    template <typename T>
    detail::FDMTSearchInput
    windowed_input(SizeType l, const T* state, Stream queue) const;

    /// Direct top above the direct low levels (fdmt_direct.hpp) when no
    /// search reads the levels in between and its rings fit the budget.
    void configure_top(plans::FDMTPlanContainer& relaid);

    /// Whether this chunk's trials come from the direct top: configured,
    /// float level k, and rings covering every trial's reach.
    [[nodiscard]] bool use_top(const std::vector<Elem>& types) const {
        return m_top && m_mode == FDMTMode::kValid &&
               types[m_top->level] == Elem::kF32 &&
               (m_top->skewed ||
                m_top_filled >= static_cast<SizeType>(m_top->max_ring));
    }

    /// The chunk's trials into `out` (rows `out_row_stride` apart, beams
    /// `out_beam_stride` apart) from level k, then level k's rings.
    void run_top(float* out,
                 int64_t out_beam_stride,
                 int out_row_stride,
                 cudaStream_t stream);

    /// Level k's newest samples of this chunk into its rings.
    void update_top_rings(cudaStream_t stream);

    /// Rewrites the tree histories of levels k+1..root from the rings at the
    /// current stream position, after top chunks left them behind.
    void rebuild_top_history(cudaStream_t stream) const;

    [[nodiscard]] SizeType top_workspace_bytes() const noexcept;

    [[nodiscard]] SizeType direct_window_bytes() const noexcept;

    [[nodiscard]] SizeType direct_workspace_bytes() const noexcept;

    /// Returns the number of leading sum coordinates whose tree-history ring
    /// the merge kernel already advanced (0 when the advance was not fused).
    int execute_iter_device(SizeType next_level,
                            cudaStream_t stream,
                            bool regular_only = false,
                            bool advance_hist = true);

    int execute_iter_levels(SizeType next_level,
                            cudaStream_t stream,
                            bool regular_only = false,
                            bool advance_hist = true);

    template <typename TIn, typename TOut>
    int execute_iter_typed(const TIn* __restrict__ in_ptr,
                           TOut* __restrict__ out_ptr,
                           SizeType next_level,
                           cudaStream_t stream,
                           bool regular_only = false,
                           bool advance_hist = true);

    /// Advances `level`'s tree history after its merges.
    template <typename TIn>
    void advance_tree_history(const TIn* in_ptr,
                              SizeType level,
                              cudaStream_t stream,
                              int coord_begin = 0);

    template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
    void launch_init(const Input& waterfall_d,
                     TOut* __restrict__ state_d,
                     dim3 grid_size,
                     dim3 block_size,
                     cudaStream_t stream);

    template <typename Input, typename TOut>
    void initialise_device(const Input& waterfall_d,
                           TOut* __restrict__ state_d,
                           cudaStream_t stream);

    // Valid mode: advance the per-channel level-0 input history by this
    // block (independent of how level 0 itself was computed).
    template <typename Input>
    void advance_level0_history(const Input& waterfall_d, cudaStream_t stream);
}; // End FDMTCudaEngine definition

} // namespace dmt::algorithms::detail::fdmt_cuda
