#pragma once

// Private: the CPU FDMT engine (detail::make_fdmt_cpu). The class is
// declared here and defined in fdmt_cpu.cpp (construction and memory
// planning), fdmt_cpu_execute.cpp (execute(), the search, the tree
// levels and their kernels) and fdmt_cpu_stream.cpp (the stepper and
// the stream history).

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_history.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_search.hpp"
#include "dmt/fdmt_skew.hpp"
#include "dmt/fdmt_tree.hpp"

namespace dmt::algorithms::detail::fdmt_cpu {

/**
 * @brief A large zero-initialised array for the engine's tree state and
 * tree history: 2 MiB-aligned, with transparent huge pages requested on
 * Linux, and zero-filled by `nthreads` threads with a static schedule.
 *
 * Huge pages: a fused time tile writes short runs of thousands of rows that
 * lie a chunk apart, one 4 KiB page each, so with small pages nearly every
 * run misses the TLB. The parallel fill first-touches the pages across the
 * threads (and so across NUMA nodes) instead of on the constructing thread.
 */
template <typename T> class FDMTLargeArray {
public:
    FDMTLargeArray() = default;
    ~FDMTLargeArray() { std::free(m_data); } // NOLINT
    FDMTLargeArray(const FDMTLargeArray&)            = delete;
    FDMTLargeArray& operator=(const FDMTLargeArray&) = delete;
    FDMTLargeArray(FDMTLargeArray&&)                 = delete;
    FDMTLargeArray& operator=(FDMTLargeArray&&)      = delete;

    void assign(SizeType n, int nthreads) {
        std::free(m_data); // NOLINT
        m_data = nullptr;
        m_size = n;
        if (n == 0) {
            return;
        }
        constexpr SizeType kAlign = SizeType{2} << 20;
        const SizeType bytes =
            (((n * sizeof(T)) + kAlign - 1) / kAlign) * kAlign;
        m_data = static_cast<T*>(std::aligned_alloc(kAlign, bytes)); // NOLINT
        if (m_data == nullptr) {
            throw std::bad_alloc();
        }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        ::madvise(m_data, bytes, MADV_HUGEPAGE); // a hint; failure is harmless
#endif
        auto* raw            = reinterpret_cast<std::byte*>(m_data);
        const SizeType pages = bytes / kAlign;
#pragma omp parallel for num_threads(nthreads) schedule(static)
        for (SizeType i = 0; i < pages; ++i) {
            std::memset(raw + (i * kAlign), 0, kAlign);
        }
    }
    [[nodiscard]] T* data() noexcept { return m_data; }
    [[nodiscard]] const T* data() const noexcept { return m_data; }
    [[nodiscard]] SizeType size() const noexcept { return m_size; }
    [[nodiscard]] bool empty() const noexcept { return m_size == 0; }
    [[nodiscard]] T* begin() noexcept { return m_data; }
    [[nodiscard]] T* end() noexcept { return m_data + m_size; }
    [[nodiscard]] const T* begin() const noexcept { return m_data; }
    [[nodiscard]] const T* end() const noexcept { return m_data + m_size; }

private:
    T* m_data{nullptr};
    SizeType m_size{0};
};

/**
 * @brief One merge (or copy) of the engine's tree, from the plan's
 * coordinate: element offsets of the output row and of its tail / head
 * child rows within their levels' buffers (re-laid out for the engine's row
 * length when it executes in chunks), row lengths, the merge delay and the
 * valid-mode tree-history slot.
 */
struct FDMTMergeOp {
    SizeType out;
    SizeType tail;
    SizeType head; ///< unused for a copy
    SizeType nsamps;
    SizeType tail_nsamps;
    SizeType delay;
    SizeType hist;
    // Row (coordinate) indices of out / tail / head within their levels, for
    // the fused time tiles, whose scratch rows have their own length.
    SizeType out_row;
    SizeType tail_row;
    SizeType head_row; ///< unused for a copy
};

/// Row lengths of one call: the ops' own, or `active` samples of every row
/// (a valid-mode chunk, where every row has the same length).
struct FDMTRowLengths {
    SizeType active{0};
    [[nodiscard]] SizeType out(const FDMTMergeOp& op) const noexcept {
        return active != 0 ? active : op.nsamps;
    }
    [[nodiscard]] SizeType tail(const FDMTMergeOp& op) const noexcept {
        return active != 0 ? active : op.tail_nsamps;
    }
};

/**
 * @brief Per-thread scratch carved out of one engine-owned buffer allocated
 * at construction (see FDMTCpuEngine::m_workspace), so execute() never
 * allocates. Thread t's slice starts at base + t * stride and holds, at
 * fixed offsets: one unpacked packed-input row (`unpack`, nsamps samples of
 * <= 4 bytes), two running level-0 box rows for mixed-sign dt grids
 * (`rows`), and the level-fusion ping-pong buffers (`fused_a`/`fused_b`).
 * Every parallel region that uses it runs with exactly `nthreads` threads.
 */
struct FDMTWorkspace {
    std::byte* base{nullptr};
    SizeType stride{0};
    SizeType row_bytes{0};   ///< one row of nsamps samples, <= 4 bytes each
    SizeType fused_bytes{0}; ///< one fusion ping-pong buffer
    int nthreads{1};

    [[nodiscard]] std::byte* slice(SizeType tid) const noexcept {
        return base + (tid * stride);
    }
    template <typename T> [[nodiscard]] T* unpack(SizeType tid) const noexcept {
        return reinterpret_cast<T*>(slice(tid));
    }
    template <typename T> [[nodiscard]] T* rows(SizeType tid) const noexcept {
        return reinterpret_cast<T*>(slice(tid) + row_bytes);
    }
    [[nodiscard]] std::byte* fused_a(SizeType tid) const noexcept {
        return slice(tid) + (3 * row_bytes);
    }
    [[nodiscard]] std::byte* fused_b(SizeType tid) const noexcept {
        return fused_a(tid) + fused_bytes;
    }
};

/**
 * @brief Level-0 box means (fdmt_box_weights.hpp), indexed by |dt| = s (box
 * length s + 1): the dyadic float weight w(s + 1) for float input, and for
 * packed input the integer rounding recipe on the plan's common scale 2^-F
 * (`unit` = 2^-F).
 */
struct FDMTBoxWeights {
    const float* w_f32{nullptr};
    const detail::FDMTBoxRound* round{nullptr};
    float unit{1.0F};
};

/**
 * @brief One beam's level-0 input: either a float waterfall or a packed
 * low-bit one (LSB-first rows of `row_bytes` bytes, see bit_pack_utils.hpp).
 * Packed rows are unpacked one sub-band at a time into a per-thread scratch
 * row, so a copy of the whole waterfall is never materialized.
 */
struct FDMTInput {
    const float* f32{nullptr};
    const uint8_t* packed{nullptr};
    SizeType nbits{32};
    SizeType row_bytes{0}; ///< packed bytes per channel row of the block
    SizeType row_samps{0}; ///< float samples per channel row of the block
    SizeType t0{0};        ///< first sample of the current chunk (packed:
                           ///< a whole number of bytes into the row)

    // Samples [t0, t0 + nsamps) of row `i_sub` as type TS: the float
    // waterfall row itself, or the packed row unpacked into `scratch`
    // (nsamps elements). TS other than float is only requested for packed
    // input.
    template <typename TS>
    [[nodiscard]] const TS* row(SizeType i_sub,
                                SizeType nsamps,
                                TS* __restrict__ scratch) const noexcept {
        if (packed == nullptr) {
            if constexpr (std::is_same_v<TS, float>) {
                return f32 + (i_sub * row_samps) + t0;
            } else {
                return nullptr;
            }
        }
        bit_pack_utils::unpack_row(packed + (i_sub * row_bytes) +
                                       ((t0 * nbits) / 8),
                                   nbits, nsamps, scratch);
        return scratch;
    }
};

class FDMTCpuEngine final : public detail::FDMTEngine {
public:
    FDMTCpuEngine(const plans::FDMTPlan& plan,
                  const detail::FDMTEngineConfig& cfg);

    ~FDMTCpuEngine() override                      = default;
    FDMTCpuEngine(const FDMTCpuEngine&)            = delete;
    FDMTCpuEngine& operator=(const FDMTCpuEngine&) = delete;
    FDMTCpuEngine(FDMTCpuEngine&&)                 = delete;
    FDMTCpuEngine& operator=(FDMTCpuEngine&&)      = delete;

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override;

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt) override;

    void execute_search(std::span<const float> waterfall,
                        detail::FDMTSearchSink& sink) override;

    void execute_search(std::span<const uint8_t> waterfall,
                        SizeType nbits,
                        detail::FDMTSearchSink& sink) override;

    template <typename TInt>
    void execute_search_typed(std::span<const uint8_t> waterfall,
                              SizeType nbits,
                              detail::FDMTSearchSink& sink,
                              TInt* history);

    void reset(std::span<const float> waterfall, std::span<float> dmt) override;

    void reset(std::span<const uint8_t> waterfall_packed,
               SizeType nbits,
               std::span<float> dmt) override;

    [[nodiscard]] SizeType get_fuse_levels() const noexcept override {
        return m_fuse_levels;
    }

    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept override {
        return m_chunk;
    }

    [[nodiscard]] uint64_t search_latency() const noexcept override {
        return m_skew ? m_skew->max_emit_lag() : 0;
    }
    [[nodiscard]] std::string search_path() const override {
        if (m_skew) {
            return fmt::format(
                "skewed direct 0-{}, fused {}-{}", m_skew->get_leaf_level(),
                m_skew->get_leaf_level(), m_skew->get_fuse_level());
        }
        return {};
    }

    [[nodiscard]] std::string summary() const override;

    [[nodiscard]] FDMTMemoryUsage get_memory_usage() const noexcept override;

    void advance(SizeType levels, Stream /*stream*/) override;

    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override;

    [[nodiscard]] std::span<const float> view_level_data() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        const auto& shape = m_plan.get_container().state_shape[m_current_level];
        return {current_level_f32(),
                shape.ncoords * m_row_len[m_current_level]};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override;

    [[nodiscard]] SizeType current_level() const noexcept override {
        return m_current_level;
    }

    [[nodiscard]] SizeType total_levels() const noexcept {
        return m_plan.get_niters() + 1;
    }

    [[nodiscard]] SizeType num_subbands() const override {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMT: Stepper is not initialized. Call reset() first.");
        }
        return m_plan.get_container().grids[m_current_level].size();
    }

    [[nodiscard]] bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(Stream stream) override;

    void reset_history() noexcept override;

    // Off the skewed path: the header (fdmt_history.hpp), then the level-0
    // input history and the tree history; 0 without a history.
    [[nodiscard]] SizeType history_state_size() const noexcept override {
        if (m_skew) {
            return m_skew->history_state_size();
        }
        const SizeType n = m_history.size() + m_tree_history.size();
        return n == 0 ? 0 : detail::kFDMTHistHeader + n;
    }

    // Layout: the level-0 input history, then the tree history of every
    // beam with each coordinate's slot in FIFO order (oldest sample first),
    // whatever the ring position.
    void save_history(std::span<float> out) const override;

    void load_history(std::span<const float> in) override;

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    // Storage type of one tree level's state (int_tree).
    using Elem = detail::FDMTLevelType;

    // Where one level's state lives: `base` of beam 0, beams `beam_stride`
    // bytes apart.
    struct LevelBuf {
        std::byte* base{nullptr};
        Elem type{Elem::kF32};
        SizeType beam_stride{0};
    };

    // Everything below is allocated in the constructor (see
    // allocate_working_memory()); execute() and the stepper only index it.
    bool m_use_box_smearing;
    bool m_int_tree;
    FDMTMode m_mode;
    SizeType m_nbeams;
    int m_nthreads;
    const plans::FDMTPlan& m_plan; // owned by the FDMT facade
    // Row length of the tree state: blocks run in chunks of m_chunk samples
    // (valid mode; otherwise one chunk of nsamps).
    SizeType m_chunk{0};
    // Per level: row length in the engine's buffers, and the merge / copy
    // ops by coordinate (levels >= 1; see FDMTMergeOp).
    std::vector<SizeType> m_row_len;
    std::vector<std::vector<FDMTMergeOp>> m_ops;
    SizeType m_ops_elems{0};
    // The two state halves: level l < niters lives in half l % 2 (the root
    // is written straight into the caller's dmt buffer). Byte storage holds
    // float, uint8_t and uint16_t levels alike. Zero-filled.
    FDMTLargeArray<std::byte> m_state_internal;
    SizeType m_half_elems{0}; // elements of one beam's level in a half
    // Valid-mode streaming state, always float: the last dt0_max raw input
    // samples of every channel, and each merge's tree-history ring (see
    // offset_add), at stream position m_stream_pos.
    std::vector<float> m_history;
    FDMTLargeArray<float> m_tree_history;
    SizeType m_stream_pos{0};
    SizeType m_search_nbits{32};
    bool m_search_upper{false};
    bool m_search_u32{false};
    std::unique_ptr<detail::FDMTSkewCpu> m_skew;
    /// Leaf level of the skewed engine (choose_skew; 0: automatic).
    SizeType m_skew_leaf{0};
    /// Plain-tree history layout (fdmt_tree_history_hash), saved in and
    /// checked against the history header.
    uint64_t m_hist_hash{0};
    /// Stream position of the last loaded history (0 after reset_history());
    /// the saved position is this plus m_stream_pos.
    uint64_t m_hist_pos_base{0};
    FDMTLargeArray<uint64_t> m_integer_history;
    FDMTLargeArray<uint32_t> m_integer32_history;

    // Level-0 box means (use_box_smearing; see fdmt_box_weights.hpp): dyadic
    // float weights, the packed-input rounding recipes on the common scale
    // 2^-F, whether a recipe's product needs 64 bits (by nbits), and the
    // value of one integer state unit, 2^-F (1 without box smearing).
    std::vector<float> m_box_w_f32;
    std::vector<detail::FDMTBoxRound> m_box_round;
    std::array<SizeType, 17> m_box_round_bytes{};
    FDMTBoxWeights m_box_weights;
    float m_int_unit{1.0F};

    SizeType m_fuse_levels{0}; // resolved fuse_levels
    // Valid mode: the fused levels run in time tiles of this many samples
    // (0: one tile per chunk, the whole-chunk fused path).
    SizeType m_fuse_tile{0};
    // Time-tiled fusion: ring position of every fused merge (levels
    // 1..fuse, by op index), advanced tile by tile.
    std::vector<std::vector<SizeType>> m_tile_ring_pos;
    // Per-level storage types: all float, and per nbits for int_tree.
    std::vector<Elem> m_float_levels;
    std::array<std::vector<Elem>, 17> m_int_levels; // indexed by nbits
    // Per-thread scratch (see FDMTWorkspace) and its layout.
    std::vector<std::byte> m_workspace;
    FDMTWorkspace m_ws;

    // Current block / chunk.
    std::vector<LevelBuf> m_levels;
    FDMTInput m_input{};
    SizeType m_input_beam_stride{0};
    SizeType m_active{0}; // samples of the current chunk (valid mode)
    SizeType m_current_level{0};
    bool m_is_initialized{false};

    template <typename F> static void with_elem(Elem e, F&& f) {
        switch (e) {
        case Elem::kU8:
            f.template operator()<uint8_t>();
            break;
        case Elem::kU16:
            f.template operator()<uint16_t>();
            break;
        case Elem::kF32:
            f.template operator()<float>();
            break;
        }
    }

    template <typename T>
    [[nodiscard]] T* level_ptr(SizeType level, SizeType beam) const noexcept {
        const auto& lb = m_levels[level];
        return reinterpret_cast<T*>(lb.base + (beam * lb.beam_stride));
    }

    [[nodiscard]] const float* current_level_f32() const {
        if (m_levels[m_current_level].type != Elem::kF32) {
            throw std::logic_error(fmt::format(
                "FDMT: level {} is stored as an integer type "
                "(int_tree); construct with int_tree=false to inspect "
                "intermediate levels.",
                m_current_level));
        }
        return level_ptr<const float>(m_current_level, 0);
    }

    // ---------------------------------------------------------------------
    // Inputs and blocks
    // ---------------------------------------------------------------------

    [[nodiscard]] FDMTInput float_input(std::span<const float> waterfall);

    [[nodiscard]] FDMTInput packed_input(std::span<const uint8_t> packed,
                                         SizeType nbits);

    [[nodiscard]] const std::vector<Elem>&
    packed_levels(SizeType nbits) const noexcept {
        return m_int_tree ? m_int_levels[nbits] : m_float_levels;
    }

    void check_dmt_size(std::span<float> dmt) const;

    // In "valid" mode every block must reach the root before the next one
    // starts: levels a stepper run never executed would otherwise skip this
    // block's cross-block history update and silently corrupt the stream.
    void check_previous_block_finished() const;

    /**
     * Assigns each level's buffer for this block: level l < niters in state
     * half l % 2 (so a merge never reads and writes the same half), the root
     * in the caller's dmt buffer at the chunk's first sample. Integer levels
     * use the halves' leading bytes.
     */
    void layout_levels(const std::vector<Elem>& types,
                       std::span<float> dmt,
                       SizeType t0);

    // One chunk [t0, t0 + n) of the block: input view, level layout, length.
    void start_chunk(const FDMTInput& input,
                     const std::vector<Elem>& types,
                     std::span<float> dmt,
                     SizeType t0,
                     SizeType n);

    // execute(): the block's chunks, each from level 0 (fused when
    // fuse_levels > 0) to the root.
    void run_block(const FDMTInput& input,
                   const std::vector<Elem>& types,
                   std::span<float> dmt);

    // reset(): one whole block, level by level (inspectable).
    void start_stepper(const FDMTInput& input,
                       const std::vector<Elem>& types,
                       std::span<float> dmt);

    // After the root of a chunk: the stream moves on.
    void end_chunk() noexcept {
        if (m_mode == FDMTMode::kValid) {
            m_stream_pos += m_active;
        }
    }

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    // Automatic fusion depth (kFDMTAutoFuse): the deepest depth whose two
    // per-thread scratch buffers fit max(kFuseBudgetTotal / nthreads,
    // kFuseBudgetPerThread) -- a rule on the plan and thread count only, no
    // hardware detection. Fitted on an ARM64 (NEON) laptop and a Xeon Gold
    // 6348H (4096 channels, dt_max 2048, 4K-64K samples, 1-8 threads): within
    // 5% of the best fixed depth except 4K-sample blocks on the Xeon (6-8.5%).
    // The per-thread floor of 8 MB keeps the same choices there and lets a
    // high-DM plan (1024 channels at 512 us, DM 2000: ~22 rows per channel,
    // 2-8K-sample chunks) fuse at 8 threads, within 1% of its best depth
    // on the ARM64 laptop, where 5 MB fell back to no fusion (+18%).
    static constexpr SizeType kFuseBudgetTotal     = SizeType{36} << 20;
    static constexpr SizeType kFuseBudgetPerThread = SizeType{8} << 20;

    // Valid mode fuses in time tiles instead (initialise_fused_tiled): the
    // deepest depth, then the longest tile of kFuseTiles, whose mean
    // per-group working set (fused_tile_working_set) is at most
    // kFuseTileBudget, with at least kFuseGroupsPerThread groups per thread
    // for load balance -- again from the plan and thread count only. Tiles
    // shorter than 256 samples lose more to per-merge overhead than they
    // gain in locality. Fitted on a Xeon Gold 6348H at 8 threads (1 MiB L2
    // per core): float input, 512 us, 1024 channels to DM 2000 (F=4, T=512,
    // -31% against the untiled F=1) and 4096 channels to DM 1000 (F=7,
    // T=256, -27%), each within 1% of the best (F, T) measured.
    static constexpr std::array<SizeType, 2> kFuseTiles = {512, 256};
    static constexpr SizeType kFuseTileBudget           = SizeType{1792} << 10;
    static constexpr SizeType kFuseGroupsPerThread      = 2;

    static constexpr SizeType round_up_64(SizeType n) noexcept {
        return ((n + 63) / 64) * 64;
    }

    /**
     * Resolves fuse_levels and allocates every buffer execute() and the
     * stepper use: the merge ops, the state halves and histories, per-level
     * type tables, the box means and the per-thread workspace (see
     * FDMTWorkspace). Nothing is allocated after this.
     */
    /// Fuse level requested of the skewed-window engine (0: automatic).
    [[nodiscard]] SizeType skew_fuse_request(SizeType fuse_levels,
                                             SizeType max_fuse_levels) const;

    /**
     * (chunk, leaf level) candidates of the skewed-window engine, fastest
     * first. Measured on parkes_uwl (6656 channels, 8 threads one per CCD
     * on an EPYC 9654, s per 2.1 s block): float ring K=1 4.96 at chunk
     * 4096 and 7.4 at 2048, K=2 6.6 / 9.4, K=3 8.8 / 11.1, K=4 13.3 /
     * 15.3, K=1 13.6 at 1024 (levels 1..K-1 of a direct leaf cost double
     * running-sum gathers, the binary merges above them are cheap); 8-bit
     * ring K=2 4.08, K=3 4.38, K=4 4.69, K=1 4.77 at 4096, K=4 5.99 and
     * K=2 6.28 at 2048. Leaf 0: the automatic level (<= 4).
     */
    static std::vector<std::pair<SizeType, SizeType>>
    skew_candidates(SizeType ring_elem);

    /**
     * Tree path or skewed-window engine (fdmt_skew.hpp), under the memory
     * budget: the tree path (fastest) when its history and state fit
     * max_memory_bytes, else the skewed engine at the fastest (chunk, leaf
     * level) of skew_candidates() whose estimate fits (an explicit
     * chunk_nsamps is kept, with the fastest leaf level that fits it).
     * Sets m_chunk and m_skew_leaf. Throws when nothing fits.
     */
    bool choose_skew(const detail::FDMTEngineConfig& cfg);

    void allocate_working_memory(SizeType fuse_levels,
                                 SizeType max_fuse_levels,
                                 bool skew);

    /**
     * The engine's merge ops, from the plan's coordinates. Levels below the
     * root are re-laid out for rows of m_row_len[l] samples (the chunk
     * length in valid mode; the plan's own rows otherwise); the root keeps
     * the plan's layout, which is the dmt output's.
     */
    void build_merge_ops();

    // Calls f(beam_offset, slot_offset, delay, ring_position) for every
    // tree-history ring of every beam.
    template <typename F> void for_each_ring(F&& f) const;

    // Row lengths of the current chunk's merges.
    [[nodiscard]] FDMTRowLengths row_lengths() const noexcept {
        return {.active = (m_mode == FDMTMode::kValid) ? m_active : 0};
    }

    // ---------------------------------------------------------------------
    // Level 0
    // ---------------------------------------------------------------------

    [[nodiscard]] FDMTInput beam_input(SizeType beam) const noexcept;

    // Beam-major layout: beam b's waterfall/state/history all live in
    // disjoint, per-beam-sized slices of the same flat buffers (the plan's
    // coordinate offsets are beam-agnostic and apply identically within each
    // slice), so the beams run one after the other.
    void initialise();

    // Level-0 rows of channel `c` of beam `beam` into `rows`, plus its
    // valid-mode input history, using thread `tid`'s scratch: the current
    // chunk, or (valid mode, `nsamps` > 0) its `nsamps` samples from
    // `input_b.t0` into rows of `stride` samples (a fused time tile).
    template <typename T>
    void init_channel(const FDMTInput& input_b,
                      T* rows,
                      SizeType c,
                      SizeType beam,
                      SizeType tid,
                      SizeType nsamps = 0,
                      SizeType stride = 0);

    // ---------------------------------------------------------------------
    // Level fusion (fuse_levels)
    // ---------------------------------------------------------------------

    // Coordinate range [begin, end) of level `l` owned by fused group `g`:
    // the level-l sub-bands descending from level-F sub-band g (see
    // fdmt_tree.hpp).
    [[nodiscard]] std::pair<SizeType, SizeType>
    fused_coord_range(SizeType l, SizeType g, SizeType fuse) const noexcept {
        return detail::fdmt_group_coords(m_plan.get_container(), fuse, g, l);
    }

    // Merges level-l coordinates [c_begin, c_end) of one fused group. `in`
    // / `out` hold the group's rows starting at coordinates in_base /
    // out_base (out_base 0 with `to_level`: the level's own buffer); the
    // merge itself is the unmodified offset_add / copy used by fdmt_iter,
    // so fused results are bit-identical.
    template <FDMTMode Mode, typename TIn, typename TOut>
    void fused_merge_group(const TIn* __restrict__ in,
                           SizeType in_base,
                           TOut* __restrict__ out,
                           SizeType out_base,
                           bool to_level,
                           float* __restrict__ hist_b,
                           SizeType l,
                           SizeType c_begin,
                           SizeType c_end) const noexcept;

    template <FDMTMode Mode>
    void fused_merge_dispatch(const std::byte* in,
                              SizeType in_base,
                              std::byte* out,
                              SizeType out_base,
                              bool to_level,
                              float* hist_b,
                              SizeType l,
                              SizeType c_begin,
                              SizeType c_end) const noexcept;

    // Level-0 rows of channels [ch_begin, ch_end) into `buf` (rows from
    // coordinate `base`), plus their valid-mode input history -- exactly
    // the per-sub-band work of initialise(), for one fused group.
    template <typename T0>
    void fused_init_group(const FDMTInput& input_b,
                          T0* __restrict__ buf,
                          SizeType base,
                          SizeType ch_begin,
                          SizeType ch_end,
                          SizeType beam,
                          SizeType tid);

    // Largest per-group scratch level buffer for fusion depth `fuse` (4
    // bytes/element covers every storage type); two are live per thread.
    // `tile` > 0: rows of `tile` samples (the time-tiled path).
    [[nodiscard]] SizeType
    fused_scratch_bytes(SizeType fuse, SizeType tile = 0) const noexcept;

    // Mean over the level-F groups of the bytes one thread touches while it
    // walks a group in tiles of `tile` samples: its two scratch buffers
    // (float-sized) and the tree-history rings of its fused merges.
    [[nodiscard]] double fused_tile_working_set(SizeType fuse,
                                                SizeType tile) const noexcept;

    // Tile length for valid-mode fusion depth `fuse`: the longest of
    // kFuseTiles within kFuseTileBudget (else the shortest), capped at the
    // chunk.
    [[nodiscard]] SizeType fuse_tile_for(SizeType fuse) const noexcept;

    // kFDMTAutoFuse (see kFuseBudgetTotal and kFuseTileBudget).
    [[nodiscard]] SizeType auto_fuse_levels() const noexcept;

    /**
     * Fused replacement for initialise() + levels 1..fuse. Level-F sub-band
     * g depends only on the 2^F input channels below it, so each thread
     * builds one such group at a time: level-0 rows into a per-thread
     * scratch buffer, then each merge level ping-pongs between two scratch
     * buffers, and only level F is written to its real state buffer. The
     * group's rows (a few rows per channel) stay cache-resident, removing
     * the main-memory write + read of every intermediate level.
     */
    void initialise_fused(SizeType fuse);

    // Merges level-l coordinates [c_begin, c_end) of one fused group over the
    // time tile [ts, ts + n) of the chunk. `in` / `out` are scratch rows of
    // m_fuse_tile samples starting at coordinates in_base / out_base, or
    // (`to_level`) `out` is the level's own state buffer (rows of the chunk
    // length, written at column ts). Valid mode only: each merge is the
    // unmodified offset_add, whose tree-history ring carries the head
    // samples across tiles exactly as it does across chunks.
    template <typename TIn, typename TOut>
    void fused_merge_tile(const TIn* __restrict__ in,
                          SizeType in_base,
                          TOut* __restrict__ out,
                          SizeType out_base,
                          bool to_level,
                          float* __restrict__ hist_b,
                          SizeType l,
                          SizeType c_begin,
                          SizeType c_end,
                          SizeType ts,
                          SizeType n) noexcept;

    // fused_merge_tile for the storage types of levels l - 1 and l.
    void fused_tile_dispatch(const std::byte* in,
                             SizeType in_base,
                             std::byte* out,
                             bool to_level,
                             float* hist_b,
                             SizeType l,
                             SizeType c_begin,
                             SizeType c_end,
                             SizeType ts,
                             SizeType n) noexcept;

    /**
     * Time-tiled level fusion (valid mode, m_fuse_tile > 0). Like
     * initialise_fused(), one thread owns one level-F channel group at a
     * time, but walks the chunk in tiles of m_fuse_tile samples: level 0 and
     * levels 1..F of the tile live in two scratch buffers of group rows x
     * tile, and level F is written to its state buffer. A merge whose delay
     * exceeds the tile reads its head samples from its tree-history ring,
     * which the previous tile of the same group has just written, so a
     * group's rings stay cache-resident while it walks the chunk. The
     * results equal the untiled path bit for bit (valid mode is independent
     * of the block length).
     */
    void initialise_fused_tiled(SizeType fuse);

    // ---------------------------------------------------------------------
    // Tree levels
    // ---------------------------------------------------------------------

    template <FDMTMode Mode, typename TIn, typename TOut>
    void execute_iter_typed(SizeType i_iter, bool regular_only = false);

    void execute_iter(SizeType i_iter, bool regular_only = false);

}; // End FDMTCpuEngine definition

} // namespace dmt::algorithms::detail::fdmt_cpu
