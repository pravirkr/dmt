#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_planned_vector.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/kadane_cuda.cuh"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {

inline constexpr int kItems = 4;
inline constexpr int kWarp =
    gpu_compat::kLogicalWarp; // a 64-lane wavefront holds two
// Width and threshold tables are staged in shared memory up to this count;
// longer tables are read through the cache.
inline constexpr SizeType kSharedFilters = 512;
// Widest boxcar of the fast path (prefix ring in shared memory); wider
// boxes take the device-memory ring (kGlobalRing), with no speed promise.
inline constexpr SizeType kFastMaxWidth = 4096;
// Device memory of the kGlobalRing rings: blocks are launched in batches
// whose rings fit it (at least one block).
inline constexpr SizeType kGlobalRingBytes = SizeType{256} << 20;
inline constexpr int kGlobalBlock          = 256;

// Radix-select key length for OverflowPolicy::kKeepBrightest:
// 18 bytes (4B snr + 8B peak + 4B rank + 2B filter).
inline constexpr int kKeepKeyBytes = 18;

struct KeepSelect {
    uint64_t n        = 0; ///< entries in the stores
    uint64_t k        = 0; ///< still to select within the prefix
    uint32_t resolved = 0; ///< selection fixed by prefix[0, depth)
    uint32_t depth    = 0;
    uint8_t prefix[kKeepKeyBytes]{};
};

// Fixed part of the block's shared memory. The dynamic part follows it:
// the prefix ring (a power of two of at least max_width + kBlockTile entries),
// then optionally the thresholds and widths of the row.
template <typename T, int kBlockThreads> struct BoxcarShared {
    BoxcarRunSummary runs[kBlockThreads];
    T warp_totals[kBlockThreads / kWarp];
};

template <typename T> DMT_D inline T shfl_up(T value, int delta) {
    return gpu_compat::shfl_up(value, static_cast<unsigned>(delta));
}

DMT_D inline uint64_t increment(uint64_t* count) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ULL);
}

/// Where a row's candidates go (kDropNew): the call's slots, or with
/// BoxcarConfig::max_candidates_per_plane its plane's counter and share of
/// a staging buffer (compacted after the call).
struct PlaneShare {
    uint64_t* slots;
    SizeType capacity;
    SizeType base;
};
DMT_HD inline PlaneShare plane_share(uint64_t* slots,
                                     SizeType capacity,
                                     uint64_t* plane_slots,
                                     SizeType plane_cap,
                                     uint32_t plane) {
    if (plane_cap == 0) {
        return {slots, capacity, 0};
    }
    return {plane_slots + plane, plane_cap, plane * plane_cap};
}

struct CandidateWriter {
    RawCandidate* candidates;
    CandidateCounts* counts;
    uint64_t* slots;
    uint64_t* per_dm;
    const uint32_t* widths;
    RawCandidate prototype;
    SizeType capacity;
    SizeType dm_cap;
    SizeType base       = 0; ///< first slot (a plane's share), else 0
    const double* means = nullptr;
    const float* scales = nullptr;
    double unit         = 1;
    bool integer        = false;

    DMT_D void operator()(const BoxcarPeak& peak) const {
        increment(&counts->total_detected);
        // Every detection counts in its row (per-plane counts); dm_cap is
        // SIZE_MAX when max_cands_per_dm is 0.
        if (increment(per_dm) >= dm_cap) {
            increment(&counts->dropped_per_dm);
            return;
        }
        const uint64_t slot = increment(slots);
        if (slot >= capacity) {
            increment(&counts->dropped_global);
            return;
        }
        RawCandidate candidate = prototype;
        candidate.peak_sample  = peak.peak;
        candidate.begin_sample = peak.begin;
        candidate.end_sample   = peak.end;
        candidate.snr =
            integer ? static_cast<float>((static_cast<double>(peak.raw_sum) -
                                          means[peak.filter]) *
                                         (scales[peak.filter] * unit))
                    : peak.snr;
        candidate.filter_idx    = peak.filter;
        candidate.width         = widths[peak.filter];
        candidates[base + slot] = candidate;
        increment(&counts->emitted_count);
    }
};

// OverflowPolicy::kKeepBrightest. Within one row: brighter() restricted to
// the fields that differ (S/N, peak sample, width).
DMT_D inline bool keep_brighter(const BoxcarKeepEntry& a,
                                const BoxcarKeepEntry& b) {
    if (a.snr != b.snr) {
        return a.snr > b.snr;
    }
    if (a.peak != b.peak) {
        return a.peak < b.peak;
    }
    return a.filter < b.filter;
}

// A row's store: a heap of at most q entries, dimmest at the top, with
// `*count` entries. One thread inserts at a time.
DMT_D inline void keep_insert(BoxcarKeepEntry* store,
                              uint64_t* count,
                              SizeType q,
                              const BoxcarKeepEntry& entry,
                              CandidateCounts* counts) {
    uint64_t n = *count;
    if (n < q) {
        uint64_t i = n;
        store[i]   = entry;
        while (i > 0) {
            const uint64_t p = (i - 1) / 2;
            if (!keep_brighter(store[p], store[i])) {
                break;
            }
            const BoxcarKeepEntry t = store[p];
            store[p]                = store[i];
            store[i]                = t;
            i                       = p;
        }
        *count = n + 1;
        return;
    }
    increment(&counts->dropped_per_dm);
    if (!keep_brighter(entry, store[0])) {
        return;
    }
    store[0]   = entry;
    uint64_t i = 0;
    for (;;) {
        const uint64_t l = (2 * i) + 1;
        if (l >= q) {
            break;
        }
        const uint64_t r = l + 1;
        const uint64_t c =
            (r < q && keep_brighter(store[l], store[r])) ? r : l; // dimmer
        if (!keep_brighter(store[i], store[c])) {
            break;
        }
        const BoxcarKeepEntry t = store[c];
        store[c]                = store[i];
        store[i]                = t;
        i                       = c;
    }
}

// A thread's candidates since the last keep_drain(): at most one per
// sample of its kItems (one width at a time, or one run end per sample).
struct KeepPending {
    BoxcarKeepEntry entry[kItems];
    int n = 0;
};

// kKeepBrightest emission in the scoring kernel: the S/N is final here;
// a candidate dimmer than a full store's dimmest is dropped at once, the
// rest wait in the thread's KeepPending for keep_drain().
struct KeepWriter {
    CandidateCounts* counts;
    BoxcarKeepEntry* store; // this row's q entries
    uint64_t* count;        // entries in the store
    uint64_t* detected;     // the row's detections (per-plane counts)
    SizeType q;
    KeepPending* pending;
    const double* means = nullptr;
    const float* scales = nullptr;
    double unit         = 1;
    bool integer        = false;

    DMT_D void operator()(const BoxcarPeak& peak) const {
        increment(&counts->total_detected);
        increment(detected);
        const float snr =
            integer ? static_cast<float>((static_cast<double>(peak.raw_sum) -
                                          means[peak.filter]) *
                                         (scales[peak.filter] * unit))
                    : peak.snr;
        if (*count >= q && snr < store[0].snr) {
            increment(&counts->dropped_per_dm);
            return;
        }
        pending->entry[pending->n++] = {peak.peak, peak.begin, peak.end, snr,
                                        peak.filter};
    }
};

// Block-uniform: every thread's pending candidates into the row's store.
// Each round a thread stages one in its slot of shared.runs (unused outside
// reduce_runs()) and thread 0 inserts them in slot order.
template <typename T, int kBlockThreads>
DMT_D inline void keep_drain(BoxcarShared<T, kBlockThreads>& shared,
                             KeepPending& pending,
                             const KeepWriter& writer) {
    const int tid = static_cast<int>(threadIdx.x);
    while (__syncthreads_or(pending.n > 0) != 0) {
        BoxcarPeak& slot = shared.runs[tid].trailing;
        slot.open        = pending.n > 0;
        if (pending.n > 0) {
            const BoxcarKeepEntry& e = pending.entry[--pending.n];
            slot.peak                = e.peak;
            slot.begin               = e.begin;
            slot.end                 = e.end;
            slot.snr                 = e.snr;
            slot.filter              = e.filter;
        }
        __syncthreads();
        if (tid == 0) {
            for (int i = 0; i < kBlockThreads; ++i) {
                const BoxcarPeak& s = shared.runs[i].trailing;
                if (s.open) {
                    keep_insert(writer.store, writer.count, writer.q,
                                {s.peak, s.begin, s.end, s.snr, s.filter},
                                writer.counts);
                }
            }
        }
        __syncthreads();
    }
}

// Ordered block prefix of each thread's four consecutive input samples,
// written to ring[position + 1 .. position + kBlockTile]: a warp shuffle scan
// and one scan of the warp totals. The ring is rebased after every tile, so
// float32 prefixes stay near the magnitude of one tile and the lookback.
template <typename T, int kBlockThreads>
DMT_D inline void prefix_tile(BoxcarShared<T, kBlockThreads>& shared,
                              T* ring,
                              uint32_t mask,
                              const T (&x)[kItems],
                              uint32_t position,
                              T base) {
    constexpr int kBlockWarps = kBlockThreads / kWarp;
    const int tid             = static_cast<int>(threadIdx.x);
    const int lane            = tid % kWarp;
    const int warp            = tid / kWarp;
    T total                   = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        total += x[j];
    }
    T inclusive = total;
#pragma unroll
    for (int offset = 1; offset < kWarp; offset *= 2) {
        const T left = shfl_up(inclusive, offset);
        if (lane >= offset) {
            inclusive += left;
        }
    }
    if (lane == kWarp - 1) {
        shared.warp_totals[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        T w = lane < kBlockWarps ? shared.warp_totals[lane] : T{0};
#pragma unroll
        for (int offset = 1; offset < kBlockWarps; offset *= 2) {
            const T left = shfl_up(w, offset);
            if (lane >= offset) {
                w += left;
            }
        }
        if (lane < kBlockWarps) {
            shared.warp_totals[lane] = w;
        }
    }
    __syncthreads();
    T running = base + (inclusive - total) +
                (warp > 0 ? shared.warp_totals[warp - 1] : T{0});
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
        running += x[j];
        ring[(position + tid * kItems + j + 1) & mask] = running;
    }
    __syncthreads();
}

// Scan trailing-run summaries to give each worker the state of every
// preceding sample. Each worker then emits only excursions ending within
// its own four samples. The last worker saves the open trailing excursion.
template <typename T, int kBlockThreads, typename Writer>
DMT_D inline void reduce_runs(BoxcarShared<T, kBlockThreads>& shared,
                              const bool (&above)[kItems],
                              const float (&snr)[kItems],
                              const uint32_t (&filters)[kItems],
                              uint64_t start,
                              int n,
                              BoxcarPeak* persistent,
                              const Writer& writer,
                              const uint64_t (&raw_sum)[kItems],
                              int integer) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int first = tid * kItems;
    BoxcarRunSummary summary;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (!above[j]) {
            summary.all           = false;
            summary.trailing.open = false;
        } else {
            const auto ignore = [] DMT_D(const BoxcarPeak&) {};
            if (integer == 2) {
                boxcar_integer_snr_peak_step(summary.trailing, true, snr[j],
                                             raw_sum[j], start + first + j,
                                             filters[j], ignore);
            } else if (integer == 1) {
                boxcar_integer_peak_step(summary.trailing, true, raw_sum[j],
                                         start + first + j, filters[j], ignore);
            } else {
                boxcar_peak_step(summary.trailing, true, snr[j],
                                 start + first + j, filters[j], ignore);
            }
        }
    }
    const BoxcarRunSummary carried{*persistent, true};
    shared.runs[tid] = summary;
    __syncthreads();
    for (int offset = 1; offset < kBlockThreads; offset *= 2) {
        const auto left =
            tid >= offset ? shared.runs[tid - offset] : BoxcarRunSummary{};
        __syncthreads();
        if (tid >= offset) {
            summary = boxcar_join(left, summary, integer);
        }
        shared.runs[tid] = summary;
        __syncthreads();
    }
    BoxcarPeak incoming =
        tid == 0 ? carried.trailing
                 : boxcar_join(carried, shared.runs[tid - 1], integer).trailing;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (integer == 2) {
            boxcar_integer_snr_peak_step(incoming, above[j], snr[j], raw_sum[j],
                                         start + first + j, filters[j], writer);
        } else if (integer == 1) {
            boxcar_integer_peak_step(incoming, above[j], raw_sum[j],
                                     start + first + j, filters[j], writer);
        } else {
            boxcar_peak_step(incoming, above[j], snr[j], start + first + j,
                             filters[j], writer);
        }
    }
    // All threads have read persistent and shared summaries before writing
    // the state for the next width/tile.
    __syncthreads();
    if (tid == kBlockThreads - 1) {
        *persistent = boxcar_join(carried, summary, integer).trailing;
    }
    __syncthreads();
}

// Per-call kill masks (float rows): the tables of the (at most two) ring
// entries a call's endpoints can reach, older first. Per entry: its first
// stream sample, steady [0] and transition [1] thresholds and inverse
// sigmas (rows x filters; null: the constructor tables) and whether it has
// a transition (its first support + W - 1 endpoints per row use [1]).
// count 0: the constructor tables only.
struct BlockTables {
    uint64_t first[2]{0, 0};
    const float* thr[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
    const float* inv[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
    /// Packed rows: the integer cutoffs (null: the constructor's).
    const uint64_t* ithr[2][2]{{nullptr, nullptr}, {nullptr, nullptr}};
    bool transition[2]{false, false};
    int count               = 0;
    const uint64_t* support = nullptr;
};

// The entry of stream sample `s` (the older one for samples before the
// newer's first).
DMT_D inline int block_entry(const BlockTables& bt, uint64_t s) {
    return bt.count == 2 && s >= bt.first[1] ? 1 : 0;
}

// Table (0 steady, 1 transition) of entry `e` for endpoint `s` and width w.
DMT_D inline int block_table(const BlockTables& bt,
                             int e,
                             uint64_t s,
                             uint32_t width,
                             uint64_t support) {
    return bt.transition[e] && s < bt.first[e] + support + width - 1 ? 1 : 0;
}

template <typename T> inline T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

template <typename T> inline const T* raw(const thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

class FDMTBoxcarCudaEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCudaEngine(const plans::FDMTPlan& plan,
                         const BoxcarConfig& config,
                         const std::vector<plans::BoxcarPlanePlan>& planes,
                         const FDMTEngineConfig& cfg,
                         std::unique_ptr<BoxcarDriver> driver,
                         std::vector<std::pair<SizeType, SizeType>> kill_runs);
    ~FDMTBoxcarCudaEngine() override;

    FDMTBoxcarCudaEngine(const FDMTBoxcarCudaEngine&)            = delete;
    FDMTBoxcarCudaEngine& operator=(const FDMTBoxcarCudaEngine&) = delete;
    FDMTBoxcarCudaEngine(FDMTBoxcarCudaEngine&&)                 = delete;
    FDMTBoxcarCudaEngine& operator=(FDMTBoxcarCudaEngine&&)      = delete;

    CandidateResultView execute(std::span<const float> waterfall) override;
    CandidateDeviceResultView execute(DeviceSpan<const float> waterfall,
                                      Stream queue) override;
    CandidateResultView execute(std::span<const uint8_t> waterfall) override;
    CandidateDeviceResultView execute(DeviceSpan<const uint8_t> waterfall,
                                      Stream queue) override;
    CandidateResultView flush() override;
    [[nodiscard]] bool zero_blocks() const noexcept override;
    CandidateResultView execute_zero() override;
    CandidateDeviceResultView flush_device(Stream queue) override;
    CandidateResultView copy_result(Stream queue) override;

    void set_search_stop(uint64_t stop) noexcept override;
    [[nodiscard]] uint64_t stream_position() const noexcept override;
    [[nodiscard]] uint64_t search_latency() const noexcept override;
    [[nodiscard]] uint64_t open_begin_min() const noexcept override;
    [[nodiscard]] SizeType block_noise_bytes() const noexcept override;
    void enable_block_noise(
        std::shared_ptr<const std::vector<uint64_t>> support) override;
    void push_block_noise(BlockNoise entry) override;
    void drop_block_noise() noexcept override;
    [[nodiscard]] uint64_t noise_catch_all_begin() const noexcept override;
    void reset_history(Stream queue) override;
    void consume(SizeType level, const FDMTSearchInput& input) override;
    void allocate() override;
    void allocate_host();
    [[nodiscard]] FDMTBoxcarMemoryUsage memory_usage() const noexcept override;
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override;
    [[nodiscard]] std::string search_path() const override;
    [[nodiscard]] SizeType fuse_levels() const noexcept override;

private:
    struct LevelGroup {
        SizeType level;
        SizeType row_begin;
        SizeType row_count;
    };

    struct TableSlot {
        thrust::device_vector<float> device; ///< thresholds, then inv sigma
        gpu_host::PinnedBuffer<float> host;
        uint64_t id          = 0; ///< NoiseTables::id held (0: none)
        cudaEvent_t copied   = nullptr;
        cudaEvent_t released = nullptr;
        bool copy_pending    = false;
        bool release_pending = false;
    };

    template <typename Kernel>
    void configure_kernel(Kernel kernel, SizeType smem = 0) const {
        gpu_utils::raise_dynamic_smem(kernel, smem != 0 ? smem : m_smem,
                                      "FDMTBoxcar: shared-memory opt-in");
#if !defined(DMT_ENABLE_HIP)
        // Several resident rows per SM need the shared-memory carveout.
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared),
            "FDMTBoxcar: shared-memory carveout");
#endif
    }

    // Prefix / kNone / wide-ring path (fdmt_boxcar_cuda_prefix.cu).
    void init_prefix_screen();
    template <typename T> void configure_shared();
    template <typename T, int kBlockThreads>
    void configure_shared_block(SizeType limit);
    template <typename T> void configure_global(SizeType limit);
    template <typename T,
              int kBlockThreads,
              bool kScreen,
              bool kBlockNoise,
              typename C,
              bool kGlobalRing = false>
    void configure_variants();
    template <typename T>
    [[nodiscard]] SizeType shared_need(int threads, bool screen) const;
    template <typename T, typename C = T>
    void launch_score(const FDMTSearchInput& input,
                      const LevelGroup& g,
                      dim3 grid,
                      dim3 block,
                      C* carry);
    template <typename T, bool kBlockNoise, typename C>
    void launch_score_global(const FDMTSearchInput& input,
                             const LevelGroup& g,
                             dim3 grid,
                             dim3 block,
                             C* carry);
    template <typename T, int kBlockThreads, typename C>
    void launch_score_block(const FDMTSearchInput& input,
                            const LevelGroup& g,
                            dim3 grid,
                            dim3 block,
                            C* carry);
    template <typename T,
              int kBlockThreads,
              bool kScreen,
              bool kBlockNoise,
              typename C>
    void launch_score_kernel(const FDMTSearchInput& input,
                             const LevelGroup& g,
                             dim3 grid,
                             dim3 block,
                             C* carry);
    template <typename T,
              int kBlockThreads,
              bool kScreen,
              bool kBlockNoise,
              typename C,
              bool kKeep,
              bool kGlobalRing = false>
    void launch_score_args(const FDMTSearchInput& stopped,
                           const LevelGroup& g,
                           dim3 grid,
                           dim3 block,
                           C* carry,
                           cudaStream_t stream,
                           SizeType task_base = 0);

    // Kadane Design C gate path (fdmt_boxcar_cuda_kadane.cu).
    void populate_row_int_m1(std::span<const double> means);
    void init_kadane(bool use_box_smearing,
                     std::span<const double> means,
                     std::span<const uint64_t> cutoffs);
    void reset_kadane(cudaStream_t stream);
    template <typename T, typename SegT>
    [[nodiscard]] SizeType kadane_shared_need(int threads) const;
    template <typename T, typename SegT>
    [[nodiscard]] bool try_configure_shared_kadane();
    template <typename T, typename SegT, int kBlockThreads>
    void configure_shared_kadane_block(SizeType limit);
    template <typename T,
              typename SegT,
              int kBlockThreads,
              bool kUseIntGate,
              bool kBlockNoise,
              typename C>
    void configure_kadane_variants();
    [[nodiscard]] kadane::KadaneGateDeviceView make_kadane_view() noexcept;
    template <typename T, typename C = T>
    void launch_score_kadane(const FDMTSearchInput& input,
                             const LevelGroup& g,
                             dim3 grid,
                             dim3 block,
                             C* carry);

    // Common helpers (fdmt_boxcar_cuda.cu).
    /// One block's search (`search()` runs the driver) between the block
    /// noise commit and the result bookkeeping, on `queue`.
    template <typename Search>
    CandidateDeviceResultView run_search(Stream queue, const Search& search);
    /// Killed rows read as zero by the transform itself (FDMT drivers, or
    /// a zero block): flags set for this block, true. False: the driver
    /// reads a staging copy with the rows zeroed (masked()).
    bool keep_rows(Stream queue);
    template <typename T>
    DeviceSpan<const T> masked(DeviceSpan<const T> block,
                               thrust::device_vector<T>& buffer,
                               Stream queue);
    /// `call_fill`: packed input under per-call masks; rows the runs kill
    /// but the constructor keeps read input_mean (flag 2).
    void set_keep(const std::vector<std::pair<SizeType, SizeType>>& runs,
                  Stream queue,
                  bool call_fill = false);
    void finish_keep(cudaStream_t stream);
    void shift_candidate_times(cudaStream_t stream);
    void update_open_begin(cudaStream_t stream);
    [[nodiscard]] SizeType ring_size() const noexcept;
    /// Entries the search latency needs: 1 + ceil(latency / nsamps); over
    /// ring_size() (2) the oldest kept entry takes the catch-all.
    [[nodiscard]] SizeType ring_needed() const noexcept;
    /// max_cands_per_dm for the writers: SIZE_MAX when 0 (no cap).
    [[nodiscard]] SizeType dm_cap() const noexcept {
        return m_config.max_cands_per_dm == 0 ? SIZE_MAX
                                              : m_config.max_cands_per_dm;
    }
    /// The call's counters per plane, from the rows' counters and the
    /// emitted candidates (after the call's candidates are final).
    void plane_counts(cudaStream_t stream);
    /// A table slot's device and pinned buffers and events.
    void allocate_slot(TableSlot& slot);
    [[nodiscard]] SizeType slot_count() const noexcept;
    [[nodiscard]] SizeType table_floats() const noexcept;
    /// Floats of one table set on the device: thresholds and inverse
    /// sigmas, plus the integer cutoffs (as pairs of floats) when packed.
    [[nodiscard]] SizeType slot_floats() const noexcept;
    /// Open packed per-width peaks take their S/N (the first per-call mask).
    void sync_peak_snr();
    void reset_block_noise();
    const float* slot_of(const NoiseTables* tables, cudaStream_t stream);
    void commit_block_noise(cudaStream_t stream);
    void undo_commit() noexcept;
    void release_slots(cudaStream_t stream);
    void validate_packed(SizeType size) const;
    void validate_input(SizeType size) const;
    void begin_result(SizeType samples, Stream queue);
    CandidateDeviceResultView device_view();

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    SizeType m_beams;
    int m_device;
    std::vector<BoxcarRow> m_rows;
    SizeType m_width;
    SizeType m_filters;
    SizeType m_peak_count;
    bool m_carry32 = false;
    std::unique_ptr<BoxcarDriver> m_driver;
    /// Killed channel runs: read as zero by the transform (m_keep_d) or
    /// zeroed in m_input_d / m_packed_input.
    std::vector<std::pair<SizeType, SizeType>> m_kill_runs;
    bool m_keep_in_transform = false;
    /// kKeepBrightest: max_cands_per_dm entries per (beam, row), the rows'
    /// brighter() ranks, the selection state and histogram (0: kDropNew).
    SizeType m_bright_q  = 0;
    bool m_bright_select = false;
    gpu_utils::PlannedVector<BoxcarKeepEntry> m_bright_d;
    gpu_utils::PlannedVector<uint32_t> m_bright_rank_d;
    gpu_utils::PlannedVector<KeepSelect> m_bright_sel_d;
    gpu_utils::PlannedVector<uint64_t> m_bright_hist_d;
    /// One byte per input row (beam, channel), for m_keep_runs.
    gpu_utils::PlannedVector<uint8_t> m_keep_d;
    std::vector<uint8_t> m_keep_h;
    std::vector<std::pair<SizeType, SizeType>> m_keep_runs;
    bool m_keep_fill = false; ///< m_keep_d holds flag 2 (call_fill)
    /// execute_zero(): the block's input reads as zero (every row killed).
    bool m_zero_block = false;
    std::vector<LevelGroup> m_level_groups;
    gpu_utils::PlannedVector<BoxcarRow> m_rows_d;
    gpu_utils::PlannedVector<uint32_t> m_widths_d;
    gpu_utils::PlannedVector<float> m_thresholds_d, m_scales_d, m_carry_d,
        m_input_d;
    gpu_utils::PlannedVector<uint64_t> m_integer_carry, m_integer_thresholds;
    gpu_utils::PlannedVector<uint32_t> m_integer_carry32;
    gpu_utils::PlannedVector<double> m_integer_means;
    gpu_utils::PlannedVector<uint8_t> m_packed_input;
    double m_integer_unit = 1;
    int m_block_threads   = 256;
    // FDMTEngineConfig::boxcar_shared_limit (0: the device's opt-in limit).
    SizeType m_shared_limit = 0;
    // Fallbacks taken to fit max_boxcar in shared memory (configure_shared).
    bool m_screen_dropped  = false;
    bool m_block_shrunk    = false;
    bool m_kadane_fallback = false;
    // kGlobalRing: rings of m_ring_batch blocks (uint64_t words, viewed as
    // T), reused by each batch of a launch.
    bool m_global_ring    = false;
    SizeType m_ring_batch = 0;
    gpu_utils::PlannedVector<uint64_t> m_ring_d;
    gpu_utils::PlannedVector<BoxcarScreenGroup> m_screen_groups_d;
    gpu_utils::PlannedVector<float> m_screen_thresholds_d;
    gpu_utils::PlannedVector<float> m_screen_m1_d; ///< packed: mean per sample
    SizeType m_screen_offset = 0;
    uint32_t m_mask          = 0;
    bool m_shared_tables     = false;
    SizeType m_smem          = 0;
    /// Shared bytes of the prefix kernel's per-call mask instances: m_smem
    /// plus the group bounds (float rows include them in m_smem as well).
    SizeType m_smem_noise = 0;
    gpu_utils::PlannedVector<BoxcarPeak> m_peaks_d;
    gpu_utils::PlannedVector<uint8_t> m_open_d; ///< any peak open, per task
    gpu_utils::PlannedVector<uint64_t> m_per_dm_d, m_slots_d;
    gpu_utils::PlannedVector<CandidateCounts> m_counts_d;
    gpu_utils::PlannedVector<RawCandidate> m_candidates_d;
    planned::PlannedHostVector<RawCandidate> m_candidates_h;
    CandidateCounts m_counts_h{};
    /// Per-plane counts: kKeepBrightest's detections per (beam, row), the
    /// first plane of each subband level (boxcar_level_offsets), and the
    /// call's counters per plane (device; host after copy_result()).
    gpu_utils::PlannedVector<uint64_t> m_row_detected_d;
    gpu_utils::PlannedVector<uint32_t> m_level_offset_d;
    gpu_utils::PlannedVector<CandidateCounts> m_plane_counts_d;
    std::vector<CandidateCounts> m_plane_counts_h;
    /// BoxcarConfig::max_candidates_per_plane (0: off): each plane's
    /// counter and, under kDropNew, the staging buffer of the shares.
    SizeType m_plane_cap = 0;
    gpu_utils::PlannedVector<uint64_t> m_plane_slots_d;
    gpu_utils::PlannedVector<RawCandidate> m_plane_scratch_d;
    /// Where the writers put candidates: the staging buffer of the planes'
    /// shares, or the call's candidates.
    [[nodiscard]] RawCandidate* candidate_out() {
        return m_plane_scratch_d.size() > 0 ? raw(m_plane_scratch_d)
                                            : raw(m_candidates_d);
    }
    gpu_utils::DeviceWorkFence m_work;
    uint64_t m_position       = 0;
    uint64_t m_search_stop    = UINT64_MAX; // drain(): end of stream
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
    SizeType m_group_offset   = 0;
    gpu_utils::PlannedVector<uint64_t> m_oldest_d; ///< candidate frontier
    gpu_host::PinnedBuffer<uint64_t> m_oldest_h;
    // Per-call kill masks: off until the first one.
    bool m_noise_on = false;
    std::vector<BlockNoise> m_ring; ///< oldest first, at most ring_size()
    std::optional<BlockNoise> m_pending;
    /// The entry the last commit evicted, kept until its call completes.
    std::optional<BlockNoise> m_evicted;
    std::optional<BlockNoise> m_prev_evicted;
    bool m_commit_open = false;
    std::vector<std::unique_ptr<TableSlot>> m_slots;
    gpu_utils::PlannedVector<uint64_t> m_support_d;
    BlockTables m_block_tables{};
    BlockTables m_prev_block_tables{};
    bool m_has_catch_all = false; ///< m_catch_all set (null: constructor)
    std::shared_ptr<const NoiseTables> m_catch_all;
    /// What the open commit replaced (restored by undo_commit()).
    struct CatchUndo {
        bool has_catch_all = false;
        std::shared_ptr<const NoiseTables> catch_all;
        bool front_set = false;
        std::shared_ptr<const NoiseTables> steady, transition_tables;
        bool transition = false;
    };
    CatchUndo m_catch_undo;

    // --- Kadane gate (`kKadane`) plan & device state ---
    kadane::KadaneGatePlan m_kadane_plan;
    bool m_kadane_int32 = false;
    bool m_any_edge_lag = false;
    gpu_utils::PlannedVector<float> m_kadane_bias_d;
    gpu_utils::PlannedVector<float> m_kadane_cut_d;
    gpu_utils::PlannedVector<int32_t> m_kadane_ibias_d;
    gpu_utils::PlannedVector<int32_t> m_kadane_icut_d;
    gpu_utils::PlannedVector<uint8_t> m_kadane_disabled_d;
    gpu_utils::PlannedVector<double> m_row_int_m1_d;
    gpu_utils::PlannedVector<float> m_kadane_fc_d;
    gpu_utils::PlannedVector<int32_t> m_kadane_ic_d;
    gpu_utils::PlannedVector<kadane::LagWindow<float>> m_lag_f32_d;
    gpu_utils::PlannedVector<kadane::LagWindow<int32_t>> m_lag_i32_d;
    /// Buffers a deferred construction left for allocate() (declared last:
    /// destroyed first, after nothing else refers to it).
    gpu_utils::PendingList m_deferred;
    bool m_allocated = false;
};

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
