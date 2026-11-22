#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_variance.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/logging.hpp"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: prefix-sum & sparse-table screen (`kPrefix`),
// unscreened (`kNone`), and wide-boxcar / small-SMEM `kGlobalRing` scorers.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

// Whether any sample of the tile crosses any width's threshold. Threads
// take samples in a striped order, so ring reads are conflict-free. The
// run bookkeeping runs only for tiles where this (or an open run) holds;
// otherwise every peak mode leaves its state unchanged.
template <typename T,
          int kBlockThreads,
          bool kFull,
          bool kCold,
          bool kTrapezoid>
DMT_D bool detect_tile(const T* ring,
                       uint32_t mask,
                       uint32_t position,
                       int n,
                       const uint32_t* widths,
                       const T* thresholds,
                       SizeType nfilters,
                       uint64_t start,
                       uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const uint32_t base = position + 1 + static_cast<uint32_t>(tid);
    T current[kItems];
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        current[k] = ring[(base + k * kBlockThreads) & mask];
    }
    bool hit = false;
    for (SizeType f = 0; f < nfilters; ++f) {
        const uint32_t width  = widths[f];
        const uint32_t w_edge = kTrapezoid ? trapezoid_edge_width(width) : 0U;
        const T threshold     = thresholds[f];
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = base + k * kBlockThreads;
            T sum              = current[k] - ring[(idx - width) & mask];
            if constexpr (kTrapezoid) {
                if (w_edge > 0) {
                    sum += ring[(idx - w_edge) & mask] -
                           ring[(idx - width + w_edge) & mask];
                }
            }
            bool h = sum >= threshold;
            if constexpr (kCold) {
                h = h && start + s >= warmup + width - 1;
            }
            hit = hit || h;
        }
    }
    return hit;
}

// detect_tile through the width screen (BoxcarScreenGroup, no trapezoid):
// `mn` (lookback + n + 1 floats of shared memory) receives the tile's
// prefixes in linear order and is raised in place to range-minimum level k
// as the groups need it (entry j holds min(P[j .. j + 2^k - 1])). Each
// update reads only entries at or above its own index, so updating in
// ascending batches (read, sync, write, sync) is safe.
//
// For packed rows (uint64 running sums, which only grow), with `m1` the
// row's mean per sample, the centred sums c(j) = (P(j) - P(0)) - m1 * j
// satisfy S_w - m1 * w = c(t) - c(t - w); a margin of 2^-20 of the tile's
// magnitude keeps the float screen conservative, and passing groups are
// checked width by width in exact integers from the ring.
template <typename T, int kBlockThreads, bool kFull, bool kCold>
DMT_D bool detect_tile_grouped(const T* ring,
                               uint32_t mask,
                               uint32_t position,
                               uint32_t lookback,
                               int n,
                               const uint32_t* widths,
                               const T* thresholds,
                               const BoxcarScreenGroup* groups,
                               int ngroups,
                               const float* group_thresholds,
                               float m1,
                               float* mn,
                               uint64_t start,
                               uint64_t warmup) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int len       = static_cast<int>(lookback) + n + 1;
    const uint32_t lin0 = position - lookback;
    float margin        = 0.0F;
    if constexpr (std::is_integral_v<T>) {
        const uint64_t p0 = ring[lin0 & mask];
        for (int j = tid; j < len; j += kBlockThreads) {
            const uint64_t e =
                ring[(lin0 + static_cast<uint32_t>(j)) & mask] - p0;
            mn[j] = __ull2float_rn(e) - (m1 * static_cast<float>(j));
        }
        const float extent =
            __ull2float_ru(
                ring[(lin0 + static_cast<uint32_t>(len - 1)) & mask] - p0) +
            (fabsf(m1) * static_cast<float>(len));
        margin = (extent * 0x1p-20F) + 1.0F;
    } else {
        for (int j = tid; j < len; j += kBlockThreads) {
            mn[j] = ring[(lin0 + static_cast<uint32_t>(j)) & mask];
        }
    }
    __syncthreads();
    struct RawTile {
        uint64_t v[kItems];
    };
    float current[kItems];
    std::conditional_t<std::is_integral_v<T>, RawTile, NoKeep> current_raw{};
#pragma unroll
    for (int k = 0; k < kItems; ++k) {
        const int j = static_cast<int>(lookback) + 1 + k * kBlockThreads + tid;
        current[k]  = j < len ? mn[j] : 0.0F;
        if constexpr (std::is_integral_v<T>) {
            current_raw.v[k] =
                ring[(position + 1U +
                      static_cast<uint32_t>(k * kBlockThreads + tid)) &
                     mask];
        }
    }
    const auto exact = [&](uint32_t f) {
        const uint32_t width = widths[f];
        const T th           = thresholds[f];
        bool hit             = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const uint32_t idx = position + 1U + static_cast<uint32_t>(s);
            const T cur = [&] {
                if constexpr (std::is_integral_v<T>) {
                    return current_raw.v[k];
                } else {
                    return current[k];
                }
            }();
            bool h = cur - ring[(idx - width) & mask] >= th;
            if constexpr (kCold) {
                h = h && start + s >= warmup + width - 1;
            }
            hit = hit || h;
        }
        return hit;
    };
    bool hit  = false;
    int level = 0;
    for (int g = 0; g < ngroups; ++g) {
        const BoxcarScreenGroup grp = groups[g];
        if (static_cast<int>(grp.level) < level) {
            // Sparse gap: one width, checked directly from the ring.
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
            continue;
        }
        while (level < static_cast<int>(grp.level)) {
            const int half  = 1 << level;
            const int valid = len - (2 * half) + 1;
            for (int b0 = 0; b0 < valid; b0 += kItems * kBlockThreads) {
                float v[kItems];
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        v[r] = fminf(mn[j], mn[j + half]);
                    }
                }
                __syncthreads();
#pragma unroll
                for (int r = 0; r < kItems; ++r) {
                    const int j = b0 + (r * kBlockThreads) + tid;
                    if (j < valid) {
                        mn[j] = v[r];
                    }
                }
                __syncthreads();
            }
            ++level;
        }
        // Bound of the whole group; a pass is checked width by width from
        // the ring (mn holds range minima now).
        const int span = 1 << grp.level;
        const float bound_thr =
            std::is_integral_v<T> ? (group_thresholds[g] - margin)
                                  : group_thresholds[g];
        bool pass = false;
#pragma unroll
        for (int k = 0; k < kItems; ++k) {
            const int s = k * kBlockThreads + tid;
            if (!kFull && s >= n) {
                break;
            }
            const int j = static_cast<int>(lookback) + 1 + s;
            const float m =
                fminf(mn[j - static_cast<int>(grp.w_hi)],
                      mn[j - static_cast<int>(grp.w_lo) - span + 1]);
            pass = pass || current[k] - m >= bound_thr;
        }
        if (pass) {
            for (uint32_t f = grp.f_lo; f < grp.f_hi; ++f) {
                hit = exact(f) || hit;
            }
        }
    }
    __syncthreads(); // mn is reused by the next tile
    return hit;
}

// kScreen: tiles are screened by width group (detect_tile_grouped*); a
// separate instantiation, so the plain kernel keeps its register budget.
// kBlockNoise: per-call mask tables are in use (packed rows: their integer
// cutoffs, S/N per sample and S/N-ranked runs, peaks carry their S/N). A
// separate instance, so the common unmasked launch carries none of their
// state (it cost ~160 B of stack per thread and 15-50% of the run time).
// kKeep: OverflowPolicy::kKeepBrightest (KeepWriter, per-row stores); a
// separate instance, so kDropNew keeps its code and output.
// kGlobalRing: the prefix ring lives in device memory (`ring_global`, one
// ring of mask + 1 entries per block, blocks launched in batches from task
// `task_base`) instead of shared memory: boxcars wider than the fast path
// (kFastMaxWidth) or than the device's shared memory. No width screen and
// no shared tables; same candidates, no speed promise.
template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise = false,
          typename C       = T,
          bool kKeep       = false,
          bool kGlobalRing = false>
__global__ void __launch_bounds__(kBlockThreads, 4)
    kernel_fdmt_boxcar(FDMTSearchInput input,
                       const BoxcarRow* rows,
                       SizeType row_base,
                       SizeType ndm,
                       SizeType total_rows,
                       const uint32_t* widths,
                       SizeType nfilters,
                       SizeType max_width,
                       const float* thresholds,
                       const float* inv_sigma,
                       C* carry_all,
                       BoxcarPeak* peaks_all,
                       uint8_t* open_all,
                       SizeType peak_count,
                       RawCandidate* candidates,
                       CandidateCounts* counts,
                       uint64_t* slots,
                       uint64_t* per_dm_all,
                       SizeType capacity,
                       SizeType dm_cap,
                       PeakFindingMode mode,
                       bool skip_cold,
                       bool root,
                       const uint64_t* integer_thresholds,
                       const double* means,
                       double unit,
                       uint32_t mask,
                       bool shared_tables,
                       bool use_trapezoid,
                       const BoxcarScreenGroup* screen_groups,
                       int screen_ngroups,
                       const float* screen_thresholds,
                       const float* screen_m1,
                       SizeType screen_offset,
                       BlockTables block_tables,
                       SizeType group_offset,
                       BoxcarKeepEntry* keep_all,
                       SizeType keep_q,
                       uint64_t* detected_all,
                       uint64_t* plane_slots,
                       SizeType plane_cap,
                       T* ring_global,
                       SizeType task_base,
                       bool bypass_screen) {
    constexpr int kBlockTile = kBlockThreads * kItems;
    const SizeType task =
        kGlobalRing ? task_base + blockIdx.x : SizeType{blockIdx.x};
    if (!setup_row_input(input, rows, row_base, ndm, task)) {
        return;
    }
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<BoxcarShared<T, kBlockThreads>*>(storage);
    T* ring = kGlobalRing
                  ? ring_global + (SizeType{blockIdx.x} * (SizeType{mask} + 1))
                  : reinterpret_cast<T*>(storage + sizeof(shared));
    const SizeType beam      = input.beam_begin + (task / ndm);
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = (beam * total_rows) + row_index;
    const BoxcarRow row      = rows[row_index];
    C* carry                 = carry_all + (index * max_width);
    BoxcarPeak* peaks        = peaks_all + (index * peak_count);
    std::conditional_t<kKeep, KeepPending, NoKeep> pending{};
    const auto out = make_row_writer<T, kBlockNoise, kKeep>(
        row, beam, row_index, index, nfilters, widths, inv_sigma, means, unit,
        mode, candidates, counts, slots, per_dm_all, capacity, dm_cap, keep_all,
        keep_q, detected_all, plane_slots, plane_cap, pending);
    const int tid       = static_cast<int>(threadIdx.x);
    const auto lookback = static_cast<uint32_t>(max_width - 1);
    const auto noise    = resolve_row_noise<T, kBlockNoise>(
        input, row_index, nfilters, max_width, widths, thresholds, inv_sigma,
        integer_thresholds, block_tables);
    const uint32_t* wtab = widths;
    const T* ttab        = noise.thr_row();
    if (shared_tables) {
        T* s_thr       = ring + mask + 1;
        auto* s_widths = reinterpret_cast<uint32_t*>(s_thr + nfilters);
        stage_row_tables<T, kBlockThreads, kBlockNoise>(s_thr, s_widths, widths,
                                                        nfilters, noise);
        wtab = s_widths;
        ttab = s_thr;
    }
    // Mixed tables without the shared row, or kNone: every tile takes the
    // exact bookkeeping (no screen bound is stored / evaluated).
    bool force_bookkeeping =
        bypass_screen || (noise.is_mixed() && !shared_tables);
    // Group screen bounds of tables other than the constructor's: the
    // per-group minimum of the row's (staged) thresholds.
    const float* group_thr =
        screen_ngroups > 0 ? screen_thresholds + row_index * screen_ngroups
                           : nullptr;
    if constexpr (kScreen && kBlockNoise) {
        if (noise.custom && group_offset == 0) {
            force_bookkeeping = true; // no room for the group bounds
        } else if (noise.custom) {
            auto* s_grp = reinterpret_cast<float*>(storage + group_offset);
            __syncthreads();
            for (int g = tid; g < screen_ngroups; g += kBlockThreads) {
                if constexpr (std::is_integral_v<T>) {
                    // Packed: cutoff - mean, rounded down to float (as
                    // init_prefix_screen()); the mean is the constructor's.
                    double th = 1.0e300;
                    for (uint32_t f = screen_groups[g].f_lo;
                         f < screen_groups[g].f_hi; ++f) {
                        const double c = static_cast<double>(ttab[f]) -
                                         means[row_index * nfilters + f];
                        th             = th < c ? th : c;
                    }
                    s_grp[g] = __double2float_rd(th);
                } else {
                    float th = 3.402823466e38F;
                    for (uint32_t f = screen_groups[g].f_lo;
                         f < screen_groups[g].f_hi; ++f) {
                        th = th < ttab[f] ? th : ttab[f];
                    }
                    s_grp[g] = th;
                }
            }
            group_thr = s_grp;
        }
    }
    // Whether a run is open, as any_open_peaks() would find: kept per row by
    // the last call, so the common call never reads the row's peaks.
    bool open = input.sample_begin > 0 && open_all[index] != 0;
    bootstrap_carry_ring<T, C, kBlockThreads>(shared, ring, mask, carry,
                                              max_width, lookback,
                                              input.sample_begin);
    for (SizeType t0 = 0; t0 < input.nsamps; t0 += kBlockTile) {
        const int n = static_cast<int>(
            input.nsamps - t0 < kBlockTile ? input.nsamps - t0 : kBlockTile);
        const auto position = static_cast<uint32_t>(lookback + t0);
        const T base        = ring[position & mask];
        T x[kItems];
        load_tile_samples<T, kBlockThreads>(input, row, beam, t0, n, root, x);
        prefix_tile<T, kBlockThreads>(shared, ring, mask, x, position, base);
        const uint64_t start = input.sample_begin + t0;
        // Tiles past the warm-up of every width skip the per-width check.
        const bool cold       = skip_cold && start + 1 < row.warmup + max_width;
        const auto run_detect = [&](auto full_tag, auto cold_tag) {
            constexpr bool kFull = decltype(full_tag)::value;
            constexpr bool kCold = decltype(cold_tag)::value;
            if constexpr (kScreen) {
                return detect_tile_grouped<T, kBlockThreads, kFull, kCold>(
                    ring, mask, position, lookback, n, wtab, ttab,
                    screen_groups, screen_ngroups, group_thr,
                    std::is_integral_v<T> ? screen_m1[row_index] : 0.0F,
                    reinterpret_cast<float*>(storage + screen_offset), start,
                    row.warmup);
            }
            return use_trapezoid
                       ? detect_tile<T, kBlockThreads, kFull, kCold, true>(
                             ring, mask, position, n, wtab, ttab, nfilters,
                             start, row.warmup)
                       : detect_tile<T, kBlockThreads, kFull, kCold, false>(
                             ring, mask, position, n, wtab, ttab, nfilters,
                             start, row.warmup);
        };
        bool hit;
        if (force_bookkeeping) {
            hit = true;
        } else if (n == kBlockTile) {
            hit = cold ? run_detect(cuda::std::true_type{},
                                    cuda::std::true_type{})
                       : run_detect(cuda::std::true_type{},
                                    cuda::std::false_type{});
        } else {
            hit = cold ? run_detect(cuda::std::false_type{},
                                    cuda::std::true_type{})
                       : run_detect(cuda::std::false_type{},
                                    cuda::std::false_type{});
        }
        if (__syncthreads_or(hit || open) != 0) {
            // Rare path: exact run bookkeeping in each thread's four
            // consecutive samples.
            score_tile_bookkeeping<T, kBlockThreads, kBlockNoise, kKeep>(
                shared, ring, mask, position, n, start, row, row_index, wtab,
                ttab, nfilters, noise, block_tables, means, unit, mode,
                skip_cold, use_trapezoid, peaks, out, pending);
            open = any_open_peaks<kBlockThreads>(peaks, peak_count, mode);
        }
        // Recenter the retained prefix boundaries only when another tile
        // follows.
        if (t0 + kBlockTile < input.nsamps) {
            recenter_prefix_ring<T, kBlockThreads>(
                ring, mask, position + static_cast<uint32_t>(n), lookback);
        }
    }
    if (tid == 0) {
        open_all[index] = open ? 1 : 0;
    }
    // The root's old FDMT history remains available until this kernel ends.
    save_row_carry<T, C, kBlockThreads>(input, row, beam, root, max_width,
                                        carry);
}

} // namespace

void FDMTBoxcarCudaEngine::init_prefix_screen() {
    if (m_config.use_trapezoid) {
        return;
    }
    const auto groups = boxcar_screen_groups(m_planes.front().widths, false);
    if (groups.empty()) {
        return;
    }
    // Per row, each group's smallest threshold (float rows) or smallest
    // cutoff - mean, rounded down (packed rows), and the packed rows' mean
    // per sample (from the width-1 mean).
    const bool packed = m_config.input_nbits != 32;
    std::vector<float> group_thr;
    std::vector<float> m1;
    group_thr.reserve(m_rows.size() * groups.size());
    for (const auto& p : m_planes) {
        for (SizeType d = 0; d < p.dm_grid.size(); ++d) {
            const SizeType base = d * m_filters;
            for (const auto& g : groups) {
                double th = std::numeric_limits<double>::infinity();
                for (uint32_t f = g.f_lo; f < g.f_hi; ++f) {
                    th = std::min(th,
                                  packed ? static_cast<double>(
                                               p.integer_thresh_lut[base + f]) -
                                               p.integer_mean_lut[base + f]
                                         : static_cast<double>(
                                               p.raw_thresh_lut[base + f]));
                }
                auto thf = static_cast<float>(th);
                if (static_cast<double>(thf) > th) {
                    thf = std::nextafter(
                        thf, -std::numeric_limits<float>::infinity());
                }
                group_thr.push_back(thf);
            }
            if (packed) {
                // The mean per sample comes from widths[0], a plain boxcar
                // (no screen with use_trapezoid).
                m1.push_back(
                    static_cast<float>(p.integer_mean_lut[base] /
                                       static_cast<double>(p.widths[0])));
            }
        }
    }
    m_screen_groups_d     = groups;
    m_screen_thresholds_d = group_thr;
    m_screen_m1_d         = m1;
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C,
          bool kGlobalRing>
void FDMTBoxcarCudaEngine::configure_variants() {
    const SizeType smem = kBlockNoise ? m_smem_noise : m_smem;
    configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, kScreen, kBlockNoise,
                                        C, false, kGlobalRing>,
                     smem);
    if (m_bright_q != 0) {
        configure_kernel(kernel_fdmt_boxcar<T, kBlockThreads, kScreen,
                                            kBlockNoise, C, true, kGlobalRing>,
                         smem);
    }
}

template <typename T, int kBlockThreads>
void FDMTBoxcarCudaEngine::configure_shared_block(SizeType limit) {
    configure_shared_layout<T, kBlockThreads>(
        sizeof(BoxcarShared<T, kBlockThreads>), limit,
        !m_screen_groups_d.empty(), "FDMTBoxcar");
    if (m_screen_groups_d.empty()) {
        configure_variants<T, kBlockThreads, false, false, T>();
        configure_variants<T, kBlockThreads, false, true, T>();
    } else {
        configure_variants<T, kBlockThreads, true, false, T>();
        configure_variants<T, kBlockThreads, true, true, T>();
    }
    if constexpr (std::is_integral_v<T>) {
        if (m_screen_groups_d.empty()) {
            configure_variants<T, kBlockThreads, false, false, uint32_t>();
            configure_variants<T, kBlockThreads, false, true, uint32_t>();
        } else {
            configure_variants<T, kBlockThreads, true, false, uint32_t>();
            configure_variants<T, kBlockThreads, true, true, uint32_t>();
        }
    }
}

template <typename T>
void FDMTBoxcarCudaEngine::configure_global(SizeType limit) {
    if (!m_screen_groups_d.empty()) {
        m_screen_groups_d.clear();
        m_screen_thresholds_d.clear();
        m_screen_m1_d.clear();
        m_screen_groups_d.shrink_to_fit();
        m_screen_thresholds_d.shrink_to_fit();
        m_screen_m1_d.shrink_to_fit();
        m_screen_dropped = true;
    }
    const SizeType ring =
        std::bit_ceil(m_width + SizeType{kGlobalBlock * kItems});
    m_global_ring   = true;
    m_block_threads = kGlobalBlock;
    m_mask          = static_cast<uint32_t>(ring - 1);
    m_shared_tables = false;
    m_screen_offset = 0;
    m_group_offset  = 0;
    m_smem          = sizeof(BoxcarShared<T, kGlobalBlock>);
    m_smem_noise    = m_smem;
    if (m_smem > limit) {
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar: the search needs {} B of shared memory per "
            "block; the device allows {} B",
            m_smem, limit));
    }
    SizeType tasks = 1;
    for (const auto& g : m_level_groups) {
        tasks = std::max(tasks, g.row_count * m_beams);
    }
    const SizeType ring_words = ring * sizeof(T) / sizeof(uint64_t);
    m_ring_batch              = std::clamp<SizeType>(
        kGlobalRingBytes / (ring_words * sizeof(uint64_t)), 1, tasks);
    m_ring_d.resize(m_ring_batch * ring_words);
    logging::debug("FDMTBoxcar: max_boxcar {}: prefix rings in device "
                   "memory ({} blocks per batch, {} B)",
                   m_width, m_ring_batch, m_ring_d.size() * sizeof(uint64_t));
    configure_variants<T, kGlobalBlock, false, false, T, true>();
    configure_variants<T, kGlobalBlock, false, true, T, true>();
    if constexpr (std::is_integral_v<T>) {
        configure_variants<T, kGlobalBlock, false, false, uint32_t, true>();
        configure_variants<T, kGlobalBlock, false, true, uint32_t, true>();
    }
}

template <typename T> void FDMTBoxcarCudaEngine::configure_shared() {
    const SizeType limit = shared_mem_limit();
    if (m_width > kFastMaxWidth) {
        configure_global<T>(limit);
        return;
    }
    const int threads = select_prefix_block_threads<T>(limit, m_block_shrunk);
    // Wide boxes on small-shared-memory GPUs: when the block's prefix
    // ring does not fit, drop the width screen (its second ring; the
    // screen only skips work, candidates are the same), then take
    // smaller blocks (shorter tile and ring, smaller run state).
    // Configurations that fit are unchanged.
    if (!m_screen_groups_d.empty() &&
        (m_block_shrunk || shared_need<T>(threads, true) > limit)) {
        m_screen_groups_d.clear();
        m_screen_thresholds_d.clear();
        m_screen_m1_d.clear();
        m_screen_groups_d.shrink_to_fit();
        m_screen_thresholds_d.shrink_to_fit();
        m_screen_m1_d.shrink_to_fit();
        m_screen_dropped = true;
    }
    const bool screen = !m_screen_groups_d.empty();
    if (shared_need<T>(threads, screen) > limit) {
        m_block_shrunk = false;
        configure_global<T>(limit);
        return;
    }
    if (m_screen_dropped || m_block_shrunk) {
        logging::debug("FDMTBoxcar: max_boxcar {} in {} B of shared "
                       "memory: {}-thread blocks{}",
                       m_width, limit, threads,
                       m_screen_dropped ? ", no width screen" : "");
    }
    if (threads == 64) {
        configure_shared_block<T, 64>(limit);
    } else if (threads == 128) {
        configure_shared_block<T, 128>(limit);
    } else {
        configure_shared_block<T, 256>(limit);
    }
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C,
          bool kKeep,
          bool kGlobalRing>
void FDMTBoxcarCudaEngine::launch_score_args(const FDMTSearchInput& stopped,
                                             const LevelGroup& g,
                                             dim3 grid,
                                             dim3 block,
                                             C* carry,
                                             cudaStream_t stream,
                                             SizeType task_base) {
    kernel_fdmt_boxcar<T, kBlockThreads, kScreen, kBlockNoise, C, kKeep,
                       kGlobalRing>
        <<<grid, block, kBlockNoise ? m_smem_noise : m_smem, stream>>>(
            stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
            raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
            raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d), m_peak_count,
            candidate_out(), raw(m_counts_d), raw(m_slots_d), raw(m_per_dm_d),
            m_config.max_candidates, dm_cap(), m_config.peak_mode,
            m_config.skip_cold_prefix, g.level == m_plan.get_niters(),
            raw(m_integer_thresholds), raw(m_integer_means), m_integer_unit,
            m_mask, m_shared_tables, m_config.use_trapezoid,
            raw(m_screen_groups_d), static_cast<int>(m_screen_groups_d.size()),
            raw(m_screen_thresholds_d), raw(m_screen_m1_d), m_screen_offset,
            kBlockNoise ? m_block_tables : BlockTables{}, m_group_offset,
            raw(m_bright_d), m_bright_q, raw(m_row_detected_d),
            raw(m_plane_slots_d), m_plane_cap,
            kGlobalRing ? reinterpret_cast<T*>(raw(m_ring_d)) : nullptr,
            task_base, m_config.screen_mode == BoxcarScreenMode::kNone);
}

template <typename T,
          int kBlockThreads,
          bool kScreen,
          bool kBlockNoise,
          typename C>
void FDMTBoxcarCudaEngine::launch_score_kernel(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    if (m_bright_q != 0) {
        launch_score_args<T, kBlockThreads, kScreen, kBlockNoise, C, true>(
            stopped, g, grid, block, carry, stream);
    } else {
        launch_score_args<T, kBlockThreads, kScreen, kBlockNoise, C, false>(
            stopped, g, grid, block, carry, stream);
    }
}

template <typename T, int kBlockThreads, typename C>
void FDMTBoxcarCudaEngine::launch_score_block(const FDMTSearchInput& input,
                                              const LevelGroup& g,
                                              dim3 grid,
                                              dim3 block,
                                              C* carry) {
    // Per-call mask tables take their own kernel instance.
    if (m_block_tables.count > 0) {
        if (m_screen_groups_d.empty()) {
            launch_score_kernel<T, kBlockThreads, false, true, C>(
                input, g, grid, block, carry);
        } else {
            launch_score_kernel<T, kBlockThreads, true, true, C>(input, g, grid,
                                                                 block, carry);
        }
        return;
    }
    if (m_screen_groups_d.empty()) {
        launch_score_kernel<T, kBlockThreads, false, false, C>(input, g, grid,
                                                               block, carry);
    } else {
        launch_score_kernel<T, kBlockThreads, true, false, C>(input, g, grid,
                                                              block, carry);
    }
}

template <typename T, bool kBlockNoise, typename C>
void FDMTBoxcarCudaEngine::launch_score_global(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    const SizeType tasks    = grid.x;
    for (SizeType base = 0; base < tasks; base += m_ring_batch) {
        const dim3 batch(
            static_cast<unsigned>(std::min(m_ring_batch, tasks - base)));
        if (m_bright_q != 0) {
            launch_score_args<T, kGlobalBlock, false, kBlockNoise, C, true,
                              true>(stopped, g, batch, block, carry, stream,
                                    base);
        } else {
            launch_score_args<T, kGlobalBlock, false, kBlockNoise, C, false,
                              true>(stopped, g, batch, block, carry, stream,
                                    base);
        }
    }
}

template <typename T, typename C>
void FDMTBoxcarCudaEngine::launch_score(const FDMTSearchInput& input,
                                        const LevelGroup& g,
                                        dim3 grid,
                                        dim3 block,
                                        C* carry) {
    if (m_global_ring) {
        if (m_block_tables.count > 0) {
            launch_score_global<T, true, C>(input, g, grid, block, carry);
            return;
        }
        launch_score_global<T, false, C>(input, g, grid, block, carry);
        return;
    }
    if (m_block_threads == 64) {
        launch_score_block<T, 64, C>(input, g, grid, block, carry);
    } else if (m_block_threads == 128) {
        launch_score_block<T, 128, C>(input, g, grid, block, carry);
    } else {
        launch_score_block<T, 256, C>(input, g, grid, block, carry);
    }
}

template void FDMTBoxcarCudaEngine::configure_shared<float>();
template void FDMTBoxcarCudaEngine::configure_shared<uint64_t>();
template void FDMTBoxcarCudaEngine::launch_score<float, float>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, float*);
template void FDMTBoxcarCudaEngine::launch_score<uint64_t, uint64_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint64_t*);
template void FDMTBoxcarCudaEngine::launch_score<uint64_t, uint32_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint32_t*);

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
