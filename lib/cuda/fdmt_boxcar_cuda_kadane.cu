#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/kadane_cuda.cuh"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/logging.hpp"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: Design C shared-memory-lean tropical Kadane monoid
// gate (`BoxcarScreenMode::kKadane`) with lazy carry & prefix-ring bootstrap,
// kKeepBrightest support, and wide-boxcar / small-SMEM fallback.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

// kKadane block size for integer (exact-sum) inputs. A row whose tile
// triggers runs Stage 2 alone in its block, and with one block per row a few
// such blocks set the length of the whole launch: small blocks keep that
// latency short and pack more blocks per SM (L40, uint8 input: 2-5% faster
// than 128 threads).
//
// Float inputs keep the kPrefix block-size policy instead: the Stage 2
// prefix sum is associated per block tile (threads * kItems), so only the
// same tile as the kPrefix kernel gives bit-identical S/N.
inline constexpr int kKadaneThreads = 64;

template <typename T, typename SegT, int kBlockThreads>
struct alignas(16) KadaneBlockShared {
    using LagV = std::conditional_t<std::is_same_v<SegT, kadane::KadaneSegI64>,
                                    int32_t,
                                    float>;
    SegT warp_segs[kadane::kMaxGateBuckets][kBlockThreads / kWarp];
    float row_cuts[kadane::kMaxGateBuckets];
    // Edge-lag windows of the leader thread: one per block in shared memory
    // (per-thread arrays were zero-filled in local memory by every thread of
    // every block, ~0.5 KB each, costing more than the whole gate).
    kadane::LagWindow<LagV> lag[kadane::kMaxGateBuckets];
    BoxcarShared<T, kBlockThreads> boxcar;
};

template <typename T,
          typename SegT,
          int kBlockThreads,
          bool kUseIntGate,
          bool kBlockNoise = false,
          typename C       = T,
          bool kKeep       = false>
__global__ void __launch_bounds__(kBlockThreads, 4)
    kernel_fdmt_boxcar_kadane(FDMTSearchInput input,
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
                              kadane::KadaneGateDeviceView gate,
                              BlockTables block_tables,
                              BoxcarKeepEntry* keep_all,
                              SizeType keep_q,
                              uint64_t* detected_all,
                              uint64_t* plane_slots,
                              SizeType plane_cap) {
    constexpr int kBlockTile  = kBlockThreads * kItems;
    constexpr int kBlockWarps = kBlockThreads / kWarp;
    constexpr int kLeaderTid  = kBlockWarps - 1;
    const SizeType task       = SizeType{blockIdx.x};

    if (!setup_row_input(input, rows, row_base, ndm, task) ||
        input.nsamps == 0) {
        return;
    }

    const SizeType beam      = input.beam_begin + (task / ndm);
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = (beam * total_rows) + row_index;
    const int tid            = static_cast<int>(threadIdx.x);
    const int lane           = tid % kWarp;
    const int warp           = tid / kWarp;

    bool open = input.sample_begin > 0 && open_all[index] != 0;
    if (gate.disabled_row[row_index] != 0U && !open) {
        if (tid == 0) {
            open_all[index] = 0;
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char storage[];
    auto& kshared =
        *reinterpret_cast<KadaneBlockShared<T, SegT, kBlockThreads>*>(storage);
    auto& shared = kshared.boxcar;
    T* ring      = reinterpret_cast<T*>(storage + sizeof(kshared));

    const BoxcarRow row = rows[row_index];
    C* carry            = carry_all + (index * max_width);
    BoxcarPeak* peaks   = peaks_all + (index * peak_count);
    std::conditional_t<kKeep, KeepPending, NoKeep> pending{};
    const auto out = make_row_writer<T, kBlockNoise, kKeep>(
        row, beam, row_index, index, nfilters, widths, inv_sigma, means, unit,
        mode, candidates, counts, slots, per_dm_all, capacity, dm_cap, keep_all,
        keep_q, detected_all, plane_slots, plane_cap, pending);

    const auto lookback = static_cast<uint32_t>(max_width - 1);
    RowNoisePlay<kBlockNoise> play{};
    const auto noise = resolve_row_noise<T, kBlockNoise>(
        input, row_index, nfilters, max_width, widths, thresholds, inv_sigma,
        integer_thresholds, block_tables, play);

    T* s_thr             = ring + mask + 1;
    auto* s_widths       = reinterpret_cast<uint32_t*>(s_thr + nfilters);
    const uint32_t* wtab = shared_tables ? s_widths : widths;
    const T* ttab        = shared_tables ? s_thr : noise.thr_row();
    bool tables_staged   = false;

    const SizeType nb = gate.num_buckets;
    float row_biases_f32[kadane::kMaxGateBuckets]{};
    float row_cuts_f32[kadane::kMaxGateBuckets]{};
    int32_t row_ibias[kadane::kMaxGateBuckets]{};
    int32_t row_icut[kadane::kMaxGateBuckets]{};

    // Smallest cutoff of the tables in play at filter f: integer (packed
    // rows) and the float one the gate compares (packed rows run the float
    // gate on centred sums: at most the centred integer cutoff, as
    // init_kadane()).
    [[maybe_unused]] const auto min_ithr_at = [&](SizeType f) -> uint64_t {
        return noise.min_ithr_at(f, play);
    };
    [[maybe_unused]] const auto min_thr_at = [&](SizeType f) -> float {
        float th = noise.min_fthr_at(f, play);
        if constexpr (std::is_integral_v<T>) {
            const uint64_t it = noise.min_ithr_at(f, play);
            if (it != std::numeric_limits<uint64_t>::max()) {
                const double centred = (static_cast<double>(it) -
                                        means[(row_index * nfilters) + f]) *
                                       unit;
                if (centred > 0.0) {
                    const auto c = static_cast<float>(centred);
                    th           = c < th ? c : th;
                }
            }
        }
        return th;
    };

    if constexpr (kUseIntGate) {
        for (SizeType k = 0; k < nb; ++k) {
            row_ibias[k] = gate.ibias[(k * gate.row_stride) + row_index];
            row_icut[k]  = gate.icut[(k * gate.row_stride) + row_index];
        }
        if constexpr (kBlockNoise) {
            // Per-call tables: the row's cuts at its fixed bias, as
            // KadaneGatePlan::recompute_row_icuts (the minimum over the
            // tables in play is the cut of their per-width minimum).
            if (noise.custom) {
                if (static_cast<SizeType>(tid) < nb) {
                    const auto k = static_cast<SizeType>(tid);
                    const int32_t c =
                        gate.disabled_row[row_index] != 0U
                            ? std::numeric_limits<int32_t>::max()
                            : kadane::compute_bucket_icut_hd(
                                  gate.buckets[k], widths, use_trapezoid,
                                  gate.int_shift, row_ibias[k], min_ithr_at);
                    kshared.row_cuts[k] = __int_as_float(c);
                }
                __syncthreads();
                for (SizeType k = 0; k < nb; ++k) {
                    row_icut[k] = __float_as_int(kshared.row_cuts[k]);
                }
                __syncthreads();
            }
        }
    } else {
        for (SizeType k = 0; k < nb; ++k) {
            row_biases_f32[k] = gate.bias[(k * gate.row_stride) + row_index];
            row_cuts_f32[k]   = gate.cut[(k * gate.row_stride) + row_index];
        }
        if constexpr (kBlockNoise) {
            if (noise.custom) {
                if (static_cast<SizeType>(tid) < nb) {
                    const auto k = static_cast<SizeType>(tid);
                    if (gate.disabled_row[row_index] != 0U) {
                        kshared.row_cuts[k] = std::numeric_limits<float>::max();
                    } else {
                        kshared.row_cuts[k] = kadane::compute_bucket_cut_hd_f32(
                            gate.buckets[k], widths, nfilters, use_trapezoid,
                            row_biases_f32[k], min_thr_at);
                    }
                }
                __syncthreads();
                for (SizeType k = 0; k < nb; ++k) {
                    row_cuts_f32[k] = kshared.row_cuts[k];
                }
                __syncthreads();
            }
        }
    }

    const double row_m1 =
        (std::is_integral_v<T> && !kUseIntGate && gate.row_int_m1 != nullptr)
            ? gate.row_int_m1[row_index]
            : 0.0;

    float c_state_f32[kadane::kMaxGateBuckets]{};
    int32_t c_state_i32[kadane::kMaxGateBuckets]{};
    using LagV = typename KadaneBlockShared<T, SegT, kBlockThreads>::LagV;
    static_assert(std::is_same_v<LagV, int32_t> == kUseIntGate,
                  "int32 gate pairs with the KadaneSegI64 monoid");
    kadane::LagWindow<LagV>* lag_state = kshared.lag;
    if (tid == kLeaderTid) {
        for (SizeType k = 0; k < nb; ++k) {
            lag_state[k].reset();
        }
    }

    if (tid == kLeaderTid && input.sample_begin > 0) {
        const SizeType state_off = index * nb;
        for (SizeType k = 0; k < nb; ++k) {
            if constexpr (kUseIntGate) {
                c_state_i32[k] = gate.state_ic[state_off + k];
                if (gate.any_edge_lag) {
                    lag_state[k] = gate.lag_i32[state_off + k];
                }
            } else {
                c_state_f32[k] = gate.state_fc[state_off + k];
                if (gate.any_edge_lag) {
                    lag_state[k] = gate.lag_f32[state_off + k];
                }
            }
        }
    }

    bool ring_ready        = false;
    SizeType ring_synced_t = 0;

    for (SizeType t0 = 0; t0 < input.nsamps; t0 += kBlockTile) {
        const int n = static_cast<int>(
            input.nsamps - t0 < kBlockTile ? input.nsamps - t0 : kBlockTile);
        const uint64_t start = input.sample_begin + t0;

        T x[kItems];
        load_tile_samples<T, kBlockThreads>(input, row, beam, t0, n, root, x);

        float x_f32[kItems]{};
        if constexpr (!kUseIntGate) {
#pragma unroll
            for (int j = 0; j < kItems; ++j) {
                const SizeType local = static_cast<SizeType>(tid * kItems + j);
                const bool valid =
                    local < static_cast<SizeType>(n) &&
                    (!skip_cold || (start + local) >= row.warmup);
                if (valid) {
                    if constexpr (std::is_integral_v<T>) {
                        x_f32[j] = static_cast<float>(
                            (static_cast<double>(x[j]) - row_m1) * unit);
                    } else {
                        x_f32[j] = x[j];
                    }
                }
            }
        }

        for (SizeType k = 0; k < nb; ++k) {
            SegT local_seg{};
#pragma unroll
            for (int j = 0; j < kItems; ++j) {
                const SizeType local = static_cast<SizeType>(tid * kItems + j);
                const bool valid =
                    local < static_cast<SizeType>(n) &&
                    (!skip_cold || (start + local) >= row.warmup);
                SegT item_seg{};
                if constexpr (kUseIntGate) {
                    item_seg = kadane::make_kadane_seg_i64(
                        static_cast<uint64_t>(x[j]), gate.int_shift,
                        row_ibias[k], valid);
                } else {
                    item_seg = kadane::make_kadane_seg_f32(
                        x_f32[j], row_biases_f32[k], valid);
                }
                local_seg = j == 0
                                ? item_seg
                                : kadane::kadane_combine(local_seg, item_seg);
            }
            const SegT warp_tot = kadane::warp_kadane_reduce(local_seg, lane);
            if (lane == kWarp - 1) {
                kshared.warp_segs[k][warp] = warp_tot;
            }
        }
        __syncthreads();

        bool my_hit = false;
        if (warp == 0) {
            const bool has_cold_start = skip_cold && start < row.warmup;
            const uint64_t tile_end   = start + static_cast<uint64_t>(n);
            for (SizeType k = 0; k < nb; ++k) {
                const SegT total_seg =
                    kadane::warp_totals_kadane_reduce<SegT, kBlockWarps>(
                        kshared.warp_segs[k], lane);
                if (lane == kLeaderTid) {
                    const uint32_t lag_len = gate.max_edge_lag[k];
                    if constexpr (kUseIntGate) {
                        const int32_t c_in =
                            has_cold_start ? int32_t{0} : c_state_i32[k];
                        const auto app = kadane::kadane_apply(c_in, total_seg);
                        c_state_i32[k] = app.c_out;
                        int64_t eff_m  = app.tile_max;
                        if (lag_len > 0U) {
                            const uint64_t w_beg =
                                start > lag_len ? (start - lag_len) : 0ULL;
                            const auto lag_v = static_cast<int64_t>(
                                lag_state[k].query_and_prune(w_beg));
                            eff_m = eff_m > lag_v ? eff_m : lag_v;
                            lag_state[k].push(
                                tile_end,
                                kadane::clamp_kadane_i32(app.tile_max));
                        }
                        if (gate.disabled_row[row_index] == 0U &&
                            eff_m >= static_cast<int64_t>(row_icut[k])) {
                            my_hit = true;
                        }
                    } else {
                        const float c_in =
                            has_cold_start ? 0.0F : c_state_f32[k];
                        const auto app = kadane::kadane_apply(c_in, total_seg);
                        c_state_f32[k] = app.c_out;
                        float eff_m    = app.tile_max;
                        if (lag_len > 0U) {
                            const uint64_t w_beg =
                                start > lag_len ? (start - lag_len) : 0ULL;
                            const float lag_v =
                                lag_state[k].query_and_prune(w_beg);
                            eff_m = kadane::monoid_max_f32(eff_m, lag_v);
                            lag_state[k].push(tile_end, app.tile_max);
                        }
                        if (gate.disabled_row[row_index] == 0U &&
                            eff_m >= row_cuts_f32[k]) {
                            my_hit = true;
                        }
                    }
                }
            }
        }

        if (__syncthreads_or(my_hit || open) != 0) {
            if (!ring_ready) {
                if (shared_tables && !tables_staged) {
                    stage_row_tables<T, kBlockThreads, kBlockNoise>(
                        s_thr, s_widths, widths, nfilters, noise, play);
                    tables_staged = true;
                }
                bootstrap_carry_ring<T, C, kBlockThreads>(
                    shared, ring, mask, carry, max_width, lookback,
                    input.sample_begin);
                ring_ready    = true;
                ring_synced_t = 0;
            }

            while (ring_synced_t < t0) {
                const auto pos_tau =
                    static_cast<uint32_t>(lookback + ring_synced_t);
                const T base_tau = ring[pos_tau & mask];
                T prev_x[kItems];
                load_tile_samples<T, kBlockThreads, true>(
                    input, row, beam, ring_synced_t, kBlockTile, root, prev_x);
                prefix_tile<T, kBlockThreads>(shared, ring, mask, prev_x,
                                              pos_tau, base_tau);
                recenter_prefix_ring<T, kBlockThreads>(
                    ring, mask, pos_tau + static_cast<uint32_t>(kBlockTile),
                    lookback);
                ring_synced_t += kBlockTile;
            }

            const auto position = static_cast<uint32_t>(lookback + t0);
            const T base        = ring[position & mask];
            prefix_tile<T, kBlockThreads>(shared, ring, mask, x, position,
                                          base);

            score_tile_bookkeeping<T, kBlockThreads, kBlockNoise, kKeep>(
                shared, ring, mask, position, n, start, row, row_index, wtab,
                ttab, nfilters, noise, block_tables, means, unit, mode,
                skip_cold, use_trapezoid, peaks, out, pending);
            open = any_open_peaks<kBlockThreads>(peaks, peak_count, mode);

            if (t0 + kBlockTile < input.nsamps) {
                recenter_prefix_ring<T, kBlockThreads>(
                    ring, mask, position + static_cast<uint32_t>(n), lookback);
                ring_synced_t = t0 + kBlockTile;
            }
        }
    }

    if (tid == kLeaderTid) {
        const SizeType state_off = index * nb;
        for (SizeType k = 0; k < nb; ++k) {
            if constexpr (kUseIntGate) {
                gate.state_ic[state_off + k] = c_state_i32[k];
                if (gate.any_edge_lag) {
                    gate.lag_i32[state_off + k] = lag_state[k];
                }
            } else {
                gate.state_fc[state_off + k] = c_state_f32[k];
                if (gate.any_edge_lag) {
                    gate.lag_f32[state_off + k] = lag_state[k];
                }
            }
        }
    }
    if (tid == 0) {
        open_all[index] = open ? 1 : 0;
    }

    // Exact raw carry update directly from input.
    save_row_carry<T, C, kBlockThreads>(input, row, beam, root, max_width,
                                        carry);
}

} // namespace

void FDMTBoxcarCudaEngine::populate_row_int_m1(std::span<const double> means) {
    if (!m_row_int_m1_d.empty()) {
        return;
    }
    const SizeType n_rows = m_rows.size();
    const uint32_t w0     = m_planes.front().widths[0];
    const uint32_t we0 = m_config.use_trapezoid ? trapezoid_edge_width(w0) : 0U;
    const double g0    = we0 == 0U ? static_cast<double>(w0)
                                   : static_cast<double>(2U * (w0 - we0));
    std::vector<double> row_m1(n_rows, 0.0);
    for (SizeType r = 0; r < n_rows; ++r) {
        row_m1[r] = means[r * m_filters] / g0;
    }
    m_row_int_m1_d = row_m1;
}

template <typename T,
          typename SegT,
          int kBlockThreads,
          bool kUseIntGate,
          bool kBlockNoise,
          typename C>
void FDMTBoxcarCudaEngine::configure_kadane_variants() {
    configure_kernel(
        kernel_fdmt_boxcar_kadane<T, SegT, kBlockThreads, kUseIntGate,
                                  kBlockNoise, C, false>);
    if (m_bright_q != 0) {
        configure_kernel(
            kernel_fdmt_boxcar_kadane<T, SegT, kBlockThreads, kUseIntGate,
                                      kBlockNoise, C, true>);
    }
}

template <typename T, typename SegT, int kBlockThreads>
void FDMTBoxcarCudaEngine::configure_shared_kadane_block(SizeType limit) {
    constexpr bool kUseIntGate = std::is_same_v<SegT, kadane::KadaneSegI64>;
    // Design C SMEM halving: no s_screen range-minimum buffer allocated.
    configure_shared_layout<T, kBlockThreads>(
        sizeof(KadaneBlockShared<T, SegT, kBlockThreads>), limit, false,
        "FDMTBoxcar (kKadane)");
    configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, false, T>();
    configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, true, T>();
    if constexpr (std::is_integral_v<T>) {
        configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, false,
                                  uint32_t>();
        configure_kadane_variants<T, SegT, kBlockThreads, kUseIntGate, true,
                                  uint32_t>();
    }
}

template <typename T, typename SegT>
bool FDMTBoxcarCudaEngine::try_configure_shared_kadane() {
    const SizeType limit   = shared_mem_limit();
    const auto kadane_need = [&](int threads) {
        return block_shared_need<T>(
            threads, false, []<int K>(std::integral_constant<int, K>) {
                return sizeof(KadaneBlockShared<T, SegT, K>);
            });
    };
    if constexpr (std::is_integral_v<T>) {
        if (kadane_need(kKadaneThreads) > limit) {
            return false;
        }
        configure_shared_kadane_block<T, SegT, kKadaneThreads>(limit);
        return true;
    } else {
        // Exactly the kPrefix kernel's block size (shared policy): bit-exact
        // float S/N. Where the Kadane kernel does not fit at that size, fall
        // back to kPrefix.
        const int threads =
            select_prefix_block_threads<T>(limit, m_block_shrunk);
        if (kadane_need(threads) > limit) {
            return false;
        }
        if (m_block_shrunk) {
            logging::debug("FDMTBoxcar (kKadane): max_boxcar {} in {} B of "
                           "shared memory: {}-thread blocks",
                           m_width, limit, threads);
        }
        if (threads == 64) {
            configure_shared_kadane_block<T, SegT, 64>(limit);
        } else if (threads == 128) {
            configure_shared_kadane_block<T, SegT, 128>(limit);
        } else {
            configure_shared_kadane_block<T, SegT, 256>(limit);
        }
        return true;
    }
}

void FDMTBoxcarCudaEngine::init_kadane(bool use_box_smearing,
                                       std::span<const double> means,
                                       std::span<const uint64_t> cutoffs) {
    // Wide boxcars (> 4096) use the device-memory prefix-ring path directly.
    if (m_width > kFastMaxWidth) {
        m_kadane_fallback = true;
        if (m_config.input_nbits == 32) {
            configure_shared<float>();
        } else {
            configure_shared<uint64_t>();
        }
        return;
    }

    const SizeType n_rows  = m_rows.size();
    const SizeType total_f = n_rows * m_filters;
    std::vector<float> all_raw_thresh;
    all_raw_thresh.reserve(total_f);
    for (const auto& p : m_planes) {
        all_raw_thresh.insert(all_raw_thresh.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
    }

    if (m_config.input_nbits != 32) {
        std::vector<float> eff_raw_thresh = all_raw_thresh;
        for (SizeType i = 0; i < total_f; ++i) {
            if (cutoffs[i] != UINT64_MAX) {
                const double centered =
                    (static_cast<double>(cutoffs[i]) - means[i]) *
                    m_integer_unit;
                if (centered > 0.0) {
                    eff_raw_thresh[i] = std::min(eff_raw_thresh[i],
                                                 static_cast<float>(centered));
                }
            }
        }
        const uint64_t max_int_sample = max_int_search_sample(
            m_plan, use_box_smearing, m_config.input_nbits);
        m_kadane_plan = kadane::make_kadane_gate_plan(
            m_planes.front().widths, m_config.use_trapezoid, n_rows,
            eff_raw_thresh, means, cutoffs, m_carry32, max_int_sample);
        m_kadane_int32 = m_carry32 && m_kadane_plan.has_int32;
    } else {
        m_kadane_plan  = kadane::make_kadane_gate_plan(m_planes.front().widths,
                                                       m_config.use_trapezoid,
                                                       n_rows, all_raw_thresh);
        m_kadane_int32 = false;
    }

    const bool configured =
        m_kadane_int32
            ? try_configure_shared_kadane<uint64_t, kadane::KadaneSegI64>()
        : m_config.input_nbits == 32
            ? try_configure_shared_kadane<float, kadane::KadaneSegF32>()
            : try_configure_shared_kadane<uint64_t, kadane::KadaneSegF32>();
    if (!configured) {
        // Tiny GPU SMEM cannot hold even a 64-thread KadaneBlockShared:
        // fall back cleanly to the unscreened / device-ring prefix scorer.
        m_kadane_fallback = true;
        m_block_shrunk    = false;
        m_kadane_plan     = {};
        m_kadane_int32    = false;
        if (m_config.input_nbits == 32) {
            configure_shared<float>();
        } else {
            configure_shared<uint64_t>();
        }
        return;
    }

    if (m_config.input_nbits != 32 && !m_kadane_int32) {
        populate_row_int_m1(means);
    }

    const SizeType nb    = m_kadane_plan.num_buckets;
    const SizeType tasks = m_rows.size() * m_beams;
    m_any_edge_lag       = false;
    for (SizeType k = 0; k < nb; ++k) {
        if (m_kadane_plan.max_edge_lag[k] > 0U) {
            m_any_edge_lag = true;
        }
    }

    m_kadane_disabled_d = m_kadane_plan.disabled_row;
    if (m_kadane_int32) {
        m_kadane_ibias_d = m_kadane_plan.ibias;
        m_kadane_icut_d  = m_kadane_plan.icut;
        m_kadane_ic_d.resize(tasks * nb, 0);
        if (m_any_edge_lag) {
            m_lag_i32_d.resize_zeroed(tasks * nb);
        }
    } else {
        m_kadane_bias_d = m_kadane_plan.bias;
        m_kadane_cut_d  = m_kadane_plan.cut;
        m_kadane_fc_d.resize(tasks * nb, 0.0F);
        if (m_any_edge_lag) {
            m_lag_f32_d.resize_zeroed(tasks * nb);
        }
    }
}

void FDMTBoxcarCudaEngine::reset_kadane(cudaStream_t stream) {
    if (!m_kadane_fc_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_kadane_fc_d), 0,
                            m_kadane_fc_d.size() * sizeof(float), stream));
    }
    if (!m_kadane_ic_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_kadane_ic_d), 0,
                            m_kadane_ic_d.size() * sizeof(int32_t), stream));
    }
    if (!m_lag_f32_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_lag_f32_d), 0,
            m_lag_f32_d.size() * sizeof(kadane::LagWindow<float>), stream));
    }
    if (!m_lag_i32_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_lag_i32_d), 0,
            m_lag_i32_d.size() * sizeof(kadane::LagWindow<int32_t>), stream));
    }
}

kadane::KadaneGateDeviceView FDMTBoxcarCudaEngine::make_kadane_view() noexcept {
    kadane::KadaneGateDeviceView view;
    view.num_buckets  = m_kadane_plan.num_buckets;
    view.row_stride   = m_kadane_plan.row_stride;
    view.int_shift    = m_kadane_plan.int_shift;
    view.has_int32    = m_kadane_int32;
    view.any_edge_lag = m_any_edge_lag;
    for (SizeType k = 0; k < m_kadane_plan.num_buckets; ++k) {
        view.buckets[k]      = m_kadane_plan.buckets[k];
        view.max_edge_lag[k] = m_kadane_plan.max_edge_lag[k];
    }
    view.bias         = raw(m_kadane_bias_d);
    view.cut          = raw(m_kadane_cut_d);
    view.ibias        = raw(m_kadane_ibias_d);
    view.icut         = raw(m_kadane_icut_d);
    view.disabled_row = raw(m_kadane_disabled_d);
    view.row_int_m1   = raw(m_row_int_m1_d);
    view.state_fc     = raw(m_kadane_fc_d);
    view.state_ic     = raw(m_kadane_ic_d);
    view.lag_f32      = raw(m_lag_f32_d);
    view.lag_i32      = raw(m_lag_i32_d);
    return view;
}

template <typename T, typename C>
void FDMTBoxcarCudaEngine::launch_score_kadane(const FDMTSearchInput& input,
                                               const LevelGroup& g,
                                               dim3 grid,
                                               dim3 block,
                                               C* carry) {
    const auto stream       = static_cast<cudaStream_t>(input.stream.native);
    FDMTSearchInput stopped = input;
    stopped.search_stop     = m_search_stop;
    const auto gate         = make_kadane_view();
    const bool root         = g.level == m_plan.get_niters();

    const auto dispatch_keep = [&](auto seg_tag, auto int_gate_tag,
                                   auto noise_tag, auto keep_tag) {
        using SegT                 = decltype(seg_tag);
        constexpr bool kUseIntGate = decltype(int_gate_tag)::value;
        constexpr bool kNoise      = decltype(noise_tag)::value;
        constexpr bool kKeep       = decltype(keep_tag)::value;
        const BlockTables bt       = kNoise ? m_block_tables : BlockTables{};
        const auto launch          = [&](auto threads_tag) {
            constexpr int kThreads = decltype(threads_tag)::value;
            kernel_fdmt_boxcar_kadane<T, SegT, kThreads, kUseIntGate, kNoise, C,
                                      kKeep><<<grid, block, m_smem, stream>>>(
                stopped, raw(m_rows_d), g.row_begin, g.row_count, m_rows.size(),
                raw(m_widths_d), m_filters, m_width, raw(m_thresholds_d),
                raw(m_scales_d), carry, raw(m_peaks_d), raw(m_open_d),
                m_peak_count, candidate_out(), raw(m_counts_d), raw(m_slots_d),
                raw(m_per_dm_d), m_config.max_candidates, dm_cap(),
                m_config.peak_mode, m_config.skip_cold_prefix, root,
                raw(m_integer_thresholds), raw(m_integer_means), m_integer_unit,
                m_mask, m_shared_tables, m_config.use_trapezoid, gate, bt,
                raw(m_bright_d), m_bright_q, raw(m_row_detected_d),
                raw(m_plane_slots_d), m_plane_cap);
        };
        if constexpr (std::is_integral_v<T>) {
            launch(std::integral_constant<int, kKadaneThreads>{});
        } else if (m_block_threads == 64) {
            launch(std::integral_constant<int, 64>{});
        } else if (m_block_threads == 128) {
            launch(std::integral_constant<int, 128>{});
        } else {
            launch(std::integral_constant<int, 256>{});
        }
    };

    const auto dispatch_threads = [&](auto seg_tag, auto int_gate_tag,
                                      auto noise_tag) {
        if (m_bright_q != 0) {
            dispatch_keep(seg_tag, int_gate_tag, noise_tag,
                          cuda::std::true_type{});
        } else {
            dispatch_keep(seg_tag, int_gate_tag, noise_tag,
                          cuda::std::false_type{});
        }
    };

    if constexpr (std::is_integral_v<T>) {
        // Per-call mask tables take their own kernel instance.
        const bool noise = m_block_tables.count > 0;
        if (m_kadane_int32) {
            if (noise) {
                dispatch_threads(kadane::KadaneSegI64{}, cuda::std::true_type{},
                                 cuda::std::true_type{});
            } else {
                dispatch_threads(kadane::KadaneSegI64{}, cuda::std::true_type{},
                                 cuda::std::false_type{});
            }
        } else if (noise) {
            dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                             cuda::std::true_type{});
        } else {
            dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                             cuda::std::false_type{});
        }
    } else if (m_block_tables.count > 0) {
        dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                         cuda::std::true_type{});
    } else {
        dispatch_threads(kadane::KadaneSegF32{}, cuda::std::false_type{},
                         cuda::std::false_type{});
    }
}

template void FDMTBoxcarCudaEngine::launch_score_kadane<float, float>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, float*);
template void FDMTBoxcarCudaEngine::launch_score_kadane<uint64_t, uint64_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint64_t*);
template void FDMTBoxcarCudaEngine::launch_score_kadane<uint64_t, uint32_t>(
    const FDMTSearchInput&, const LevelGroup&, dim3, dim3, uint32_t*);

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda
