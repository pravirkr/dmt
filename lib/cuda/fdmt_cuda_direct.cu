#include "fdmt_cuda_engine.cuh"
#include "fdmt_direct.cuh"

// CUDA FDMT engine: the direct levels and the direct top, with
// their shared-memory opt-ins next to their launches.

namespace dmt::algorithms::detail::fdmt_cuda {

template <typename TInt>
void FDMTCudaEngine::launch_direct_int(const FDMTInputPacked& input,
                                       TInt* out,
                                       cudaStream_t stream) {
    auto& d          = *m_direct;
    const int nchans = static_cast<int>(m_plan.get_nchans());
    const int active = static_cast<int>(m_active);
    if (d.skewed) {
        const dim3 wgrid((active + 255) / 256, nchans, m_nbeams);
        const int ntiles = (active + d.max_extra + detail::kDirectTile - 1) /
                           detail::kDirectTile;
        const int row_stride = static_cast<int>(m_chunk);
        const auto b_stride_k =
            static_cast<int64_t>(level_beam_stride(d.levels));
        const auto lvl_ptr = [&](SizeType l) -> TInt* {
            return state_level_ptr<TInt>(l);
        };
        if (m_search_nbits <= 8) {
            auto* wring = thrust::raw_pointer_cast(d.win_ring_u8.data());
            detail::kernel_direct_write_ring<FDMTInputPacked, uint8_t>
                <<<wgrid, 256, 0, stream>>>(
                    wring, d.win_ring_size,
                    thrust::raw_pointer_cast(d.chan_ring_off.data()),
                    thrust::raw_pointer_cast(d.chan_ring_len.data()), input,
                    nchans, active, m_chunk_pos);
            for (SizeType b = 0; b < m_skew_nbatches; ++b) {
                const int32_t sub0    = d.batch_sub_off[b];
                const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
                if (nsubs_b > 0) {
                    const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                    gpu_utils::check_kernel_launch_params(
                        lgrid, dim3(detail::kDirectWarps * 32));
                    detail::kernel_direct_level_skewed<uint8_t, uint32_t, TInt>
                        <<<lgrid, detail::kDirectWarps * 32, d.smem, stream>>>(
                            wring, d.win_ring_size,
                            thrust::raw_pointer_cast(d.chan_ring_off.data()),
                            thrust::raw_pointer_cast(d.chan_ring_len.data()),
                            thrust::raw_pointer_cast(d.subs.data()) + sub0,
                            thrust::raw_pointer_cast(d.leaf_off.data()),
                            thrust::raw_pointer_cast(d.leaves.data()), out,
                            row_stride, b_stride_k, d.cpass, m_chunk_pos,
                            active);
                }
                if (m_skew_nbatches > 1 && m_top) {
                    launch_skewed_batch_mid<TInt>(b, lvl_ptr, stream);
                }
            }
        } else {
            auto* wring = thrust::raw_pointer_cast(d.win_ring_u16.data());
            detail::kernel_direct_write_ring<FDMTInputPacked, uint16_t>
                <<<wgrid, 256, 0, stream>>>(
                    wring, d.win_ring_size,
                    thrust::raw_pointer_cast(d.chan_ring_off.data()),
                    thrust::raw_pointer_cast(d.chan_ring_len.data()), input,
                    nchans, active, m_chunk_pos);
            for (SizeType b = 0; b < m_skew_nbatches; ++b) {
                const int32_t sub0    = d.batch_sub_off[b];
                const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
                if (nsubs_b > 0) {
                    const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                    gpu_utils::check_kernel_launch_params(
                        lgrid, dim3(detail::kDirectWarps * 32));
                    detail::kernel_direct_level_skewed<uint16_t, uint32_t, TInt>
                        <<<lgrid, detail::kDirectWarps * 32, d.smem, stream>>>(
                            wring, d.win_ring_size,
                            thrust::raw_pointer_cast(d.chan_ring_off.data()),
                            thrust::raw_pointer_cast(d.chan_ring_len.data()),
                            thrust::raw_pointer_cast(d.subs.data()) + sub0,
                            thrust::raw_pointer_cast(d.leaf_off.data()),
                            thrust::raw_pointer_cast(d.leaves.data()), out,
                            row_stride, b_stride_k, d.cpass, m_chunk_pos,
                            active);
                }
                if (m_skew_nbatches > 1 && m_top) {
                    launch_skewed_batch_mid<TInt>(b, lvl_ptr, stream);
                }
            }
        }
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): integer skewed direct levels failed");
        return;
    }
    auto* prefix =
        reinterpret_cast<uint32_t*>(thrust::raw_pointer_cast(d.prefix.data()));
    detail::kernel_direct_prefix<FDMTInputPacked, uint32_t>
        <<<dim3(nchans, 1, m_nbeams), detail::kPrefixThreads, 0, stream>>>(
            input, m_direct_win_in, d.window, nchans, active, prefix, d.pitch);
    constexpr int kRows = detail::kDirectRowsPerWarp * detail::kDirectWarps;
    const int ntiles = (active + detail::kDirectTile - 1) / detail::kDirectTile;
    const dim3 grid((d.max_rows + kRows - 1) / kRows, d.nsubs,
                    m_nbeams * ntiles);
    gpu_utils::check_kernel_launch_params(grid,
                                          dim3(detail::kDirectWarps * 32));
    detail::kernel_direct_level<uint32_t, TInt>
        <<<grid, detail::kDirectWarps * 32, d.smem, stream>>>(
            prefix, d.pitch, d.window, nchans,
            thrust::raw_pointer_cast(d.subs.data()),
            thrust::raw_pointer_cast(d.leaf_off.data()),
            thrust::raw_pointer_cast(d.leaves.data()), out,
            static_cast<int>(m_chunk), static_cast<int64_t>(m_half_elems),
            d.cpass, active, ntiles);
    gpu_utils::check_last_gpu_error(
        "FDMT (cuda): integer direct levels failed");
}

template <typename TInt>
void FDMTCudaEngine::run_top_int(const TInt* level_state,
                                 TInt* out,
                                 int64_t out_beam_stride,
                                 cudaStream_t stream) {
    auto& t              = *m_top;
    constexpr int kWarps = 8;
    constexpr int kSamps = 32 * detail::kTopSampsPerLane;
    const int active     = static_cast<int>(m_active);
    auto* rings =
        reinterpret_cast<TInt*>(thrust::raw_pointer_cast(t.ring.data()));
    if (t.target_ring) {
        const dim3 grid(t.ntrials, 1, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
        detail::kernel_direct_top_target<TInt>
            <<<grid, kWarps * 32, 0, stream>>>(
                level_state, static_cast<int64_t>(m_half_elems),
                static_cast<int>(m_chunk), rings, t.ring_size,
                thrust::raw_pointer_cast(t.ring_off.data()),
                thrust::raw_pointer_cast(t.ring_len.data()),
                thrust::raw_pointer_cast(t.leaf_off.data()),
                thrust::raw_pointer_cast(t.leaves.data()),
                thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
                out_beam_stride, static_cast<int>(m_chunk), m_chunk_pos, active,
                0, m_skew_nbatches > 1 ? 0 : -1, true);
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): integer direct top target failed");
        return;
    }
    const dim3 grid((t.ntrials + kWarps - 1) / kWarps,
                    (active + kSamps - 1) / kSamps, m_nbeams);
    gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
    detail::kernel_direct_top<TInt><<<grid, kWarps * 32, 0, stream>>>(
        level_state, static_cast<int64_t>(m_half_elems),
        static_cast<int>(m_chunk), rings, t.ring_size,
        thrust::raw_pointer_cast(t.ring_off.data()),
        thrust::raw_pointer_cast(t.ring_len.data()),
        thrust::raw_pointer_cast(t.leaf_off.data()),
        thrust::raw_pointer_cast(t.leaves.data()),
        thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
        out_beam_stride, static_cast<int>(m_chunk), m_chunk_pos, active);
    constexpr int kSegment = 256;
    const int newest       = std::min(t.max_ring, active);
    if (newest > 0) {
        const dim3 rgrid((t.nrows + kWarps - 1) / kWarps,
                         (newest + kSegment - 1) / kSegment, m_nbeams);
        detail::kernel_direct_top_ring<TInt><<<rgrid, kWarps * 32, 0, stream>>>(
            level_state, static_cast<int64_t>(m_half_elems),
            static_cast<int>(m_chunk), rings, t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()), t.nrows, m_chunk_pos,
            active);
    }
    gpu_utils::check_last_gpu_error("FDMT (cuda): integer direct top failed");
}

void FDMTCudaEngine::configure_direct(SizeType requested) {
    m_direct.reset();
    m_direct_eligible     = false;
    const SizeType levels = kDirectLevels;
    // The packed search (exact integers) takes them too: its level-0
    // boxes use the plan's integer dyadic weights unrounded, so level K
    // is an exact integer sum of weighted input boxes (uint32_t running
    // sums, kernel_direct_level<uint32_t, TInt>).
    const bool packed = m_search_nbits != 32;
    if (packed) {
        requested = m_requested_fuse;
    }
    if (m_mode != FDMTMode::kValid || requested != kFDMTAutoFuse ||
        m_plan.get_niters() <= levels || m_search_max_fuse < levels ||
        get_fuse_levels() >= levels) {
        return;
    }
    m_direct_eligible = true;
    std::vector<float> box_w(m_box_w_d.size());
    if (packed) {
        // Integer weights (exact in float), so the leaves give the
        // integer search's level-0 values.
        const auto weights = m_plan.get_box_weights();
        for (SizeType i = 0; i < box_w.size(); ++i) {
            box_w[i] = static_cast<float>(weights[i]);
        }
    } else {
        thrust::copy(m_box_w_d.begin(), m_box_w_d.end(), box_w.begin());
    }
    auto plan = detail::build_fdmt_direct_plan(m_plan, levels,
                                               m_use_box_smearing, box_w);
    int span  = 0;
    for (const auto& sub : plan.subs) {
        span = std::max(span, sub.span);
    }
    const auto per_channel =
        static_cast<SizeType>(span + detail::kDirectTile + 1) * sizeof(float);
    const SizeType optin = max_optin_smem_bytes();
    const auto rows      = m_nbeams * m_plan.get_nchans();
    if (per_channel > optin ||
        rows * static_cast<SizeType>(plan.window) * sizeof(float) >
            (SizeType{256} << 20)) {
        return;
    }
    if (packed) {
        // uint32_t running sums over the window and a chunk.
        const auto maxval = static_cast<double>(
            bit_pack_utils::max_sample_value(m_search_nbits));
        if (static_cast<double>(plan.window + m_chunk + 1) * maxval >
            4294967295.0) {
            return;
        }
    }
    auto d         = std::make_unique<DirectLevelsD>();
    d->levels      = levels;
    d->window      = plan.window;
    d->nsubs       = static_cast<int>(plan.subs.size());
    d->max_rows    = plan.max_rows;
    d->cpass       = static_cast<int>(std::min<SizeType>(
        static_cast<SizeType>(plan.max_nc), optin / per_channel));
    d->smem        = static_cast<SizeType>(d->cpass) * per_channel;
    d->subs        = plan.subs;
    d->leaf_off    = plan.leaf_off;
    d->leaves      = plan.leaves;
    d->hist_leaves = plan.hist_leaves;
    d->hist_coords.resize(levels + 1);
    d->hist_max_delay.assign(levels + 1, 0);
    for (SizeType l = 1; l <= levels; ++l) {
        d->hist_coords[l] = plan.hist_coords[l];
        for (const auto& hc : plan.hist_coords[l]) {
            d->hist_max_delay[l] = std::max(d->hist_max_delay[l], hc.delay);
        }
    }
    for (auto& w : d->win) {
        w.resize(rows * static_cast<SizeType>(d->window), 0.0F);
    }
    d->pitch = d->window + static_cast<int>(m_chunk) + 1;
    d->prefix.resize(rows * static_cast<SizeType>(d->pitch));
    const auto opt_in = [&](auto kernel) {
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(d->smem)),
            "FDMT (cuda): direct levels shared-memory opt-in");
    };
    if (!packed) {
        opt_in(detail::kernel_direct_level<>);
    } else if (m_search_u32) {
        opt_in(detail::kernel_direct_level<uint32_t, uint32_t>);
    } else {
        opt_in(detail::kernel_direct_level<uint32_t, uint64_t>);
    }
    logging::debug("FDMT (cuda): levels 0..{} direct ({} subbands, {} "
                   "leaves, window {}, {} channels per pass, {} B shared)",
                   levels, d->nsubs, plan.leaves.size(), d->window, d->cpass,
                   d->smem);
    m_direct_filled = static_cast<SizeType>(d->window);
    m_direct        = std::move(d);
    m_fused.reset();
}

template <typename T, typename PtrFn>
void FDMTCudaEngine::launch_skewed_batch_mid(SizeType b,
                                             PtrFn&& lvl_ptr,
                                             cudaStream_t stream) {
    const SizeType k     = m_direct->levels;
    const SizeType top_k = m_top->level;
    for (SizeType l = k + 1; l <= top_k; ++l) {
        const int sum_b0         = m_skew_batch_sum_off[l][b];
        const int sum_b1         = m_skew_batch_sum_off[l][b + 1];
        const int copy_b0        = m_skew_batch_copy_off[l][b];
        const int copy_b1        = m_skew_batch_copy_off[l][b + 1];
        const int ncoords_sum_b  = sum_b1 - sum_b0;
        const int ncoords_copy_b = copy_b1 - copy_b0;
        const int coords_max     = std::max(ncoords_sum_b, ncoords_copy_b);
        if (coords_max <= 0) {
            continue;
        }
        auto sums   = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto copies = m_plan_d.coordinates_copy.get_raw_ptrs();
        sums.update_offsets(m_coords_sum_offsets[l] + sum_b0);
        copies.update_offsets(m_coords_copy_offsets[l] + copy_b0);
        const int nsamps_l = static_cast<int>(
            m_active +
            (l < m_level_max_extra.size() ? m_level_max_extra[l] : 0));
        constexpr int kBlockSamps = kIterThreads * kIterSampsPerTh;
        const dim3 block(kIterThreads, 1);
        const dim3 grid((nsamps_l + kBlockSamps - 1) / kBlockSamps,
                        std::min(coords_max, kMaxGridY), m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, block);
        kernel_execute_iter<FDMTMode::kValid, T, T, T>
            <<<grid, block, 0, stream>>>(
                lvl_ptr(l - 1), lvl_ptr(l), sums, copies, nullptr, nsamps_l,
                ncoords_sum_b, ncoords_copy_b, level_beam_stride(l - 1),
                level_beam_stride(l), 0, 0, m_chunk_pos, 1.0F,
                static_cast<int>(m_active), false, static_cast<int>(m_chunk));
    }
    if (m_top && m_top->target_ring) {
        auto& t              = *m_top;
        constexpr int kWarps = 8;
        const dim3 grid(t.ntrials, 1, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
        auto* rings =
            reinterpret_cast<T*>(thrust::raw_pointer_cast(t.ring.data()));
        detail::kernel_direct_top_target<T><<<grid, kWarps * 32, 0, stream>>>(
            lvl_ptr(top_k), static_cast<int64_t>(m_half_elems),
            static_cast<int>(m_chunk), rings, t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()),
            thrust::raw_pointer_cast(t.leaf_off.data()),
            thrust::raw_pointer_cast(t.leaves.data()),
            thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, nullptr, 0,
            0, m_chunk_pos, static_cast<int>(m_active),
            m_skew_batch_top_grid[b], m_skew_batch_top_grid[b + 1], false);
    }
}

template <typename Input>
void FDMTCudaEngine::launch_direct(const Input& input, cudaStream_t stream) {
    auto& d          = *m_direct;
    const int nchans = static_cast<int>(m_plan.get_nchans());
    const int active = static_cast<int>(m_active);
    if (d.skewed) {
        auto* wring = thrust::raw_pointer_cast(d.win_ring.data());
        const dim3 wgrid((active + 255) / 256, nchans, m_nbeams);
        detail::kernel_direct_write_ring<Input, float>
            <<<wgrid, 256, 0, stream>>>(
                wring, d.win_ring_size,
                thrust::raw_pointer_cast(d.chan_ring_off.data()),
                thrust::raw_pointer_cast(d.chan_ring_len.data()), input, nchans,
                active, m_chunk_pos);
        const int ntiles   = (active + d.max_extra + detail::kDirectTile - 1) /
                             detail::kDirectTile;
        const auto lvl_ptr = [&](SizeType l) {
            return reinterpret_cast<float*>(m_levels_d[l].base);
        };
        for (SizeType b = 0; b < m_skew_nbatches; ++b) {
            const int32_t sub0    = d.batch_sub_off[b];
            const int32_t nsubs_b = d.batch_sub_off[b + 1] - sub0;
            if (nsubs_b > 0) {
                const dim3 lgrid(nsubs_b, ntiles, m_nbeams);
                gpu_utils::check_kernel_launch_params(
                    lgrid, dim3(detail::kDirectWarps * 32));
                const auto args = std::make_tuple(
                    static_cast<const float*>(wring), d.win_ring_size,
                    thrust::raw_pointer_cast(d.chan_ring_off.data()),
                    thrust::raw_pointer_cast(d.chan_ring_len.data()),
                    thrust::raw_pointer_cast(d.subs.data()) + sub0,
                    thrust::raw_pointer_cast(d.leaf_off.data()),
                    thrust::raw_pointer_cast(d.leaves.data()),
                    lvl_ptr(d.levels), static_cast<int>(m_chunk),
                    level_beam_stride(d.levels), d.cpass, m_chunk_pos, active);
                if constexpr (detail::kFDMTPackedInput<Input>) {
                    // Packed input keeps the tree's rounded box means
                    // (exact integer sums, see FDMTDirectRound).
                    const auto bm = box_means_for(input);
                    const detail::FDMTDirectRound rnd{
                        .round     = m_use_box_smearing ? bm.round : nullptr,
                        .round_u64 = bm.round_u64,
                        .unit      = m_use_box_smearing ? m_int_unit : 1.0F,
                    };
                    std::apply(
                        [&](auto... a) {
                            detail::kernel_direct_level_skewed<
                                float, uint32_t, uint32_t, true, float>
                                <<<lgrid, detail::kDirectWarps * 32, d.smem,
                                   stream>>>(a..., rnd);
                        },
                        args);
                } else {
                    std::apply(
                        [&](auto... a) {
                            detail::kernel_direct_level_skewed<float, float,
                                                               float>
                                <<<lgrid, detail::kDirectWarps * 32, d.smem,
                                   stream>>>(a...);
                        },
                        args);
                }
            }
            if (m_skew_nbatches > 1 && m_top) {
                launch_skewed_batch_mid<float>(b, lvl_ptr, stream);
            }
        }
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): skewed direct levels failed");
        return;
    }
    detail::kernel_direct_prefix<Input>
        <<<dim3(nchans, 1, m_nbeams), detail::kPrefixThreads, 0, stream>>>(
            input, m_direct_win_in, d.window, nchans, active,
            thrust::raw_pointer_cast(d.prefix.data()), d.pitch);
    constexpr int kRows = detail::kDirectRowsPerWarp * detail::kDirectWarps;
    const int ntiles = (active + detail::kDirectTile - 1) / detail::kDirectTile;
    const dim3 grid((d.max_rows + kRows - 1) / kRows, d.nsubs,
                    m_nbeams * ntiles);
    gpu_utils::check_kernel_launch_params(grid,
                                          dim3(detail::kDirectWarps * 32));
    detail::kernel_direct_level<>
        <<<grid, detail::kDirectWarps * 32, d.smem, stream>>>(
            thrust::raw_pointer_cast(d.prefix.data()), d.pitch, d.window,
            nchans, thrust::raw_pointer_cast(d.subs.data()),
            thrust::raw_pointer_cast(d.leaf_off.data()),
            thrust::raw_pointer_cast(d.leaves.data()),
            reinterpret_cast<float*>(m_levels_d[d.levels].base),
            static_cast<int>(m_chunk), level_beam_stride(d.levels), d.cpass,
            active, ntiles);
    gpu_utils::check_last_gpu_error("FDMT (cuda): direct levels failed");
}

template <typename Input>
void FDMTCudaEngine::advance_direct_window(const Input& input,
                                           cudaStream_t stream) {
    if (!m_direct || m_direct->skewed || m_mode != FDMTMode::kValid) {
        return;
    }
    const auto& d   = *m_direct;
    const int nsubs = static_cast<int>(m_plan.get_nchans());
    const dim3 block(256);
    const dim3 grid((d.window + 255) / 256, nsubs, m_nbeams);
    kernel_advance_history_window<Input><<<grid, block, 0, stream>>>(
        m_direct_win_out, m_direct_win_in, input, nsubs,
        static_cast<int>(m_active), d.window);
    gpu_utils::check_last_gpu_error("FDMT (cuda): direct window update failed");
    m_direct_filled = std::min<SizeType>(static_cast<SizeType>(d.window),
                                         m_direct_filled + m_active);
}

void FDMTCudaEngine::rebuild_direct_history(cudaStream_t stream) const {
    if (!m_direct || !m_direct_stale || m_direct->skewed) {
        return;
    }
    const auto& d = *m_direct;
    const auto* win =
        thrust::raw_pointer_cast(m_direct->win[m_parity ? 1 : 0].data());
    auto* ring = const_cast<float*>( // NOLINT: device buffer, host const
        thrust::raw_pointer_cast(m_tree_ring_d.data()));
    for (SizeType l = 1; l <= d.levels; ++l) {
        const int n = static_cast<int>(d.hist_coords[l].size());
        if (n == 0) {
            continue;
        }
        const dim3 grid((d.hist_max_delay[l] + 255) / 256,
                        std::min(n, kMaxGridY), m_nbeams);
        detail::kernel_direct_rebuild_history<<<grid, 256, 0, stream>>>(
            win, d.window, static_cast<int>(m_plan.get_nchans()),
            thrust::raw_pointer_cast(d.hist_coords[l].data()), n,
            thrust::raw_pointer_cast(d.hist_leaves.data()), ring,
            static_cast<int64_t>(m_ring_size),
            static_cast<int64_t>(m_fifo_size), m_next_pos);
    }
    gpu_utils::check_last_gpu_error(
        "FDMT (cuda): direct history rebuild failed");
    m_direct_stale = false;
}

void FDMTCudaEngine::build_skewed_direct_from_windows(
    const std::vector<int32_t>& s_hi_k,
    const std::vector<int32_t>& sp_k,
    const std::vector<SizeType>& batch_grid_off) {
    const SizeType k  = kDirectLevels;
    const bool packed = m_search_nbits != 32;
    std::vector<float> box_w(m_box_w_d.size());
    if (packed) {
        const auto weights = m_plan.get_box_weights();
        for (SizeType i = 0; i < box_w.size(); ++i) {
            box_w[i] = static_cast<float>(weights[i]);
        }
    } else {
        thrust::copy(m_box_w_d.begin(), m_box_w_d.end(), box_w.begin());
    }
    auto dplan = detail::build_fdmt_direct_skewed_plan(
        m_plan, k, m_use_box_smearing, box_w, m_chunk, s_hi_k, sp_k,
        batch_grid_off);
    const auto per_channel =
        static_cast<SizeType>(dplan.max_span + detail::kDirectTile + 1) *
        sizeof(float);
    const SizeType optin = max_optin_smem_bytes();
    if (per_channel > optin) {
        return;
    }
    auto d                = std::make_unique<DirectLevelsD>();
    d->levels             = k;
    d->skewed             = true;
    d->window             = dplan.window;
    d->nsubs              = static_cast<int>(dplan.subs.size());
    d->max_rows           = dplan.max_rows;
    d->max_extra          = dplan.max_extra;
    d->total_elems        = static_cast<SizeType>(dplan.total_elems);
    d->win_ring_size      = dplan.win_ring_size;
    d->row_off            = std::move(dplan.row_off);
    d->batch_sub_off      = std::move(dplan.batch_sub_off);
    d->cpass              = static_cast<int>(std::min<SizeType>(
        static_cast<SizeType>(dplan.max_nc), optin / per_channel));
    d->smem               = static_cast<SizeType>(d->cpass) * per_channel;
    d->subs               = dplan.subs;
    d->leaf_off           = dplan.leaf_off;
    d->leaves             = dplan.leaves;
    d->chan_ring_len      = dplan.chan_ring_len;
    d->chan_ring_off      = dplan.chan_ring_off;
    const auto total_ring = m_nbeams * static_cast<SizeType>(d->win_ring_size);
    if (!packed) {
        d->win_ring.resize(total_ring, 0.0F);
    } else if (m_search_nbits <= 8) {
        d->win_ring_u8.resize(total_ring, 0);
    } else {
        d->win_ring_u16.resize(total_ring, 0);
    }
    const auto opt_in = [&](auto kernel) {
        gpu_utils::check_gpu_call(
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(d->smem)),
            "FDMT (cuda): skewed direct levels shared-memory opt-in");
    };
    if (!packed) {
        opt_in(detail::kernel_direct_level_skewed<float, float, float>);
        // Packed blocks into the float ring (rounded box means).
        opt_in(detail::kernel_direct_level_skewed<float, uint32_t, uint32_t,
                                                  true, float>);
    } else if (m_search_nbits <= 8 && m_search_u32) {
        opt_in(detail::kernel_direct_level_skewed<uint8_t, uint32_t, uint32_t>);
    } else if (m_search_nbits <= 8) {
        opt_in(detail::kernel_direct_level_skewed<uint8_t, uint32_t, uint64_t>);
    } else if (m_search_u32) {
        opt_in(
            detail::kernel_direct_level_skewed<uint16_t, uint32_t, uint32_t>);
    } else {
        opt_in(
            detail::kernel_direct_level_skewed<uint16_t, uint32_t, uint64_t>);
    }
    logging::debug(
        "FDMT (cuda): levels 0..{} skewed direct ({} slices, {} leaves, "
        "win_ring {} MiB, max_span {}, max_extra {}, {} B shared)",
        k, d->nsubs, dplan.leaves.size(),
        (total_ring * (packed ? (m_search_nbits <= 8 ? 1 : 2) : 4)) >> 20,
        dplan.max_span, d->max_extra, d->smem);
    m_direct_filled = static_cast<SizeType>(d->window);
    m_direct        = std::move(d);
    m_fused.reset();
}

void FDMTCudaEngine::run_top(float* out,
                             int64_t out_beam_stride,
                             int out_row_stride,
                             cudaStream_t stream) {
    auto& t              = *m_top;
    constexpr int kWarps = 8;
    constexpr int kSamps = 32 * detail::kTopSampsPerLane;
    const int active     = static_cast<int>(m_active);
    if (t.target_ring) {
        const dim3 grid(t.ntrials, 1, m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
        detail::kernel_direct_top_target<><<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<const float*>(m_levels_d[t.level].base),
            level_beam_stride(t.level), static_cast<int>(m_chunk),
            thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()),
            thrust::raw_pointer_cast(t.leaf_off.data()),
            thrust::raw_pointer_cast(t.leaves.data()),
            thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
            out_beam_stride, out_row_stride, m_chunk_pos, active, 0,
            m_skew_nbatches > 1 ? 0 : -1, true);
        gpu_utils::check_last_gpu_error(
            "FDMT (cuda): direct top target failed");
        m_top_stale     = false;
        m_current_level = t.target;
        return;
    }
    const dim3 grid((t.ntrials + kWarps - 1) / kWarps,
                    (active + kSamps - 1) / kSamps, m_nbeams);
    gpu_utils::check_kernel_launch_params(grid, dim3(kWarps * 32));
    detail::kernel_direct_top<><<<grid, kWarps * 32, 0, stream>>>(
        reinterpret_cast<const float*>(m_levels_d[t.level].base),
        level_beam_stride(t.level), static_cast<int>(m_chunk),
        thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
        thrust::raw_pointer_cast(t.ring_off.data()),
        thrust::raw_pointer_cast(t.ring_len.data()),
        thrust::raw_pointer_cast(t.leaf_off.data()),
        thrust::raw_pointer_cast(t.leaves.data()),
        thrust::raw_pointer_cast(t.out_row.data()), t.ntrials, out,
        out_beam_stride, out_row_stride, m_chunk_pos, active);
    gpu_utils::check_last_gpu_error("FDMT (cuda): direct top failed");
    update_top_rings(stream);
    m_top_stale     = !t.skewed;
    m_current_level = t.target;
}

void FDMTCudaEngine::update_top_rings(cudaStream_t stream) {
    if (m_top->target_ring) {
        return;
    }
    auto& t                = *m_top;
    constexpr int kWarps   = 8;
    constexpr int kSegment = 256;
    const int newest       = std::min(t.max_ring, static_cast<int>(m_active));
    if (newest > 0) {
        const dim3 grid((t.nrows + kWarps - 1) / kWarps,
                        (newest + kSegment - 1) / kSegment, m_nbeams);
        detail::kernel_direct_top_ring<><<<grid, kWarps * 32, 0, stream>>>(
            reinterpret_cast<const float*>(m_levels_d[t.level].base),
            level_beam_stride(t.level), static_cast<int>(m_chunk),
            thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()), t.nrows, m_chunk_pos,
            static_cast<int>(m_active));
        gpu_utils::check_last_gpu_error("FDMT (cuda): direct top rings failed");
    }
    m_top_filled = std::min<SizeType>(static_cast<SizeType>(t.max_ring),
                                      m_top_filled + m_active);
}

void FDMTCudaEngine::rebuild_top_history(cudaStream_t stream) const {
    if (!m_top || !m_top_stale || m_top->skewed) {
        return;
    }
    const auto& t = *m_top;
    auto* ring    = const_cast<float*>( // NOLINT: device buffer, host const
        thrust::raw_pointer_cast(m_tree_ring_d.data()));
    for (SizeType l = t.level + 1; l <= t.target; ++l) {
        const int n = static_cast<int>(t.hist_coords[l].size());
        if (n == 0) {
            continue;
        }
        const dim3 grid((t.hist_max_delay[l] + 255) / 256,
                        std::min(n, kMaxGridY), m_nbeams);
        detail::kernel_direct_top_rebuild_history<<<grid, 256, 0, stream>>>(
            thrust::raw_pointer_cast(t.ring.data()), t.ring_size,
            thrust::raw_pointer_cast(t.ring_off.data()),
            thrust::raw_pointer_cast(t.ring_len.data()),
            thrust::raw_pointer_cast(t.hist_coords[l].data()), n,
            thrust::raw_pointer_cast(t.hist_leaves.data()), ring,
            static_cast<int64_t>(m_ring_size),
            static_cast<int64_t>(m_fifo_size), m_next_pos);
    }
    gpu_utils::check_last_gpu_error(
        "FDMT (cuda): direct top history rebuild failed");
    m_top_stale = false;
}

// Called from fdmt_cuda_execute.cu.
template void FDMTCudaEngine::launch_direct_int<uint32_t>(
    const FDMTInputPacked& input, uint32_t* out, cudaStream_t stream);
template void FDMTCudaEngine::launch_direct_int<uint64_t>(
    const FDMTInputPacked& input, uint64_t* out, cudaStream_t stream);
template void FDMTCudaEngine::run_top_int<uint32_t>(const uint32_t* level_state,
                                                    uint32_t* out,
                                                    int64_t out_beam_stride,
                                                    cudaStream_t stream);
template void FDMTCudaEngine::run_top_int<uint64_t>(const uint64_t* level_state,
                                                    uint64_t* out,
                                                    int64_t out_beam_stride,
                                                    cudaStream_t stream);
template void FDMTCudaEngine::launch_direct<detail::FDMTInputF32>(
    const detail::FDMTInputF32& input, cudaStream_t stream);
template void FDMTCudaEngine::launch_direct<detail::FDMTInputPacked>(
    const detail::FDMTInputPacked& input, cudaStream_t stream);
template void FDMTCudaEngine::advance_direct_window<detail::FDMTInputF32>(
    const detail::FDMTInputF32& input, cudaStream_t stream);
template void FDMTCudaEngine::advance_direct_window<detail::FDMTInputPacked>(
    const detail::FDMTInputPacked& input, cudaStream_t stream);

} // namespace dmt::algorithms::detail::fdmt_cuda
