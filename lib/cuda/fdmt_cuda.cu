#include "fdmt_cuda_engine.cuh"

// CUDA FDMT engine: construction, path and memory planning.

namespace dmt::algorithms::detail::fdmt_cuda {

FDMTCudaEngine::FDMTCudaEngine(const plans::FDMTPlan& plan,
                               const detail::FDMTEngineConfig& cfg)
    : m_device_id(cfg.exec.device),
      m_use_box_smearing(cfg.use_box_smearing),
      m_int_tree(cfg.int_tree),
      m_mode(cfg.mode),
      m_nbeams(cfg.nbeams),
      m_plan(plan) {
    if (m_nbeams == 0) {
        throw std::invalid_argument("FDMT (cuda): nbeams must be at least 1");
    }
    m_search_nbits = cfg.search_nbits;
    m_search_u32 =
        detail::int_search_fits_u32(m_plan, m_use_box_smearing, m_search_nbits);
    // Staggered subbands cross fused-group boundaries: fuse (and compute
    // direct levels) only below them.
    m_search_max_fuse =
        std::min(cfg.max_fuse_levels, m_plan.get_max_fuse_levels());
    m_search_upper = cfg.search_upper_levels;
    m_max_memory   = cfg.max_memory_bytes;
    m_force_skewed = cfg.force_skewed_direct;
    m_chunk = detail::fdmt_chunk_nsamps(m_plan, m_mode, m_nbeams,
                                        cfg.chunk_nsamps, m_search_upper);
    // Automatic chunk: L2-resident state when it fits, else the budget
    // rule (halved only when the packed search needs uint64 state).
    const SizeType elem = (m_search_nbits == 32 || m_search_u32)
                              ? sizeof(uint32_t)
                              : sizeof(uint64_t);
    const SizeType cached =
        (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0)
            ? detail::fdmt_cache_chunk(m_plan, m_nbeams, elem, l2_bytes(),
                                       m_search_upper)
            : 0;
    if (cached != 0) {
        m_chunk = cached;
    } else if (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0 &&
               (m_plan.get_history_size() + m_plan.get_tree_history_size()) *
                       sizeof(float) >
                   detail::kFDMTChunkStateBudget) {
        m_chunk = std::min<SizeType>(1024, m_plan.get_nsamps());
    } else if (elem == sizeof(uint64_t) && cfg.chunk_nsamps == 0) {
        m_chunk =
            std::min(m_plan.get_nsamps(), std::max<SizeType>(64, m_chunk / 2));
    }
    m_requested_fuse = cfg.fuse_levels;
    init_device_structures(m_search_nbits == 32 ? cfg.fuse_levels : 0);
    if (m_mode == FDMTMode::kValid && !skew_history()) {
        m_hist_hash = detail::fdmt_tree_history_hash(m_plan, m_nbeams,
                                                     m_use_box_smearing);
    }
}

void FDMTCudaEngine::init_device_structures(SizeType fuse_levels) {
    // Chunked execution (fdmt_chunk.hpp): the device tables describe rows
    // of the chunk length below the root (the root keeps the dmt
    // output's layout), so the kernels run unchanged on every chunk.
    const SizeType niters = m_plan.get_niters();
    const bool chunked    = m_chunk < m_plan.get_nsamps();
    plans::FDMTPlanContainer relaid =
        chunked ? detail::fdmt_relayout_container(m_plan.get_container(),
                                                  niters, m_chunk)
                : plans::FDMTPlanContainer{};
    gpu_utils::set_device(m_device_id);
    if (m_use_box_smearing) {
        // Level-0 box means (fdmt_box_weights.hpp), by |dt|: dyadic float
        // weights, and the packed-input rounding recipes on 2^-F.
        const auto weights = m_plan.get_box_weights();
        const auto box_max = weights.size();
        const auto scale_f = m_plan.get_box_mean_bits();
        const float w_unit =
            std::ldexp(1.0F, -static_cast<int>(m_plan.get_box_fraction_bits()));
        m_int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
        std::vector<float> box_w(box_max);
        std::vector<detail::FDMTBoxRound> box_round(box_max);
        for (SizeType s = 0; s < box_max; ++s) {
            box_w[s]     = static_cast<float>(weights[s]) * w_unit;
            box_round[s] = m_search_nbits == 32
                               ? detail::box_round(s + 1, scale_f)
                               : detail::FDMTBoxRound{.mul    = weights[s],
                                                      .add    = 0,
                                                      .rshift = 0,
                                                      .lshift = 0};
        }
        m_box_w_d     = box_w;
        m_box_round_d = box_round;
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            // The rounding product's width is independent of the u32/u64
            // search state: it must hold S * mul + add (box_round_max()).
            m_box_round_u64[nbits] = !detail::box_round_fits_u32(
                bit_pack_utils::max_sample_value(nbits), box_max);
        }
    }

    // Precompute cumulative coordinate offsets per iteration
    const auto niters_plus_one = m_plan.get_niters() + 1;
    m_coords_sum_offsets.resize(niters_plus_one, 0);
    m_coords_copy_offsets.resize(niters_plus_one, 0);
    int sum_off  = 0;
    int copy_off = 0;
    for (size_t i = 0; i < niters_plus_one; ++i) {
        m_coords_sum_offsets[i]  = sum_off;
        m_coords_copy_offsets[i] = copy_off;
        sum_off +=
            static_cast<int>(m_plan.get_container().state_shape[i].ncoords_sum);
        copy_off += static_cast<int>(
            m_plan.get_container().state_shape[i].ncoords_copy);
    }

    m_float_levels.assign(m_plan.get_niters() + 1, Elem::kF32);
    for (const SizeType nbits : {1, 2, 4, 8, 16}) {
        m_int_levels[nbits] =
            detail::int_tree_level_types(m_plan, m_use_box_smearing, nbits);
    }
    m_levels_d.resize(m_plan.get_niters() + 1);
    const auto& pc_init = chunked ? relaid : m_plan.get_container();
    configure_fusion(pc_init, fuse_levels);
    configure_direct(fuse_levels);
    configure_top(relaid);
    if (m_direct && m_direct->skewed) {
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            m_int_levels[nbits].assign(niters + 1, Elem::kF32);
        }
    }
    const auto& pc =
        !relaid.state_shape.empty() ? relaid : m_plan.get_container();
    m_row_len0   = pc.state_shape[0].nsamps;
    m_half_elems = 0;
    for (SizeType l = 0; l < niters; ++l) {
        if (m_direct && m_direct->skewed && l < m_direct->levels) {
            continue;
        }
        if (m_direct && m_direct->skewed && m_top && m_top->skewed &&
            m_top->target == niters &&
            (m_top->windowed ? l == m_top->level : l > m_top->level)) {
            continue;
        }
        const bool skew_level = m_direct && m_direct->skewed && m_top &&
                                (l <= m_top->level || m_top->windowed);
        const SizeType elems =
            skew_level ? pc.state_shape[l].nelements
                       : (m_search_upper ? detail::fdmt_regular_coords(pc, l)
                                         : pc.state_shape[l].ncoords) *
                             pc.state_shape[l].nsamps;
        m_half_elems = std::max(m_half_elems, elems);
    }
    check_int32_extents();
    // The two state halves: level l < niters lives in half l % 2; the
    // root is written straight into the caller's dmt buffer.
    m_state_internal_d.resize(((m_search_nbits == 32 || m_search_u32) ? 2 : 4) *
                                  m_nbeams * m_half_elems,
                              0.0F);
    if (m_mode == FDMTMode::kValid && !((m_direct && m_direct->skewed) ||
                                        (m_search_nbits != 32 && m_direct))) {
        const auto hist_size = m_nbeams * m_plan.get_history_size();
        m_history_a_d.resize(hist_size, 0.0F);
        m_history_b_d.resize(hist_size, 0.0F);
    }
    plans::transfer_fdmt_plan_to_device(pc, m_plan_d);
    if (m_mode == FDMTMode::kValid) {
        // FIFOs for the fused levels, a ring above (see m_tree_fifo_d).
        // Only active tree ring levels (outside m_direct and m_top) are
        // allocated; save_history()/load_history() expand on demand.
        const auto tree = m_plan.get_tree_history_size();
        m_fifo_levels   = (m_search_nbits == 32 && m_fused)
                              ? static_cast<SizeType>(m_fused->args.fuse)
                              : 0;
        m_fifo_size     = tree_history_through(m_fifo_levels);
        const SizeType low_first =
            m_direct ? (m_direct->levels + 1) : (m_fifo_levels + 1);
        const SizeType low_last = m_top ? m_top->level : niters;
        m_ring_low_base         = tree_history_through(low_first - 1);
        m_ring_low_size =
            (m_direct && m_direct->skewed && m_top)
                ? 0
                : (low_last >= low_first
                       ? tree_history_through(low_last) - m_ring_low_base
                       : 0);
        m_ring_high_orig_base = (m_top && m_top->target < niters)
                                    ? tree_history_through(m_top->target)
                                    : tree;
        m_ring_high_size      = tree - m_ring_high_orig_base;
        m_ring_size           = m_ring_low_size + m_ring_high_size;
        m_full_tree_ring      = (m_ring_size == tree - m_fifo_size);
        check_tree_ring_extent(m_ring_high_size > 0 ? tree
                               : m_ring_low_size > 0
                                   ? m_ring_low_base + m_ring_low_size
                                   : m_fifo_size);
        if (m_search_nbits == 32) {
            for (auto& fifo : m_tree_fifo_d) {
                fifo.resize(m_nbeams * m_fifo_size, 0.0F);
            }
            m_tree_ring_d.resize(m_nbeams * m_ring_size, 0.0F);
        } else if (m_search_u32) {
            m_integer32_ring.resize(m_nbeams * m_ring_size, 0);
        } else {
            m_integer_ring.resize(m_nbeams * m_ring_size, 0);
        }
    }
    gpu_utils::check_last_gpu_error("FDMT (gpu): constructor failed");
}

std::string FDMTCudaEngine::summary() const {
    const auto mem = get_memory_usage();
    const auto mib = [](SizeType b) {
        return static_cast<double>(b) / 1048576.0;
    };
    const auto summary = fmt::format(
        "*** FDMT (cuda) ***\n"
        "Mode: {}, device {}, nbeams {}, chunk {} of {} samples, "
        "fuse_levels {}, int_tree {}\n"
        "Device memory: plan {:.1f} MiB, state {:.1f} MiB, history {:.1f} "
        "MiB, workspace {:.1f} MiB (total {:.1f} MiB); output buffer "
        "{:.1f} MiB "
        "per execute()\n",
        fdmt_mode_to_string(m_mode), m_device_id, m_nbeams, m_chunk,
        m_plan.get_nsamps(), get_fuse_levels(), m_int_tree, mib(mem.plan),
        mib(mem.state), mib(mem.history), mib(mem.workspace),
        mib(mem.plan + mem.state + mem.history + mem.workspace),
        mib(mem.output));
    return m_plan.summary() + summary;
}

std::string FDMTCudaEngine::search_path() const {
    std::string path;
    if (m_direct) {
        path = fmt::format("{}direct levels 0-{}",
                           m_direct->skewed ? "skewed " : "", m_direct->levels);
    }
    if (m_top && m_top->windowed) {
        path += fmt::format("{}batched mid levels {}-{}, windowed top "
                            "{} -> {}",
                            path.empty() ? "" : ", ",
                            m_direct ? m_direct->levels : 0, m_top->level,
                            m_top->level, m_top->target);
    } else if (m_top) {
        path += fmt::format("{}direct top {} -> {}", path.empty() ? "" : ", ",
                            m_top->level, m_top->target);
    }
    return path;
}

FDMTMemoryUsage FDMTCudaEngine::get_memory_usage() const noexcept {
    const auto coord_bytes = [](const plans::FDMTCoordD& c) {
        return (c.nsamps.size() + c.buf_offset.size() + c.offset.size() +
                c.tail_buf_offset.size() + c.tail_nsamps.size() +
                c.head_buf_offset.size() + c.head_nsamps.size() +
                c.hist_offset.size()) *
               sizeof(int);
    };
    const auto& sh = m_plan_d.state_shape;
    const auto& g0 = m_plan_d.grids0;
    const SizeType plan_bytes =
        coord_bytes(m_plan_d.coordinates_sum) +
        coord_bytes(m_plan_d.coordinates_copy) +
        ((sh.nchans.size() + sh.ncoords_sum.size() + sh.ncoords_copy.size() +
          sh.nsamps.size() + sh.dt_max.size() + g0.dt_grid.size() +
          g0.ndt.size() + g0.coord_offset.size()) *
         sizeof(int));
    SizeType fused_bytes = 0;
    if (m_fused) {
        fused_bytes = (m_fused->group_info.size() + m_fused->coords.size() +
                       m_fused->dt_grid0.size() + m_fused->ndt0.size() +
                       m_fused->coord_offset0.size()) *
                      sizeof(int);
    }
    return {
        .plan    = plan_bytes,
        .state   = m_state_internal_d.size() * sizeof(float),
        .history = (m_history_a_d.size() + m_history_b_d.size() +
                    m_tree_fifo_d[0].size() + m_tree_fifo_d[1].size() +
                    m_tree_ring_d.size() + (m_top ? m_top->ring.size() : 0)) *
                       sizeof(float) +
                   windowed_top_bytes() + direct_window_bytes() +
                   m_integer_ring.size() * sizeof(uint64_t) +
                   m_integer32_ring.size() * sizeof(uint32_t),
        // Fused or direct tables, plus the host-execute() staging once
        // allocated.
        .workspace =
            fused_bytes + direct_workspace_bytes() + top_workspace_bytes() +
            (m_stage_wf_f32_d.size() * sizeof(float)) +
            m_stage_wf_packed_d.size() + (m_stage_dmt_d.size() * sizeof(float)),
        .output = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
    };
}

void FDMTCudaEngine::check_int32_extents() const {
    constexpr auto kMax =
        static_cast<SizeType>(std::numeric_limits<int32_t>::max());
    const auto check = [&](SizeType value, std::string_view what) {
        if (value > kMax) {
            throw std::invalid_argument(fmt::format(
                "FDMT (cuda): per-beam {} ({} elements) exceeds the 32-bit "
                "index range of the CUDA backend ({}); use a smaller "
                "chunk_nsamps or block (nsamps), or fewer DM trials",
                what, value, kMax));
        }
    };
    check(m_half_elems, "state level");
    check(m_plan.get_dmt_size(), "output");
    check(m_plan.get_nchans() * m_plan.get_nsamps(), "waterfall");
    check(m_plan.get_history_size(), "level-0 history");
}

void FDMTCudaEngine::check_tree_ring_extent(SizeType ring_elems) {
    if (ring_elems >
        static_cast<SizeType>(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument(fmt::format(
            "FDMT (cuda): per-beam tree history ({} elements) exceeds the "
            "unsigned 32-bit index range of the CUDA backend ({}); use "
            "fewer DM trials or a lower maximum DM",
            ring_elems, std::numeric_limits<uint32_t>::max()));
    }
}

SizeType FDMTCudaEngine::max_optin_smem_bytes() const {
    int bytes = 0;
    if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               m_device_id) != cudaSuccess ||
        bytes <= 0) {
        return kDefaultSmemBytes;
    }
    return static_cast<SizeType>(bytes);
}

SizeType FDMTCudaEngine::l2_bytes() const {
    int bytes = 0;
    if (cudaDeviceGetAttribute(&bytes, cudaDevAttrL2CacheSize, m_device_id) !=
            cudaSuccess ||
        bytes <= 0) {
        return 0;
    }
    return static_cast<SizeType>(bytes);
}

SizeType FDMTCudaEngine::auto_smem_bytes() const {
    constexpr SizeType kReserved = 1024;
    int per_sm                   = 0;
    if (cudaDeviceGetAttribute(&per_sm,
                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                               m_device_id) != cudaSuccess ||
        per_sm <= 0) {
        return kDefaultSmemBytes;
    }
    const auto share = static_cast<SizeType>(per_sm) / kAutoBlocksPerSM;
    return std::min(kDefaultSmemBytes,
                    share > kReserved ? share - kReserved : share);
}

void FDMTCudaEngine::configure_fusion(const plans::FDMTPlanContainer& pc,
                                      SizeType requested) {
    m_fused.reset();
    const SizeType niters  = m_plan.get_niters();
    const SizeType deepest = std::min(
        {niters, SizeType{detail::kMaxFusedLevels}, m_search_max_fuse});
    if (requested == 0 || deepest == 0) {
        return;
    }
    const bool automatic  = requested == kFDMTAutoFuse;
    const SizeType budget = auto_smem_bytes();
    const SizeType optin  = automatic ? budget : max_optin_smem_bytes();
    for (SizeType f = automatic ? deepest : std::min(requested, deepest);
         f >= 1; --f) {
        const auto plan = detail::build_fused_tile_plan(pc, niters, f);
        // Never wider than the (32-rounded) level-F row itself.
        const int max_tile =
            std::min(kMaxTile, ((plan.ntiles_nsamps + 31) / 32) * 32);
        int tile = plan.max_tile_nsamps(budget, max_tile);
        if (automatic) {
            const int min_tile = std::min(kMinAutoTile, max_tile);
            if (tile < min_tile || 2 * plan.max_level0_halo > tile) {
                continue;
            }
        } else if (tile < 64) {
            tile = std::max(tile, plan.max_tile_nsamps(optin, max_tile));
        }
        if (tile > 0) {
            adopt_fused_plan(plan, tile);
            return;
        }
    }
}

void FDMTCudaEngine::adopt_fused_plan(const detail::FDMTFusedTilePlan& plan,
                                      int tile) {
    gpu_utils::set_device(m_device_id);
    auto fused           = std::make_unique<FusedLevelsD>();
    fused->group_info    = plan.group_info;
    fused->coords        = plan.coords;
    fused->dt_grid0      = plan.dt_grid0;
    fused->ndt0          = plan.ndt0;
    fused->coord_offset0 = plan.coord_offset0;
    const auto hist_len =
        static_cast<int>(m_plan.get_container().state_shape[0].dt_max);
    fused->args = detail::make_fused_tile_args(
        plan, tile, hist_len,
        thrust::raw_pointer_cast(fused->group_info.data()),
        thrust::raw_pointer_cast(fused->coords.data()),
        thrust::raw_pointer_cast(fused->dt_grid0.data()),
        thrust::raw_pointer_cast(fused->ndt0.data()),
        thrust::raw_pointer_cast(fused->coord_offset0.data()), box_means(0));
    const auto [cap_a, cap_b] = plan.smem_floats(tile);
    fused->smem_bytes         = (cap_a + cap_b) * sizeof(float);
    fused->tile               = tile;
    fused->ntiles             = (plan.ntiles_nsamps + tile - 1) / tile;
    fused->ngroups            = plan.ngroups;
    gpu_utils::check_last_gpu_error("FDMT (cuda): fused plan upload failed");
    logging::debug("FDMT (cuda): fusing levels 0..{} (tile {} samples, {} "
                   "groups, {} B shared)",
                   plan.fuse, tile, plan.ngroups, fused->smem_bytes);
    m_fused = std::move(fused);
}

void FDMTCudaEngine::configure_skewed_direct_and_mid(
    plans::FDMTPlanContainer& relaid,
    detail::FDMTDirectTopPlan& top_plan,
    const detail::FDMTSkewWindows* win,
    const detail::FDMTSkewWindows* top_win) {
    if (relaid.state_shape.empty()) {
        relaid = detail::fdmt_relayout_container(m_plan.get_container(),
                                                 m_plan.get_niters(), m_chunk);
    }
    const auto& pc       = m_plan.get_container();
    const SizeType k     = kDirectLevels;
    const SizeType top_k = top_plan.level;
    std::vector<std::vector<int32_t>> s_lo(top_k + 1);
    std::vector<std::vector<int32_t>> s_hi(top_k + 1);
    for (SizeType l = k; l <= top_k; ++l) {
        s_lo[l].assign(pc.state_shape[l].ncoords, INT32_MAX);
        s_hi[l].assign(pc.state_shape[l].ncoords, -1);
    }
    if (win != nullptr) {
        for (SizeType l = k; l <= top_k; ++l) {
            for (SizeType r = 0; r < pc.state_shape[l].ncoords; ++r) {
                if (win->used(l, r)) {
                    s_lo[l][r] = static_cast<int32_t>(win->s_lo[l][r]);
                    s_hi[l][r] = static_cast<int32_t>(win->s_hi[l][r]);
                }
            }
        }
    } else {
        for (SizeType r = 0; r < pc.state_shape[top_k].ncoords; ++r) {
            if (top_plan.row_s_hi[r] >= 0) {
                s_lo[top_k][r] = top_plan.row_s_lo[r];
                s_hi[top_k][r] = top_plan.row_s_lo[r];
            }
        }
    }
    for (SizeType l = top_k; l > k && win == nullptr; --l) {
        const auto in_stride  = pc.state_shape[l - 1].nsamps;
        const auto out_stride = pc.state_shape[l].nsamps;
        for (const auto& c : pc.coordinates_copy[l]) {
            const auto p = c.buf_offset / out_stride;
            const auto t = c.tail_buf_offset / in_stride;
            if (s_hi[l][p] < 0) {
                continue;
            }
            s_lo[l - 1][t] = std::min(s_lo[l - 1][t], s_lo[l][p]);
            s_hi[l - 1][t] = std::max(s_hi[l - 1][t], s_hi[l][p]);
        }
        for (const auto& c : pc.coordinates_sum[l]) {
            const auto p = c.buf_offset / out_stride;
            const auto t = c.tail_buf_offset / in_stride;
            const auto h = c.head_buf_offset / in_stride;
            const auto d = static_cast<int32_t>(c.delay);
            if (s_hi[l][p] < 0) {
                continue;
            }
            s_lo[l - 1][t] = std::min(s_lo[l - 1][t], s_lo[l][p]);
            s_hi[l - 1][t] = std::max(s_hi[l - 1][t], s_hi[l][p]);
            s_lo[l - 1][h] = std::min(s_lo[l - 1][h], s_lo[l][p] + d);
            s_hi[l - 1][h] = std::max(s_hi[l - 1][h], s_hi[l][p] + d);
        }
    }
    m_level_max_extra.assign(m_plan.get_niters() + 1, 0);
    std::vector<std::vector<int32_t>> sp(top_k + 1);
    for (SizeType l = k; l <= top_k; ++l) {
        sp[l].assign(pc.state_shape[l].ncoords, 0);
        for (SizeType r = 0; r < pc.state_shape[l].ncoords; ++r) {
            if (s_hi[l][r] >= 0) {
                sp[l][r]             = s_hi[l][r] - s_lo[l][r];
                m_level_max_extra[l] = std::max(
                    m_level_max_extra[l], static_cast<SizeType>(sp[l][r]));
            }
        }
    }
    const SizeType ngrids_top = pc.grids[top_k].size();
    for (SizeType f = 0; f < top_plan.out_row.size(); ++f) {
        std::ranges::stable_sort(
            top_plan.leaves.begin() + top_plan.leaf_off[f],
            top_plan.leaves.begin() + top_plan.leaf_off[f + 1], {},
            [](const detail::FDMTDirectTopLeaf& lf) { return lf.row; });
    }
    SizeType nbatches =
        ((win != nullptr || (!m_search_upper && top_plan.target_ring)) &&
         ngrids_top >= 24)
            ? 24
            : 1;
    std::vector<std::vector<SizeType>> batch_grid(top_k + 1);
    std::vector<std::vector<SizeType>> batch_coord(top_k + 1);
    const auto build_batch_bounds = [&](SizeType nb) {
        std::vector<SizeType> top_bounds(nb + 1, 0);
        top_bounds[nb] = ngrids_top;
        if (nb > 1) {
            const SizeType ngrids_k = pc.grids[k].size();
            std::vector<SizeType> cum_w(ngrids_top + 1, 0);
            for (SizeType g = 0; g < ngrids_top; ++g) {
                const SizeType gk0 = std::min(ngrids_k, g << (top_k - k));
                const SizeType gk1 = std::min(ngrids_k, (g + 1) << (top_k - k));
                const SizeType c0  = gk0 < ngrids_k
                                         ? pc.grids[k][gk0].coord_offset
                                         : pc.state_shape[k].ncoords;
                const SizeType c1  = gk1 < ngrids_k
                                         ? pc.grids[k][gk1].coord_offset
                                         : pc.state_shape[k].ncoords;
                SizeType w         = 1;
                for (SizeType r = c0; r < c1; ++r) {
                    if (s_hi[k][r] >= 0) {
                        w += m_chunk + static_cast<SizeType>(sp[k][r]);
                    }
                }
                cum_w[g + 1] = cum_w[g] + w;
            }
            const SizeType total_w = cum_w[ngrids_top];
            for (SizeType b = 1; b < nb; ++b) {
                const SizeType target_w = (b * total_w) / nb;
                SizeType g              = top_bounds[b - 1] + 1;
                const SizeType max_g    = ngrids_top - (nb - b);
                while (g < max_g && cum_w[g] < target_w) {
                    ++g;
                }
                top_bounds[b] = g;
            }
        }
        for (SizeType l = k; l <= top_k; ++l) {
            batch_grid[l].resize(nb + 1, 0);
            batch_coord[l].resize(nb + 1, 0);
            const SizeType ngrids_l = pc.grids[l].size();
            for (SizeType b = 0; b <= nb; ++b) {
                const SizeType g_top = top_bounds[b];
                const SizeType g_l =
                    (b == nb) ? ngrids_l
                              : std::min(ngrids_l, g_top << (top_k - l));
                batch_grid[l][b]  = g_l;
                batch_coord[l][b] = (g_l < ngrids_l)
                                        ? pc.grids[l][g_l].coord_offset
                                        : pc.state_shape[l].ncoords;
            }
        }
    };
    build_batch_bounds(nbatches);
    if (nbatches > 1) {
        bool valid_batches = true;
        for (SizeType f = 0; f < top_plan.out_row.size() && valid_batches;
             ++f) {
            const auto f0 = static_cast<SizeType>(top_plan.leaf_off[f]);
            const auto f1 = static_cast<SizeType>(top_plan.leaf_off[f + 1]);
            if (f1 - f0 != ngrids_top) {
                valid_batches = false;
                break;
            }
            for (SizeType b = 0; b < nbatches && valid_batches; ++b) {
                const auto c_lo = batch_coord[top_k][b];
                const auto c_hi = batch_coord[top_k][b + 1];
                for (SizeType m = batch_grid[top_k][b];
                     m < batch_grid[top_k][b + 1]; ++m) {
                    const auto r =
                        static_cast<SizeType>(top_plan.leaves[f0 + m].row);
                    if (r < c_lo || r >= c_hi) {
                        valid_batches = false;
                        break;
                    }
                }
            }
        }
        for (SizeType l = k + 1; l <= top_k && valid_batches; ++l) {
            const SizeType orig_in  = pc.state_shape[l - 1].nsamps;
            const SizeType orig_out = pc.state_shape[l].nsamps;
            SizeType b_cur          = 0;
            for (const auto& o : pc.coordinates_copy[l]) {
                const auto p = o.buf_offset / orig_out;
                const auto t = o.tail_buf_offset / orig_in;
                if (p < batch_coord[l][b_cur]) {
                    valid_batches = false;
                    break;
                }
                while (b_cur < nbatches && p >= batch_coord[l][b_cur + 1]) {
                    ++b_cur;
                }
                if (b_cur >= nbatches || t < batch_coord[l - 1][b_cur] ||
                    t >= batch_coord[l - 1][b_cur + 1]) {
                    valid_batches = false;
                    break;
                }
            }
            b_cur = 0;
            for (const auto& o : pc.coordinates_sum[l]) {
                const auto p = o.buf_offset / orig_out;
                const auto t = o.tail_buf_offset / orig_in;
                const auto h = o.head_buf_offset / orig_in;
                if (p < batch_coord[l][b_cur]) {
                    valid_batches = false;
                    break;
                }
                while (b_cur < nbatches && p >= batch_coord[l][b_cur + 1]) {
                    ++b_cur;
                }
                if (b_cur >= nbatches || t < batch_coord[l - 1][b_cur] ||
                    t >= batch_coord[l - 1][b_cur + 1] ||
                    h < batch_coord[l - 1][b_cur] ||
                    h >= batch_coord[l - 1][b_cur + 1]) {
                    valid_batches = false;
                    break;
                }
            }
        }
        if (!valid_batches) {
            nbatches = 1;
            build_batch_bounds(1);
        }
    }
    build_skewed_direct_from_windows(s_hi[k], sp[k], batch_grid[k]);
    if (!m_direct || !m_direct->skewed) {
        m_level_max_extra.clear();
        return;
    }
    m_skew_nbatches = nbatches;
    m_skew_batch_top_grid.resize(nbatches + 1, 0);
    for (SizeType b = 0; b <= nbatches; ++b) {
        m_skew_batch_top_grid[b] = static_cast<int>(batch_grid[top_k][b]);
    }
    m_skew_batch_sum_off.assign(top_k + 1, {});
    m_skew_batch_copy_off.assign(top_k + 1, {});
    relaid.state_shape[k].nsamps    = m_chunk + m_level_max_extra[k];
    relaid.state_shape[k].nelements = m_direct->total_elems;
    std::vector<std::vector<SizeType>> row_off(top_k + 1);
    row_off[k].resize(m_direct->row_off.size());
    for (SizeType r = 0; r < m_direct->row_off.size(); ++r) {
        row_off[k][r] = static_cast<SizeType>(m_direct->row_off[r]);
    }
    for (SizeType l = k + 1; l <= top_k; ++l) {
        const SizeType orig_in  = pc.state_shape[l - 1].nsamps;
        const SizeType orig_out = pc.state_shape[l].nsamps;
        const SizeType ncoords  = pc.state_shape[l].ncoords;
        row_off[l].resize(ncoords, 0);
        if (l == top_k && win != nullptr) {
            // Whole level in its own buffer (DirectTopD::win_top): each
            // row's full window (`top_win`), the batches writing its
            // newest samples after the carried spread.
            SizeType off = 0;
            for (SizeType p = 0; p < ncoords; ++p) {
                const SizeType full =
                    top_win->used(l, p) ? top_win->spread(l, p) : 0;
                row_off[l][p] = off + full;
                if (s_hi[l][p] >= 0) {
                    off += m_chunk + full;
                }
            }
            relaid.state_shape[l].nsamps    = m_chunk + m_level_max_extra[l];
            relaid.state_shape[l].nelements = off;
        } else if (l == top_k) {
            SizeType max_used_rows = 0;
            std::vector<int32_t> remap(ncoords, -1);
            for (SizeType b = 0; b < nbatches; ++b) {
                SizeType used_rows = 0;
                for (SizeType p = batch_coord[l][b]; p < batch_coord[l][b + 1];
                     ++p) {
                    row_off[l][p] = used_rows * m_chunk;
                    if (s_hi[l][p] >= 0) {
                        remap[p] = static_cast<int32_t>(used_rows);
                        ++used_rows;
                    }
                }
                max_used_rows = std::max(max_used_rows, used_rows);
            }
            for (auto& lf : top_plan.leaves) {
                lf.row = remap[static_cast<SizeType>(lf.row)];
            }
            if (!top_plan.target_ring) {
                std::vector<int32_t> new_ring_len(max_used_rows, 0);
                std::vector<int64_t> new_ring_off(max_used_rows, 0);
                for (SizeType p = 0; p < ncoords; ++p) {
                    if (remap[p] >= 0) {
                        const auto u    = static_cast<SizeType>(remap[p]);
                        new_ring_len[u] = top_plan.ring_len[p];
                        new_ring_off[u] = top_plan.ring_off[p];
                    }
                }
                top_plan.ring_len = std::move(new_ring_len);
                top_plan.ring_off = std::move(new_ring_off);
            }
            relaid.state_shape[l].nsamps    = m_chunk;
            relaid.state_shape[l].nelements = max_used_rows * m_chunk;
        } else {
            SizeType max_off = 0;
            for (SizeType b = 0; b < nbatches; ++b) {
                SizeType off = 0;
                for (SizeType p = batch_coord[l][b]; p < batch_coord[l][b + 1];
                     ++p) {
                    row_off[l][p] = off;
                    if (s_hi[l][p] >= 0) {
                        off += m_chunk + static_cast<SizeType>(sp[l][p]);
                    }
                }
                max_off = std::max(max_off, off);
            }
            relaid.state_shape[l].nsamps    = m_chunk + m_level_max_extra[l];
            relaid.state_shape[l].nelements = max_off;
        }
        m_skew_batch_copy_off[l].resize(nbatches + 1, 0);
        m_skew_batch_sum_off[l].resize(nbatches + 1, 0);
        SizeType i_copy = 0;
        SizeType i_sum  = 0;
        for (SizeType b = 0; b <= nbatches; ++b) {
            const SizeType c_lim = batch_coord[l][b];
            while (i_copy < pc.coordinates_copy[l].size() &&
                   (pc.coordinates_copy[l][i_copy].buf_offset / orig_out) <
                       c_lim) {
                ++i_copy;
            }
            m_skew_batch_copy_off[l][b] = static_cast<int>(i_copy);
            while (i_sum < pc.coordinates_sum[l].size() &&
                   (pc.coordinates_sum[l][i_sum].buf_offset / orig_out) <
                       c_lim) {
                ++i_sum;
            }
            m_skew_batch_sum_off[l][b] = static_cast<int>(i_sum);
        }
        relaid.state_shape[l].dt_max = 0;
        for (SizeType i = 0; i < relaid.coordinates_copy[l].size(); ++i) {
            auto& c                = relaid.coordinates_copy[l][i];
            const auto& o          = pc.coordinates_copy[l][i];
            const auto p           = o.buf_offset / orig_out;
            const auto t           = o.tail_buf_offset / orig_in;
            const bool used        = s_hi[l][p] >= 0;
            const SizeType extra_p = used ? static_cast<SizeType>(sp[l][p]) : 0;
            const SizeType shift_t =
                (used && s_hi[l - 1][t] >= s_hi[l][p])
                    ? static_cast<SizeType>(s_hi[l - 1][t] - s_hi[l][p])
                    : 0;
            c.nsamps          = used ? (m_chunk + extra_p) : 0;
            c.buf_offset      = row_off[l][p];
            c.tail_buf_offset = row_off[l - 1][t] + shift_t;
            c.tail_nsamps     = c.nsamps;
        }
        for (SizeType i = 0; i < relaid.coordinates_sum[l].size(); ++i) {
            auto& c                = relaid.coordinates_sum[l][i];
            const auto& o          = pc.coordinates_sum[l][i];
            const auto p           = o.buf_offset / orig_out;
            const auto t           = o.tail_buf_offset / orig_in;
            const auto h           = o.head_buf_offset / orig_in;
            const auto d           = static_cast<int32_t>(o.delay);
            const bool used        = s_hi[l][p] >= 0;
            const SizeType extra_p = used ? static_cast<SizeType>(sp[l][p]) : 0;
            const SizeType shift_t =
                (used && s_hi[l - 1][t] >= s_hi[l][p])
                    ? static_cast<SizeType>(s_hi[l - 1][t] - s_hi[l][p])
                    : 0;
            const SizeType shift_h =
                (used && s_hi[l - 1][h] >= s_hi[l][p] + d)
                    ? static_cast<SizeType>(s_hi[l - 1][h] - (s_hi[l][p] + d))
                    : 0;
            c.nsamps          = used ? (m_chunk + extra_p) : 0;
            c.buf_offset      = row_off[l][p];
            c.tail_buf_offset = row_off[l - 1][t] + shift_t;
            c.tail_nsamps     = c.nsamps;
            c.head_buf_offset = row_off[l - 1][h] + shift_h;
            c.head_nsamps     = c.nsamps;
            c.delay           = 0;
        }
    }
}

bool FDMTCudaEngine::configure_windowed_top(plans::FDMTPlanContainer& relaid,
                                            SizeType top_k) {
    const auto& pc           = m_plan.get_container();
    const SizeType root      = m_plan.get_niters();
    const SizeType k         = kDirectLevels;
    const SizeType sub_depth = m_plan.get_subband_depth();
    const auto win = detail::compute_skew_windows(m_plan, k, sub_depth);
    // Levels k..top_k compute only level top_k's n newest samples
    // (local windows); each level-top_k row carries the rest of its
    // window, spread(top_k, r) samples, over from the previous chunk
    // (as FDMTSkewCpu's level-F ring).
    const auto local =
        detail::compute_local_skew_windows(m_plan, win, k, top_k);
    detail::FDMTDirectTopPlan mid;
    mid.level  = top_k;
    mid.skewed = true;
    configure_skewed_direct_and_mid(relaid, mid, &local, &win);
    if (!m_direct || !m_direct->skewed) {
        m_level_max_extra.clear();
        return false;
    }
    const auto spread = [&](SizeType l, SizeType r) {
        return win.used(l, r) ? win.s_hi[l][r] - win.s_lo[l][r] : SizeType{0};
    };
    // Row offsets of levels top_k .. root: whole levels; the root's rows
    // m_chunk apart (its trials buffer).
    std::vector<std::vector<SizeType>> row_off(root + 1);
    for (SizeType l = top_k; l <= root; ++l) {
        const SizeType n = pc.state_shape[l].ncoords;
        row_off[l].assign(n, 0);
        SizeType off   = 0;
        SizeType extra = 0;
        for (SizeType r = 0; r < n; ++r) {
            if (l == root) {
                row_off[l][r] = r * m_chunk;
                continue;
            }
            row_off[l][r] = off;
            if (win.used(l, r)) {
                off += m_chunk + spread(l, r);
                extra = std::max(extra, spread(l, r));
            }
        }
        if (l > top_k) {
            m_level_max_extra[l]            = l == root ? 0 : extra;
            relaid.state_shape[l].nsamps    = m_chunk + m_level_max_extra[l];
            relaid.state_shape[l].nelements = l == root ? n * m_chunk : off;
            relaid.state_shape[l].dt_max    = 0;
        }
    }
    for (SizeType l = top_k + 1; l <= root; ++l) {
        const SizeType orig_in  = pc.state_shape[l - 1].nsamps;
        const SizeType orig_out = pc.state_shape[l].nsamps;
        const auto place = [&](plans::FDMTCoord& c, const plans::FDMTCoord& o,
                               bool sum) {
            const auto p    = o.buf_offset / orig_out;
            const bool used = win.used(l, p);
            const auto lo_p = used ? win.s_hi[l][p] : 0;
            const auto t    = o.tail_buf_offset / orig_in;
            c.nsamps        = used ? m_chunk + spread(l, p) : 0;
            c.buf_offset    = row_off[l][p];
            c.tail_buf_offset =
                row_off[l - 1][t] + (used ? win.s_hi[l - 1][t] - lo_p : 0);
            c.tail_nsamps = c.nsamps;
            if (sum) {
                const auto h = o.head_buf_offset / orig_in;
                c.head_buf_offset =
                    row_off[l - 1][h] +
                    (used ? win.s_hi[l - 1][h] - lo_p - o.delay : 0);
                c.head_nsamps = c.nsamps;
            }
            c.delay = 0;
        };
        for (SizeType i = 0; i < relaid.coordinates_copy[l].size(); ++i) {
            place(relaid.coordinates_copy[l][i], pc.coordinates_copy[l][i],
                  false);
        }
        for (SizeType i = 0; i < relaid.coordinates_sum[l].size(); ++i) {
            place(relaid.coordinates_sum[l][i], pc.coordinates_sum[l][i], true);
        }
    }
    auto t         = std::make_unique<DirectTopD>();
    t->level       = top_k;
    t->target      = root;
    t->skewed      = true;
    t->windowed    = true;
    t->target_ring = false;
    t->ntrials     = static_cast<int>(pc.state_shape[root].ncoords);
    t->hist_coords.resize(root + 1);
    t->hist_max_delay.assign(root + 1, 0);
    t->win_top.resize(relaid.state_shape[top_k].nelements * top_words(), 0.0F);
    t->trials.resize(pc.state_shape[root].ncoords * m_chunk * top_words(),
                     0.0F);
    {
        std::vector<SizeType> c_off;
        std::vector<int32_t> c_len;
        for (SizeType r = 0; r < pc.state_shape[top_k].ncoords; ++r) {
            if (win.used(top_k, r) && spread(top_k, r) > 0) {
                c_off.push_back(row_off[top_k][r]);
                c_len.push_back(static_cast<int32_t>(spread(top_k, r)));
            }
        }
        t->carry_off = c_off;
        t->carry_len = c_len;
    }
    t->emit_off.resize(root);
    t->emit_lag.resize(root);
    for (SizeType l = win.search_from; l < root; ++l) {
        const SizeType n = pc.state_shape[l].ncoords;
        std::vector<SizeType> off(n);
        std::vector<uint64_t> lag(n);
        for (SizeType r = 0; r < n; ++r) {
            lag[r]     = win.lag[l][r];
            off[r]     = row_off[l][r] + (win.s_hi[l][r] - win.lag[l][r]);
            t->max_lag = std::max<uint64_t>(t->max_lag, lag[r]);
        }
        t->emit_off[l] = off;
        t->emit_lag[l] = lag;
    }
    logging::debug("FDMT (cuda): windowed top from level {} (subband "
                   "depth {}, level {} {} MiB)",
                   top_k, sub_depth, top_k,
                   (t->win_top.size() * sizeof(float)) >> 20);
    m_top_filled = 0;
    m_top        = std::move(t);
    return true;
}

SizeType FDMTCudaEngine::windowed_top_bytes() const noexcept {
    if (!m_top || !m_top->windowed) {
        return 0;
    }
    SizeType b =
        (m_top->win_top.size() + m_top->trials.size()) * sizeof(float) +
        m_top->carry_off.size() * sizeof(SizeType) +
        m_top->carry_len.size() * sizeof(int32_t);
    for (SizeType l = 0; l < m_top->emit_off.size(); ++l) {
        b += m_top->emit_off[l].size() * sizeof(SizeType) +
             m_top->emit_lag[l].size() * sizeof(uint64_t);
    }
    return b;
}

void FDMTCudaEngine::configure_top(plans::FDMTPlanContainer& relaid) {
    m_top.reset();
    if (!m_direct && !m_direct_eligible) {
        return;
    }
    const auto& pc           = m_plan.get_container();
    const SizeType root      = m_plan.get_niters();
    const SizeType sub_depth = m_plan.get_subband_depth();
    const SizeType k_direct  = m_direct ? m_direct->levels : kDirectLevels;
    const auto fallback_skewed_direct = [&]() {
        if (!m_direct && m_direct_eligible) {
            const std::vector<int32_t> zeros(pc.state_shape[k_direct].ncoords,
                                             0);
            build_skewed_direct_from_windows(zeros, zeros);
        }
    };
    // Subband search of a plan too wide for the unskewed direct levels:
    // the windowed top (no tree history above the direct levels).
    // The tree levels' history (all levels: the subband search reads
    // them) over the budget.
    const bool big_history =
        (m_plan.get_history_size() + m_plan.get_tree_history_size()) *
            sizeof(float) * top_words() >
        m_max_memory;
    if (m_search_upper && m_nbeams == 1 && m_direct_eligible &&
        (big_history || m_force_skewed) && sub_depth + k_direct + 2 <= root) {
        const SizeType top_k =
            std::min(k_direct + (pc.state_shape[0].nchans > 4096 ? 3 : 2),
                     root - sub_depth - 1);

        if (configure_windowed_top(relaid, top_k)) {
            return;
        }
    }
    if (m_search_upper && sub_depth + 1 >= root) {
        fallback_skewed_direct();
        return;
    }
    const SizeType target  = m_search_upper ? (root - sub_depth - 1) : root;
    const SizeType ntrials = detail::fdmt_regular_coords(pc, target);
    const SizeType target_subs =
        std::max<SizeType>(1, pc.state_shape[target].nchans);
    SizeType level = k_direct + 1;
    while (level + 2 < target) {
        const SizeType leaves_saved =
            (ntrials * (pc.state_shape[level].nchans -
                        pc.state_shape[level + 1].nchans)) /
            target_subs;
        if (leaves_saved < 8 * pc.state_shape[level + 1].ncoords) {
            break;
        }
        ++level;
    }
    // When writing into m_levels_d[target], level and target must occupy
    // opposite ping-pong state halves (level % 2 != target % 2).
    if (target < root && (target - level) % 2 == 0) {
        ++level;
    }
    if (level + 1 >= target) {
        fallback_skewed_direct();
        return;
    }
    const SizeType orig_level        = level;
    constexpr SizeType kMaxRingBytes = SizeType{2} << 30;
    const SizeType step              = target < root ? 2 : 1;
    detail::FDMTDirectTopPlan plan;
    bool use_skewed = !m_direct || m_force_skewed;
    if (!use_skewed) {
        plan = detail::build_fdmt_direct_top(m_plan, level, target, false);
        while (m_nbeams * static_cast<SizeType>(plan.ring_size) *
                   sizeof(float) * top_words() >
               kMaxRingBytes) {
            if (level + step + 1 >= target) {
                use_skewed = true;
                break;
            }
            level += step;
            plan = detail::build_fdmt_direct_top(m_plan, level, target, false);
        }
    }
    if (use_skewed) {
        level = std::max<SizeType>(
            orig_level, k_direct + (pc.state_shape[0].nchans > 4096 ? 3 : 2));
        if (target < root && (target - level) % 2 == 0) {
            ++level;
        }
        if (level + 1 >= target) {
            level = orig_level;
        }
        plan = detail::build_fdmt_direct_top(m_plan, level, target, true,
                                             m_chunk, true);
        if (m_nbeams * static_cast<SizeType>(plan.ring_size) * sizeof(float) *
                top_words() >
            kMaxRingBytes) {
            plan = detail::build_fdmt_direct_top(m_plan, level, target, true);
            while (m_nbeams * static_cast<SizeType>(plan.ring_size) *
                       sizeof(float) * top_words() >
                   kMaxRingBytes) {
                if (level + step + 1 >= target) {
                    logging::debug(
                        "FDMT (cuda): skewed direct top rings over budget "
                        "at every level; top disabled");
                    fallback_skewed_direct();
                    return;
                }
                level += step;
                plan =
                    detail::build_fdmt_direct_top(m_plan, level, target, true);
            }
        }
        configure_skewed_direct_and_mid(relaid, plan);
        if (!m_direct || !m_direct->skewed) {
            return;
        }
    }
    auto t         = std::make_unique<DirectTopD>();
    t->level       = level;
    t->target      = target;
    t->skewed      = plan.skewed;
    t->target_ring = plan.target_ring;
    t->ntrials     = static_cast<int>(plan.out_row.size());
    t->nrows       = static_cast<int>(plan.ring_len.size());
    t->max_ring    = plan.max_ring;
    t->ring_size   = plan.ring_size;
    t->out_row     = plan.out_row;
    t->leaf_off    = plan.leaf_off;
    t->leaves      = plan.leaves;
    t->ring_len    = plan.ring_len;
    t->ring_off    = plan.ring_off;
    t->hist_leaves = plan.hist_leaves;
    // uint64_t rings (packed search) take two floats' storage each.
    t->ring.resize(
        m_nbeams * static_cast<SizeType>(plan.ring_size) * top_words(), 0.0F);
    t->hist_coords.resize(target + 1);
    t->hist_max_delay.assign(target + 1, 0);
    for (SizeType l = level + 1; l <= target; ++l) {
        if (l < plan.hist_coords.size()) {
            t->hist_coords[l] = plan.hist_coords[l];
            for (const auto& hc : plan.hist_coords[l]) {
                t->hist_max_delay[l] = std::max(t->hist_max_delay[l], hc.delay);
            }
        }
    }
    if (target == root) {
        t->trials.resize(m_nbeams * plan.out_row.size() * m_chunk *
                         top_words());
    }
    logging::debug("FDMT (cuda): level {} summed from level {} (skewed={}, "
                   "target_ring={}, {} leaves, rings {} floats per beam)",
                   target, level, t->skewed, t->target_ring, plan.leaves.size(),
                   plan.ring_size);
    m_top_filled = static_cast<SizeType>(t->max_ring);
    m_top        = std::move(t);
}

SizeType FDMTCudaEngine::top_workspace_bytes() const noexcept {
    if (!m_top) {
        return 0;
    }
    const auto& t = *m_top;
    SizeType hist = 0;
    for (const auto& hc : t.hist_coords) {
        hist += hc.size() * sizeof(detail::FDMTDirectHistCoord);
    }
    return (t.trials.size() * sizeof(float)) +
           ((t.out_row.size() + t.leaf_off.size() + t.ring_len.size()) *
            sizeof(int32_t)) +
           (t.ring_off.size() * sizeof(int64_t)) +
           ((t.leaves.size() + t.hist_leaves.size()) *
            sizeof(detail::FDMTDirectTopLeaf)) +
           hist;
}

SizeType FDMTCudaEngine::direct_window_bytes() const noexcept {
    if (!m_direct) {
        return 0;
    }
    return ((m_direct->win[0].size() + m_direct->win[1].size() +
             m_direct->win_ring.size()) *
            sizeof(float)) +
           (m_direct->win_ring_u8.size() * sizeof(uint8_t)) +
           (m_direct->win_ring_u16.size() * sizeof(uint16_t));
}

SizeType FDMTCudaEngine::direct_workspace_bytes() const noexcept {
    if (!m_direct) {
        return 0;
    }
    const auto& d = *m_direct;
    SizeType hist = 0;
    for (const auto& hc : d.hist_coords) {
        hist += hc.size() * sizeof(detail::FDMTDirectHistCoord);
    }
    return (d.prefix.size() * sizeof(float)) +
           (d.subs.size() * sizeof(detail::FDMTDirectSub)) +
           ((d.leaf_off.size() + d.chan_ring_len.size()) * sizeof(int32_t)) +
           (d.chan_ring_off.size() * sizeof(int64_t)) +
           ((d.leaves.size() + d.hist_leaves.size()) *
            sizeof(detail::FDMTDirectLeaf)) +
           hist;
}

} // namespace dmt::algorithms::detail::fdmt_cuda

namespace dmt::algorithms {

std::unique_ptr<detail::FDMTEngine>
detail::make_fdmt_gpu(const plans::FDMTPlan& plan,
                      const detail::FDMTEngineConfig& cfg) {
    // Memory budget (FDMTEngineConfig::max_memory_bytes). Construction is
    // cheap next to the plan (about 1 s on parkes_uwl), so the automatic
    // chunk is chosen by building and measuring: the windowed top (subband
    // search on the skewed path) runs faster at longer chunks, so it takes
    // the longest that fits; any other path keeps its automatic chunk and
    // halves it while over budget. A longer windowed-top chunk is only
    // built when predicted to fit (its memory is affine in the chunk: rows
    // of chunk + spread samples, rings of reach + chunk; extrapolated from
    // two measured engines), so construction never holds more than the
    // budget to find out (it held 12 GiB at chunk 4096 for float32 UWL).
    const auto fits = [&](const detail::fdmt_cuda::FDMTCudaEngine& e) {
        return e.get_memory_usage().total() <= cfg.max_memory_bytes;
    };
    const auto over = [&](const detail::fdmt_cuda::FDMTCudaEngine& e) {
        return std::invalid_argument(fmt::format(
            "FDMT (cuda): this configuration needs {:.2f} GiB, over "
            "max_memory_bytes ({:.2f} GiB)",
            static_cast<double>(e.get_memory_usage().total()) / (1 << 30),
            static_cast<double>(cfg.max_memory_bytes) / (1 << 30)));
    };
    auto engine =
        std::make_unique<detail::fdmt_cuda::FDMTCudaEngine>(plan, cfg);
    if (cfg.chunk_nsamps != 0 || cfg.mode != FDMTMode::kValid) {
        if (!fits(*engine)) {
            throw over(*engine);
        }
        return engine;
    }
    const auto try_chunk =
        [&](SizeType c) -> std::unique_ptr<detail::fdmt_cuda::FDMTCudaEngine> {
        auto c_cfg         = cfg;
        c_cfg.chunk_nsamps = c;
        try {
            auto e = std::make_unique<detail::fdmt_cuda::FDMTCudaEngine>(plan,
                                                                         c_cfg);
            return fits(*e) ? std::move(e) : nullptr;
        } catch (const std::bad_alloc&) {
            return nullptr;
        } catch (const thrust::system_error&) {
            return nullptr;
        }
    };
    const SizeType nsamps = plan.get_nsamps();
    const SizeType c_auto = engine->get_chunk_nsamps();
    if (engine->search_path().find("windowed") != std::string::npos &&
        fits(*engine) && c_auto >= 512 &&
        c_auto < std::min<SizeType>(4096, nsamps)) {
        const SizeType c0 = c_auto;
        const auto m0 = static_cast<double>(engine->get_memory_usage().total());
        const auto budget = static_cast<double>(cfg.max_memory_bytes);
        // Free the device memory before each build. The second point is a
        // shorter chunk, which fits whenever c0 does.
        engine.reset();
        const SizeType c1 = c0 / 2;
        double slope      = 0.0;
        if (auto e1 = try_chunk(c1)) {
            const auto m1 = static_cast<double>(e1->get_memory_usage().total());
            slope         = (m0 - m1) / static_cast<double>(c0 - c1);
        }
        for (const SizeType c : {4096, 2048}) {
            if (c <= c0 || c > nsamps || slope <= 0.0 ||
                m0 + (slope * static_cast<double>(c - c0)) > budget) {
                continue;
            }
            if (auto e = try_chunk(c)) {
                return e;
            }
        }
        engine = std::make_unique<detail::fdmt_cuda::FDMTCudaEngine>(plan, cfg);
    }
    SizeType chunk = engine->get_chunk_nsamps();
    while (!fits(*engine)) {
        if (chunk <= 256) {
            throw over(*engine);
        }
        chunk /= 2;
        auto c_cfg         = cfg;
        c_cfg.chunk_nsamps = chunk;
        engine.reset();
        engine =
            std::make_unique<detail::fdmt_cuda::FDMTCudaEngine>(plan, c_cfg);
    }
    return engine;
}

} // namespace dmt::algorithms
