#include "fdmt_cuda_engine.cuh"

// CUDA FDMT engine: execute(), the boxcar search input, the tree and
// fused levels, with their kernels.

namespace dmt::algorithms::detail::fdmt_cuda {

namespace {

/**
 * @brief Level-0 (per-channel) FDMT state initialization kernel.
 *
 * `dt_grid_sub` (this sub-band's local dt grid) is dense, contiguous, and
 * sorted ascending by construction (see `make_plan_iter0` in
 * fdmt_plan_build.cpp), but may start below zero when the plan's overall dt
 * range is negative or straddles zero. Level 0 has no notion of merge
 * *direction* -- that is resolved entirely by the sign-aware tail/head
 * reference swap in `make_plan` (see its doc comment), once per merge, above
 * this level. A row here only ever represents "this single channel's own
 * delay/smearing width of |dt| samples": rows +s and -s are always identical,
 * since a single leaf channel has no other channel to be earlier/later than.
 * This mirrors the CPU implementation's `fdmt_init_subband` in fdmt.cpp exactly
 * (see its doc comment for the full rationale) -- every row lookup below is
 * indexed by magnitude `s`, and `write_matches` fans a computed row out to
 * whichever of the +s / -s dt indices are actually present in this
 * sub-band's grid.
 */
template <FDMTMode Mode, bool UseBoxSmearing, typename Input, typename TOut>
__global__ void
kernel_init_fdmt(const Input waterfall,
                 TOut* __restrict__ state,
                 const int* __restrict__ grids0_dt_grid_ptr,
                 const int* __restrict__ grids0_ndt_ptr,
                 const int* __restrict__ grids0_coord_offset_ptr,
                 int nsubs,
                 int nsamps,
                 int row_stride,
                 int hist_len,
                 int state0_nelements,
                 const float* __restrict__ hist,
                 const detail::FDMTBoxMeans box) {
    const auto isamp =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (i_sub >= nsubs || isamp >= nsamps) {
        return;
    }

    const auto* dt_grid_sub =
        &grids0_dt_grid_ptr[grids0_coord_offset_ptr[i_sub]];
    // `nsamps` samples of this chunk; rows `row_stride` apart.
    const auto buffer_offset = (i_beam * state0_nelements) +
                               (grids0_coord_offset_ptr[i_sub] * row_stride);
    const auto ndt_grid_sub  = grids0_ndt_ptr[i_sub];
    const auto wf_row        = (i_beam * nsubs) + i_sub;
    const auto hist_offset   = (i_beam * nsubs * hist_len) + (i_sub * hist_len);
    const auto dt_first      = dt_grid_sub[0];
    const auto dt_last       = dt_grid_sub[ndt_grid_sub - 1];

    // Lambda to fetch sample at relative index (isamp - shift), shift >= 0.
    auto get_sample = [&](int t) -> float {
        if (t >= 0) {
            return waterfall.load(wf_row, t);
        }
        if constexpr (Mode == FDMTMode::kRoll) {
            return waterfall.load(wf_row, t + nsamps);
        } else if constexpr (Mode == FDMTMode::kValid) {
            return (hist != nullptr) ? hist[hist_offset + hist_len + t] : 0.0F;
        } else {
            return 0.0F; // kFull: zero-padded before t=0
        }
    };

    // Box smearing writes each box's mean (fdmt_box_weights.hpp): the
    // dyadic float mean for float input; for packed input the exact integer
    // box sum (uint32) rounded by its recipe, as integer units for integer
    // TOut (int_tree, exact within the type's plan bound) or times 2^-F. The
    // column helper is shared with the fused kernel, so both agree bit for
    // bit.
    const auto put = [&](int i_dt, auto val) {
        state[buffer_offset + (static_cast<int64_t>(i_dt) * row_stride) +
              isamp] = static_cast<TOut>(val);
    };
    if constexpr (detail::kFDMTPackedInput<Input> &&
                  std::is_same_v<TOut, uint64_t>) {
        detail::fdmt_init_column<UseBoxSmearing, uint64_t>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, uint64_t sum) {
                return detail::box_round_apply<uint64_t>(sum, box.round[s]);
            });
    } else if constexpr (detail::kFDMTPackedInput<Input>) {
        detail::fdmt_init_column<UseBoxSmearing, uint32_t>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, uint32_t sum) {
                return box.of_packed(s, sum, std::is_integral_v<TOut>);
            });
    } else {
        detail::fdmt_init_column<UseBoxSmearing, float>(
            dt_first, dt_last, isamp, get_sample, put,
            [&](int s, float sum) { return box.of_float(s, sum); });
    }
}

/// Moves the last len[r] samples of each row's window (from index n) to
/// its front, in rounds of n so a round's reads never meet its writes.
template <typename T>
__global__ void kernel_carry_front(T* __restrict__ base,
                                   const SizeType* __restrict__ off,
                                   const int32_t* __restrict__ len,
                                   int n) {
    T* row       = base + off[blockIdx.x];
    const int sp = len[blockIdx.x];
    for (int b = 0; b < sp; b += n) {
        const int end = b + n < sp ? b + n : sp;
        for (int i = b + static_cast<int>(threadIdx.x); i < end;
             i += static_cast<int>(blockDim.x)) {
            row[i] = row[i + n];
        }
        __syncthreads();
    }
}

/// kernel_advance_tree_history: one warp per (coordinate, segment of
/// kAdvanceSegment samples), kAdvanceWarps coordinates per block, so the many
/// short-delay rows of the low levels do not idle a block each.
inline constexpr int kAdvanceWarps   = 8;
inline constexpr int kAdvanceSegment = 256;

/**
 * @brief Advances tree-level cross-block history on GPU (layout as in
 * kernel_execute_iter). A FIFO is shifted from hist_in into hist_out; a
 * ring (hist_in == hist_out) only receives the chunk's newest
 * min(delay, chunk) head samples in place, after the level's merges have
 * read it. Supports arbitrary block sizes and offsets.
 * Grid: x over coordinate groups (strided), y over segments, z over beams.
 */
template <typename TIn = float, typename THist = float>
__global__ __launch_bounds__(
    kAdvanceWarps *
    32) void kernel_advance_tree_history(const TIn* __restrict__ state_in,
                                         const plans::FDMTCoordDPtrs coords_sum,
                                         const THist* hist_in,
                                         THist* hist_out,
                                         int ncoords_sum_cur,
                                         int in_state_nelements,
                                         int64_t hist_stride,
                                         int64_t hist_base,
                                         uint64_t ring_pos,
                                         float int_unit,
                                         int active) {
    constexpr int kWarp = 32;
    const int lane      = static_cast<int>(threadIdx.x) % kWarp;
    const int warp      = static_cast<int>(threadIdx.x) / kWarp;
    const int seg0      = static_cast<int>(blockIdx.y) * kAdvanceSegment;
    // Existing FDMT keeps canonical float history; packed search keeps
    // exact integer history without a conversion through float.
    const float unit      = std::is_integral_v<TIn> ? int_unit : 1.0F;
    const auto to_history = [unit](TIn value) -> THist {
        if constexpr (std::is_integral_v<THist>) {
            return static_cast<THist>(value);
        } else {
            return static_cast<float>(value) * unit;
        }
    };
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    for (int i_coord = (static_cast<int>(blockIdx.x) * kAdvanceWarps) + warp;
         i_coord < ncoords_sum_cur;
         i_coord += static_cast<int>(gridDim.x) * kAdvanceWarps) {
        const auto offset = coords_sum.offset[i_coord];
        if (offset <= 0 || seg0 >= offset) {
            continue;
        }
        // `active` > 0: this chunk's length (see kernel_execute_iter).
        const auto nsamps_head =
            active > 0 ? active : coords_sum.head_nsamps[i_coord];
        const auto hist_off =
            (i_beam * hist_stride) +
            (static_cast<int64_t>(coords_sum.hist_offset[i_coord]) - hist_base);
        const auto head_idx_base =
            (i_beam * in_state_nelements) + coords_sum.head_buf_offset[i_coord];
        const bool ring  = ring_pos != kFifoHistory;
        const int newest = nsamps_head < offset ? nsamps_head : offset;
        const int count  = ring ? newest : offset;
        const int end =
            seg0 + kAdvanceSegment < count ? seg0 + kAdvanceSegment : count;
        const int first = nsamps_head - newest;
        const int slot0 =
            ring ? static_cast<int>((ring_pos + static_cast<uint64_t>(first)) %
                                    static_cast<uint64_t>(offset))
                 : 0;
        for (int k = seg0 + lane; k < end; k += kWarp) {
            if (ring) {
                int slot = slot0 + k;
                slot -= slot >= offset ? offset : 0;
                hist_out[hist_off + slot] =
                    to_history(state_in[head_idx_base + first + k]);
            } else if (nsamps_head >= offset) {
                hist_out[hist_off + k] = to_history(
                    state_in[head_idx_base + (nsamps_head - offset + k)]);
            } else if (k < offset - nsamps_head) {
                hist_out[hist_off + k] =
                    (hist_in != nullptr) ? hist_in[hist_off + k + nsamps_head]
                                         : THist{0};
            } else {
                hist_out[hist_off + k] = to_history(
                    state_in[head_idx_base + (k - (offset - nsamps_head))]);
            }
        }
    } // i_coord
}

/// @brief Thread-block execution policy of detail::fdmt_fused_tile: rows
/// strided over the block's warps, a row's cells over the warp's lanes, and
/// __syncthreads() as the barrier.
struct DeviceBlock {
    template <typename Row>
    __host__ __device__ void
    for_each_row(int nrows, int width, Row&& row) const {
#if DMT_GPU_DEVICE_PASS
        constexpr int kWarp = 32;
        const int lane      = static_cast<int>(threadIdx.x) % kWarp;
        const int nwarps    = static_cast<int>(blockDim.x) / kWarp;
        for (int r = static_cast<int>(threadIdx.x) / kWarp; r < nrows;
             r += nwarps) {
            auto cell = row(r);
            for (int j = lane; j < width; j += kWarp) {
                cell(j);
            }
        }
#else
        (void)nrows;
        (void)width;
        (void)row;
#endif
    }
    __host__ __device__ void sync() const {
#if DMT_GPU_DEVICE_PASS
        __syncthreads();
#endif
    }
};

/**
 * @brief Fused level-0 initialisation + tree merges 1..F (see
 * fdmt_fused_tile.hpp). One block per (time tile, channel group, beam);
 * dynamic shared memory holds the group's intermediate levels. Writes level
 * F, and in valid mode the fused levels' tree history, exactly as the
 * unfused kernels would.
 */
template <FDMTMode Mode, bool UseBoxSmearing, typename Input, typename TOut>
__global__ void kernel_fused_levels(const Input waterfall,
                                    TOut* __restrict__ state_out,
                                    const detail::FDMTFusedTileArgs args,
                                    int64_t out_beam_stride,
                                    const float* __restrict__ hist0,
                                    const float* __restrict__ thist_in,
                                    float* __restrict__ thist_out,
                                    int64_t tree_hist_size) {
    extern __shared__ float smem[];
    const auto i_beam       = static_cast<int64_t>(blockIdx.z);
    const auto hist0_stride = static_cast<int64_t>(args.nsubs) * args.hist_len;
    detail::fdmt_fused_tile<Mode, UseBoxSmearing>(
        DeviceBlock{}, waterfall, i_beam * args.nsubs,
        state_out + (i_beam * out_beam_stride), args,
        static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y), smem,
        (hist0 != nullptr) ? hist0 + (i_beam * hist0_stride) : nullptr,
        (thist_in != nullptr) ? thist_in + (i_beam * tree_hist_size) : nullptr,
        (thist_out != nullptr) ? thist_out + (i_beam * tree_hist_size)
                               : nullptr);
}

} // namespace

void FDMTCudaEngine::execute_search(DeviceSpan<const float> waterfall,
                                    detail::FDMTSearchSink& sink,
                                    Stream queue) {
    check_inputs(waterfall.size(), m_nbeams * m_plan.get_dmt_size());
    const SizeType root = m_plan.get_niters();
    if (m_mode != FDMTMode::kValid || get_fuse_levels() >= root) {
        throw std::logic_error(
            "FDMT search requires valid mode and a separate root");
    }
    const auto stream = to_cuda(queue);
    const detail::FDMTInputF32 input{waterfall.data(),
                                     static_cast<int>(m_plan.get_nsamps())};
    for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
        start(chunk_input(input, t0), m_float_levels, nullptr,
              std::min(m_chunk, m_plan.get_nsamps() - t0), stream, true);
        const auto consume = [&](SizeType l, SizeType source) {
            const auto hist = tree_history(l);
            sink.consume(
                l,
                {
                    .state =
                        reinterpret_cast<const float*>(m_levels_d[source].base),
                    .prev_state = l > 0 ? reinterpret_cast<const float*>(
                                              m_levels_d[l - 1].base)
                                        : nullptr,
                    .history    = hist.in,
                    .row_stride = m_chunk + (source < m_level_max_extra.size()
                                                 ? m_level_max_extra[source]
                                                 : 0),
                    .beam_stride =
                        static_cast<SizeType>(level_beam_stride(source)),
                    .history_stride = static_cast<SizeType>(hist.stride),
                    .history_base   = static_cast<SizeType>(hist.base),
                    .nsamps         = m_active,
                    .sample_begin   = m_chunk_pos,
                    .history_ring   = hist.ring != kFifoHistory,
                    .stream         = queue,
                });
        };
        consume(m_current_level, m_current_level);
        if (m_top_chunk && m_top->target < root) {
            while (m_current_level < m_top->level) {
                const SizeType next = m_current_level + 1;
                execute_iter_device(next, stream, /*regular_only=*/true,
                                    /*advance_hist=*/true);
                m_current_level = next;
            }
            run_top(reinterpret_cast<float*>(m_levels_d[m_top->target].base),
                    level_beam_stride(m_top->target), static_cast<int>(m_chunk),
                    stream);
        }
        const SizeType last =
            (m_top_chunk && m_top->target == root) ? m_top->level : root - 1;
        while (m_current_level < last) {
            const SizeType next = m_current_level + 1;
            // The merge kernel may have advanced the regular rows'
            // rings itself; advance only the rest after the search read
            // the history.
            const int coord_begin =
                execute_iter_device(next, stream, /*regular_only=*/true,
                                    /*advance_hist=*/false);
            m_current_level = next;
            consume(next, next);
            advance_tree_history<float>(
                reinterpret_cast<const float*>(m_levels_d[next - 1].base), next,
                stream, coord_begin);
        }
        if (m_top_chunk && m_top->windowed) {
            run_windowed_top<float>(stream, [&](SizeType l, const float* st) {
                sink.consume(l, windowed_input<float>(l, st, queue));
            });
        } else if (m_top_chunk && m_top->target == root) {
            // The direct top materialises this chunk's trials.
            auto* trials = thrust::raw_pointer_cast(m_top->trials.data());
            const auto trial_stride =
                m_plan.get_container().coordinates[root].size() * m_chunk;
            run_top(trials, static_cast<int64_t>(trial_stride),
                    static_cast<int>(m_chunk), stream);
            sink.consume(root, {
                                   .state             = trials,
                                   .row_stride        = m_chunk,
                                   .beam_stride       = trial_stride,
                                   .nsamps            = m_active,
                                   .sample_begin      = m_chunk_pos,
                                   .materialized_root = true,
                                   .stream            = queue,
                               });
        } else {
            consume(root, root - 1);
            advance_tree_history<float>(
                reinterpret_cast<const float*>(m_levels_d[root - 1].base), root,
                stream);
        }
        m_current_level  = root;
        m_is_initialized = false;
    }
    m_device_work.mark(stream);
}

void FDMTCudaEngine::execute_search(DeviceSpan<const uint8_t> waterfall,
                                    SizeType nbits,
                                    detail::FDMTSearchSink& sink,
                                    Stream queue) {
    if (nbits == 32 || nbits != m_search_nbits) {
        throw std::invalid_argument(
            "FDMT integer search: input bit depth mismatch");
    }
    check_packed_inputs(waterfall.size(), nbits,
                        m_nbeams * m_plan.get_dmt_size());
    gpu_utils::set_device(m_device_id);
    const auto stream   = to_cuda(queue);
    const SizeType root = m_plan.get_niters();
    const FDMTInputPacked full{
        .data      = waterfall.data(),
        .row_bytes = static_cast<int>(
            bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits)),
        .nbits = static_cast<int>(nbits)};
    if (m_search_u32) {
        execute_search_int_typed<uint32_t>(
            full, thrust::raw_pointer_cast(m_integer32_ring.data()), sink,
            queue, stream);
    } else {
        execute_search_int_typed<uint64_t>(
            full, thrust::raw_pointer_cast(m_integer_ring.data()), sink, queue,
            stream);
    }
    m_device_work.mark(stream);
}

template <typename TInt>
void FDMTCudaEngine::execute_search_int_typed(const FDMTInputPacked& full,
                                              TInt* ring,
                                              detail::FDMTSearchSink& sink,
                                              Stream queue,
                                              cudaStream_t stream) {
    const SizeType root = m_plan.get_niters();
    const auto tree     = static_cast<int64_t>(m_plan.get_tree_history_size());
    if (m_direct) {
        execute_search_int_direct<TInt>(full, ring, sink, queue, stream);
        return;
    }
    for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
        m_active    = std::min(m_chunk, m_plan.get_nsamps() - t0);
        m_chunk_pos = m_next_pos;
        m_next_pos += m_active;
        m_level0_hist_in_ptr = thrust::raw_pointer_cast(
            (m_parity ? m_history_b_d : m_history_a_d).data());
        m_level0_hist_out_ptr = thrust::raw_pointer_cast(
            (m_parity ? m_history_a_d : m_history_b_d).data());
        m_parity    = !m_parity;
        auto* state = reinterpret_cast<TInt*>(
            thrust::raw_pointer_cast(m_state_internal_d.data()));
        const auto level_ptr = [&](SizeType l) {
            return state + (l % 2) * m_nbeams * m_half_elems;
        };
        initialise_device<FDMTInputPacked, TInt>(chunk_input(full, t0),
                                                 level_ptr(0), stream);
        for (SizeType l = 0; l <= root; ++l) {
            const auto& shape = m_plan.get_container().state_shape[l];
            auto sums         = m_plan_d.coordinates_sum.get_raw_ptrs();
            auto copies       = m_plan_d.coordinates_copy.get_raw_ptrs();
            sums.update_offsets(m_coords_sum_offsets[l]);
            copies.update_offsets(m_coords_copy_offsets[l]);
            int adv_begin = 0;
            if (l > 0 && l < root) {
                const auto reg_sum =
                    detail::fdmt_regular_sum_coords(m_plan.get_container(), l);
                const int count =
                    static_cast<int>(std::max(reg_sum, shape.ncoords_copy));
                const dim3 block(kIterThreads);
                const dim3 grid(
                    (m_active + kIterThreads * kIterSampsPerTh - 1) /
                        (kIterThreads * kIterSampsPerTh),
                    std::min(count, kMaxGridY), m_nbeams);
                gpu_utils::check_kernel_launch_params(grid, block);
                const bool fuse_ring = grid.x == 1;
                kernel_execute_iter<FDMTMode::kValid, TInt, TInt, TInt>
                    <<<grid, block, 0, stream>>>(
                        level_ptr(l - 1), level_ptr(l), sums, copies, ring,
                        static_cast<int>(m_active), static_cast<int>(reg_sum),
                        static_cast<int>(shape.ncoords_copy),
                        static_cast<int>(m_half_elems),
                        static_cast<int>(m_half_elems), tree, 0, m_chunk_pos,
                        1.0F, static_cast<int>(m_active), fuse_ring);
                if (fuse_ring) {
                    adv_begin = static_cast<int>(reg_sum);
                }
            }
            detail::FDMTSearchInput in{
                .row_stride     = m_chunk,
                .beam_stride    = m_half_elems,
                .history_stride = m_plan.get_tree_history_size(),
                .nsamps         = m_active,
                .sample_begin   = m_chunk_pos,
                .history_ring   = true,
                .stream         = queue,
            };
            if constexpr (std::is_same_v<TInt, uint32_t>) {
                in.integer32_state      = level_ptr(l == root ? root - 1 : l);
                in.integer32_prev_state = l > 0 ? level_ptr(l - 1) : nullptr;
                in.integer32_history    = ring;
            } else {
                in.integer_state      = level_ptr(l == root ? root - 1 : l);
                in.integer_prev_state = l > 0 ? level_ptr(l - 1) : nullptr;
                in.integer_history    = ring;
            }
            sink.consume(l, in);
            const int adv_count =
                static_cast<int>(shape.ncoords_sum) - adv_begin;
            if (l > 0 && adv_count > 0 && shape.dt_max > 0) {
                // Ring: only the chunk's newest min(delay, chunk) samples.
                const auto newest = static_cast<int>(
                    std::min<SizeType>(shape.dt_max, m_active));
                sums.update_offsets(adv_begin);
                launch_advance_tree_history(level_ptr(l - 1), sums, ring, ring,
                                            adv_count, newest,
                                            static_cast<int>(m_half_elems),
                                            tree, 0, m_chunk_pos, 1.0F, stream);
            }
            gpu_utils::check_last_gpu_error(
                "FDMT integer search launch failed");
        }
    }
}

template <typename TInt>
void FDMTCudaEngine::execute_search_int_direct(const FDMTInputPacked& full,
                                               TInt* ring,
                                               detail::FDMTSearchSink& sink,
                                               Stream queue,
                                               cudaStream_t stream) {
    const SizeType root = m_plan.get_niters();
    const auto h_stride = static_cast<int64_t>(m_ring_size);
    auto* state         = reinterpret_cast<TInt*>(
        thrust::raw_pointer_cast(m_state_internal_d.data()));
    const auto level_ptr = [&](SizeType l) { return state_level_ptr<TInt>(l); };
    const auto beam_stride = static_cast<int64_t>(m_half_elems);
    const auto consume     = [&](SizeType l, const TInt* src, const TInt* prev,
                                 int64_t src_beam_stride, bool materialized) {
        const SizeType src_lvl = l == root ? root - 1 : l;
        detail::FDMTSearchInput in{
            .row_stride        = m_chunk + (src_lvl < m_level_max_extra.size()
                                                ? m_level_max_extra[src_lvl]
                                                : 0),
            .beam_stride       = static_cast<SizeType>(src_beam_stride),
            .history_stride    = m_ring_size,
            .history_base      = ring_base_for_level(l),
            .nsamps            = m_active,
            .sample_begin      = m_chunk_pos,
            .materialized_root = materialized,
            .history_ring      = true,
            .stream            = queue,
        };
        if constexpr (std::is_same_v<TInt, uint32_t>) {
            in.integer32_state      = src;
            in.integer32_prev_state = prev;
            in.integer32_history    = ring;
        } else {
            in.integer_state      = src;
            in.integer_prev_state = prev;
            in.integer_history    = ring;
        }
        sink.consume(l, in);
    };
    const auto merge = [&](SizeType l) {
        const auto& shape = m_plan.get_container().state_shape[l];
        auto sums         = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto copies       = m_plan_d.coordinates_copy.get_raw_ptrs();
        sums.update_offsets(m_coords_sum_offsets[l]);
        copies.update_offsets(m_coords_copy_offsets[l]);
        const auto reg_sum =
            detail::fdmt_regular_sum_coords(m_plan.get_container(), l);
        const int count =
            static_cast<int>(std::max(reg_sum, shape.ncoords_copy));
        const int nsamps_l = static_cast<int>(
            m_active +
            (l < m_level_max_extra.size() ? m_level_max_extra[l] : 0));
        const dim3 block(kIterThreads);
        const dim3 grid((nsamps_l + kIterThreads * kIterSampsPerTh - 1) /
                            (kIterThreads * kIterSampsPerTh),
                        std::min(count, kMaxGridY), m_nbeams);
        gpu_utils::check_kernel_launch_params(grid, block);
        const bool skewed_mid = m_direct->skewed && m_top && l <= m_top->level;
        const bool fuse_ring  = !skewed_mid && grid.x == 1;
        const auto h_base     = static_cast<int64_t>(ring_base_for_level(l));
        kernel_execute_iter<FDMTMode::kValid, TInt, TInt, TInt>
            <<<grid, block, 0, stream>>>(
                level_ptr(l - 1), level_ptr(l), sums, copies,
                skewed_mid ? nullptr : ring, nsamps_l,
                static_cast<int>(reg_sum), static_cast<int>(shape.ncoords_copy),
                static_cast<int>(m_half_elems), static_cast<int>(m_half_elems),
                h_stride, h_base, m_chunk_pos, 1.0F, static_cast<int>(m_active),
                fuse_ring, skewed_mid ? static_cast<int>(m_chunk) : 0);
        return (skewed_mid || fuse_ring) ? static_cast<int>(reg_sum) : 0;
    };
    const auto advance = [&](SizeType l, int adv_begin) {
        if (m_direct->skewed && m_top && l <= m_top->level) {
            return;
        }
        const auto& shape   = m_plan.get_container().state_shape[l];
        const int adv_count = static_cast<int>(shape.ncoords_sum) - adv_begin;
        if (adv_count > 0 && shape.dt_max > 0 && ring != nullptr) {
            auto sums = m_plan_d.coordinates_sum.get_raw_ptrs();
            sums.update_offsets(m_coords_sum_offsets[l] + adv_begin);
            const auto newest =
                static_cast<int>(std::min<SizeType>(shape.dt_max, m_active));
            const auto h_base = static_cast<int64_t>(ring_base_for_level(l));
            launch_advance_tree_history(
                level_ptr(l - 1), sums, ring, ring, adv_count, newest,
                static_cast<int>(m_half_elems), h_stride, h_base, m_chunk_pos,
                1.0F, stream);
        }
    };
    const SizeType levels = m_direct->levels;
    for (SizeType t0 = 0; t0 < m_plan.get_nsamps(); t0 += m_chunk) {
        m_active    = std::min(m_chunk, m_plan.get_nsamps() - t0);
        m_chunk_pos = m_next_pos;
        m_next_pos += m_active;
        const int in = m_parity ? 1 : 0;
        if (!m_direct->skewed) {
            m_direct_win_in =
                thrust::raw_pointer_cast(m_direct->win[in].data());
            m_direct_win_out =
                thrust::raw_pointer_cast(m_direct->win[1 - in].data());
        }
        m_parity         = !m_parity;
        const auto input = chunk_input(full, t0);
        launch_direct_int<TInt>(input, level_ptr(levels), stream);
        advance_direct_window(input, stream);
        SizeType level = (m_skew_nbatches > 1 && m_top) ? m_top->level : levels;
        consume(level, level_ptr(level), nullptr, beam_stride, false);
        const SizeType last = m_top ? m_top->level : root - 1;
        while (level < last) {
            ++level;
            const int adv_begin = merge(level);
            consume(level, level_ptr(level), level_ptr(level - 1), beam_stride,
                    false);
            advance(level, adv_begin);
        }
        if (m_top && m_top->windowed) {
            run_windowed_top<TInt>(stream, [&](SizeType l, const TInt* st) {
                sink.consume(l, windowed_input<TInt>(l, st, queue));
            });
            continue;
        }
        if (m_top) {
            if (m_top->target == root) {
                // The trials, materialised: the sink's root.
                auto* trials = reinterpret_cast<TInt*>(
                    thrust::raw_pointer_cast(m_top->trials.data()));
                const auto trial_stride = static_cast<int64_t>(
                    m_plan.get_container().coordinates[root].size() * m_chunk);
                run_top_int<TInt>(level_ptr(m_top->level), trials, trial_stride,
                                  stream);
                consume(root, trials, nullptr, trial_stride, true);
                gpu_utils::check_last_gpu_error(
                    "FDMT integer search (direct) launch failed");
                continue;
            }
            run_top_int<TInt>(level_ptr(m_top->level), level_ptr(m_top->target),
                              beam_stride, stream);
            level = m_top->target;
            consume(level, level_ptr(level), nullptr, beam_stride, false);
        }
        while (level + 1 < root) {
            ++level;
            const int adv_begin = merge(level);
            consume(level, level_ptr(level), level_ptr(level - 1), beam_stride,
                    false);
            advance(level, adv_begin);
        }
        // The root: merged on the fly by the sink from level root - 1.
        consume(root, level_ptr(root - 1), level_ptr(root - 1), beam_stride,
                false);
        advance(root, 0);
        gpu_utils::check_last_gpu_error(
            "FDMT integer search (direct) launch failed");
    }
}

void FDMTCudaEngine::execute(std::span<const float> waterfall_h,
                             std::span<float> dmt_h) {
    check_inputs(waterfall_h.size(), dmt_h.size());
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    stage_to_device(m_stage_wf_f32_d, waterfall_h);
    execute_d(cuda::std::span<const float>(
                  thrust::raw_pointer_cast(m_stage_wf_f32_d.data()),
                  waterfall_h.size()),
              stage_dmt_span(), m_host.stream());
    copy_result_to_host(dmt_h);
}

void FDMTCudaEngine::execute_d(cuda::std::span<const float> waterfall_d,
                               cuda::std::span<float> dmt_d,
                               cudaStream_t stream) {
    check_inputs(waterfall_d.size(), dmt_d.size());
    const FDMTInputF32 input{
        .data   = waterfall_d.data(),
        .nsamps = static_cast<int>(m_plan.get_nsamps()),
    };
    run_block(input, m_float_levels, dmt_d, stream);
}

template <typename Input>
void FDMTCudaEngine::run_block(const Input& input,
                               const std::vector<detail::FDMTLevelType>& types,
                               cuda::std::span<float> dmt_d,
                               cudaStream_t stream) {
    const SizeType nsamps = m_plan.get_nsamps();
    for (SizeType t0 = 0; t0 < nsamps; t0 += m_chunk) {
        start(chunk_input(input, t0), types, dmt_d.data() + t0,
              std::min(m_chunk, nsamps - t0), stream, /*fuse=*/true);
        if (m_top_chunk) {
            advance(m_top->level - m_current_level, Stream{stream});
            run_top(dmt_d.data() + t0, level_beam_stride(m_plan.get_niters()),
                    static_cast<int>(nsamps), stream);
        } else {
            advance_until_remaining(0, Stream{stream});
        }
    }
    finalize(Stream{stream});
}

void FDMTCudaEngine::execute(std::span<const uint8_t> waterfall_h,
                             SizeType nbits,
                             std::span<float> dmt_h) {
    check_packed_inputs(waterfall_h.size(), nbits, dmt_h.size());
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    // Only the packed bytes cross PCIe (32/nbits times less than float).
    stage_to_device(m_stage_wf_packed_d, waterfall_h);
    execute_d(cuda::std::span<const uint8_t>(
                  thrust::raw_pointer_cast(m_stage_wf_packed_d.data()),
                  waterfall_h.size()),
              nbits, stage_dmt_span(), m_host.stream());
    copy_result_to_host(dmt_h);
}

template <typename T>
void FDMTCudaEngine::stage_to_device(thrust::device_vector<T>& dst,
                                     std::span<const T> src) {
    if (dst.size() < src.size()) {
        dst.resize(src.size());
    }
    m_host.to_device(thrust::raw_pointer_cast(dst.data()), src.data(),
                     src.size_bytes());
}

void FDMTCudaEngine::copy_result_to_host(std::span<float> dmt_h) {
    std::vector<gpu_host::ChunkedStager::Segment> segs;
    segs.push_back({dmt_h.data(),
                    thrust::raw_pointer_cast(m_stage_dmt_d.data()),
                    m_nbeams * m_plan.get_dmt_size() * sizeof(float)});
    m_host.to_host(segs);
}

void FDMTCudaEngine::execute_d(cuda::std::span<const uint8_t> waterfall_d,
                               SizeType nbits,
                               cuda::std::span<float> dmt_d,
                               cudaStream_t stream) {
    check_packed_inputs(waterfall_d.size(), nbits, dmt_d.size());
    const auto row_bytes =
        bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
    const FDMTInputPacked input{.data      = waterfall_d.data(),
                                .row_bytes = static_cast<int>(row_bytes),
                                .nbits     = static_cast<int>(nbits)};
    run_block(input,
              (m_int_tree && !(m_direct && m_direct->skewed))
                  ? m_int_levels[nbits]
                  : m_float_levels,
              dmt_d, stream);
}

void FDMTCudaEngine::execute(DeviceSpan<const float> d_waterfall,
                             DeviceSpan<float> d_dmt,
                             Stream stream) {
    execute_d(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream));
    m_device_work.mark(to_cuda(stream));
}

void FDMTCudaEngine::execute(DeviceSpan<const uint8_t> d_waterfall,
                             SizeType nbits,
                             DeviceSpan<float> d_dmt,
                             Stream stream) {
    execute_d(to_cuda(d_waterfall), nbits, to_cuda(d_dmt), to_cuda(stream));
    m_device_work.mark(to_cuda(stream));
}

void FDMTCudaEngine::reset(DeviceSpan<const uint8_t> d_waterfall,
                           SizeType nbits,
                           DeviceSpan<float> d_dmt,
                           Stream stream) {
    reset_impl(to_cuda(d_waterfall), nbits, to_cuda(d_dmt), to_cuda(stream),
               /*fuse=*/false);
    m_device_work.mark(to_cuda(stream));
}

void FDMTCudaEngine::reset(DeviceSpan<const float> d_waterfall,
                           DeviceSpan<float> d_dmt,
                           Stream stream) {
    reset_impl(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream),
               /*fuse=*/false);
    m_device_work.mark(to_cuda(stream));
}

void FDMTCudaEngine::reset(std::span<const float> waterfall_h,
                           std::span<float> dmt_h) {
    check_inputs(waterfall_h.size(), dmt_h.size());
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    stage_to_device(m_stage_wf_f32_d, waterfall_h);
    reset_impl(cuda::std::span<const float>(
                   thrust::raw_pointer_cast(m_stage_wf_f32_d.data()),
                   waterfall_h.size()),
               stage_dmt_span(), m_host.stream(), /*fuse=*/false);
    m_host_dmt = dmt_h;
}

void FDMTCudaEngine::reset(std::span<const uint8_t> waterfall_h,
                           SizeType nbits,
                           std::span<float> dmt_h) {
    check_packed_inputs(waterfall_h.size(), nbits, dmt_h.size());
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    stage_to_device(m_stage_wf_packed_d, waterfall_h);
    reset_impl(cuda::std::span<const uint8_t>(
                   thrust::raw_pointer_cast(m_stage_wf_packed_d.data()),
                   waterfall_h.size()),
               nbits, stage_dmt_span(), m_host.stream(), /*fuse=*/false);
    m_host_dmt = dmt_h;
}

void FDMTCudaEngine::check_stepper_block() const {
    if (m_chunk < m_plan.get_nsamps()) {
        throw std::logic_error(fmt::format(
            "FDMT (cuda): the stepper runs whole blocks, but this engine "
            "executes blocks of {} samples in chunks of {} "
            "(chunk_nsamps); construct with chunk_nsamps >= nsamps to "
            "step through levels.",
            m_plan.get_nsamps(), m_chunk));
    }
}

void FDMTCudaEngine::reset_impl(cuda::std::span<const uint8_t> d_waterfall,
                                SizeType nbits,
                                cuda::std::span<float> d_dmt,
                                cudaStream_t stream,
                                bool fuse) {
    check_stepper_block();
    check_packed_inputs(d_waterfall.size(), nbits, d_dmt.size());
    const auto row_bytes =
        bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
    const FDMTInputPacked input{.data      = d_waterfall.data(),
                                .row_bytes = static_cast<int>(row_bytes),
                                .nbits     = static_cast<int>(nbits)};
    start(input, m_int_tree ? m_int_levels[nbits] : m_float_levels,
          d_dmt.data(), m_plan.get_nsamps(), stream, fuse);
}

void FDMTCudaEngine::reset_impl(cuda::std::span<const float> d_waterfall,
                                cuda::std::span<float> d_dmt,
                                cudaStream_t stream,
                                bool fuse) {
    check_stepper_block();
    check_inputs(d_waterfall.size(), d_dmt.size());
    const FDMTInputF32 input{
        .data   = d_waterfall.data(),
        .nsamps = static_cast<int>(m_plan.get_nsamps()),
    };
    start(input, m_float_levels, d_dmt.data(), m_plan.get_nsamps(), stream,
          fuse);
}

void FDMTCudaEngine::check_inputs(SizeType waterfall_size,
                                  SizeType dmt_size) const {
    const auto nchans = m_plan.get_nchans();
    const auto nsamps = m_plan.get_nsamps();
    if (waterfall_size != m_nbeams * nchans * nsamps) {
        throw std::invalid_argument(
            fmt::format("FDMT (cuda): Invalid size of waterfall. "
                        "Expected {}, got {}",
                        m_nbeams * nchans * nsamps, waterfall_size));
    }
    if (dmt_size < m_nbeams * m_plan.get_dmt_size()) {
        throw std::invalid_argument(
            fmt::format("FDMT (cuda): Invalid size of dmt. Expected at "
                        "least {}, got {}",
                        m_nbeams * m_plan.get_dmt_size(), dmt_size));
    }
}

void FDMTCudaEngine::check_packed_inputs(SizeType waterfall_bytes,
                                         SizeType nbits,
                                         SizeType dmt_size) const {
    if (nbits != 1 && nbits != 2 && nbits != 4 && nbits != 8 && nbits != 16) {
        throw std::invalid_argument(fmt::format(
            "FDMT (cuda): nbits={} must be one of 1, 2, 4, 8, 16", nbits));
    }
    const auto row_bytes =
        bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(), nbits);
    const auto expected = m_nbeams * m_plan.get_nchans() * row_bytes;
    if (waterfall_bytes != expected) {
        throw std::invalid_argument(fmt::format(
            "FDMT (cuda): Invalid size of packed waterfall (nbits={}). "
            "Expected {} bytes, got {}",
            nbits, expected, waterfall_bytes));
    }
    if (dmt_size < m_nbeams * m_plan.get_dmt_size()) {
        throw std::invalid_argument(
            fmt::format("FDMT (cuda): Invalid size of dmt. Expected at "
                        "least {}, got {}",
                        m_nbeams * m_plan.get_dmt_size(), dmt_size));
    }
}

template <typename TIn, typename THist>
void FDMTCudaEngine::launch_advance_tree_history(
    const TIn* in_ptr,
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
    cudaStream_t stream) const {
    const dim3 block(kAdvanceWarps * 32);
    const dim3 grid((count + kAdvanceWarps - 1) / kAdvanceWarps,
                    (max_offset + kAdvanceSegment - 1) / kAdvanceSegment,
                    m_nbeams);
    gpu_utils::check_kernel_launch_params(grid, block);
    kernel_advance_tree_history<TIn, THist><<<grid, block, 0, stream>>>(
        in_ptr, coords, hist_in, hist_out, count, in_state_nelements,
        hist_stride, hist_base, ring_pos, int_unit, static_cast<int>(m_active));
}

template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
void FDMTCudaEngine::launch_fused_typed(const Input& input,
                                        TOut* out,
                                        cudaStream_t stream) {
    const auto& f = *m_fused;
    if (f.smem_bytes > kDefaultSmemBytes) {
        cudaFuncSetAttribute(kernel_fused_levels<Mode, Smear, Input, TOut>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(f.smem_bytes));
    }
    const bool valid = Mode == FDMTMode::kValid;
    // Valid mode: this chunk's samples only.
    const int ntiles =
        valid ? static_cast<int>((m_active + f.tile - 1) / f.tile) : f.ntiles;
    const dim3 grid_size(static_cast<unsigned>(ntiles),
                         static_cast<unsigned>(f.ngroups),
                         static_cast<unsigned>(m_nbeams));
    const dim3 block_size(kFusedThreads);
    gpu_utils::check_kernel_launch_params(grid_size, block_size);
    auto args   = f.args;
    args.box    = box_means_for(input);
    args.active = valid ? static_cast<int>(m_active) : 0;
    kernel_fused_levels<Mode, Smear, Input, TOut>
        <<<grid_size, block_size, f.smem_bytes, stream>>>(
            input, out, args,
            static_cast<int64_t>(level_beam_stride(f.args.fuse)),
            valid ? m_level0_hist_in_ptr : nullptr,
            valid ? m_fifo_in_ptr : nullptr, valid ? m_fifo_out_ptr : nullptr,
            static_cast<int64_t>(m_fifo_size));
    gpu_utils::check_last_gpu_error("kernel_fused_levels launch failed");
}

template <typename Input>
void FDMTCudaEngine::launch_fused(const Input& input, cudaStream_t stream) {
    const SizeType fuse = get_fuse_levels();
    with_elem(m_levels_d[fuse].type, [&](auto tag) {
        using TOut        = decltype(tag);
        auto* out         = reinterpret_cast<TOut*>(m_levels_d[fuse].base);
        const auto launch = [&](auto mode_tag, auto smear_tag) {
            launch_fused_typed<decltype(mode_tag)::value,
                               decltype(smear_tag)::value, Input, TOut>(
                input, out, stream);
        };
        using Full  = std::integral_constant<FDMTMode, FDMTMode::kFull>;
        using Roll  = std::integral_constant<FDMTMode, FDMTMode::kRoll>;
        using Valid = std::integral_constant<FDMTMode, FDMTMode::kValid>;
        if (m_mode == FDMTMode::kFull) {
            m_use_box_smearing ? launch(Full{}, std::true_type{})
                               : launch(Full{}, std::false_type{});
        } else if (m_mode == FDMTMode::kRoll) {
            m_use_box_smearing ? launch(Roll{}, std::true_type{})
                               : launch(Roll{}, std::false_type{});
        } else {
            m_use_box_smearing ? launch(Valid{}, std::true_type{})
                               : launch(Valid{}, std::false_type{});
        }
    });
    advance_level0_history(input, stream);
}

void FDMTCudaEngine::layout_levels(const std::vector<Elem>& types, float* dmt) {
    const SizeType niters = m_plan.get_niters();
    auto* internal        = reinterpret_cast<std::byte*>(
        thrust::raw_pointer_cast(m_state_internal_d.data()));
    const SizeType half = m_nbeams * m_half_elems * sizeof(float);
    for (SizeType l = 0; l < niters; ++l) {
        m_levels_d[l] = {.base = internal + ((l % 2) * half), .type = types[l]};
    }
    if (m_top && m_top->windowed) {
        m_levels_d[m_top->level].base = reinterpret_cast<std::byte*>(
            thrust::raw_pointer_cast(m_top->win_top.data()));
    }
    m_levels_d[niters] = {.base = reinterpret_cast<std::byte*>(dmt),
                          .type = Elem::kF32};
}

template <typename Input>
void FDMTCudaEngine::start(const Input& input,
                           const std::vector<Elem>& types,
                           float* dmt,
                           SizeType active,
                           cudaStream_t stream,
                           bool fuse) {
    gpu_utils::set_device(m_device_id);
    // In "valid" mode every block must reach the root before the next
    // one starts: levels a stepper run never executed would skip this
    // block's history update, and the next block would read the
    // two-blocks-old ping-pong buffer.
    if (m_mode == FDMTMode::kValid && m_is_initialized && !is_finished()) {
        throw std::logic_error(fmt::format(
            "FDMT (cuda): the previous block's stepper run stopped at "
            "level "
            "{} of {}. In mode='valid' every block must be finalized "
            "before the next reset()/execute(), or its cross-block "
            "history is lost. Call finalize(), or reset_history() to "
            "abandon the stream.",
            m_current_level, total_levels() - 1));
    }
    m_stream           = stream;
    m_dmt_target_ptr   = dmt;
    m_active           = active;
    m_current_level    = 0;
    m_host_dmt         = {};
    m_host_level_valid = false;

    const bool direct = fuse && use_direct<Input>(types);
    m_top_chunk       = fuse && use_top(types);
    if (m_mode == FDMTMode::kValid) {
        if ((!direct && m_direct) || (!m_top_chunk && m_top)) {
            ensure_full_tree_history(stream);
        }
        // Tree levels after direct chunks lag behind: rebuild their
        // histories (at the current position) before the tree runs.
        if (!direct) {
            rebuild_direct_history(stream);
        }
        if (!m_top_chunk) {
            rebuild_top_history(stream);
        }
        // Ping-pong the level-0 history and the fused levels' FIFOs once
        // per chunk (not per level): all levels of one chunk must agree
        // on which buffer is "previous" vs "this". The ring stays put;
        // its slots follow the stream position.
        const int in = m_parity ? 1 : 0;
        if (m_direct && !m_direct->skewed) {
            m_direct_win_in =
                thrust::raw_pointer_cast(m_direct->win[in].data());
            m_direct_win_out =
                thrust::raw_pointer_cast(m_direct->win[1 - in].data());
        }
        m_fifo_in_ptr  = thrust::raw_pointer_cast(m_tree_fifo_d[in].data());
        m_fifo_out_ptr = thrust::raw_pointer_cast(m_tree_fifo_d[1 - in].data());
        m_level0_hist_in_ptr = thrust::raw_pointer_cast(
            (m_parity ? m_history_b_d : m_history_a_d).data());
        m_level0_hist_out_ptr = thrust::raw_pointer_cast(
            (m_parity ? m_history_a_d : m_history_b_d).data());
        m_parity    = !m_parity;
        m_chunk_pos = m_next_pos;
        m_next_pos += active;
    }

    layout_levels(types, dmt);
    if (direct) {
        launch_direct(input, stream);
        if (!m_direct->skewed) {
            advance_level0_history(input, stream);
        }
        m_current_level =
            (m_skew_nbatches > 1 && m_top) ? m_top->level : m_direct->levels;
        m_direct_stale = !m_direct->skewed;
    } else if (fuse && m_fused) {
        launch_fused(input, stream);
        m_current_level = get_fuse_levels();
    } else {
        with_elem(m_levels_d[0].type, [&](auto tag0) {
            using T0 = decltype(tag0);
            initialise_device<Input, T0>(
                input, reinterpret_cast<T0*>(m_levels_d[0].base), stream);
        });
    }
    if (m_direct && !m_direct->skewed) {
        advance_direct_window(input, stream);
    }
    m_is_initialized = true;
}

template <typename T, typename Consume>
void FDMTCudaEngine::run_windowed_top(cudaStream_t stream, Consume&& consume) {
    auto& t                       = *m_top;
    const SizeType root           = m_plan.get_niters();
    const SizeType first_searched = root - m_plan.get_subband_depth();
    for (SizeType l = t.level + 1; l <= root; ++l) {
        const auto& shape = m_plan.get_container().state_shape[l];
        auto sums         = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto copies       = m_plan_d.coordinates_copy.get_raw_ptrs();
        sums.update_offsets(m_coords_sum_offsets[l]);
        copies.update_offsets(m_coords_copy_offsets[l]);
        const int nsum  = static_cast<int>(shape.ncoords_sum);
        const int ncopy = static_cast<int>(shape.ncoords_copy);
        const int count = std::max(nsum, ncopy);
        T* out = l == root ? reinterpret_cast<T*>(
                                 thrust::raw_pointer_cast(t.trials.data()))
                           : state_level_ptr<T>(l);
        const int nsamps_l = static_cast<int>(m_active + m_level_max_extra[l]);
        constexpr int kBlockSamps = kIterThreads * kIterSampsPerTh;
        const dim3 grid((nsamps_l + kBlockSamps - 1) / kBlockSamps,
                        std::min(count, kMaxGridY), 1);
        gpu_utils::check_kernel_launch_params(grid, dim3(kIterThreads));
        kernel_execute_iter<FDMTMode::kValid, T, T, T>
            <<<grid, kIterThreads, 0, stream>>>(
                state_level_ptr<T>(l - 1), out, sums, copies, nullptr, nsamps_l,
                nsum, ncopy, 0, 0, 0, 0, m_chunk_pos, 1.0F,
                static_cast<int>(m_active), false, static_cast<int>(m_chunk));
        if (l == t.level + 1 && !t.carry_len.empty()) {
            // Level top_k has been read: carry each row's last spread
            // samples to the front for the next chunk.
            const auto nrows = static_cast<int>(t.carry_len.size());
            gpu_utils::check_kernel_launch_params(dim3(nrows), dim3(256));
            kernel_carry_front<T><<<nrows, 256, 0, stream>>>(
                state_level_ptr<T>(t.level),
                thrust::raw_pointer_cast(t.carry_off.data()),
                thrust::raw_pointer_cast(t.carry_len.data()),
                static_cast<int>(m_active));
        }
        if (l >= first_searched) {
            consume(l, static_cast<const T*>(out));
        }
    }
    gpu_utils::check_last_gpu_error("FDMT (cuda): windowed top failed");
    m_top_stale     = false;
    m_current_level = root;
}

template <typename T>
detail::FDMTSearchInput
FDMTCudaEngine::windowed_input(SizeType l, const T* state, Stream queue) const {
    detail::FDMTSearchInput in{
        .row_stride        = m_chunk,
        .nsamps            = m_active,
        .sample_begin      = m_chunk_pos,
        .materialized_root = true,
        .stream            = queue,
    };
    if constexpr (std::is_same_v<T, float>) {
        in.state = state;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        in.integer32_state = state;
    } else {
        in.integer_state = state;
    }
    if (l < m_plan.get_niters()) {
        in.row_offset = thrust::raw_pointer_cast(m_top->emit_off[l].data());
        in.row_lag    = thrust::raw_pointer_cast(m_top->emit_lag[l].data());
    }
    return in;
}

int FDMTCudaEngine::execute_iter_device(SizeType next_level,
                                        cudaStream_t stream,
                                        bool regular_only,
                                        bool advance_hist) {
    const int fused_coords =
        execute_iter_levels(next_level, stream, regular_only, advance_hist);
    // The tree produced the direct top's level: keep its rings current
    // (a top chunk stores them after its trials read them).
    if (m_top && next_level == m_top->level && !m_top_chunk &&
        m_mode == FDMTMode::kValid) {
        if (m_levels_d[next_level].type == Elem::kF32) {
            update_top_rings(stream);
        } else {
            m_top_filled = 0; // integer level: the rings must refill
        }
    }
    return fused_coords;
}

int FDMTCudaEngine::execute_iter_levels(SizeType next_level,
                                        cudaStream_t stream,
                                        bool regular_only,
                                        bool advance_hist) {
    int fused_coords = 0;
    with_elem(m_levels_d[next_level - 1].type, [&](auto tag_in) {
        using TIn = decltype(tag_in);
        with_elem(m_levels_d[next_level].type, [&](auto tag_out) {
            using TOut = decltype(tag_out);
            // Levels only ever widen (see detail::int_tree_level_types).
            if constexpr (sizeof(TIn) <= sizeof(TOut) &&
                          !(std::is_floating_point_v<TIn> &&
                            !std::is_floating_point_v<TOut>)) {
                fused_coords = execute_iter_typed<TIn, TOut>(
                    reinterpret_cast<const TIn*>(
                        m_levels_d[next_level - 1].base),
                    reinterpret_cast<TOut*>(m_levels_d[next_level].base),
                    next_level, stream, regular_only, advance_hist);
            } else {
                throw std::logic_error(
                    "FDMT (cuda): invalid narrowing level transition");
            }
        });
    });
    return fused_coords;
}

template <typename TIn, typename TOut>
int FDMTCudaEngine::execute_iter_typed(const TIn* __restrict__ in_ptr,
                                       TOut* __restrict__ out_ptr,
                                       SizeType next_level,
                                       cudaStream_t stream,
                                       bool regular_only,
                                       bool advance_hist) {
    const auto& pc    = m_plan.get_container();
    const auto& shape = pc.state_shape[next_level];
    int fused_coords  = 0;
    // Valid mode: this chunk's samples (rows keep their table strides).
    const bool valid  = m_mode == FDMTMode::kValid;
    const int active  = valid ? static_cast<int>(m_active) : 0;
    const int extra_l = (valid && next_level < m_level_max_extra.size())
                            ? static_cast<int>(m_level_max_extra[next_level])
                            : 0;
    const int nsamps =
        valid ? (active + extra_l) : static_cast<int>(shape.nsamps);
    const int ncoords_sum_cur = static_cast<int>(
        regular_only ? detail::fdmt_regular_sum_coords(pc, next_level)
                     : shape.ncoords_sum);
    const int ncoords_copy_cur = static_cast<int>(shape.ncoords_copy);

    auto coords_sum_cur  = m_plan_d.coordinates_sum.get_raw_ptrs();
    auto coords_copy_cur = m_plan_d.coordinates_copy.get_raw_ptrs();
    coords_sum_cur.update_offsets(m_coords_sum_offsets[next_level]);
    coords_copy_cur.update_offsets(m_coords_copy_offsets[next_level]);

    const auto coords_max     = std::max(ncoords_sum_cur, ncoords_copy_cur);
    constexpr int kBlockSamps = kIterThreads * kIterSampsPerTh;
    const dim3 block_size     = dim3(kIterThreads, 1);
    // Coordinates beyond grid.y's limit are strided in the kernel.
    const dim3 grid_size = dim3((nsamps + kBlockSamps - 1) / kBlockSamps,
                                std::min(coords_max, kMaxGridY), m_nbeams);
    gpu_utils::check_kernel_launch_params(grid_size, block_size);

    // Beams are level_beam_stride() elements apart in each level.
    const auto in_state_nelements  = level_beam_stride(next_level - 1);
    const auto out_state_nelements = level_beam_stride(next_level);
    const auto hist                = tree_history(next_level);

    if (m_mode == FDMTMode::kFull) {
        kernel_execute_iter<FDMTMode::kFull, TIn, TOut>
            <<<grid_size, block_size, 0, stream>>>(
                in_ptr, out_ptr, coords_sum_cur, coords_copy_cur, hist.in,
                nsamps, ncoords_sum_cur, ncoords_copy_cur, in_state_nelements,
                out_state_nelements, hist.stride, hist.base, hist.ring,
                m_int_unit, active);
    } else if (m_mode == FDMTMode::kRoll) {
        kernel_execute_iter<FDMTMode::kRoll, TIn, TOut>
            <<<grid_size, block_size, 0, stream>>>(
                in_ptr, out_ptr, coords_sum_cur, coords_copy_cur, hist.in,
                nsamps, ncoords_sum_cur, ncoords_copy_cur, in_state_nelements,
                out_state_nelements, hist.stride, hist.base, hist.ring,
                m_int_unit, active);
    } else {
        const bool skewed_mid =
            m_direct && m_direct->skewed && m_top && next_level <= m_top->level;
        const bool fuse_ring = valid && !skewed_mid &&
                               hist.ring != kFifoHistory &&
                               hist.out != nullptr && grid_size.x == 1;
        kernel_execute_iter<FDMTMode::kValid, TIn, TOut>
            <<<grid_size, block_size, 0, stream>>>(
                in_ptr, out_ptr, coords_sum_cur, coords_copy_cur,
                skewed_mid ? nullptr : hist.in, nsamps, ncoords_sum_cur,
                ncoords_copy_cur, in_state_nelements, out_state_nelements,
                hist.stride, hist.base, hist.ring, m_int_unit, active,
                fuse_ring, skewed_mid ? static_cast<int>(m_chunk) : 0);

        fused_coords = (skewed_mid || fuse_ring) ? ncoords_sum_cur : 0;
        if (advance_hist && !skewed_mid) {
            advance_tree_history<TIn>(in_ptr, next_level, stream, fused_coords);
        }
    }
    gpu_utils::check_last_gpu_error("kernel_execute_iter launch failed");
    return fused_coords;
}

template <typename TIn>
void FDMTCudaEngine::advance_tree_history(const TIn* in_ptr,
                                          SizeType level,
                                          cudaStream_t stream,
                                          int coord_begin) {
    const auto& shape = m_plan.get_container().state_shape[level];
    const int count   = static_cast<int>(shape.ncoords_sum) - coord_begin;
    const auto hist   = tree_history(level);
    // A ring takes only the chunk's newest min(delay, chunk) samples.
    const int max_offset = static_cast<int>(
        hist.ring == kFifoHistory ? shape.dt_max
                                  : std::min<SizeType>(shape.dt_max, m_active));
    if (hist.out == nullptr || count <= 0 || max_offset == 0) {
        return;
    }
    auto coords = m_plan_d.coordinates_sum.get_raw_ptrs();
    coords.update_offsets(m_coords_sum_offsets[level] + coord_begin);
    launch_advance_tree_history(in_ptr, coords, hist.in, hist.out, count,
                                max_offset, level_beam_stride(level - 1),
                                hist.stride, hist.base, hist.ring, m_int_unit,
                                stream);
    gpu_utils::check_last_gpu_error(
        "kernel_advance_tree_history launch failed");
}

template <FDMTMode Mode, bool Smear, typename Input, typename TOut>
void FDMTCudaEngine::launch_init(const Input& waterfall_d,
                                 TOut* __restrict__ state_d,
                                 dim3 grid_size,
                                 dim3 block_size,
                                 cudaStream_t stream) {
    const auto& plan_c = m_plan.get_container();
    kernel_init_fdmt<Mode, Smear, Input, TOut>
        <<<grid_size, block_size, 0, stream>>>(
            waterfall_d, state_d, m_plan_d.grids0.dt_grid.data().get(),
            m_plan_d.grids0.ndt.data().get(),
            m_plan_d.grids0.coord_offset.data().get(),
            static_cast<int>(plan_c.state_shape[0].nchans),
            static_cast<int>(m_active), static_cast<int>(m_row_len0),
            static_cast<int>(plan_c.state_shape[0].dt_max),
            level_beam_stride(0), m_level0_hist_in_ptr,
            box_means_for(waterfall_d));
}

template <typename Input, typename TOut>
void FDMTCudaEngine::initialise_device(const Input& waterfall_d,
                                       TOut* __restrict__ state_d,
                                       cudaStream_t stream) {
    const auto& plan_c    = m_plan.get_container();
    const int nsubs       = static_cast<int>(plan_c.state_shape[0].nchans);
    const int nsamps      = static_cast<int>(m_active);
    const int threads     = nsamps <= 256 ? 256 : (nsamps <= 512 ? 512 : 1024);
    const dim3 block_size = dim3(threads, 1);
    const dim3 grid_size =
        dim3((nsamps + block_size.x - 1) / block_size.x, nsubs, m_nbeams);
    gpu_utils::check_kernel_launch_params(grid_size, block_size);

    const auto launch = [&](auto mode_tag, auto smear_tag) {
        launch_init<decltype(mode_tag)::value, decltype(smear_tag)::value,
                    Input, TOut>(waterfall_d, state_d, grid_size, block_size,
                                 stream);
    };
    using Full  = std::integral_constant<FDMTMode, FDMTMode::kFull>;
    using Roll  = std::integral_constant<FDMTMode, FDMTMode::kRoll>;
    using Valid = std::integral_constant<FDMTMode, FDMTMode::kValid>;
    if (m_mode == FDMTMode::kFull) {
        if (m_use_box_smearing) {
            launch(Full{}, std::true_type{});
        } else {
            launch(Full{}, std::false_type{});
        }
    } else if (m_mode == FDMTMode::kRoll) {
        if (m_use_box_smearing) {
            launch(Roll{}, std::true_type{});
        } else {
            launch(Roll{}, std::false_type{});
        }
    } else {
        if (m_use_box_smearing) {
            launch(Valid{}, std::true_type{});
        } else {
            launch(Valid{}, std::false_type{});
        }
    }
    gpu_utils::check_last_gpu_error("kernel_init_fdmt launch failed");

    advance_level0_history(waterfall_d, stream);
}

template <typename Input>
void FDMTCudaEngine::advance_level0_history(const Input& waterfall_d,
                                            cudaStream_t stream) {
    const auto& plan_c = m_plan.get_container();
    const int nsubs    = static_cast<int>(plan_c.state_shape[0].nchans);
    const int nsamps   = static_cast<int>(m_active);
    // The last dt0_max raw samples of every channel.
    const int dt_max_final = static_cast<int>(plan_c.state_shape[0].dt_max);
    if (m_mode == FDMTMode::kValid && m_level0_hist_out_ptr != nullptr &&
        dt_max_final > 0) {
        const dim3 block_hist = dim3(256, 1);
        const dim3 grid_hist  = dim3(
            (dt_max_final + block_hist.x - 1) / block_hist.x, nsubs, m_nbeams);
        kernel_advance_history_window<Input>
            <<<grid_hist, block_hist, 0, stream>>>(
                m_level0_hist_out_ptr, m_level0_hist_in_ptr, waterfall_d, nsubs,
                nsamps, dt_max_final);
        gpu_utils::check_last_gpu_error(
            "kernel_advance_history_window launch failed");
    }
}

} // namespace dmt::algorithms::detail::fdmt_cuda
