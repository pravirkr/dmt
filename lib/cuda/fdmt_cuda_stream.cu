#include "fdmt_cuda_engine.cuh"

// CUDA FDMT engine: the stepper and the stream history.

namespace dmt::algorithms::detail::fdmt_cuda {

namespace {

/**
 * @brief Linearizes ring tree histories (kernel_execute_iter) into the
 * oldest-first layout of save_history(): element k of a coordinate with
 * delay d is the ring slot (pos + k) mod d, pos being the next chunk's
 * stream sample.
 */
__global__ void kernel_tree_ring_to_fifo(const float* __restrict__ ring,
                                         float* __restrict__ fifo,
                                         const plans::FDMTCoordDPtrs coords,
                                         int ncoords,
                                         int64_t ring_stride,
                                         int64_t fifo_stride,
                                         int64_t ring_base,
                                         uint64_t pos) {
    const auto k = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    for (int i = static_cast<int>(blockIdx.y); i < ncoords;
         i += static_cast<int>(gridDim.y)) {
        const int d = coords.offset[i];
        if (d <= 0 || k >= d) {
            continue;
        }
        const auto off = static_cast<int64_t>(coords.hist_offset[i]);
        int slot       = static_cast<int>(pos % static_cast<uint64_t>(d)) + k;
        slot -= slot >= d ? d : 0;
        fifo[(i_beam * fifo_stride) + off + k] =
            ring[(i_beam * ring_stride) + off - ring_base + slot];
    }
}

} // namespace

void FDMTCudaEngine::reset_search_history(Stream queue) {
    gpu_utils::set_device(m_device_id);
    const auto stream = to_cuda(queue);
    for (auto* v : {&m_history_a_d, &m_history_b_d, &m_tree_fifo_d[0],
                    &m_tree_fifo_d[1], &m_tree_ring_d}) {
        if (!v->empty()) {
            gpu_utils::check_gpu_call(
                cudaMemsetAsync(thrust::raw_pointer_cast(v->data()), 0,
                                v->size() * sizeof(float), stream));
        }
    }
    if (m_integer_ring.size() > 0) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(thrust::raw_pointer_cast(m_integer_ring.data()), 0,
                            m_integer_ring.size() * sizeof(uint64_t), stream));
    }
    if (m_integer32_ring.size() > 0) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            thrust::raw_pointer_cast(m_integer32_ring.data()), 0,
            m_integer32_ring.size() * sizeof(uint32_t), stream));
    }
    if (m_direct) {
        for (auto& w : m_direct->win) {
            if (!w.empty()) {
                gpu_utils::check_gpu_call(
                    cudaMemsetAsync(thrust::raw_pointer_cast(w.data()), 0,
                                    w.size() * sizeof(float), stream));
            }
        }
        if (!m_direct->win_ring.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                thrust::raw_pointer_cast(m_direct->win_ring.data()), 0,
                m_direct->win_ring.size() * sizeof(float), stream));
        }
        if (!m_direct->win_ring_u8.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                thrust::raw_pointer_cast(m_direct->win_ring_u8.data()), 0,
                m_direct->win_ring_u8.size() * sizeof(uint8_t), stream));
        }
        if (!m_direct->win_ring_u16.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                thrust::raw_pointer_cast(m_direct->win_ring_u16.data()), 0,
                m_direct->win_ring_u16.size() * sizeof(uint16_t), stream));
        }
        m_direct_filled = static_cast<SizeType>(m_direct->window);
    }
    m_direct_stale = false;
    if (m_top && !m_top->ring.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(thrust::raw_pointer_cast(m_top->ring.data()), 0,
                            m_top->ring.size() * sizeof(float), stream));
        m_top_filled = static_cast<SizeType>(m_top->max_ring);
    }
    m_top_stale      = false;
    m_parity         = false;
    m_is_initialized = false;
    m_chunk_pos      = 0;
    m_next_pos       = 0;
    m_device_work.mark(stream);
}

void FDMTCudaEngine::advance(SizeType levels, Stream stream_in) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    gpu_utils::set_device(m_device_id);
    m_host_level_valid         = false;
    cudaStream_t stream        = to_cuda(stream_in);
    cudaStream_t active_stream = stream ? stream : m_stream;
    const auto total_lvl       = total_levels();
    const auto target_level = std::min(m_current_level + levels, total_lvl - 1);

    while (m_current_level < target_level) {
        execute_iter_device(m_current_level + 1, active_stream);
        m_current_level++;
    }
    m_device_work.mark(active_stream);
}

void FDMTCudaEngine::advance_until_remaining(SizeType remaining_levels,
                                             Stream stream) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    const auto total_lvl = total_levels();
    if (remaining_levels >= total_lvl) {
        return;
    }
    const auto target_level = total_lvl - 1 - remaining_levels;
    if (target_level > m_current_level) {
        advance(target_level - m_current_level, stream);
    }
}

DeviceSpan<const float> FDMTCudaEngine::view_level_data_device() const {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    const auto& shape = m_plan.get_container().state_shape[m_current_level];
    // The span covers through the last beam's valid elements.
    return {current_level_f32(),
            ((m_nbeams - 1) * level_beam_stride(m_current_level)) +
                shape.nelements,
            device()};
}

std::span<const float> FDMTCudaEngine::view_level_data() const {
    const float* level = host_level_snapshot();
    return {level,
            m_plan.get_container().state_shape[m_current_level].nelements};
}

cuda::std::span<const float>
FDMTCudaEngine::view_subband_data(SizeType subband_idx) const {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    check_subband_idx(subband_idx);
    const auto& grid =
        m_plan.get_container().grids[m_current_level][subband_idx];
    const auto nsamps =
        m_plan.get_container().state_shape[m_current_level].nsamps;
    const auto offset = grid.coord_offset * nsamps;
    const auto count  = grid.ndt * nsamps;
    return cuda::std::span<const float>(current_level_f32() + offset, count);
}

FDMTSubbandView FDMTCudaEngine::view_subband(SizeType subband_idx) const {
    const float* level = host_level_snapshot();
    check_subband_idx(subband_idx);
    const auto& grid =
        m_plan.get_container().grids[m_current_level][subband_idx];
    const auto nsamps =
        m_plan.get_container().state_shape[m_current_level].nsamps;
    return FDMTSubbandView{
        .data = std::span<const float>(level + (grid.coord_offset * nsamps),
                                       grid.ndt * nsamps),
        .subband_idx = subband_idx,
        .ndt         = grid.ndt,
        .nsamps      = nsamps,
        .f_start     = grid.f_start,
        .f_end       = grid.f_end,
        .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                  grid.dt_grid.size()),
        .chan_begin  = grid.chan_begin,
        .chan_end    = grid.chan_end,
        .staggered   = grid.staggered,
    };
}

FDMTSubbandDeviceView
FDMTCudaEngine::view_subband_device(SizeType subband_idx) const {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    check_subband_idx(subband_idx);
    const auto& grid =
        m_plan.get_container().grids[m_current_level][subband_idx];
    const auto nsamps =
        m_plan.get_container().state_shape[m_current_level].nsamps;
    const auto data = view_subband_data(subband_idx);
    return FDMTSubbandDeviceView{
        .data = DeviceSpan<const float>(data.data(), data.size(), device()),
        .subband_idx = subband_idx,
        .ndt         = grid.ndt,
        .nsamps      = nsamps,
        .f_start     = grid.f_start,
        .f_end       = grid.f_end,
        .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                  grid.dt_grid.size()),
        .chan_begin  = grid.chan_begin,
        .chan_end    = grid.chan_end,
        .staggered   = grid.staggered,
    };
}

void FDMTCudaEngine::finalize(Stream stream_in) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    gpu_utils::set_device(m_device_id);
    cudaStream_t stream        = to_cuda(stream_in);
    cudaStream_t active_stream = stream ? stream : m_stream;
    advance_until_remaining(0, Stream{active_stream});

    // The root is written straight into dmt: nothing to copy.
    m_device_work.mark(active_stream);
    if (!m_host_dmt.empty()) {
        // Host-memory stepper: the result goes back to the caller's
        // buffer (through the staging stream).
        gpu_utils::check_gpu_call(cudaStreamSynchronize(active_stream));
        copy_result_to_host(m_host_dmt);
        m_host_dmt = {};
    }
    m_host_level_valid = false;
    m_is_initialized   = false;
}

void FDMTCudaEngine::reset_history() noexcept {
    m_chunk_pos     = 0;
    m_next_pos      = 0;
    m_hist_pos_base = 0;
    gpu_utils::set_device(m_device_id);
    for (auto* v : {&m_history_a_d, &m_history_b_d, &m_tree_fifo_d[0],
                    &m_tree_fifo_d[1], &m_tree_ring_d}) {
        if (!v->empty()) {
            v->assign(v->size(), 0.0F);
        }
    }
    if (!m_integer_ring.empty()) {
        m_integer_ring.assign(m_integer_ring.size(), 0);
    }
    if (!m_integer32_ring.empty()) {
        m_integer32_ring.assign(m_integer32_ring.size(), 0);
    }
    if (m_direct) {
        for (auto& w : m_direct->win) {
            w.assign(w.size(), 0.0F);
        }
        if (!m_direct->win_ring.empty()) {
            m_direct->win_ring.assign(m_direct->win_ring.size(), 0.0F);
        }
        if (!m_direct->win_ring_u8.empty()) {
            m_direct->win_ring_u8.assign(m_direct->win_ring_u8.size(), 0);
        }
        if (!m_direct->win_ring_u16.empty()) {
            m_direct->win_ring_u16.assign(m_direct->win_ring_u16.size(), 0);
        }
        // An all-zero window is consistent with all-zero tree histories.
        m_direct_filled = static_cast<SizeType>(m_direct->window);
    }
    m_direct_stale = false;
    if (m_top) {
        m_top->ring.assign(m_top->ring.size(), 0.0F);
        m_top->win_top.assign(m_top->win_top.size(), 0.0F);
        m_top_filled = static_cast<SizeType>(m_top->max_ring);
    }
    m_top_stale = false;
    m_parity    = false;
    // Abandons any unfinished stepper block along with its stream.
    m_is_initialized = false;
}

SizeType FDMTCudaEngine::history_state_size() const noexcept {
    if (skew_history()) {
        SizeType n = kSkewHistHeader;
        for (const auto& b : skew_hist_buffers()) {
            n += b.size;
        }
        return n;
    }
    // The header (fdmt_history.hpp), then the level-0 input history and the
    // tree history, as on the CPU; 0 without a history.
    const SizeType n =
        m_mode == FDMTMode::kValid && m_search_nbits == 32
            ? m_history_a_d.size() +
                  (m_plan.get_tree_history_size() == 0
                       ? 0
                       : m_nbeams * m_plan.get_tree_history_size())
            : 0;
    return n == 0 ? 0 : detail::kFDMTHistHeader + n;
}

void FDMTCudaEngine::save_history(DeviceSpan<float> out_d,
                                  Stream stream_in) const {
    const cuda::std::span<float> out = to_cuda(out_d);
    if (out.size() != history_state_size()) {
        throw std::invalid_argument(
            fmt::format("FDMT::save_history: Invalid output size. Expected {}, "
                        "got {}",
                        history_state_size(), out.size()));
    }
    gpu_utils::set_device(m_device_id);
    save_history_d(out.data(), to_cuda(stream_in));
    m_device_work.mark(to_cuda(stream_in));
}

void FDMTCudaEngine::load_history(DeviceSpan<const float> in_d,
                                  Stream stream_in) {
    const cuda::std::span<const float> in = to_cuda(in_d);
    if (in.size() != history_state_size()) {
        detail::throw_fdmt_history_size(history_state_size(), in.size());
    }
    gpu_utils::set_device(m_device_id);
    load_history_d(in.data(), to_cuda(stream_in));
    m_device_work.mark(to_cuda(stream_in));
}

void FDMTCudaEngine::save_history(std::span<float> out) const {
    if (out.size() != history_state_size()) {
        throw std::invalid_argument(
            fmt::format("FDMT::save_history: Invalid output size. Expected {}, "
                        "got {}",
                        history_state_size(), out.size()));
    }
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    thrust::device_vector<float> staged(out.size());
    save_history_d(thrust::raw_pointer_cast(staged.data()), nullptr);
    gpu_utils::check_gpu_call(
        cudaMemcpy(out.data(), thrust::raw_pointer_cast(staged.data()),
                   out.size_bytes(), cudaMemcpyDeviceToHost),
        "FDMT::save_history");
}

void FDMTCudaEngine::load_history(std::span<const float> in) {
    if (in.size() != history_state_size()) {
        detail::throw_fdmt_history_size(history_state_size(), in.size());
    }
    if (!skew_history() && !in.empty()) {
        // Before the staging copy, so a wrong buffer replaces nothing.
        static_cast<void>(
            detail::check_fdmt_tree_history_header(in, m_hist_hash));
    }
    gpu_utils::set_device(m_device_id);
    m_device_work.wait();
    thrust::device_vector<float> staged(in.begin(), in.end());
    load_history_d(thrust::raw_pointer_cast(staged.data()), nullptr);
    gpu_utils::check_gpu_call(cudaStreamSynchronize(nullptr),
                              "FDMT::load_history");
}

const float* FDMTCudaEngine::host_level_snapshot() const {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT (cuda): Stepper is not initialized. Call reset() first.");
    }
    const float* level_d = current_level_f32();
    if (!m_host_level_valid) {
        const auto nelements =
            m_plan.get_container().state_shape[m_current_level].nelements;
        m_host_level.resize(nelements);
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        cudaMemcpy(m_host_level.data(), level_d, nelements * sizeof(float),
                   cudaMemcpyDeviceToHost);
        gpu_utils::check_last_gpu_error("FDMT::view (host): D->H copy failed");
        m_host_level_valid = true;
    }
    return m_host_level.data();
}

void FDMTCudaEngine::check_subband_idx(SizeType subband_idx) const {
    const auto nsubs = m_plan.get_container().grids[m_current_level].size();
    if (subband_idx >= nsubs) {
        throw std::out_of_range(
            fmt::format("FDMT (cuda): Subband index {} out of range "
                        "(current level has {} "
                        "subbands)",
                        subband_idx, nsubs));
    }
}

void FDMTCudaEngine::ensure_full_tree_history(cudaStream_t stream) const {
    if (m_full_tree_ring || m_mode != FDMTMode::kValid ||
        m_search_nbits != 32) {
        return;
    }
    check_tree_ring_extent(m_plan.get_tree_history_size());
    const SizeType full_ring = m_plan.get_tree_history_size() - m_fifo_size;
    thrust::device_vector<float> expanded(m_nbeams * full_ring, 0.0F);
    if (m_ring_low_size > 0 && !m_tree_ring_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemcpy2DAsync(thrust::raw_pointer_cast(expanded.data()) +
                                  (m_ring_low_base - m_fifo_size),
                              full_ring * sizeof(float),
                              thrust::raw_pointer_cast(m_tree_ring_d.data()),
                              m_ring_size * sizeof(float),
                              m_ring_low_size * sizeof(float), m_nbeams,
                              cudaMemcpyDeviceToDevice, stream),
            "FDMT::ensure_full_tree_history (low)");
    }
    if (m_ring_high_size > 0 && !m_tree_ring_d.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemcpy2DAsync(thrust::raw_pointer_cast(expanded.data()) +
                                  (m_ring_high_orig_base - m_fifo_size),
                              full_ring * sizeof(float),
                              thrust::raw_pointer_cast(m_tree_ring_d.data()) +
                                  m_ring_low_size,
                              m_ring_size * sizeof(float),
                              m_ring_high_size * sizeof(float), m_nbeams,
                              cudaMemcpyDeviceToDevice, stream),
            "FDMT::ensure_full_tree_history (high)");
    }
    m_tree_ring_d.swap(expanded);
    m_ring_size      = full_ring;
    m_full_tree_ring = true;
}

std::vector<FDMTCudaEngine::SkewHistBuffer>
FDMTCudaEngine::skew_hist_buffers() const {
    std::vector<SkewHistBuffer> v;
    const auto add = [&](const thrust::device_vector<float>& b) {
        if (!b.empty()) {
            v.push_back({const_cast<float*>( // NOLINT
                             thrust::raw_pointer_cast(b.data())),
                         b.size()});
        }
    };
    add(m_direct->win_ring);
    if (m_top) {
        add(m_top->ring);
        add(m_top->win_top); // windowed top: carried level-top_k spreads
    }
    add(m_tree_ring_d);
    add(next_tree_fifo());
    add(next_level0_history());
    return v;
}

uint64_t FDMTCudaEngine::skew_layout_hash() const {
    uint64_t h     = 1469598103934665603ULL;
    const auto mix = [&](uint64_t x) {
        for (int i = 0; i < 8; ++i) {
            h = (h ^ ((x >> (8 * i)) & 0xFFU)) * 1099511628211ULL;
        }
    };
    mix(m_plan.get_nchans());
    mix(m_nbeams);
    mix(m_chunk);
    mix(m_direct->levels);
    mix(static_cast<uint64_t>(m_direct->win_ring_size));
    mix(m_top ? m_top->level : 0);
    mix(m_top ? static_cast<uint64_t>(m_top->ring_size) : 0);
    for (const auto& b : skew_hist_buffers()) {
        mix(b.size);
    }
    return h;
}

void FDMTCudaEngine::save_skew_history_d(float* dst,
                                         cudaStream_t stream) const {
    std::array<float, kSkewHistHeader> hdr{};
    const auto put16 = [&](SizeType at, uint64_t x, SizeType words) {
        for (SizeType i = 0; i < words; ++i) {
            hdr[at + i] = static_cast<float>((x >> (16 * i)) & 0xFFFFU);
        }
    };
    put16(0, kSkewHistMagic, 1);
    put16(1, kSkewHistVersion, 1);
    put16(2, skew_layout_hash(), 4);
    put16(6, m_next_pos, 4);
    // Pageable source: returns once the header has been staged.
    gpu_utils::check_gpu_call(cudaMemcpyAsync(dst, hdr.data(), sizeof(hdr),
                                              cudaMemcpyHostToDevice, stream),
                              "FDMT::save_history");
    SizeType off = kSkewHistHeader;
    for (const auto& b : skew_hist_buffers()) {
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(dst + off, b.data, b.size * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMT::save_history");
        off += b.size;
    }
}

void FDMTCudaEngine::load_skew_history_d(const float* src,
                                         cudaStream_t stream) {
    std::array<float, kSkewHistHeader> hdr{};
    gpu_utils::check_gpu_call(cudaMemcpyAsync(hdr.data(), src, sizeof(hdr),
                                              cudaMemcpyDeviceToHost, stream),
                              "FDMT::load_history");
    gpu_utils::check_gpu_call(cudaStreamSynchronize(stream),
                              "FDMT::load_history");
    const auto get16 = [&](SizeType at, SizeType words) {
        uint64_t x = 0;
        for (SizeType i = 0; i < words; ++i) {
            const float f = hdr[at + i];
            if (!(f >= 0.0F && f <= 65535.0F) || f != std::floor(f)) {
                throw std::invalid_argument(
                    "FDMT::load_history: not a skewed-window history");
            }
            x |= static_cast<uint64_t>(f) << (16 * i);
        }
        return x;
    };
    if (get16(0, 1) != kSkewHistMagic || get16(1, 1) != kSkewHistVersion) {
        throw std::invalid_argument(
            "FDMT::load_history: not a skewed-window history (it was "
            "saved by a different engine path)");
    }
    if (get16(2, 4) != skew_layout_hash()) {
        throw std::invalid_argument(
            "FDMT::load_history: the history was saved by an engine with "
            "a different skewed-window layout (plan, chunk_nsamps, beams "
            "or direct levels)");
    }
    const uint64_t pos = get16(6, 4);
    SizeType off       = kSkewHistHeader;
    for (const auto& b : skew_hist_buffers()) {
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(b.data, src + off, b.size * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMT::load_history");
        off += b.size;
    }
    m_chunk_pos      = pos;
    m_next_pos       = pos;
    m_is_initialized = false;
    m_direct_stale   = false;
    m_top_stale      = false;
    m_direct_filled  = static_cast<SizeType>(m_direct->window);
    if (m_top) {
        m_top_filled = static_cast<SizeType>(m_top->max_ring);
    }
}

void FDMTCudaEngine::save_history_d(float* dst, cudaStream_t stream) const {
    if (skew_history()) {
        save_skew_history_d(dst, stream);
        return;
    }
    if (history_state_size() == 0) {
        return;
    }
    std::array<float, detail::kFDMTHistHeader> hdr{};
    detail::write_fdmt_tree_history_header(hdr, m_hist_hash,
                                           m_hist_pos_base + m_next_pos);
    // Pageable source: returns once the header has been staged.
    gpu_utils::check_gpu_call(cudaMemcpyAsync(dst, hdr.data(), sizeof(hdr),
                                              cudaMemcpyHostToDevice, stream),
                              "FDMT::save_history");
    dst += detail::kFDMTHistHeader;
    ensure_full_tree_history(stream);
    rebuild_direct_history(stream);
    rebuild_top_history(stream);
    const auto& level0 = next_level0_history();
    if (!level0.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(dst, thrust::raw_pointer_cast(level0.data()),
                            level0.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMT::save_history");
    }
    float* tree      = dst + level0.size();
    const auto total = m_plan.get_tree_history_size();
    if (m_fifo_size > 0) {
        gpu_utils::check_gpu_call(
            cudaMemcpy2DAsync(tree, total * sizeof(float),
                              thrust::raw_pointer_cast(next_tree_fifo().data()),
                              m_fifo_size * sizeof(float),
                              m_fifo_size * sizeof(float), m_nbeams,
                              cudaMemcpyDeviceToDevice, stream),
            "FDMT::save_history");
    }
    for_each_ring_level([&](plans::FDMTCoordDPtrs coords, int count,
                            int max_delay) {
        const dim3 block(256);
        const dim3 grid((max_delay + 255) / 256, std::min(count, kMaxGridY),
                        m_nbeams);
        kernel_tree_ring_to_fifo<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(m_tree_ring_d.data()), tree, coords, count,
            static_cast<int64_t>(m_ring_size), static_cast<int64_t>(total),
            static_cast<int64_t>(m_fifo_size), m_next_pos);
    });
    gpu_utils::check_last_gpu_error("FDMT::save_history failed");
}

void FDMTCudaEngine::load_history_d(const float* src, cudaStream_t stream) {
    if (skew_history()) {
        load_skew_history_d(src, stream);
        return;
    }
    uint64_t pos_base = 0;
    if (history_state_size() != 0) {
        // The header is checked on the host before anything is replaced.
        std::array<float, detail::kFDMTHistHeader> hdr{};
        gpu_utils::check_gpu_call(cudaMemcpyAsync(hdr.data(), src, sizeof(hdr),
                                                  cudaMemcpyDeviceToHost,
                                                  stream),
                                  "FDMT::load_history");
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream),
                                  "FDMT::load_history");
        pos_base = detail::check_fdmt_tree_history_header(hdr, m_hist_hash);
        src += detail::kFDMTHistHeader;
    }
    m_hist_pos_base = pos_base;
    ensure_full_tree_history(stream);
    auto& level0 = next_level0_history();
    if (!level0.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(level0.data()), src,
                            level0.size() * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMT::load_history");
    }
    const float* tree = src + level0.size();
    const auto total  = m_plan.get_tree_history_size();
    const auto copy   = [&](float* dst, SizeType pitch, SizeType from,
                            SizeType width) {
        if (width > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(dst, pitch * sizeof(float), tree + from,
                                  total * sizeof(float), width * sizeof(float),
                                  m_nbeams, cudaMemcpyDeviceToDevice, stream),
                "FDMT::load_history");
        }
    };
    if (!m_tree_fifo_d[0].empty() || !m_tree_ring_d.empty()) {
        copy(thrust::raw_pointer_cast(next_tree_fifo().data()), m_fifo_size, 0,
             m_fifo_size);
        copy(thrust::raw_pointer_cast(m_tree_ring_d.data()), m_ring_size,
             m_fifo_size, m_ring_size);
    }
    m_chunk_pos      = 0;
    m_next_pos       = 0;
    m_is_initialized = false;
    // The loaded format holds no input window: execute() runs the tree
    // (from the loaded histories) until the window has refilled.
    m_direct_filled = 0;
    m_direct_stale  = false;
    m_top_filled    = 0;
    m_top_stale     = false;
}

template <typename F> void FDMTCudaEngine::for_each_ring_level(F&& f) const {
    if (m_tree_ring_d.empty()) {
        return;
    }
    const auto& pc   = m_plan.get_container();
    const auto first = m_fifo_levels + 1;
    int count        = 0;
    int max_delay    = 0;
    for (SizeType l = first; l <= m_plan.get_niters(); ++l) {
        count += static_cast<int>(pc.state_shape[l].ncoords_sum);
        max_delay =
            std::max(max_delay, static_cast<int>(pc.state_shape[l].dt_max));
    }
    if (count == 0 || max_delay == 0) {
        return;
    }
    auto coords = m_plan_d.coordinates_sum.get_raw_ptrs();
    coords.update_offsets(m_coords_sum_offsets[first]);
    f(coords, count, max_delay);
}

} // namespace dmt::algorithms::detail::fdmt_cuda
