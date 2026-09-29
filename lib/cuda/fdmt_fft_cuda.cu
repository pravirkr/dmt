#include "dmt/algorithms/fdmt_fft.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "dmt/fft_cuda.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/modes.hpp"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms {

namespace {

__global__ void kernel_fill_window(const float* __restrict__ wf,
                                   const float* __restrict__ overlap,
                                   float* __restrict__ window,
                                   int nchans,
                                   int nsamps,
                                   int n_fft,
                                   int overlap_len,
                                   int mode,
                                   int nbeams) {
    const auto t = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_chan = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (t >= n_fft || i_chan >= nchans || i_beam >= nbeams) {
        return;
    }
    const auto out_idx = (((i_beam * nchans) + i_chan) * n_fft) + t;
    float val          = 0.0F;
    if (mode == 2) { // roll
        if (t < nsamps) {
            val = wf[(((i_beam * nchans) + i_chan) * nsamps) + t];
        }
    } else if (mode == 0) { // full: pad zeros at the end
        if (t < nsamps) {
            val = wf[(((i_beam * nchans) + i_chan) * nsamps) + t];
        }
    } else { // valid: [overlap | block]
        if (t < overlap_len) {
            if (overlap != nullptr) {
                val = overlap[(((i_beam * nchans) + i_chan) * overlap_len) + t];
            }
        } else if (t < overlap_len + nsamps) {
            val =
                wf[(((i_beam * nchans) + i_chan) * nsamps) + (t - overlap_len)];
        }
    }
    window[out_idx] = val;
}

__global__ void kernel_update_overlap(const float* __restrict__ wf,
                                      const float* __restrict__ overlap_in,
                                      float* __restrict__ overlap_out,
                                      int nchans,
                                      int nsamps,
                                      int overlap_len,
                                      int nbeams) {
    const auto i = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_chan = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (i >= overlap_len || i_chan >= nchans || i_beam >= nbeams) {
        return;
    }
    const auto base  = ((i_beam * nchans) + i_chan) * overlap_len;
    const float* src = wf + ((((i_beam * nchans) + i_chan) * nsamps));
    float val        = 0.0F;
    if (nsamps >= overlap_len) {
        val = src[nsamps - overlap_len + i];
    } else if (i < overlap_len - nsamps) {
        val = overlap_in[base + i + nsamps];
    } else {
        val = src[i - (overlap_len - nsamps)];
    }
    overlap_out[base + i] = val;
}

__global__ void
kernel_init_fdmt_fft(const ComplexTypeGPU* __restrict__ spectra,
                     const ComplexTypeGPU* __restrict__ win_table,
                     ComplexTypeGPU* __restrict__ state,
                     const int* __restrict__ grids0_coord_offset_ptr,
                     const int* __restrict__ grids0_ndt_ptr,
                     const int* __restrict__ grids0_dt_grid_ptr,
                     int nsubs,
                     int n_bins,
                     int max_coords) {
    const auto k = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (k >= n_bins || i_sub >= nsubs) {
        return;
    }
    const auto coord_base  = grids0_coord_offset_ptr[i_sub];
    const auto ndt         = grids0_ndt_ptr[i_sub];
    const auto* dt_grid    = &grids0_dt_grid_ptr[coord_base];
    const auto spec_offset = (i_beam * nsubs * n_bins) + (i_sub * n_bins) + k;
    const ComplexTypeGPU sample = spectra[spec_offset];
    const auto beam_state_base  = i_beam * max_coords * n_bins;
    for (int i_dt = 0; i_dt < ndt; ++i_dt) {
        const int s              = abs(dt_grid[i_dt]);
        const ComplexTypeGPU win = win_table[(s * n_bins) + k];
        const auto coord_idx     = coord_base + i_dt;
        state[beam_state_base + (coord_idx * n_bins) + k] = sample * win;
    }
}

__global__ void
kernel_execute_iter_fft(const ComplexTypeGPU* __restrict__ state_in,
                        ComplexTypeGPU* __restrict__ state_out,
                        const plans::FDMTCoordDPtrs coords_sum,
                        const plans::FDMTCoordDPtrs coords_copy,
                        const ComplexTypeGPU* __restrict__ phasors,
                        int n_bins,
                        int ncoords_sum_cur,
                        int ncoords_copy_cur,
                        int max_coords) {
    const auto linear =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam         = static_cast<int64_t>(blockIdx.y);
    const int max_iter_coords = (ncoords_sum_cur > ncoords_copy_cur)
                                    ? ncoords_sum_cur
                                    : ncoords_copy_cur;
    const int total           = max_iter_coords * n_bins;
    if (linear >= total) {
        return;
    }
    const int k       = linear % n_bins;
    const int i_coord = linear / n_bins;

    if (i_coord < ncoords_sum_cur) {
        const auto s        = coords_sum.offset[i_coord];
        const auto tail_idx = coords_sum.tail_buf_offset[i_coord] /
                              coords_sum.tail_nsamps[i_coord];
        const auto head_idx = coords_sum.head_buf_offset[i_coord] /
                              coords_sum.head_nsamps[i_coord];
        const auto out_idx =
            coords_sum.buf_offset[i_coord] / coords_sum.nsamps[i_coord];
        const auto tail_base =
            (i_beam * max_coords * n_bins) + (tail_idx * n_bins);
        const auto head_base =
            (i_beam * max_coords * n_bins) + (head_idx * n_bins);
        const auto out_base =
            (i_beam * max_coords * n_bins) + (out_idx * n_bins);
        const ComplexTypeGPU tail = state_in[tail_base + k];
        const ComplexTypeGPU head = state_in[head_base + k];
        const ComplexTypeGPU p    = phasors[(s * n_bins) + k];
        state_out[out_base + k]   = tail + (head * p);
    }
    if (i_coord < ncoords_copy_cur) {
        const auto tail_idx = coords_copy.tail_buf_offset[i_coord] /
                              coords_copy.tail_nsamps[i_coord];
        const auto out_idx =
            coords_copy.buf_offset[i_coord] / coords_copy.nsamps[i_coord];
        const auto tail_base =
            (i_beam * max_coords * n_bins) + (tail_idx * n_bins);
        const auto out_base =
            (i_beam * max_coords * n_bins) + (out_idx * n_bins);
        state_out[out_base + k] = state_in[tail_base + k];
    }
}

// (n * k) mod N for n, k < N < 2^31, exact: Barrett reduction with
// magic = floor((2^64 - 1) / N), whose quotient is at most one short (no
// 64-bit division in the hot loop).
__device__ __forceinline__ unsigned
fft_mod_mul(unsigned n, unsigned k, unsigned big_n, unsigned long long magic) {
    const unsigned long long nk =
        static_cast<unsigned long long>(n) * static_cast<unsigned long long>(k);
    const unsigned long long q = __umul64hi(nk, magic);
    unsigned long long r       = nk - (q * big_n);
    if (r >= big_n) {
        r -= big_n;
    }
    return static_cast<unsigned>(r);
}

// kernel_execute_iter_fft with real-valued merge shifts (fractional delays):
// W^(s k) for s = s_int + s_frac, the integer part reduced exactly modulo N
// (fft_mod_mul), the fractional part added in float, then sin/cos of the
// turn count reduced to [-1/2, 1/2].
__global__ void
kernel_execute_iter_fft_frac(const ComplexTypeGPU* __restrict__ state_in,
                             ComplexTypeGPU* __restrict__ state_out,
                             const plans::FDMTCoordDPtrs coords_sum,
                             const plans::FDMTCoordDPtrs coords_copy,
                             const int* __restrict__ shift_int,
                             const float* __restrict__ shift_frac,
                             int n_fft,
                             unsigned long long magic,
                             int n_bins,
                             int ncoords_sum_cur,
                             int ncoords_copy_cur,
                             int max_coords) {
    const auto linear =
        static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_beam         = static_cast<int64_t>(blockIdx.y);
    const int max_iter_coords = (ncoords_sum_cur > ncoords_copy_cur)
                                    ? ncoords_sum_cur
                                    : ncoords_copy_cur;
    const int total           = max_iter_coords * n_bins;
    if (linear >= total) {
        return;
    }
    const int k       = linear % n_bins;
    const int i_coord = linear / n_bins;

    if (i_coord < ncoords_sum_cur) {
        const auto tail_idx = coords_sum.tail_buf_offset[i_coord] /
                              coords_sum.tail_nsamps[i_coord];
        const auto head_idx = coords_sum.head_buf_offset[i_coord] /
                              coords_sum.head_nsamps[i_coord];
        const auto out_idx =
            coords_sum.buf_offset[i_coord] / coords_sum.nsamps[i_coord];
        const auto beam_base = i_beam * max_coords * n_bins;
        const ComplexTypeGPU tail =
            state_in[beam_base + (tail_idx * n_bins) + k];
        const ComplexTypeGPU head =
            state_in[beam_base + (head_idx * n_bins) + k];
        const auto m = fft_mod_mul(static_cast<unsigned>(shift_int[i_coord]),
                                   static_cast<unsigned>(k),
                                   static_cast<unsigned>(n_fft), magic);
        float u      = fmaf(shift_frac[i_coord], static_cast<float>(k),
                            static_cast<float>(m)) /
                       static_cast<float>(n_fft);
        u -= rintf(u);
        float sn = 0.0F;
        float cs = 0.0F;
        sincospif(-2.0F * u, &sn, &cs);
        state_out[beam_base + (out_idx * n_bins) + k] =
            tail + (head * ComplexTypeGPU(cs, sn));
    }
    if (i_coord < ncoords_copy_cur) {
        const auto tail_idx = coords_copy.tail_buf_offset[i_coord] /
                              coords_copy.tail_nsamps[i_coord];
        const auto out_idx =
            coords_copy.buf_offset[i_coord] / coords_copy.nsamps[i_coord];
        const auto beam_base = i_beam * max_coords * n_bins;
        state_out[beam_base + (out_idx * n_bins) + k] =
            state_in[beam_base + (tail_idx * n_bins) + k];
    }
}

__global__ void kernel_trim_scale(const float* __restrict__ time_out,
                                  float* __restrict__ dmt,
                                  int ndms,
                                  int n_fft,
                                  int nsamps_out,
                                  int skip,
                                  int max_coords,
                                  int nbeams,
                                  float norm) {
    const auto t    = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_dm = static_cast<int>(blockIdx.y);
    // 64-bit: beam-strided offsets (i_beam * per-beam size) exceed
    // int32 for large blocks with several beams.
    const auto i_beam = static_cast<int64_t>(blockIdx.z);
    if (t >= nsamps_out || i_dm >= ndms || i_beam >= nbeams) {
        return;
    }
    const auto src =
        time_out[(((i_beam * max_coords) + i_dm) * n_fft) + skip + t];
    dmt[(((i_beam * ndms) + i_dm) * nsamps_out) + t] = src * norm;
}

__global__ void kernel_materialize_view(const float* __restrict__ time_out,
                                        float* __restrict__ view,
                                        int ncoords,
                                        int n_fft,
                                        int nsamps_view,
                                        int skip,
                                        int max_coords,
                                        float norm) {
    const auto t   = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_c = static_cast<int>(blockIdx.y);
    if (t >= nsamps_view || i_c >= ncoords) {
        return;
    }
    const auto src                = time_out[(i_c * n_fft) + skip + t];
    view[(i_c * nsamps_view) + t] = src * norm;
}

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

class FDMTFFTCudaEngine final : public detail::FDMTFFTEngine {
public:
    FDMTFFTCudaEngine(const plans::FDMTPlan& plan,
                      const detail::FDMTFFTEngineConfig& cfg)
        : m_nchans(plan.get_nchans()),
          m_nsamps(plan.get_nsamps()),
          m_nbeams(cfg.nbeams),
          m_device_id(cfg.exec.device),
          m_use_box_smearing(cfg.use_box_smearing),
          m_mode(cfg.mode),
          m_frac(cfg.fractional_delays),
          m_plan(&plan) {
        initialize();
    }

    // Device-memory and stream entry points: forward to the cuda::std::span
    // / cudaStream_t overloads below, then mark the end of their work for
    // the next host-memory call.
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        execute(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream));
        m_device_work.mark(to_cuda(stream));
    }
    void reset(DeviceSpan<const float> d_waterfall,
               DeviceSpan<float> d_dmt,
               Stream stream) override {
        reset(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream));
        m_device_work.mark(to_cuda(stream));
    }
    void advance(SizeType levels, Stream stream) override {
        advance(levels, to_cuda(stream));
        m_device_work.mark(active(stream));
    }
    void advance_until_remaining(SizeType remaining_levels,
                                 Stream stream) override {
        advance_until_remaining(remaining_levels, to_cuda(stream));
        m_device_work.mark(active(stream));
    }
    void finalize(Stream stream) override {
        const cudaStream_t used = active(stream);
        finalize(to_cuda(stream));
        m_device_work.mark(used);
    }

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: expected waterfall size {}, got {}",
                total_in, waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: dmt buffer size {} must be >= {}",
                dmt.size(), total_out));
        }
        m_waterfall_d.resize(total_in);
        m_dmt_host_d.resize(total_out);
        gpu_utils::check_gpu_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_waterfall_d.data()),
                       waterfall.data(), total_in * sizeof(float),
                       cudaMemcpyHostToDevice),
            "FDMTFFT::execute: H2D waterfall");
        execute(cuda::std::span<const float>(
                    thrust::raw_pointer_cast(m_waterfall_d.data()), total_in),
                cuda::std::span<float>(
                    thrust::raw_pointer_cast(m_dmt_host_d.data()), total_out),
                nullptr);
        gpu_utils::check_gpu_call(cudaDeviceSynchronize(),
                                  "FDMTFFT::execute: sync");
        gpu_utils::check_gpu_call(
            cudaMemcpy(dmt.data(),
                       thrust::raw_pointer_cast(m_dmt_host_d.data()),
                       total_out * sizeof(float), cudaMemcpyDeviceToHost),
            "FDMTFFT::execute: D2H dmt");
    }

    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream) {
        run_transform(d_waterfall, d_dmt, stream, true);
    }

    void reset(std::span<const float> waterfall,
               std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        m_device_work.wait();
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: expected waterfall size {}, got {}", total_in,
                waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(
                std::format("FDMTFFT::reset: dmt buffer size {} must be >= {}",
                            dmt.size(), total_out));
        }
        m_waterfall_d.resize(total_in);
        m_dmt_host_d.resize(total_out);
        gpu_utils::check_gpu_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_waterfall_d.data()),
                       waterfall.data(), total_in * sizeof(float),
                       cudaMemcpyHostToDevice),
            "FDMTFFT::reset: H2D waterfall");
        reset(cuda::std::span<const float>(
                  thrust::raw_pointer_cast(m_waterfall_d.data()), total_in),
              cuda::std::span<float>(
                  thrust::raw_pointer_cast(m_dmt_host_d.data()), total_out),
              nullptr);
        m_host_dmt_ptr  = dmt.data();
        m_host_dmt_size = total_out;
    }

    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream) {
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (d_waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: expected d_waterfall size {}, got {}",
                total_in, d_waterfall.size()));
        }
        if (d_dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::reset: d_dmt buffer size {} must be >= {}",
                d_dmt.size(), total_out));
        }
        m_d_waterfall_ptr = d_waterfall.data();
        m_dmt_target_ptr  = d_dmt.data();
        m_host_dmt_ptr    = nullptr;
        m_host_dmt_size   = 0;
        m_stream          = stream;
        m_current_level   = 0;
        m_state_in        = reinterpret_cast<ComplexTypeGPU*>(
            thrust::raw_pointer_cast(m_state_a_d.data()));
        m_state_out = reinterpret_cast<ComplexTypeGPU*>(
            thrust::raw_pointer_cast(m_state_b_d.data()));
        m_view_valid = false;
        fill_and_init(d_waterfall.data(), stream);
        m_is_initialized = true;
    }

    void advance(SizeType levels, cudaStream_t stream) {
        require_stepper();
        if (stream == nullptr) {
            stream = m_stream;
        }
        const auto total_lvl = total_levels();
        while (levels > 0 && m_current_level < total_lvl - 1) {
            const SizeType next_level = m_current_level + 1;
            merge_iter_device(m_state_in, m_state_out, next_level, stream);
            std::swap(m_state_in, m_state_out);
            m_current_level = next_level;
            m_view_valid    = false;
            --levels;
        }
    }

    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream) {
        require_stepper();
        const auto total_lvl = total_levels();
        if (remaining_levels >= total_lvl) {
            return;
        }
        const SizeType target_level = total_lvl - 1 - remaining_levels;
        if (target_level > m_current_level) {
            advance(target_level - m_current_level, stream);
        }
    }

    [[nodiscard]] DeviceSpan<const float>
    view_level_data_device() const override {
        require_stepper();
        materialize_view();
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        const auto n = shape.ncoords * view_nsamps();
        return {thrust::raw_pointer_cast(m_view_d.data()), n, device()};
    }

    [[nodiscard]] std::span<const float> view_level_data() const override {
        const float* view = host_view();
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        return {view, shape.ncoords * view_nsamps()};
    }

    [[nodiscard]] FDMTSubbandView
    view_subband(SizeType subband_idx) const override {
        const auto dev    = view_subband_device(subband_idx);
        const float* view = host_view();
        const auto offset = static_cast<SizeType>(
            dev.data.data() - thrust::raw_pointer_cast(m_view_d.data()));
        return FDMTSubbandView{
            .data = std::span<const float>(view + offset, dev.data.size()),
            .subband_idx = dev.subband_idx,
            .ndt         = dev.ndt,
            .nsamps      = dev.nsamps,
            .f_start     = dev.f_start,
            .f_end       = dev.f_end,
            .dt_grid     = dev.dt_grid,
        };
    }

    [[nodiscard]] FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const override {
        require_stepper();
        materialize_view();
        const auto& plan_c = m_plan->get_container();
        const auto& shape  = plan_c.state_shape[m_current_level];
        if (subband_idx >= shape.nchans) {
            throw std::out_of_range("FDMTFFT (cuda): subband_idx out of range");
        }
        const auto& grid    = plan_c.grids[m_current_level][subband_idx];
        const auto nsamps_v = view_nsamps();
        const auto offset   = grid.coord_offset * nsamps_v;
        const auto count    = grid.ndt * nsamps_v;
        return FDMTSubbandDeviceView{
            .data = DeviceSpan<const float>(
                thrust::raw_pointer_cast(m_view_d.data()) + offset, count,
                device()),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps_v,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    SizeType current_level() const noexcept override { return m_current_level; }
    SizeType total_levels() const noexcept { return m_plan->get_niters() + 1; }
    SizeType num_subbands() const override {
        require_stepper();
        return m_plan->get_container().state_shape[m_current_level].nchans;
    }
    bool is_finished() const noexcept override {
        return m_is_initialized && (m_current_level >= total_levels() - 1);
    }

    void finalize(cudaStream_t stream) {
        require_stepper();
        if (stream == nullptr) {
            stream = m_stream;
        }
        if (!is_finished()) {
            advance_until_remaining(0, stream);
        }
        inverse_and_store(m_state_in, m_dmt_target_ptr, stream);
        if (m_mode == FDMTMode::kValid && m_d_waterfall_ptr != nullptr) {
            launch_update_overlap(m_d_waterfall_ptr, stream);
        }
        if (m_host_dmt_ptr != nullptr) {
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream),
                                      "FDMTFFT::finalize: sync");
            gpu_utils::check_gpu_call(
                cudaMemcpy(m_host_dmt_ptr,
                           thrust::raw_pointer_cast(m_dmt_host_d.data()),
                           m_host_dmt_size * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "FDMTFFT::finalize: D2H dmt");
            m_host_dmt_ptr = nullptr;
        }
        m_is_initialized = false;
        m_view_valid     = false;
    }

    void reset_history() noexcept override {
        if (!m_overlap_d.empty()) {
            thrust::fill(m_overlap_d.begin(), m_overlap_d.end(), 0.0F);
        }
        if (!m_overlap_scratch_d.empty()) {
            thrust::fill(m_overlap_scratch_d.begin(), m_overlap_scratch_d.end(),
                         0.0F);
        }
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_nbeams;
    int m_device_id;
    bool m_use_box_smearing;
    FDMTMode m_mode;
    bool m_frac; // real-valued merge shifts (fdmt_fft_common.hpp)
    const plans::FDMTPlan* m_plan; // owned by the FDMTFFT facade
    plans::FDMTPlanContainerD m_plan_d;
    // Fractional merge shifts, flattened like the device coordinates_sum.
    thrust::device_vector<int> m_shift_int_d;
    thrust::device_vector<float> m_shift_frac_d;
    std::vector<int> m_coords_sum_offsets;
    std::vector<int> m_coords_copy_offsets;

    SizeType m_ndms{};
    SizeType m_n_bins{};
    SizeType m_n_fft{};
    SizeType m_fft_buf_size{};
    SizeType m_max_s{};
    SizeType m_max_coords{};
    SizeType m_overlap_len{};
    SizeType m_nsamps_out{};
    SizeType m_out_skip{};

    thrust::device_vector<ComplexTypeGPU> m_phasors_d;
    thrust::device_vector<ComplexTypeGPU> m_boxcar_window_d;
    thrust::device_vector<ComplexTypeGPU> m_spectra_d;
    thrust::device_vector<ComplexTypeGPU> m_state_a_d;
    thrust::device_vector<ComplexTypeGPU> m_state_b_d;
    mutable thrust::device_vector<ComplexTypeGPU> m_ifft_d;
    thrust::device_vector<float> m_window_d;
    mutable thrust::device_vector<float> m_time_out_d;
    mutable thrust::device_vector<float> m_view_d;
    thrust::device_vector<float> m_overlap_d;
    thrust::device_vector<float> m_overlap_scratch_d;
    thrust::device_vector<float> m_waterfall_d;
    thrust::device_vector<float> m_dmt_host_d;

    std::unique_ptr<utils::CUFFTManager> m_fft_forward;
    std::unique_ptr<utils::CUFFTManager> m_fft_backward;

    const float* m_d_waterfall_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
    float* m_host_dmt_ptr{nullptr};
    SizeType m_host_dmt_size{0};
    ComplexTypeGPU* m_state_in{nullptr};
    ComplexTypeGPU* m_state_out{nullptr};
    cudaStream_t m_stream{nullptr};
    // End of the latest device-memory call; host-memory calls wait for it.
    gpu_utils::DeviceWorkFence m_device_work;

    // The stream a stepper call runs on (null: the one reset() was given).
    [[nodiscard]] cudaStream_t active(Stream stream) const noexcept {
        const auto s = to_cuda(stream);
        return s != nullptr ? s : m_stream;
    }
    SizeType m_current_level{0};
    bool m_is_initialized{false};
    mutable bool m_view_valid{false};
    // Host copy of m_view_d for the host view_* methods; stale whenever
    // materialize_view() recomputes the device view.
    mutable std::vector<float> m_view_h;
    mutable bool m_view_h_valid{false};

    [[nodiscard]] Device device() const noexcept {
        return {.backend = detail::kGPUBackend, .id = m_device_id};
    }

    [[nodiscard]] const float* host_view() const {
        require_stepper();
        materialize_view();
        if (!m_view_h_valid) {
            const auto& shape =
                m_plan->get_container().state_shape[m_current_level];
            const auto n = shape.ncoords * view_nsamps();
            m_view_h.resize(n);
            gpu_utils::check_gpu_call(
                cudaMemcpy(m_view_h.data(),
                           thrust::raw_pointer_cast(m_view_d.data()),
                           n * sizeof(float), cudaMemcpyDeviceToHost),
                "FDMTFFT (cuda): host view");
            m_view_h_valid = true;
        }
        return m_view_h.data();
    }

    void initialize() {
        gpu_utils::set_device(m_device_id);
        // Same transform geometry as the CPU engine (fdmt_fft_common.hpp):
        // the plan's for integer delays, longer with fractional ones.
        const auto geom = fdmt_fft::make_geometry(*m_plan, m_mode, m_frac,
                                                  m_use_box_smearing);
        m_ndms          = m_plan->get_dmt_ndms();
        m_n_fft         = geom.n_fft;
        m_n_bins        = (m_n_fft / 2) + 1;
        m_max_s         = m_plan->get_max_shift();
        m_overlap_len   = geom.overlap;
        m_nsamps_out    = m_plan->get_dmt_nsamps();
        m_out_skip      = geom.skip;
        if (m_mode == FDMTMode::kValid &&
            m_nsamps_out != m_plan->get_nsamps()) {
            throw std::logic_error("FDMTFFT: valid mode expects nsamps output");
        }
        if (m_out_skip + m_nsamps_out > m_n_fft) {
            throw std::logic_error("FDMTFFT: output exceeds the FFT length");
        }
        m_max_coords = 0;
        for (const auto& shape : m_plan->get_container().state_shape) {
            m_max_coords = std::max(m_max_coords, shape.ncoords);
        }
        m_fft_buf_size = m_max_coords * m_n_bins;
        if (m_frac) {
            const auto frac =
                fdmt_fft::fractional_delays(*m_plan, m_use_box_smearing);
            std::vector<int> si;
            std::vector<float> sf;
            for (SizeType l = 1; l < frac.shift.size(); ++l) {
                for (const double v : frac.shift[l]) {
                    const double fl = std::floor(v);
                    si.push_back(static_cast<int>(fl));
                    sf.push_back(static_cast<float>(v - fl));
                }
            }
            m_shift_int_d  = si;
            m_shift_frac_d = sf;
        }

        plans::transfer_fdmt_plan_to_device(m_plan->get_container(), m_plan_d);
        const auto niters_plus_one = m_plan->get_niters() + 1;
        m_coords_sum_offsets.resize(niters_plus_one, 0);
        m_coords_copy_offsets.resize(niters_plus_one, 0);
        int sum_off  = 0;
        int copy_off = 0;
        for (size_t i = 0; i < niters_plus_one; ++i) {
            m_coords_sum_offsets[i]  = sum_off;
            m_coords_copy_offsets[i] = copy_off;
            sum_off += static_cast<int>(
                m_plan->get_container().state_shape[i].ncoords_sum);
            copy_off += static_cast<int>(
                m_plan->get_container().state_shape[i].ncoords_copy);
        }

        // Phasors W^(s k) at this engine's transform length (exact in
        // double): merges read them up to max_shift with integer delays;
        // with fractional delays only the level-0 windows (s <= level-0
        // dt_max) do.
        const SizeType table_s =
            m_frac ? m_plan->get_container().state_shape[0].dt_max : m_max_s;
        std::vector<ComplexTypeGPU> phasors_cuda((table_s + 1) * m_n_bins);
        for (SizeType sft = 0; sft <= table_s; ++sft) {
            for (SizeType k = 0; k < m_n_bins; ++k) {
                const double a = -2.0 * 3.14159265358979323846 *
                                 static_cast<double>((sft * k) % m_n_fft) /
                                 static_cast<double>(m_n_fft);
                phasors_cuda[(sft * m_n_bins) + k] =
                    ComplexTypeGPU(static_cast<float>(std::cos(a)),
                                   static_cast<float>(std::sin(a)));
            }
        }
        m_phasors_d = phasors_cuda;
        std::vector<ComplexTypeGPU> boxcar_h((table_s + 1) * m_n_bins);
        for (SizeType k = 0; k < m_n_bins; ++k) {
            ComplexTypeGPU accum{0.0F, 0.0F};
            for (SizeType s = 0; s <= table_s; ++s) {
                accum += phasors_cuda[(s * m_n_bins) + k];
                boxcar_h[(s * m_n_bins) + k] = accum;
            }
        }
        m_boxcar_window_d = boxcar_h;

        m_spectra_d.resize(m_nbeams * m_nchans * m_n_bins);
        m_state_a_d.resize(m_nbeams * m_fft_buf_size);
        m_state_b_d.resize(m_nbeams * m_fft_buf_size);
        m_ifft_d.resize(m_nbeams * m_fft_buf_size);
        m_window_d.resize(m_nbeams * m_nchans * m_n_fft, 0.0F);
        m_time_out_d.resize(m_nbeams * m_max_coords * m_n_fft, 0.0F);
        m_view_d.resize(m_max_coords * m_nsamps_out, 0.0F);
        if (m_mode == FDMTMode::kValid) {
            const auto ov_n = m_nbeams * m_nchans * m_overlap_len;
            m_overlap_d.resize(ov_n, 0.0F);
            m_overlap_scratch_d.resize(ov_n, 0.0F);
        }

        m_fft_forward = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kR2C, m_n_fft, m_nbeams * m_nchans, m_device_id);
        m_fft_backward = std::make_unique<utils::CUFFTManager>(
            utils::FFTKind::kC2R, m_n_fft, m_nbeams * m_max_coords,
            m_device_id);
    }

    void run_transform(cuda::std::span<const float> d_waterfall,
                       cuda::std::span<float> d_dmt,
                       cudaStream_t stream,
                       bool update_hist) {
        gpu_utils::set_device(m_device_id);
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (d_waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: expected d_waterfall size {}, got {}",
                total_in, d_waterfall.size()));
        }
        if (d_dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::execute: d_dmt buffer size {} must be >= {}",
                d_dmt.size(), total_out));
        }
        fill_and_init(d_waterfall.data(), stream);
        ComplexTypeGPU* state_in = reinterpret_cast<ComplexTypeGPU*>(
            thrust::raw_pointer_cast(m_state_a_d.data()));
        ComplexTypeGPU* state_out = reinterpret_cast<ComplexTypeGPU*>(
            thrust::raw_pointer_cast(m_state_b_d.data()));
        const auto niters = m_plan->get_niters();
        for (SizeType i_iter = 1; i_iter <= niters; ++i_iter) {
            merge_iter_device(state_in, state_out, i_iter, stream);
            std::swap(state_in, state_out);
        }
        inverse_and_store(state_in, d_dmt.data(), stream);
        if (update_hist && m_mode == FDMTMode::kValid) {
            launch_update_overlap(d_waterfall.data(), stream);
        }
        gpu_utils::check_last_gpu_error("FDMTFFT::execute kernels");
    }

    void fill_and_init(const float* d_wf, cudaStream_t stream) {
        const dim3 block(256);
        const dim3 grid_fill((m_n_fft + 255) / 256,
                             static_cast<unsigned>(m_nchans),
                             static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid_fill, block);
        const int mode_i = static_cast<int>(m_mode);
        const float* ov_ptr =
            m_overlap_d.empty() ? nullptr
                                : thrust::raw_pointer_cast(m_overlap_d.data());
        kernel_fill_window<<<grid_fill, block, 0, stream>>>(
            d_wf, ov_ptr, thrust::raw_pointer_cast(m_window_d.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_nsamps),
            static_cast<int>(m_n_fft), static_cast<int>(m_overlap_len), mode_i,
            static_cast<int>(m_nbeams));
        m_fft_forward->execute(
            cuda::std::span<float>(thrust::raw_pointer_cast(m_window_d.data()),
                                   m_window_d.size()),
            cuda::std::span<ComplexTypeGPU>(
                thrust::raw_pointer_cast(m_spectra_d.data()),
                m_spectra_d.size()),
            stream);

        const auto block_x = 256u;
        const dim3 init_grid((m_n_bins + block_x - 1) / block_x,
                             static_cast<unsigned>(m_nchans),
                             static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(init_grid, dim3(block_x));
        const auto* win_ptr =
            m_use_box_smearing
                ? reinterpret_cast<const ComplexTypeGPU*>(
                      thrust::raw_pointer_cast(m_boxcar_window_d.data()))
                : reinterpret_cast<const ComplexTypeGPU*>(
                      thrust::raw_pointer_cast(m_phasors_d.data()));
        kernel_init_fdmt_fft<<<init_grid, block_x, 0, stream>>>(
            reinterpret_cast<const ComplexTypeGPU*>(
                thrust::raw_pointer_cast(m_spectra_d.data())),
            win_ptr,
            reinterpret_cast<ComplexTypeGPU*>(
                thrust::raw_pointer_cast(m_state_a_d.data())),
            thrust::raw_pointer_cast(m_plan_d.grids0.coord_offset.data()),
            thrust::raw_pointer_cast(m_plan_d.grids0.ndt.data()),
            thrust::raw_pointer_cast(m_plan_d.grids0.dt_grid.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_n_bins),
            static_cast<int>(m_max_coords));
    }

    void merge_iter_device(ComplexTypeGPU* state_in,
                           ComplexTypeGPU* state_out,
                           SizeType i_iter,
                           cudaStream_t stream) {
        const auto& shape = m_plan->get_container().state_shape[i_iter];
        const int n_sum   = static_cast<int>(shape.ncoords_sum);
        const int n_copy  = static_cast<int>(shape.ncoords_copy);
        const int max_c   = std::max(n_sum, n_copy);
        if (max_c == 0) {
            return;
        }
        auto coords_sum  = m_plan_d.coordinates_sum.get_raw_ptrs();
        auto coords_copy = m_plan_d.coordinates_copy.get_raw_ptrs();
        coords_sum.update_offsets(m_coords_sum_offsets[i_iter]);
        coords_copy.update_offsets(m_coords_copy_offsets[i_iter]);

        const int total = max_c * static_cast<int>(m_n_bins);
        const dim3 block(256);
        const dim3 grid((total + 255) / 256, static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        if (m_frac) {
            const auto off = m_coords_sum_offsets[i_iter];
            kernel_execute_iter_fft_frac<<<grid, block, 0, stream>>>(
                state_in, state_out, coords_sum, coords_copy,
                thrust::raw_pointer_cast(m_shift_int_d.data()) + off,
                thrust::raw_pointer_cast(m_shift_frac_d.data()) + off,
                static_cast<int>(m_n_fft),
                ~0ULL / static_cast<unsigned long long>(m_n_fft),
                static_cast<int>(m_n_bins), n_sum, n_copy,
                static_cast<int>(m_max_coords));
            return;
        }
        kernel_execute_iter_fft<<<grid, block, 0, stream>>>(
            state_in, state_out, coords_sum, coords_copy,
            reinterpret_cast<const ComplexTypeGPU*>(
                thrust::raw_pointer_cast(m_phasors_d.data())),
            static_cast<int>(m_n_bins), n_sum, n_copy,
            static_cast<int>(m_max_coords));
    }

    // Full-mode tail t >= nsamps is the Fourier linear-convolution
    // continuation and is not required to match the CPU FDMT (see
    // docs/fdmt-fft.md).
    void inverse_and_store(const ComplexTypeGPU* state_root,
                           float* d_dmt,
                           cudaStream_t stream) {
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ifft_d.data()),
                            state_root,
                            m_nbeams * m_fft_buf_size * sizeof(ComplexTypeGPU),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMTFFT (cuda): snapshot for C2R");
        m_fft_backward->execute(
            cuda::std::span<float>(
                thrust::raw_pointer_cast(m_time_out_d.data()),
                m_time_out_d.size()),
            cuda::std::span<ComplexTypeGPU>(
                thrust::raw_pointer_cast(m_ifft_d.data()), m_ifft_d.size()),
            stream);
        const float norm = 1.0F / static_cast<float>(m_n_fft);
        const dim3 block(256);
        const dim3 grid((m_nsamps_out + 255) / 256,
                        static_cast<unsigned>(m_ndms),
                        static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        kernel_trim_scale<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(m_time_out_d.data()), d_dmt,
            static_cast<int>(m_ndms), static_cast<int>(m_n_fft),
            static_cast<int>(m_nsamps_out), static_cast<int>(m_out_skip),
            static_cast<int>(m_max_coords), static_cast<int>(m_nbeams), norm);
    }

    void launch_update_overlap(const float* d_wf, cudaStream_t stream) {
        if (m_overlap_len == 0) {
            return;
        }
        const dim3 block(256);
        const dim3 grid((m_overlap_len + 255) / 256,
                        static_cast<unsigned>(m_nchans),
                        static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        kernel_update_overlap<<<grid, block, 0, stream>>>(
            d_wf, thrust::raw_pointer_cast(m_overlap_d.data()),
            thrust::raw_pointer_cast(m_overlap_scratch_d.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_nsamps),
            static_cast<int>(m_overlap_len), static_cast<int>(m_nbeams));
        m_overlap_d.swap(m_overlap_scratch_d);
    }

    [[nodiscard]] SizeType view_nsamps() const {
        if (m_mode == FDMTMode::kFull) {
            return m_plan->get_container().state_shape[m_current_level].nsamps;
        }
        return m_nsamps;
    }
    [[nodiscard]] SizeType view_skip() const {
        return (m_mode == FDMTMode::kValid) ? m_overlap_len : 0;
    }

    void materialize_view() const {
        if (m_view_valid) {
            return;
        }
        m_view_h_valid = false;
        // IFFT beam 0 of the live state into time_out, then trim.
        gpu_utils::check_gpu_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_ifft_d.data()), m_state_in,
                       m_fft_buf_size * sizeof(ComplexTypeGPU),
                       cudaMemcpyDeviceToDevice),
            "FDMTFFT (cuda): view snapshot");
        m_fft_backward->execute(
            cuda::std::span<float>(
                thrust::raw_pointer_cast(m_time_out_d.data()),
                m_time_out_d.size()),
            cuda::std::span<ComplexTypeGPU>(
                thrust::raw_pointer_cast(m_ifft_d.data()), m_ifft_d.size()));
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        const auto nsamps_v = view_nsamps();
        const auto skip     = view_skip();
        const float norm    = 1.0F / static_cast<float>(m_n_fft);
        const dim3 block(256);
        const dim3 grid((nsamps_v + 255) / 256,
                        static_cast<unsigned>(shape.ncoords));
        kernel_materialize_view<<<grid, block>>>(
            thrust::raw_pointer_cast(m_time_out_d.data()),
            thrust::raw_pointer_cast(m_view_d.data()),
            static_cast<int>(shape.ncoords), static_cast<int>(m_n_fft),
            static_cast<int>(nsamps_v), static_cast<int>(skip),
            static_cast<int>(m_max_coords), norm);
        gpu_utils::check_gpu_call(cudaDeviceSynchronize(),
                                  "FDMTFFT (cuda): view sync");
        m_view_valid = true;
    }

    void require_stepper() const {
        if (!m_is_initialized) {
            throw std::logic_error("FDMTFFT (cuda): Stepper is not "
                                   "initialized. Call reset() first.");
        }
    }
};

} // namespace

std::unique_ptr<detail::FDMTFFTEngine>
detail::make_fdmt_fft_gpu(const plans::FDMTPlan& plan,
                          const detail::FDMTFFTEngineConfig& cfg) {
    return std::make_unique<FDMTFFTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
