#include "dmt/algorithms/fdmt_fft.hpp"

#ifdef DMT_ENABLE_CUDA

#include <algorithm>
#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda/std/complex>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include <spdlog/spdlog.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms {

namespace {

using Cplx = cuda::std::complex<float>;

enum class FDMTMode : uint8_t { kFull = 0, kValid = 1, kRoll = 2 };

FDMTMode parse_mode(std::string_view mode) {
    if (mode == "full") {
        return FDMTMode::kFull;
    }
    if (mode == "roll") {
        return FDMTMode::kRoll;
    }
    if (mode == "valid") {
        return FDMTMode::kValid;
    }
    throw std::invalid_argument(std::format(
        "Invalid mode '{}'. Expected 'full', 'roll', or 'valid'", mode));
}

__global__ void kernel_fill_window(const float* __restrict__ wf,
                                   const float* __restrict__ overlap,
                                   float* __restrict__ window,
                                   int nchans,
                                   int nsamps,
                                   int n_fft,
                                   int overlap_len,
                                   int mode,
                                   int nbeams) {
    const auto t      = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_chan = static_cast<int>(blockIdx.y);
    const auto i_beam = static_cast<int>(blockIdx.z);
    if (t >= n_fft || i_chan >= nchans || i_beam >= nbeams) {
        return;
    }
    const auto out_idx =
        (((i_beam * nchans) + i_chan) * n_fft) + t;
    float val = 0.0F;
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
            val = wf[(((i_beam * nchans) + i_chan) * nsamps) + (t - overlap_len)];
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
    const auto i_beam = static_cast<int>(blockIdx.z);
    if (i >= overlap_len || i_chan >= nchans || i_beam >= nbeams) {
        return;
    }
    const auto base = ((i_beam * nchans) + i_chan) * overlap_len;
    const float* src = wf + ((((i_beam * nchans) + i_chan) * nsamps));
    float val = 0.0F;
    if (nsamps >= overlap_len) {
        val = src[nsamps - overlap_len + i];
    } else if (i < overlap_len - nsamps) {
        val = overlap_in[base + i + nsamps];
    } else {
        val = src[i - (overlap_len - nsamps)];
    }
    overlap_out[base + i] = val;
}

__global__ void kernel_init_fdmt_fft(const Cplx* __restrict__ spectra,
                                     const Cplx* __restrict__ win_table,
                                     Cplx* __restrict__ state,
                                     const int* __restrict__ grids0_coord_offset_ptr,
                                     const int* __restrict__ grids0_ndt_ptr,
                                     const int* __restrict__ grids0_dt_grid_ptr,
                                     int nsubs,
                                     int n_bins,
                                     int max_coords) {
    const auto k      = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_sub  = static_cast<int>(blockIdx.y);
    const auto i_beam = static_cast<int>(blockIdx.z);
    if (k >= n_bins || i_sub >= nsubs) {
        return;
    }
    const auto coord_base = grids0_coord_offset_ptr[i_sub];
    const auto ndt        = grids0_ndt_ptr[i_sub];
    const auto* dt_grid   = &grids0_dt_grid_ptr[coord_base];
    const auto spec_offset =
        (i_beam * nsubs * n_bins) + (i_sub * n_bins) + k;
    const Cplx sample          = spectra[spec_offset];
    const auto beam_state_base = i_beam * max_coords * n_bins;
    for (int i_dt = 0; i_dt < ndt; ++i_dt) {
        const int s        = abs(dt_grid[i_dt]);
        const Cplx win     = win_table[(s * n_bins) + k];
        const auto coord_idx = coord_base + i_dt;
        state[beam_state_base + (coord_idx * n_bins) + k] = sample * win;
    }
}

__global__ void kernel_execute_iter_fft(const Cplx* __restrict__ state_in,
                                        Cplx* __restrict__ state_out,
                                        const plans::FDMTCoordDPtrs coords_sum,
                                        const plans::FDMTCoordDPtrs coords_copy,
                                        const Cplx* __restrict__ phasors,
                                        int n_bins,
                                        int ncoords_sum_cur,
                                        int ncoords_copy_cur,
                                        int max_coords) {
    const auto linear = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_beam = static_cast<int>(blockIdx.y);
    const int max_iter_coords =
        (ncoords_sum_cur > ncoords_copy_cur) ? ncoords_sum_cur
                                             : ncoords_copy_cur;
    const int total = max_iter_coords * n_bins;
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
        const Cplx tail = state_in[tail_base + k];
        const Cplx head = state_in[head_base + k];
        const Cplx p    = phasors[(s * n_bins) + k];
        state_out[out_base + k] = tail + (head * p);
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

__global__ void kernel_trim_scale(const float* __restrict__ time_out,
                                  float* __restrict__ dmt,
                                  int ndms,
                                  int n_fft,
                                  int nsamps_out,
                                  int skip,
                                  int max_coords,
                                  int nbeams,
                                  float norm) {
    const auto t      = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_dm   = static_cast<int>(blockIdx.y);
    const auto i_beam = static_cast<int>(blockIdx.z);
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
    const auto t    = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto i_c  = static_cast<int>(blockIdx.y);
    if (t >= nsamps_view || i_c >= ncoords) {
        return;
    }
    const auto src = time_out[(i_c * n_fft) + skip + t];
    view[(i_c * nsamps_view) + t] = src * norm;
}

} // namespace

class FDMTFFTCUDA::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         IndexType dt_max,
         IndexType dt_min,
         SizeType dt_step,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int device_id,
         SizeType nbeams)
        : m_nchans(nchans),
          m_nsamps(nsamps),
          m_nbeams(nbeams),
          m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(std::make_unique<plans::FDMTPlan>(f_min,
                                                   f_max,
                                                   nchans,
                                                   nsamps,
                                                   tsamp,
                                                   dt_max,
                                                   dt_min,
                                                   dt_step,
                                                   mode,
                                                   verbose)) {
        initialize();
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int device_id,
         SizeType nbeams)
        : m_nchans(nchans),
          m_nsamps(nsamps),
          m_nbeams(nbeams),
          m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(std::make_unique<plans::FDMTPlan>(f_min,
                                                   f_max,
                                                   nchans,
                                                   nsamps,
                                                   tsamp,
                                                   dt_grid,
                                                   mode,
                                                   verbose)) {
        initialize();
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         bool use_box_smearing,
         std::string_view mode,
         bool verbose,
         int device_id,
         SizeType nbeams)
        : m_nchans(nchans),
          m_nsamps(nsamps),
          m_nbeams(nbeams),
          m_device_id(device_id),
          m_use_box_smearing(use_box_smearing),
          m_mode(parse_mode(mode)),
          m_plan(std::make_unique<plans::FDMTPlan>(f_min,
                                                   f_max,
                                                   nchans,
                                                   nsamps,
                                                   tsamp,
                                                   dm_grid,
                                                   mode,
                                                   verbose)) {
        initialize();
    }

    ~Impl() {
        cuda_utils::set_device(m_device_id);
        if (m_plan_forward != 0) {
            cufftDestroy(m_plan_forward);
            m_plan_forward = 0;
        }
        if (m_plan_backward != 0) {
            cufftDestroy(m_plan_backward);
            m_plan_backward = 0;
        }
    }

    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept {
        return *m_plan;
    }
    [[nodiscard]] SizeType get_nbeams() const noexcept { return m_nbeams; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        cuda_utils::set_device(m_device_id);
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::execute: expected waterfall size {}, got {}",
                total_in, waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::execute: dmt buffer size {} must be >= {}",
                dmt.size(), total_out));
        }
        m_waterfall_d.resize(total_in);
        m_dmt_host_d.resize(total_out);
        cuda_utils::check_cuda_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_waterfall_d.data()),
                       waterfall.data(), total_in * sizeof(float),
                       cudaMemcpyHostToDevice),
            "FDMTFFTCUDA::execute: H2D waterfall");
        execute(cuda::std::span<const float>(
                    thrust::raw_pointer_cast(m_waterfall_d.data()), total_in),
                cuda::std::span<float>(
                    thrust::raw_pointer_cast(m_dmt_host_d.data()), total_out),
                nullptr);
        cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                    "FDMTFFTCUDA::execute: sync");
        cuda_utils::check_cuda_call(
            cudaMemcpy(dmt.data(),
                       thrust::raw_pointer_cast(m_dmt_host_d.data()),
                       total_out * sizeof(float), cudaMemcpyDeviceToHost),
            "FDMTFFTCUDA::execute: D2H dmt");
    }

    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream) {
        run_transform(d_waterfall, d_dmt, stream, true);
    }

    void reset(std::span<const float> waterfall, std::span<float> dmt) {
        cuda_utils::set_device(m_device_id);
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::reset: expected waterfall size {}, got {}",
                total_in, waterfall.size()));
        }
        if (dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::reset: dmt buffer size {} must be >= {}",
                dmt.size(), total_out));
        }
        m_waterfall_d.resize(total_in);
        m_dmt_host_d.resize(total_out);
        cuda_utils::check_cuda_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_waterfall_d.data()),
                       waterfall.data(), total_in * sizeof(float),
                       cudaMemcpyHostToDevice),
            "FDMTFFTCUDA::reset: H2D waterfall");
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
                "FDMTFFTCUDA::reset: expected d_waterfall size {}, got {}",
                total_in, d_waterfall.size()));
        }
        if (d_dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::reset: d_dmt buffer size {} must be >= {}",
                d_dmt.size(), total_out));
        }
        m_d_waterfall_ptr = d_waterfall.data();
        m_dmt_target_ptr  = d_dmt.data();
        m_host_dmt_ptr    = nullptr;
        m_host_dmt_size   = 0;
        m_stream          = stream;
        m_current_level   = 0;
        m_state_in        = reinterpret_cast<Cplx*>(
            thrust::raw_pointer_cast(m_state_a_d.data()));
        m_state_out = reinterpret_cast<Cplx*>(
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

    cuda::std::span<const float> view_level_data() const {
        require_stepper();
        materialize_view();
        const auto& shape =
            m_plan->get_container().state_shape[m_current_level];
        const auto n = shape.ncoords * view_nsamps();
        return {thrust::raw_pointer_cast(m_view_d.data()), n};
    }

    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const {
        return view_subband(subband_idx).data;
    }

    FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const {
        require_stepper();
        materialize_view();
        const auto& plan_c = m_plan->get_container();
        const auto& shape  = plan_c.state_shape[m_current_level];
        if (subband_idx >= shape.nchans) {
            throw std::out_of_range("FDMTFFTCUDA: subband_idx out of range");
        }
        const auto& grid    = plan_c.grids[m_current_level][subband_idx];
        const auto nsamps_v = view_nsamps();
        const auto offset   = grid.coord_offset * nsamps_v;
        const auto count    = grid.ndt * nsamps_v;
        return FDMTSubbandViewCUDA{
            .data = cuda::std::span<const float>(
                thrust::raw_pointer_cast(m_view_d.data()) + offset, count),
            .subband_idx = subband_idx,
            .ndt         = grid.ndt,
            .nsamps      = nsamps_v,
            .f_start     = grid.f_start,
            .f_end       = grid.f_end,
            .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                      grid.dt_grid.size()),
        };
    }

    SizeType current_level() const noexcept { return m_current_level; }
    SizeType total_levels() const noexcept { return m_plan->get_niters() + 1; }
    SizeType remaining_levels() const noexcept {
        if (total_levels() <= 1 || m_current_level >= total_levels() - 1) {
            return 0;
        }
        return (total_levels() - 1) - m_current_level;
    }
    SizeType num_subbands() const {
        require_stepper();
        return m_plan->get_container().state_shape[m_current_level].nchans;
    }
    bool is_finished() const noexcept {
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
            cuda_utils::check_cuda_call(cudaStreamSynchronize(stream),
                                        "FDMTFFTCUDA::finalize: sync");
            cuda_utils::check_cuda_call(
                cudaMemcpy(m_host_dmt_ptr,
                           thrust::raw_pointer_cast(m_dmt_host_d.data()),
                           m_host_dmt_size * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "FDMTFFTCUDA::finalize: D2H dmt");
            m_host_dmt_ptr = nullptr;
        }
        m_is_initialized = false;
        m_view_valid     = false;
    }

    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width) const {
        return m_plan->get_effective_variance(dm_idx, boxcar_width,
                                              m_use_box_smearing);
    }
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width) const {
        return m_plan->get_effective_sigma(dm_idx, boxcar_width,
                                           m_use_box_smearing);
    }
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width) const {
        return m_plan->get_effective_variance_grid(boxcar_width,
                                                   m_use_box_smearing);
    }
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width) const {
        return m_plan->get_effective_sigma_grid(boxcar_width,
                                                m_use_box_smearing);
    }

    void reset_history() noexcept {
        if (!m_overlap_d.empty()) {
            thrust::fill(m_overlap_d.begin(), m_overlap_d.end(), 0.0F);
        }
        if (!m_overlap_scratch_d.empty()) {
            thrust::fill(m_overlap_scratch_d.begin(), m_overlap_scratch_d.end(),
                         0.0F);
        }
    }

private:
    SizeType m_nchans;
    SizeType m_nsamps;
    SizeType m_nbeams;
    int m_device_id;
    bool m_use_box_smearing;
    FDMTMode m_mode;
    std::unique_ptr<plans::FDMTPlan> m_plan;
    plans::FDMTPlanContainerD m_plan_d;
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

    thrust::device_vector<ComplexTypeCUDA> m_phasors_d;
    thrust::device_vector<ComplexTypeCUDA> m_boxcar_window_d;
    thrust::device_vector<ComplexTypeCUDA> m_spectra_d;
    thrust::device_vector<ComplexTypeCUDA> m_state_a_d;
    thrust::device_vector<ComplexTypeCUDA> m_state_b_d;
    mutable thrust::device_vector<ComplexTypeCUDA> m_ifft_d;
    thrust::device_vector<float> m_window_d;
    mutable thrust::device_vector<float> m_time_out_d;
    mutable thrust::device_vector<float> m_view_d;
    thrust::device_vector<float> m_overlap_d;
    thrust::device_vector<float> m_overlap_scratch_d;
    thrust::device_vector<float> m_waterfall_d;
    thrust::device_vector<float> m_dmt_host_d;

    cufftHandle m_plan_forward{0};
    cufftHandle m_plan_backward{0};

    const float* m_d_waterfall_ptr{nullptr};
    float* m_dmt_target_ptr{nullptr};
    float* m_host_dmt_ptr{nullptr};
    SizeType m_host_dmt_size{0};
    Cplx* m_state_in{nullptr};
    Cplx* m_state_out{nullptr};
    cudaStream_t m_stream{nullptr};
    SizeType m_current_level{0};
    bool m_is_initialized{false};
    mutable bool m_view_valid{false};

    void initialize() {
        cuda_utils::set_device(m_device_id);
        m_ndms         = m_plan->get_dmt_ndms();
        m_n_fft        = m_plan->get_fft_size();
        m_n_bins       = m_plan->get_fft_n_bins();
        m_fft_buf_size = m_plan->get_fft_buffer_size();
        m_max_s        = m_plan->get_max_shift();
        m_overlap_len  = m_plan->get_fft_overlap();
        m_nsamps_out   = m_plan->get_dmt_nsamps();
        m_out_skip     = (m_mode == FDMTMode::kValid) ? m_overlap_len : 0;
        m_max_coords   = 0;
        for (const auto& shape : m_plan->get_container().state_shape) {
            m_max_coords = std::max(m_max_coords, shape.ncoords);
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

        const auto phasors_h = m_plan->get_fft_phasor_table();
        std::vector<ComplexTypeCUDA> phasors_cuda(phasors_h.size());
        for (SizeType i = 0; i < phasors_h.size(); ++i) {
            phasors_cuda[i] =
                ComplexTypeCUDA(phasors_h[i].real(), phasors_h[i].imag());
        }
        m_phasors_d = phasors_cuda;
        std::vector<ComplexTypeCUDA> boxcar_h((m_max_s + 1) * m_n_bins);
        for (SizeType k = 0; k < m_n_bins; ++k) {
            ComplexTypeCUDA accum{0.0F, 0.0F};
            for (SizeType s = 0; s <= m_max_s; ++s) {
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

        int n_time        = static_cast<int>(m_n_fft);
        int howmany_fwd   = static_cast<int>(m_nbeams * m_nchans);
        int inembed_fwd   = n_time;
        int onembed_fwd   = static_cast<int>(m_n_bins);
        cuda_utils::check_cuda_call(
            cufftPlanMany(&m_plan_forward, 1, &n_time, &inembed_fwd, 1,
                          inembed_fwd, &onembed_fwd, 1, onembed_fwd, CUFFT_R2C,
                          howmany_fwd),
            "FDMTFFTCUDA: cufftPlanMany forward");

        int howmany_bwd = static_cast<int>(m_nbeams * m_max_coords);
        int inembed_bwd = static_cast<int>(m_n_bins);
        int onembed_bwd = n_time;
        cuda_utils::check_cuda_call(
            cufftPlanMany(&m_plan_backward, 1, &n_time, &inembed_bwd, 1,
                          inembed_bwd, &onembed_bwd, 1, onembed_bwd, CUFFT_C2R,
                          howmany_bwd),
            "FDMTFFTCUDA: cufftPlanMany backward");
    }

    void run_transform(cuda::std::span<const float> d_waterfall,
                       cuda::std::span<float> d_dmt,
                       cudaStream_t stream,
                       bool update_hist) {
        cuda_utils::set_device(m_device_id);
        const auto total_in  = m_nbeams * m_nchans * m_nsamps;
        const auto total_out = m_nbeams * m_ndms * m_nsamps_out;
        if (d_waterfall.size() != total_in) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::execute: expected d_waterfall size {}, got {}",
                total_in, d_waterfall.size()));
        }
        if (d_dmt.size() < total_out) {
            throw std::invalid_argument(std::format(
                "FDMTFFTCUDA::execute: d_dmt buffer size {} must be >= {}",
                d_dmt.size(), total_out));
        }
        fill_and_init(d_waterfall.data(), stream);
        Cplx* state_in = reinterpret_cast<Cplx*>(
            thrust::raw_pointer_cast(m_state_a_d.data()));
        Cplx* state_out = reinterpret_cast<Cplx*>(
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
        cuda_utils::check_last_cuda_error("FDMTFFTCUDA::execute kernels");
    }

    void fill_and_init(const float* d_wf, cudaStream_t stream) {
        cuda_utils::check_cuda_call(cufftSetStream(m_plan_forward, stream),
                                    "FDMTFFTCUDA: cufftSetStream forward");
        const dim3 block(256);
        const dim3 grid_fill((m_n_fft + 255) / 256,
                             static_cast<unsigned>(m_nchans),
                             static_cast<unsigned>(m_nbeams));
        cuda_utils::check_kernel_launch_params(grid_fill, block);
        const int mode_i = static_cast<int>(m_mode);
        const float* ov_ptr =
            m_overlap_d.empty()
                ? nullptr
                : thrust::raw_pointer_cast(m_overlap_d.data());
        kernel_fill_window<<<grid_fill, block, 0, stream>>>(
            d_wf, ov_ptr, thrust::raw_pointer_cast(m_window_d.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_nsamps),
            static_cast<int>(m_n_fft), static_cast<int>(m_overlap_len), mode_i,
            static_cast<int>(m_nbeams));
        cuda_utils::check_cuda_call(
            cufftExecR2C(m_plan_forward,
                         thrust::raw_pointer_cast(m_window_d.data()),
                         reinterpret_cast<cufftComplex*>(
                             thrust::raw_pointer_cast(m_spectra_d.data()))),
            "FDMTFFTCUDA: cufftExecR2C");

        const auto block_x = 256u;
        const dim3 init_grid((m_n_bins + block_x - 1) / block_x,
                             static_cast<unsigned>(m_nchans),
                             static_cast<unsigned>(m_nbeams));
        cuda_utils::check_kernel_launch_params(init_grid, dim3(block_x));
        const auto* win_ptr = m_use_box_smearing
            ? reinterpret_cast<const Cplx*>(
                  thrust::raw_pointer_cast(m_boxcar_window_d.data()))
            : reinterpret_cast<const Cplx*>(
                  thrust::raw_pointer_cast(m_phasors_d.data()));
        kernel_init_fdmt_fft<<<init_grid, block_x, 0, stream>>>(
            reinterpret_cast<const Cplx*>(
                thrust::raw_pointer_cast(m_spectra_d.data())),
            win_ptr,
            reinterpret_cast<Cplx*>(
                thrust::raw_pointer_cast(m_state_a_d.data())),
            thrust::raw_pointer_cast(m_plan_d.grids0.coord_offset.data()),
            thrust::raw_pointer_cast(m_plan_d.grids0.ndt.data()),
            thrust::raw_pointer_cast(m_plan_d.grids0.dt_grid.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_n_bins),
            static_cast<int>(m_max_coords));
    }

    void merge_iter_device(Cplx* state_in,
                           Cplx* state_out,
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
        cuda_utils::check_kernel_launch_params(grid, block);
        kernel_execute_iter_fft<<<grid, block, 0, stream>>>(
            state_in, state_out, coords_sum, coords_copy,
            reinterpret_cast<const Cplx*>(
                thrust::raw_pointer_cast(m_phasors_d.data())),
            static_cast<int>(m_n_bins), n_sum, n_copy,
            static_cast<int>(m_max_coords));
    }

    // Full-mode tail t >= nsamps is the Fourier linear-convolution
    // continuation and is not required to match FDMTCPU (see docs/fdmt-fft.md).
    void inverse_and_store(const Cplx* state_root,
                           float* d_dmt,
                           cudaStream_t stream) {
        cuda_utils::check_cuda_call(cufftSetStream(m_plan_backward, stream),
                                    "FDMTFFTCUDA: cufftSetStream backward");
        cuda_utils::check_cuda_call(
            cudaMemcpyAsync(thrust::raw_pointer_cast(m_ifft_d.data()),
                            state_root,
                            m_nbeams * m_fft_buf_size * sizeof(Cplx),
                            cudaMemcpyDeviceToDevice, stream),
            "FDMTFFTCUDA: snapshot for C2R");
        cuda_utils::check_cuda_call(
            cufftExecC2R(m_plan_backward,
                         reinterpret_cast<cufftComplex*>(
                             thrust::raw_pointer_cast(m_ifft_d.data())),
                         thrust::raw_pointer_cast(m_time_out_d.data())),
            "FDMTFFTCUDA: cufftExecC2R");
        const float norm = 1.0F / static_cast<float>(m_n_fft);
        const dim3 block(256);
        const dim3 grid((m_nsamps_out + 255) / 256,
                        static_cast<unsigned>(m_ndms),
                        static_cast<unsigned>(m_nbeams));
        cuda_utils::check_kernel_launch_params(grid, block);
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
        cuda_utils::check_kernel_launch_params(grid, block);
        kernel_update_overlap<<<grid, block, 0, stream>>>(
            d_wf, thrust::raw_pointer_cast(m_overlap_d.data()),
            thrust::raw_pointer_cast(m_overlap_scratch_d.data()),
            static_cast<int>(m_nchans), static_cast<int>(m_nsamps),
            static_cast<int>(m_overlap_len), static_cast<int>(m_nbeams));
        m_overlap_d.swap(m_overlap_scratch_d);
    }

    [[nodiscard]] SizeType view_nsamps() const {
        if (m_mode == FDMTMode::kFull) {
            return m_plan->get_container()
                .state_shape[m_current_level]
                .nsamps;
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
        // IFFT beam 0 of the live state into time_out, then trim.
        cuda_utils::check_cuda_call(
            cudaMemcpy(thrust::raw_pointer_cast(m_ifft_d.data()), m_state_in,
                       m_fft_buf_size * sizeof(Cplx), cudaMemcpyDeviceToDevice),
            "FDMTFFTCUDA: view snapshot");
        cuda_utils::check_cuda_call(
            cufftExecC2R(m_plan_backward,
                         reinterpret_cast<cufftComplex*>(
                             thrust::raw_pointer_cast(m_ifft_d.data())),
                         thrust::raw_pointer_cast(m_time_out_d.data())),
            "FDMTFFTCUDA: view C2R");
        const auto& shape   = m_plan->get_container().state_shape[m_current_level];
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
        cuda_utils::check_cuda_call(cudaDeviceSynchronize(),
                                    "FDMTFFTCUDA: view sync");
        m_view_valid = true;
    }

    void require_stepper() const {
        if (!m_is_initialized) {
            throw std::logic_error(
                "FDMTFFTCUDA: Stepper is not initialized. Call reset() first.");
        }
    }
};

FDMTFFTCUDA::FDMTFFTCUDA(float f_min,
                         float f_max,
                         SizeType nchans,
                         SizeType nsamps,
                         float tsamp,
                         IndexType dt_max,
                         IndexType dt_min,
                         SizeType dt_step,
                         bool use_box_smearing,
                         std::string_view mode,
                         bool verbose,
                         int device_id,
                         SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_min,
                                    dt_step,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id,
                                    nbeams)) {}

FDMTFFTCUDA::FDMTFFTCUDA(float f_min,
                         float f_max,
                         SizeType nchans,
                         SizeType nsamps,
                         float tsamp,
                         const std::vector<IndexType>& dt_grid,
                         bool use_box_smearing,
                         std::string_view mode,
                         bool verbose,
                         int device_id,
                         SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id,
                                    nbeams)) {}

FDMTFFTCUDA::FDMTFFTCUDA(float f_min,
                         float f_max,
                         SizeType nchans,
                         SizeType nsamps,
                         float tsamp,
                         const std::vector<float>& dm_grid,
                         bool use_box_smearing,
                         std::string_view mode,
                         bool verbose,
                         int device_id,
                         SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dm_grid,
                                    use_box_smearing,
                                    mode,
                                    verbose,
                                    device_id,
                                    nbeams)) {}

FDMTFFTCUDA::~FDMTFFTCUDA()                               = default;
FDMTFFTCUDA::FDMTFFTCUDA(FDMTFFTCUDA&&) noexcept           = default;
FDMTFFTCUDA& FDMTFFTCUDA::operator=(FDMTFFTCUDA&&) noexcept = default;

const plans::FDMTPlan& FDMTFFTCUDA::get_plan() const noexcept {
    return m_impl->get_plan();
}
SizeType FDMTFFTCUDA::get_nbeams() const noexcept {
    return m_impl->get_nbeams();
}
void FDMTFFTCUDA::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}
void FDMTFFTCUDA::execute(cuda::std::span<const float> d_waterfall,
                         cuda::std::span<float> d_dmt,
                         cudaStream_t stream) {
    m_impl->execute(d_waterfall, d_dmt, stream);
}
void FDMTFFTCUDA::reset(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->reset(waterfall, dmt);
}
void FDMTFFTCUDA::reset(cuda::std::span<const float> d_waterfall,
                       cuda::std::span<float> d_dmt,
                       cudaStream_t stream) {
    m_impl->reset(d_waterfall, d_dmt, stream);
}
void FDMTFFTCUDA::advance(SizeType levels, cudaStream_t stream) {
    m_impl->advance(levels, stream);
}
void FDMTFFTCUDA::advance_until_remaining(SizeType remaining_levels,
                                         cudaStream_t stream) {
    m_impl->advance_until_remaining(remaining_levels, stream);
}
cuda::std::span<const float> FDMTFFTCUDA::view_level_data() const {
    return m_impl->view_level_data();
}
cuda::std::span<const float>
FDMTFFTCUDA::view_subband_data(SizeType subband_idx) const {
    return m_impl->view_subband_data(subband_idx);
}
FDMTSubbandViewCUDA FDMTFFTCUDA::view_subband(SizeType subband_idx) const {
    return m_impl->view_subband(subband_idx);
}
SizeType FDMTFFTCUDA::current_level() const noexcept {
    return m_impl->current_level();
}
SizeType FDMTFFTCUDA::total_levels() const noexcept {
    return m_impl->total_levels();
}
SizeType FDMTFFTCUDA::remaining_levels() const noexcept {
    return m_impl->remaining_levels();
}
SizeType FDMTFFTCUDA::num_subbands() const { return m_impl->num_subbands(); }
bool FDMTFFTCUDA::is_finished() const noexcept { return m_impl->is_finished(); }
void FDMTFFTCUDA::finalize(cudaStream_t stream) { m_impl->finalize(stream); }
float FDMTFFTCUDA::get_effective_variance(SizeType dm_idx,
                                          SizeType boxcar_width) const {
    return m_impl->get_effective_variance(dm_idx, boxcar_width);
}
float FDMTFFTCUDA::get_effective_sigma(SizeType dm_idx,
                                       SizeType boxcar_width) const {
    return m_impl->get_effective_sigma(dm_idx, boxcar_width);
}
std::vector<float>
FDMTFFTCUDA::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_variance_grid(boxcar_width);
}
std::vector<float>
FDMTFFTCUDA::get_effective_sigma_grid(SizeType boxcar_width) const {
    return m_impl->get_effective_sigma_grid(boxcar_width);
}
void FDMTFFTCUDA::reset_history() noexcept { m_impl->reset_history(); }

} // namespace dmt::algorithms

#endif // DMT_ENABLE_CUDA
