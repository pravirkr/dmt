#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <tuple>
#include <vector>

#ifdef DMT_ENABLE_CUDA
#include <cuda/std/span>
#include <cuda_runtime_api.h>
#endif // DMT_ENABLE_CUDA

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms {

/**
 * @brief Fast Dispersion Measure Transform using Fourier shifts (FDMT-FFT)
 * for CPU execution.
 *
 * Implements Algorithm 2 of Zackay & Ofek (2014). Converts time-domain
 * dedispersion shifts into complex phase rotations in the Fourier domain:
 *
 *   out[k] = in_tail[k] + in_head[k] * exp(-2*pi*i * k * delay / N_fft)
 *
 * Reuses FDMTPlan (same coordinate DAG as FDMTCPU). The FFT length and
 * padding depend on mode:
 * - "roll": cyclic, N_fft = nsamps (matches FDMTCPU roll).
 * - "full": zero-pad; N_fft = nsamps + L + max_shift. Linear convolution
 *   of the input-aligned region t in [0, nsamps) matches FDMTCPU full
 *   (~1e-4). The extra delay tail t >= nsamps is produced by per-level
 *   buffer growth in the time-domain algorithm and is not required to
 *   match the Fourier tail (see docs/fdmt-fft.md).
 * - "valid": overlap-save of the same linear operator across blocks.
 *   Overlap length L = max(|dt_min|, |dt_max|).
 *
 * Stepper views IFFT the current level on demand (more expensive than
 * FDMTCPU views). Inspection exposes beam 0; advance()/finalize()
 * process every beam.
 *
 * Default mode is "valid", matching FDMTCPU.
 */
class FDMTFFTCPU {
public:
    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               IndexType dt_max,
               IndexType dt_min      = 0,
               SizeType dt_step      = 1,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               bool verbose          = false,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               const std::vector<IndexType>& dt_grid,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               bool verbose          = false,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    FDMTFFTCPU(float f_min,
               float f_max,
               SizeType nchans,
               SizeType nsamps,
               float tsamp,
               const std::vector<float>& dm_grid,
               bool use_box_smearing = true,
               std::string_view mode = "valid",
               bool verbose          = false,
               int nthreads          = 1,
               SizeType nbeams       = 1);

    ~FDMTFFTCPU();
    FDMTFFTCPU(FDMTFFTCPU&&) noexcept;
    FDMTFFTCPU& operator=(FDMTFFTCPU&&) noexcept;
    FDMTFFTCPU(const FDMTFFTCPU&)            = delete;
    FDMTFFTCPU& operator=(const FDMTFFTCPU&) = delete;

    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    /**
     * @brief Executes the end-to-end FDMT-FFT transform in a single shot.
     *
     * @param waterfall Input waterfall, beam-major (nbeams * nchans * nsamps).
     * @param dmt Output DM-time, beam-major (nbeams * ndms * dmt_nsamps).
     */
    void execute(std::span<const float> waterfall, std::span<float> dmt);

    void reset(std::span<const float> waterfall, std::span<float> dmt);
    void advance(SizeType levels = 1);
    void advance_until_remaining(SizeType remaining_levels);

    [[nodiscard]] std::span<const float> view_level_data() const;
    [[nodiscard]] std::span<const float>
    view_subband_data(SizeType subband_idx) const;
    [[nodiscard]] FDMTSubbandView view_subband(SizeType subband_idx) const;

    [[nodiscard]] SizeType current_level() const noexcept;
    [[nodiscard]] SizeType total_levels() const noexcept;
    [[nodiscard]] SizeType remaining_levels() const noexcept;
    [[nodiscard]] SizeType num_subbands() const;
    [[nodiscard]] bool is_finished() const noexcept;
    void finalize();

    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 IndexType dt_max,
                 IndexType dt_min      = 0,
                 SizeType dt_step      = 1,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 bool verbose          = false,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<IndexType>& dt_grid,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 bool verbose          = false,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

[[nodiscard]] std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing = true,
                 std::string_view mode = "valid",
                 bool verbose          = false,
                 int nthreads          = 1,
                 SizeType nbeams       = 1);

#ifdef DMT_ENABLE_CUDA
class FDMTFFTCUDA {
public:
    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                IndexType dt_max,
                IndexType dt_min      = 0,
                SizeType dt_step      = 1,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                bool verbose          = false,
                int device_id         = 0,
                SizeType nbeams       = 1);

    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                const std::vector<IndexType>& dt_grid,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                bool verbose          = false,
                int device_id         = 0,
                SizeType nbeams       = 1);

    FDMTFFTCUDA(float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp,
                const std::vector<float>& dm_grid,
                bool use_box_smearing = true,
                std::string_view mode = "valid",
                bool verbose          = false,
                int device_id         = 0,
                SizeType nbeams       = 1);

    ~FDMTFFTCUDA();
    FDMTFFTCUDA(FDMTFFTCUDA&&) noexcept;
    FDMTFFTCUDA& operator=(FDMTFFTCUDA&&) noexcept;
    FDMTFFTCUDA(const FDMTFFTCUDA&)            = delete;
    FDMTFFTCUDA& operator=(const FDMTFFTCUDA&) = delete;

    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    [[nodiscard]] SizeType get_nbeams() const noexcept;

    void execute(std::span<const float> waterfall, std::span<float> dmt);
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream = nullptr);

    void reset(std::span<const float> waterfall, std::span<float> dmt);
    void reset(cuda::std::span<const float> d_waterfall,
               cuda::std::span<float> d_dmt,
               cudaStream_t stream = nullptr);
    void advance(SizeType levels = 1, cudaStream_t stream = nullptr);
    void advance_until_remaining(SizeType remaining_levels,
                                 cudaStream_t stream = nullptr);

    cuda::std::span<const float> view_level_data() const;
    cuda::std::span<const float> view_subband_data(SizeType subband_idx) const;
    FDMTSubbandViewCUDA view_subband(SizeType subband_idx) const;

    SizeType current_level() const noexcept;
    SizeType total_levels() const noexcept;
    SizeType remaining_levels() const noexcept;
    SizeType num_subbands() const;
    bool is_finished() const noexcept;
    void finalize(cudaStream_t stream = nullptr);

    [[nodiscard]] float get_effective_variance(SizeType dm_idx,
                                               SizeType boxcar_width = 1) const;
    [[nodiscard]] float get_effective_sigma(SizeType dm_idx,
                                            SizeType boxcar_width = 1) const;
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width = 1) const;
    [[nodiscard]] std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width = 1) const;

    void reset_history() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};
#endif // DMT_ENABLE_CUDA

} // namespace dmt::algorithms
