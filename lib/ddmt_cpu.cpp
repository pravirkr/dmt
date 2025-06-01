#include "dmt/algorithms/ddmt.hpp"

#include <cstddef>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"

namespace dmt::algorithms {

class DDMTCPU::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min,
         int nthreads)
        : m_plan(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min),
          m_nthreads(nthreads) {
#ifdef DMT_ENABLE_OPENMP
        if (m_nthreads <= 0) {
            m_nthreads = omp_get_max_threads();
        }
        omp_set_num_threads(m_nthreads);
        spdlog::debug("DDMTCPU::Impl: Using {} OpenMP threads", m_nthreads);
#endif
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const std::vector<float>& dm_arr,
         int n_threads)
        : m_plan(f_min, f_max, nchans, tsamp, dm_arr),
          m_nthreads(n_threads) {
#ifdef DMT_ENABLE_OPENMP
        if (m_nthreads <= 0) {
            m_nthreads = omp_get_max_threads();
        }
        omp_set_num_threads(m_nthreads);
        spdlog::debug("DDMTCPU::Impl: Using {} OpenMP threads", m_nthreads);
#endif
    }

    const plans::DDMTPlan& get_plan() const { return m_plan; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        const auto& plan_c         = m_plan.get_container();
        const auto nchans          = plan_c.nchans;
        const auto nsamps          = waterfall.size() / nchans;
        const auto max_delay       = plan_c.delay_table.back();
        const auto nsamps_reduced  = nsamps - max_delay;
        const auto out_dm_stride   = nsamps_reduced;
        const auto out_samp_stride = 1;
        const auto in_chan_stride  = nsamps;
        const auto in_samp_stride  = 1;
        const auto* delay_table    = plan_c.delay_table.data();
        const auto dm_count        = plan_c.dm_arr.size();

        if (dmt.size() != dm_count * nsamps_reduced) {
            spdlog::error("Output buffer size mismatch: expected {}, got {}",
                          dm_count * nsamps_reduced, dmt.size());
            return;
        }

        execute_dedisp(waterfall.data(), in_chan_stride, in_samp_stride,
                       dmt.data(), out_dm_stride, out_samp_stride, delay_table,
                       dm_count, nchans, nsamps_reduced);
    }

    void execute_dedisp(const float* __restrict__ d_in,
                        size_t in_chan_stride,
                        size_t in_samp_stride,
                        float* __restrict__ d_out,
                        size_t out_dm_stride,
                        size_t out_samp_stride,
                        const size_t* __restrict__ delay_table,
                        size_t dm_count,
                        size_t nchans,
                        size_t nsamps_reduced) {
#pragma omp parallel for default(none)                                         \
    shared(d_in, d_out, delay_table, dm_count, nchans, nsamps_reduced,         \
               in_chan_stride, in_samp_stride, out_dm_stride, out_samp_stride)
        for (size_t i_dm = 0; i_dm < dm_count; ++i_dm) {
            const auto& delays = &delay_table[i_dm * nchans];
            const auto out_idx = i_dm * out_dm_stride;
            for (size_t i_samp = 0; i_samp < nsamps_reduced; ++i_samp) {
                float sum = 0.0F;
#pragma omp simd reduction(+ : sum)
                for (size_t i_chan = 0; i_chan < nchans; ++i_chan) {
                    const auto& delay = delays[i_chan];
                    sum += d_in[(i_chan * in_chan_stride) +
                                ((i_samp + delay) * in_samp_stride)];
                }
                d_out[out_idx + (i_samp * out_samp_stride)] = sum;
            }
        }
    }

private:
    plans::DDMTPlan m_plan;
    int m_nthreads;
};

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min, nthreads)) {}

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 const std::vector<float>& dm_arr,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, dm_arr, nthreads)) {}
DDMTCPU::~DDMTCPU()                                   = default;
DDMTCPU::DDMTCPU(DDMTCPU&& other) noexcept            = default;
DDMTCPU& DDMTCPU::operator=(DDMTCPU&& other) noexcept = default;
const plans::DDMTPlan& DDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
void DDMTCPU::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}

} // namespace dmt::algorithms