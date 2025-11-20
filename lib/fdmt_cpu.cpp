#include "dmt/algorithms/fdmt.hpp"

#include <cstddef>
#include <format>
#include <utility>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"
#include "dmt/kernels.hpp"

namespace dmt::algorithms {

class FDMTCPU::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         SizeType dt_max,
         SizeType dt_step,
         SizeType dt_min,
         bool use_history,
         bool verbose,
         int nthreads)
        : m_use_history(use_history),
          m_plan(f_min,
                 f_max,
                 nchans,
                 nsamps,
                 tsamp,
                 dt_max,
                 dt_step,
                 dt_min,
                 verbose),
          m_state_in(m_plan.get_buffer_size(), 0.0F),
          m_state_out(m_plan.get_buffer_size(), 0.0F),
          m_history(use_history ? m_plan.get_history_size() : 0, 0.0F),
          m_nthreads(set_dmt_openmp_threads(nthreads)) {}

    ~Impl()                      = default;
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    const plans::FDMTPlan& get_plan() const { return m_plan; }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        check_inputs(waterfall.size(), dmt.size());
        initialise(waterfall, std::span(m_state_in));

        float* state_in_ptr  = m_state_in.data();
        float* state_out_ptr = m_state_out.data();
        const auto niters    = m_plan.get_niters();
        for (SizeType i_iter = 1; i_iter < niters; ++i_iter) {
            execute_iter(state_in_ptr, state_out_ptr, i_iter);
            std::swap(state_in_ptr, state_out_ptr);
        }
        // Last iteration directly writes to the output buffer
        execute_iter(state_in_ptr, dmt.data(), niters);
        spdlog::debug("FDMTCPU::Impl: Execution complete.");
    }

private:
    bool m_use_history;
    plans::FDMTPlan m_plan;
    // State buffers
    std::vector<float> m_state_in;
    std::vector<float> m_state_out;
    std::vector<float> m_history;
    int m_nthreads;

    void initialise(std::span<const float> waterfall, std::span<float> state) {
        const auto& plan_c     = m_plan.get_container();
        const auto& grids_init = plan_c.grids[0];
        const auto nsamps      = plan_c.state_shape[0].nsamps;
        const auto dt_max      = plan_c.state_shape[0].dt_max;
        // Use history data only if enabled and buffer is allocated
        std::span<const float> hist_span =
            (m_use_history && !m_history.empty())
                ? std::span<const float>(m_history)
                : std::span<const float>();

#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(waterfall, state, grids_init, nsamps, dt_max, hist_span)
#endif
        for (SizeType i_sub = 0; i_sub < grids_init.size(); ++i_sub) {
            const auto& dt_grid_sub  = grids_init[i_sub].dt_grid;
            const auto buffer_offset = grids_init[i_sub].coord_offset * nsamps;
            const auto waterfall_offset = i_sub * nsamps;
            const auto hist_offset      = i_sub * dt_max;

            // Initialise state for [:, dt_init_min, dt_init_min:]
            const auto dt_min = dt_grid_sub[0];
            for (SizeType isamp = dt_min; isamp < nsamps; ++isamp) {
                float sum = 0.0F;
                for (SizeType i = isamp - dt_min; i <= isamp; ++i) {
                    sum += waterfall[waterfall_offset + i];
                }
                state[buffer_offset + isamp] = sum;
            }
            for (SizeType i_dt = 1; i_dt < dt_grid_sub.size(); ++i_dt) {
                const auto dt_cur           = dt_grid_sub[i_dt];
                const auto dt_prev          = dt_grid_sub[i_dt - 1];
                const auto state_offset_cur = buffer_offset + (i_dt * nsamps);
                const auto state_offset_prev =
                    buffer_offset + ((i_dt - 1) * nsamps);

                // Initialise state for [i_sub, i_dt, dt_cur:]
                for (SizeType isamp = dt_cur; isamp < nsamps; ++isamp) {
                    float sum = 0.0F;
                    for (SizeType i = isamp - dt_cur; i < isamp - dt_prev;
                         ++i) {
                        sum += waterfall[waterfall_offset + i];
                    }
                    state[state_offset_cur + isamp] =
                        state[state_offset_prev + isamp] + sum;
                }
                // Initialise state for [i_sub, i_dt, 0:dt_cur]
                for (SizeType isamp = 0; isamp < dt_cur; ++isamp) {
                    float sum = 0.0F;
                    const auto i_start_rel =
                        static_cast<IndexType>(isamp - dt_cur);
                    const auto i_end_rel =
                        static_cast<IndexType>(isamp - dt_prev);
                    // Sum from history buffer if needed and available
                    if (!hist_span.empty()) {
                        for (IndexType i_rel = i_start_rel;
                             i_rel < 0 && i_rel < i_end_rel; ++i_rel) {
                            // hist contains last dt_max samples
                            sum += hist_span[hist_offset + (dt_max + i_rel)];
                        }
                    }
                    // Sum from waterfall buffer for the remaining part
                    for (IndexType i_rel =
                             std::max(i_start_rel, static_cast<IndexType>(0));
                         i_rel < i_end_rel; ++i_rel) {
                        // Access waterfall using absolute index
                        sum += waterfall[waterfall_offset + i_rel];
                    }
                    state[state_offset_cur + isamp] =
                        state[state_offset_prev + isamp] + sum;
                }
            }
        }
        // Update history buffer if enabled
        if (m_use_history && !m_history.empty()) {
            float* hist_ptr = m_history.data();
            // Copy the last nchans x dt_max elements from waterfall to hist
            for (SizeType i_sub = 0; i_sub < grids_init.size(); ++i_sub) {
                const auto wf_last_elements = waterfall.subspan(
                    (i_sub * nsamps) + nsamps - dt_max, dt_max);
                float* hist_sub_buffer = &hist_ptr[i_sub * dt_max];
                std::copy_n(wf_last_elements.data(), dt_max, hist_sub_buffer);
            }
        }
    }

    void execute_iter(const float* __restrict__ state_in,
                      float* __restrict__ state_out,
                      SizeType i_iter) {
        const auto& plan_c          = m_plan.get_container();
        const auto& coords_sum_cur  = plan_c.coordinates_sum[i_iter];
        const auto& coords_copy_cur = plan_c.coordinates_copy[i_iter];

#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel default(none)                                             \
    shared(state_in, state_out, coords_sum_cur, coords_copy_cur)
#endif
        {
#ifdef DMT_ENABLE_OPENMP
#pragma omp for nowait
#endif
            for (SizeType i_coord = 0; i_coord < coords_sum_cur.size();
                 ++i_coord) {
                const auto& coord = coords_sum_cur[i_coord];
                const float* __restrict__ tail =
                    &state_in[coord.tail_buf_offset];
                const float* __restrict__ head =
                    &state_in[coord.head_buf_offset];
                float* __restrict__ out = &state_out[coord.buf_offset];
                kernels::offset_add(tail, coord.tail_nsamps, head,
                                    coord.head_nsamps, out, coord.nsamps,
                                    coord.delay);
            }
#ifdef DMT_ENABLE_OPENMP
#pragma omp for
#endif
            for (SizeType i_coord = 0; i_coord < coords_copy_cur.size();
                 ++i_coord) {
                const auto& coord = coords_copy_cur[i_coord];
                const float* __restrict__ tail =
                    &state_in[coord.tail_buf_offset];
                float* __restrict__ out = &state_out[coord.buf_offset];
                std::copy_n(tail, coord.tail_nsamps, out);
            }
        }
    }

    void check_inputs(SizeType waterfall_size, SizeType dmt_size) const {
        const auto nchans = m_plan.get_nchans();
        const auto nsamps = m_plan.get_nsamps();
        if (waterfall_size != nchans * nsamps) {
            throw std::invalid_argument(std::format(
                "FDMTCPU: Invalid size of waterfall. Expected {}, got {}",
                nchans * nsamps, waterfall_size));
        }
        if (dmt_size != m_plan.get_dmt_size()) {
            throw std::invalid_argument(
                std::format("FDMTCPU: Invalid size of dmt. Expected {}, got {}",
                            m_plan.get_dmt_size(), dmt_size));
        }
        spdlog::debug("FDMTCPU: Input dimensions check passed: {}x{}", nchans,
                      nsamps);
    }
}; // End FDMTCPU::Impl definition

FDMTCPU::FDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 SizeType dt_max,
                 SizeType dt_step,
                 SizeType dt_min,
                 bool use_history,
                 bool verbose,
                 int nthreads)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_step,
                                    dt_min,
                                    use_history,
                                    verbose,
                                    nthreads)) {}
FDMTCPU::~FDMTCPU()                                   = default;
FDMTCPU::FDMTCPU(FDMTCPU&& other) noexcept            = default;
FDMTCPU& FDMTCPU::operator=(FDMTCPU&& other) noexcept = default;
const plans::FDMTPlan& FDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
void FDMTCPU::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}

} // namespace dmt::algorithms