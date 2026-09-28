#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <format>
#include <span>
#include <stdexcept>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bb_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft.hpp"
#include "dmt/unpacker.hpp"

namespace dmt::algorithms {

namespace {

class CohFDMTCpuEngine final : public detail::CohFDMTEngine {
public:
    CohFDMTCpuEngine(const plans::CohFDMTPlan& plan,
                     const detail::CohFDMTEngineConfig& cfg)
        : m_plan(plan),
          m_nthreads(std::max(1, cfg.exec.nthreads)) {
        initialise();
    }

    // Keep the base's device-memory overloads (throwing defaults) visible
    // next to the host overrides and the templated implementation below.
    using detail::CohFDMTEngine::execute;

    void execute(std::span<const uint8_t> data_in,
                 std::span<float> dmt) override {
        execute<uint8_t>(data_in, dmt);
    }
    void execute(std::span<const int8_t> data_in,
                 std::span<float> dmt) override {
        execute<int8_t>(data_in, dmt);
    }

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) {
        if (dmt.size() < m_plan.get_buffer_size()) {
            throw std::invalid_argument(std::format(
                "CohFDMT: dmt buffer too small. Expected at least {} "
                "(get_buffer_size()), got {}",
                m_plan.get_buffer_size(), dmt.size()));
        }
        m_theunpacker->execute<DataType>(data_in, m_unpack_buf_p1,
                                         m_unpack_buf_p2);
        m_fft_forward->execute(m_unpack_buf_p1);
        m_fft_forward->execute(m_unpack_buf_p2);
        bb_utils::swap_spectrum(
            m_unpack_buf_p1, m_unpack_buf_p2,
            static_cast<int>(m_plan.get_nbin()),
            static_cast<int>(m_plan.get_nfft() * m_plan.get_nsub()),
            m_nthreads);
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
            // Apply chirp
            apply_chirp(m_unpack_buf_p1, m_unpack_buf_p2, m_delay_buf_p1,
                        m_delay_buf_p2, idm);
            bb_utils::swap_spectrum(
                m_delay_buf_p1, m_delay_buf_p2,
                static_cast<int>(m_plan.get_mbin()),
                static_cast<int>(m_plan.get_nfft() * m_plan.get_nsub() *
                                 m_plan.get_nchan()),
                m_nthreads);
            m_fft_backward->execute(m_delay_buf_p1);
            m_fft_backward->execute(m_delay_buf_p2);
            // Detect and unpad
            unpad_detect(m_delay_buf_p1, m_delay_buf_p2, m_intensity_buf);
            // Apply causal streaming inter-channel delay line for this coarse
            // trial
            m_delay_line.process(m_intensity_buf, m_aligned_buf, idm,
                                 m_plan.get_mchan(), m_plan.get_msamp());

            // Perform the FDMT for this coarse-DM trial's fine residual
            // search on the aligned waterfall. All coarse trials share one
            // CPU FDMT instance (same plan geometry, "valid" mode); only the
            // small per-trial streaming history differs, so it's swapped in/out
            // around the shared instance's working-state buffers.
            //
            // Sliding arena: the trial runs in place at offset idm * D with
            // its full B-sized ping-pong span. Its scratch tail only reaches
            // later trials' slots, which are overwritten afterwards, so no
            // copy is needed (see CohFDMTPlan::get_buffer_size()).
            const auto& fine_plan = m_thefdmt->get_plan();
            m_thefdmt->load_history(m_dm_histories[idm]);
            m_thefdmt->execute(m_aligned_buf,
                               dmt.subspan(idm * fine_plan.get_dmt_size(),
                                           fine_plan.get_buffer_size()));
            m_thefdmt->save_history(m_dm_histories[idm]);
        }
    }

    void reset_history() noexcept override {
        m_delay_line.reset_history();
        for (auto& hist : m_dm_histories) {
            std::ranges::fill(hist, 0.0F);
        }
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    const plans::CohFDMTPlan& m_plan; // owned by the CohFDMT facade
    int m_nthreads;
    std::unique_ptr<utils::FFTWManager> m_fft_forward;
    std::unique_ptr<utils::FFTWManager> m_fft_backward;
    std::unique_ptr<algorithms::FDMT> m_thefdmt;
    std::unique_ptr<utils::DataUnpackerCPU> m_theunpacker;
    utils::ChannelDelayLineCPU m_delay_line;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<float> m_aligned_buf;
    std::vector<ComplexType> m_chirp_table;
    // Small per-coarse-DM-trial "valid"-mode streaming history, swapped
    // into the one shared m_thefdmt instance around each trial's execute()
    // call. Persists across CohFDMT::execute() calls (future contiguous
    // baseband blocks); reset via reset_history().
    std::vector<std::vector<float>> m_dm_histories;

    void initialise() {
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_aligned_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());

        m_fft_forward = std::make_unique<utils::FFTWManager>(
            utils::FFTKind::kC2CForward, m_plan.get_nbin(),
            m_plan.get_nfft() * m_plan.get_nsub(), m_nthreads);
        m_fft_backward = std::make_unique<utils::FFTWManager>(
            utils::FFTKind::kC2CBackward, m_plan.get_mbin(),
            m_plan.get_nfft() * m_plan.get_nsub() * m_plan.get_nchan(),
            m_nthreads);
        m_thefdmt = std::make_unique<algorithms::FDMT>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max(),
            m_plan.get_dt_min(), 1, true, "valid", Exec::cpu(m_nthreads));
        m_theunpacker = std::make_unique<utils::DataUnpackerCPU>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order(), m_nthreads);

        // One small streaming-history slot per coarse-DM trial.
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        m_dm_histories.assign(
            dm_grid_coh.size(),
            std::vector<float>(m_thefdmt->history_state_size(), 0.0F));

        // Initialise the causal inter-channel streaming delay line
        m_delay_line.initialise(dm_grid_coh, m_plan.get_f_min(),
                                m_plan.get_f_max(), m_plan.get_mchan(),
                                m_plan.get_tsamp());

        // Compute the chirp table
        bb_utils::compute_chirp(
            dm_grid_coh, m_chirp_table, m_plan.get_f_center(), m_plan.get_bw(),
            m_plan.get_nbin(), m_plan.get_nsub(), m_plan.get_nchan());
    }

    void apply_chirp(std::span<const ComplexType> data1_in,
                     std::span<const ComplexType> data2_in,
                     std::span<ComplexType> data1_out,
                     std::span<ComplexType> data2_out,
                     SizeType idm) {
        const auto scale = m_plan.get_chirp_scale();
        bb_utils::apply_chirp(data1_in, data2_in, m_chirp_table, data1_out,
                              data2_out, m_plan.get_nsub(), m_plan.get_nbin(),
                              m_plan.get_nfft(), idm, scale, m_nthreads);
    }

    void unpad_detect(std::span<const ComplexType> fft_p1,
                      std::span<const ComplexType> fft_p2,
                      std::span<float> intensity) const {
        bb_utils::unpad_detect(fft_p1, fft_p2, intensity, m_plan.get_nchan(),
                               m_plan.get_nfft(), m_plan.get_nsub(),
                               m_plan.get_mbin(), m_plan.get_noverlap(),
                               m_nthreads);
    }

}; // End CohFDMTCpuEngine definition

} // namespace

std::unique_ptr<detail::CohFDMTEngine>
detail::make_cfdmt_cpu(const plans::CohFDMTPlan& plan,
                       const detail::CohFDMTEngineConfig& cfg) {
    return std::make_unique<CohFDMTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
