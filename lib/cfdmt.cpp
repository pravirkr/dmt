#include "dmt/algorithms/cfdmt.hpp"

#include <span>
#include <stdexcept>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif
#include <fftw3.h>

#include <spdlog/spdlog.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bb_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/utils/fft.hpp"
#include "dmt/utils/unpacker.hpp"

namespace dmt::algorithms {

class CohFDMTCPU::Impl {
public:
    Impl(float f_center,
         float bw_sub,
         SizeType nsub,
         float tbin,
         SizeType nbin,
         SizeType nfft,
         float t_p,
         float dm_max,
         float dm_min,
         SizeType noverlap,
         std::string_view data_order,
         bool verbose,
         int nthreads)
        : m_plan(f_center,
                 bw_sub,
                 nsub,
                 tbin,
                 nbin,
                 nfft,
                 t_p,
                 dm_max,
                 dm_min,
                 noverlap,
                 data_order,
                 verbose),
          m_nthreads(set_dmt_openmp_threads(nthreads)) {
        initialise();
    }

    const plans::CohFDMTPlan& get_plan() const { return m_plan; }

    template <IntegralDataType DataType>
    void execute(std::span<const DataType> data_in, std::span<float> dmt) {
        if (dmt.size() != m_plan.get_dmt_size()) {
            throw std::runtime_error("Invalid DMT size");
        }
        m_theunpacker->execute<DataType>(data_in, m_unpack_buf_p1,
                                         m_unpack_buf_p2);
        // Forward FFT
        m_thefft->forward_fft(m_unpack_buf_p1, m_unpack_buf_p2);
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        for (SizeType idm = 0; idm < dm_grid_coh.size(); ++idm) {
            // Apply chirp
            apply_chirp(m_unpack_buf_p1, m_unpack_buf_p2, m_delay_buf_p1,
                        m_delay_buf_p2, idm);
            // Backward FFT
            m_thefft->backward_fft(m_delay_buf_p1, m_delay_buf_p2);
            // Detect and unpad
            unpad_detect(m_delay_buf_p1, m_delay_buf_p2, m_intensity_buf);
            // Apply causal streaming inter-channel delay line for this coarse trial
            m_delay_line.process(m_intensity_buf, m_aligned_buf, idm,
                                 m_plan.get_mchan(), m_plan.get_msamp());

            // Perform the FDMT for this coarse-DM trial's fine residual
            // search on the aligned waterfall. All coarse trials share one
            // FDMTCPU instance (same plan geometry, "valid" mode); only the
            // small per-trial streaming history differs, so it's swapped in/out
            // around the shared instance's working-state buffers.
            m_thefdmt->load_history(m_dm_histories[idm]);
            m_thefdmt->execute(m_aligned_buf, m_fdmt_scratch);
            m_thefdmt->save_history(m_dm_histories[idm]);
            const auto dmt_cur_size = m_thefdmt->get_plan().get_dmt_size();
            auto dmt_cur_span = dmt.subspan(idm * dmt_cur_size, dmt_cur_size);
            std::copy_n(m_fdmt_scratch.begin(), dmt_cur_size,
                       dmt_cur_span.begin());
        }
    }

    void reset_history() noexcept {
        m_delay_line.reset_history();
        for (auto& hist : m_dm_histories) {
            std::fill(hist.begin(), hist.end(), 0.0F);
        }
    }

private:
    plans::CohFDMTPlan m_plan;
    int m_nthreads;
    std::unique_ptr<utils::FFTManagerCPU> m_thefft;
    std::unique_ptr<algorithms::FDMTCPU> m_thefdmt;
    std::unique_ptr<utils::DataUnpackerCPU> m_theunpacker;
    utils::ChannelDelayLineCPU m_delay_line;

    std::vector<ComplexType> m_unpack_buf_p1;
    std::vector<ComplexType> m_unpack_buf_p2;
    std::vector<ComplexType> m_delay_buf_p1;
    std::vector<ComplexType> m_delay_buf_p2;
    std::vector<float> m_intensity_buf;
    std::vector<float> m_aligned_buf;
    std::vector<ComplexType> m_chirp_table;
    // FDMT ping-pong scratch, reused across coarse-DM trials -- sized to
    // get_buffer_size() (>= get_dmt_size()); see execute()'s comment.
    std::vector<float> m_fdmt_scratch;
    // Small per-coarse-DM-trial "valid"-mode streaming history, swapped
    // into the one shared m_thefdmt instance around each trial's execute()
    // call. Persists across CohFDMTCPU::execute() calls (future contiguous
    // baseband blocks); reset via reset_history().
    std::vector<std::vector<float>> m_dm_histories;

    void initialise() {
        // Allocate buffers first: initialize_plans() (FFTW_MEASURE) actually
        // reads/writes through the buffers while planning, so they must
        // already be sized before the FFT manager touches them.
        m_unpack_buf_p1.resize(m_plan.get_unpack_buf_size());
        m_unpack_buf_p2.resize(m_plan.get_unpack_buf_size());
        m_delay_buf_p1.resize(m_plan.get_delay_buf_size());
        m_delay_buf_p2.resize(m_plan.get_delay_buf_size());
        m_intensity_buf.resize(m_plan.get_intensity_buf_size());
        m_aligned_buf.resize(m_plan.get_intensity_buf_size());
        m_chirp_table.resize(m_plan.get_chirp_table_size());

        // Initialise the FFT manager, FDMT and data unpacker
        m_thefft = std::make_unique<utils::FFTManagerCPU>(
            m_plan.get_nfft(), m_plan.get_nsub(), m_plan.get_nbin(),
            m_plan.get_mbin(), m_plan.get_nchan(), m_nthreads);
        m_thefft->initialize_plans(std::span<ComplexType>(m_unpack_buf_p1),
                                   std::span<ComplexType>(m_delay_buf_p1));
        m_thefdmt = std::make_unique<algorithms::FDMTCPU>(
            m_plan.get_f_min(), m_plan.get_f_max(), m_plan.get_mchan(),
            m_plan.get_msamp(), m_plan.get_tsamp(), m_plan.get_dt_max(),
            m_plan.get_dt_min(), 1, true, "valid", false, m_nthreads);
        m_theunpacker = std::make_unique<utils::DataUnpackerCPU>(
            m_plan.get_nsub(), m_plan.get_nbin(), m_plan.get_noverlap(),
            m_plan.get_nfft(), m_plan.get_data_order(), m_nthreads);

        m_fdmt_scratch.resize(m_thefdmt->get_plan().get_buffer_size());

        // One small streaming-history slot per coarse-DM trial.
        const auto& dm_grid_coh = m_plan.get_dm_grid_coh();
        m_dm_histories.assign(
            dm_grid_coh.size(),
            std::vector<float>(m_thefdmt->history_state_size(), 0.0F));

        // Initialise the causal inter-channel streaming delay line
        m_delay_line.initialise(dm_grid_coh, m_plan.get_f_min(), m_plan.get_f_max(),
                                m_plan.get_mchan(), m_plan.get_tsamp());

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
                              m_plan.get_nfft(), idm, scale);
    }

    void unpad_detect(std::span<const ComplexType> fft_p1,
                      std::span<const ComplexType> fft_p2,
                      std::span<float> intensity) const {
        bb_utils::unpad_detect(fft_p1, fft_p2, intensity, m_plan.get_nchan(),
                               m_plan.get_nfft(), m_plan.get_nsub(),
                               m_plan.get_mbin(), m_plan.get_noverlap());
    }

}; // End CohFDMTCPU::Impl definition

CohFDMTCPU::CohFDMTCPU(float f_center,
                       float bw_sub,
                       SizeType nsub,
                       float tbin,
                       SizeType nbin,
                       SizeType nfft,
                       float t_p,
                       float dm_max,
                       float dm_min,
                       SizeType noverlap,
                       std::string_view data_order,
                       bool verbose,
                       int nthreads)
    : m_impl(std::make_unique<Impl>(f_center,
                                    bw_sub,
                                    nsub,
                                    tbin,
                                    nbin,
                                    nfft,
                                    t_p,
                                    dm_max,
                                    dm_min,
                                    noverlap,
                                    data_order,
                                    verbose,
                                    nthreads)) {}

CohFDMTCPU::~CohFDMTCPU()                                      = default;
CohFDMTCPU::CohFDMTCPU(CohFDMTCPU&& other) noexcept            = default;
CohFDMTCPU& CohFDMTCPU::operator=(CohFDMTCPU&& other) noexcept = default;
const plans::CohFDMTPlan& CohFDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
SizeType CohFDMTCPU::get_dmt_size() const noexcept {
    return m_impl->get_plan().get_dmt_size();
}
template <IntegralDataType DataType>
void CohFDMTCPU::execute(std::span<const DataType> data_in,
                         std::span<float> dmt) const {
    m_impl->execute<DataType>(data_in, dmt);
}
void CohFDMTCPU::reset_history() noexcept { m_impl->reset_history(); }

// Instantiate the public execute method for each supported DataType
template void CohFDMTCPU::execute<int8_t>(std::span<const int8_t>,
                                          std::span<float>) const;
template void CohFDMTCPU::execute<uint8_t>(std::span<const uint8_t>,
                                           std::span<float>) const;

} // namespace dmt::algorithms
