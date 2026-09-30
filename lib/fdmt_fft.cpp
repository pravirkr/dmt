#include "dmt/algorithms/fdmt_fft.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "dmt/logging.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms {

namespace {

std::vector<uint8_t> checked_kill_mask(std::span<const uint8_t> mask,
                                       SizeType nchans) {
    if (!mask.empty() && mask.size() != nchans) {
        throw std::invalid_argument(
            std::format("FDMTFFT: kill_mask has {} entries, expected nchans = "
                        "{}",
                        mask.size(), nchans));
    }
    if (std::ranges::all_of(mask, [](uint8_t v) { return v != 0; })) {
        return {}; // keeps every channel
    }
    return {mask.begin(), mask.end()};
}

std::unique_ptr<detail::FDMTFFTEngine>
make_fdmt_fft_engine(const plans::FDMTPlan& plan,
                     const detail::FDMTFFTEngineConfig& cfg) {
    if (cfg.exec.backend == Backend::kCPU) {
        return detail::make_fdmt_fft_cpu(plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (cfg.exec.backend == detail::kGPUBackend) {
        return detail::make_fdmt_fft_gpu(plan, cfg);
    }
#endif
    detail::throw_unavailable("FDMTFFT", cfg.exec.backend);
}

} // namespace

class FDMTFFT::Impl {
public:
    Impl(plans::FDMTPlan plan,
         bool use_box_smearing,
         std::string_view mode,
         Exec exec,
         SizeType nbeams,
         bool fractional_delays,
         std::span<const uint8_t> kill_mask)
        : m_plan(std::move(plan)),
          m_cfg{
              .use_box_smearing  = use_box_smearing,
              .mode              = parse_fdmt_mode(mode),
              .nbeams            = nbeams,
              .exec              = exec,
              .fractional_delays = fractional_delays,
              .kill_mask = checked_kill_mask(kill_mask, m_plan.get_nchans()),
          },
          m_geom(fdmt_fft::make_geometry(
              m_plan, m_cfg.mode, fractional_delays, use_box_smearing)),
          m_engine(make_fdmt_fft_engine(m_plan, m_cfg)) {
        const auto ns = m_plan.get_nsamps();
        if (m_cfg.mode == FDMTMode::kValid && (2 * ns) < (3 * m_geom.overlap)) {
            logging::debug("FDMTFFT: blocks of {} samples against a {}-sample "
                           "overlap keep only {:.0f}% of each transform; "
                           "blocks of >= {} samples (get_suggested_nsamps()) "
                           "keep 80%",
                           ns, m_geom.overlap,
                           100.0 * static_cast<double>(ns) /
                               static_cast<double>(ns + m_geom.overlap),
                           suggested_nsamps());
        }
    }

    [[nodiscard]] SizeType suggested_nsamps() const {
        if (m_cfg.mode != FDMTMode::kValid) {
            return m_plan.get_nsamps();
        }
        const auto h = std::max<SizeType>(m_geom.overlap, 1);
        return utils::next_fft_size(5 * h) - h;
    }

    // The engine holds a pointer to m_plan; declaration order matters.
    plans::FDMTPlan m_plan;
    detail::FDMTFFTEngineConfig m_cfg;
    fdmt_fft::Geometry m_geom;
    std::unique_ptr<detail::FDMTFFTEngine> m_engine;

    [[nodiscard]] bool on_cpu() const noexcept {
        return m_cfg.exec.backend == Backend::kCPU;
    }

    void check_stream(Stream stream, std::string_view what) const {
        if (on_cpu() && stream.native != nullptr) {
            throw std::invalid_argument(std::format(
                "FDMTFFT::{}: a stream was given, but the cpu backend has "
                "none",
                what));
        }
    }

    template <typename T>
    void check_device(const DeviceSpan<T>& span, std::string_view what) const {
        detail::check_device(span.device, m_cfg.exec.backend, m_cfg.exec.device,
                             what);
    }
};

FDMTFFT::FDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 IndexType dt_max,
                 IndexType dt_min,
                 SizeType dt_step,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays,
                 std::span<const uint8_t> kill_mask)
    : m_impl(std::make_unique<Impl>(plans::FDMTPlan(f_min,
                                                    f_max,
                                                    nchans,
                                                    nsamps,
                                                    tsamp,
                                                    dt_max,
                                                    dt_min,
                                                    dt_step,
                                                    mode),
                                    use_box_smearing,
                                    mode,
                                    exec,
                                    nbeams,
                                    fractional_delays,
                                    kill_mask)) {}

FDMTFFT::FDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<IndexType>& dt_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays,
                 std::span<const uint8_t> kill_mask)
    : m_impl(std::make_unique<Impl>(
          plans::FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode),
          use_box_smearing,
          mode,
          exec,
          nbeams,
          fractional_delays,
          kill_mask)) {}

FDMTFFT::FDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays,
                 std::span<const uint8_t> kill_mask)
    : m_impl(std::make_unique<Impl>(
          plans::FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode),
          use_box_smearing,
          mode,
          exec,
          nbeams,
          fractional_delays,
          kill_mask)) {}

FDMTFFT::~FDMTFFT()                                   = default;
FDMTFFT::FDMTFFT(FDMTFFT&& other) noexcept            = default;
FDMTFFT& FDMTFFT::operator=(FDMTFFT&& other) noexcept = default;

const plans::FDMTPlan& FDMTFFT::get_plan() const noexcept {
    return m_impl->m_plan;
}
SizeType FDMTFFT::get_nbeams() const noexcept { return m_impl->m_cfg.nbeams; }
Backend FDMTFFT::backend() const noexcept { return m_impl->m_cfg.exec.backend; }
int FDMTFFT::nthreads() const noexcept {
    return m_impl->on_cpu() ? std::max(1, m_impl->m_cfg.exec.nthreads) : 1;
}
int FDMTFFT::device() const noexcept {
    return m_impl->on_cpu() ? -1 : m_impl->m_cfg.exec.device;
}
bool FDMTFFT::fractional_delays() const noexcept {
    return m_impl->m_cfg.fractional_delays;
}
SizeType FDMTFFT::get_output_latency() const noexcept {
    return m_impl->m_geom.latency;
}
SizeType FDMTFFT::get_suggested_nsamps() const noexcept {
    return m_impl->suggested_nsamps();
}

void FDMTFFT::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->m_engine->execute(waterfall, dmt);
}
void FDMTFFT::execute(DeviceSpan<const float> d_waterfall,
                      DeviceSpan<float> d_dmt,
                      Stream stream) {
    m_impl->check_device(d_waterfall, "FDMTFFT::execute");
    m_impl->check_device(d_dmt, "FDMTFFT::execute");
    m_impl->m_engine->execute(d_waterfall, d_dmt, stream);
}
void FDMTFFT::execute(std::span<const uint8_t> waterfall_packed,
                      SizeType nbits,
                      std::span<float> dmt) {
    m_impl->m_engine->execute(waterfall_packed, nbits, false, dmt);
}
void FDMTFFT::execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                      SizeType nbits,
                      DeviceSpan<float> d_dmt,
                      Stream stream) {
    m_impl->check_device(d_waterfall_packed, "FDMTFFT::execute");
    m_impl->check_device(d_dmt, "FDMTFFT::execute");
    m_impl->m_engine->execute(d_waterfall_packed, nbits, d_dmt, stream);
}
void FDMTFFT::execute_time_major(std::span<const uint8_t> filterbank_packed,
                                 SizeType nbits,
                                 std::span<float> dmt) {
    m_impl->m_engine->execute(filterbank_packed, nbits, true, dmt);
}
void FDMTFFT::reset(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->m_engine->reset(waterfall, dmt);
}
void FDMTFFT::reset(std::span<const uint8_t> waterfall_packed,
                    SizeType nbits,
                    std::span<float> dmt) {
    m_impl->m_engine->reset(waterfall_packed, nbits, dmt);
}
void FDMTFFT::reset(DeviceSpan<const uint8_t> d_waterfall_packed,
                    SizeType nbits,
                    DeviceSpan<float> d_dmt,
                    Stream stream) {
    m_impl->check_device(d_waterfall_packed, "FDMTFFT::reset");
    m_impl->check_device(d_dmt, "FDMTFFT::reset");
    m_impl->m_engine->reset(d_waterfall_packed, nbits, d_dmt, stream);
}
void FDMTFFT::reset(DeviceSpan<const float> d_waterfall,
                    DeviceSpan<float> d_dmt,
                    Stream stream) {
    m_impl->check_device(d_waterfall, "FDMTFFT::reset");
    m_impl->check_device(d_dmt, "FDMTFFT::reset");
    m_impl->m_engine->reset(d_waterfall, d_dmt, stream);
}

void FDMTFFT::advance(SizeType levels, Stream stream) {
    m_impl->check_stream(stream, "advance");
    m_impl->m_engine->advance(levels, stream);
}
void FDMTFFT::advance_until_remaining(SizeType remaining_levels,
                                      Stream stream) {
    m_impl->check_stream(stream, "advance_until_remaining");
    m_impl->m_engine->advance_until_remaining(remaining_levels, stream);
}
void FDMTFFT::finalize(Stream stream) {
    m_impl->check_stream(stream, "finalize");
    m_impl->m_engine->finalize(stream);
}

std::span<const float> FDMTFFT::view_level_data() const {
    return m_impl->m_engine->view_level_data();
}
std::span<const float> FDMTFFT::view_subband_data(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband(subband_idx).data;
}
FDMTSubbandView FDMTFFT::view_subband(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband(subband_idx);
}
DeviceSpan<const float> FDMTFFT::view_level_data_device() const {
    return m_impl->m_engine->view_level_data_device();
}
FDMTSubbandDeviceView FDMTFFT::view_subband_device(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband_device(subband_idx);
}

SizeType FDMTFFT::current_level() const noexcept {
    return m_impl->m_engine->current_level();
}
SizeType FDMTFFT::total_levels() const noexcept {
    return m_impl->m_plan.get_niters() + 1;
}
SizeType FDMTFFT::remaining_levels() const noexcept {
    const auto total = total_levels();
    const auto cur   = current_level();
    return (total <= 1 || cur >= total - 1) ? 0 : (total - 1) - cur;
}
SizeType FDMTFFT::num_subbands() const {
    return m_impl->m_engine->num_subbands();
}
bool FDMTFFT::is_finished() const noexcept {
    return m_impl->m_engine->is_finished();
}

float FDMTFFT::get_effective_variance(SizeType dm_idx,
                                      SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_variance(
        dm_idx, boxcar_width, m_impl->m_cfg.use_box_smearing);
}
float FDMTFFT::get_effective_sigma(SizeType dm_idx,
                                   SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_sigma(dm_idx, boxcar_width,
                                              m_impl->m_cfg.use_box_smearing);
}
std::vector<float>
FDMTFFT::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_variance_grid(
        boxcar_width, m_impl->m_cfg.use_box_smearing);
}
std::vector<float>
FDMTFFT::get_effective_sigma_grid(SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_sigma_grid(
        boxcar_width, m_impl->m_cfg.use_box_smearing);
}

void FDMTFFT::reset_history() noexcept { m_impl->m_engine->reset_history(); }

SizeType FDMTFFT::history_state_size() const noexcept {
    return m_impl->m_engine->history_state_size();
}
void FDMTFFT::save_history(std::span<float> out) const {
    m_impl->m_engine->save_history(out);
}
void FDMTFFT::load_history(std::span<const float> in) {
    m_impl->m_engine->load_history(in);
}
void FDMTFFT::save_history(DeviceSpan<float> d_out, Stream stream) const {
    m_impl->check_device(d_out, "FDMTFFT::save_history");
    m_impl->m_engine->save_history(d_out, stream);
}
void FDMTFFT::load_history(DeviceSpan<const float> d_in, Stream stream) {
    m_impl->check_device(d_in, "FDMTFFT::load_history");
    m_impl->m_engine->load_history(d_in, stream);
}

namespace {

std::tuple<std::vector<float>, plans::FDMTPlan>
run_compute_fdmt_fft(FDMTFFT& fdmt, std::span<const float> waterfall) {
    std::vector<float> dmt(fdmt.get_plan().get_dmt_size() * fdmt.get_nbeams());
    fdmt.execute(waterfall, dmt);
    return {std::move(dmt), fdmt.get_plan()};
}

} // namespace

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 IndexType dt_max,
                 IndexType dt_min,
                 SizeType dt_step,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays) {
    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                 use_box_smearing, mode, exec, nbeams, fractional_delays);
    return run_compute_fdmt_fft(fdmt, waterfall);
}

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<IndexType>& dt_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays) {
    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid, use_box_smearing,
                 mode, exec, nbeams, fractional_delays);
    return run_compute_fdmt_fft(fdmt, waterfall);
}

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt_fft(std::span<const float> waterfall,
                 float f_min,
                 float f_max,
                 SizeType nchans,
                 SizeType nsamps,
                 float tsamp,
                 const std::vector<float>& dm_grid,
                 bool use_box_smearing,
                 std::string_view mode,
                 Exec exec,
                 SizeType nbeams,
                 bool fractional_delays) {
    FDMTFFT fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid, use_box_smearing,
                 mode, exec, nbeams, fractional_delays);
    return run_compute_fdmt_fft(fdmt, waterfall);
}

} // namespace dmt::algorithms
