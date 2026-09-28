#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms {

namespace {

std::unique_ptr<detail::FDMTEngine>
make_fdmt_engine(const plans::FDMTPlan& plan,
                 const detail::FDMTEngineConfig& cfg) {
    if (cfg.exec.backend == Backend::kCPU) {
        return detail::make_fdmt_cpu(plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (cfg.exec.backend == detail::kGPUBackend) {
        return detail::make_fdmt_gpu(plan, cfg);
    }
#endif
    detail::throw_unavailable("FDMT", cfg.exec.backend);
}

} // namespace

class FDMT::Impl {
public:
    Impl(plans::FDMTPlan plan,
         bool use_box_smearing,
         std::string_view mode,
         SizeType nbeams,
         SizeType fuse_levels,
         bool int_tree,
         Exec exec)
        : m_plan(std::move(plan)),
          m_cfg{
              .use_box_smearing = use_box_smearing,
              .mode             = parse_fdmt_mode(mode),
              .nbeams           = nbeams,
              .fuse_levels      = fuse_levels,
              .int_tree         = int_tree,
              .exec             = exec,
          },
          m_engine(make_fdmt_engine(m_plan, m_cfg)) {}

    // The engine holds a reference to m_plan; declaration order matters.
    plans::FDMTPlan m_plan;
    detail::FDMTEngineConfig m_cfg;
    std::unique_ptr<detail::FDMTEngine> m_engine;

    [[nodiscard]] bool on_cpu() const noexcept {
        return m_cfg.exec.backend == Backend::kCPU;
    }

    // The CPU stepper has no queue: a stream there is a caller mistake.
    void check_stream(Stream stream, std::string_view what) const {
        if (on_cpu() && stream.native != nullptr) {
            throw std::invalid_argument(std::format(
                "FDMT::{}: a stream was given, but the cpu backend has none",
                what));
        }
    }

    template <typename T>
    void check_device(const DeviceSpan<T>& span, std::string_view what) const {
        detail::check_device(span.device, m_cfg.exec.backend, m_cfg.exec.device,
                             what);
    }
};

FDMT::FDMT(float f_min,
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
           SizeType fuse_levels,
           bool int_tree)
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
                                    nbeams,
                                    fuse_levels,
                                    int_tree,
                                    exec)) {}

FDMT::FDMT(float f_min,
           float f_max,
           SizeType nchans,
           SizeType nsamps,
           float tsamp,
           const std::vector<IndexType>& dt_grid,
           bool use_box_smearing,
           std::string_view mode,
           Exec exec,
           SizeType nbeams,
           SizeType fuse_levels,
           bool int_tree)
    : m_impl(std::make_unique<Impl>(
          plans::FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode),
          use_box_smearing,
          mode,
          nbeams,
          fuse_levels,
          int_tree,
          exec)) {}

FDMT::FDMT(float f_min,
           float f_max,
           SizeType nchans,
           SizeType nsamps,
           float tsamp,
           const std::vector<float>& dm_grid,
           bool use_box_smearing,
           std::string_view mode,
           Exec exec,
           SizeType nbeams,
           SizeType fuse_levels,
           bool int_tree)
    : m_impl(std::make_unique<Impl>(
          plans::FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode),
          use_box_smearing,
          mode,
          nbeams,
          fuse_levels,
          int_tree,
          exec)) {}

FDMT::~FDMT()                                = default;
FDMT::FDMT(FDMT&& other) noexcept            = default;
FDMT& FDMT::operator=(FDMT&& other) noexcept = default;

const plans::FDMTPlan& FDMT::get_plan() const noexcept {
    return m_impl->m_plan;
}
SizeType FDMT::get_nbeams() const noexcept { return m_impl->m_cfg.nbeams; }
Backend FDMT::backend() const noexcept { return m_impl->m_cfg.exec.backend; }
int FDMT::nthreads() const noexcept {
    return m_impl->on_cpu() ? std::max(1, m_impl->m_cfg.exec.nthreads) : 1;
}
int FDMT::device() const noexcept {
    return m_impl->on_cpu() ? -1 : m_impl->m_cfg.exec.device;
}

void FDMT::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->m_engine->execute(waterfall, dmt);
}
void FDMT::execute(std::span<const uint8_t> waterfall_packed,
                   SizeType nbits,
                   std::span<float> dmt) {
    m_impl->m_engine->execute(waterfall_packed, nbits, dmt);
}
void FDMT::execute(DeviceSpan<const float> d_waterfall,
                   DeviceSpan<float> d_dmt,
                   Stream stream) {
    m_impl->check_device(d_waterfall, "FDMT::execute");
    m_impl->check_device(d_dmt, "FDMT::execute");
    m_impl->m_engine->execute(d_waterfall, d_dmt, stream);
}
void FDMT::execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                   SizeType nbits,
                   DeviceSpan<float> d_dmt,
                   Stream stream) {
    m_impl->check_device(d_waterfall_packed, "FDMT::execute");
    m_impl->check_device(d_dmt, "FDMT::execute");
    m_impl->m_engine->execute(d_waterfall_packed, nbits, d_dmt, stream);
}

SizeType FDMT::get_fuse_levels() const noexcept {
    return m_impl->m_engine->get_fuse_levels();
}
bool FDMT::get_int_tree() const noexcept { return m_impl->m_cfg.int_tree; }
FDMTMemoryUsage FDMT::get_memory_usage() const noexcept {
    return m_impl->m_engine->get_memory_usage();
}
std::string FDMT::summary() const { return m_impl->m_engine->summary(); }

void FDMT::reset(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->m_engine->reset(waterfall, dmt);
}
void FDMT::reset(std::span<const uint8_t> waterfall_packed,
                 SizeType nbits,
                 std::span<float> dmt) {
    m_impl->m_engine->reset(waterfall_packed, nbits, dmt);
}
void FDMT::reset(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) {
    m_impl->check_device(d_waterfall, "FDMT::reset");
    m_impl->check_device(d_dmt, "FDMT::reset");
    m_impl->m_engine->reset(d_waterfall, d_dmt, stream);
}
void FDMT::reset(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nbits,
                 DeviceSpan<float> d_dmt,
                 Stream stream) {
    m_impl->check_device(d_waterfall_packed, "FDMT::reset");
    m_impl->check_device(d_dmt, "FDMT::reset");
    m_impl->m_engine->reset(d_waterfall_packed, nbits, d_dmt, stream);
}

void FDMT::advance(SizeType levels, Stream stream) {
    m_impl->check_stream(stream, "advance");
    m_impl->m_engine->advance(levels, stream);
}
void FDMT::advance_until_remaining(SizeType remaining_levels, Stream stream) {
    m_impl->check_stream(stream, "advance_until_remaining");
    m_impl->m_engine->advance_until_remaining(remaining_levels, stream);
}
void FDMT::finalize(Stream stream) {
    m_impl->check_stream(stream, "finalize");
    m_impl->m_engine->finalize(stream);
}

std::span<const float> FDMT::view_level_data() const {
    return m_impl->m_engine->view_level_data();
}
std::span<const float> FDMT::view_subband_data(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband(subband_idx).data;
}
FDMTSubbandView FDMT::view_subband(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband(subband_idx);
}
DeviceSpan<const float> FDMT::view_level_data_device() const {
    return m_impl->m_engine->view_level_data_device();
}
FDMTSubbandDeviceView FDMT::view_subband_device(SizeType subband_idx) const {
    return m_impl->m_engine->view_subband_device(subband_idx);
}

SizeType FDMT::current_level() const noexcept {
    return m_impl->m_engine->current_level();
}
SizeType FDMT::total_levels() const noexcept {
    return m_impl->m_plan.get_niters() + 1;
}
SizeType FDMT::remaining_levels() const noexcept {
    const auto total = total_levels();
    const auto cur   = current_level();
    return (total <= 1 || cur >= total - 1) ? 0 : (total - 1) - cur;
}
SizeType FDMT::num_subbands() const { return m_impl->m_engine->num_subbands(); }
bool FDMT::is_finished() const noexcept {
    return m_impl->m_engine->is_finished();
}

float FDMT::get_effective_variance(SizeType dm_idx,
                                   SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_variance(
        dm_idx, boxcar_width, m_impl->m_cfg.use_box_smearing);
}
float FDMT::get_effective_sigma(SizeType dm_idx, SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_sigma(dm_idx, boxcar_width,
                                              m_impl->m_cfg.use_box_smearing);
}
std::vector<float>
FDMT::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_variance_grid(
        boxcar_width, m_impl->m_cfg.use_box_smearing);
}
std::vector<float> FDMT::get_effective_sigma_grid(SizeType boxcar_width) const {
    return m_impl->m_plan.get_effective_sigma_grid(
        boxcar_width, m_impl->m_cfg.use_box_smearing);
}

void FDMT::reset_history() noexcept { m_impl->m_engine->reset_history(); }
SizeType FDMT::history_state_size() const noexcept {
    return m_impl->m_engine->history_state_size();
}
void FDMT::save_history(std::span<float> out) const {
    m_impl->m_engine->save_history(out);
}
void FDMT::load_history(std::span<const float> in) {
    m_impl->m_engine->load_history(in);
}
void FDMT::save_history(DeviceSpan<float> out, Stream stream) const {
    m_impl->check_device(out, "FDMT::save_history");
    m_impl->m_engine->save_history(out, stream);
}
void FDMT::load_history(DeviceSpan<const float> in, Stream stream) {
    m_impl->check_device(in, "FDMT::load_history");
    m_impl->m_engine->load_history(in, stream);
}

namespace {

// Runs one block and compacts each beam's leading get_dmt_size() samples
// (discarding the buffer_size-sized scratch tail) into a contiguous
// (nbeams, dmt_size) result.
std::tuple<std::vector<float>, plans::FDMTPlan>
run_compute_fdmt(FDMT& fdmt, std::span<const float> waterfall) {
    const plans::FDMTPlan& fdmt_plan = fdmt.get_plan();
    const auto nbeams                = fdmt.get_nbeams();
    const auto buffer_size           = fdmt_plan.get_buffer_size();
    std::vector<float> dmt(nbeams * buffer_size, 0.0F);
    fdmt.execute(waterfall, dmt);
    const auto dmt_size = fdmt_plan.get_dmt_size();
    std::vector<float> dmt_out(nbeams * dmt_size);
    for (SizeType b = 0; b < nbeams; ++b) {
        std::copy_n(dmt.data() + (b * buffer_size), dmt_size,
                    dmt_out.data() + (b * dmt_size));
    }
    return std::make_tuple(std::move(dmt_out), fdmt_plan);
}

} // namespace

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
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
             SizeType nbeams) {
    FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
              use_box_smearing, mode, exec, nbeams);
    return run_compute_fdmt(fdmt, waterfall);
}

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<IndexType>& dt_grid,
             bool use_box_smearing,
             std::string_view mode,
             Exec exec,
             SizeType nbeams) {
    FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_grid, use_box_smearing,
              mode, exec, nbeams);
    return run_compute_fdmt(fdmt, waterfall);
}

std::tuple<std::vector<float>, plans::FDMTPlan>
compute_fdmt(std::span<const float> waterfall,
             float f_min,
             float f_max,
             SizeType nchans,
             SizeType nsamps,
             float tsamp,
             const std::vector<float>& dm_grid,
             bool use_box_smearing,
             std::string_view mode,
             Exec exec,
             SizeType nbeams) {
    FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dm_grid, use_box_smearing,
              mode, exec, nbeams);
    return run_compute_fdmt(fdmt, waterfall);
}

void add_frb_track(std::span<float> waterfall,
                   const plans::FDMTPlan& plan,
                   SizeType dm_idx,
                   float amplitude,
                   IndexType toffset,
                   SizeType width) {
    if (width == 0) {
        throw std::invalid_argument(
            "add_frb_track: width must be greater than 0");
    }
    const auto nchans = plan.get_nchans();
    const auto nsamps = plan.get_nsamps();
    if (waterfall.size() != nchans * nsamps) {
        throw std::invalid_argument(std::format(
            "add_frb_track: Invalid size of waterfall. Expected {}, got {}",
            nchans * nsamps, waterfall.size()));
    }
    // trace_dm() throws std::out_of_range if dm_idx is invalid.
    const auto shifts = plan.trace_dm(dm_idx);

    for (SizeType c = 0; c < nchans; ++c) {
        const auto start = toffset + shifts[c];
        const auto end   = start + static_cast<IndexType>(width);
        if (start < 0 || end > static_cast<IndexType>(nsamps)) {
            throw std::out_of_range(std::format(
                "add_frb_track: injected pulse for channel {} falls outside "
                "the waterfall (start={}, end={}, nsamps={}); choose a "
                "smaller |toffset| or a dm_idx with less total delay",
                c, start, end, nsamps));
        }
        float* __restrict__ row = waterfall.data() + (c * nsamps);
        for (IndexType t = start; t < end; ++t) {
            row[static_cast<SizeType>(t)] += amplitude;
        }
    }
}

} // namespace dmt::algorithms
