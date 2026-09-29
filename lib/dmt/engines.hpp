#pragma once

// Private engine interfaces: one abstract class per algorithm, implemented
// once per backend (lib/cpu/*_cpu.cpp, lib/cuda/*_cuda.cu). The public classes
// in include/dmt/algorithms/ own the plan and one engine, validate the
// backend and forward to it; nothing here includes a GPU runtime header.
//
// Device-memory entry points have throwing defaults, so a host-only engine
// (the CPU) implements only the host half.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

// The backend lib/cuda/ is compiled for: DMT_ENABLE_CUDA (nvcc) or
// DMT_ENABLE_HIP (hip-clang, through dmt/gpu_compat.cuh); DMT_ENABLE_GPU is
// set with either.
#ifdef DMT_ENABLE_HIP
inline constexpr Backend kGPUBackend = Backend::kHIP;
#elif defined(DMT_ENABLE_CUDA)
inline constexpr Backend kGPUBackend = Backend::kCUDA;
#endif

/// Throws std::invalid_argument: `what` needs device memory, which the
/// `backend` engine does not have.
[[noreturn]] void throw_no_device_memory(std::string_view what,
                                         Backend backend);

/// Throws std::invalid_argument naming available_backends(): `algorithm` was
/// asked for a backend this build does not contain.
[[noreturn]] void throw_unavailable(std::string_view algorithm,
                                    Backend backend);

/// Throws std::invalid_argument if a DeviceSpan states a device other than
/// the engine's.
void check_device(const Device& view,
                  Backend backend,
                  int device,
                  std::string_view what);

// ---------------------------------------------------------------------------
// FDMT
// ---------------------------------------------------------------------------

struct FDMTEngineConfig {
    bool use_box_smearing{true};
    FDMTMode mode{FDMTMode::kValid};
    SizeType nbeams{1};
    SizeType fuse_levels{algorithms::kFDMTAutoFuse};
    bool int_tree{true};
    Exec exec{};
};

class FDMTEngine {
public:
    FDMTEngine()                             = default;
    virtual ~FDMTEngine()                    = default;
    FDMTEngine(const FDMTEngine&)            = delete;
    FDMTEngine& operator=(const FDMTEngine&) = delete;
    FDMTEngine(FDMTEngine&&)                 = delete;
    FDMTEngine& operator=(FDMTEngine&&)      = delete;

    // Host memory: every backend.
    virtual void execute(std::span<const float> waterfall,
                         std::span<float> dmt)                           = 0;
    virtual void execute(std::span<const uint8_t> waterfall_packed,
                         SizeType nbits,
                         std::span<float> dmt)                           = 0;
    virtual void reset(std::span<const float> waterfall,
                       std::span<float> dmt)                             = 0;
    virtual void reset(std::span<const uint8_t> waterfall_packed,
                       SizeType nbits,
                       std::span<float> dmt)                             = 0;
    [[nodiscard]] virtual std::span<const float> view_level_data() const = 0;
    [[nodiscard]] virtual algorithms::FDMTSubbandView
    view_subband(SizeType subband_idx) const              = 0;
    virtual void save_history(std::span<float> out) const = 0;
    virtual void load_history(std::span<const float> in)  = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const float> waterfall,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void execute(DeviceSpan<const uint8_t> waterfall_packed,
                         SizeType nbits,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void reset(DeviceSpan<const float> waterfall,
                       DeviceSpan<float> dmt,
                       Stream stream);
    virtual void reset(DeviceSpan<const uint8_t> waterfall_packed,
                       SizeType nbits,
                       DeviceSpan<float> dmt,
                       Stream stream);
    [[nodiscard]] virtual DeviceSpan<const float>
    view_level_data_device() const;
    [[nodiscard]] virtual algorithms::FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const;
    virtual void save_history(DeviceSpan<float> out, Stream stream) const;
    virtual void load_history(DeviceSpan<const float> in, Stream stream);

    // Stepper. `stream` is always empty on the CPU (the facade checks).
    virtual void advance(SizeType levels, Stream stream)                    = 0;
    virtual void advance_until_remaining(SizeType remaining, Stream stream) = 0;
    virtual void finalize(Stream stream)                                    = 0;
    [[nodiscard]] virtual SizeType current_level() const noexcept           = 0;
    [[nodiscard]] virtual SizeType num_subbands() const                     = 0;
    [[nodiscard]] virtual bool is_finished() const noexcept                 = 0;

    virtual void reset_history() noexcept                              = 0;
    [[nodiscard]] virtual SizeType history_state_size() const noexcept = 0;

    [[nodiscard]] virtual SizeType get_fuse_levels() const noexcept = 0;
    [[nodiscard]] virtual algorithms::FDMTMemoryUsage
    get_memory_usage() const noexcept                 = 0;
    [[nodiscard]] virtual std::string summary() const = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

// The plan outlives the engine (both are owned by the FDMT facade).
std::unique_ptr<FDMTEngine> make_fdmt_cpu(const plans::FDMTPlan& plan,
                                          const FDMTEngineConfig& cfg);
std::unique_ptr<FDMTEngine> make_fdmt_gpu(const plans::FDMTPlan& plan,
                                          const FDMTEngineConfig& cfg);

// ---------------------------------------------------------------------------
// DDMT
// ---------------------------------------------------------------------------

struct DDMTEngineConfig {
    SizeType nbeams{1};
    Exec exec{};
};

class DDMTEngine {
public:
    DDMTEngine()                             = default;
    virtual ~DDMTEngine()                    = default;
    DDMTEngine(const DDMTEngine&)            = delete;
    DDMTEngine& operator=(const DDMTEngine&) = delete;
    DDMTEngine(DDMTEngine&&)                 = delete;
    DDMTEngine& operator=(DDMTEngine&&)      = delete;

    // Host memory: every backend.
    virtual void execute(std::span<const float> waterfall,
                         std::span<float> dmt)              = 0;
    virtual void execute(std::span<const uint8_t> waterfall_packed,
                         SizeType nsamps,
                         std::span<int32_t> dmt)            = 0;
    virtual void execute_time_major(std::span<const uint8_t> filterbank_packed,
                                    SizeType nsamps,
                                    std::span<int32_t> dmt) = 0;
    virtual void save_history(std::span<float> out) const   = 0;
    virtual void save_history(std::span<uint8_t> out) const = 0;
    virtual void load_history(std::span<const float> in)    = 0;
    virtual void load_history(std::span<const uint8_t> in)  = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const float> waterfall,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void execute(DeviceSpan<const uint8_t> waterfall_packed,
                         SizeType nsamps,
                         DeviceSpan<int32_t> dmt,
                         Stream stream);
    virtual void execute_time_major(DeviceSpan<const uint8_t> filterbank_packed,
                                    SizeType nsamps,
                                    DeviceSpan<int32_t> dmt,
                                    Stream stream);
    virtual void save_history(DeviceSpan<float> out, Stream stream) const;
    virtual void save_history(DeviceSpan<uint8_t> out, Stream stream) const;
    virtual void load_history(DeviceSpan<const float> in, Stream stream);
    virtual void load_history(DeviceSpan<const uint8_t> in, Stream stream);

    [[nodiscard]] virtual SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept            = 0;
    virtual void reset_history() noexcept                              = 0;
    [[nodiscard]] virtual SizeType history_state_size() const noexcept = 0;

    // Host-path chunk length in input samples (GPU; stored but unused on
    // the CPU). 0 restores the default.
    virtual void set_gulp_size(SizeType gulp_size)                = 0;
    [[nodiscard]] virtual SizeType get_gulp_size() const noexcept = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

std::unique_ptr<DDMTEngine> make_ddmt_cpu(const plans::DDMTPlan& plan,
                                          const DDMTEngineConfig& cfg);
std::unique_ptr<DDMTEngine> make_ddmt_gpu(const plans::DDMTPlan& plan,
                                          const DDMTEngineConfig& cfg);
// SDMT: exact shared-partial-sum engines (same interface as DDMT).
std::unique_ptr<DDMTEngine> make_sdmt_cpu(const plans::DDMTPlan& plan,
                                          const DDMTEngineConfig& cfg);
std::unique_ptr<DDMTEngine> make_sdmt_gpu(const plans::DDMTPlan& plan,
                                          const DDMTEngineConfig& cfg);
// Testing hook: while set, SDMT GPU engines constructed afterwards run the
// shared-sum kernel whenever its programs fit, even where the DDMT kernel is
// estimated to be faster (so tests cover it on small plans).
void set_sdmt_gpu_always_shared(bool on) noexcept;
[[nodiscard]] bool sdmt_gpu_always_shared() noexcept;

// ---------------------------------------------------------------------------
// FDMT-FFT
// ---------------------------------------------------------------------------

struct FDMTFFTEngineConfig {
    bool use_box_smearing{true};
    FDMTMode mode{FDMTMode::kValid};
    SizeType nbeams{1};
    Exec exec{};
};

class FDMTFFTEngine {
public:
    FDMTFFTEngine()                                = default;
    virtual ~FDMTFFTEngine()                       = default;
    FDMTFFTEngine(const FDMTFFTEngine&)            = delete;
    FDMTFFTEngine& operator=(const FDMTFFTEngine&) = delete;
    FDMTFFTEngine(FDMTFFTEngine&&)                 = delete;
    FDMTFFTEngine& operator=(FDMTFFTEngine&&)      = delete;

    // Host memory: every backend.
    virtual void execute(std::span<const float> waterfall,
                         std::span<float> dmt)                           = 0;
    virtual void reset(std::span<const float> waterfall,
                       std::span<float> dmt)                             = 0;
    [[nodiscard]] virtual std::span<const float> view_level_data() const = 0;
    [[nodiscard]] virtual algorithms::FDMTSubbandView
    view_subband(SizeType subband_idx) const = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const float> waterfall,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void reset(DeviceSpan<const float> waterfall,
                       DeviceSpan<float> dmt,
                       Stream stream);
    [[nodiscard]] virtual DeviceSpan<const float>
    view_level_data_device() const;
    [[nodiscard]] virtual algorithms::FDMTSubbandDeviceView
    view_subband_device(SizeType subband_idx) const;

    // Stepper. `stream` is always empty on the CPU (the facade checks).
    virtual void advance(SizeType levels, Stream stream)                    = 0;
    virtual void advance_until_remaining(SizeType remaining, Stream stream) = 0;
    virtual void finalize(Stream stream)                                    = 0;
    [[nodiscard]] virtual SizeType current_level() const noexcept           = 0;
    [[nodiscard]] virtual SizeType num_subbands() const                     = 0;
    [[nodiscard]] virtual bool is_finished() const noexcept                 = 0;

    virtual void reset_history() noexcept = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

std::unique_ptr<FDMTFFTEngine>
make_fdmt_fft_cpu(const plans::FDMTPlan& plan, const FDMTFFTEngineConfig& cfg);
std::unique_ptr<FDMTFFTEngine>
make_fdmt_fft_gpu(const plans::FDMTPlan& plan, const FDMTFFTEngineConfig& cfg);

// ---------------------------------------------------------------------------
// CohFDMT
// ---------------------------------------------------------------------------

struct CohFDMTEngineConfig {
    Exec exec{};
};

class CohFDMTEngine {
public:
    CohFDMTEngine()                                = default;
    virtual ~CohFDMTEngine()                       = default;
    CohFDMTEngine(const CohFDMTEngine&)            = delete;
    CohFDMTEngine& operator=(const CohFDMTEngine&) = delete;
    CohFDMTEngine(CohFDMTEngine&&)                 = delete;
    CohFDMTEngine& operator=(CohFDMTEngine&&)      = delete;

    // Host memory: every backend. One overload per supported sample type.
    virtual void execute(std::span<const uint8_t> data_in,
                         std::span<float> dmt) = 0;
    virtual void execute(std::span<const int8_t> data_in,
                         std::span<float> dmt) = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const uint8_t> data_in,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void execute(DeviceSpan<const int8_t> data_in,
                         DeviceSpan<float> dmt,
                         Stream stream);

    virtual void reset_history() noexcept = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

std::unique_ptr<CohFDMTEngine> make_cfdmt_cpu(const plans::CohFDMTPlan& plan,
                                              const CohFDMTEngineConfig& cfg);
std::unique_ptr<CohFDMTEngine> make_cfdmt_gpu(const plans::CohFDMTPlan& plan,
                                              const CohFDMTEngineConfig& cfg);

} // namespace dmt::algorithms::detail
