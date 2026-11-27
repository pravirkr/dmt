#pragma once

// Private engine interfaces: one abstract class per algorithm, implemented
// once per backend (lib/cpu/*_cpu.cpp, lib/cuda/*_cuda.cu). The public classes
// in include/dmt/algorithms/ own the plan and one engine, validate the
// backend and forward to it; nothing here includes a GPU runtime header.
//
// Device-memory entry points have throwing defaults, so a host-only engine
// (the CPU) implements only the host half.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_search.hpp"

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

// GPU host-memory D2H staging parameters (shared by ChunkedStager and the
// memory estimators so the threshold and pinned footprint cannot drift).
inline constexpr SizeType kHostStagingChunkBytes = SizeType{8} << 20;
inline constexpr int kHostStagingMaxWorkers      = 4;

[[nodiscard]] constexpr int host_staging_workers(int nthreads) noexcept {
    return std::clamp(nthreads, 1, kHostStagingMaxWorkers);
}

[[nodiscard]] constexpr SizeType
host_staging_pinned_bytes(SizeType output_bytes, int nthreads) noexcept {
    const SizeType need =
        SizeType{2} * static_cast<SizeType>(host_staging_workers(nthreads)) *
        kHostStagingChunkBytes;
    return output_bytes >= need ? need : 0;
}

// ---------------------------------------------------------------------------
// FDMT
// ---------------------------------------------------------------------------

/// Transform path of an FDMT engine (private: chosen automatically by
/// default, or pinned by the FDMTBoxcar search list and internal tests).
enum class FDMTPath : uint8_t {
    kAuto,     ///< choose the fastest path that fits max_memory_bytes
    kTree,     ///< plain tree (CPU: no skewed window; GPU: no direct/skewed)
    kDirect,   ///< GPU unskewed direct levels / top (CPU: alias of kTree)
    kSkewed,   ///< skewed-window path (CPU: FDMTSkewCpu; GPU: skewed direct /
               ///< windowed top)
    kWindowed, ///< reported by GPU subband search with windowed top
};

[[nodiscard]] constexpr bool is_skewed_path(FDMTPath path) noexcept {
    return path == FDMTPath::kSkewed || path == FDMTPath::kWindowed;
}

struct FDMTEngineConfig {
    bool use_box_smearing{true};
    FDMTMode mode{FDMTMode::kValid};
    SizeType nbeams{1};
    SizeType fuse_levels{algorithms::kFDMTAutoFuse};
    bool int_tree{true};
    Exec exec{};
    SizeType chunk_nsamps{0};           ///< 0: automatic (see fdmt_chunk.hpp)
    SizeType max_fuse_levels{SIZE_MAX}; ///< Private search boundary.
    /// The search reads levels below the root (subband search), so they must
    /// be materialised (no direct top, see fdmt_direct.hpp).
    bool search_upper_levels{false};
    SizeType search_nbits{
        32}; ///< Exact dyadic integer search; ordinary FDMT unchanged.
    /// Memory budget of everything the engine allocates
    /// (FDMTMemoryUsage::total(); see algorithms::kFDMTDefaultMaxMemory).
    SizeType max_memory_bytes{algorithms::kFDMTDefaultMaxMemory};
    /// Private: requested transform path (kAuto selects by budget).
    FDMTPath path{FDMTPath::kAuto};
    /// Private (CPU, tests): leaf level of a skewed engine when path ==
    /// kSkewed with an explicit chunk (0: automatic / fastest fitting leaf).
    SizeType skew_leaf{0};
    /// Private (GPU): the engine only runs execute_search() (FDMTBoxcar),
    /// which never writes a full-block root plane, so the root output is
    /// exempt from the 32-bit extent check; root-writing calls then throw.
    bool search_only{false};
    /// Private: build the engine's tables and size every working buffer
    /// (get_memory_usage()), but allocate the buffers only on allocate().
    /// Memory estimates and budget searches build engines this way.
    bool defer_allocation{false};
};

class FDMTEngine {
public:
    FDMTEngine()                             = default;
    virtual ~FDMTEngine()                    = default;
    FDMTEngine(const FDMTEngine&)            = delete;
    FDMTEngine& operator=(const FDMTEngine&) = delete;
    FDMTEngine(FDMTEngine&&)                 = delete;
    FDMTEngine& operator=(FDMTEngine&&)      = delete;

    // Search consumes chunks without allocating a root output plane.
    virtual void execute_search(std::span<const float> /*unused*/,
                                FDMTSearchSink& /*unused*/) {
        throw_no_device_memory("FDMT search host execution", backend());
    }
    virtual void execute_search(DeviceSpan<const float> /*unused*/,
                                FDMTSearchSink& /*unused*/,
                                Stream /*unused*/) {
        throw_no_device_memory("FDMT search device execution", backend());
    }

    virtual void execute_search(std::span<const uint8_t> /*unused*/,
                                SizeType /*unused*/,
                                FDMTSearchSink& /*unused*/) {
        throw_no_device_memory("FDMT packed search host execution", backend());
    }
    virtual void execute_search(DeviceSpan<const uint8_t> /*unused*/,
                                SizeType /*unused*/,
                                FDMTSearchSink& /*unused*/,
                                Stream /*unused*/) {
        throw_no_device_memory("FDMT packed search device execution",
                               backend());
    }
    /// Device search only: level-0 rows (beam * nchans + channel) whose
    /// byte in `keep` is 0 read as zero in later execute_search() calls
    /// (device input, and host input staged per chunk); nullptr reads
    /// every row. The bytes are read by the
    /// queued kernels, so they must stay valid until those complete.
    /// Returns false (and ignores `keep`) where unsupported.
    /// `fill`: the code packed rows flagged 2 read (FDMTInputPacked).
    virtual bool set_search_keep(const uint8_t* /*keep*/, float /*fill*/ = 0) {
        return false;
    }
    /// Host search of one all-zero input block of the search bit depth
    /// (FDMTBoxcar::drain() padding) without reading or staging one, where
    /// search_zero_supported(); the result equals execute_search() on such a
    /// block.
    [[nodiscard]] virtual bool search_zero_supported() const noexcept {
        return false;
    }
    virtual void execute_search_zero(FDMTSearchSink& /*unused*/) {
        throw std::logic_error("FDMT search: no zero-block search here");
    }
    /// Short final block (FDMT::execute_tail, FDMTBoxcar::drain(tail, n_tail)):
    /// valid samples per channel row of the next block (SIZE_MAX: all nsamps).
    virtual void set_valid_nsamps(SizeType /*n_valid*/) noexcept {}

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

    virtual void reset_search_history(Stream /*unused*/) { reset_history(); }
    virtual void reset_history() noexcept                              = 0;
    [[nodiscard]] virtual SizeType history_state_size() const noexcept = 0;

    [[nodiscard]] virtual SizeType get_fuse_levels() const noexcept  = 0;
    [[nodiscard]] virtual SizeType get_chunk_nsamps() const noexcept = 0;
    [[nodiscard]] virtual algorithms::FDMTMemoryUsage
    get_memory_usage() const noexcept = 0;
    /// Allocates the working buffers of an engine built with
    /// FDMTEngineConfig::defer_allocation (a no-op once allocated). The
    /// allocation equals get_memory_usage(), checked here.
    virtual void allocate()                           = 0;
    [[nodiscard]] virtual std::string summary() const = 0;
    /// The transform path this engine built.
    [[nodiscard]] virtual FDMTPath path() const noexcept = 0;
    /// The search's transform path when it differs from the plain tree
    /// (GPU direct levels and direct top), e.g. "direct levels 0-3, direct
    /// top 5 -> 9"; empty otherwise.
    [[nodiscard]] virtual std::string search_path() const { return {}; }
    /// Search latency of execute_search(): input samples that a stream
    /// sample waits for before every searched row has emitted it (the
    /// largest row lag of a subband search on the skewed paths; 0 otherwise).
    [[nodiscard]] virtual uint64_t search_latency() const noexcept { return 0; }

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

/// Outcome of building an FDMT engine within cfg.max_memory_bytes.
/// On success, `engine` is non-null. On budget overflow, `engine` is null,
/// `needed` is the smallest total bytes among the tried configurations
/// (SIZE_MAX if none could be built), and `error` describes the failure.
/// Non-budget errors (invalid parameters, 32-bit extent limits) still throw.
struct FDMTEngineAttempt {
    std::unique_ptr<FDMTEngine> engine;
    SizeType needed{0};
    std::string error;
};

// The plan outlives the engine (both are owned by the FDMT facade).
FDMTEngineAttempt try_make_fdmt_cpu(const plans::FDMTPlan& plan,
                                    const FDMTEngineConfig& cfg);
FDMTEngineAttempt try_make_fdmt_gpu(const plans::FDMTPlan& plan,
                                    const FDMTEngineConfig& cfg);
std::unique_ptr<FDMTEngine> make_fdmt_cpu(const plans::FDMTPlan& plan,
                                          const FDMTEngineConfig& cfg);
std::unique_ptr<FDMTEngine> make_fdmt_gpu(const plans::FDMTPlan& plan,
                                          const FDMTEngineConfig& cfg);
/// GPU: the automatic chunk whose two state halves fit a fixed 96 MiB cache
/// target, the L40S's L2 (fdmt_cache_chunk()) for this plan and config; 0 when
/// none fits.
SizeType fdmt_gpu_cache_chunk(const plans::FDMTPlan& plan,
                              const FDMTEngineConfig& cfg);

// ---------------------------------------------------------------------------
// DDMT
// ---------------------------------------------------------------------------

struct DDMTEngineConfig {
    SizeType nbeams{1};
    Exec exec{};
    // Estimate-only engine: buffers are sized, never allocated (estimate()).
    bool defer_allocation{false};
    // SDMT GPU: run the shared-sum kernel whenever its programs fit, even
    // where the DDMT kernel is estimated faster (for small-plan tests).
    bool force_shared_sums{false};
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
    virtual void set_gulp_size(SizeType gulp_size)                      = 0;
    [[nodiscard]] virtual SizeType get_gulp_size() const noexcept       = 0;
    [[nodiscard]] virtual DDMTMemoryUsage memory_usage() const noexcept = 0;
    // Estimate-only engines: what calls of `nsamps` samples hold.
    [[nodiscard]] virtual DDMTMemoryEstimate estimate(SizeType nsamps) = 0;

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
// ---------------------------------------------------------------------------
// DDMTFFT
// ---------------------------------------------------------------------------

struct DDMTFFTEngineConfig {
    SizeType nbeams{1};
    Exec exec{};
    DDMTFFTOptions options{}; // method resolved by the facade (never kAuto)
    // Longest overlap-save transform length (0: default budget).
    SizeType max_segment{0};
    // Estimate-only engine: buffers are sized, never allocated (estimate()).
    bool defer_allocation{false};
};

class DDMTFFTEngine {
public:
    DDMTFFTEngine()                                = default;
    virtual ~DDMTFFTEngine()                       = default;
    DDMTFFTEngine(const DDMTFFTEngine&)            = delete;
    DDMTFFTEngine& operator=(const DDMTFFTEngine&) = delete;
    DDMTFFTEngine(DDMTFFTEngine&&)                 = delete;
    DDMTFFTEngine& operator=(DDMTFFTEngine&&)      = delete;

    // Host memory: every backend.
    virtual void execute(std::span<const float> waterfall,
                         std::span<float> dmt)            = 0;
    virtual void execute(std::span<const uint8_t> waterfall_packed,
                         SizeType nsamps,
                         std::span<float> dmt)            = 0;
    virtual void execute_time_major(std::span<const uint8_t> filterbank_packed,
                                    SizeType nsamps,
                                    std::span<float> dmt) = 0;
    virtual void save_history(std::span<float> out) const = 0;
    virtual void load_history(std::span<const float> in)  = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const float> waterfall,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void execute(DeviceSpan<const uint8_t> waterfall_packed,
                         SizeType nsamps,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void save_history(DeviceSpan<float> out, Stream stream) const;
    virtual void load_history(DeviceSpan<const float> in, Stream stream);

    [[nodiscard]] virtual SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept            = 0;
    virtual void reset_history() noexcept                              = 0;
    [[nodiscard]] virtual SizeType history_state_size() const noexcept = 0;
    [[nodiscard]] virtual SizeType max_delay() const noexcept          = 0;
    /// "nufft", "piecewise_nufft" or "brute": how the grid's trials are
    /// summed, the same on every backend (a GPU engine transforms a run
    /// too long for shared memory as equal NUFFT sub-runs).
    [[nodiscard]] virtual std::string_view method_used() const noexcept    = 0;
    [[nodiscard]] virtual DDMTFFTMemoryUsage memory_usage() const noexcept = 0;
    // Estimate-only engines: sizes what a cold stream of calls of `nsamps`
    // holds once warm, replaying the calls' buffer requests.
    [[nodiscard]] virtual DDMTFFTMemoryEstimate estimate(SizeType nsamps) = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

/// method_used() of an engine running `nruns` NUFFT runs plus `nbrute`
/// brute-force trials.
[[nodiscard]] inline std::string_view
ddmt_fft_method_used(SizeType nruns, SizeType nbrute) noexcept {
    if (nruns == 0) {
        return "brute";
    }
    return (nruns == 1 && nbrute == 0) ? "nufft" : "piecewise_nufft";
}

std::unique_ptr<DDMTFFTEngine>
make_ddmt_fft_cpu(const plans::DDMTPlan& plan, const DDMTFFTEngineConfig& cfg);
std::unique_ptr<DDMTFFTEngine>
make_ddmt_fft_gpu(const plans::DDMTPlan& plan, const DDMTFFTEngineConfig& cfg);

// ---------------------------------------------------------------------------
// FDMT-FFT
// ---------------------------------------------------------------------------

struct FDMTFFTEngineConfig {
    bool use_box_smearing{true};
    FDMTMode mode{FDMTMode::kValid};
    SizeType nbeams{1};
    Exec exec{};
    // Fractional merge delays (fdmt_fft_common.hpp); false only for
    // FDMT-equivalence tests.
    bool fractional_delays{true};
    // Per channel, 1 = keep (empty: keep all); killed channels read as zero.
    std::vector<uint8_t> kill_mask;
    // Longest overlap-save transform execute() may use (0: no cap). The
    // FDMTFFTBoxcar memory budget sets it.
    SizeType max_segment{0};
    // The engine drives a boxcar search (execute_search()): allocate the
    // search's row buffer at construction, so memory_usage() is complete
    // before the first block (the budget compares against it).
    bool search{false};
    // Size every buffer (memory_usage() reports them) but allocate them
    // only on allocate(): the memory estimate and the budget search.
    bool defer_allocation{false};
};

class FDMTFFTEngine {
public:
    FDMTFFTEngine()                                = default;
    virtual ~FDMTFFTEngine()                       = default;
    FDMTFFTEngine(const FDMTFFTEngine&)            = delete;
    FDMTFFTEngine& operator=(const FDMTFFTEngine&) = delete;
    FDMTFFTEngine(FDMTFFTEngine&&)                 = delete;
    FDMTFFTEngine& operator=(FDMTFFTEngine&&)      = delete;

    // Valid-mode search: each overlap-save segment hands its DM rows to
    // `sink` (level niters, materialized) instead of writing a DM-time plane.
    // Rows are in input-stream sample coordinates: output sample o of a
    // block is stream sample block_start - latency + o, and the first
    // `latency` output samples of a stream (before sample 0) are dropped.
    virtual void execute_search(std::span<const float> /*unused*/,
                                FDMTSearchSink& /*unused*/) {
        throw_no_device_memory("FDMTFFT search host execution", backend());
    }
    virtual void execute_search(DeviceSpan<const float> /*unused*/,
                                FDMTSearchSink& /*unused*/,
                                Stream /*unused*/) {
        throw_no_device_memory("FDMTFFT search device execution", backend());
    }
    /// Clears the overlap history and restarts the search stream at 0.
    virtual void reset_search_history(Stream /*unused*/) { reset_history(); }
    /// Engine-owned buffers (output: the caller's DM-time plane).
    [[nodiscard]] virtual FDMTMemoryUsage memory_usage() const noexcept {
        return {};
    }
    /// Allocates what a construction with defer_allocation sized;
    /// memory_usage() is unchanged. A no-op once allocated.
    virtual void allocate() = 0;
    /// Output samples per overlap-save segment (the search chunk).
    [[nodiscard]] virtual SizeType segment_nsamps() const noexcept { return 0; }

    // Host memory: every backend. Packed input is (nbeams, nchans,
    // row bytes) at `nbits` (LSB first), or with `time_major` (nbeams,
    // nsamps, sample bytes), as FDMT's and DDMT's packed layouts.
    virtual void execute(std::span<const float> waterfall,
                         std::span<float> dmt)                           = 0;
    virtual void execute(std::span<const uint8_t> packed,
                         SizeType nbits,
                         bool time_major,
                         std::span<float> dmt)                           = 0;
    virtual void reset(std::span<const float> waterfall,
                       std::span<float> dmt)                             = 0;
    virtual void reset(std::span<const uint8_t> packed,
                       SizeType nbits,
                       std::span<float> dmt)                             = 0;
    [[nodiscard]] virtual std::span<const float> view_level_data() const = 0;
    [[nodiscard]] virtual algorithms::FDMTSubbandView
    view_subband(SizeType subband_idx) const = 0;

    // Device memory: GPU backends.
    virtual void execute(DeviceSpan<const float> waterfall,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void execute(DeviceSpan<const uint8_t> packed,
                         SizeType nbits,
                         DeviceSpan<float> dmt,
                         Stream stream);
    virtual void reset(DeviceSpan<const float> waterfall,
                       DeviceSpan<float> dmt,
                       Stream stream);
    virtual void reset(DeviceSpan<const uint8_t> packed,
                       SizeType nbits,
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
    // Valid-mode overlap history, [beam][chan][overlap] on every backend
    // (empty in the other modes).
    [[nodiscard]] virtual SizeType history_state_size() const noexcept = 0;
    virtual void save_history(std::span<float> out) const              = 0;
    virtual void load_history(std::span<const float> in)               = 0;
    virtual void save_history(DeviceSpan<float> out, Stream stream) const;
    virtual void load_history(DeviceSpan<const float> in, Stream stream);

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

/// GPU coherent-stage implementation (internal; tests and benchmarks pin
/// one, the facade always uses kAuto).
enum class CohFDMTCoherentPath : uint8_t {
    kAuto,  ///< fused when the channel FFT length allows, else cuFFT
    kFused, ///< fused shared-memory kernel (throws if unsupported)
    kCuFFT, ///< gather -> batched cuFFT -> detect through a work buffer
};

struct CohFDMTEngineConfig {
    Exec exec{};
    CohFDMTCoherentPath coherent_path{CohFDMTCoherentPath::kAuto};
    /// cuFFT path work-buffer budget in bytes (0 = 1 GiB).
    SizeType work_bytes{0};
    /// Size every working buffer (and the fine FDMT) without allocating;
    /// allocate() allocates them (CohFDMT::estimate_memory()).
    bool defer_allocation{false};
};

// Stateless: one self-contained block per execute(), one input span per
// subband group (raw bytes, decoded per the plan's BasebandFormat).
class CohFDMTEngine {
public:
    CohFDMTEngine()                                = default;
    virtual ~CohFDMTEngine()                       = default;
    CohFDMTEngine(const CohFDMTEngine&)            = delete;
    CohFDMTEngine& operator=(const CohFDMTEngine&) = delete;
    CohFDMTEngine(CohFDMTEngine&&)                 = delete;
    CohFDMTEngine& operator=(CohFDMTEngine&&)      = delete;

    // Host memory: every backend.
    virtual void execute(std::span<const std::span<const uint8_t>> groups,
                         std::span<float> dmt) = 0;

    // Device memory: GPU backends.
    virtual void execute(std::span<const DeviceSpan<const uint8_t>> groups,
                         DeviceSpan<float> dmt,
                         Stream stream);

    [[nodiscard]] virtual plans::CohFDMTMemoryUsage
    memory_usage() const noexcept = 0;

    /// Allocates what a deferred construction sized (no-op otherwise).
    virtual void allocate() = 0;

protected:
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
};

std::unique_ptr<CohFDMTEngine> make_cfdmt_cpu(const plans::CohFDMTPlan& plan,
                                              const CohFDMTEngineConfig& cfg);
std::unique_ptr<CohFDMTEngine> make_cfdmt_gpu(const plans::CohFDMTPlan& plan,
                                              const CohFDMTEngineConfig& cfg);

} // namespace dmt::algorithms::detail
