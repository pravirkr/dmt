#pragma once

#include <cstddef>
#include <mutex>
#include <new>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

#include "dmt/gpu_compat.cuh"

namespace dmt {

// Device-side complex type for shared GPU sources (never in public headers).
using ComplexTypeGPU                     = cuda::std::complex<float>;
template <typename T> using DeviceVector = thrust::device_vector<T>;

} // namespace dmt

namespace dmt::gpu_utils {

constexpr std::string_view fft_error_string(cufftResult error) noexcept {
    switch (error) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    default:
        return "Unknown FFT library error";
    }
}

/** Runtime or FFT errors from the GPU toolchain (CUDA or HIP via gpu_compat).
 */
class GPUException : public std::runtime_error {
public:
    explicit GPUException(
        cudaError_t code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_runtime_error(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_runtime(true),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    explicit GPUException(
        cufftResult code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_fft_error(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_runtime(false),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    [[nodiscard]] constexpr int code() const noexcept { return m_code; }
    [[nodiscard]] constexpr bool is_runtime_error() const noexcept {
        return m_is_runtime;
    }
    [[nodiscard]] constexpr const char* file() const noexcept { return m_file; }
    [[nodiscard]] constexpr uint32_t line() const noexcept { return m_line; }
    [[nodiscard]] constexpr const char* function() const noexcept {
        return m_func;
    }
    // Not constexpr: libstdc++ 11 (the GCC 11 floor) has no constexpr
    // std::string, and nvcc host builds with it reject the conversion.
    [[nodiscard]] std::string_view user_message() const noexcept {
        return m_user_msg;
    }
    [[nodiscard]] std::string error_string() const {
        return m_is_runtime
                   ? cudaGetErrorString(static_cast<cudaError_t>(m_code))
                   : std::string(
                         fft_error_string(static_cast<cufftResult>(m_code)));
    }

private:
    int m_code;
    bool m_is_runtime;
    const char* m_file;
    uint32_t m_line;
    const char* m_func;
    std::string m_user_msg;

    static std::string format_runtime_error(cudaError_t code,
                                            std::string_view user_msg,
                                            const std::source_location& loc) {
        auto base_msg =
            fmt::format("{} runtime error [{}]: {}", DMT_GPU_NAME,
                        static_cast<int>(code), cudaGetErrorString(code));
        return user_msg.empty()
                   ? fmt::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : fmt::format("{} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }

    static std::string format_fft_error(cufftResult code,
                                        std::string_view user_msg,
                                        const std::source_location& loc) {
        auto base_msg =
            fmt::format("{} FFT error [{}]: {}", DMT_GPU_NAME,
                        static_cast<int>(code), fft_error_string(code));
        return user_msg.empty()
                   ? fmt::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : fmt::format("{} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }
};

inline void
check_gpu_call(cudaError_t result,
               std::string_view msg     = "",
               std::source_location loc = std::source_location::current()) {
    if (result != cudaSuccess) {
        throw GPUException(result, msg, loc);
    }
}

inline void
check_gpu_call(cufftResult result,
               std::string_view msg     = "",
               std::source_location loc = std::source_location::current()) {
    if (result != CUFFT_SUCCESS) {
        throw GPUException(result, msg, loc);
    }
}

/// Lets `kernel` launch with up to `bytes` of dynamic shared memory on the
/// current device. The limit is the kernel's, shared by every engine in the
/// process: it is only ever raised, so an engine built later with a smaller
/// need never lowers it under an engine that needs more.
template <typename Kernel>
inline void
raise_dynamic_smem(Kernel kernel,
                   std::size_t bytes,
                   std::string_view msg     = "",
                   std::source_location loc = std::source_location::current()) {
    static std::mutex lock;
    const std::scoped_lock guard(lock);
    cudaFuncAttributes attr{};
    check_gpu_call(cudaFuncGetAttributes(&attr, kernel), msg, loc);
    if (static_cast<std::size_t>(attr.maxDynamicSharedSizeBytes) < bytes) {
        check_gpu_call(cudaFuncSetAttribute(
                           kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(bytes)),
                       msg, loc);
    }
}

inline void check_last_gpu_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw GPUException(error, msg, loc);
    }
}

/// An engine's device buffers did not fit: a std::bad_alloc (Python
/// MemoryError, as before) that says what the configuration needs.
class DeviceAllocationError : public std::bad_alloc {
public:
    explicit DeviceAllocationError(std::string message, std::string cause = {})
        : m_message(std::move(message)),
          m_cause(std::move(cause)) {}
    [[nodiscard]] const char* what() const noexcept override {
        return m_message.c_str();
    }
    /// The failed allocation itself (empty: what() is all there is).
    [[nodiscard]] const std::string& cause() const noexcept { return m_cause; }

private:
    std::string m_message;
    std::string m_cause;
};

/// The device's free and total memory, if it can be read.
struct DeviceMemoryInfo {
    std::size_t free{0};
    std::size_t total{0};
    bool known{false};
};

/// Free device memory now (read only to fail early and for messages; no
/// engine sizes itself from it). Engines read it before allocating
/// (check_free_memory()), so a failure message reports what was free
/// before their own partial allocations.
[[nodiscard]] inline DeviceMemoryInfo device_memory_info() noexcept {
    DeviceMemoryInfo m;
    m.known = cudaMemGetInfo(&m.free, &m.total) == cudaSuccess;
    if (!m.known) {
        (void)cudaGetLastError();
    }
    return m;
}

/// The underlying failure of an allocation exception: a nested engine's
/// DeviceAllocationError contributes its cause, not its whole message.
[[nodiscard]] inline std::string allocation_cause(const std::exception& e) {
    if (const auto* d = dynamic_cast<const DeviceAllocationError*>(&e);
        d != nullptr && !d->cause().empty()) {
        return d->cause();
    }
    return e.what();
}

/// What an engine's memory error tells the user to change: engines with a
/// max_memory_bytes lower it (nsamps is not a memory knob for them); the
/// others take shorter blocks.
inline constexpr std::string_view kAdviceBudget =
    "lower max_memory_bytes or use fewer beams (estimate_memory() gives what "
    "a budget needs)";
inline constexpr std::string_view kAdviceBoxcar =
    "lower max_memory_bytes or use fewer beams (estimate_memory() gives what "
    "a budget needs, FDMTBoxcar::estimate_minimum_memory() the smallest "
    "budget)";
inline constexpr std::string_view kAdviceBlocks =
    "use shorter blocks (nsamps) or fewer beams (estimate_memory() gives the "
    "requirement)";

/// Construction's check of the chosen configuration against the device,
/// before anything is allocated: throws a DeviceAllocationError
/// (std::bad_alloc, Python MemoryError) when `needed` is over the free
/// memory in `before` (device_memory_info()). The check never changes
/// which configuration is chosen (max_memory_bytes alone does); the
/// allocation itself can still fail (fragmentation, another process), and
/// allocation_failure() then reports it.
inline void check_free_memory(std::string_view what,
                              std::size_t needed,
                              const DeviceMemoryInfo& before,
                              std::string_view advice) {
    if (!before.known || needed <= before.free) {
        return;
    }
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    auto cause = fmt::format("{:.2f} GiB needed, {:.2f} GiB free",
                             static_cast<double>(needed) / kGiB,
                             static_cast<double>(before.free) / kGiB);
    throw DeviceAllocationError(
        fmt::format("{}: this configuration needs {:.2f} GiB of device "
                    "memory and only {:.2f} GiB of {:.2f} GiB is free on "
                    "the device (nothing allocated); {}",
                    what, static_cast<double>(needed) / kGiB,
                    static_cast<double>(before.free) / kGiB,
                    static_cast<double>(before.total) / kGiB, advice),
        std::move(cause));
}

/// The error for an engine whose planned device buffers failed to
/// allocate: what it needs and what the device had free before it
/// allocated (`before`, from device_memory_info()), and `advice`.
[[nodiscard]] inline DeviceAllocationError
allocation_failure(std::string_view what,
                   std::size_t needed,
                   const DeviceMemoryInfo& before,
                   const std::exception& e,
                   std::string_view advice) {
    (void)cudaGetLastError(); // clear the failed allocation's error
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    auto cause            = allocation_cause(e);
    auto message          = fmt::format(
        "{}: this configuration needs {:.2f} GiB of device memory{}; {}: {}",
        what, static_cast<double>(needed) / kGiB,
        before.known ? fmt::format(" and {:.2f} GiB of {:.2f} GiB was free",
                                   static_cast<double>(before.free) / kGiB,
                                   static_cast<double>(before.total) / kGiB)
                     : std::string{},
        advice, cause);
    return DeviceAllocationError(std::move(message), std::move(cause));
}

/// check_gpu_call() for an allocation of `bytes`: out of memory throws a
/// DeviceAllocationError (std::bad_alloc, Python MemoryError) naming the
/// size; other errors throw GPUException as usual.
inline void
check_allocation(cudaError_t result,
                 std::size_t bytes,
                 std::string_view msg,
                 std::source_location loc = std::source_location::current()) {
    if (result == cudaErrorMemoryAllocation) {
        (void)cudaGetLastError();
        auto cause = fmt::format("{} ({:.3f} GiB): out of memory", msg,
                                 static_cast<double>(bytes) /
                                     (1024.0 * 1024.0 * 1024.0));
        throw DeviceAllocationError(
            cause + "; estimate_memory() gives an engine's requirement", cause);
    }
    check_gpu_call(result, msg, loc);
}

[[nodiscard]] inline const cudaDeviceProp& device_properties(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, cudaDeviceProp> cache;
    const std::lock_guard lock(mutex);
    if (const auto it = cache.find(device); it != cache.end()) {
        return it->second;
    }
    cudaDeviceProp props{};
    check_gpu_call(cudaGetDeviceProperties(&props, device),
                   "Failed to get device properties");
    return cache.emplace(device, props).first->second;
}

inline void check_kernel_launch_params(
    dim3 grid,
    dim3 block,
    const std::source_location loc = std::source_location::current()) {
    int device;
    check_gpu_call(cudaGetDevice(&device), "Failed to get device", loc);
    const cudaDeviceProp& props = device_properties(device);

    auto check_limit = [&](auto val, auto max, std::string_view dim) {
        if (val > static_cast<unsigned>(max)) {
            throw std::runtime_error(
                fmt::format("{} dimension {} exceeds device limit {} at {}:{}",
                            dim, val, max, loc.file_name(), loc.line()));
        }
    };

    check_limit(block.x, props.maxThreadsDim[0], "Block X");
    check_limit(block.y, props.maxThreadsDim[1], "Block Y");
    check_limit(block.z, props.maxThreadsDim[2], "Block Z");
    check_limit(block.x * block.y * block.z, props.maxThreadsPerBlock,
                "Total threads");
    check_limit(grid.x, props.maxGridSize[0], "Grid X");
    check_limit(grid.y, props.maxGridSize[1], "Grid Y");
    check_limit(grid.z, props.maxGridSize[2], "Grid Z");
}

class DeviceWorkFence {
public:
    DeviceWorkFence() = default;
    ~DeviceWorkFence() {
        if (m_event != nullptr) {
            cudaEventDestroy(m_event);
        }
    }
    DeviceWorkFence(const DeviceWorkFence&)            = delete;
    DeviceWorkFence& operator=(const DeviceWorkFence&) = delete;
    DeviceWorkFence(DeviceWorkFence&&)                 = delete;
    DeviceWorkFence& operator=(DeviceWorkFence&&)      = delete;

    void mark(cudaStream_t stream) {
        if (m_event == nullptr) {
            check_gpu_call(
                cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming),
                "DeviceWorkFence: event creation failed");
        }
        check_gpu_call(cudaEventRecord(m_event, stream),
                       "DeviceWorkFence: event record failed");
        m_pending = true;
    }

    /// Makes later work on @p stream wait for the marked work (no host
    /// blocking).
    void order(cudaStream_t stream) const {
        if (m_pending) {
            check_gpu_call(cudaStreamWaitEvent(stream, m_event, 0),
                           "DeviceWorkFence: stream wait failed");
        }
    }

    void wait() {
        if (m_pending) {
            check_gpu_call(cudaEventSynchronize(m_event),
                           "DeviceWorkFence: waiting for device work failed");
            m_pending = false;
        }
    }

private:
    cudaEvent_t m_event{nullptr};
    bool m_pending{false};
};

inline void set_device(int device_id) {
    int device_count;
    check_gpu_call(cudaGetDeviceCount(&device_count),
                   "Failed to get device count");
    if (device_id < 0 || device_id >= device_count) {
        throw GPUException(
            cudaErrorInvalidDevice,
            fmt::format("Invalid device_id: {}. Must be between 0 and {}",
                        device_id, device_count - 1));
    }
    check_gpu_call(cudaSetDevice(device_id),
                   fmt::format("Failed to set device {}", device_id));
}
} // namespace dmt::gpu_utils
