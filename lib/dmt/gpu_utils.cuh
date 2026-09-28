#pragma once

#include <format>
#include <mutex>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include "dmt/gpu_compat.cuh"
#include <thrust/complex.h>
#include <thrust/device_vector.h>

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

/** Runtime or FFT errors from the GPU toolchain (CUDA or HIP via gpu_compat). */
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
    [[nodiscard]] constexpr std::string_view user_message() const noexcept {
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
        auto base_msg = std::format("{} runtime error [{}]: {}", DMT_GPU_NAME,
                                    static_cast<int>(code),
                                    cudaGetErrorString(code));
        return user_msg.empty()
                   ? std::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("{} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }

    static std::string format_fft_error(cufftResult code,
                                        std::string_view user_msg,
                                        const std::source_location& loc) {
        auto base_msg = std::format("{} FFT error [{}]: {}", DMT_GPU_NAME,
                                    static_cast<int>(code),
                                    fft_error_string(code));
        return user_msg.empty()
                   ? std::format("{} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("{} in {} ({}:{}): {}", base_msg,
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

inline void check_last_gpu_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw GPUException(error, msg, loc);
    }
}

inline void check_gpu_sync_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    check_gpu_call(cudaDeviceSynchronize(), "Synchronization failed", loc);
    check_last_gpu_error(msg, loc);
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
                std::format("{} dimension {} exceeds device limit {} at {}:{}",
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

[[nodiscard]] inline std::string get_device_info() noexcept {
    int device;
    cudaDeviceProp props{};
    if (auto err = cudaGetDevice(&device); err != cudaSuccess) {
        return std::format("Failed to get device: {}", cudaGetErrorString(err));
    }
    if (auto err = cudaGetDeviceProperties(&props, device);
        err != cudaSuccess) {
        return std::format("Failed to get properties: {}",
                           cudaGetErrorString(err));
    }

    return std::format("{} device {}: {} (CC {}.{}), Mem: {} MiB", DMT_GPU_NAME,
                       device, props.name, props.major, props.minor,
                       props.totalGlobalMem >> 20);
}

inline void set_device(int device_id) {
    int device_count;
    check_gpu_call(cudaGetDeviceCount(&device_count),
                   "Failed to get device count");
    if (device_id < 0 || device_id >= device_count) {
        throw GPUException(
            cudaErrorInvalidDevice,
            std::format("Invalid device_id: {}. Must be between 0 and {}",
                        device_id, device_count - 1));
    }
    check_gpu_call(cudaSetDevice(device_id),
                   std::format("Failed to set device {}", device_id));
}
} // namespace dmt::gpu_utils
