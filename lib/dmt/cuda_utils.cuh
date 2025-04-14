#pragma once

#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>
#include <cufft.h>

// Formatter specialization for cudaError_t
template <>
struct std::formatter<cudaError_t> : std::formatter<int> {
    constexpr auto format(cudaError_t error, format_context& ctx) const {
        return std::formatter<int>::format(static_cast<int>(error), ctx);
    }
};

// Formatter specialization for cufftResult
template <>
struct std::formatter<cufftResult> : std::formatter<int> {
    constexpr auto format(cufftResult error, format_context& ctx) const {
        return std::formatter<int>::format(static_cast<int>(error), ctx);
    }
};

namespace dmt::error {

/**
 * @brief Custom exception class for CUDA errors.
 */
class CudaException : public std::runtime_error {
public:
    // Constructor for cudaError_t
    explicit CudaException(
        cudaError_t code,
        std::string_view user_msg       = "",
        const std::source_location& loc = std::source_location::current())
        : std::runtime_error(format_what(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_cuda(true),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    // Constructor for cufftResult
    explicit CudaException(
        cufftResult code,
        std::string_view user_msg = "",
        std::source_location loc  = std::source_location::current())
        : std::runtime_error(format_what(code, user_msg, loc)),
          m_code(static_cast<int>(code)),
          m_is_cuda(false),
          m_file(loc.file_name()),
          m_line(loc.line()),
          m_func(loc.function_name()),
          m_user_msg(user_msg) {}

    [[nodiscard]] constexpr int code() const noexcept { return m_code; }
    [[nodiscard]] constexpr bool is_cuda_error() const noexcept {
        return m_is_cuda;
    }
    [[nodiscard]] constexpr const char* file() const noexcept { return m_file; }
    [[nodiscard]] constexpr std::uint32_t line() const noexcept {
        return m_line;
    }
    [[nodiscard]] constexpr const char* function() const noexcept {
        return m_func;
    }
    [[nodiscard]] constexpr std::string_view user_message() const noexcept {
        return m_user_msg;
    }
    [[nodiscard]] std::string error_string() const {
        if (m_is_cuda) {
            return cudaGetErrorString(static_cast<cudaError_t>(m_code));
        }
        // cuFFT doesn't provide a standard string function, so we map manually
        switch (static_cast<cufftResult>(m_code)) {
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
            return "Unknown cuFFT error";
        }
    }

private:
    int m_code;
    bool m_is_cuda;
    const char* m_file;
    std::uint32_t m_line;
    const char* m_func;
    std::string m_user_msg;

    // Helper to format the what() message for the base class
    static std::string format_what(cudaError_t code,
                                   std::string_view user_msg,
                                   const std::source_location& loc) {
        auto base_msg = std::format("[{}] {}", code, cudaGetErrorString(code));
        return user_msg.empty()
                   ? std::format("CUDA Error {} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("CUDA Error {} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }

    static std::string format_what(cufftResult code,
                                   std::string_view user_msg,
                                   const std::source_location& loc) {
        auto base_msg = std::format("[{}] ", code);
        switch (code) {
        case CUFFT_SUCCESS:
            base_msg += "CUFFT_SUCCESS";
            break;
        case CUFFT_INVALID_PLAN:
            base_msg += "CUFFT_INVALID_PLAN";
            break;
        case CUFFT_ALLOC_FAILED:
            base_msg += "CUFFT_ALLOC_FAILED";
            break;
        case CUFFT_INVALID_TYPE:
            base_msg += "CUFFT_INVALID_TYPE";
            break;
        case CUFFT_INVALID_VALUE:
            base_msg += "CUFFT_INVALID_VALUE";
            break;
        case CUFFT_INTERNAL_ERROR:
            base_msg += "CUFFT_INTERNAL_ERROR";
            break;
        case CUFFT_EXEC_FAILED:
            base_msg += "CUFFT_EXEC_FAILED";
            break;
        case CUFFT_SETUP_FAILED:
            base_msg += "CUFFT_SETUP_FAILED";
            break;
        case CUFFT_INVALID_SIZE:
            base_msg += "CUFFT_INVALID_SIZE";
            break;
        case CUFFT_UNALIGNED_DATA:
            base_msg += "CUFFT_UNALIGNED_DATA";
            break;
        default:
            base_msg += "Unknown cuFFT error";
            break;
        }
        return user_msg.empty()
                   ? std::format("cuFFT Error {} in {} ({}:{})", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line())
                   : std::format("cuFFT Error {} in {} ({}:{}): {}", base_msg,
                                 loc.function_name(), loc.file_name(),
                                 loc.line(), user_msg);
    }
};

// Generic error checking function
inline void
check_cuda_call(cudaError_t result,
                std::string_view msg     = "",
                std::source_location loc = std::source_location::current()) {
    if (result != cudaSuccess) {
        throw CudaException(result, msg, loc);
    }
}

inline void
check_cuda_call(cufftResult result,
                std::string_view msg     = "",
                std::source_location loc = std::source_location::current()) {
    if (result != CUFFT_SUCCESS) {
        throw CudaException(result, msg, loc);
    }
}

// Specialized checks
inline void check_last_cuda_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(error, msg, loc);
    }
}

inline void check_cuda_sync_error(
    std::string_view msg     = "",
    std::source_location loc = std::source_location::current()) {
    check_cuda_call(cudaDeviceSynchronize(), "Synchronization failed", loc);
    check_last_cuda_error(msg, loc);
}

/**
 * @brief Checks kernel launch parameters against device limits before
 * launching. Throws std::runtime_error on failure.
 * @param grid Grid dimensions.
 * @param block Block dimensions.
 * @param location Source location information (automatically captured).
 * @throws std::runtime_error if dimensions exceed device limits.
 */
inline void check_kernel_launch_params(
    dim3 grid,
    dim3 block,
    const std::source_location loc = std::source_location::current()) {
    int device;
    cudaDeviceProp props{};
    // Get current device, check for errors
    check_cuda_call(cudaGetDevice(&device), "Failed to get device", loc);
    check_cuda_call(cudaGetDeviceProperties(&props, device),
                    "Failed to get device properties", loc);

    auto throw_if_exceeds = [&](auto val, auto max, std::string_view dim) {
        if (val > static_cast<unsigned>(max)) {
            throw std::runtime_error(
                std::format("{} dimension {} exceeds device limit {} at {}:{}",
                            dim, val, max, loc.file_name(), loc.line()));
        }
    };

    throw_if_exceeds(block.x, props.maxThreadsDim[0], "Block X");
    throw_if_exceeds(block.y, props.maxThreadsDim[1], "Block Y");
    throw_if_exceeds(block.z, props.maxThreadsDim[2], "Block Z");
    throw_if_exceeds(block.x * block.y * block.z, props.maxThreadsPerBlock,
                     "Total threads");
    throw_if_exceeds(grid.x, props.maxGridSize[0], "Grid X");
    throw_if_exceeds(grid.y, props.maxGridSize[1], "Grid Y");
    throw_if_exceeds(grid.z, props.maxGridSize[2], "Grid Z");
}

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

    return std::format("Device {}: {} (CC {}.{}), Mem: {} MiB", device,
                       props.name, props.major, props.minor,
                       props.totalGlobalMem >> 20);
}

} // namespace dmt::error

#define DMT_CHECK_CUDA_CALL(call, ...)                                         \
    dmt::error::check_cuda_call(call, __VA_ARGS__)
#define DMT_CHECK_CUFFT_CALL(call, ...)                                        \
    dmt::error::check_cuda_call(call, __VA_ARGS__)

/**
 * @brief Macro to check the last asynchronous CUDA error (e.g., after kernel
 * launch <<<>>>).
 * @param msg Optional user message string literal.
 */
#define DMT_CHECK_LAST_CUDA_ERROR(...)                                         \
    dmt::error::check_last_cuda_error(__VA_ARGS__)

/**
 * @brief Macro to synchronize the device/stream and check for errors.
 * @param msg Optional user message string literal.
 */
#define DMT_CHECK_CUDA_SYNC(...) dmt::error::check_cuda_sync(__VA_ARGS__)

/**
 * @brief Macro to check kernel launch parameters before launch.
 * @param grid The dim3 grid dimensions.
 * @param block The dim3 block dimensions.
 */
#define DMT_CHECK_KERNEL_PARAMS(grid, block)                                   \
    dmt::error::check_kernel_launch_params(grid, block)
