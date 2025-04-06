#pragma once

#include <format>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

// Add formatter specialization for cudaError_t
template <>
struct std::formatter<cudaError_t> : std::formatter<int> {
    auto format(cudaError_t error, format_context& ctx) const {
        return std::formatter<int>::format(static_cast<int>(error), ctx);
    }
};

namespace dmt::error {

/**
 * @brief Custom exception class for CUDA errors.
 */
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t code,
                  std::string_view user_msg,
                  const std::source_location& location)
        : std::runtime_error(format_what(code, user_msg, location)),
          m_error_code(code),
          m_file_name(location.file_name()),
          m_line_number(location.line()),
          m_function_name(location.function_name()),
          m_user_message(user_msg) {}

    /**
     * @brief Gets the CUDA error code.
     * @return cudaError_t The specific CUDA error enum value.
     */
    [[nodiscard]] cudaError_t code() const noexcept { return m_error_code; }

    /**
     * @brief Gets the name of the source file where the error occurred.
     */
    [[nodiscard]] const char* file_name() const noexcept { return m_file_name; }

    /**
     * @brief Gets the line number where the error occurred.
     */
    [[nodiscard]] std::uint_least32_t line() const noexcept {
        return m_line_number;
    }

    /**
     * @brief Gets the name of the function where the error occurred.
     */
    [[nodiscard]] const char* function_name() const noexcept {
        return m_function_name;
    }

    /**
     * @brief Gets the user-provided message associated with the error check.
     */
    [[nodiscard]] const std::string& user_message() const noexcept {
        return m_user_message;
    }

    /**
     * @brief Gets the official CUDA error string for the error code.
     */
    [[nodiscard]] const char* cuda_error_string() const noexcept {
        return cudaGetErrorString(m_error_code);
    }

private:
    cudaError_t m_error_code;
    const char* m_file_name;
    std::uint_least32_t m_line_number;
    const char* m_function_name;
    std::string m_user_message; // Store user message

    // Helper to format the what() message for the base class
    static std::string format_what(cudaError_t code,
                                   std::string_view user_msg,
                                   const std::source_location& location) {
        std::string combined_msg = cudaGetErrorString(code);
        if (!user_msg.empty()) {
            combined_msg += " - User message: ";
            combined_msg += user_msg;
        }
        return std::format("CUDA error [{}] in function '{}' ({}:{}): {}",
                           static_cast<int>(code), location.function_name(),
                           location.file_name(), location.line(), combined_msg);
    }
};

/**
 * @brief Checks the result of a synchronous CUDA API call. Throws CudaException
 * on failure.
 * @param result The cudaError_t code returned by the CUDA API call.
 * @param location Source location information (automatically captured).
 * @param msg Optional user message to include in the exception.
 * @throws CudaException if result is not cudaSuccess.
 */
inline void check_cuda_call(
    cudaError_t result,
    const std::source_location location = std::source_location::current(),
    std::string_view msg                = "") {
    if (result != cudaSuccess) {
        throw CudaException(result, msg, location);
    }
}

/**
 * @brief Checks the last asynchronous CUDA error (e.g., after kernel launch).
 * Throws CudaException on failure.
 * @param location Source location information (automatically captured).
 * @param msg Optional user message to include in the exception.
 * @throws CudaException if cudaGetLastError() returns an error.
 */
inline void check_last_cuda_error(
    const std::source_location location = std::source_location::current(),
    std::string_view msg                = "") {
    // cudaGetLastError resets the error state, so call it unconditionally
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException(error, msg, location);
    }
}

/**
 * @brief Synchronizes the default stream (or all streams on device pre-CUDA 11)
 * and checks for asynchronous errors.
 * @param location Source location information (automatically captured).
 * @param msg Optional user message to include in the exception.
 * @throws CudaException if synchronization or subsequent error check fails.
 */
inline void check_cuda_sync_error(
    const std::source_location location = std::source_location::current(),
    std::string_view msg                = "") {
    // Explicitly check synchronize result first
    check_cuda_call(cudaDeviceSynchronize(), location,
                    "cudaDeviceSynchronize failed");
    // Then check if any async error occurred before the sync
    check_last_cuda_error(location, msg);
}

inline void check_last_cuda_error_sync(
    const std::source_location location = std::source_location::current(),
    std::string_view msg                = "") {
    check_cuda_sync_error(location, msg);
}

inline void check_cuda_sync(
    const std::source_location location = std::source_location::current(),
    std::string_view msg                = "") {
    check_cuda_sync_error(location, msg);
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
    const std::source_location location = std::source_location::current()) {
    cudaDeviceProp props{};
    int device = -1;
    // Get current device, check for errors
    check_cuda_call(cudaGetDevice(&device), location,
                    "check_kernel_launch_params: cudaGetDevice failed");
    check_cuda_call(
        cudaGetDeviceProperties(&props, device), location,
        "check_kernel_launch_params: cudaGetDeviceProperties failed");

    // Check block dimensions
    if (block.x > static_cast<unsigned int>(props.maxThreadsDim[0]) ||
        block.y > static_cast<unsigned int>(props.maxThreadsDim[1]) ||
        block.z > static_cast<unsigned int>(props.maxThreadsDim[2])) {
        throw std::runtime_error(std::format(
            "Invalid block dimensions ({},{},{}) exceed device {} limits "
            "({},{},{}) at {}:{}",
            block.x, block.y, block.z, device, props.maxThreadsDim[0],
            props.maxThreadsDim[1], props.maxThreadsDim[2],
            location.file_name(), location.line()));
    }
    // Check total threads per block
    if (block.x * block.y * block.z >
        static_cast<unsigned int>(props.maxThreadsPerBlock)) {
        throw std::runtime_error(std::format(
            "Total threads per block ({}) exceed device {} limit ({}) at {}:{}",
            block.x * block.y * block.z, device, props.maxThreadsPerBlock,
            location.file_name(), location.line()));
    }

    // Check grid dimensions
    if (grid.x > static_cast<unsigned int>(props.maxGridSize[0]) ||
        grid.y > static_cast<unsigned int>(props.maxGridSize[1]) ||
        grid.z > static_cast<unsigned int>(props.maxGridSize[2])) {
        throw std::runtime_error(
            std::format("Invalid grid dimensions ({},{},{}) exceed device {} "
                        "limits ({},{},{}) at {}:{}",
                        grid.x, grid.y, grid.z, device, props.maxGridSize[0],
                        props.maxGridSize[1], props.maxGridSize[2],
                        location.file_name(), location.line()));
    }
}

/**
 * @brief Gets formatted information about the current CUDA device. Useful for
 * error reporting context.
 * @return std::string String containing device name, compute capability, and
 * global memory. Returns error message on failure.
 */
[[nodiscard]] inline std::string get_current_device_info() {
    int device          = -1;
    cudaError_t err_dev = cudaGetDevice(&device);
    if (err_dev != cudaSuccess) {
        return std::format("Failed to get current CUDA device: {}",
                           cudaGetErrorString(err_dev));
    }

    cudaDeviceProp props{};
    cudaError_t err_props = cudaGetDeviceProperties(&props, device);
    if (err_props != cudaSuccess) {
        return std::format("Failed to get properties for CUDA device {}: {}",
                           device, cudaGetErrorString(err_props));
    }

    // Use MiB for memory size for better readability
    const auto mem_mib =
        props.totalGlobalMem / (static_cast<std::size_t>(1024) * 1024);
    return std::format("Device {}: {} (CC {}.{}), Global Mem: {} MB", device,
                       props.name, props.major, props.minor, mem_mib);
}

} // namespace dmt::error

/**
 * @brief Macro to check the result of a synchronous CUDA API call.
 * @param call The CUDA API call (e.g., cudaMalloc(...)).
 * @param msg Optional user message string literal.
 */
#define DMT_CHECK_CUDA_CALL(call, msg)                                         \
    dmt::error::check_cuda_call(call, std::source_location::current(), msg)

/**
 * @brief Macro to check the last asynchronous CUDA error (e.g., after kernel
 * launch <<<>>>).
 * @param msg Optional user message string literal.
 */
#define DMT_CHECK_LAST_CUDA_ERROR(msg)                                         \
    dmt::error::check_last_cuda_error(std::source_location::current(), msg)

/**
 * @brief Macro to synchronize the device/stream and check for errors.
 * @param msg Optional user message string literal.
 */
#define DMT_CHECK_CUDA_SYNC(msg)                                               \
    dmt::error::check_cuda_sync(std::source_location::current(), msg)

/**
 * @brief Macro to check kernel launch parameters before launch.
 * @param grid The dim3 grid dimensions.
 * @param block The dim3 block dimensions.
 */
#define DMT_CHECK_KERNEL_PARAMS(grid, block)                                   \
    dmt::error::check_kernel_launch_params(grid, block,                        \
                                           std::source_location::current())
