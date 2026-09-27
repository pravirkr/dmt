#pragma once

#include <cstdint>

namespace dmt {

enum class LogLevel : std::uint8_t {
    kOff   = 0, ///< Nothing (default).
    kDebug = 1, ///< Construction-time details (plans, FFTs, CUDA fusion).
};

/// @brief Sets dmt's process-wide log level (thread-safe).
void set_log_level(LogLevel level) noexcept;

/// @brief Current process-wide log level.
[[nodiscard]] LogLevel get_log_level() noexcept;

} // namespace dmt
