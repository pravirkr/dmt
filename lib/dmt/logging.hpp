#pragma once

#include <format>
#include <string_view>
#include <utility>

#include "dmt/common/logging.hpp"

// Internal logging front end: construction-time debug messages only.
// Formatting uses std::format and happens only when debug logging is on;
// spdlog is confined to lib/logging.cpp, so no other translation unit (and
// nothing downstream) sees it.

namespace dmt::logging {

/// Whether debug messages are emitted (one relaxed atomic load).
[[nodiscard]] bool enabled() noexcept;

/// Writes an already formatted debug message.
void emit(std::string_view message) noexcept;

template <typename... Args>
void debug(std::format_string<Args...> fmt, Args&&... args) {
    if (enabled()) {
        emit(std::format(fmt, std::forward<Args>(args)...));
    }
}

} // namespace dmt::logging
