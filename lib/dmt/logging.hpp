#pragma once

#include <string_view>
#include <utility>

#include <fmt/format.h>

#include "dmt/common/logging.hpp"

// Internal logging front end: construction-time debug messages only.
// Formatting uses fmt::format and happens only when debug logging is on;
// spdlog is confined to lib/logging.cpp, so no other translation unit (and
// nothing downstream) sees it.

namespace dmt::logging {

/// Whether debug messages are emitted (one relaxed atomic load).
[[nodiscard]] bool enabled() noexcept;

/// Writes an already formatted debug message.
void emit(std::string_view message) noexcept;

template <typename... Args>
void debug(fmt::format_string<Args...> fmt, Args&&... args) {
    if (enabled()) {
        emit(fmt::format(fmt, std::forward<Args>(args)...));
    }
}

} // namespace dmt::logging
