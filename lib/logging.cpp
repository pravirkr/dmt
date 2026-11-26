#include "dmt/logging.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string_view>

namespace dmt {

namespace {

std::atomic<LogLevel>& level_ref() noexcept {
    static std::atomic<LogLevel> level{LogLevel::kOff};
    return level;
}

// One lock for every message, so lines from concurrent constructions do not
// interleave.
std::mutex& stderr_mutex() noexcept {
    static std::mutex mutex;
    return mutex;
}

} // namespace

void set_log_level(LogLevel level) noexcept {
    level_ref().store(level, std::memory_order_relaxed);
}

LogLevel get_log_level() noexcept {
    return level_ref().load(std::memory_order_relaxed);
}

namespace logging {

bool enabled() noexcept { return get_log_level() == LogLevel::kDebug; }

void emit(std::string_view message) noexcept {
    try {
        const std::scoped_lock lock(stderr_mutex());
        std::fputs("[dmt] [debug] ", stderr);
        std::fwrite(message.data(), 1, message.size(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    } catch (...) { // NOLINT(bugprone-empty-catch): logging must not throw
    }
}

} // namespace logging

} // namespace dmt
