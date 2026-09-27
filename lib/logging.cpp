#include "dmt/logging.hpp"

#include <atomic>
#include <memory>

#include <spdlog/logger.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace dmt {

namespace {

std::atomic<LogLevel>& level_ref() noexcept {
    static std::atomic<LogLevel> level{LogLevel::kOff};
    return level;
}

spdlog::logger& logger() {
    static const auto kLogger = [] {
        auto sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        auto log  = std::make_shared<spdlog::logger>("dmt", std::move(sink));
        log->set_pattern("[dmt] [%l] %v");
        // Filtering happens in logging::enabled(); the sink emits the rest.
        log->set_level(spdlog::level::trace);
        return log;
    }();
    return *kLogger;
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
        logger().debug("{}", message);
    } catch (...) { // NOLINT(bugprone-empty-catch): logging must not throw
    }
}

} // namespace logging

} // namespace dmt
