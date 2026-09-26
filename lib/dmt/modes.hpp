#pragma once

#include <format>
#include <stdexcept>
#include <string_view>

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"

namespace dmt {

// Sets the process-wide spdlog level: the most recently constructed object's
// `verbose` wins. 0 still reports warnings (e.g. a clamped fuse_levels).
inline void apply_log_verbosity(int verbose) {
    if (verbose <= 0) {
        spdlog::set_level(spdlog::level::warn);
    } else if (verbose == 1) {
        spdlog::set_level(spdlog::level::info);
    } else {
        spdlog::set_level(spdlog::level::debug);
    }
}

[[nodiscard]] inline FDMTMode parse_fdmt_mode(std::string_view mode) {
    if (mode == "full") {
        return FDMTMode::kFull;
    }
    if (mode == "roll") {
        return FDMTMode::kRoll;
    }
    if (mode == "valid") {
        return FDMTMode::kValid;
    }
    throw std::invalid_argument(std::format(
        "Invalid mode '{}'. Expected 'full', 'roll', or 'valid'", mode));
}

[[nodiscard]] inline std::string_view fdmt_mode_to_string(FDMTMode mode) {
    switch (mode) {
    case FDMTMode::kFull:
        return "full";
    case FDMTMode::kRoll:
        return "roll";
    case FDMTMode::kValid:
        return "valid";
    }
    throw std::invalid_argument("Invalid FDMTMode");
}

[[nodiscard]] inline BasebandDataOrder
parse_baseband_data_order(std::string_view name) {
    if (name == "FTPRI") {
        return BasebandDataOrder::kFTPRI;
    }
    if (name == "PRITF") {
        return BasebandDataOrder::kPRITF;
    }
    if (name == "RITFP") {
        return BasebandDataOrder::kRITFP;
    }
    throw std::invalid_argument(
        std::format("Invalid baseband data order: {}. Expected 'FTPRI', "
                    "'PRITF', or 'RITFP'",
                    name));
}

[[nodiscard]] inline std::string_view
baseband_data_order_to_string(BasebandDataOrder order) {
    switch (order) {
    case BasebandDataOrder::kFTPRI:
        return "FTPRI";
    case BasebandDataOrder::kPRITF:
        return "PRITF";
    case BasebandDataOrder::kRITFP:
        return "RITFP";
    }
    throw std::invalid_argument("Invalid BasebandDataOrder");
}

} // namespace dmt
