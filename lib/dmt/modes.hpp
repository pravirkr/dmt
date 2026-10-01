#pragma once

#include <format>
#include <stdexcept>
#include <string_view>

#include "dmt/common/types.hpp"

namespace dmt {

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

} // namespace dmt
