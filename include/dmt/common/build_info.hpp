#pragma once

#include <string>
#include <string_view>

namespace dmt {

/**
 * @brief Identity of this libdmt build, for provenance of results.
 *
 * Benchmarks and sensitivity datasets store it next to their numbers, so a
 * result can be traced to the exact source and toolchain. The git commit and
 * dirty flag are refreshed on every build; a dirty build cannot be
 * reproduced from its commit alone.
 */
struct BuildInfo {
    std::string_view version;   ///< Project version (CMake project())
    std::string_view git_sha;   ///< Full commit SHA, "unknown" without git
    bool git_dirty;             ///< Sources differed from git_sha (or were new)
    std::string_view compiler;  ///< C++ compiler id and version
    std::string_view cxx_flags; ///< C++ flags of the build type, -march
    std::string_view build_type; ///< CMAKE_BUILD_TYPE
    std::string_view gpu;  ///< "CUDA <version> (<archs>)", "HIP ..." or "none"
    std::string_view fftw; ///< Linked FFTW (single precision) version
    std::string_view openmp; ///< OpenMP version of the C++ compiler
};

/// @brief The identity of the linked libdmt.
[[nodiscard]] const BuildInfo& build_info() noexcept;

/// @brief build_info() on one line, e.g. for a log or a file header.
[[nodiscard]] std::string build_info_string();

} // namespace dmt
