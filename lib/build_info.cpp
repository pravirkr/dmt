#include "dmt/common/build_info.hpp"

#include <string_view>

#include <fftw3.h>
#include <fmt/format.h>

#include "dmt_build_config.hpp"
#include "dmt_git_info.hpp"

namespace dmt {

namespace {

// One string with the commit, so that tools can read a binary's identity
// without running it (bench/scripts/run_suite.py rejects stale binaries).
// build_info() refers to it, which keeps it in every binary that links it.
constexpr std::string_view kPrefix = "DMT_BUILD_INFO|sha=";
constexpr const char* kMarker =
    "DMT_BUILD_INFO|sha=" DMT_GIT_SHA "|dirty=" DMT_BUILD_INFO_DIRTY_STR "|";

} // namespace

const BuildInfo& build_info() noexcept {
    static const BuildInfo kInfo{
        .version = DMT_BI_VERSION,
        .git_sha = std::string_view(kMarker).substr(
            kPrefix.size(), std::string_view(DMT_GIT_SHA).size()),
        .git_dirty  = DMT_GIT_DIRTY != 0,
        .compiler   = DMT_BI_COMPILER,
        .cxx_flags  = DMT_BI_CXX_FLAGS,
        .build_type = DMT_BI_BUILD_TYPE,
        .gpu        = DMT_BI_GPU,
        .fftw       = fftwf_version,
        .openmp     = DMT_BI_OPENMP,
    };
    return kInfo;
}

std::string build_info_string() {
    const auto& b = build_info();
    return fmt::format("dmt {} ({}{}), {} {}, {}, GPU {}, FFTW {}, OpenMP {}",
                       b.version, b.git_sha, b.git_dirty ? ", dirty" : "",
                       b.build_type, b.compiler, b.cxx_flags, b.gpu, b.fftw,
                       b.openmp);
}

} // namespace dmt
