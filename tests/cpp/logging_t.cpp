#include <string>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/logging.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/logging.hpp"

namespace dmt {

TEST_CASE("Log level is process-wide, defaults to off and round-trips",
          "[logging][cpu]") {
    CHECK(get_log_level() == LogLevel::kOff);
    CHECK_FALSE(logging::enabled());

    set_log_level(LogLevel::kDebug);
    CHECK(get_log_level() == LogLevel::kDebug);
    CHECK(logging::enabled());
    // A debug-logged construction must not throw.
    CHECK_NOTHROW(algorithms::FDMTCPU(1000.0F, 1500.0F, 64, 256, 0.001F, 32));

    set_log_level(LogLevel::kOff);
    CHECK_FALSE(logging::enabled());
}

TEST_CASE("Plans and engines describe themselves via summary()",
          "[logging][cpu]") {
    const algorithms::FDMTCPU fdmt(1000.0F, 1500.0F, 64, 256, 0.001F, 32);
    const auto plan_text = fdmt.get_plan().summary();
    CHECK(plan_text.find("FDMT Plan Summary") != std::string::npos);
    CHECK(fdmt.get_plan().summary("  ").starts_with("  "));

    const auto engine_text = fdmt.summary();
    CHECK(engine_text.starts_with(plan_text));
    CHECK(engine_text.find("FDMTCPU") != std::string::npos);
    CHECK(engine_text.find("fuse_levels") != std::string::npos);
    CHECK(engine_text.find("Host memory") != std::string::npos);

    const plans::DDMTPlan ddmt(1000.0F, 1500.0F, 64, 0.001F, 10.0F, 1.0F);
    const auto ddmt_text = ddmt.summary();
    CHECK(ddmt_text.find("DDMT Plan Summary") != std::string::npos);
    CHECK(ddmt_text.find("11 over") != std::string::npos);
}

} // namespace dmt
