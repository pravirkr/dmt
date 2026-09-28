#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "dmt/dmt.hpp"

// Compiled with the host compiler only: the public headers must not need a
// GPU runtime (see the umbrella-header test in simulate_t.cpp as well).
#if defined(CUDART_VERSION) || defined(__CUDA_RUNTIME_H__)
#error "dmt public headers pulled in the CUDA runtime"
#endif

namespace dmt {

using algorithms::DDMT;
using algorithms::FDMT;
using algorithms::FDMTFFT;

TEST_CASE("Backend names round-trip", "[backend][cpu]") {
    for (const auto b : {Backend::kCPU, Backend::kCUDA, Backend::kHIP}) {
        CHECK(parse_backend(to_string(b)) == b);
    }
    CHECK_THROWS_AS(parse_backend("opencl"), std::invalid_argument);
}

TEST_CASE("available_backends matches the build", "[backend][cpu]") {
    const auto avail = available_backends();
    REQUIRE_FALSE(avail.empty());
    CHECK(avail.front() == Backend::kCPU);
    CHECK(is_available(Backend::kCPU));
#ifdef DMT_ENABLE_CUDA
    CHECK(is_available(Backend::kCUDA));
#else
    CHECK_FALSE(is_available(Backend::kCUDA));
#endif
#ifdef DMT_ENABLE_HIP
    CHECK(is_available(Backend::kHIP));
#else
    CHECK_FALSE(is_available(Backend::kHIP));
#endif
}

TEST_CASE("Constructing on a backend missing from the build throws",
          "[backend][cpu]") {
    const auto expect_unavailable = [](auto make) {
        try {
            make();
            FAIL("expected std::invalid_argument");
        } catch (const std::invalid_argument& e) {
            CHECK_THAT(std::string(e.what()),
                       Catch::Matchers::ContainsSubstring("not available") &&
                           Catch::Matchers::ContainsSubstring("cpu"));
        }
    };
    for (const auto b : {Backend::kCUDA, Backend::kHIP}) {
        if (is_available(b)) {
            continue;
        }
        const Exec exec{.backend = b, .nthreads = 1, .device = 0};
        expect_unavailable([&] {
            return FDMT(1000.0F, 1500.0F, 16, 64, 0.001F, 8, 0, 1, true,
                        "valid", exec);
        });
        expect_unavailable([&] {
            return FDMTFFT(1000.0F, 1500.0F, 16, 64, 0.001F, 8, 0, 1, true,
                           "valid", exec);
        });
        expect_unavailable([&] {
            return DDMT(1000.0F, 1500.0F, 16, 0.001F, 10.0F, 1.0F, 0.0F, exec);
        });
    }
}

TEST_CASE("The CPU backend rejects device memory and streams",
          "[backend][cpu]") {
    FDMT fdmt(1000.0F, 1500.0F, 16, 64, 0.001F, 8, 0, 1, true, "roll",
              Exec::cpu(2));
    CHECK(fdmt.backend() == Backend::kCPU);
    CHECK(fdmt.nthreads() == 2);
    CHECK(fdmt.device() == -1);

    std::vector<float> wf(SizeType{16} * 64, 1.0F);
    std::vector<float> out(fdmt.get_plan().get_buffer_size());
    const DeviceSpan<const float> d_wf(wf.data(), wf.size());
    const DeviceSpan<float> d_out(out.data(), out.size());
    CHECK_THROWS_AS(fdmt.execute(d_wf, d_out), std::invalid_argument);
    CHECK_THROWS_AS(fdmt.reset(d_wf, d_out), std::invalid_argument);
    CHECK_THROWS_AS(fdmt.save_history(d_out), std::invalid_argument);
    CHECK_THROWS_AS(fdmt.view_level_data_device(), std::invalid_argument);

    // A device view that names another device is rejected before any work.
    const DeviceSpan<const float> d_other(wf.data(), wf.size(),
                                          {.backend = Backend::kCUDA, .id = 0});
    CHECK_THROWS_AS(fdmt.execute(d_other, d_out), std::invalid_argument);

    // The host path still works, and a stream is a mistake on the CPU.
    fdmt.reset(wf, out);
    int dummy = 0;
    CHECK_THROWS_AS(fdmt.advance(1, Stream{&dummy}), std::invalid_argument);
    CHECK_NOTHROW(fdmt.advance(1));
    CHECK_NOTHROW(fdmt.finalize());

    DDMT ddmt(1000.0F, 1500.0F, 16, 0.001F, 10.0F, 1.0F);
    CHECK(ddmt.device() == -1);
    CHECK_THROWS_AS(ddmt.execute(d_wf, d_out), std::invalid_argument);

    FDMTFFT fft(1000.0F, 1500.0F, 16, 64, 0.001F, 8);
    CHECK_THROWS_AS(fft.execute(d_wf, d_out), std::invalid_argument);
}

TEST_CASE("Default and empty DeviceSpan::data is nullptr", "[backend][cpu]") {
    const DeviceSpan<float> empty{};
    CHECK(empty.data() == nullptr);
    CHECK(empty.empty());
    const DeviceSpan<float> zero_count{nullptr, 0};
    CHECK(zero_count.data() == nullptr);
}

TEST_CASE("DeviceSpan::subspan keeps the handle and advances the offset",
          "[backend][cpu]") {
    std::vector<float> buf(16);
    const DeviceSpan<float> whole(buf.data(), buf.size(),
                                  {.backend = Backend::kCUDA, .id = 1});
    const auto part = whole.subspan(4, 8);
    CHECK(part.handle == whole.handle);
    CHECK(part.byte_offset == 4 * sizeof(float));
    CHECK(part.size() == 8);
    CHECK(part.data() == buf.data() + 4);
    CHECK(part.device.id == 1);
    const DeviceSpan<const float> cpart = part.subspan(2, 2);
    CHECK(cpart.data() == buf.data() + 6);
}

TEST_CASE("Exec replaces the old nthreads / device_id slot", "[backend][cpu]") {
    constexpr auto kCpu = Exec::cpu(4);
    STATIC_REQUIRE(kCpu.backend == Backend::kCPU);
    STATIC_REQUIRE(kCpu.nthreads == 4);
    constexpr auto kGPU = Exec::cuda(1);
    STATIC_REQUIRE(kGPU.backend == Backend::kCUDA);
    STATIC_REQUIRE(kGPU.device == 1);
    // An int never converts to Exec, so a stale positional nthreads argument
    // fails to compile instead of landing in the next parameter.
    STATIC_REQUIRE_FALSE(std::is_convertible_v<int, Exec>);
    STATIC_REQUIRE_FALSE(
        std::is_convertible_v<std::vector<float>&, DeviceSpan<float>>);
}

} // namespace dmt
