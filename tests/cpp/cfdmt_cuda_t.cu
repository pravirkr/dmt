#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/utils/simulate.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::CohFDMT;

namespace {

// Same search as tests/cpp/cfdmt_cpu_t.cpp: 16 x 1 MHz at 392-408 MHz,
// t_p = 4 us, DM 10-11 (5 coarse trials).
CohFDMTConfig small_config() {
    return {.f_center = 400.0F,
            .bw_sub   = 1.0F,
            .nsub     = 16,
            .t_p      = 4.0E-6F,
            .dm_min   = 10.0F,
            .dm_max   = 11.0F,
            .format   = BasebandFormat{.order = "FTPRI"}};
}

// Noise plus a dispersed pulse, quantised into @p cfg's format.
std::vector<uint8_t> make_block(const CohFDMTConfig& cfg, SizeType n) {
    const plans::CohFDMTPlan plan(cfg);
    const double dm    = plan.get_dm_grid_coh()[2];
    const double t_ref = plan.get_output_time_offset() +
                         (100.0 * static_cast<double>(plan.get_tsamp()));
    const utils::BasebandPulse pulse{
        .dm        = dm,
        .t_arrival = t_ref - (static_cast<double>(kDispConst) * dm /
                              (static_cast<double>(plan.get_f_ref()) *
                               static_cast<double>(plan.get_f_ref()))),
        .fluence   = 1.0E7};
    const auto v = utils::simulate_baseband(cfg.f_center, cfg.bw_sub, cfg.nsub,
                                            n, std::span(&pulse, 1), 4.0F, 5, 4);
    return utils::pack_baseband(v, cfg.nsub, n, cfg.format, 1.0F);
}

// Relative agreement: |gpu - cpu| <= rel * max|cpu| everywhere.
void require_close(const std::vector<float>& gpu,
                   const std::vector<float>& cpu,
                   double rel) {
    REQUIRE(gpu.size() == cpu.size());
    double scale = 0.0;
    for (const float x : cpu) {
        scale = std::max(scale, static_cast<double>(std::abs(x)));
    }
    double worst = 0.0;
    for (SizeType i = 0; i < cpu.size(); ++i) {
        worst = std::max(worst,
                         std::abs(static_cast<double>(gpu[i] - cpu[i])));
    }
    INFO("max |gpu - cpu| / max |cpu| = " << worst / scale);
    CHECK(worst <= rel * scale);
}

} // namespace

TEST_CASE("CohFDMT (gpu) host execute matches the CPU engine",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    auto cfg = GENERATE(small_config(), [] {
        auto c      = small_config();
        c.normalize = false;
        return c;
    }(), [] {
        auto c         = small_config();
        c.format.order = "PRITF";
        c.format.nbits = 4;
        c.dt_step      = 4;
        return c;
    }());
    CAPTURE(cfg.format.order, cfg.format.nbits, cfg.normalize, cfg.dt_step);
    const CohFDMT cpu(cfg, Exec::cpu(4));
    const CohFDMT gpu(cfg, test::gpu_exec());
    REQUIRE(gpu.get_dmt_size() == cpu.get_dmt_size());
    const auto in = make_block(cfg, cpu.get_block_nsamps());
    std::vector<float> out_cpu(cpu.get_dmt_size());
    std::vector<float> out_gpu(gpu.get_dmt_size());
    cpu.execute<uint8_t>(std::span<const uint8_t>(in), out_cpu);
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), out_gpu);
    require_close(out_gpu, out_cpu, 1.0E-4);
    // Stateless: a second call gives the same result.
    std::vector<float> again(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), again);
    CHECK(again == out_gpu);
    CHECK(gpu.get_memory_usage().total() > 0);
}

TEST_CASE("CohFDMT (gpu) device and multi-group execute match the host path",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    auto cfg           = small_config();
    cfg.subband_groups = {6, 10};
    const CohFDMT gpu(cfg, test::gpu_exec());
    const auto n    = gpu.get_block_nsamps();
    auto one        = small_config();
    const auto full = make_block(one, n);
    // FTPRI is subband-major: the groups are consecutive byte ranges.
    const SizeType split = gpu.get_input_size(0);
    const std::vector<uint8_t> g0(full.begin(),
                                  full.begin() +
                                      static_cast<std::ptrdiff_t>(split));
    const std::vector<uint8_t> g1(
        full.begin() + static_cast<std::ptrdiff_t>(split), full.end());

    const CohFDMT host_ref(one, test::gpu_exec());
    std::vector<float> want(host_ref.get_dmt_size());
    host_ref.execute<uint8_t>(std::span<const uint8_t>(full), want);

    const std::array<std::span<const uint8_t>, 2> h_groups{
        std::span<const uint8_t>(g0), std::span<const uint8_t>(g1)};
    std::vector<float> got_host(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const std::span<const uint8_t>>(h_groups),
                         got_host);
    CHECK(got_host == want);

    thrust::device_vector<uint8_t> d0(g0.begin(), g0.end());
    thrust::device_vector<uint8_t> d1(g1.begin(), g1.end());
    const std::array<DeviceSpan<const uint8_t>, 2> d_groups{
        DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(d0.data()),
                                  d0.size()),
        DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(d1.data()),
                                  d1.size())};
    thrust::device_vector<float> d_out(gpu.get_dmt_size());
    gpu.execute<uint8_t>(
        std::span<const DeviceSpan<const uint8_t>>(d_groups),
        DeviceSpan<float>(thrust::raw_pointer_cast(d_out.data()),
                          d_out.size()));
    cudaDeviceSynchronize();
    const thrust::host_vector<float> h_out = d_out;
    CHECK(std::vector<float>(h_out.begin(), h_out.end()) == want);
}

TEST_CASE("CohFDMT (gpu) validates buffers", "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const CohFDMT gpu(small_config(), test::gpu_exec());
    std::vector<uint8_t> in(gpu.get_input_size());
    std::vector<uint8_t> bad_in(gpu.get_input_size() - 4);
    std::vector<float> out(gpu.get_dmt_size());
    std::vector<float> bad_out(gpu.get_dmt_size() - 1);
    CHECK_THROWS_AS(gpu.execute<uint8_t>(std::span<const uint8_t>(bad_in), out),
                    std::invalid_argument);
    CHECK_THROWS_AS(gpu.execute<uint8_t>(std::span<const uint8_t>(in), bad_out),
                    std::invalid_argument);
}

} // namespace dmt
