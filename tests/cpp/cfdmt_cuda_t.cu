#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/algorithms/cfdmt.hpp"
#include "dmt/engines.hpp"
#include "dmt/utils/simulate.hpp"
#include "cfdmt_behaviour.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::CohFDMT;
using algorithms::detail::CohFDMTCoherentPath;
using algorithms::detail::CohFDMTEngineConfig;
using test::cfdmt::small_config;

namespace {

// Noise plus a dispersed pulse, quantised into @p cfg's format.
std::vector<uint8_t> make_block(const CohFDMTConfig& cfg, SizeType n) {
    const plans::CohFDMTPlan plan(cfg);
    const double dm =
        plan.get_dm_grid_coh()[std::min<SizeType>(2, plan.get_ndm_coh() - 1)];
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

std::vector<float> run_cpu(const CohFDMTConfig& cfg,
                           const std::vector<uint8_t>& in) {
    const CohFDMT cpu(cfg, Exec::cpu(4));
    std::vector<float> out(cpu.get_dmt_size());
    cpu.execute<uint8_t>(std::span<const uint8_t>(in), out);
    return out;
}

// One engine with a pinned coherent-stage path (host memory).
std::vector<float> run_engine(const CohFDMTConfig& cfg,
                              const std::vector<uint8_t>& in,
                              CohFDMTCoherentPath path,
                              SizeType work_bytes = 0) {
    const plans::CohFDMTPlan plan(cfg);
    const auto engine = algorithms::detail::make_cfdmt_gpu(
        plan, CohFDMTEngineConfig{.exec          = test::gpu_exec(),
                                  .coherent_path = path,
                                  .work_bytes    = work_bytes});
    std::vector<float> out(plan.get_dmt_size());
    const std::span<const uint8_t> group(in);
    engine->execute(std::span(&group, 1), out);
    return out;
}

// Coarse-trial-rich variant of the small search: 16 coarse trials.
CohFDMTConfig many_trials_config() {
    auto c   = small_config();
    c.dm_max = 13.0F;
    return c;
}

} // namespace

// --- The CPU behaviour suite (cfdmt_behaviour.hpp) on the GPU ---

TEST_CASE("CohFDMT (gpu) recovers an injected dispersed impulse",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_impulse_recovery(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) impulse response does not depend on position",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_position_independence(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) skipback blocks tile one long block",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_skipback_tiling(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) normalised noise matches the variance grid",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_noise_statistics(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) output is independent of layout, encoding and "
          "grouping",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_layout_independence(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) 4-bit input and dt_step recover the pulse",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_4bit_dt_step(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) execute validates buffers", "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_validation(test::gpu_exec());
    const CohFDMT gpu(small_config(), test::gpu_exec());
    thrust::device_vector<uint8_t> d_in(gpu.get_input_size());
    thrust::device_vector<float> d_out(gpu.get_dmt_size());
    const DeviceSpan<const uint8_t> din(thrust::raw_pointer_cast(d_in.data()),
                                        d_in.size());
    const DeviceSpan<float> dout(thrust::raw_pointer_cast(d_out.data()),
                                 d_out.size());
    CHECK_THROWS_AS(gpu.execute<uint8_t>(din.subspan(0, din.size() - 4), dout),
                    std::invalid_argument);
    CHECK_THROWS_AS(gpu.execute<uint8_t>(din, dout.subspan(0, dout.size() - 1)),
                    std::invalid_argument);
    const std::array<DeviceSpan<const uint8_t>, 2> two{din, din};
    CHECK_THROWS_AS(
        gpu.execute<uint8_t>(std::span<const DeviceSpan<const uint8_t>>(two),
                             dout),
        std::invalid_argument);
}

TEST_CASE("CohFDMT (gpu) trimmed filter margin matches a 1e-6 margin",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_filter_leakage(test::gpu_exec());
}

TEST_CASE("CohFDMT (gpu) concurrent calls on one engine", "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    test::cfdmt::check_concurrent_calls(test::gpu_exec());
}

// --- CPU parity ---

TEST_CASE("CohFDMT (gpu) host execute matches the CPU engine",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto order     = GENERATE(std::string("FTPRI"), std::string("PRITF"),
                                    std::string("TFPRI"));
    const auto nbits     = GENERATE(SizeType{8}, SizeType{4}, SizeType{2});
    const auto normalize = GENERATE(true, false);
    const auto dt_step   = GENERATE(SizeType{1}, SizeType{4});
    auto cfg             = small_config();
    cfg.format.order     = order;
    cfg.format.nbits     = nbits;
    cfg.normalize        = normalize;
    cfg.dt_step          = dt_step;
    CAPTURE(order, nbits, normalize, dt_step);
    const CohFDMT gpu(cfg, test::gpu_exec());
    const auto in  = make_block(cfg, gpu.get_block_nsamps());
    const auto cpu = run_cpu(cfg, in);
    std::vector<float> out(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), out);
    require_close(out, cpu, 1.0E-4);
    // Stateless: a second call gives the same result.
    std::vector<float> again(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), again);
    CHECK(again == out);
    CHECK(gpu.get_memory_usage().total() > 0);
}

TEST_CASE("CohFDMT (gpu) matches the CPU with many coarse trials",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto cfg = many_trials_config();
    const CohFDMT gpu(cfg, test::gpu_exec());
    REQUIRE(gpu.get_plan().get_ndm_coh() >= 10);
    const auto in = make_block(cfg, gpu.get_block_nsamps());
    std::vector<float> out(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), out);
    require_close(out, run_cpu(cfg, in), 1.0E-4);
}

// --- Coherent-stage implementations ---

TEST_CASE("CohFDMT (gpu) fused and cuFFT coherent stages agree",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto cfg = many_trials_config();
    const plans::CohFDMTPlan plan(cfg);
    const auto in    = make_block(cfg, plan.get_block_nsamps());
    const auto cpu   = run_cpu(cfg, in);
    const auto fused = run_engine(cfg, in, CohFDMTCoherentPath::kFused);
    const auto cufft = run_engine(cfg, in, CohFDMTCoherentPath::kCuFFT);
    require_close(fused, cpu, 1.0E-4);
    require_close(cufft, cpu, 1.0E-4);
    require_close(fused, cufft, 1.0E-5);
    SECTION("cuFFT path in channel chunks") {
        // A work buffer of a few channels forces several chunks per trial.
        const auto nwin = ((plan.get_fdmt_nsamps() +
                            (plan.get_mbin() - (2 * plan.get_noverlap() /
                                                plan.get_n_p())) -
                            1) /
                           (plan.get_mbin() -
                            (2 * plan.get_noverlap() / plan.get_n_p()))) +
                          1;
        const SizeType per_chan = nwin * 2 * plan.get_mbin() * 8;
        const auto chunked = run_engine(cfg, in, CohFDMTCoherentPath::kCuFFT,
                                        (5 * per_chan) + 7);
        CHECK(chunked == cufft);
    }
}

TEST_CASE("CohFDMT (gpu) fused kernel at every channel FFT length",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto log2n = GENERATE(range(4, 13));
    // Short margins (high frequency, tiny DM range, loose filter margin) so
    // that even 16-bin channel FFTs are valid.
    CohFDMTConfig cfg{.f_center       = 1400.0F,
                      .bw_sub         = 1.0F,
                      .nsub           = 8,
                      .t_p            = 4.0E-6F,
                      .dm_min         = 0.0F,
                      .dm_max         = 0.05F,
                      .filter_leakage = 0.05F,
                      .format         = BasebandFormat{.order = "FTPRI"}};
    const plans::CohFDMTPlan base(cfg);
    cfg.nbin = base.get_n_p() * (SizeType{1} << log2n);
    CAPTURE(cfg.nbin, log2n);
    bool valid = true;
    try {
        const plans::CohFDMTPlan probe(cfg);
    } catch (const std::invalid_argument&) {
        valid = false; // shorter than the coherent filter margin
    }
    if (!valid) {
        SKIP("mbin 2^" << log2n << " below the filter margin");
    }
    const plans::CohFDMTPlan plan(cfg);
    const auto in    = make_block(cfg, plan.get_block_nsamps());
    const auto fused = run_engine(cfg, in, CohFDMTCoherentPath::kFused);
    require_close(fused, run_cpu(cfg, in), 1.0E-4);
}

TEST_CASE("CohFDMT (gpu) falls back to cuFFT for non-power-of-two channels",
          "[cfdmt_gpu][gpu][parity]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    auto cfg = small_config();
    const plans::CohFDMTPlan base(cfg);
    cfg.nbin = base.get_n_p() * 3 * base.get_mbin() / 2;
    const plans::CohFDMTPlan plan(cfg);
    REQUIRE((plan.get_mbin() & (plan.get_mbin() - 1)) != 0);
    CHECK_THROWS_AS(run_engine(cfg, {}, CohFDMTCoherentPath::kFused),
                    std::invalid_argument);
    const auto in = make_block(cfg, plan.get_block_nsamps());
    const CohFDMT gpu(cfg, test::gpu_exec()); // automatic: cuFFT path
    std::vector<float> out(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), out);
    require_close(out, run_cpu(cfg, in), 1.0E-4);
}

// --- Device memory, groups and streams ---

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

    // Device groups inside larger buffers (non-zero offsets).
    constexpr SizeType kPad = 37;
    thrust::device_vector<uint8_t> d0(g0.size() + kPad);
    thrust::device_vector<uint8_t> d1(g1.size() + (2 * kPad));
    thrust::copy(g0.begin(), g0.end(), d0.begin() + kPad);
    thrust::copy(g1.begin(), g1.end(), d1.begin() + (2 * kPad));
    const std::array<DeviceSpan<const uint8_t>, 2> d_groups{
        DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(d0.data()),
                                  d0.size())
            .subspan(kPad, g0.size()),
        DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(d1.data()),
                                  d1.size())
            .subspan(2 * kPad, g1.size())};
    thrust::device_vector<float> d_out(gpu.get_dmt_size() + 3);
    gpu.execute<uint8_t>(
        std::span<const DeviceSpan<const uint8_t>>(d_groups),
        DeviceSpan<float>(thrust::raw_pointer_cast(d_out.data()),
                          d_out.size())
            .subspan(3, gpu.get_dmt_size()));
    cudaDeviceSynchronize();
    const thrust::host_vector<float> h_out = d_out;
    CHECK(std::vector<float>(h_out.begin() + 3, h_out.end()) == want);
}

TEST_CASE("CohFDMT (gpu) device execute on user streams",
          "[cfdmt_gpu][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto cfg = small_config();
    const CohFDMT a(cfg, test::gpu_exec());
    const CohFDMT b(cfg, test::gpu_exec());
    const auto in = make_block(cfg, a.get_block_nsamps());
    std::vector<float> want(a.get_dmt_size());
    a.execute<uint8_t>(std::span<const uint8_t>(in), want);

    cudaStream_t s1 = nullptr;
    cudaStream_t s2 = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking) ==
            cudaSuccess);
    REQUIRE(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking) ==
            cudaSuccess);
    const thrust::device_vector<uint8_t> d_in(in.begin(), in.end());
    const DeviceSpan<const uint8_t> din(thrust::raw_pointer_cast(d_in.data()),
                                        d_in.size());
    const auto n = a.get_dmt_size();
    thrust::device_vector<float> d_out(4 * n);
    const DeviceSpan<float> dout(thrust::raw_pointer_cast(d_out.data()),
                                 d_out.size());
    // Two engines at once on two streams, then one engine on both streams
    // back to back (its buffers are shared: the second call must wait).
    a.execute<uint8_t>(din, dout.subspan(0, n), Stream{s1});
    b.execute<uint8_t>(din, dout.subspan(n, n), Stream{s2});
    a.execute<uint8_t>(din, dout.subspan(2 * n, n), Stream{s2});
    a.execute<uint8_t>(din, dout.subspan(3 * n, n), Stream{s1});
    REQUIRE(cudaStreamSynchronize(s1) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s2) == cudaSuccess);
    const thrust::host_vector<float> h = d_out;
    for (SizeType i = 0; i < 4; ++i) {
        CAPTURE(i);
        CHECK(std::vector<float>(h.begin() + static_cast<std::ptrdiff_t>(i * n),
                                 h.begin() +
                                     static_cast<std::ptrdiff_t>((i + 1) * n)) ==
              want);
    }
    // A host call right after device calls on another stream.
    std::vector<float> host_again(n);
    a.execute<uint8_t>(din, dout.subspan(0, n), Stream{s1});
    a.execute<uint8_t>(std::span<const uint8_t>(in), host_again);
    CHECK(host_again == want);
    REQUIRE(cudaStreamSynchronize(s1) == cudaSuccess);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

// --- 64-bit indexing: ~40 GB of device memory, hidden; run it with
// dmt_tests "[cfdmt_gpu_large]" ---

TEST_CASE("CohFDMT (gpu) spectrum beyond 2^31 elements",
          "[.][cfdmt_gpu_large]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    // 32 x 1 MHz, t_p 4 us, DM 0-0.2, a block of ~1.1 * 2^31 spectrum
    // elements (~19 GB of spectrum, a 4.7 GB input block).
    CohFDMTConfig cfg{.f_center = 1400.0F,
                      .bw_sub   = 1.0F,
                      .nsub     = 32,
                      .t_p      = 4.0E-6F,
                      .dm_min   = 0.0F,
                      .dm_max   = 0.2F,
                      .format   = BasebandFormat{.order = "FTPRI"}};
    cfg.block_nsamps = ((SizeType{9} << 28U) / (2 * cfg.nsub));
    const plans::CohFDMTPlan plan(cfg);
    const auto spec_elems =
        SizeType{2} * plan.get_nsub() * plan.get_nfft() * plan.get_nbin();
    INFO(plan.summary());
    REQUIRE(spec_elems > (SizeType{1} << 31U));
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    REQUIRE(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess);
    const double need = static_cast<double>(plan.get_memory_estimate().total() +
                                            plan.get_input_size(0));
    if (need > 0.95 * static_cast<double>(free_bytes)) {
        SKIP("needs " << need / 1.0737e9 << " GB of device memory");
    }
    // Raw impulse (DM 0) in every subband near the end of the block.
    const auto nout  = plan.get_output_nsamps();
    const auto n_p   = plan.get_n_p();
    const SizeType j = nout - 10;
    const SizeType s0 = static_cast<SizeType>(std::llround(
                            plan.get_output_time_offset() / plan.get_tbin())) +
                        (j * n_p);
    std::vector<uint8_t> in(plan.get_input_size(0), 0);
    const SizeType per_sub = 4 * plan.get_block_nsamps(); // FTPRI int8
    for (SizeType s = 0; s < cfg.nsub; ++s) {
        in[(s * per_sub) + (4 * s0)]     = 100; // pol 0 real
        in[(s * per_sub) + (4 * s0) + 2] = 100; // pol 1 real
    }
    cfg.normalize = false;
    const CohFDMT gpu(cfg, test::gpu_exec());
    std::vector<float> out(gpu.get_dmt_size());
    gpu.execute<uint8_t>(std::span<const uint8_t>(in), out);
    const auto it  = std::ranges::max_element(out);
    const auto idx = static_cast<SizeType>(std::distance(out.begin(), it));
    const auto& grid = plan.get_dm_grid_final();
    CHECK(std::abs(grid[idx / nout]) <= std::abs(grid[1] - grid[0]));
    CHECK(std::abs(static_cast<double>(idx % nout) - static_cast<double>(j)) <=
          1.0);
}

} // namespace dmt
