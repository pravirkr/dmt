#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <thrust/device_vector.h>

#include "dmt/algorithms/ddmt_fft.hpp"

#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_planned_vector.cuh"
#include "dmt/gpu_utils.cuh"

#include "ddmt_fft_estimate_cases.hpp"
#include "test_helpers.hpp"

namespace dmt {

using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;

namespace {

std::vector<float> noise(SizeType n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    double ref  = 0.0;
    for (SizeType i = 0; i < a.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
        ref  = std::max(ref, static_cast<double>(std::abs(b[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

} // namespace

// GPU vs CPU over a stream of blocks (history, beams, kill mask), both
// methods. (Segmentation needs spectra beyond the 1 GiB cap: not exercised.)
TEST_CASE("DDMTFFT GPU matches CPU", "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 3000;
    for (const auto method : {DDMTFFTMethod::kBrute, DDMTFFTMethod::kNUFFT}) {
        CAPTURE(algorithms::to_string(method));
        const DDMTFFTOptions opts{.method = method, .guard = 32};
        std::vector<uint8_t> mask(nchans, 1);
        mask[5] = 0;
        DDMTFFT cpu(1100.0F, 1500.0F, nchans, 1e-3F, 40.0F, 0.4F, 0.0F,
                    Exec::cpu(4), 32, mask, 2, opts);
        DDMTFFT gpu(1100.0F, 1500.0F, nchans, 1e-3F, 40.0F, 0.4F, 0.0F,
                    test::gpu_exec(), 32, mask, 2, opts);
        CHECK(gpu.get_max_delay() == cpu.get_max_delay());
        const auto ndm = cpu.get_plan().get_dm_arr().size();
        for (unsigned blk = 0; blk < 3; ++blk) {
            auto wf = noise(2 * nchans * nsamps, 10 + blk);
            REQUIRE(gpu.get_output_nsamps(nsamps) ==
                    cpu.get_output_nsamps(nsamps));
            const auto n = 2 * ndm * cpu.get_output_nsamps(nsamps);
            std::vector<float> a(n);
            std::vector<float> b(n);
            cpu.execute(wf, a);
            gpu.execute(wf, b);
            CHECK(rel_err(b, a) < 2e-5);
        }
        std::vector<float> ha(cpu.history_state_size());
        std::vector<float> hb(gpu.history_state_size());
        cpu.save_history(ha);
        gpu.save_history(hb);
        CHECK(rel_err(hb, ha) < 1e-7);
    }
}

TEST_CASE("DDMTFFT GPU device memory and packed input",
          "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nchans = 32;
    const SizeType nsamps = 2048;
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> packed(nchans * nsamps);
    for (auto& v : packed) {
        v = static_cast<uint8_t>(byte(rng));
    }
    DDMTFFT cpu(1100.0F, 1500.0F, nchans, 1e-3F, 20.0F, 0.5F, 0.0F,
                Exec::cpu(2), 8);
    DDMTFFT gpu(1100.0F, 1500.0F, nchans, 1e-3F, 20.0F, 0.5F, 0.0F,
                test::gpu_exec(), 8);
    const auto n =
        cpu.get_plan().get_dm_arr().size() * cpu.get_output_nsamps(nsamps);
    std::vector<float> a(n);
    cpu.execute(packed, nsamps, a);

    thrust::device_vector<uint8_t> in_d(packed.begin(), packed.end());
    thrust::device_vector<float> out_d(n, 0.0F);
    gpu.execute(DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(in_d.data()),
                                          in_d.size()),
                nsamps,
                DeviceSpan<float>(thrust::raw_pointer_cast(out_d.data()),
                                  out_d.size()));
    cudaDeviceSynchronize();
    std::vector<float> b(n);
    thrust::copy(out_d.begin(), out_d.end(), b.begin());
    CHECK(rel_err(b, a) < 2e-5);
}

TEST_CASE("DDMTFFT GPU piecewise-uniform grid matches CPU",
          "[ddmt_fft_gpu][gpu][parity]") {
    // Two uniform runs around scattered trials: NUFFT per run + brute force.
    std::vector<float> dms;
    for (int i = 0; i < 40; ++i) {
        dms.push_back(0.5F * static_cast<float>(i));
    }
    for (const float v : {20.3F, 21.7F, 23.0F}) {
        dms.push_back(v);
    }
    for (int i = 0; i < 64; ++i) {
        dms.push_back(24.0F + (0.75F * static_cast<float>(i)));
    }
    const SizeType nchans = 48;
    const SizeType nsamps = 2500;
    DDMTFFT cpu(1100.0F, 1500.0F, nchans, 1e-3F, dms, Exec::cpu(4));
    DDMTFFT gpu(1100.0F, 1500.0F, nchans, 1e-3F, dms, test::gpu_exec());
    CHECK(cpu.method_used() == "piecewise_nufft");
    CHECK(gpu.method_used() == cpu.method_used());
    for (unsigned blk = 0; blk < 2; ++blk) {
        auto wf      = noise(nchans * nsamps, 40 + blk);
        const auto n = dms.size() * cpu.get_output_nsamps(nsamps);
        std::vector<float> a(n);
        std::vector<float> b(n);
        cpu.execute(wf, a);
        gpu.execute(wf, b);
        CHECK(rel_err(b, a) < 2e-5);
    }
}

TEST_CASE("DDMTFFT GPU host packed, time-major and history reset",
          "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nchans = 32;
    const SizeType nsamps = 1500;
    for (const SizeType nbits : {2, 8, 16}) {
        CAPTURE(nbits);
        const auto rb   = (nsamps * nbits + 7) / 8;
        const auto vmax = static_cast<int>((1U << nbits) - 1U);
        std::mt19937 rng(static_cast<unsigned>(nbits));
        std::uniform_int_distribution<int> val(0, vmax);
        std::vector<uint8_t> packed(nchans * rb, 0);
        std::vector<uint32_t> vals(nchans * nsamps);
        for (SizeType c = 0; c < nchans; ++c) {
            for (SizeType t = 0; t < nsamps; ++t) {
                const auto v           = static_cast<uint32_t>(val(rng));
                vals[(c * nsamps) + t] = v;
                const auto bit         = t * nbits;
                for (SizeType q = 0; q < nbits; ++q) {
                    if (((v >> q) & 1U) != 0U) {
                        packed[(c * rb) + ((bit + q) / 8)] |=
                            static_cast<uint8_t>(1U << ((bit + q) % 8));
                    }
                }
            }
        }
        DDMTFFT cpu(1100.0F, 1500.0F, nchans, 1e-3F, 20.0F, 0.5F, 0.0F,
                    Exec::cpu(2), nbits);
        DDMTFFT gpu(1100.0F, 1500.0F, nchans, 1e-3F, 20.0F, 0.5F, 0.0F,
                    test::gpu_exec(), nbits);
        const auto n =
            cpu.get_plan().get_dm_arr().size() * cpu.get_output_nsamps(nsamps);
        std::vector<float> a(n);
        std::vector<float> b(n);
        cpu.execute(packed, nsamps, a);
        gpu.execute(packed, nsamps, b);
        CHECK(rel_err(b, a) < 2e-5);

        // Time-major bytes of the same samples, after a history reset.
        const auto sb = (nchans * nbits + 7) / 8;
        std::vector<uint8_t> tm(nsamps * sb, 0);
        for (SizeType t = 0; t < nsamps; ++t) {
            for (SizeType c = 0; c < nchans; ++c) {
                const auto v   = vals[(c * nsamps) + t];
                const auto bit = c * nbits;
                for (SizeType q = 0; q < nbits; ++q) {
                    if (((v >> q) & 1U) != 0U) {
                        tm[(t * sb) + ((bit + q) / 8)] |=
                            static_cast<uint8_t>(1U << ((bit + q) % 8));
                    }
                }
            }
        }
        gpu.reset_history();
        std::vector<float> c2(n);
        gpu.execute_time_major(tm, nsamps, c2);
        CHECK(rel_err(c2, a) < 2e-5);
    }
}

namespace {

// A device-memory call of `nsamps` zero samples (float or packed).
void device_call(DDMTFFT& eng, SizeType nsamps, SizeType out) {
    const auto& plan = eng.get_plan();
    const auto nbits = plan.get_nbits();
    const auto rows  = eng.get_nbeams() * plan.get_nchans();
    thrust::device_vector<float> dmt(eng.get_nbeams() *
                                     plan.get_dm_arr().size() * out);
    const DeviceSpan<float> out_d(thrust::raw_pointer_cast(dmt.data()),
                                  dmt.size());
    if (nbits == 32) {
        const thrust::device_vector<float> wf(rows * nsamps, 0.0F);
        eng.execute(DeviceSpan<const float>(thrust::raw_pointer_cast(wf.data()),
                                            wf.size()),
                    out_d);
    } else {
        const thrust::device_vector<uint8_t> wf(
            rows * bit_pack_utils::packed_row_bytes(nsamps, nbits), 0);
        eng.execute(DeviceSpan<const uint8_t>(
                        thrust::raw_pointer_cast(wf.data()), wf.size()),
                    nsamps, out_d);
    }
    cudaDeviceSynchronize();
}

} // namespace

TEST_CASE("DDMTFFT (gpu) estimate_memory equals a warm stream",
          "[ddmt_fft_gpu][gpu][budget]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const auto exec = test::gpu_exec();
    for (const auto& c : ddmt_fft_estimate::cases()) {
        ddmt_fft_estimate::check_host(c, exec);
        INFO(c.name << " (device memory)");
        const ddmt_fft_estimate::SegmentCap cap(c.cap);
        const auto e = DDMTFFT::estimate_memory(c.plan, c.nsamps, exec,
                                                c.nbeams, c.options);
        DDMTFFT eng(c.plan, exec, c.nbeams, c.options);
        ddmt_fft_estimate::check_stream(
            eng, c.nsamps, e, e.device_staging,
            [&](SizeType out) { device_call(eng, c.nsamps, out); });
        if (c.plan.get_nbits() == 32) {
            CHECK(e.device_staging == 0);
        }
    }
    // The estimate allocates nothing: a history larger than the device
    // (2^23 beams: two copies of 32 GiB) is sized; a buffer made outside the
    // engine's own deferral scope would defer into this outer list instead.
    gpu_utils::PendingList stray;
    algorithms::DDMTFFTMemoryEstimate e{};
    {
        const gpu_utils::DeferScope outer(&stray);
        const auto big = ddmt_fft_estimate::many_beams(SizeType{1} << 23);
        e = DDMTFFT::estimate_memory(big.plan, big.nsamps, exec, big.nbeams,
                                     big.options);
    }
    CHECK(stray.registered() == 0);
    const auto& props = gpu_utils::device_properties(exec.device);
    if (e.usage.total() <= props.totalGlobalMem) {
        SKIP("device larger than the probe (" << props.totalGlobalMem
                                              << " bytes)");
    }
    CHECK(e.usage.total() > props.totalGlobalMem);
    // Building it fails with a std::bad_alloc (Python MemoryError) naming
    // the buffer that did not fit.
    const auto big = ddmt_fft_estimate::many_beams(SizeType{1} << 23);
    CHECK_THROWS_MATCHES(
        DDMTFFT(big.plan, exec, big.nbeams, big.options), std::bad_alloc,
        Catch::Matchers::MessageMatches(
            Catch::Matchers::ContainsSubstring("out of memory") &&
            Catch::Matchers::ContainsSubstring("estimate_memory()")));
}

} // namespace dmt
