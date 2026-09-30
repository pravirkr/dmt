#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>
#include "dmt/gpu_compat.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_fft.hpp"
#include "dmt/engines.hpp"
#include "test_helpers.hpp"

// Robustness and parity of the GPU Fourier engines against the CPU ones:
// modes, grids, stepper, streams, history, packed input, kill masks,
// segmentation and edge cases.

namespace dmt {

using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;
using algorithms::FDMT;
using algorithms::FDMTFFT;

namespace {

constexpr float kFMin  = 1100.0F;
constexpr float kFMax  = 1500.0F;
constexpr float kTsamp = 0.001F;

std::vector<float> noise(SizeType n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(n);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    double diff = 0.0;
    double ref  = 0.0;
    for (SizeType i = 0; i < a.size(); ++i) {
        diff = std::max(diff, static_cast<double>(std::abs(a[i] - b[i])));
        ref  = std::max(ref, static_cast<double>(std::abs(b[i])));
    }
    return ref > 0.0 ? diff / ref : diff;
}

std::vector<uint8_t> pack_rows(const std::vector<uint32_t>& vals,
                               SizeType rows,
                               SizeType n,
                               SizeType nbits) {
    const auto rb = ((n * nbits) + 7) / 8;
    std::vector<uint8_t> out(rows * rb, 0);
    for (SizeType r = 0; r < rows; ++r) {
        for (SizeType i = 0; i < n; ++i) {
            const auto v   = vals[(r * n) + i];
            const auto bit = i * nbits;
            for (SizeType q = 0; q < nbits; ++q) {
                if (((v >> q) & 1U) != 0U) {
                    out[(r * rb) + ((bit + q) / 8)] |=
                        static_cast<uint8_t>(1U << ((bit + q) % 8));
                }
            }
        }
    }
    return out;
}

template <typename T>
std::vector<T> to_host(const thrust::device_vector<T>& d) {
    std::vector<T> h(d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
}

template <typename T> DeviceSpan<T> dspan(thrust::device_vector<T>& d) {
    return {thrust::raw_pointer_cast(d.data()), d.size()};
}
template <typename T>
DeviceSpan<const T> dcspan(const thrust::device_vector<T>& d) {
    return {thrust::raw_pointer_cast(d.data()), d.size()};
}

struct SegmentCap {
    explicit SegmentCap(SizeType n) {
        algorithms::detail::set_fft_segment_cap(n);
    }
    ~SegmentCap() { algorithms::detail::set_fft_segment_cap(0); }
    SegmentCap(const SegmentCap&)            = delete;
    SegmentCap& operator=(const SegmentCap&) = delete;
    SegmentCap(SegmentCap&&)                 = delete;
    SegmentCap& operator=(SegmentCap&&)      = delete;
};

// A non-blocking stream (no implicit ordering with the legacy stream).
struct NonBlockingStream {
    cudaStream_t s{nullptr};
    NonBlockingStream() {
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    }
    ~NonBlockingStream() { cudaStreamDestroy(s); }
    NonBlockingStream(const NonBlockingStream&)            = delete;
    NonBlockingStream& operator=(const NonBlockingStream&) = delete;
    NonBlockingStream(NonBlockingStream&&)                 = delete;
    NonBlockingStream& operator=(NonBlockingStream&&)      = delete;
    [[nodiscard]] Stream stream() const { return Stream{s}; }
};

} // namespace

// ---------------------------------------------------------------------------
// FDMTFFT
// ---------------------------------------------------------------------------

TEST_CASE("FDMTFFT GPU integer mode reproduces FDMT",
          "[fdmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 64;
    const SizeType ns  = 512;
    for (const auto* mode : {"roll", "valid"}) {
        CAPTURE(mode);
        FDMT fdmt(kFMin, kFMax, nch, ns, kTsamp, 64, 0, 1, true, mode);
        FDMTFFT gpu(kFMin, kFMax, nch, ns, kTsamp, 64, 0, 1, true, mode,
                    test::gpu_exec(), 1, false);
        const auto n = fdmt.get_plan().get_dmt_size();
        std::vector<float> a(fdmt.get_plan().get_buffer_size());
        std::vector<float> b(n);
        for (unsigned blk = 0; blk < 3; ++blk) {
            const auto wf = noise(nch * ns, 50 + blk);
            fdmt.execute(wf, a);
            gpu.execute(wf, b);
            a.resize(n);
            CHECK(max_rel(b, a) < 1e-5);
            a.resize(fdmt.get_plan().get_buffer_size());
        }
    }
}

TEST_CASE("FDMTFFT GPU matches CPU on varied grids and plans",
          "[fdmt_fft_gpu][gpu][parity]") {
    struct Case {
        SizeType nchans;
        IndexType dt_max;
        IndexType dt_min;
        bool box;
        const char* mode;
    };
    // Odd channel counts (copy nodes), negative delays, no box smearing,
    // and a plan big enough for several tree stages.
    const std::vector<Case> cases{
        {24, 40, 0, true, "valid"},   {48, 64, -16, true, "valid"},
        {64, 48, 0, false, "full"},   {37, 32, 0, true, "roll"},
        {512, 256, 0, true, "valid"},
    };
    for (const auto& c : cases) {
        for (const bool frac : {true, false}) {
            CAPTURE(c.nchans, c.dt_max, c.dt_min, c.box, c.mode, frac);
            const SizeType ns = 1024;
            FDMTFFT cpu(kFMin, kFMax, c.nchans, ns, kTsamp, c.dt_max, c.dt_min,
                        1, c.box, c.mode, Exec::cpu(4), 1, frac);
            FDMTFFT gpu(kFMin, kFMax, c.nchans, ns, kTsamp, c.dt_max, c.dt_min,
                        1, c.box, c.mode, test::gpu_exec(), 1, frac);
            const auto n = cpu.get_plan().get_dmt_size();
            std::vector<float> a(n);
            std::vector<float> b(n);
            for (unsigned blk = 0; blk < 2; ++blk) {
                const auto wf = noise(c.nchans * ns, 60 + blk);
                cpu.execute(wf, a);
                gpu.execute(wf, b);
                CHECK(max_rel(b, a) < 2e-5);
            }
        }
    }
    // Custom DM grid.
    const std::vector<float> dms{0.0F, 1.5F, 3.0F, 7.0F, 12.0F, 20.0F};
    FDMTFFT cpu(kFMin, kFMax, 32, 1024, kTsamp, dms, true, "valid");
    FDMTFFT gpu(kFMin, kFMax, 32, 1024, kTsamp, dms, true, "valid",
                test::gpu_exec());
    std::vector<float> a(cpu.get_plan().get_dmt_size());
    std::vector<float> b(a.size());
    const auto wf = noise(32 * 1024, 7);
    cpu.execute(wf, a);
    gpu.execute(wf, b);
    CHECK(max_rel(b, a) < 2e-5);
}

TEST_CASE("FDMTFFT GPU stepper and level views match the CPU",
          "[fdmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    const SizeType ns  = 512;
    for (const auto* mode : {"valid", "full", "roll"}) {
        for (const bool frac : {true, false}) {
            CAPTURE(mode, frac);
            FDMTFFT cpu(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, mode,
                        Exec::cpu(2), 2, frac);
            FDMTFFT gpu(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, mode,
                        test::gpu_exec(), 2, frac);
            FDMTFFT gpu_exec(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true,
                             mode, test::gpu_exec(), 2, frac);
            const auto n = 2 * cpu.get_plan().get_dmt_size();
            for (unsigned blk = 0; blk < 2; ++blk) { // valid: history
                const auto wf = noise(2 * nch * ns, 70 + blk);
                std::vector<float> a(n);
                std::vector<float> b(n);
                std::vector<float> e(n);
                cpu.reset(wf, a);
                gpu.reset(wf, b);
                while (!cpu.is_finished()) {
                    const auto va = cpu.view_level_data();
                    const auto vb = gpu.view_level_data();
                    REQUIRE(va.size() == vb.size());
                    CHECK(max_rel({vb.begin(), vb.end()},
                                  {va.begin(), va.end()}) < 2e-5);
                    const auto sa = cpu.view_subband(0);
                    const auto sb = gpu.view_subband(0);
                    CHECK(sa.ndt == sb.ndt);
                    CHECK(sa.nsamps == sb.nsamps);
                    cpu.advance();
                    gpu.advance();
                }
                cpu.finalize();
                gpu.finalize();
                gpu_exec.execute(wf, e);
                CHECK(max_rel(b, a) < 2e-5);
                CHECK(max_rel(e, b) < 2e-5);
            }
        }
    }
}

TEST_CASE("FDMTFFT GPU streams like one long call",
          "[fdmt_fft_gpu][gpu][parity]") {
    // Fractional valid mode: output lags by the guard; blocks of one stream
    // equal a single long block (to the sinc truncation).
    const SizeType nch  = 32;
    const SizeType blk  = 1024;
    const SizeType nblk = 4;
    const auto long_wf  = noise(nch * blk * nblk, 90);
    FDMTFFT one(kFMin, kFMax, nch, blk * nblk, kTsamp, 40, 0, 1, true, "valid",
                test::gpu_exec());
    FDMTFFT stream(kFMin, kFMax, nch, blk, kTsamp, 40, 0, 1, true, "valid",
                   test::gpu_exec());
    FDMTFFT cpu_stream(kFMin, kFMax, nch, blk, kTsamp, 40, 0, 1, true, "valid",
                       Exec::cpu(2));
    std::vector<float> cpu_out(one.get_plan().get_dmt_ndms() * blk);
    REQUIRE(stream.get_output_latency() == one.get_output_latency());
    const auto ndm = one.get_plan().get_dmt_ndms();
    std::vector<float> ref(ndm * blk * nblk);
    one.execute(long_wf, ref);
    std::vector<float> got(ndm * blk * nblk);
    std::vector<float> out(ndm * blk);
    std::vector<float> wf(nch * blk);
    for (SizeType b = 0; b < nblk; ++b) {
        for (SizeType c = 0; c < nch; ++c) {
            std::copy_n(long_wf.begin() + static_cast<std::ptrdiff_t>(
                                              (c * blk * nblk) + (b * blk)),
                        blk, wf.begin() + static_cast<std::ptrdiff_t>(c * blk));
        }
        stream.execute(wf, out);
        cpu_stream.execute(wf, cpu_out);
        CHECK(max_rel(out, cpu_out) < 2e-5);
        for (SizeType d = 0; d < ndm; ++d) {
            std::copy_n(out.begin() + static_cast<std::ptrdiff_t>(d * blk), blk,
                        got.begin() + static_cast<std::ptrdiff_t>(
                                          (d * blk * nblk) + (b * blk)));
        }
    }
    // Guard-level sinc truncation at the block edges (the same on the CPU).
    CHECK(max_rel(got, ref) < 2e-2);
}

TEST_CASE("FDMTFFT GPU streams, history and stepper interleaving",
          "[fdmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    const SizeType ns  = 512;
    const NonBlockingStream nb;
    FDMTFFT cpu(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "valid",
                Exec::cpu(2));
    FDMTFFT gpu(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "valid",
                test::gpu_exec());
    const auto n = cpu.get_plan().get_dmt_size();
    std::vector<float> a(n);
    // Mixed host and device calls on a non-blocking stream.
    for (unsigned blk = 0; blk < 4; ++blk) {
        const auto wf = noise(nch * ns, 100 + blk);
        cpu.execute(wf, a);
        if (blk % 2 == 0) {
            std::vector<float> b(n);
            gpu.execute(wf, b);
            CHECK(max_rel(b, a) < 2e-5);
        } else {
            thrust::device_vector<float> in(wf.begin(), wf.end());
            thrust::device_vector<float> out(n, 0.0F);
            gpu.execute(dcspan(in), dspan(out), nb.stream());
            cudaStreamSynchronize(nb.s);
            CHECK(max_rel(to_host(out), a) < 2e-5);
        }
    }
    // Device save/load history round trip, then reset_history.
    thrust::device_vector<float> hist(gpu.history_state_size());
    gpu.save_history(dspan(hist), nb.stream());
    FDMTFFT other(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "valid",
                  test::gpu_exec());
    other.load_history(dcspan(hist), nb.stream());
    cudaStreamSynchronize(nb.s);
    const auto wf = noise(nch * ns, 120);
    std::vector<float> x(n);
    std::vector<float> y(n);
    gpu.execute(wf, x);
    other.execute(wf, y);
    CHECK(x == y);
    FDMTFFT fresh(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "valid",
                  test::gpu_exec());
    gpu.reset_history();
    fresh.execute(wf, x);
    gpu.execute(wf, y);
    CHECK(x == y);

    // execute() in the middle of a stepper run leaves the stepper intact.
    std::vector<float> step(n);
    std::vector<float> ref(n);
    std::vector<float> junk(n);
    FDMTFFT s1(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "roll",
               test::gpu_exec());
    FDMTFFT s2(kFMin, kFMax, nch, ns, kTsamp, 40, 0, 1, true, "roll",
               test::gpu_exec());
    s1.execute(wf, ref);
    s2.reset(wf, step);
    s2.advance(2);
    s2.execute(noise(nch * ns, 121), junk);
    s2.finalize();
    CHECK(max_rel(step, ref) < 1e-6);
}

TEST_CASE("FDMTFFT GPU packed, time-major, kill mask and history",
          "[fdmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 64;
    const SizeType ns  = 1024;
    std::vector<uint8_t> mask(nch, 1);
    mask[2]  = 0;
    mask[40] = 0;
    for (const SizeType nbits : {1, 4, 8, 16}) {
        CAPTURE(nbits);
        std::mt19937 rng(static_cast<unsigned>(nbits));
        std::uniform_int_distribution<uint32_t> val(
            0, (nbits == 16) ? 65535U : ((1U << nbits) - 1U));
        std::vector<uint32_t> vals(2 * nch * ns);
        for (auto& v : vals) {
            v = val(rng);
        }
        std::vector<uint32_t> tm(vals.size());
        for (SizeType b = 0; b < 2; ++b) {
            for (SizeType c = 0; c < nch; ++c) {
                for (SizeType t = 0; t < ns; ++t) {
                    tm[(((b * ns) + t) * nch) + c] =
                        vals[(((b * nch) + c) * ns) + t];
                }
            }
        }
        const auto packed    = pack_rows(vals, 2 * nch, ns, nbits);
        const auto tm_packed = pack_rows(tm, 2 * ns, nch, nbits);
        FDMTFFT cpu(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, "valid",
                    Exec::cpu(2), 2, true, mask);
        FDMTFFT gpu(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, "valid",
                    test::gpu_exec(), 2, true, mask);
        FDMTFFT gpu_tm(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, "valid",
                       test::gpu_exec(), 2, true, mask);
        FDMTFFT gpu_dev(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, "valid",
                        test::gpu_exec(), 2, true, mask);
        const auto n = 2 * cpu.get_plan().get_dmt_size();
        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c(n);
        thrust::device_vector<uint8_t> in(packed.begin(), packed.end());
        thrust::device_vector<float> out(n, 0.0F);
        for (int blk = 0; blk < 2; ++blk) {
            cpu.execute(packed, nbits, a);
            gpu.execute(packed, nbits, b);
            gpu_tm.execute_time_major(tm_packed, nbits, c);
            gpu_dev.execute(dcspan(in), nbits, dspan(out));
            cudaDeviceSynchronize();
            CHECK(max_rel(b, a) < 2e-5);
            CHECK(max_rel(c, a) < 2e-5);
            CHECK(max_rel(to_host(out), a) < 2e-5);
        }
        // Histories share one layout across backends.
        std::vector<float> ha(cpu.history_state_size());
        std::vector<float> hb(gpu.history_state_size());
        cpu.save_history(ha);
        gpu.save_history(hb);
        CHECK(ha == hb);
    }
}

TEST_CASE("FDMTFFT GPU forced segments match the CPU",
          "[fdmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    const SizeType ns  = 8192;
    for (const auto* mode : {"valid", "full"}) {
        for (const bool frac : {true, false}) {
            CAPTURE(mode, frac);
            const SegmentCap cap(1024);
            FDMTFFT cpu(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, mode,
                        Exec::cpu(2), 1, frac);
            FDMTFFT gpu(kFMin, kFMax, nch, ns, kTsamp, 48, 0, 1, true, mode,
                        test::gpu_exec(), 1, frac);
            const auto n = cpu.get_plan().get_dmt_size();
            std::vector<float> a(n);
            std::vector<float> b(n);
            for (unsigned blk = 0; blk < 2; ++blk) {
                const auto wf = noise(nch * ns, 130 + blk);
                cpu.execute(wf, a);
                gpu.execute(wf, b);
                CHECK(max_rel(b, a) < 2e-5);
            }
        }
    }
}

TEST_CASE("FDMTFFT GPU rejects bad arguments", "[fdmt_fft_gpu][gpu]") {
    FDMTFFT gpu(kFMin, kFMax, 32, 256, kTsamp, 32, 0, 1, true, "valid",
                test::gpu_exec());
    std::vector<float> out(gpu.get_plan().get_dmt_size());
    std::vector<float> small(10);
    std::vector<uint8_t> bytes(32 * 256);
    CHECK_THROWS_AS(gpu.execute(small, out), std::invalid_argument);
    CHECK_THROWS_AS(gpu.execute(bytes, 3, out), std::invalid_argument);
    CHECK_THROWS_AS(gpu.execute(bytes, 4, out), std::invalid_argument);
    CHECK_THROWS_AS(gpu.advance(), std::logic_error);
    std::vector<float> hist(gpu.history_state_size() + 1);
    CHECK_THROWS_AS(gpu.load_history(hist), std::invalid_argument);
    CHECK_THROWS_AS(FDMTFFT(kFMin, kFMax, 32, 256, kTsamp, 32, 0, 1, true,
                            "valid", test::gpu_exec(), 0),
                    std::invalid_argument);
}

// ---------------------------------------------------------------------------
// DDMTFFT
// ---------------------------------------------------------------------------

TEST_CASE("DDMTFFT GPU edge cases", "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    SECTION("all channels masked") {
        const std::vector<uint8_t> mask(nch, 0);
        DDMTFFT gpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                    test::gpu_exec(), 32, mask);
        const auto ns = SizeType{1000};
        const auto n =
            gpu.get_plan().get_dm_arr().size() * gpu.get_output_nsamps(ns);
        std::vector<float> out(n, 1.0F);
        gpu.execute(noise(nch * ns, 1), out);
        CHECK(std::ranges::all_of(out, [](float v) { return v == 0.0F; }));
    }
    SECTION("short blocks, changing lengths (plan cache eviction)") {
        for (const auto method :
             {DDMTFFTMethod::kBrute, DDMTFFTMethod::kNUFFT}) {
            CAPTURE(algorithms::to_string(method));
            const DDMTFFTOptions opts{.method = method, .guard = 16};
            DDMTFFT cpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                        Exec::cpu(2), 32, {}, 1, opts);
            DDMTFFT gpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                        test::gpu_exec(), 32, {}, 1, opts);
            const auto ndm = cpu.get_plan().get_dm_arr().size();
            unsigned seed  = 200;
            for (const SizeType ns :
                 {10, 30, 700, 1100, 1700, 2600, 3100, 900}) {
                CAPTURE(ns);
                REQUIRE(gpu.get_output_nsamps(ns) == cpu.get_output_nsamps(ns));
                const auto n = ndm * cpu.get_output_nsamps(ns);
                std::vector<float> a(n);
                std::vector<float> b(n);
                const auto wf = noise(nch * ns, seed++);
                cpu.execute(wf, a);
                gpu.execute(wf, b);
                if (n > 0) {
                    CHECK(max_rel(b, a) < 2e-5);
                }
            }
        }
    }
}

TEST_CASE("DDMTFFT GPU NUFFT accuracy follows the tolerance",
          "[ddmt_fft_gpu][gpu]") {
    const SizeType nch = 64;
    const SizeType ns  = 2048;
    const auto wf      = noise(nch * ns, 300);
    DDMTFFT brute(kFMin, kFMax, nch, kTsamp, 40.0F, 0.25F, 0.0F,
                  test::gpu_exec(), 32, {}, 1,
                  DDMTFFTOptions{.method = DDMTFFTMethod::kBrute});
    const auto n =
        brute.get_plan().get_dm_arr().size() * brute.get_output_nsamps(ns);
    std::vector<float> ref(n);
    brute.execute(wf, ref);
    for (const double tol : {1e-7, 1e-5, 1e-3}) {
        CAPTURE(tol);
        DDMTFFT nu(
            kFMin, kFMax, nch, kTsamp, 40.0F, 0.25F, 0.0F, test::gpu_exec(), 32,
            {}, 1,
            DDMTFFTOptions{.method = DDMTFFTMethod::kNUFFT, .tolerance = tol});
        std::vector<float> got(n);
        nu.execute(wf, got);
        // Relative to the output scale (~ sum |a| per bin); the float
        // pipeline floor is ~1e-6.
        CHECK(max_rel(got, ref) < std::max(20.0 * tol, 5e-6));
    }
}

TEST_CASE("DDMTFFT GPU large uniform run and forced segments",
          "[ddmt_fft_gpu][gpu][parity]") {
    // 5000 trials: the GPU transforms the run as sub-runs (shared memory),
    // still reported as one NUFFT.
    const SizeType nch = 16;
    const SizeType ns  = 3000;
    const DDMTFFTOptions opts{.method = DDMTFFTMethod::kNUFFT};
    DDMTFFT cpu(kFMin, kFMax, nch, kTsamp, 50.0F, 0.01F, 0.0F, Exec::cpu(4), 32,
                {}, 1, opts);
    DDMTFFT gpu(kFMin, kFMax, nch, kTsamp, 50.0F, 0.01F, 0.0F, test::gpu_exec(),
                32, {}, 1, opts);
    CHECK(gpu.method_used() == cpu.method_used());
    const auto n =
        cpu.get_plan().get_dm_arr().size() * cpu.get_output_nsamps(ns);
    std::vector<float> a(n);
    std::vector<float> b(n);
    const auto wf = noise(nch * ns, 400);
    cpu.execute(wf, a);
    gpu.execute(wf, b);
    CHECK(max_rel(b, a) < 2e-5);

    const SegmentCap cap(1024);
    DDMTFFT cpu_s(kFMin, kFMax, 32, kTsamp, 20.0F, 0.5F, 0.0F, Exec::cpu(2));
    DDMTFFT gpu_s(kFMin, kFMax, 32, kTsamp, 20.0F, 0.5F, 0.0F,
                  test::gpu_exec());
    const auto ns2 = SizeType{6000};
    const auto n2 =
        cpu_s.get_plan().get_dm_arr().size() * cpu_s.get_output_nsamps(ns2);
    std::vector<float> c(n2);
    std::vector<float> d(n2);
    const auto wf2 = noise(32 * ns2, 401);
    cpu_s.execute(wf2, c);
    gpu_s.execute(wf2, d);
    CHECK(max_rel(d, c) < 2e-5);
}

TEST_CASE("DDMTFFT GPU device input, device history and streams",
          "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    const SizeType ns  = 1500;
    const NonBlockingStream nb;
    DDMTFFT cpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F, Exec::cpu(2), 32,
                {}, 2);
    DDMTFFT gpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F, test::gpu_exec(),
                32, {}, 2);
    const auto ndm = cpu.get_plan().get_dm_arr().size();
    for (unsigned blk = 0; blk < 3; ++blk) {
        const auto wf = noise(2 * nch * ns, 500 + blk);
        const auto n  = 2 * ndm * cpu.get_output_nsamps(ns);
        std::vector<float> a(n);
        cpu.execute(wf, a);
        thrust::device_vector<float> in(wf.begin(), wf.end());
        thrust::device_vector<float> out(n, 0.0F);
        gpu.execute(dcspan(in), dspan(out), nb.stream());
        cudaStreamSynchronize(nb.s);
        CHECK(max_rel(to_host(out), a) < 2e-5);
    }
    thrust::device_vector<float> hist(gpu.history_state_size());
    gpu.save_history(dspan(hist), nb.stream());
    DDMTFFT other(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                  test::gpu_exec(), 32, {}, 2);
    other.load_history(dcspan(hist), nb.stream());
    cudaStreamSynchronize(nb.s);
    const auto wf = noise(2 * nch * ns, 510);
    const auto n  = 2 * ndm * gpu.get_output_nsamps(ns);
    std::vector<float> x(n);
    std::vector<float> y(n);
    gpu.execute(wf, x);
    other.execute(wf, y);
    CHECK(x == y);
}

TEST_CASE("DDMTFFT GPU packed 1 and 4 bits, time-major with beams",
          "[ddmt_fft_gpu][gpu][parity]") {
    const SizeType nch = 32;
    const SizeType ns  = 1200;
    for (const SizeType nbits : {1, 4}) {
        CAPTURE(nbits);
        std::mt19937 rng(static_cast<unsigned>(nbits) + 7);
        std::uniform_int_distribution<uint32_t> val(0, (1U << nbits) - 1U);
        std::vector<uint32_t> vals(2 * nch * ns);
        for (auto& v : vals) {
            v = val(rng);
        }
        std::vector<uint32_t> tm(vals.size());
        for (SizeType b = 0; b < 2; ++b) {
            for (SizeType c = 0; c < nch; ++c) {
                for (SizeType t = 0; t < ns; ++t) {
                    tm[(((b * ns) + t) * nch) + c] =
                        vals[(((b * nch) + c) * ns) + t];
                }
            }
        }
        const auto packed    = pack_rows(vals, 2 * nch, ns, nbits);
        const auto tm_packed = pack_rows(tm, 2 * ns, nch, nbits);
        DDMTFFT cpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F, Exec::cpu(2),
                    nbits, {}, 2);
        DDMTFFT gpu(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                    test::gpu_exec(), nbits, {}, 2);
        DDMTFFT gpu_tm(kFMin, kFMax, nch, kTsamp, 20.0F, 0.5F, 0.0F,
                       test::gpu_exec(), nbits, {}, 2);
        const auto n =
            2 * cpu.get_plan().get_dm_arr().size() * cpu.get_output_nsamps(ns);
        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c(n);
        cpu.execute(packed, ns, a);
        gpu.execute(packed, ns, b);
        gpu_tm.execute_time_major(tm_packed, ns, c);
        CHECK(max_rel(b, a) < 2e-5);
        CHECK(max_rel(c, a) < 2e-5);
    }
}

} // namespace dmt
