#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <string>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/bit_pack_utils.hpp"

// FDMT allocates all working memory in its constructor: execute() and the
// stepper must not allocate. Checked by replacing the global operator new of
// this test binary with a counting one that is only armed around the calls
// under test.

namespace {

std::atomic<bool> g_counting{false};
std::atomic<std::size_t> g_allocations{0};

} // namespace

void* allocate_test_heap(std::size_t size) {
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new(std::size_t size) {
    if (g_counting.load(std::memory_order_relaxed)) {
        g_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = allocate_test_heap(size)) {
        return p;
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return allocate_test_heap(size);
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return allocate_test_heap(size);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t /*size*/) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t /*size*/) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}

namespace dmt {

using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;

namespace {

// Allocations made by `f` (armed only while it runs).
template <typename F> std::size_t count_allocations(F&& f) {
    g_allocations.store(0);
    g_counting.store(true);
    f();
    g_counting.store(false);
    return g_allocations.load();
}

} // namespace

TEST_CASE("FDMT execute and stepper do not allocate", "[fdmt_cpu][cpu]") {
    const SizeType nchans = 64;
    const SizeType nsamps = 256;
    const SizeType nbeams = 2;
    const SizeType nbits  = 2;
    std::vector<float> wf(nbeams * nchans * nsamps, 1.0F);
    std::vector<uint8_t> packed(
        nbeams * nchans * bit_pack_utils::packed_row_bytes(nsamps, nbits),
        0x5A);
    // Mixed-sign dt range exercises the running box rows of level 0.
    for (const std::string mode : {"full", "roll", "valid"}) {
        for (const SizeType fuse : {SizeType{0}, SizeType{3}, kFDMTAutoFuse}) {
            for (const int nthreads : {1, 4}) {
                DYNAMIC_SECTION("mode=" << mode << " fuse=" << fuse
                                        << " nthreads=" << nthreads) {
                    FDMT fdmt(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 32, -16,
                              1, true, mode, Exec::cpu(nthreads), nbeams, fuse);
                    std::vector<float> dmt(nbeams *
                                           fdmt.get_plan().get_buffer_size());
                    // Warm-up: starts the OpenMP thread pool.
                    fdmt.execute(wf, dmt);
                    const auto n_float =
                        count_allocations([&] { fdmt.execute(wf, dmt); });
                    const auto n_packed  = count_allocations([&] {
                        fdmt.execute(std::span<const uint8_t>(packed), nbits,
                                     dmt);
                    });
                    const auto n_stepper = count_allocations([&] {
                        fdmt.reset(wf, dmt);
                        fdmt.advance(2);
                        fdmt.finalize();
                    });
                    CHECK(n_float == 0);
                    CHECK(n_packed == 0);
                    CHECK(n_stepper == 0);
                }
            }
        }
    }
}

} // namespace dmt

namespace dmt {
TEST_CASE("FDMTBoxcar warmed execution and flush do not allocate",
          "[boxcar][cpu]") {
    plans::FDMTPlan plan(700, 1200, 7, 256, 0.001F, 48, -32, 1, "valid", 2);
    BoxcarConfig cfg;
    cfg.max_boxcar     = 31;
    cfg.wtsp           = 1;
    cfg.detect_thresh  = 1;
    cfg.max_candidates = 7;
    algorithms::FDMTBoxcar search(plan, cfg, true, Exec::cpu(3), 1,
                                  algorithms::kFDMTAutoFuse, 64);
    std::vector<float> input(7 * 256, 3.0F);
    (void)search.execute(input);
    CHECK(count_allocations([&] { (void)search.execute(input); }) == 0);
    CHECK(count_allocations([&] { (void)search.flush(); }) == 0);
    CHECK(count_allocations([&] { search.reset_history(); }) == 0);
}
} // namespace dmt

namespace dmt {
TEST_CASE("Packed FDMTBoxcar warmed execution does not allocate",
          "[boxcar][packed][cpu]") {
    plans::FDMTPlan plan(700, 1200, 7, 139, 0.001F, 97, -73, 1, "valid", 2);
    BoxcarConfig cfg;
    cfg.input_nbits = 8;
    cfg.input_mean  = 127;
    cfg.noise_sigma = 2;
    cfg.max_boxcar  = 4096;
    algorithms::FDMTBoxcar search(plan, cfg, true, Exec::cpu(3), 1,
                                  algorithms::kFDMTAutoFuse, 64);
    std::vector<uint8_t> input(7 * 139, 130);
    (void)search.execute(input);
    CHECK(count_allocations([&] { (void)search.execute(input); }) == 0);
    CHECK(count_allocations([&] { (void)search.flush(); }) == 0);
    CHECK(count_allocations([&] { search.reset_history(); }) == 0);
}
} // namespace dmt
