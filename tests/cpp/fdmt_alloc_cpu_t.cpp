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
#include "dmt/bit_pack_utils.hpp"

// FDMTCPU allocates all working memory in its constructor: execute() and the
// stepper must not allocate. Checked by replacing the global operator new of
// this test binary with a counting one that is only armed around the calls
// under test.

namespace {

std::atomic<bool> g_counting{false};
std::atomic<std::size_t> g_allocations{0};

} // namespace

void* operator new(std::size_t size) {
    if (g_counting.load(std::memory_order_relaxed)) {
        g_allocations.fetch_add(1, std::memory_order_relaxed);
    }
    if (void* p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t /*size*/) noexcept { std::free(p); }

namespace dmt {

using algorithms::FDMTCPU;
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

TEST_CASE("FDMTCPU execute and stepper do not allocate", "[fdmt_cpu][cpu]") {
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
                    FDMTCPU fdmt(1000.0F, 1500.0F, nchans, nsamps, 0.001F, 32,
                                 -16, 1, true, mode, false, nthreads, nbeams,
                                 fuse);
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
