// Exhaustive GPU vs CPU parity for the DDMT GPU engine: every input width,
// the tiled (wide / narrow DM tiles) and direct kernels, kill masks,
// multiple beams, and streaming in uneven chunks through the device-span,
// host-span and time-major entry points. Both engines sum each output in
// ascending channel order from zero, so float results must match bitwise
// and integer results exactly.

#include <bit>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <thrust/device_vector.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "test_helpers.hpp"

namespace dmt {
namespace {

using algorithms::DDMT;
using plans::DDMTPlan;

template <typename F> void with_nbits(SizeType nbits, F&& f) {
    switch (nbits) {
    case 1:
        f.template operator()<1>();
        break;
    case 2:
        f.template operator()<2>();
        break;
    case 4:
        f.template operator()<4>();
        break;
    case 8:
        f.template operator()<8>();
        break;
    case 16:
        f.template operator()<16>();
        break;
    default:
        FAIL("unsupported nbits");
    }
}

/// Channel-major input of `rows` rows x `nsamps` samples at `nbits`
/// (float values when nbits == 32, packed bytes otherwise).
struct Input {
    SizeType nbits{32};
    SizeType rows{0};
    SizeType nsamps{0};
    std::vector<float> f;
    std::vector<uint8_t> p;

    static Input
    random(SizeType nbits, SizeType rows, SizeType nsamps, uint32_t seed) {
        Input in{nbits, rows, nsamps, {}, {}};
        std::mt19937 rng(seed);
        if (nbits == 32) {
            std::uniform_real_distribution<float> dist(-3.0F, 7.0F);
            in.f.resize(rows * nsamps);
            for (auto& v : in.f) {
                v = dist(rng);
            }
        } else {
            in.p.resize(rows * bit_pack_utils::packed_row_bytes(nsamps, nbits));
            for (auto& v : in.p) {
                v = static_cast<uint8_t>(rng());
            }
        }
        return in;
    }

    [[nodiscard]] SizeType row_bytes() const {
        return bit_pack_utils::packed_row_bytes(nsamps, nbits);
    }

    /// Samples [n0, n1) of every row as a standalone input.
    [[nodiscard]] Input slice(SizeType n0, SizeType n1) const {
        Input out{nbits, rows, n1 - n0, {}, {}};
        if (nbits == 32) {
            out.f.resize(rows * (n1 - n0));
            for (SizeType r = 0; r < rows; ++r) {
                std::copy_n(&f[(r * nsamps) + n0], n1 - n0,
                            &out.f[r * (n1 - n0)]);
            }
            return out;
        }
        out.p.assign(rows * out.row_bytes(), 0);
        with_nbits(nbits, [&]<unsigned NBITS>() {
            for (SizeType r = 0; r < rows; ++r) {
                bit_pack_utils::copy_packed_samples<NBITS>(
                    &p[r * row_bytes()], n0, &out.p[r * out.row_bytes()], 0,
                    n1 - n0);
            }
        });
        return out;
    }

    /// Time-major (nbeams, nsamps, nchans) packed layout of this input.
    [[nodiscard]] std::vector<uint8_t> time_major(SizeType nbeams) const {
        const auto nchans     = rows / nbeams;
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        std::vector<uint8_t> tm(nbeams * nsamps * samp_bytes, 0);
        with_nbits(nbits, [&]<unsigned NBITS>() {
            for (SizeType b = 0; b < nbeams; ++b) {
                for (SizeType c = 0; c < nchans; ++c) {
                    const auto* row = &p[((b * nchans) + c) * row_bytes()];
                    for (SizeType s = 0; s < nsamps; ++s) {
                        bit_pack_utils::write_packed_sample<NBITS>(
                            &tm[((b * nsamps) + s) * samp_bytes], c,
                            bit_pack_utils::read_packed_sample<NBITS>(row, s));
                    }
                }
            }
        });
        return tm;
    }
};

/// Output of one execute() call, float or int32 by nbits.
struct Output {
    std::vector<float> f;
    std::vector<int32_t> i;
};

Output run_host(DDMT& ddmt, const Input& in) {
    Output out;
    const auto n = ddmt.get_nbeams() * ddmt.get_plan().get_dm_arr().size() *
                   ddmt.get_output_nsamps(in.nsamps);
    if (in.nbits == 32) {
        out.f.resize(n);
        ddmt.execute(in.f, out.f);
    } else {
        out.i.resize(n);
        ddmt.execute(in.p, in.nsamps, out.i);
    }
    return out;
}

Output run_device(DDMT& ddmt, const Input& in) {
    Output out;
    const auto n = ddmt.get_nbeams() * ddmt.get_plan().get_dm_arr().size() *
                   ddmt.get_output_nsamps(in.nsamps);
    if (in.nbits == 32) {
        const thrust::device_vector<float> d_in(in.f.begin(), in.f.end());
        thrust::device_vector<float> d_out(n);
        ddmt.execute(DeviceSpan<const float>(
                         thrust::raw_pointer_cast(d_in.data()), d_in.size()),
                     DeviceSpan<float>(thrust::raw_pointer_cast(d_out.data()),
                                       d_out.size()));
        out.f.resize(n);
        thrust::copy(d_out.begin(), d_out.end(), out.f.begin());
    } else {
        const thrust::device_vector<uint8_t> d_in(in.p.begin(), in.p.end());
        thrust::device_vector<int32_t> d_out(n);
        ddmt.execute(DeviceSpan<const uint8_t>(
                         thrust::raw_pointer_cast(d_in.data()), d_in.size()),
                     in.nsamps,
                     DeviceSpan<int32_t>(thrust::raw_pointer_cast(d_out.data()),
                                         d_out.size()));
        out.i.resize(n);
        thrust::copy(d_out.begin(), d_out.end(), out.i.begin());
    }
    return out;
}

Output run_time_major(DDMT& ddmt, const Input& in, bool on_device) {
    Output out;
    const auto n  = ddmt.get_nbeams() * ddmt.get_plan().get_dm_arr().size() *
                    ddmt.get_output_nsamps(in.nsamps);
    const auto tm = in.time_major(ddmt.get_nbeams());
    out.i.resize(n);
    if (on_device) {
        const thrust::device_vector<uint8_t> d_in(tm.begin(), tm.end());
        thrust::device_vector<int32_t> d_out(n);
        ddmt.execute_time_major(
            DeviceSpan<const uint8_t>(thrust::raw_pointer_cast(d_in.data()),
                                      d_in.size()),
            in.nsamps,
            DeviceSpan<int32_t>(thrust::raw_pointer_cast(d_out.data()),
                                d_out.size()));
        thrust::copy(d_out.begin(), d_out.end(), out.i.begin());
    } else {
        ddmt.execute_time_major(tm, in.nsamps, out.i);
    }
    return out;
}

/// Appends a chunk's (nbeams, ndm, n) output to the growing (nbeams, ndm,
/// total) stream output, one row at a time.
template <typename T>
void append_rows(std::vector<std::vector<T>>& rows_acc,
                 const std::vector<T>& chunk) {
    const auto nrows = rows_acc.size();
    const auto n     = chunk.size() / nrows;
    for (SizeType r = 0; r < nrows; ++r) {
        rows_acc[r].insert(rows_acc[r].end(), chunk.begin() + (r * n),
                           chunk.begin() + ((r + 1) * n));
    }
}

template <typename T>
std::vector<T> flatten(const std::vector<std::vector<T>>& rows) {
    std::vector<T> out;
    for (const auto& r : rows) {
        out.insert(out.end(), r.begin(), r.end());
    }
    return out;
}

void require_same(const Output& gpu, const Output& cpu) {
    REQUIRE(gpu.f.size() == cpu.f.size());
    REQUIRE(gpu.i.size() == cpu.i.size());
    if (!cpu.f.empty()) {
        // Bitwise: identical summation order on both backends.
        REQUIRE(std::equal(
            gpu.f.begin(), gpu.f.end(), cpu.f.begin(), [](float a, float b) {
                return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b);
            }));
    }
    if (!cpu.i.empty()) {
        REQUIRE_THAT(gpu.i, Catch::Matchers::Equals(cpu.i));
    }
}

struct Scenario {
    const char* name;
    std::vector<float> dms;
    SizeType nchans;
    SizeType nbeams;
    bool mask;
};

std::vector<Scenario> scenarios() {
    std::vector<float> many(101);
    std::iota(many.begin(), many.end(), 0.0F); // wide DM tiles
    std::vector<float> few    = {0.0F, 3.0F, 7.5F, 12.0F, 30.0F}; // narrow
    std::vector<float> sparse = {0.0F, 4000.0F}; // direct fallback
    return {
        {"wide tiles", many, 64, 1, false},
        {"wide tiles, masked, 2 beams", many, 64, 2, true},
        {"narrow tiles", few, 40, 1, true},
        {"direct fallback", sparse, 8, 1, false},
    };
}

std::vector<uint8_t> make_mask(SizeType nchans) {
    std::vector<uint8_t> mask(nchans, 1);
    for (SizeType c = 0; c < nchans; c += 3) {
        mask[c] = 0;
    }
    mask[nchans - 1] = 0;
    return mask;
}

} // namespace

TEST_CASE("parity: DDMT (gpu) tiled/direct kernels match CPU bitwise, "
          "monolithic and streaming",
          "[ddmt][gpu][parity][streaming]") {
    for (const auto& sc : scenarios()) {
        for (const SizeType nbits : {32, 16, 8, 4, 2, 1}) {
            DYNAMIC_SECTION(sc.name << ", nbits = " << nbits) {
                const auto mask =
                    sc.mask ? make_mask(sc.nchans) : std::vector<uint8_t>{};
                const DDMTPlan plan(test::kFMin, test::kFMax, sc.nchans,
                                    test::kTsamp, sc.dms, nbits, mask);
                const auto max_delay =
                    *std::ranges::max_element(plan.get_container().delay_table);
                const auto nsamps = max_delay + 1001;
                const auto in     = Input::random(nbits, sc.nbeams * sc.nchans,
                                                  nsamps, 7 + nbits);

                DDMT cpu(plan, Exec::cpu(4), sc.nbeams);
                const auto ref = run_host(cpu, in);

                // Monolithic: host and device spans.
                DDMT gpu(plan, test::gpu_exec(), sc.nbeams);
                require_same(run_host(gpu, in), ref);
                gpu.reset_history();
                require_same(run_device(gpu, in), ref);

                // Streaming in uneven chunks (some shorter than the max
                // delay, odd lengths), alternating host and device calls.
                gpu.reset_history();
                gpu.set_gulp_size(40); // many host-path chunks per call
                const auto nrows = sc.nbeams * sc.dms.size();
                std::vector<std::vector<float>> acc_f(nrows);
                std::vector<std::vector<int32_t>> acc_i(nrows);
                const SizeType bounds[] = {0,
                                           1,
                                           8,
                                           max_delay / 2 + 3,
                                           max_delay + 9,
                                           max_delay + 400,
                                           nsamps};
                for (SizeType k = 0; k + 1 < std::size(bounds); ++k) {
                    const auto chunk = in.slice(bounds[k], bounds[k + 1]);
                    const auto out   = (k % 2 == 0) ? run_device(gpu, chunk)
                                                    : run_host(gpu, chunk);
                    if (nbits == 32) {
                        append_rows(acc_f, out.f);
                    } else {
                        append_rows(acc_i, out.i);
                    }
                }
                Output streamed;
                streamed.f = flatten(acc_f);
                streamed.i = flatten(acc_i);
                require_same(streamed, ref);
            }
        }
    }
}

TEST_CASE("parity: DDMT (gpu) time-major streaming matches CPU channel-major",
          "[ddmt][gpu][parity][streaming]") {
    std::vector<float> dms(70);
    std::iota(dms.begin(), dms.end(), 0.0F);
    const SizeType nchans = 48;
    const SizeType nbeams = 2;
    for (const SizeType nbits : {16, 8, 4, 2, 1}) {
        for (const bool on_device : {false, true}) {
            DYNAMIC_SECTION("nbits = " << nbits
                                       << (on_device ? ", device" : ", host")) {
                const DDMTPlan plan(test::kFMin, test::kFMax, nchans,
                                    test::kTsamp, dms, nbits);
                const auto max_delay =
                    *std::ranges::max_element(plan.get_container().delay_table);
                const auto nsamps = max_delay + 777;
                const auto in =
                    Input::random(nbits, nbeams * nchans, nsamps, 99 + nbits);
                DDMT cpu(plan, Exec::cpu(4), nbeams);
                const auto ref = run_host(cpu, in);

                DDMT gpu(plan, test::gpu_exec(), nbeams);
                gpu.set_gulp_size(64);
                std::vector<std::vector<int32_t>> acc(nbeams * dms.size());
                const SizeType bounds[] = {0, 5, max_delay + 1, 300 + max_delay,
                                           nsamps};
                for (SizeType k = 0; k + 1 < std::size(bounds); ++k) {
                    const auto out = run_time_major(
                        gpu, in.slice(bounds[k], bounds[k + 1]), on_device);
                    append_rows(acc, out.i);
                }
                Output streamed;
                streamed.i = flatten(acc);
                require_same(streamed, ref);
            }
        }
    }
}

TEST_CASE("parity: DDMT (gpu) device history save/load resumes a stream",
          "[ddmt][gpu][streaming]") {
    std::vector<float> dms(40);
    std::iota(dms.begin(), dms.end(), 0.0F);
    for (const SizeType nbits : {32, 8, 2}) {
        DYNAMIC_SECTION("nbits = " << nbits) {
            const DDMTPlan plan(test::kFMin, test::kFMax, 32, test::kTsamp, dms,
                                nbits);
            const auto max_delay =
                *std::ranges::max_element(plan.get_container().delay_table);
            const auto in =
                Input::random(nbits, 32, (2 * max_delay) + 300, 5 + nbits);
            const auto part1 = in.slice(0, max_delay + 150);
            const auto part2 = in.slice(max_delay + 150, in.nsamps);

            DDMT a(plan, test::gpu_exec());
            run_device(a, part1);
            const auto size = a.history_state_size();
            thrust::device_vector<uint8_t> d_state(
                nbits == 32 ? size * sizeof(float) : size);
            if (nbits == 32) {
                a.save_history(DeviceSpan<float>(
                    reinterpret_cast<float*>(
                        thrust::raw_pointer_cast(d_state.data())),
                    size));
            } else {
                a.save_history(DeviceSpan<uint8_t>(
                    thrust::raw_pointer_cast(d_state.data()), size));
            }
            const auto expected = run_device(a, part2);

            DDMT b(plan, test::gpu_exec());
            if (nbits == 32) {
                b.load_history(DeviceSpan<const float>(
                    reinterpret_cast<const float*>(
                        thrust::raw_pointer_cast(d_state.data())),
                    size));
            } else {
                b.load_history(DeviceSpan<const uint8_t>(
                    thrust::raw_pointer_cast(d_state.data()), size));
            }
            require_same(run_host(b, part2), expected);

            // Host snapshot of the same state round-trips too.
            DDMT c(plan, test::gpu_exec());
            if (nbits == 32) {
                std::vector<float> h(size);
                REQUIRE(cudaMemcpy(h.data(),
                                   thrust::raw_pointer_cast(d_state.data()),
                                   size * sizeof(float),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                c.load_history(h);
                std::vector<float> back(size);
                c.save_history(back);
                REQUIRE(back == h);
            } else {
                std::vector<uint8_t> h(d_state.begin(), d_state.end());
                c.load_history(h);
                std::vector<uint8_t> back(size);
                c.save_history(back);
                REQUIRE(back == h);
            }
            require_same(run_device(c, part2), expected);
        }
    }
}

} // namespace dmt
