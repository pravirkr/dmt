#include <algorithm>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/fdmt.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/backend.hpp"
#include "test_helpers.hpp"

// The GPU backend of this build (CUDA or HIP) against the CPU,
// through the host-memory API only, so the one source covers all of them.
// Integer-valued data keeps every sum exact whatever each backend's float
// flags, so those comparisons are bitwise.

namespace dmt {

using algorithms::DDMT;
using algorithms::FDMT;
using algorithms::kFDMTAutoFuse;

namespace {

// GPU backends of this build that can construct an instance here (a build
// may contain a backend whose device is absent, e.g. a CI VM).
std::vector<Backend> usable_gpu_backends() {
    std::vector<Backend> out;
    for (const auto b : available_backends()) {
        if (b == Backend::kCPU) {
            continue;
        }
        try {
            FDMT probe(1000.0F, 1500.0F, 16, 64, 0.001F, 8, 0, 1, true, "valid",
                       Exec{.backend = b, .nthreads = 1, .device = 0});
            out.push_back(b);
        } catch (const std::exception& e) {
            WARN("skipping " << to_string(b) << ": " << e.what());
        }
    }
    return out;
}

Exec exec_of(Backend b) { return {.backend = b, .nthreads = 1, .device = 0}; }

std::vector<uint32_t> random_ints(SizeType n, uint32_t max, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, max);
    std::vector<uint32_t> v(n);
    for (auto& x : v) {
        x = dis(gen);
    }
    return v;
}

std::vector<float> as_floats(const std::vector<uint32_t>& v) {
    return {v.begin(), v.end()};
}

// Rows of `nsamps` samples packed at `nbits`.
std::vector<uint8_t> pack_rows(const std::vector<uint32_t>& v,
                               SizeType rows,
                               SizeType nsamps,
                               SizeType nbits) {
    const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
    std::vector<uint8_t> out(rows * row_bytes, 0);
    for (SizeType r = 0; r < rows; ++r) {
        auto* row = out.data() + (r * row_bytes);
        for (SizeType s = 0; s < nsamps; ++s) {
            const auto val = v[(r * nsamps) + s];
            switch (nbits) {
            case 1:
                bit_pack_utils::write_packed_sample<1>(row, s, val);
                break;
            case 2:
                bit_pack_utils::write_packed_sample<2>(row, s, val);
                break;
            case 4:
                bit_pack_utils::write_packed_sample<4>(row, s, val);
                break;
            case 8:
                bit_pack_utils::write_packed_sample<8>(row, s, val);
                break;
            default:
                bit_pack_utils::write_packed_sample<16>(row, s, val);
                break;
            }
        }
    }
    return out;
}

// Each beam's leading get_dmt_size() values (the rest is scratch).
std::vector<float> dmt_result(const FDMT& f, const std::vector<float>& buf) {
    const auto& plan = f.get_plan();
    std::vector<float> out;
    for (SizeType b = 0; b < f.get_nbeams(); ++b) {
        const auto* p = buf.data() + (b * plan.get_buffer_size());
        out.insert(out.end(), p, p + plan.get_dmt_size());
    }
    return out;
}

constexpr SizeType kNchans = 64;
constexpr SizeType kNsamps = 256;
constexpr IndexType kDtMax = 96;

} // namespace

TEST_CASE("parity: FDMT on every GPU backend matches the CPU bitwise",
          "[fdmt][gpu][parity]") {
    const auto backends = usable_gpu_backends();
    if (backends.empty()) {
        return;
    }
    for (const auto backend : backends) {
        for (const std::string mode : {"full", "roll", "valid"}) {
            for (const bool smear : {true, false}) {
                for (const SizeType nbeams : {1, 3}) {
                    // nbits 0: float input.
                    for (const SizeType nbits : {0, 1, 2, 4, 8, 16}) {
                        for (const SizeType fuse :
                             {kFDMTAutoFuse, SizeType{0}, SizeType{2}}) {
                            for (const bool int_tree : {true, false}) {
                                if (nbits == 0 && !int_tree) {
                                    continue; // int_tree is ignored for float
                                }
                                DYNAMIC_SECTION(to_string(backend)
                                                << " mode=" << mode << " smear="
                                                << smear << " nbeams=" << nbeams
                                                << " nbits=" << nbits
                                                << " fuse=" << fuse
                                                << " int_tree=" << int_tree) {
                                    FDMT cpu(1000.0F, 1500.0F, kNchans, kNsamps,
                                             0.001F, kDtMax, 0, 1, smear, mode,
                                             Exec::cpu(2), nbeams, fuse,
                                             int_tree);
                                    FDMT gpu(1000.0F, 1500.0F, kNchans, kNsamps,
                                             0.001F, kDtMax, 0, 1, smear, mode,
                                             exec_of(backend), nbeams, fuse,
                                             int_tree);
                                    REQUIRE(gpu.backend() == backend);
                                    const auto n =
                                        nbeams *
                                        cpu.get_plan().get_buffer_size();
                                    const auto max_val =
                                        nbits == 0
                                            ? 7U
                                            : bit_pack_utils::max_sample_value(
                                                  nbits);
                                    // Three consecutive blocks: valid mode
                                    // streams history across them.
                                    for (unsigned blk = 0; blk < 3; ++blk) {
                                        const auto ints = random_ints(
                                            nbeams * kNchans * kNsamps, max_val,
                                            1234U + blk);
                                        std::vector<float> out_cpu(n, 0.0F);
                                        std::vector<float> out_gpu(n, -1.0F);
                                        if (nbits == 0) {
                                            const auto wf = as_floats(ints);
                                            cpu.execute(wf, out_cpu);
                                            gpu.execute(wf, out_gpu);
                                        } else {
                                            const auto wf = pack_rows(
                                                ints, nbeams * kNchans, kNsamps,
                                                nbits);
                                            cpu.execute(wf, nbits, out_cpu);
                                            gpu.execute(wf, nbits, out_gpu);
                                        }
                                        test::require_exact(
                                            dmt_result(gpu, out_gpu),
                                            dmt_result(cpu, out_cpu));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("parity: FDMT GPU float input matches the CPU closely",
          "[fdmt][gpu][parity]") {
    const auto backends = usable_gpu_backends();
    if (backends.empty()) {
        return;
    }
    std::mt19937 gen(7);
    std::normal_distribution<float> dis(0.0F, 1.0F);
    std::vector<float> wf(kNchans * kNsamps);
    for (auto& x : wf) {
        x = dis(gen);
    }
    for (const auto backend : backends) {
        FDMT cpu(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0, 1, true,
                 "valid", Exec::cpu(1));
        FDMT gpu(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0, 1, true,
                 "valid", exec_of(backend));
        std::vector<float> out_cpu(cpu.get_plan().get_buffer_size());
        std::vector<float> out_gpu(out_cpu.size());
        cpu.execute(wf, out_cpu);
        gpu.execute(wf, out_gpu);
        test::require_approx(dmt_result(gpu, out_gpu), dmt_result(cpu, out_cpu),
                             1.0E-4);
    }
}

TEST_CASE("parity: FDMT GPU host stepper and history match the CPU",
          "[fdmt][gpu][parity]") {
    const auto backends = usable_gpu_backends();
    if (backends.empty()) {
        return;
    }
    for (const auto backend : backends) {
        DYNAMIC_SECTION(to_string(backend)) {
            FDMT cpu(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0, 1,
                     true, "valid", Exec::cpu(1), 1, 0, false);
            FDMT gpu(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0, 1,
                     true, "valid", exec_of(backend), 1, 0, false);
            const auto n   = cpu.get_plan().get_buffer_size();
            const auto wf0 = as_floats(random_ints(kNchans * kNsamps, 5, 1));
            const auto wf1 = as_floats(random_ints(kNchans * kNsamps, 5, 2));
            const auto wf2 = as_floats(random_ints(kNchans * kNsamps, 5, 3));

            // Stepper: every level agrees, then the root.
            std::vector<float> out_cpu(n);
            std::vector<float> out_gpu(n);
            cpu.reset(wf0, out_cpu);
            gpu.reset(wf0, out_gpu);
            while (!cpu.is_finished()) {
                REQUIRE(gpu.current_level() == cpu.current_level());
                const auto lc = cpu.view_level_data();
                const auto lg = gpu.view_level_data();
                REQUIRE(lg.size() == lc.size());
                REQUIRE(std::ranges::equal(lg, lc));
                const auto sub = cpu.num_subbands() / 2;
                REQUIRE(std::ranges::equal(gpu.view_subband(sub).data,
                                           cpu.view_subband(sub).data));
                cpu.advance(1);
                gpu.advance(1);
            }
            cpu.finalize();
            gpu.finalize();
            test::require_exact(dmt_result(gpu, out_gpu),
                                dmt_result(cpu, out_cpu));

            // save_history() into a fresh instance continues the stream.
            cpu.execute(wf1, out_cpu);
            gpu.execute(wf1, out_gpu);
            std::vector<float> hist(gpu.history_state_size());
            REQUIRE_FALSE(hist.empty());
            gpu.save_history(hist);
            FDMT resumed(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0,
                         1, true, "valid", exec_of(backend), 1, 0, false);
            resumed.load_history(hist);
            cpu.execute(wf2, out_cpu);
            std::vector<float> out_resumed(n);
            resumed.execute(wf2, out_resumed);
            test::require_exact(dmt_result(resumed, out_resumed),
                                dmt_result(cpu, out_cpu));

            // reset_history() starts cold again.
            gpu.reset_history();
            FDMT cold(1000.0F, 1500.0F, kNchans, kNsamps, 0.001F, kDtMax, 0, 1,
                      true, "valid", Exec::cpu(1), 1, 0, false);
            std::vector<float> out_cold(n);
            cold.execute(wf2, out_cold);
            gpu.execute(wf2, out_gpu);
            test::require_exact(dmt_result(gpu, out_gpu),
                                dmt_result(cold, out_cold));
        }
    }
}

TEST_CASE("parity: DDMT on every GPU backend matches the CPU bitwise",
          "[ddmt][gpu][parity]") {
    const auto backends = usable_gpu_backends();
    if (backends.empty()) {
        return;
    }
    constexpr SizeType kDdmtNchans = 32;
    for (const auto backend : backends) {
        for (const SizeType nbeams : {1, 2}) {
            for (const SizeType nbits : {32, 1, 2, 4, 8, 16}) {
                for (const bool masked : {false, true}) {
                    DYNAMIC_SECTION(to_string(backend)
                                    << " nbeams=" << nbeams << " nbits="
                                    << nbits << " masked=" << masked) {
                        std::vector<uint8_t> kill(kDdmtNchans, 1);
                        if (masked) {
                            kill[3] = kill[17] = kill[30] = 0;
                        }
                        DDMT cpu(1000.0F, 1500.0F, kDdmtNchans, 0.001F, 60.0F,
                                 2.0F, 0.0F, Exec::cpu(1), nbits, kill, nbeams);
                        DDMT gpu(1000.0F, 1500.0F, kDdmtNchans, 0.001F, 60.0F,
                                 2.0F, 0.0F, exec_of(backend), nbits, kill,
                                 nbeams);
                        const auto max_val =
                            nbits == 32
                                ? 9U
                                : bit_pack_utils::max_sample_value(nbits);
                        const auto dm_count =
                            cpu.get_plan().get_dm_arr().size();
                        // Warm-up blocks shorter than the delay, then normal
                        // ones: the history is joined on every call.
                        for (const SizeType nsamps : {37, 50, 700, 1024, 333}) {
                            const auto ints = random_ints(
                                nbeams * kDdmtNchans * nsamps, max_val,
                                static_cast<unsigned>(nsamps));
                            const auto nout = cpu.get_output_nsamps(nsamps);
                            REQUIRE(gpu.get_output_nsamps(nsamps) == nout);
                            if (nbits == 32) {
                                const auto wf = as_floats(ints);
                                std::vector<float> oc(nbeams * dm_count * nout);
                                std::vector<float> og(oc.size(), -1.0F);
                                cpu.execute(wf, oc);
                                gpu.execute(wf, og);
                                test::require_exact(og, oc);
                            } else {
                                const auto wf = pack_rows(
                                    ints, nbeams * kDdmtNchans, nsamps, nbits);
                                std::vector<int32_t> oc(nbeams * dm_count *
                                                        nout);
                                std::vector<int32_t> og(oc.size(), -1);
                                cpu.execute(wf, nsamps, oc);
                                gpu.execute(wf, nsamps, og);
                                REQUIRE(og == oc);
                            }
                        }
                        // The history round-trips through save/load.
                        if (nbits == 32) {
                            std::vector<float> hc(cpu.history_state_size());
                            std::vector<float> hg(gpu.history_state_size());
                            cpu.save_history(hc);
                            gpu.save_history(hg);
                            REQUIRE(hg == hc);
                            gpu.load_history(hg);
                        } else {
                            std::vector<uint8_t> hc(cpu.history_state_size());
                            std::vector<uint8_t> hg(gpu.history_state_size());
                            cpu.save_history(hc);
                            gpu.save_history(hg);
                            REQUIRE(hg == hc);
                            gpu.load_history(hg);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("parity: DDMT GPU time-major and multi-gulp inputs match the CPU",
          "[ddmt][gpu][parity]") {
    const auto backends = usable_gpu_backends();
    if (backends.empty()) {
        return;
    }
    constexpr SizeType nchans = 16;
    for (const auto backend : backends) {
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            DYNAMIC_SECTION(to_string(backend)
                            << " time-major nbits=" << nbits) {
                DDMT cpu(1000.0F, 1500.0F, nchans, 0.001F, 40.0F, 4.0F, 0.0F,
                         Exec::cpu(1), nbits);
                DDMT gpu(1000.0F, 1500.0F, nchans, 0.001F, 40.0F, 4.0F, 0.0F,
                         exec_of(backend), nbits);
                constexpr SizeType nsamps = 900;
                // (nsamps, nchans): channels of one sample packed together.
                const auto ints =
                    random_ints(nsamps * nchans,
                                bit_pack_utils::max_sample_value(nbits), 5);
                const auto fb       = pack_rows(ints, nsamps, nchans, nbits);
                const auto dm_count = cpu.get_plan().get_dm_arr().size();
                // Stateless: nsamps - max_delay, as for a fresh stream.
                const auto nout = cpu.get_output_nsamps(nsamps);
                std::vector<int32_t> oc(dm_count * nout);
                std::vector<int32_t> og(oc.size(), -1);
                cpu.execute_time_major(fb, nsamps, oc);
                gpu.execute_time_major(fb, nsamps, og);
                REQUIRE(og == oc);
            }
        }
        DYNAMIC_SECTION(to_string(backend) << " multi-gulp float") {
            DDMT cpu(1000.0F, 1500.0F, 8, 0.001F, 20.0F, 5.0F, 0.0F,
                     Exec::cpu(1));
            DDMT gpu(1000.0F, 1500.0F, 8, 0.001F, 20.0F, 5.0F, 0.0F,
                     exec_of(backend));
            constexpr SizeType nsamps = 150000; // > 2 gulps of 65536 outputs
            const auto wf       = as_floats(random_ints(8 * nsamps, 3, 11));
            const auto dm_count = cpu.get_plan().get_dm_arr().size();
            const auto nout     = cpu.get_output_nsamps(nsamps);
            std::vector<float> oc(dm_count * nout);
            std::vector<float> og(oc.size(), -1.0F);
            cpu.execute(wf, oc);
            gpu.execute(wf, og);
            test::require_exact(og, oc);
        }
    }
}

} // namespace dmt
