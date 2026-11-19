// Memory ground rules 3 and 4 on the GPU: every budget from the minimum up
// builds, and a subband search's configuration ranks in the ordered list
// (detail::gpu_search_list) never get worse as the budget grows. The
// chooser's prediction against building every entry: gpu_search_list_t.cpp.

#include <bit>
#include <optional>

#include <catch2/catch_test_macros.hpp>

#include "dmt/gpu_search_list.hpp"

#include "budget_sweep.hpp"
#include "test_helpers.hpp"

using namespace dmt;
using algorithms::FDMTBoxcarMemoryEstimate;
using dmt::BoxcarConfig;
using Path = algorithms::detail::FDMTEngineConfig::GPUPath;

namespace {
test::SweepRank
gpu_rank(const plans::FDMTPlan& plan, const BoxcarConfig& cfg, Exec exec) {
    if (plan.get_subband_depth() == 0) {
        // Full band: the automatic chunk, halved while over the budget.
        return
            [](const FDMTBoxcarMemoryEstimate& e) -> std::optional<SizeType> {
                return 64 - std::bit_width(e.chunk_nsamps);
            };
    }
    algorithms::detail::FDMTEngineConfig ec;
    ec.exec                = exec;
    ec.search_nbits        = cfg.input_nbits;
    ec.search_upper_levels = true;
    const auto list        = algorithms::detail::gpu_search_list(
        algorithms::detail::fdmt_gpu_cache_chunk(plan, ec), cfg.max_boxcar);
    const SizeType nsamps = plan.get_nsamps();
    return [list, nsamps](
               const FDMTBoxcarMemoryEstimate& e) -> std::optional<SizeType> {
        // The windowed top (skewed path of a subband search) delays the
        // upper subbands; the direct top does not.
        const bool skewed = e.search_latency > 0;
        for (SizeType i = 0; i < list.size(); ++i) {
            if (std::min(list[i].chunk, nsamps) == e.chunk_nsamps &&
                (list[i].path == Path::kSkewed) == skewed) {
                return i;
            }
        }
        return std::nullopt;
    };
}
} // namespace

TEST_CASE("FDMTBoxcar budget sweep: the list's rank never worsens (gpu)",
          "[boxcar][budget][sweep][gpu]") {
    const Exec gpu = test::gpu_exec();
    // A 1152-channel subband search (cryopaf_lband-like: both paths and a
    // cache chunk), a short block (chunks capped at nsamps), the full band,
    // and a UWL-like search (1024 channels, dt_max 60000: the direct path
    // needs far more than the skewed one). Short boxcars (128: direct from
    // chunk 256) and, on the UWL-like plan, wide ones (4096: direct from
    // chunk 512, so skewed 4096 comes before direct 256). Estimates only:
    // nothing is allocated.
    struct Case {
        SizeType nchans;
        IndexType dt_max;
        SizeType depth;
        SizeType nsamps;
        SizeType max_boxcar;
    };
    for (const auto c :
         {Case{1152, 2700, 1, 16384, 128}, Case{1152, 2700, 1, 2048, 128},
          Case{1152, 2700, 0, 16384, 128}, Case{1024, 60000, 1, 16384, 128},
          Case{1024, 60000, 1, 16384, 4096}}) {
        const plans::FDMTPlan plan(1100.0F, 1714.0F, c.nchans, c.nsamps,
                                   455.6e-6F, c.dt_max, 0, 1, "valid", c.depth);
        for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
            INFO("nchans " << c.nchans << " dt_max " << c.dt_max << " depth "
                           << c.depth << " nsamps " << c.nsamps
                           << " max_boxcar " << c.max_boxcar << " nbits "
                           << nbits);
            BoxcarConfig cfg;
            cfg.max_boxcar = c.max_boxcar;
            if (nbits != 32) {
                cfg.input_nbits = nbits;
                cfg.input_mean  = 127;
                cfg.noise_sigma = 8;
            }
            test::check_budget_sweep(plan, cfg, gpu, gpu_rank(plan, cfg, gpu));
        }
    }
}
