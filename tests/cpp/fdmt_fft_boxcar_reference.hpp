#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "dmt/engines.hpp"
#include "dmt/fdmt_fft_common.hpp"

#include "fdmt_boxcar_reference.hpp"

namespace boxcar_test {

// FDMT-FFT rows from the existing transform (one beam), moved to input-stream
// coordinates (output sample j of the stream is input sample j - latency),
// then direct windows with FDMTFFTBoxcar's planes.
inline std::vector<RawCandidate>
fft_reference(const plans::FDMTPlan& plan,
              const BoxcarConfig& config,
              bool smear,
              bool frac,
              std::span<const uint8_t> kill,
              Exec exec,
              const std::vector<std::vector<float>>& input,
              SizeType max_segment = 0) {
    using algorithms::FDMTFFT;
    using algorithms::FDMTFFTBoxcar;
    FDMTFFTBoxcar metadata(plan, config, smear, Exec::cpu(), 1, frac, kill);
    const auto planes = metadata.get_search_plans();
    REQUIRE(planes.size() == 1);
    const SizeType ndm = planes.front().dm_grid.size();
    std::vector<std::vector<float>> rows(ndm);
    std::vector<float> output(plan.get_dmt_size());
    SizeType latency = 0;
    if (max_segment == 0) {
        FDMTFFT fft(plan.get_f_min(), plan.get_f_max(), plan.get_nchans(),
                    plan.get_nsamps(), plan.get_tsamp(),
                    plan.get_dt_grid_final(), smear, "valid", exec, 1, frac,
                    kill);
        latency = fft.get_output_latency();
        for (const auto& block : input) {
            fft.execute(block, output);
            for (SizeType d = 0; d < ndm; ++d) {
                const auto* row = output.data() + d * plan.get_nsamps();
                rows[d].insert(rows[d].end(), row, row + plan.get_nsamps());
            }
        }
    } else {
        const auto geom = algorithms::fdmt_fft::make_geometry(
            plan, FDMTMode::kValid, frac, smear);
        latency = geom.latency;
        const auto eng = algorithms::detail::make_fdmt_fft_cpu(
            plan, {.use_box_smearing  = smear,
                   .mode              = FDMTMode::kValid,
                   .nbeams            = 1,
                   .exec              = exec,
                   .fractional_delays = frac,
                   .kill_mask         = {kill.begin(), kill.end()},
                   .max_segment       = max_segment});
        for (const auto& block : input) {
            eng->execute(block, output);
            for (SizeType d = 0; d < ndm; ++d) {
                const auto* row = output.data() + d * plan.get_nsamps();
                rows[d].insert(rows[d].end(), row, row + plan.get_nsamps());
            }
        }
    }
    REQUIRE(latency == metadata.get_output_latency());
    for (auto& row : rows) {
        row.erase(row.begin(),
                  row.begin() + static_cast<std::ptrdiff_t>(
                                    std::min(latency, row.size())));
    }
    return to_arrival(plan, direct_windows(planes, rows, config));
}

} // namespace boxcar_test
