// The C++ quickstart (docs/getting_started/quickstart_cpp.md), built against an
// installed dmt:
// - FDMT: an impulse at t0 in every channel of a zero-DM waterfall comes out
//   at DM 0, sample t0;
// - FDMTBoxcar: the same impulse in noise is a candidate at sample t0;
// - CohFDMT: a simulated dispersed baseband pulse is found at its DM;
// - FFTW is linked by the consumer itself as well.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#include <dmt/dmt.hpp>
#include <fftw3.h>

namespace {

bool check_fdmt() {
    const size_t nchans = 256;
    const size_t nsamps = 1024;
    dmt::algorithms::FDMT fdmt(1200.0F, 1600.0F, nchans, nsamps, 1e-3F, 128,
                               /*dt_min=*/0, /*dt_step=*/1,
                               /*use_box_smearing=*/true, /*mode=*/"full",
                               dmt::Exec::cpu(/*nthreads=*/1));
    const auto& plan = fdmt.get_plan();
    const size_t t0  = 300;
    std::vector<float> waterfall(nchans * nsamps, 0.0F);
    for (size_t c = 0; c < nchans; ++c) {
        waterfall[(c * nsamps) + t0] = 1.0F;
    }
    std::vector<float> out(plan.get_dmt_size());
    fdmt.execute(waterfall, out);
    const auto peak =
        static_cast<size_t>(std::ranges::max_element(out) - out.begin());
    const size_t i_dm = peak / plan.get_dmt_nsamps();
    const size_t t    = peak % plan.get_dmt_nsamps();
    std::cout << "FDMT: " << plan.get_dmt_ndms()
              << " DM trials; peak at DM index " << i_dm << ", sample " << t
              << '\n';
    return i_dm == 0 && t == t0;
}

bool check_fdmt_boxcar() {
    const size_t nchans = 64;
    const size_t nsamps = 1024;
    const dmt::plans::FDMTPlan plan(1200.0F, 1600.0F, nchans, nsamps, 1e-3F,
                                    32);
    const dmt::BoxcarConfig config{.max_boxcar = 8, .detect_thresh = 8.0F};
    dmt::algorithms::FDMTBoxcar search(plan, config);
    const size_t t0 = 500;
    std::vector<float> waterfall(nchans * nsamps, 0.0F);
    for (size_t c = 0; c < nchans; ++c) {
        waterfall[(c * nsamps) + t0] = 3.0F;
    }
    const auto first = search.execute(waterfall);
    std::vector<dmt::RawCandidate> found(first.candidates.begin(),
                                         first.candidates.end());
    const auto tail = search.drain();
    found.insert(found.end(), tail.candidates.begin(), tail.candidates.end());
    const auto best = std::ranges::max_element(
        found, {}, [](const dmt::RawCandidate& c) { return c.snr; });
    if (best == found.end()) {
        return false;
    }
    std::cout << "FDMTBoxcar: " << found.size() << " candidate(s); best S/N "
              << best->snr << " at sample " << best->peak_sample << '\n';
    return best->peak_sample == t0 && best->dm_idx == 0;
}

bool check_cfdmt() {
    const dmt::CohFDMTConfig config{.f_center = 400.0F,
                                    .bw_sub   = 1.0F,
                                    .nsub     = 16,
                                    .t_p      = 4.0E-6F,
                                    .dm_min   = 10.0F,
                                    .dm_max   = 11.0F};
    const dmt::algorithms::CohFDMT search(config);
    const auto& plan = search.get_plan();
    const double dm  = plan.get_dm_grid_coh()[1];
    const double t =
        plan.get_output_time_offset() +
        (100.0 * static_cast<double>(plan.get_tsamp())) -
        (dmt::kDispConst * dm /
         (static_cast<double>(plan.get_f_ref()) * plan.get_f_ref()));
    const dmt::utils::BasebandPulse pulse{
        .dm = dm, .t_arrival = t, .fluence = 1.0E6};
    const auto voltages = dmt::utils::simulate_baseband(
        plan.get_f_center(), plan.get_bw_sub(), plan.get_nsub(),
        plan.get_block_nsamps(), std::span(&pulse, 1), 0.0F);
    float peak_v = 0.0F;
    for (const auto& v : voltages) {
        peak_v = std::max({peak_v, std::abs(v.real()), std::abs(v.imag())});
    }
    const auto bytes = dmt::utils::pack_baseband(
        voltages, plan.get_nsub(), plan.get_block_nsamps(), plan.get_format(),
        100.0F / peak_v);
    std::vector<float> out(search.get_dmt_size());
    search.execute<uint8_t>(std::span<const uint8_t>(bytes), out);
    const auto best =
        static_cast<size_t>(std::ranges::max_element(out) - out.begin());
    const auto nout = plan.get_output_nsamps();
    const auto row  = best / nout;
    std::cout << "CohFDMT: peak DM " << plan.get_dm_grid_final()[row]
              << " (injected " << dm << "), sample " << best % nout << '\n';
    return std::abs(plan.get_dm_grid_final()[row] - dm) <=
           2.0 * plan.get_dm_step_coh();
}

} // namespace

int main() {
    std::cout << "dmt consumer: backends";
    for (const auto backend : dmt::available_backends()) {
        std::cout << ' ' << dmt::to_string(backend);
    }
    std::cout << "; " << fftwf_version << '\n';
    bool ok = true;
    for (const auto& [name, check] :
         {std::pair{"FDMT", &check_fdmt},
          std::pair{"FDMTBoxcar", &check_fdmt_boxcar},
          std::pair{"CohFDMT", &check_cfdmt}}) {
        if (!check()) {
            std::cerr << name << ": unexpected result\n";
            ok = false;
        }
    }
    return ok ? 0 : 1;
}
