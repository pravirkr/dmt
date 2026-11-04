// The C++ quickstart (docs/getting_started/quickstart_cpp.md), built against an
// installed dmt: an impulse at t0 in every channel of a zero-DM waterfall must
// come out at DM 0, sample t0.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <dmt/dmt.hpp>

int main() {
    const float f_min    = 1200.0F;
    const float f_max    = 1600.0F;
    const size_t nchans  = 256;
    const size_t nsamps  = 1024;
    const float tsamp    = 1e-3F;
    const int32_t dt_max = 128;

    dmt::algorithms::FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                               /*dt_min=*/0, /*dt_step=*/1,
                               /*use_box_smearing=*/true, /*mode=*/"full",
                               dmt::Exec::cpu(/*nthreads=*/2));
    const auto& plan = fdmt.get_plan();

    const size_t t0 = 300;
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

    std::cout << "dmt consumer: backends";
    for (const auto backend : dmt::available_backends()) {
        std::cout << ' ' << dmt::to_string(backend);
    }
    std::cout << "; " << plan.get_dmt_ndms() << " DM trials; peak at DM index "
              << i_dm << ", sample " << t << '\n';
    if (i_dm != 0) {
        std::cerr << "expected the peak at DM index 0\n";
        return 1;
    }
    return 0;
}
