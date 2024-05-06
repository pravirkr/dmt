#pragma once

#include <dmt/fdmt_base.hpp>

class FDMTCPU : public FDMT {
public:
    FDMTCPU(float f_min, float f_max, size_t nchans, size_t nsamps, float tsamp,
            size_t dt_max, size_t dt_step = 1, size_t dt_min = 0);
    static void set_num_threads(int nthreads);
    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override;
    void initialise(std::span<const float> waterfall,
                    std::span<float> state) override;

private:
    // Buffers
    std::vector<float> m_state_in;
    std::vector<float> m_state_out;

    void execute_iter(std::span<const float> state_in,
                      std::span<float> state_out, SizeType i_iter);
};
