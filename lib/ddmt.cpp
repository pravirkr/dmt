#include "dmt/algorithms/ddmt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef DMT_ENABLE_OPENMP
#include <omp.h>
#endif

#include <spdlog/spdlog.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/omp_helper.hpp"

namespace dmt::algorithms {

class DDMTCPU::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min,
         int nthreads,
         SizeType nbits,
         std::span<const uint8_t> kill_mask,
         SizeType nbeams)
        : m_plan(f_min,
                 f_max,
                 nchans,
                 tsamp,
                 dm_max,
                 dm_step,
                 dm_min,
                 false,
                 nbits,
                 kill_mask),
          m_nthreads(set_dmt_openmp_threads(nthreads)),
          m_nbeams(nbeams) {}

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         std::span<const float> dm_arr,
         int nthreads,
         SizeType nbits,
         std::span<const uint8_t> kill_mask,
         SizeType nbeams)
        : m_plan(f_min, f_max, nchans, tsamp, dm_arr, false, nbits, kill_mask),
          m_nthreads(set_dmt_openmp_threads(nthreads)),
          m_nbeams(nbeams) {}

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const plans::LevinConfig& levin,
         int nthreads,
         SizeType nbits,
         std::span<const uint8_t> kill_mask,
         SizeType nbeams)
        : m_plan(f_min, f_max, nchans, tsamp, levin, false, nbits, kill_mask),
          m_nthreads(set_dmt_openmp_threads(nthreads)),
          m_nbeams(nbeams) {}

    Impl(const plans::DDMTPlan& plan, int nthreads, SizeType nbeams)
        : m_plan(plan),
          m_nthreads(set_dmt_openmp_threads(nthreads)),
          m_nbeams(nbeams) {}

    const plans::DDMTPlan& get_plan() const { return m_plan; }
    SizeType get_nbeams() const noexcept { return m_nbeams; }

    // Number of output samples execute(float) will produce for a block of
    // `input_nsamps` new samples, given the currently retained history
    // (see reset_history()'s doc comment for the streaming model).
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept {
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        const auto total = m_history_len + input_nsamps;
        return total > max_delay ? total - max_delay : 0;
    }

    void reset_history() noexcept {
        m_history.clear();
        m_history_packed.clear();
        m_history_len = 0;
    }

    [[nodiscard]] SizeType history_state_size() const noexcept {
        const auto& plan_c   = m_plan.get_container();
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        if (plan_c.nbits == 32) {
            return m_nbeams * plan_c.nchans * max_delay;
        }
        return m_nbeams * plan_c.nchans *
               utils::packed_row_bytes(max_delay, plan_c.nbits);
    }

    bool save_history(std::span<float> out) const {
        if (m_plan.get_nbits() != 32) {
            spdlog::error(
                "DDMTCPU::save_history(float): plan nbits={} != 32; "
                "use the packed save_history(uint8_t) overload instead",
                m_plan.get_nbits());
            return false;
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            spdlog::error("DDMTCPU::save_history: stream is not fully "
                          "warmed up yet ({} of {} history samples/channel)",
                          m_history_len, max_delay);
            return false;
        }
        if (out.size() != m_history.size()) {
            spdlog::error("DDMTCPU::save_history: buffer size mismatch: "
                          "expected {}, got {}",
                          m_history.size(), out.size());
            return false;
        }
        std::ranges::copy(m_history, out.begin());
        return true;
    }

    bool save_history(std::span<uint8_t> out) const {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            spdlog::error("DDMTCPU::save_history(uint8_t): plan nbits=32; "
                          "use the float save_history(float) overload instead");
            return false;
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            spdlog::error("DDMTCPU::save_history: stream is not fully "
                          "warmed up yet ({} of {} history samples/channel)",
                          m_history_len, max_delay);
            return false;
        }
        const auto expected_bytes = history_state_size();
        if (out.size() != expected_bytes) {
            spdlog::error("DDMTCPU::save_history: buffer size mismatch: "
                          "expected {} bytes, got {}",
                          expected_bytes, out.size());
            return false;
        }
        std::ranges::copy(m_history_packed, out.begin());
        return true;
    }

    bool load_history(std::span<const float> in) {
        if (m_plan.get_nbits() != 32) {
            spdlog::error(
                "DDMTCPU::load_history(float): plan nbits={} != 32; "
                "use the packed load_history(uint8_t) overload instead",
                m_plan.get_nbits());
            return false;
        }
        if (in.size() != history_state_size()) {
            spdlog::error("DDMTCPU::load_history: buffer size mismatch: "
                          "expected {}, got {}",
                          history_state_size(), in.size());
            return false;
        }
        m_history.assign(in.begin(), in.end());
        m_history_len = in.size() / (m_nbeams * m_plan.get_nchans());
        return true;
    }

    bool load_history(std::span<const uint8_t> in) {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            spdlog::error("DDMTCPU::load_history(uint8_t): plan nbits=32; "
                          "use the float load_history(float) overload instead");
            return false;
        }
        const auto expected_bytes = history_state_size();
        if (in.size() != expected_bytes) {
            spdlog::error("DDMTCPU::load_history: buffer size mismatch: "
                          "expected {} bytes, got {}",
                          expected_bytes, in.size());
            return false;
        }
        m_history_packed.assign(in.begin(), in.end());
        m_history_len =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        return true;
    }

    void execute(std::span<const float> waterfall, std::span<float> dmt) {
        const auto& plan_c = m_plan.get_container();
        if (plan_c.nbits != 32) {
            spdlog::error("DDMTCPU::execute(float): plan nbits={} != 32; use "
                          "the packed-integer execute() overload instead",
                          plan_c.nbits);
            return;
        }
        const auto nchans       = plan_c.nchans;
        const auto nsamps_new   = waterfall.size() / (nchans * m_nbeams);
        const auto max_delay    = *std::ranges::max_element(plan_c.delay_table);
        const auto total        = m_history_len + nsamps_new;
        const auto n_out        = total > max_delay ? total - max_delay : 0;
        const auto* delay_table = plan_c.delay_table.data();
        const auto* kill_mask   = plan_c.kill_mask.data();
        const auto dm_count     = plan_c.dm_arr.size();

        if (dmt.size() != m_nbeams * dm_count * n_out) {
            spdlog::error("Output buffer size mismatch: expected {}, got {}",
                          m_nbeams * dm_count * n_out, dmt.size());
            return;
        }

        // Beam-major (nbeams, nchans, total): prepend each beam's retained
        // history tail to that beam's new block.
        std::vector<float> combined(m_nbeams * nchans * total);
        for (SizeType ibeam = 0; ibeam < m_nbeams; ++ibeam) {
            for (SizeType ichan = 0; ichan < nchans; ++ichan) {
                const auto row = (ibeam * nchans) + ichan;
                auto* dst      = &combined[row * total];
                if (m_history_len > 0) {
                    std::copy_n(&m_history[(row * m_history_len)],
                                m_history_len, dst);
                }
                std::copy_n(&waterfall[row * nsamps_new], nsamps_new,
                            dst + m_history_len);
            }
        }

        if (n_out > 0) {
            execute_dedisp(combined.data(), total, 1, dmt.data(), n_out, 1,
                           delay_table, kill_mask, dm_count, nchans, n_out,
                           m_nbeams);
        }

        const auto new_history_len = std::min(total, max_delay);
        std::vector<float> new_history(m_nbeams * nchans * new_history_len);
        for (SizeType ibeam = 0; ibeam < m_nbeams; ++ibeam) {
            for (SizeType ichan = 0; ichan < nchans; ++ichan) {
                const auto row = (ibeam * nchans) + ichan;
                std::copy_n(
                    &combined[(row * total) + (total - new_history_len)],
                    new_history_len, &new_history[row * new_history_len]);
            }
        }
        m_history     = std::move(new_history);
        m_history_len = new_history_len;
    }

    // `d_in`/`d_out` are beam-major: (nbeams, nchans, in_chan_stride) and
    // (nbeams, dm_count, out_dm_stride) respectively. Folding (beam, dm)
    // into one flat OpenMP dimension (rather than a nested loop) preserves
    // the exact tiled/SIMD inner loop below and reduces to the original
    // single-beam behavior bit-for-bit when nbeams == 1.
    void execute_dedisp(const float* __restrict__ d_in,
                        size_t in_chan_stride,
                        size_t in_samp_stride,
                        float* __restrict__ d_out,
                        size_t out_dm_stride,
                        size_t out_samp_stride,
                        const size_t* __restrict__ delay_table,
                        const uint8_t* __restrict__ kill_mask,
                        size_t dm_count,
                        size_t nchans,
                        size_t nsamps_reduced,
                        size_t nbeams) {
        const size_t in_beam_stride  = nchans * in_chan_stride;
        const size_t out_beam_stride = dm_count * out_dm_stride;
        const size_t total_units     = nbeams * dm_count;
#pragma omp parallel for default(none)                                         \
    shared(d_in, d_out, delay_table, kill_mask, dm_count, nchans,              \
               nsamps_reduced, in_chan_stride, in_samp_stride, out_dm_stride,  \
               out_samp_stride, in_beam_stride, out_beam_stride, total_units)
        for (size_t u = 0; u < total_units; ++u) {
            const size_t i_beam    = u / dm_count;
            const size_t i_dm      = u % dm_count;
            const auto* d_in_b     = d_in + (i_beam * in_beam_stride);
            auto* d_out_b          = d_out + (i_beam * out_beam_stride);
            const size_t* delays   = delay_table + (i_dm * nchans);
            const auto out_idx     = i_dm * out_dm_stride;
            constexpr size_t kTile = 2048;
            for (size_t s_block = 0; s_block < nsamps_reduced;
                 s_block += kTile) {
                const size_t s_count =
                    std::min(kTile, nsamps_reduced - s_block);
                float* out_tile =
                    d_out_b + out_idx + (s_block * out_samp_stride);
                std::fill_n(out_tile, s_count, 0.0F);
                for (size_t i_chan = 0; i_chan < nchans; ++i_chan) {
                    if (!kill_mask[i_chan]) {
                        continue;
                    }
                    const auto delay    = delays[i_chan];
                    const float* in_ptr = d_in_b + (i_chan * in_chan_stride) +
                                          ((s_block + delay) * in_samp_stride);
#pragma omp simd
                    for (size_t s = 0; s < s_count; ++s) {
                        out_tile[s] += in_ptr[s];
                    }
                }
            }
        }
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt) {
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            spdlog::error("DDMTCPU::execute(packed): plan nbits={} is not a "
                          "supported packed width (1,2,4,8,16); use the "
                          "float execute() overload for nbits==32",
                          nbits);
            return;
        }
        const auto nchans    = plan_c.nchans;
        const auto in_rows   = m_nbeams * nchans;
        const auto row_bytes = utils::packed_row_bytes(nsamps, nbits);
        if (waterfall_packed.size() != in_rows * row_bytes) {
            spdlog::error("Packed input buffer size mismatch: expected {} "
                          "bytes ({} beams x {} chans x {} bytes/row), got {}",
                          in_rows * row_bytes, m_nbeams, nchans, row_bytes,
                          waterfall_packed.size());
            return;
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto total     = m_history_len + nsamps;
        const auto nsamps_reduced = total > max_delay ? total - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            spdlog::error("Output buffer size mismatch: expected {}, got {}",
                          m_nbeams * dm_count * nsamps_reduced, dmt.size());
            return;
        }

        const auto combined_row_bytes = utils::packed_row_bytes(total, nbits);
        const auto hist_row_bytes =
            utils::packed_row_bytes(m_history_len, nbits);

        auto run_dedisp = [&](const uint8_t* ptr, SizeType stride_bytes) {
            if (nsamps_reduced == 0)
                return;
            switch (nbits) {
            case 1:
                execute_dedisp_packed<1>(ptr, stride_bytes, dmt.data(),
                                         plan_c.delay_table.data(),
                                         plan_c.kill_mask.data(), dm_count,
                                         nchans, nsamps_reduced, m_nbeams);
                break;
            case 2:
                execute_dedisp_packed<2>(ptr, stride_bytes, dmt.data(),
                                         plan_c.delay_table.data(),
                                         plan_c.kill_mask.data(), dm_count,
                                         nchans, nsamps_reduced, m_nbeams);
                break;
            case 4:
                execute_dedisp_packed<4>(ptr, stride_bytes, dmt.data(),
                                         plan_c.delay_table.data(),
                                         plan_c.kill_mask.data(), dm_count,
                                         nchans, nsamps_reduced, m_nbeams);
                break;
            case 8:
                execute_dedisp_packed<8>(ptr, stride_bytes, dmt.data(),
                                         plan_c.delay_table.data(),
                                         plan_c.kill_mask.data(), dm_count,
                                         nchans, nsamps_reduced, m_nbeams);
                break;
            case 16:
                execute_dedisp_packed<16>(ptr, stride_bytes, dmt.data(),
                                          plan_c.delay_table.data(),
                                          plan_c.kill_mask.data(), dm_count,
                                          nchans, nsamps_reduced, m_nbeams);
                break;
            }
        };

        const auto new_history_len = std::min(total, max_delay);
        const auto new_hist_row_bytes =
            utils::packed_row_bytes(new_history_len, nbits);
        std::vector<uint8_t> new_history_packed(in_rows * new_hist_row_bytes,
                                                0);

        if (m_history_len == 0) {
            run_dedisp(waterfall_packed.data(), row_bytes);

            auto extract_tail = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* src_row =
                        waterfall_packed.data() + (row * row_bytes);
                    auto* dst_row =
                        new_history_packed.data() + (row * new_hist_row_bytes);
                    utils::copy_packed_samples<NBITS>(
                        src_row, total - new_history_len, dst_row, 0,
                        new_history_len);
                }
            };
            switch (nbits) {
            case 1:
                extract_tail.template operator()<1>();
                break;
            case 2:
                extract_tail.template operator()<2>();
                break;
            case 4:
                extract_tail.template operator()<4>();
                break;
            case 8:
                extract_tail.template operator()<8>();
                break;
            case 16:
                extract_tail.template operator()<16>();
                break;
            }
        } else {
            std::vector<uint8_t> combined_packed(in_rows * combined_row_bytes,
                                                 0);
            auto combine_rows = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* hist_row =
                        m_history_packed.data() + (row * hist_row_bytes);
                    const auto* new_row =
                        waterfall_packed.data() + (row * row_bytes);
                    auto* comb_row =
                        combined_packed.data() + (row * combined_row_bytes);
                    utils::copy_packed_samples<NBITS>(hist_row, 0, comb_row, 0,
                                                      m_history_len);
                    utils::copy_packed_samples<NBITS>(new_row, 0, comb_row,
                                                      m_history_len, nsamps);
                }
            };
            switch (nbits) {
            case 1:
                combine_rows.template operator()<1>();
                break;
            case 2:
                combine_rows.template operator()<2>();
                break;
            case 4:
                combine_rows.template operator()<4>();
                break;
            case 8:
                combine_rows.template operator()<8>();
                break;
            case 16:
                combine_rows.template operator()<16>();
                break;
            }

            run_dedisp(combined_packed.data(), combined_row_bytes);

            auto extract_tail = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* comb_row =
                        combined_packed.data() + (row * combined_row_bytes);
                    auto* dst_row =
                        new_history_packed.data() + (row * new_hist_row_bytes);
                    utils::copy_packed_samples<NBITS>(
                        comb_row, total - new_history_len, dst_row, 0,
                        new_history_len);
                }
            };
            switch (nbits) {
            case 1:
                extract_tail.template operator()<1>();
                break;
            case 2:
                extract_tail.template operator()<2>();
                break;
            case 4:
                extract_tail.template operator()<4>();
                break;
            case 8:
                extract_tail.template operator()<8>();
                break;
            case 16:
                extract_tail.template operator()<16>();
                break;
            }
        }

        m_history_packed = std::move(new_history_packed);
        m_history_len    = new_history_len;
    }

    // `d_in`/`d_out` are beam-major: (nbeams, nchans, row_bytes) and
    // (nbeams, dm_count, nsamps_reduced) respectively; see
    // execute_dedisp's doc comment for the same (beam, dm) flattening.
    template <unsigned NBITS>
    void execute_dedisp_packed(const uint8_t* __restrict__ d_in,
                               size_t row_bytes,
                               int32_t* __restrict__ d_out,
                               const size_t* __restrict__ delay_table,
                               const uint8_t* __restrict__ kill_mask,
                               size_t dm_count,
                               size_t nchans,
                               size_t nsamps_reduced,
                               size_t nbeams) {
        const size_t in_beam_stride  = nchans * row_bytes;
        const size_t out_beam_stride = dm_count * nsamps_reduced;
        const size_t total_units     = nbeams * dm_count;
#pragma omp parallel for default(none) shared(                                 \
        d_in, d_out, delay_table, kill_mask, dm_count, nchans, nsamps_reduced, \
            row_bytes, in_beam_stride, out_beam_stride, total_units)
        for (size_t u = 0; u < total_units; ++u) {
            const size_t i_beam    = u / dm_count;
            const size_t i_dm      = u % dm_count;
            const auto* d_in_b     = d_in + (i_beam * in_beam_stride);
            auto* d_out_b          = d_out + (i_beam * out_beam_stride);
            const size_t* delays   = delay_table + (i_dm * nchans);
            const auto out_idx     = i_dm * nsamps_reduced;
            constexpr size_t kTile = 2048;
            for (size_t s_block = 0; s_block < nsamps_reduced;
                 s_block += kTile) {
                const size_t s_count =
                    std::min(kTile, nsamps_reduced - s_block);
                int32_t* out_tile = d_out_b + out_idx + s_block;
                std::fill_n(out_tile, s_count, 0);
                for (size_t i_chan = 0; i_chan < nchans; ++i_chan) {
                    if (!kill_mask[i_chan]) {
                        continue;
                    }
                    const auto delay = delays[i_chan];
                    if constexpr (NBITS == 8) {
                        const auto* in_ptr =
                            d_in_b + (i_chan * row_bytes) + s_block + delay;
#pragma omp simd
                        for (size_t s = 0; s < s_count; ++s) {
                            out_tile[s] += static_cast<int32_t>(in_ptr[s]);
                        }
                    } else if constexpr (NBITS == 16) {
                        // Native uint16_t load, not read_packed_sample<16>'s
                        // manual little-endian byte assembly -- numerically
                        // identical on every little-endian target this
                        // library ships on (x86_64, ARM64/Apple Silicon,
                        // NVIDIA GPUs), but would diverge on a big-endian
                        // host.
                        const auto* in_ptr =
                            reinterpret_cast<const uint16_t*>(
                                d_in_b + (i_chan * row_bytes)) +
                            s_block + delay;
#pragma omp simd
                        for (size_t s = 0; s < s_count; ++s) {
                            out_tile[s] += static_cast<int32_t>(in_ptr[s]);
                        }
                    } else {
                        const auto* row = d_in_b + (i_chan * row_bytes);
                        for (size_t s = 0; s < s_count; ++s) {
                            const auto sample =
                                utils::read_packed_sample<NBITS>(
                                    row, s_block + s + delay);
                            out_tile[s] += static_cast<int32_t>(sample);
                        }
                    }
                }
            }
        }
    }

    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt) {
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            spdlog::error(
                "DDMTCPU::execute_time_major: plan nbits={} is not supported",
                nbits);
            return;
        }
        const auto nchans     = plan_c.nchans;
        const auto samp_bytes = utils::packed_row_bytes(nchans, nbits);
        if (filterbank_packed.size() != m_nbeams * nsamps * samp_bytes) {
            spdlog::error(
                "Time-major input buffer size mismatch: expected {}, got {}",
                m_nbeams * nsamps * samp_bytes, filterbank_packed.size());
            return;
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto nsamps_reduced = nsamps > max_delay ? nsamps - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            spdlog::error("Output buffer size mismatch: expected {}, got {}",
                          m_nbeams * dm_count * nsamps_reduced, dmt.size());
            return;
        }
        if (nsamps_reduced == 0) {
            return;
        }

        switch (nbits) {
        case 1:
            execute_dedisp_time_major<1>(
                filterbank_packed.data(), samp_bytes, dmt.data(),
                plan_c.delay_table.data(), plan_c.kill_mask.data(), dm_count,
                nchans, nsamps_reduced, nsamps, m_nbeams);
            break;
        case 2:
            execute_dedisp_time_major<2>(
                filterbank_packed.data(), samp_bytes, dmt.data(),
                plan_c.delay_table.data(), plan_c.kill_mask.data(), dm_count,
                nchans, nsamps_reduced, nsamps, m_nbeams);
            break;
        case 4:
            execute_dedisp_time_major<4>(
                filterbank_packed.data(), samp_bytes, dmt.data(),
                plan_c.delay_table.data(), plan_c.kill_mask.data(), dm_count,
                nchans, nsamps_reduced, nsamps, m_nbeams);
            break;
        case 8:
            execute_dedisp_time_major<8>(
                filterbank_packed.data(), samp_bytes, dmt.data(),
                plan_c.delay_table.data(), plan_c.kill_mask.data(), dm_count,
                nchans, nsamps_reduced, nsamps, m_nbeams);
            break;
        case 16:
            execute_dedisp_time_major<16>(
                filterbank_packed.data(), samp_bytes, dmt.data(),
                plan_c.delay_table.data(), plan_c.kill_mask.data(), dm_count,
                nchans, nsamps_reduced, nsamps, m_nbeams);
            break;
        }
    }

    // `d_in`/`d_out` are beam-major: (nbeams, nsamps_total, samp_bytes) and
    // (nbeams, dm_count, nsamps_reduced) respectively; see
    // execute_dedisp's doc comment for the same (beam, dm) flattening.
    template <unsigned NBITS>
    void execute_dedisp_time_major(const uint8_t* __restrict__ d_in,
                                   size_t samp_bytes,
                                   int32_t* __restrict__ d_out,
                                   const size_t* __restrict__ delay_table,
                                   const uint8_t* __restrict__ kill_mask,
                                   size_t dm_count,
                                   size_t nchans,
                                   size_t nsamps_reduced,
                                   size_t nsamps_total,
                                   size_t nbeams) {
        const size_t in_beam_stride  = nsamps_total * samp_bytes;
        const size_t out_beam_stride = dm_count * nsamps_reduced;
        const size_t total_units     = nbeams * dm_count;
#pragma omp parallel for default(none) shared(                                 \
        d_in, d_out, delay_table, kill_mask, dm_count, nchans, nsamps_reduced, \
            samp_bytes, in_beam_stride, out_beam_stride, total_units)
        for (size_t u = 0; u < total_units; ++u) {
            const size_t i_beam  = u / dm_count;
            const size_t i_dm    = u % dm_count;
            const auto* d_in_b   = d_in + (i_beam * in_beam_stride);
            auto* d_out_b        = d_out + (i_beam * out_beam_stride);
            const size_t* delays = delay_table + (i_dm * nchans);
            const auto out_idx   = i_dm * nsamps_reduced;
            for (size_t i_samp = 0; i_samp < nsamps_reduced; ++i_samp) {
                int32_t sum = 0;
                for (size_t i_chan = 0; i_chan < nchans; ++i_chan) {
                    if (!kill_mask[i_chan]) {
                        continue;
                    }
                    const auto samp_idx  = i_samp + delays[i_chan];
                    const auto* samp_ptr = d_in_b + (samp_idx * samp_bytes);
                    const auto val =
                        utils::read_packed_sample<NBITS>(samp_ptr, i_chan);
                    sum += static_cast<int32_t>(val);
                }
                d_out_b[out_idx + i_samp] = sum;
            }
        }
    }

private:
    plans::DDMTPlan m_plan;
    int m_nthreads;
    SizeType m_nbeams;
    // Retained tail from previous execute calls; see reset_history().
    std::vector<float> m_history;
    std::vector<uint8_t> m_history_packed;
    SizeType m_history_len = 0;
};

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min,
                 int nthreads,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    tsamp,
                                    dm_max,
                                    dm_step,
                                    dm_min,
                                    nthreads,
                                    nbits,
                                    kill_mask,
                                    nbeams)) {}

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 std::span<const float> dm_arr,
                 int nthreads,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    tsamp,
                                    dm_arr,
                                    nthreads,
                                    nbits,
                                    kill_mask,
                                    nbeams)) {}

DDMTCPU::DDMTCPU(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 const plans::LevinConfig& levin,
                 int nthreads,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    tsamp,
                                    levin,
                                    nthreads,
                                    nbits,
                                    kill_mask,
                                    nbeams)) {}

DDMTCPU::DDMTCPU(const plans::DDMTPlan& plan, int nthreads, SizeType nbeams)
    : m_impl(std::make_unique<Impl>(plan, nthreads, nbeams)) {}

DDMTCPU::~DDMTCPU()                                   = default;
DDMTCPU::DDMTCPU(DDMTCPU&& other) noexcept            = default;
DDMTCPU& DDMTCPU::operator=(DDMTCPU&& other) noexcept = default;
const plans::DDMTPlan& DDMTCPU::get_plan() const noexcept {
    return m_impl->get_plan();
}
SizeType DDMTCPU::get_nbeams() const noexcept { return m_impl->get_nbeams(); }
void DDMTCPU::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->execute(waterfall, dmt);
}
void DDMTCPU::execute(std::span<const uint8_t> waterfall_packed,
                      SizeType nsamps,
                      std::span<int32_t> dmt) {
    m_impl->execute(waterfall_packed, nsamps, dmt);
}
void DDMTCPU::execute_time_major(std::span<const uint8_t> filterbank_packed,
                                 SizeType nsamps,
                                 std::span<int32_t> dmt) {
    m_impl->execute_time_major(filterbank_packed, nsamps, dmt);
}
SizeType DDMTCPU::get_output_nsamps(SizeType input_nsamps) const noexcept {
    return m_impl->get_output_nsamps(input_nsamps);
}
void DDMTCPU::reset_history() noexcept { m_impl->reset_history(); }
SizeType DDMTCPU::history_state_size() const noexcept {
    return m_impl->history_state_size();
}
bool DDMTCPU::save_history(std::span<float> out) const {
    return m_impl->save_history(out);
}
bool DDMTCPU::save_history(std::span<uint8_t> out) const {
    return m_impl->save_history(out);
}
bool DDMTCPU::load_history(std::span<const float> in) {
    return m_impl->load_history(in);
}
bool DDMTCPU::load_history(std::span<const uint8_t> in) {
    return m_impl->load_history(in);
}

} // namespace dmt::algorithms
