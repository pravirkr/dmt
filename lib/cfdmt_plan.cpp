#include "dmt/common/plans.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dmt/baseband_layout.hpp"
#include "dmt/common/baseband.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"

namespace dmt::plans {

namespace {

constexpr double kHzPerMHz = 1.0E6;

// Channel response: a steep Butterworth-like edge at 0.47 of the channel
// width (as in cdmt), sampled on the mbin DC-centred bins of a channel.
std::vector<float> make_channel_taper(SizeType mbin) {
    std::vector<float> taper(mbin);
    for (SizeType b = 0; b < mbin; ++b) {
        const double x =
            (static_cast<double>(b) - (static_cast<double>(mbin) / 2.0)) /
            static_cast<double>(mbin);
        taper[b] =
            static_cast<float>(1.0 / std::sqrt(1.0 + std::pow(x / 0.47, 80.0)));
    }
    return taper;
}

// Half-length (channel samples) of the channel filter's impulse response
// holding all but @p leakage of its energy. Independent of mbin for mbin >>
// 1, so it is measured on a reference length.
SizeType taper_ringing_length(double leakage) {
    constexpr SizeType kRef = 1024;
    const auto taper        = make_channel_taper(kRef);
    std::vector<double> energy((kRef / 2) + 1, 0.0);
    double total = 0.0;
    for (SizeType t = 0; t <= kRef / 2; ++t) {
        std::complex<double> h{0.0, 0.0};
        for (SizeType b = 0; b < kRef; ++b) {
            const double ang = 2.0 * std::numbers::pi *
                               static_cast<double>(b * t % kRef) /
                               static_cast<double>(kRef);
            h += static_cast<double>(taper[b]) *
                 std::complex<double>(std::cos(ang), std::sin(ang));
        }
        // Lags +t and -t (except t = 0 and t = kRef / 2).
        const double mult = (t == 0 || t == kRef / 2) ? 1.0 : 2.0;
        energy[t]         = mult * std::norm(h);
        total += energy[t];
    }
    double tail = total;
    for (SizeType t = 0; t <= kRef / 2; ++t) {
        tail -= energy[t];
        if (tail < leakage * total) {
            return t + 1;
        }
    }
    return kRef / 2;
}

bool is_smooth7(SizeType n) {
    if (n == 0) {
        return false;
    }
    for (const SizeType p : {2U, 3U, 5U, 7U}) {
        while (n % p == 0) {
            n /= p;
        }
    }
    return n == 1;
}

// 7-smooth integer nearest to x (>= 1); ties go down.
SizeType nearest_smooth7(double x) {
    const auto lo0 = static_cast<SizeType>(std::max(1.0, std::floor(x)));
    SizeType lo    = lo0;
    while (lo > 1 && !is_smooth7(lo)) {
        --lo;
    }
    SizeType hi = std::max<SizeType>(1, static_cast<SizeType>(std::ceil(x)));
    while (!is_smooth7(hi)) {
        ++hi;
    }
    return (x - static_cast<double>(lo) <= static_cast<double>(hi) - x) ? lo
                                                                        : hi;
}

// Dispersion delay (s) between frequencies f_lo < f_hi (MHz) at DM dm.
double disp_delay(double dm, double f_lo, double f_hi) {
    return static_cast<double>(kDispConst) * dm *
           ((1.0 / (f_lo * f_lo)) - (1.0 / (f_hi * f_hi)));
}

SizeType ceil_to(SizeType v, SizeType m) { return ((v + m - 1) / m) * m; }

// Autocorrelation sum A(tau) of the per-channel weights of an FDMT row: a
// box of s samples (box smearing) convolved with the boxcar of w samples.
// A = tri_s * tri_w evaluated at tau (tri_n(u) = max(0, n - |u|)).
double window_autocorr(IndexType s, IndexType w, IndexType tau) {
    double acc = 0.0;
    for (IndexType u = -(s - 1); u <= s - 1; ++u) {
        const auto tw = w - std::abs(tau - u);
        if (tw > 0) {
            acc +=
                static_cast<double>(s - std::abs(u)) * static_cast<double>(tw);
        }
    }
    return acc;
}

} // namespace

class CohFDMTPlan::Impl {
public:
    explicit Impl(const CohFDMTConfig& config) : m_cfg(config) {
        validate();
        configure();
    }

    // --- getters ---
    const CohFDMTConfig& config() const noexcept { return m_cfg; }
    float f_min() const noexcept { return m_f_min; }
    float f_max() const noexcept { return m_f_max; }
    double tbin() const noexcept { return m_tbin; }
    SizeType n_p() const noexcept { return m_n_p; }
    SizeType nchans() const noexcept { return m_nchans; }
    float bw_chan() const noexcept { return m_bw_chan; }
    float tsamp() const noexcept { return m_tsamp; }
    float f_ref() const noexcept { return m_f_min + (m_bw_chan / 2.0F); }
    SizeType nbin() const noexcept { return m_nbin; }
    SizeType mbin() const noexcept { return m_mbin; }
    SizeType noverlap() const noexcept { return m_noverlap; }
    SizeType nfft() const noexcept { return m_nfft; }
    SizeType block_nsamps() const noexcept { return m_block_nsamps; }
    SizeType stride_nsamps() const noexcept { return m_nout * m_n_p; }
    SizeType msamp() const noexcept { return m_msamp; }
    SizeType nout() const noexcept { return m_nout; }
    double output_time_offset() const noexcept {
        return static_cast<double>(m_noverlap + (m_max_delay * m_n_p)) * m_tbin;
    }
    const std::vector<SizeType>& groups() const noexcept { return m_groups; }
    SizeType input_size(SizeType igroup) const {
        if (igroup >= m_groups.size()) {
            throw std::out_of_range(
                std::format("CohFDMTPlan: group {} out of range [0, {})",
                            igroup, m_groups.size()));
        }
        return utils::baseband_block_bytes(m_cfg.format, m_block_nsamps,
                                           m_groups[igroup]);
    }
    const std::vector<float>& dm_grid_coh() const noexcept {
        return m_dm_grid_coh;
    }
    float dm_step_coh() const noexcept { return m_dm_step_coh; }
    SizeType ndm_fine() const noexcept { return m_fdmt_plan->get_dmt_ndms(); }
    SizeType fine_dt_max() const noexcept { return m_fine_dt_max; }
    const std::vector<float>& dm_grid_final() const noexcept {
        return m_dm_grid_final;
    }
    SizeType max_delay() const noexcept { return m_max_delay; }
    SizeType fdmt_nsamps() const noexcept {
        return m_nout + (2 * m_fine_dt_max) + m_fdmt_margin;
    }
    IndexType fdmt_window_start() const noexcept {
        return static_cast<IndexType>(m_max_delay) -
               static_cast<IndexType>(m_fine_dt_max + m_fdmt_margin);
    }
    std::span<const IndexType> channel_shifts(SizeType idm) const {
        if (idm >= m_dm_grid_coh.size()) {
            throw std::out_of_range(
                std::format("CohFDMTPlan: coarse trial {} out of range [0, {})",
                            idm, m_dm_grid_coh.size()));
        }
        return {m_shifts.data() + (idm * m_nchans), m_nchans};
    }
    const std::vector<SizeType>& row_offsets() const noexcept {
        return m_row_offsets;
    }
    float intra_channel_smear() const noexcept { return m_intra_smear; }
    SizeType dmt_size() const noexcept {
        return m_dm_grid_final.size() * m_nout;
    }
    const std::vector<float>& taper() const noexcept { return m_taper; }
    const FDMTPlan& fdmt_plan() const noexcept { return *m_fdmt_plan; }

    std::vector<double> lag_correlation(SizeType max_lag) const {
        // Detected noise of a complex Gaussian channel: Cov(I_t, I_t+tau) is
        // proportional to |rho_tau|^2, rho the autocorrelation of the
        // channel response |H|^2 (circular over mbin; beyond mbin / 2 it is
        // negligible and set to 0).
        std::vector<double> r(max_lag + 1, 0.0);
        double norm = 0.0;
        for (const float w : m_taper) {
            norm += static_cast<double>(w) * static_cast<double>(w);
        }
        const SizeType lag_end = std::min(max_lag, m_mbin / 2);
        for (SizeType tau = 0; tau <= lag_end; ++tau) {
            std::complex<double> acc{0.0, 0.0};
            for (SizeType b = 0; b < m_mbin; ++b) {
                const double w2  = static_cast<double>(m_taper[b]) *
                                   static_cast<double>(m_taper[b]);
                const double ang = 2.0 * std::numbers::pi *
                                   static_cast<double>((b * tau) % m_mbin) /
                                   static_cast<double>(m_mbin);
                acc += w2 * std::complex<double>(std::cos(ang), std::sin(ang));
            }
            r[tau] = std::norm(acc) / (norm * norm);
        }
        return r;
    }

    std::vector<float> variance_grid(SizeType boxcar_width) const {
        if (boxcar_width == 0) {
            throw std::invalid_argument("boxcar_width must be greater than 0");
        }
        const auto smearing = m_fdmt_plan->get_smearing_grid_final();
        const auto nfine    = ndm_fine();
        const auto w        = static_cast<IndexType>(boxcar_width);
        IndexType s_max     = 1;
        for (const float v : smearing) {
            s_max = std::max(s_max, static_cast<IndexType>(std::lround(v)) + 1);
        }
        const auto r = lag_correlation(static_cast<SizeType>(s_max + w));
        // Variance of one channel's window (unit-variance correlated
        // samples), cached per smearing length.
        std::map<IndexType, double> cache;
        const auto channel_var = [&](IndexType s) {
            const auto it = cache.find(s);
            if (it != cache.end()) {
                return it->second;
            }
            double v = window_autocorr(s, w, 0);
            for (IndexType tau = 1; tau < s + w - 1; ++tau) {
                v += 2.0 * r[static_cast<SizeType>(tau)] *
                     window_autocorr(s, w, tau);
            }
            cache.emplace(s, v);
            return v;
        };
        std::vector<float> fine(nfine);
        for (SizeType i = 0; i < nfine; ++i) {
            double var = 0.0;
            for (SizeType c = 0; c < m_nchans; ++c) {
                const auto s = static_cast<IndexType>(
                                   std::lround(smearing[(i * m_nchans) + c])) +
                               1;
                var += channel_var(s);
            }
            fine[i] = static_cast<float>(var);
        }
        return tile_fine(fine);
    }

    std::vector<float> count_grid() const {
        const auto smearing = m_fdmt_plan->get_smearing_grid_final();
        const auto nfine    = ndm_fine();
        std::vector<float> fine(nfine, 0.0F);
        for (SizeType i = 0; i < nfine; ++i) {
            double count = 0.0;
            for (SizeType c = 0; c < m_nchans; ++c) {
                count += std::round(smearing[(i * m_nchans) + c]) + 1.0;
            }
            fine[i] = static_cast<float>(count);
        }
        return tile_fine(fine);
    }

    CohFDMTMemoryUsage memory_estimate() const noexcept {
        const SizeType spectrum =
            SizeType{2} * m_nfft * m_cfg.nsub * m_nbin * sizeof(ComplexType);
        const SizeType waterfall = m_nchans * fdmt_nsamps() * sizeof(float);
        const SizeType fdmt =
            (2 * m_fdmt_plan->get_buffer_size() * sizeof(float)) +
            (m_nchans * m_fine_dt_max * sizeof(float));
        const SizeType workspace =
            (m_nchans * m_mbin * 2 * sizeof(unsigned long long)) +
            (m_shifts.size() * sizeof(IndexType)) +
            (m_nchans * 2 * sizeof(float));
        return {.spectrum  = spectrum,
                .waterfall = waterfall,
                .fdmt      = fdmt,
                .workspace = workspace,
                .output    = dmt_size() * sizeof(float)};
    }

    std::string summary() const {
        const auto mib = [](SizeType bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        };
        const auto mem  = memory_estimate();
        std::string out = "*** CohFDMT Plan Summary ***\n";
        out += std::format(
            "Band: {:.4f}-{:.4f} MHz, {} subbands x {:.6f} MHz ({} group(s)), "
            "tbin {:.4g} s\n",
            m_f_min, m_f_max, m_cfg.nsub, m_cfg.bw_sub, m_groups.size(),
            m_tbin);
        out += std::format(
            "Input: {} order, {}-bit {}, {:.1f} MiB per block\n",
            m_cfg.format.order, m_cfg.format.nbits,
            m_cfg.format.nbits == 2
                ? "levels"
                : (m_cfg.format.is_signed ? "signed" : "offset-binary"),
            mib(utils::baseband_block_bytes(m_cfg.format, m_block_nsamps,
                                            m_cfg.nsub)));
        out += std::format(
            "Channels: n_p={} per subband, {} total of {:.6f} MHz, tsamp "
            "{:.4g} s (t_p requested {:.4g} s)\n",
            m_n_p, m_nchans, m_bw_chan, m_tsamp, m_cfg.t_p);
        out += std::format(
            "FFT: nbin={} (fwd), mbin={} (inv), noverlap={} raw samples/side, "
            "nfft={}\n",
            m_nbin, m_mbin, m_noverlap, m_nfft);
        out += std::format(
            "Block: {} raw samples/subband, stride {}, overlap {} ({:.1f}%); "
            "{} valid output samples (max delay {} samples)\n",
            m_block_nsamps, stride_nsamps(), m_block_nsamps - stride_nsamps(),
            100.0 * static_cast<double>(m_block_nsamps - stride_nsamps()) /
                static_cast<double>(m_block_nsamps),
            m_nout, m_max_delay);
        out += std::format(
            "DM: [{}, {}] pc/cc, {} coarse trials (step {:.6g}, residual "
            "intra-channel smear {:.3f} tsamp), {} fine rows each "
            "(dt in [-{}, {}] step {}), {} rows total\n",
            m_cfg.dm_min, m_cfg.dm_max, m_dm_grid_coh.size(), m_dm_step_coh,
            m_intra_smear, ndm_fine(), m_fine_dt_max, m_fine_dt_max,
            m_cfg.dt_step, m_dm_grid_final.size());
        out += std::format(
            "Memory (est.): spectrum {:.1f} MiB, waterfall {:.1f} MiB, FDMT "
            "{:.1f} MiB, workspace {:.1f} MiB; output {:.1f} MiB\n",
            mib(mem.spectrum), mib(mem.waterfall), mib(mem.fdmt),
            mib(mem.workspace), mib(mem.output));
        out += m_fdmt_plan->summary("\t");
        out += std::format("{:*>80}\n", "");
        return out;
    }

private:
    CohFDMTConfig m_cfg;
    std::vector<SizeType> m_groups;
    float m_f_min{};
    float m_f_max{};
    double m_tbin{};
    SizeType m_n_p{};
    SizeType m_nchans{};
    float m_bw_chan{};
    float m_tsamp{};
    SizeType m_nbin{};
    SizeType m_mbin{};
    SizeType m_noverlap{};
    SizeType m_nfft{};
    SizeType m_block_nsamps{};
    SizeType m_msamp{};
    SizeType m_nout{};
    SizeType m_max_delay{};
    SizeType m_fine_dt_max{};
    SizeType m_fdmt_margin{};
    float m_dm_step_coh{};
    float m_intra_smear{};
    std::vector<float> m_dm_grid_coh;
    std::vector<float> m_dm_grid_final;
    std::vector<IndexType> m_shifts; // (ndm_coh, nchans)
    std::vector<SizeType> m_row_offsets;
    std::vector<float> m_taper;
    std::unique_ptr<FDMTPlan> m_fdmt_plan;

    std::vector<float> tile_fine(const std::vector<float>& fine) const {
        std::vector<float> grid;
        grid.reserve(fine.size() * m_dm_grid_coh.size());
        for (SizeType k = 0; k < m_dm_grid_coh.size(); ++k) {
            grid.insert(grid.end(), fine.begin(), fine.end());
        }
        return grid;
    }

    void validate() {
        const auto& c     = m_cfg;
        const auto finite = [](double v) { return utils::is_finite_bits(v); };
        if (c.nsub == 0) {
            throw std::invalid_argument("CohFDMT: nsub must be > 0");
        }
        if (!finite(c.bw_sub) || c.bw_sub <= 0.0F) {
            throw std::invalid_argument("CohFDMT: bw_sub must be positive");
        }
        const double bw =
            static_cast<double>(c.bw_sub) * static_cast<double>(c.nsub);
        if (!finite(c.f_center) || c.f_center - (bw / 2.0) <= 0.0) {
            throw std::invalid_argument(std::format(
                "CohFDMT: f_center={} MHz leaves the band below 0 MHz "
                "(total bandwidth {} MHz)",
                c.f_center, bw));
        }
        if (!finite(c.t_p) || c.t_p <= 0.0F) {
            throw std::invalid_argument("CohFDMT: t_p must be positive");
        }
        if (!finite(c.dm_min) || !finite(c.dm_max) || c.dm_min < 0.0F ||
            c.dm_max < c.dm_min) {
            throw std::invalid_argument(
                std::format("CohFDMT: need 0 <= dm_min <= dm_max, got [{}, {}]",
                            c.dm_min, c.dm_max));
        }
        if (!finite(c.smear_tol) || c.smear_tol <= 0.0F) {
            throw std::invalid_argument("CohFDMT: smear_tol must be positive");
        }
        if (c.dt_step == 0) {
            throw std::invalid_argument("CohFDMT: dt_step must be > 0");
        }
        if (!finite(c.filter_leakage) || c.filter_leakage <= 0.0F ||
            c.filter_leakage >= 1.0F) {
            throw std::invalid_argument(
                "CohFDMT: filter_leakage must be in (0, 1)");
        }
        utils::validate_baseband_format(c.format);
        m_groups = c.subband_groups.empty() ? std::vector<SizeType>{c.nsub}
                                            : c.subband_groups;
        if (std::ranges::any_of(m_groups, [](SizeType g) { return g == 0; }) ||
            std::accumulate(m_groups.begin(), m_groups.end(), SizeType{0}) !=
                c.nsub) {
            throw std::invalid_argument(
                "CohFDMT: subband_groups must be positive and sum to nsub");
        }
    }

    void configure() {
        const auto& c = m_cfg;
        const double bw =
            static_cast<double>(c.bw_sub) * static_cast<double>(c.nsub);
        const double f_min = static_cast<double>(c.f_center) - (bw / 2.0);
        const double f_max = static_cast<double>(c.f_center) + (bw / 2.0);
        m_f_min            = static_cast<float>(f_min);
        m_f_max            = static_cast<float>(f_max);

        // Channelisation: n_p channels per subband, output sample n_p * tbin.
        m_tbin   = 1.0 / (static_cast<double>(c.bw_sub) * kHzPerMHz);
        m_n_p    = nearest_smooth7(static_cast<double>(c.t_p) / m_tbin);
        m_nchans = c.nsub * m_n_p;
        const double bw_chan =
            static_cast<double>(c.bw_sub) / static_cast<double>(m_n_p);
        const double tsamp = static_cast<double>(m_n_p) * m_tbin;
        m_bw_chan          = static_cast<float>(bw_chan);
        m_tsamp            = static_cast<float>(tsamp);

        // Coarse grid: residual intra-channel smearing of the bottom channel
        // at a window edge (half a step from the trial) <= smear_tol * tsamp.
        const double smear_per_dm = disp_delay(1.0, f_min, f_min + bw_chan);
        const double step_max =
            2.0 * static_cast<double>(c.smear_tol) * tsamp / smear_per_dm;
        const double range =
            static_cast<double>(c.dm_max) - static_cast<double>(c.dm_min);
        const auto ncoh =
            range > 0.0 ? std::max<SizeType>(
                              1, static_cast<SizeType>(std::ceil(
                                     (range / step_max) * (1.0 - 1.0E-9))))
                        : SizeType{1};
        const double step = range / static_cast<double>(ncoh);
        m_dm_step_coh     = static_cast<float>(step);
        m_intra_smear = static_cast<float>(smear_per_dm * (step / 2.0) / tsamp);
        m_dm_grid_coh.resize(ncoh);
        for (SizeType k = 0; k < ncoh; ++k) {
            m_dm_grid_coh[k] =
                static_cast<float>(static_cast<double>(c.dm_min) +
                                   ((static_cast<double>(k) + 0.5) * step));
        }

        // Fine FDMT: dt in [-Delta, Delta] covers each window's half-width.
        const double dt_exact = disp_delay(step / 2.0, f_min, f_max) / tsamp;
        const auto dt_step    = static_cast<double>(c.dt_step);
        m_fine_dt_max         = std::max<SizeType>(
            c.dt_step,
            static_cast<SizeType>(std::ceil(dt_exact / dt_step)) * c.dt_step);

        // Inter-channel shifts at every coarse trial, relative to the centre
        // of the lowest channel.
        const double f_ref = f_min + (bw_chan / 2.0);
        m_shifts.resize(ncoh * m_nchans);
        for (SizeType k = 0; k < ncoh; ++k) {
            for (SizeType ch = 0; ch < m_nchans; ++ch) {
                const double f_c =
                    f_min + ((static_cast<double>(ch) + 0.5) * bw_chan);
                m_shifts[(k * m_nchans) + ch] =
                    static_cast<IndexType>(std::nearbyint(
                        disp_delay(static_cast<double>(m_dm_grid_coh[k]), f_ref,
                                   f_c) /
                        tsamp));
            }
        }
        // The FDMT window starts m_fdmt_margin samples before any valid
        // output reads (box-smearing lookback of the bottom channel plus tree
        // rounding), so its streaming history never reaches a cropped
        // output: no reset between coarse trials is needed.
        const double smear_frac = disp_delay(1.0, f_min, f_min + bw_chan) /
                                  disp_delay(1.0, f_min, f_max);
        m_fdmt_margin = static_cast<SizeType>(std::ceil(
                            static_cast<double>(m_fine_dt_max) * smear_frac)) +
                        4;
        const auto delta     = static_cast<IndexType>(m_fine_dt_max);
        const auto top_first = m_shifts[m_nchans - 1];
        const auto top_last  = m_shifts[(ncoh * m_nchans) - 1];
        // Earliest sample any row reads is D before the output sample; the
        // latest is E after it (negative residual rows on the first trial).
        m_max_delay     = static_cast<SizeType>(top_last + delta + 1);
        const auto lead = static_cast<SizeType>(
            std::max<IndexType>(0, delta - top_first) + 2);

        // Coherent filter margin: the chirp response of the bottom channel at
        // the largest coarse DM (its lower edge is the longer side) plus the
        // channel filter's ringing.
        const double tau_lo =
            disp_delay(static_cast<double>(m_dm_grid_coh.back()),
                       f_ref - (bw_chan / 2.0), f_ref);
        const SizeType ring =
            taper_ringing_length(static_cast<double>(c.filter_leakage));
        m_noverlap = ceil_to(static_cast<SizeType>(std::ceil(tau_lo / m_tbin)) +
                                 (ring * m_n_p),
                             m_n_p);
        const SizeType novc = m_noverlap / m_n_p;

        choose_fft_lengths(novc);
        const SizeType lc = m_mbin - (2 * novc);
        const SizeType l  = lc * m_n_p;
        choose_block(novc, lc, l, lead);

        // Fine FDMT over the aligned window [D - Delta - margin,
        // D + nout + Delta).
        m_fdmt_plan = std::make_unique<FDMTPlan>(m_f_min, m_f_max, m_nchans,
                                                 fdmt_nsamps(), m_tsamp, delta,
                                                 -delta, c.dt_step, "valid");
        const auto dt_grid = m_fdmt_plan->get_dt_grid_final();
        m_row_offsets.resize(dt_grid.size());
        for (SizeType r = 0; r < dt_grid.size(); ++r) {
            // Negative-dt rows are referenced to the top channel by the
            // FDMT; shift them back to the bottom-channel reference.
            m_row_offsets[r] =
                m_fine_dt_max + m_fdmt_margin +
                static_cast<SizeType>(std::max<IndexType>(0, -dt_grid[r]));
        }
        const auto fine_dms = m_fdmt_plan->get_dm_grid_final();
        m_dm_grid_final.resize(ncoh * fine_dms.size());
        for (SizeType k = 0; k < ncoh; ++k) {
            for (SizeType r = 0; r < fine_dms.size(); ++r) {
                m_dm_grid_final[(k * fine_dms.size()) + r] =
                    m_dm_grid_coh[k] + fine_dms[r];
            }
        }
        m_taper = make_channel_taper(m_mbin);
    }

    void choose_fft_lengths(SizeType novc) {
        const auto& c = m_cfg;
        if (c.nbin != 0) {
            if (c.nbin % m_n_p != 0 || (c.nbin / m_n_p) % 2 != 0) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: nbin={} must be an even multiple of n_p={}",
                    c.nbin, m_n_p));
            }
            m_mbin = c.nbin / m_n_p;
            if (m_mbin <= 2 * novc) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: nbin={} too short for the coherent filter "
                    "margin: need nbin > {} (2 * noverlap)",
                    c.nbin, 2 * m_noverlap));
            }
        } else {
            // Cost per valid channel sample of the per-trial work (chirp,
            // inverse FFT, detection) ~ (log2(mbin) + 3) / useful fraction;
            // very long transforms leave the caches.
            double best = std::numeric_limits<double>::max();
            for (SizeType m = 16; m <= (SizeType{1} << 20); m *= 2) {
                if (m <= (2 * novc) + 1) {
                    continue;
                }
                const double useful = 1.0 - (2.0 * static_cast<double>(novc) /
                                             static_cast<double>(m));
                double cost =
                    (std::log2(static_cast<double>(m)) + 3.0) / useful;
                if (m > 8192) {
                    cost *= 1.25;
                }
                if (cost < best) {
                    best   = cost;
                    m_mbin = m;
                }
            }
        }
        m_nbin = m_n_p * m_mbin;
    }

    void choose_block(SizeType novc, SizeType lc, SizeType l, SizeType lead) {
        const auto& c        = m_cfg;
        const SizeType guard = m_max_delay + lead; // msamp - nout >= guard
        if (c.block_nsamps != 0) {
            if (c.block_nsamps <= 2 * m_noverlap ||
                (c.block_nsamps - (2 * m_noverlap)) / l == 0) {
                throw std::invalid_argument(std::format(
                    "CohFDMT: block_nsamps={} shorter than one FFT block ({})",
                    c.block_nsamps, l + (2 * m_noverlap)));
            }
            m_nfft = (c.block_nsamps - (2 * m_noverlap)) / l;
        } else {
            // Aim for 4x as many valid samples as the block spends on the
            // sweep (80% of the forward transforms are useful; longer blocks
            // mainly help single-trial searches, where the forward transform
            // dominates) and at least 16 FFT blocks of output, within
            // kAutoSpectrumBytes of spectrum; never below the shortest valid
            // block.
            constexpr SizeType kAutoSpectrumBytes = SizeType{1} << 30U;
            const SizeType target =
                ceil_to(std::max({4 * guard, 2 * m_fine_dt_max, 16 * lc}), lc);
            const SizeType nfft_min = (guard + (2 * lc) - 1) / lc;
            const SizeType nfft_cap =
                kAutoSpectrumBytes /
                (c.nsub * m_nbin * 2 * sizeof(ComplexType));
            m_nfft = std::max(
                nfft_min, std::min((target + guard + lc - 1) / lc, nfft_cap));
        }
        m_msamp = m_nfft * lc;
        if (m_msamp < guard + lc) {
            const SizeType need_nfft = (guard + lc + lc - 1) / lc;
            throw std::invalid_argument(std::format(
                "CohFDMT: block too short for the dispersion sweep: "
                "block_nsamps={} gives {} channel samples, need >= {} "
                "(block_nsamps >= {})",
                c.block_nsamps, m_msamp, guard + lc,
                (need_nfft * l) + (2 * m_noverlap)));
        }
        m_nout         = ((m_msamp - guard) / lc) * lc;
        m_block_nsamps = (m_nfft * l) + (2 * m_noverlap);
        static_cast<void>(novc);
    }
};

// --- Definitions for CohFDMTPlan ---
CohFDMTPlan::CohFDMTPlan(const CohFDMTConfig& config)
    : m_impl(std::make_unique<Impl>(config)) {}
CohFDMTPlan::~CohFDMTPlan()                                 = default;
CohFDMTPlan::CohFDMTPlan(CohFDMTPlan&&) noexcept            = default;
CohFDMTPlan& CohFDMTPlan::operator=(CohFDMTPlan&&) noexcept = default;
CohFDMTPlan::CohFDMTPlan(const CohFDMTPlan& other)
    : m_impl(std::make_unique<Impl>(other.m_impl->config())) {}
CohFDMTPlan& CohFDMTPlan::operator=(const CohFDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(other.m_impl->config());
    }
    return *this;
}

const CohFDMTConfig& CohFDMTPlan::get_config() const noexcept {
    return m_impl->config();
}
const BasebandFormat& CohFDMTPlan::get_format() const noexcept {
    return m_impl->config().format;
}
float CohFDMTPlan::get_f_center() const noexcept {
    return m_impl->config().f_center;
}
float CohFDMTPlan::get_bw_sub() const noexcept {
    return m_impl->config().bw_sub;
}
SizeType CohFDMTPlan::get_nsub() const noexcept {
    return m_impl->config().nsub;
}
const std::vector<SizeType>& CohFDMTPlan::get_subband_groups() const noexcept {
    return m_impl->groups();
}
float CohFDMTPlan::get_bw() const noexcept {
    return m_impl->f_max() - m_impl->f_min();
}
float CohFDMTPlan::get_f_min() const noexcept { return m_impl->f_min(); }
float CohFDMTPlan::get_f_max() const noexcept { return m_impl->f_max(); }
double CohFDMTPlan::get_tbin() const noexcept { return m_impl->tbin(); }
float CohFDMTPlan::get_dm_min() const noexcept {
    return m_impl->config().dm_min;
}
float CohFDMTPlan::get_dm_max() const noexcept {
    return m_impl->config().dm_max;
}
SizeType CohFDMTPlan::get_n_p() const noexcept { return m_impl->n_p(); }
SizeType CohFDMTPlan::get_nchans() const noexcept { return m_impl->nchans(); }
float CohFDMTPlan::get_bw_chan() const noexcept { return m_impl->bw_chan(); }
float CohFDMTPlan::get_tsamp() const noexcept { return m_impl->tsamp(); }
float CohFDMTPlan::get_f_ref() const noexcept { return m_impl->f_ref(); }
SizeType CohFDMTPlan::get_nbin() const noexcept { return m_impl->nbin(); }
SizeType CohFDMTPlan::get_mbin() const noexcept { return m_impl->mbin(); }
SizeType CohFDMTPlan::get_noverlap() const noexcept {
    return m_impl->noverlap();
}
SizeType CohFDMTPlan::get_nfft() const noexcept { return m_impl->nfft(); }
SizeType CohFDMTPlan::get_block_nsamps() const noexcept {
    return m_impl->block_nsamps();
}
SizeType CohFDMTPlan::get_stride_nsamps() const noexcept {
    return m_impl->stride_nsamps();
}
SizeType CohFDMTPlan::get_overlap_nsamps() const noexcept {
    return m_impl->block_nsamps() - m_impl->stride_nsamps();
}
SizeType CohFDMTPlan::get_msamp() const noexcept { return m_impl->msamp(); }
SizeType CohFDMTPlan::get_output_nsamps() const noexcept {
    return m_impl->nout();
}
double CohFDMTPlan::get_output_time_offset() const noexcept {
    return m_impl->output_time_offset();
}
SizeType CohFDMTPlan::get_input_size(SizeType igroup) const {
    return m_impl->input_size(igroup);
}
const std::vector<float>& CohFDMTPlan::get_dm_grid_coh() const noexcept {
    return m_impl->dm_grid_coh();
}
float CohFDMTPlan::get_dm_step_coh() const noexcept {
    return m_impl->dm_step_coh();
}
SizeType CohFDMTPlan::get_ndm_coh() const noexcept {
    return m_impl->dm_grid_coh().size();
}
SizeType CohFDMTPlan::get_ndm_fine() const noexcept {
    return m_impl->ndm_fine();
}
SizeType CohFDMTPlan::get_fine_dt_max() const noexcept {
    return m_impl->fine_dt_max();
}
const std::vector<float>& CohFDMTPlan::get_dm_grid_final() const noexcept {
    return m_impl->dm_grid_final();
}
SizeType CohFDMTPlan::get_ndm() const noexcept {
    return m_impl->dm_grid_final().size();
}
SizeType CohFDMTPlan::get_max_delay() const noexcept {
    return m_impl->max_delay();
}
SizeType CohFDMTPlan::get_fdmt_nsamps() const noexcept {
    return m_impl->fdmt_nsamps();
}
IndexType CohFDMTPlan::get_fdmt_window_start() const noexcept {
    return m_impl->fdmt_window_start();
}
std::span<const IndexType>
CohFDMTPlan::get_channel_shifts(SizeType idm_coh) const {
    return m_impl->channel_shifts(idm_coh);
}
const std::vector<SizeType>& CohFDMTPlan::get_row_offsets() const noexcept {
    return m_impl->row_offsets();
}
float CohFDMTPlan::get_intra_channel_smear() const noexcept {
    return m_impl->intra_channel_smear();
}
SizeType CohFDMTPlan::get_dmt_ndms() const noexcept {
    return m_impl->dm_grid_final().size();
}
SizeType CohFDMTPlan::get_dmt_nsamps() const noexcept { return m_impl->nout(); }
SizeType CohFDMTPlan::get_dmt_size() const noexcept {
    return m_impl->dmt_size();
}
SizeType CohFDMTPlan::get_buffer_size() const noexcept {
    return m_impl->dmt_size();
}
const std::vector<float>& CohFDMTPlan::get_channel_taper() const noexcept {
    return m_impl->taper();
}
std::vector<double> CohFDMTPlan::get_lag_correlation(SizeType max_lag) const {
    return m_impl->lag_correlation(max_lag);
}
std::vector<float>
CohFDMTPlan::get_effective_variance_grid(SizeType boxcar_width) const {
    return m_impl->variance_grid(boxcar_width);
}
std::vector<float>
CohFDMTPlan::get_effective_sigma_grid(SizeType boxcar_width) const {
    auto grid = m_impl->variance_grid(boxcar_width);
    for (auto& v : grid) {
        v = std::sqrt(v);
    }
    return grid;
}
std::vector<float> CohFDMTPlan::get_cumulative_count_grid() const {
    return m_impl->count_grid();
}
const FDMTPlan& CohFDMTPlan::get_fdmt_plan() const noexcept {
    return m_impl->fdmt_plan();
}
CohFDMTMemoryUsage CohFDMTPlan::get_memory_estimate() const noexcept {
    return m_impl->memory_estimate();
}
std::string CohFDMTPlan::summary() const { return m_impl->summary(); }

} // namespace dmt::plans
