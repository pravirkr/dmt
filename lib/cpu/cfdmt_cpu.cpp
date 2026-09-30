#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <omp.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/cfdmt_common.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"
#include "dmt/fft.hpp"
#include "dmt/simd_math.hpp"
#include "dmt/unpacker.hpp"

namespace dmt::algorithms {

namespace {

// CPU engine. Per block:
//   unpack -> forward FFT (once) -> channel noise statistics (once) ->
//   per coarse trial: fused per-channel coherent stage (gather + chirp +
//   inverse FFT + detect + normalise + trim + delay-aligned write) -> fine
//   FDMT -> crop copy of the valid output window.
class CohFDMTCpuEngine final : public detail::CohFDMTEngine {
public:
    CohFDMTCpuEngine(const plans::CohFDMTPlan& plan,
                     const detail::CohFDMTEngineConfig& cfg)
        : m_plan(plan),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_unpacker(plan.get_format(),
                     plan.get_subband_groups(),
                     plan.get_nbin(),
                     plan.get_nfft(),
                     plan.get_noverlap(),
                     m_nthreads),
          m_fwd(utils::FFTKind::kC2CForward,
                plan.get_nbin(),
                2,
                plan.get_nbin(),
                plan.get_nbin()),
          m_mstride(utils::fft_row_stride<ComplexType>(plan.get_mbin())),
          m_fdmt(plan.get_f_min(),
                 plan.get_f_max(),
                 plan.get_nchans(),
                 plan.get_fdmt_nsamps(),
                 plan.get_tsamp(),
                 static_cast<IndexType>(plan.get_fine_dt_max()),
                 -static_cast<IndexType>(plan.get_fine_dt_max()),
                 plan.get_config().dt_step,
                 true,
                 "valid",
                 Exec::cpu(m_nthreads)) {
        const auto nchans = plan.get_nchans();
        const auto mbin   = plan.get_mbin();
        m_spec.resize(m_unpacker.output_size());
        m_phases = cfdmt::make_chirp_phases(plan);
        const auto& taper = plan.get_channel_taper();
        const float scale = 1.0F / static_cast<float>(plan.get_nbin());
        m_taper.resize(mbin);
        for (SizeType b = 0; b < mbin; ++b) {
            m_taper[b] = taper[b] * scale;
        }
        m_mean.assign(nchans, 0.0F);
        m_inv_sigma.assign(nchans, 1.0F);
        m_power.assign(plan.get_nsub() * plan.get_nfft() * plan.get_n_p() * 2,
                       0.0F);
        m_waterfall.resize(nchans * plan.get_fdmt_nsamps());
        m_fdmt_out.resize(m_fdmt.get_plan().get_buffer_size());
        // Inverse transforms of up to kChunk FFT blocks (both pols) per call.
        const auto lc   = mbin - (2 * (plan.get_noverlap() / plan.get_n_p()));
        const auto nwin = ((plan.get_fdmt_nsamps() + lc - 1) / lc) + 1;
        m_chunk         = std::min<SizeType>(kChunk, nwin);
        for (SizeType n = 1; n <= m_chunk; ++n) {
            m_inv.emplace_back(utils::FFTKind::kC2CBackward, mbin, 2 * n,
                               m_mstride, m_mstride);
        }
        // Per thread: chirp (mbin) + two pol rows (2 * mstride).
        m_scratch.resize(static_cast<SizeType>(m_nthreads) * scratch_stride());
    }

    void execute(std::span<const std::span<const uint8_t>> groups,
                 std::span<float> dmt) override {
        if (dmt.size() < m_plan.get_dmt_size()) {
            throw std::invalid_argument(std::format(
                "CohFDMT: dmt buffer too small. Expected at least {} "
                "(get_dmt_size()), got {}",
                m_plan.get_dmt_size(), dmt.size()));
        }
        front_end(groups);
        const auto ndm_coh = m_plan.get_ndm_coh();
        for (SizeType k = 0; k < ndm_coh; ++k) {
            // No reset_history(): the FDMT window's lead-in keeps its
            // history out of every cropped output (see CohFDMTPlan).
            coherent_trial(k);
            m_fdmt.execute(m_waterfall, m_fdmt_out);
            crop_rows(k, dmt);
        }
    }

    [[nodiscard]] plans::CohFDMTMemoryUsage
    memory_usage() const noexcept override {
        const auto fdmt_mem = m_fdmt.get_memory_usage();
        return {.spectrum  = m_spec.size() * sizeof(ComplexType),
                .waterfall = m_waterfall.size() * sizeof(float),
                .fdmt      = fdmt_mem.total() +
                        (m_fdmt_out.size() * sizeof(float)),
                .workspace = ((m_phases.base.size() + m_phases.inc.size()) *
                              sizeof(uint64_t)) +
                             (m_scratch.size() * sizeof(ComplexType)) +
                             ((m_mean.size() + m_inv_sigma.size() +
                               m_taper.size()) *
                              sizeof(float)),
                .output    = m_plan.get_dmt_size() * sizeof(float)};
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    const plans::CohFDMTPlan& m_plan; // owned by the CohFDMT facade
    int m_nthreads;
    static constexpr SizeType kChunk = 8;

    utils::BasebandUnpackerCPU m_unpacker;
    utils::FFTWRowPlan m_fwd; // both pols of one FFT block
    SizeType m_mstride;
    algorithms::FDMT m_fdmt;
    SizeType m_chunk{1};
    std::vector<utils::FFTWRowPlan> m_inv; // [n - 1]: 2 * n inverse rows

    utils::FFTVector<ComplexType> m_spec; // (sub, ifft, pol, nbin)
    std::vector<float> m_power;           // (sub, ifft, chan, pol) partials
    cfdmt::ChirpPhases m_phases;
    std::vector<float> m_taper; // channel response / nbin
    std::vector<float> m_mean;
    std::vector<float> m_inv_sigma;
    std::vector<float> m_waterfall; // (nchans, fdmt_nsamps)
    std::vector<float> m_fdmt_out;
    utils::FFTVector<ComplexType> m_scratch;

    [[nodiscard]] SizeType scratch_stride() const noexcept {
        return m_mstride * (1 + (2 * m_chunk));
    }

    // Offset of channel ichan's first bin in the unshifted spectrum: the
    // fftshift is folded into the gather index.
    [[nodiscard]] SizeType bin_offset(SizeType ichan) const noexcept {
        const auto nbin = m_plan.get_nbin();
        return ((ichan * m_plan.get_mbin()) + (nbin / 2)) % nbin;
    }

    // Fused front end, one task per (subband, FFT block): unpack both
    // polarisations, forward transform them in place while they are in
    // cache and (if normalising) accumulate each channel's filtered power.
    //
    // Per channel the mean power of each polarisation after the channel
    // filter is E|y_p|^2 = sum_b |c_b|^2 |X_p(b)|^2 (Parseval; independent of
    // the chirp phase, so shared by all coarse trials). Detected I = |y_0|^2
    // + |y_1|^2 of complex Gaussian noise has mean m0 + m1 and variance
    // m0^2 + m1^2.
    void front_end(std::span<const std::span<const uint8_t>> groups) {
        m_unpacker.validate(groups);
        const auto n_p    = m_plan.get_n_p();
        const auto nsub   = m_plan.get_nsub();
        const auto nfft   = m_plan.get_nfft();
        const auto nbin   = m_plan.get_nbin();
        const auto mbin   = m_plan.get_mbin();
        const bool norm   = m_plan.get_config().normalize;
        ComplexType* spec = m_spec.data();
        const float* taper = m_taper.data();
        float* power       = m_power.data();
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType task = 0; task < nsub * nfft; ++task) {
            ComplexType* row0 = spec + (task * 2 * nbin);
            ComplexType* row1 = row0 + nbin;
            m_unpacker.unpack_block(groups, task / nfft, task % nfft, row0,
                                    row1);
            m_fwd.c2c(row0);
            if (!norm) {
                continue;
            }
            for (SizeType ichan = 0; ichan < n_p; ++ichan) {
                const SizeType off  = bin_offset(ichan);
                const SizeType head = std::min(mbin, nbin - off);
                for (SizeType p = 0; p < 2; ++p) {
                    const auto* x =
                        reinterpret_cast<const float*>(p == 0 ? row0 : row1);
                    float acc = 0.0F;
#pragma omp simd reduction(+ : acc)
                    for (SizeType b = 0; b < head; ++b) {
                        const float re = x[2 * (off + b)];
                        const float im = x[(2 * (off + b)) + 1];
                        acc += taper[b] * taper[b] * ((re * re) + (im * im));
                    }
#pragma omp simd reduction(+ : acc)
                    for (SizeType b = head; b < mbin; ++b) {
                        const float re = x[2 * (b - head)];
                        const float im = x[(2 * (b - head)) + 1];
                        acc += taper[b] * taper[b] * ((re * re) + (im * im));
                    }
                    power[(((task * n_p) + ichan) * 2) + p] = acc;
                }
            }
        }
        if (!norm) {
            return;
        }
        const auto nchans = m_plan.get_nchans();
        for (SizeType c = 0; c < nchans; ++c) {
            const SizeType isub  = c / n_p;
            const SizeType ichan = c % n_p;
            double m[2]          = {0.0, 0.0};
            for (SizeType j = 0; j < nfft; ++j) {
                const float* pw =
                    power + (((((isub * nfft) + j) * n_p) + ichan) * 2);
                m[0] += static_cast<double>(pw[0]);
                m[1] += static_cast<double>(pw[1]);
            }
            m[0] /= static_cast<double>(nfft);
            m[1] /= static_cast<double>(nfft);
            const double sigma = std::sqrt((m[0] * m[0]) + (m[1] * m[1]));
            m_mean[c]          = static_cast<float>(m[0] + m[1]);
            m_inv_sigma[c] = sigma > 0.0 ? static_cast<float>(1.0 / sigma) : 0.0F;
        }
    }

    // Aligned waterfall of coarse trial k: row c holds channel c's detected
    // intensity at aligned times [a, a + fdmt_nsamps) (a = the plan's FDMT
    // window start), i.e. channel samples m = t - S_c(k); samples outside the
    // block are zero.
    void coherent_trial(SizeType k) {
        const auto nchans  = m_plan.get_nchans();
        const auto n_p     = m_plan.get_n_p();
        const auto nsub    = m_plan.get_nsub();
        const auto nfft    = m_plan.get_nfft();
        const auto nbin    = m_plan.get_nbin();
        const auto mbin    = m_plan.get_mbin();
        const auto novc    = m_plan.get_noverlap() / n_p;
        const auto lc      = mbin - (2 * novc);
        const auto msamp   = static_cast<IndexType>(m_plan.get_msamp());
        const auto nf      = static_cast<IndexType>(m_plan.get_fdmt_nsamps());
        const auto a       = m_plan.get_fdmt_window_start();
        const auto shifts  = m_plan.get_channel_shifts(k);
        const bool norm    = m_plan.get_config().normalize;
        const auto kk      = static_cast<uint64_t>(k);
        const ComplexType* spec  = m_spec.data();
        ComplexType* scratch_all = m_scratch.data();
        const SizeType sstride   = scratch_stride();
        const SizeType ms        = m_mstride;

#pragma omp parallel for num_threads(m_nthreads) schedule(dynamic, 4)
        for (SizeType c = 0; c < nchans; ++c) {
            ComplexType* scratch =
                scratch_all +
                (static_cast<SizeType>(omp_get_thread_num()) * sstride);
            ComplexType* chirp = scratch;
            ComplexType* rows  = scratch + ms; // (block, pol) rows
            float* row = m_waterfall.data() + (c * static_cast<SizeType>(nf));
            const IndexType s    = shifts[c];
            const IndexType m_lo = std::max<IndexType>(0, a - s);
            const IndexType m_hi = std::min<IndexType>(msamp, a + nf - s);
            if (m_lo >= m_hi) {
                std::fill_n(row, nf, 0.0F);
                continue;
            }
            // Zero the parts of the window outside the block.
            std::fill_n(row, m_lo + s - a, 0.0F);
            std::fill(row + (m_hi + s - a), row + nf, 0.0F);

            // Chirp of trial k for this channel.
            const uint64_t* base = m_phases.base.data() + (c * mbin);
            const uint64_t* inc  = m_phases.inc.data() + (c * mbin);
            const float* taper   = m_taper.data();
#pragma omp simd
            for (SizeType b = 0; b < mbin; ++b) {
                float sn = 0.0F;
                float cs = 0.0F;
                simd::sincos_turns(cfdmt::turn_of(base[b] + (kk * inc[b])), sn,
                                   cs);
                chirp[b] = {taper[b] * cs, taper[b] * sn};
            }

            const SizeType isub  = c / n_p;
            const SizeType off   = bin_offset(c % n_p);
            const SizeType head  = std::min(mbin, nbin - off); // before wrap
            const float mean     = m_mean[c];
            const float inv_sig  = m_inv_sigma[c];
            const auto j_first   = static_cast<SizeType>(m_lo) / lc;
            const auto j_last    = static_cast<SizeType>(m_hi - 1) / lc;
            float* dst           = row + (s - a);
            for (SizeType j0 = j_first; j0 <= j_last; j0 += m_chunk) {
                const SizeType nj = std::min(m_chunk, j_last + 1 - j0);
                for (SizeType q = 0; q < nj; ++q) {
                    const ComplexType* x =
                        spec + ((((isub * nfft) + j0 + q) * 2) * nbin);
                    gather(x + off, x, chirp, rows + (2 * q * ms), head, mbin);
                    gather(x + nbin + off, x + nbin, chirp,
                           rows + (((2 * q) + 1) * ms), head, mbin);
                }
                m_inv[nj - 1].c2c(rows);
                for (SizeType q = 0; q < nj; ++q) {
                    // Kept samples of block j: channel samples [j lc,
                    // (j + 1) lc) from inverse-FFT bins [novc, novc + lc).
                    const auto m0 = static_cast<IndexType>((j0 + q) * lc);
                    const IndexType beg = std::max(m_lo, m0);
                    const IndexType end =
                        std::min(m_hi, m0 + static_cast<IndexType>(lc));
                    if (beg >= end) {
                        continue;
                    }
                    const auto first = novc + static_cast<SizeType>(beg - m0);
                    detect(reinterpret_cast<const float*>(rows + (2 * q * ms) +
                                                          first),
                           reinterpret_cast<const float*>(
                               rows + (((2 * q) + 1) * ms) + first),
                           dst + beg, static_cast<SizeType>(end - beg), norm,
                           mean, inv_sig);
                }
            }
        }
    }

    // dst[i] = (|y0[i]|^2 + |y1[i]|^2 - mean) * inv_sig, i < n (interleaved
    // complex y).
    static void detect(const float* __restrict__ y0,
                       const float* __restrict__ y1,
                       float* __restrict__ dst,
                       SizeType n,
                       bool norm,
                       float mean,
                       float inv_sig) noexcept {
        if (!norm) {
            mean    = 0.0F;
            inv_sig = 1.0F;
        }
#pragma omp simd
        for (SizeType i = 0; i < n; ++i) {
            const float v = (y0[2 * i] * y0[2 * i]) +
                            (y0[(2 * i) + 1] * y0[(2 * i) + 1]) +
                            (y1[2 * i] * y1[2 * i]) +
                            (y1[(2 * i) + 1] * y1[(2 * i) + 1]);
            dst[i] = (v - mean) * inv_sig;
        }
    }

    // dst[b] = src[(off + b) mod nbin] * chirp[b]: bins [0, head) come from
    // src_off, the rest wrap to the start of the row.
    static void gather(const ComplexType* src_off,
                       const ComplexType* src_row,
                       const ComplexType* chirp,
                       ComplexType* dst,
                       SizeType head,
                       SizeType mbin) noexcept {
        const auto* s = reinterpret_cast<const float*>(src_off);
        const auto* w = reinterpret_cast<const float*>(src_row);
        const auto* h = reinterpret_cast<const float*>(chirp);
        auto* d       = reinterpret_cast<float*>(dst);
#pragma omp simd
        for (SizeType b = 0; b < head; ++b) {
            const float xr = s[2 * b];
            const float xi = s[(2 * b) + 1];
            const float hr = h[2 * b];
            const float hi = h[(2 * b) + 1];
            d[2 * b]       = (xr * hr) - (xi * hi);
            d[(2 * b) + 1] = (xr * hi) + (xi * hr);
        }
#pragma omp simd
        for (SizeType b = head; b < mbin; ++b) {
            const SizeType q = b - head;
            const float xr   = w[2 * q];
            const float xi   = w[(2 * q) + 1];
            const float hr   = h[2 * b];
            const float hi   = h[(2 * b) + 1];
            d[2 * b]         = (xr * hr) - (xi * hi);
            d[(2 * b) + 1]   = (xr * hi) + (xi * hr);
        }
    }

    // Row r of trial k: the FDMT output row from its reference offset.
    void crop_rows(SizeType k, std::span<float> dmt) const {
        const auto nfine   = m_plan.get_ndm_fine();
        const auto nout    = m_plan.get_output_nsamps();
        const auto nf      = m_plan.get_fdmt_nsamps();
        const auto& offs   = m_plan.get_row_offsets();
        const float* src   = m_fdmt_out.data();
        float* dst         = dmt.data() + (k * nfine * nout);
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
        for (SizeType r = 0; r < nfine; ++r) {
            std::copy_n(src + (r * nf) + offs[r], nout, dst + (r * nout));
        }
    }
};

} // namespace

std::unique_ptr<detail::CohFDMTEngine>
detail::make_cfdmt_cpu(const plans::CohFDMTPlan& plan,
                       const detail::CohFDMTEngineConfig& cfg) {
    return std::make_unique<CohFDMTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
