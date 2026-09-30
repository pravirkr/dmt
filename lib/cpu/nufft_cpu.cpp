#include "dmt/nufft.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "dmt/dm_utils.hpp"
#include "dmt/fft.hpp"
#include "dmt/simd_math.hpp"

namespace dmt::nufft {

namespace {

// Grid padding (cells) on each side, >= the widest kernel, so spreading
// never wraps inside the point loop; the pads are folded back once. 16
// complex floats keep the transformed part 64-byte aligned.
constexpr SizeType kPad = 16;
constexpr int kMaxWidth = 16;

// GCC drops __attribute__((vector_size(N))) on an alias template when N
// depends on the parameter, so kernel_vec_t<KP> becomes plain float and
// vec[i] fails ("subscripted value is neither array nor pointer"). Literal
// sizes in explicit specializations stay real vectors on GCC and Clang.
// Only the two widths the spreader instantiates are provided.
template <int KP> struct kernel_vec_type;

template <> struct kernel_vec_type<8> {
    using type = float __attribute__((vector_size(32)));
};

template <> struct kernel_vec_type<16> {
    using type = float __attribute__((vector_size(64)));
};

template <int KP> using kernel_vec_t = typename kernel_vec_type<KP>::type;

// Copy KP lanes into a real float[KP]. GNU vectors do not subscript reliably
// in C++ (a reference or a dependent type is not an array), and handing the
// vector object itself to spread_point trips -Warray-bounds. memcpy into an
// ordinary array is the portable materialization.
template <int KP>
inline void store_kernel_lanes(kernel_vec_t<KP> vec, float* dst) noexcept {
    static_assert(sizeof(kernel_vec_t<KP>) == KP * sizeof(float));
    std::memcpy(dst, &vec, sizeof(vec));
}

// dst[2k] += ar * v[k], dst[2k + 1] += ai * v[k] for k < KP: one point's
// kernel values (padded to KP lanes with zeros) spread onto KP interleaved
// complex cells.
template <int KP>
inline void spread_point(float* __restrict__ dst,
                         const float* __restrict__ v,
                         float ar,
                         float ai) noexcept {
#if defined(__AVX2__)
    const __m256 a = _mm256_setr_ps(ar, ai, ar, ai, ar, ai, ar, ai);
    for (int k = 0; k < KP; k += 8) {
        const __m256 kv = _mm256_loadu_ps(v + k);
        const __m256 lo = _mm256_unpacklo_ps(kv, kv); // k0 k0 k1 k1 | k4 ..
        const __m256 hi = _mm256_unpackhi_ps(kv, kv); // k2 k2 k3 k3 | k6 ..
        const __m256 d0 = _mm256_permute2f128_ps(lo, hi, 0x20);
        const __m256 d1 = _mm256_permute2f128_ps(lo, hi, 0x31);
        float* p        = dst + (2 * k);
        _mm256_storeu_ps(p, _mm256_fmadd_ps(d0, a, _mm256_loadu_ps(p)));
        _mm256_storeu_ps(p + 8, _mm256_fmadd_ps(d1, a, _mm256_loadu_ps(p + 8)));
    }
#elif defined(__ARM_NEON)
    const float32x4_t a = {ar, ai, ar, ai};
    for (int k = 0; k < KP; k += 4) {
        const float32x4_t kv = vld1q_f32(v + k);
        float* p             = dst + (2 * k);
        vst1q_f32(p, vfmaq_f32(vld1q_f32(p), vzip1q_f32(kv, kv), a));
        vst1q_f32(p + 4, vfmaq_f32(vld1q_f32(p + 4), vzip2q_f32(kv, kv), a));
    }
#else
#pragma omp simd
    for (int j = 0; j < 2 * KP; ++j) {
        dst[j] += ((j & 1) != 0 ? ai : ar) * v[j >> 1];
    }
#endif
}

// Gauss-Legendre nodes and weights on [-1, 1] (Newton on P_n).
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(static_cast<SizeType>(n));
    w.resize(static_cast<SizeType>(n));
    for (int i = 0; i < n; ++i) {
        double z  = std::cos(std::numbers::pi * (i + 0.75) / (n + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1.0;
            double p1 = z;
            for (int k = 2; k <= n; ++k) {
                const double p2 =
                    (((2.0 * k - 1.0) * z * p1) - ((k - 1.0) * p0)) / k;
                p0 = p1;
                p1 = p2;
            }
            dp             = n * ((z * p1) - p0) / ((z * z) - 1.0);
            const double d = p1 / dp;
            z -= d;
            if (std::abs(d) < 1e-16) {
                break;
            }
        }
        x[static_cast<SizeType>(i)] = z;
        w[static_cast<SizeType>(i)] = 2.0 / ((1.0 - (z * z)) * dp * dp);
    }
}

} // namespace

class Type1Plan::Impl {
public:
    Impl(SizeType modes, double tol) : m_modes(modes) {
        if (modes == 0) {
            throw std::invalid_argument("NUFFT: modes must be >= 1");
        }
        if (!utils::is_finite_bits(tol) || tol < 1.0E-7 || tol > 1.0E-2) {
            throw std::invalid_argument(
                std::format("NUFFT: tolerance {} outside [1e-7, 1e-2]", tol));
        }
        // Width and shape for upsampling 2 (Barnett et al. 2019).
        m_w = std::clamp(static_cast<int>(std::ceil(-std::log10(tol))) + 1, 2,
                         kMaxWidth);
        m_beta = 2.30 * m_w;
        // Fine grid: >= 2 M cells; among the FFT-friendly lengths up to 15%
        // longer, the one FFTW expects to transform fastest.
        const SizeType n_min =
            std::max<SizeType>(2 * modes, 2 * static_cast<SizeType>(m_w));
        m_nf        = utils::next_fft_size(n_min);
        double best = std::numeric_limits<double>::max();
        for (SizeType n = m_nf; n <= n_min + (n_min / 7);
             n          = utils::next_fft_size(n + 1)) {
            const double nd = static_cast<double>(n);
            const double cost =
                utils::r2c_cost_per_nlogn(n) * nd * std::log2(nd);
            if (cost < best) {
                best = cost;
                m_nf = n;
            }
        }
        m_half = modes / 2;
        m_fft  = std::make_unique<utils::FFTWRowPlan>(
            utils::FFTKind::kC2CBackward, m_nf);

        // Kernel Fourier transform at the centred modes m = d - half:
        // Psi(m) = int psi(t) cos(2 pi m t / nf) dt, t in [-w/2, w/2].
        std::vector<double> gx;
        std::vector<double> gw;
        gauss_legendre(2 * m_w + 40, gx, gw);
        const double hw = 0.5 * m_w;
        m_inv_psi.resize(modes);
        for (SizeType d = 0; d < modes; ++d) {
            const double m =
                static_cast<double>(d) - static_cast<double>(m_half);
            double acc = 0.0;
            for (SizeType q = 0; q < gx.size(); ++q) {
                const double t = hw * gx[q];
                const double z = gx[q];
                acc += gw[q] * hw *
                       std::exp(m_beta * (std::sqrt(1.0 - (z * z)) - 1.0)) *
                       std::cos(2.0 * std::numbers::pi * m * t /
                                static_cast<double>(m_nf));
            }
            m_inv_psi[d] = static_cast<float>(1.0 / acc);
        }

        // Kernel values on the w cells a point touches, as polynomials in
        // u = 2 s - 1, s in [0, 1) the point's offset from the first cell
        // (cell k holds psi(s - w/2 + k)): Chebyshev interpolation in double,
        // converted to monomials for Horner evaluation (the kernel is smooth
        // on each cell, so the float Horner stays accurate, as in FINUFFT).
        m_deg        = m_w + 2;
        const int np = m_deg + 1;
        m_cheb.assign(static_cast<SizeType>(m_w * np), 0.0F);
        for (int k = 0; k < m_w; ++k) {
            std::vector<double> val(static_cast<SizeType>(np));
            for (int j = 0; j < np; ++j) {
                const double u = std::cos(std::numbers::pi * (j + 0.5) / np);
                const double t = (0.5 * (u + 1.0)) - hw + k;
                const double z = t / hw;
                val[static_cast<SizeType>(j)] =
                    std::abs(z) >= 1.0
                        ? 0.0
                        : std::exp(m_beta * (std::sqrt(1.0 - (z * z)) - 1.0));
            }
            std::vector<double> cheb_d(static_cast<SizeType>(np));
            for (int n = 0; n < np; ++n) {
                double c = 0.0;
                for (int j = 0; j < np; ++j) {
                    c += val[static_cast<SizeType>(j)] *
                         std::cos(std::numbers::pi * n * (j + 0.5) / np);
                }
                c *= (n == 0 ? 1.0 : 2.0) / np;
                cheb_d[static_cast<SizeType>(n)] = c;
            }
            // Chebyshev -> monomial in u (double), for Horner evaluation.
            std::vector<double> mono(static_cast<SizeType>(np), 0.0);
            std::vector<double> tm2(static_cast<SizeType>(np), 0.0); // T_{n-2}
            std::vector<double> tm1(static_cast<SizeType>(np), 0.0); // T_{n-1}
            tm2[0] = 1.0;                                            // T_0
            if (np > 1) {
                tm1[1] = 1.0; // T_1
            }
            for (int n = 0; n < np; ++n) {
                std::vector<double> tn(static_cast<SizeType>(np), 0.0);
                if (n == 0) {
                    tn = tm2;
                } else if (n == 1) {
                    tn = tm1;
                } else {
                    for (int q = 0; q < np; ++q) {
                        const double up =
                            q > 0 ? tm1[static_cast<SizeType>(q - 1)] : 0.0;
                        tn[static_cast<SizeType>(q)] =
                            (2.0 * up) - tm2[static_cast<SizeType>(q)];
                    }
                    tm2 = tm1;
                    tm1 = tn;
                }
                for (int q = 0; q < np; ++q) {
                    mono[static_cast<SizeType>(q)] +=
                        cheb_d[static_cast<SizeType>(n)] *
                        tn[static_cast<SizeType>(q)];
                }
            }
            for (int q = 0; q < np; ++q) {
                m_cheb[static_cast<SizeType>((k * np) + q)] =
                    static_cast<float>(mono[static_cast<SizeType>(q)]);
            }
        }
        // Transposed [n][kp] copy, zero past w, for the per-point Horner.
        m_kp = m_w <= 8 ? 8 : 16;
        m_coef_t.assign(static_cast<SizeType>(np * m_kp), 0.0F);
        for (int k = 0; k < m_w; ++k) {
            for (int q = 0; q < np; ++q) {
                m_coef_t[static_cast<SizeType>((q * m_kp) + k)] =
                    m_cheb[static_cast<SizeType>((k * np) + q)];
            }
        }
    }

    // Generic points: locate, centre, then the located-point path.
    void execute(std::span<const double> x,
                 std::span<const ComplexType> a,
                 std::span<ComplexType> f,
                 utils::FFTVector<ComplexType>& scratch,
                 bool centred) const {
        const auto np = x.size();
        const Type1Plan::Locator loc{static_cast<double>(m_nf), 0.5 * m_w};
        const auto halfd = static_cast<double>(m_half);
        std::vector<std::int32_t> l0(np);
        std::vector<float> u(np);
        std::vector<float> ar(np);
        std::vector<float> ai(np);
        for (SizeType j = 0; j < np; ++j) {
            locate(x[j], loc, l0[j], u[j]);
            ComplexType aj = a[j];
            if (!centred) { // shift to centred modes: exp(2 pi i half x_j)
                float sn = 0.0F;
                float cs = 0.0F;
                simd::sincos_turns(
                    static_cast<float>(simd::wrap_turns(halfd * x[j])), sn, cs);
                aj *= ComplexType(cs, sn);
            }
            ar[j] = aj.real();
            ai[j] = aj.imag();
        }
        execute_points(l0.data(), u.data(), ar.data(), ai.data(), np, f,
                       scratch);
    }

    // Located points: per point, the kernel on its KP cells by one Horner
    // pass over a KP-lane vector (coefficients transposed, zero past w), then
    // one vector multiply-add into the grid. Consecutive points alternate
    // between two grid copies, so the read-modify-write of overlapping cells
    // (sorted channels land on neighbouring cells) does not serialise.
    template <int KP, int NC>
    void spread_located(const std::int32_t* l0,
                        const float* u,
                        const float* ar,
                        const float* ai,
                        SizeType np,
                        float* g0,
                        float* g1) const noexcept {
        // KP-lane kernel vectors as compiler vector types (GCC/Clang), so
        // the kP accumulators stay in registers on every ISA: kP * KP /
        // lanes independent Horner chains keep the FMA pipes busy (one point
        // alone is latency bound).
        using V          = kernel_vec_t<KP>;
        constexpr int kP = 8;
        const auto* c    = reinterpret_cast<const V*>(m_coef_t.data());
        V v[kP];
        SizeType j = 0;
        for (; j + kP <= np; j += kP) {
            for (int p = 0; p < kP; ++p) {
                v[p] = c[NC - 1];
            }
#pragma GCC unroll 24
            for (int n = NC - 2; n >= 0; --n) {
                const V cn = c[n];
                for (int p = 0; p < kP; ++p) {
                    v[p] = (v[p] * u[j + p]) + cn;
                }
            }
            for (int p = 0; p < kP; p += 2) {
                alignas(64) float t0[KP];
                alignas(64) float t1[KP];
                store_kernel_lanes<KP>(v[p], t0);
                store_kernel_lanes<KP>(v[p + 1], t1);
                spread_point<KP>(g0 + (2 * l0[j + p]), t0, ar[j + p],
                                 ai[j + p]);
                spread_point<KP>(g1 + (2 * l0[j + p + 1]), t1, ar[j + p + 1],
                                 ai[j + p + 1]);
            }
        }
        for (; j < np; ++j) {
            V w = c[NC - 1];
            for (int n = NC - 2; n >= 0; --n) {
                w = (w * u[j]) + c[n];
            }
            alignas(64) float t0[KP];
            store_kernel_lanes<KP>(w, t0);
            spread_point<KP>(g0 + (2 * l0[j]), t0, ar[j], ai[j]);
        }
    }

    template <int KP>
    void spread_located_nc(const std::int32_t* l0,
                           const float* u,
                           const float* ar,
                           const float* ai,
                           SizeType np,
                           float* g0,
                           float* g1) const noexcept {
        switch (m_deg + 1) {
#define DMT_NUFFT_NC(n)                                                        \
    case n:                                                                    \
        spread_located<KP, n>(l0, u, ar, ai, np, g0, g1);                      \
        return;
            DMT_NUFFT_NC(5)
            DMT_NUFFT_NC(6)
            DMT_NUFFT_NC(7)
            DMT_NUFFT_NC(8)
            DMT_NUFFT_NC(9)
            DMT_NUFFT_NC(10)
            DMT_NUFFT_NC(11)
            DMT_NUFFT_NC(12)
            DMT_NUFFT_NC(13)
            DMT_NUFFT_NC(14)
            DMT_NUFFT_NC(15)
            DMT_NUFFT_NC(16)
            DMT_NUFFT_NC(17)
            DMT_NUFFT_NC(18)
            DMT_NUFFT_NC(19)
            DMT_NUFFT_NC(20)
            DMT_NUFFT_NC(21)
#undef DMT_NUFFT_NC
        default:
            break;
        }
    }

    [[nodiscard]] SizeType points_scratch_size() const noexcept {
        return 2 * (m_nf + (2 * kPad));
    }

    void execute_points(const std::int32_t* l0,
                        const float* u,
                        const float* ar,
                        const float* ai,
                        SizeType np,
                        std::span<ComplexType> f,
                        utils::FFTVector<ComplexType>& scratch) const {
        const auto ext_n = m_nf + (2 * kPad);
        if (scratch.size() < 2 * ext_n) {
            scratch.assign(2 * ext_n, ComplexType{});
        }
        ComplexType* e0 = scratch.data();
        ComplexType* e1 = e0 + ext_n;
        std::fill_n(e0, 2 * ext_n, ComplexType{});
        auto* g0 = reinterpret_cast<float*>(e0 + kPad);
        auto* g1 = reinterpret_cast<float*>(e1 + kPad);
        if (m_kp == 8) {
            spread_located_nc<8>(l0, u, ar, ai, np, g0, g1);
        } else {
            spread_located_nc<16>(l0, u, ar, ai, np, g0, g1);
        }
        // Merge the copies, then fold the pads onto the periodic grid.
        auto* a       = reinterpret_cast<float*>(e0);
        const auto* b = reinterpret_cast<const float*>(e1);
#pragma omp simd
        for (SizeType i = 0; i < 2 * ext_n; ++i) {
            a[i] += b[i];
        }
        finish(e0, f);
    }

    // Pads -> periodic grid, FFT, deconvolution of the wanted modes.
    void finish(ComplexType* ext, std::span<ComplexType> f) const {
        ComplexType* grid = ext + kPad;
        for (SizeType l = 0; l < kPad; ++l) {
            grid[l] += grid[m_nf + l];       // right overflow
            grid[m_nf - kPad + l] += ext[l]; // left overflow
        }
        m_fft->c2c(grid);
        const auto half = static_cast<SizeType>(m_half);
        // m = d - half >= 0: grid[m]; m < 0: grid[nf + m].
        for (SizeType d = 0; d < std::min(half, m_modes); ++d) {
            f[d] = grid[m_nf - half + d] * m_inv_psi[d];
        }
        for (SizeType d = half; d < m_modes; ++d) {
            f[d] = grid[d - half] * m_inv_psi[d];
        }
    }

    SizeType m_modes;
    int m_w{};
    int m_deg{};
    std::vector<float> m_cheb; // [w][deg + 1] monomial coefficients
    utils::FFTVector<float>
        m_coef_t; // [deg + 1][kp], zero past w (64 B aligned)
    int m_kp{8};
    double m_beta{};
    SizeType m_nf{};
    SizeType m_half{};
    std::vector<float> m_inv_psi;
    std::unique_ptr<utils::FFTWRowPlan> m_fft;
};

Type1Plan::Type1Plan(SizeType modes, double tol, int /*nthreads*/)
    : m_impl(std::make_unique<Impl>(modes, tol)) {}
Type1Plan::~Type1Plan()                                     = default;
Type1Plan::Type1Plan(Type1Plan&& other) noexcept            = default;
Type1Plan& Type1Plan::operator=(Type1Plan&& other) noexcept = default;

SizeType Type1Plan::scratch_size() const noexcept {
    return m_impl->points_scratch_size();
}
SizeType Type1Plan::modes() const noexcept { return m_impl->m_modes; }
SizeType Type1Plan::fine_grid() const noexcept { return m_impl->m_nf; }
int Type1Plan::width() const noexcept { return m_impl->m_w; }

void Type1Plan::execute(std::span<const double> x,
                        std::span<const ComplexType> a,
                        std::span<ComplexType> f,
                        utils::FFTVector<ComplexType>& scratch,
                        bool centred) const {
    if (x.size() != a.size() || f.size() != m_impl->m_modes) {
        throw std::invalid_argument("NUFFT: size mismatch in execute()");
    }
    if (x.empty()) {
        std::fill(f.begin(), f.end(), ComplexType{});
        return;
    }
    m_impl->execute(x, a, f, scratch, centred);
}
SizeType Type1Plan::half() const noexcept { return m_impl->m_half; }
Type1Plan::Locator Type1Plan::locator() const noexcept {
    return {static_cast<double>(m_impl->m_nf), 0.5 * m_impl->m_w};
}
void Type1Plan::execute_points(std::span<const std::int32_t> l0,
                               std::span<const float> u,
                               std::span<const float> ar,
                               std::span<const float> ai,
                               std::span<ComplexType> f,
                               utils::FFTVector<ComplexType>& scratch) const {
    const auto np = l0.size();
    if (u.size() != np || ar.size() != np || ai.size() != np ||
        f.size() != m_impl->m_modes) {
        throw std::invalid_argument("NUFFT: size mismatch in execute_points()");
    }
    m_impl->execute_points(l0.data(), u.data(), ar.data(), ai.data(), np, f,
                           scratch);
}
double Type1Plan::beta() const noexcept { return m_impl->m_beta; }
std::span<const float> Type1Plan::deconvolution() const noexcept {
    return m_impl->m_inv_psi;
}
std::span<const float> Type1Plan::kernel_monomials() const noexcept {
    return m_impl->m_cheb;
}
int Type1Plan::kernel_degree() const noexcept { return m_impl->m_deg; }

} // namespace dmt::nufft
