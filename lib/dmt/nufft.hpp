#pragma once

/**
 * @file nufft.hpp
 * @brief 1D type-1 non-uniform FFT (CPU), used by DDMTFFT's NUFFT method.
 *
 * Computes, for M uniform modes d = 0 .. M-1,
 * @f[ f(d) = \sum_j a_j \, e^{+2\pi i\, d\, x_j} @f]
 * for arbitrary real points x_j (in turns; only x mod 1 matters) with
 * complex strengths a_j, to a relative accuracy ~tol of sum_j |a_j|.
 *
 * Standard algorithm (Dutt-Rokhlin, Greengard-Lee; kernel of Barnett et al.,
 * FINUFFT): shift to centred modes m = d - M/2, spread the points onto an
 * oversampled periodic grid of nf >= 2M cells with the "exponential of
 * semicircle" kernel psi(t) = exp(beta (sqrt(1 - (2t/w)^2) - 1)), |t| <=
 * w/2, take one length-nf FFT and divide the M wanted modes by the kernel's
 * Fourier transform.
 */

#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"
#include "dmt/fft.hpp"

namespace dmt::nufft {

class Type1Plan {
public:
    /// @param modes Number of output modes M (>= 1).
    /// @param tol Target relative accuracy (1e-7 .. 1e-2).
    /// @param nthreads Unused (execute() is single-threaded and thread-safe;
    /// callers run independent transforms in parallel).
    Type1Plan(SizeType modes, double tol, int nthreads = 1);
    ~Type1Plan();
    Type1Plan(Type1Plan&&) noexcept;
    Type1Plan& operator=(Type1Plan&&) noexcept;
    Type1Plan(const Type1Plan&)            = delete;
    Type1Plan& operator=(const Type1Plan&) = delete;

    /// @brief Complex scratch values execute() needs (64-byte aligned).
    [[nodiscard]] SizeType scratch_size() const noexcept;
    [[nodiscard]] SizeType modes() const noexcept;
    [[nodiscard]] SizeType fine_grid() const noexcept;
    [[nodiscard]] int width() const noexcept;

    /// @brief Mode offset of the centred transform: f(d) is computed as the
    /// centred mode m = d - half().
    [[nodiscard]] SizeType half() const noexcept;
    /// @brief Kernel shape: psi(t) = exp(beta (sqrt(1 - (2t/w)^2) - 1)).
    [[nodiscard]] double beta() const noexcept;
    /// @brief 1 / Psi(d - half()) for d < modes(): the deconvolution
    /// factors (other backends reuse them to compute the same transform).
    [[nodiscard]] std::span<const float> deconvolution() const noexcept;
    /// @brief The kernel on the w cells a point touches as polynomials in
    /// its offset u (see locate()): coefficient q of cell i at
    /// [i * (kernel_degree() + 1) + q], value sum_q c_q u^q (Horner, float),
    /// so other backends evaluate exactly the same kernel.
    [[nodiscard]] std::span<const float> kernel_monomials() const noexcept;
    [[nodiscard]] int kernel_degree() const noexcept;

    /// @brief f(d) for d < modes(); x.size() == a.size(), f.size() ==
    /// modes(). Thread-safe with a distinct @p scratch per thread.
    /// @param centred True if the caller already multiplied every strength
    /// by exp(2 pi i half() x_j) (saves a sincos per point).
    void execute(std::span<const double> x,
                 std::span<const ComplexType> a,
                 std::span<ComplexType> f,
                 utils::FFTVector<ComplexType>& scratch,
                 bool centred = false) const;

    /// @brief Scale of locate(): nf and w / 2.
    struct Locator {
        double nf;
        double half_width;
    };
    [[nodiscard]] Locator locator() const noexcept;

    /**
     * @brief f(d) from points already located on the fine grid: point j has
     * first cell @p l0[j] and kernel offset @p u[j] (see locate()) and
     * strength ar[j] + i ai[j], already centred (times exp(2 pi i half()
     * x_j)). The fast path: callers prepare the points vectorised.
     */
    void execute_points(std::span<const std::int32_t> l0,
                        std::span<const float> u,
                        std::span<const float> ar,
                        std::span<const float> ai,
                        std::span<ComplexType> f,
                        utils::FFTVector<ComplexType>& scratch) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Fine-grid position of a point x (turns): first touched cell l0 (in
 * [-w/2, nf)) and the kernel offset u in [-1, 1]. Vectorisable.
 */
#pragma omp declare simd uniform(loc)
inline void
locate(double x, Type1Plan::Locator loc, std::int32_t& l0, float& u) noexcept {
    const double g  = (x - std::floor(x)) * loc.nf; // [0, nf)
    const double lc = std::ceil(g - loc.half_width);
    l0              = static_cast<std::int32_t>(lc);
    u = static_cast<float>((2.0 * (lc - g + loc.half_width)) - 1.0);
}

} // namespace dmt::nufft
