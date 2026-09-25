#include "dmt/utils/fft.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

#include <fftw3.h>
#include <spdlog/spdlog.h>

namespace dmt::utils {

namespace {

std::mutex& fftw_planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

void destroy_fftw_plan(fftwf_plan plan) noexcept {
    if (plan == nullptr) {
        return;
    }
    const std::scoped_lock lock(fftw_planner_mutex());
    fftwf_destroy_plan(plan);
}

class FFTWPlan {
public:
    explicit FFTWPlan(fftwf_plan plan) noexcept : m_plan(plan) {}
    ~FFTWPlan() { destroy_fftw_plan(m_plan); }
    FFTWPlan(const FFTWPlan&)            = delete;
    FFTWPlan& operator=(const FFTWPlan&) = delete;
    FFTWPlan(FFTWPlan&& other) noexcept
        : m_plan(std::exchange(other.m_plan, nullptr)) {}
    FFTWPlan& operator=(FFTWPlan&& other) noexcept {
        if (this != &other) {
            destroy_fftw_plan(m_plan);
            m_plan = std::exchange(other.m_plan, nullptr);
        }
        return *this;
    }

    [[nodiscard]] fftwf_plan get() const noexcept { return m_plan; }

private:
    fftwf_plan m_plan{nullptr};
};

int to_fftw_int(SizeType value, std::string_view what) {
    if (value > static_cast<SizeType>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::format("FFTWManager: {} exceeds FFTW int limit", what));
    }
    return static_cast<int>(value);
}

void check_extent(SizeType count, SizeType stride, std::string_view what) {
    const bool fits =
        stride == 0 || count <= (std::numeric_limits<SizeType>::max() / stride);
    if (!fits) {
        throw std::invalid_argument(
            std::format("FFTWManager: {} * batch overflows SizeType", what));
    }
}

FFTWPlan make_c2c_plan(SizeType length, SizeType howmany, int sign) {
    const int n         = to_fftw_int(length, "length");
    const int howmany_i = to_fftw_int(howmany, "howmany");
    fftwf_plan raw      = nullptr;
    {
        const std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_many_dft(1, &n, howmany_i, nullptr, nullptr, 1, n,
                                  nullptr, nullptr, 1, n, sign, FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        throw std::runtime_error(std::format(
            "FFTWManager: failed to create C2C plan (n={}, howmany={})", length,
            howmany));
    }
    return FFTWPlan{raw};
}

FFTWPlan make_r2c_plan(SizeType length, SizeType n_complex, SizeType howmany) {
    const int n_real_i    = to_fftw_int(length, "length");
    const int n_complex_i = to_fftw_int(n_complex, "n_complex");
    const int howmany_i   = to_fftw_int(howmany, "howmany");
    fftwf_plan raw        = nullptr;
    {
        const std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_many_dft_r2c(1, &n_real_i, howmany_i, nullptr, nullptr,
                                      1, n_real_i, nullptr, nullptr, 1,
                                      n_complex_i, FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        throw std::runtime_error(std::format(
            "FFTWManager: failed to create R2C plan (n={}, howmany={})", length,
            howmany));
    }
    return FFTWPlan{raw};
}

FFTWPlan make_c2r_plan(SizeType length, SizeType n_complex, SizeType howmany) {
    const int n_real_i    = to_fftw_int(length, "length");
    const int n_complex_i = to_fftw_int(n_complex, "n_complex");
    const int howmany_i   = to_fftw_int(howmany, "howmany");
    fftwf_plan raw        = nullptr;
    {
        const std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_many_dft_c2r(1, &n_real_i, howmany_i, nullptr, nullptr,
                                      1, n_complex_i, nullptr, nullptr, 1,
                                      n_real_i, FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        throw std::runtime_error(std::format(
            "FFTWManager: failed to create C2R plan (n={}, howmany={})", length,
            howmany));
    }
    return FFTWPlan{raw};
}

struct Slice {
    SizeType offset{};
    SizeType count{};
    bool use_extra{};
};

Slice slice_for(int worker, SizeType base, SizeType n_extra) {
    const auto w = static_cast<SizeType>(worker);
    if (w < n_extra) {
        return Slice{
            .offset    = w * (base + 1),
            .count     = base + 1,
            .use_extra = true,
        };
    }
    return Slice{
        .offset    = (n_extra * (base + 1)) + ((w - n_extra) * base),
        .count     = base,
        .use_extra = false,
    };
}

} // namespace

class FFTWManager::Impl {
public:
    Impl(FFTKind kind, SizeType length, SizeType howmany, int nthreads)
        : m_kind(kind),
          m_length(length),
          m_howmany(howmany),
          m_n_complex(is_real(kind) ? (length / 2) + 1 : length) {
        if (length == 0 || howmany == 0) {
            throw std::invalid_argument(
                std::format("FFTWManager: length and howmany must be positive: "
                            "length={}, howmany={}",
                            length, howmany));
        }
        check_extent(howmany, length, "length");
        if (is_real(kind)) {
            check_extent(howmany, m_n_complex, "n_complex");
        }

        const int threads = std::max(1, nthreads);
        m_n_workers =
            static_cast<int>(std::min(static_cast<SizeType>(threads), howmany));
        m_base    = howmany / static_cast<SizeType>(m_n_workers);
        m_n_extra = howmany % static_cast<SizeType>(m_n_workers);

        const SizeType extra_howmany = m_base + 1;
        if (m_n_extra == 0) {
            m_plan_base = make_plan(m_base);
        } else {
            m_plan_extra = make_plan(extra_howmany);
            m_plan_base  = make_plan(m_base);
        }
        spdlog::debug("FFTWManager: kind={} length={} howmany={} workers={} "
                      "base={} extra_workers={}",
                      static_cast<int>(kind), length, howmany, m_n_workers,
                      m_base, m_n_extra);
    }

    void execute(std::span<ComplexType> data) const {
        if (m_kind != FFTKind::kC2CForward && m_kind != FFTKind::kC2CBackward) {
            throw std::logic_error(
                "FFTWManager: execute(complex) requires a C2C plan");
        }
        const SizeType expected = m_howmany * m_length;
        if (data.size() != expected) {
            throw std::invalid_argument(
                std::format("FFTWManager: complex span size {} != {}",
                            data.size(), expected));
        }
        auto* ptr              = reinterpret_cast<fftwf_complex*>(data.data());
        const int n_workers    = m_n_workers;
        const SizeType base    = m_base;
        const SizeType n_extra = m_n_extra;
        const SizeType length  = m_length;
        fftwf_plan plan_base   = m_plan_base.get();
        fftwf_plan plan_extra  = m_plan_extra.get();
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(ptr, n_workers, base, n_extra, length, plan_base, plan_extra)
        for (int worker = 0; worker < n_workers; ++worker) {
            const Slice slice  = slice_for(worker, base, n_extra);
            fftwf_plan plan    = slice.use_extra ? plan_extra : plan_base;
            fftwf_complex* row = ptr + (slice.offset * length);
            fftwf_execute_dft(plan, row, row);
        }
    }

    void execute(std::span<float> real, std::span<ComplexType> freq) const {
        if (m_kind != FFTKind::kR2C && m_kind != FFTKind::kC2R) {
            throw std::logic_error("FFTWManager: execute(real, freq) requires "
                                   "an R2C or C2R plan");
        }
        const SizeType n_real    = m_howmany * m_length;
        const SizeType n_complex = m_howmany * m_n_complex;
        if (real.size() != n_real || freq.size() != n_complex) {
            throw std::invalid_argument(std::format(
                "FFTWManager: span sizes real={} freq={} != {} and {}",
                real.size(), freq.size(), n_real, n_complex));
        }
        auto* real_ptr         = real.data();
        auto* freq_ptr         = reinterpret_cast<fftwf_complex*>(freq.data());
        const int n_workers    = m_n_workers;
        const SizeType base    = m_base;
        const SizeType n_extra = m_n_extra;
        const SizeType length  = m_length;
        const SizeType n_freq  = m_n_complex;
        const bool forward     = m_kind == FFTKind::kR2C;
        fftwf_plan plan_base   = m_plan_base.get();
        fftwf_plan plan_extra  = m_plan_extra.get();
#pragma omp parallel for num_threads(n_workers) schedule(static) default(none) \
    shared(real_ptr, freq_ptr, n_workers, base, n_extra, length, n_freq,       \
               forward, plan_base, plan_extra)
        for (int worker = 0; worker < n_workers; ++worker) {
            const Slice slice       = slice_for(worker, base, n_extra);
            fftwf_plan plan         = slice.use_extra ? plan_extra : plan_base;
            float* real_row         = real_ptr + (slice.offset * length);
            fftwf_complex* freq_row = freq_ptr + (slice.offset * n_freq);
            if (forward) {
                fftwf_execute_dft_r2c(plan, real_row, freq_row);
            } else {
                fftwf_execute_dft_c2r(plan, freq_row, real_row);
            }
        }
    }

private:
    static bool is_real(FFTKind kind) {
        return kind == FFTKind::kR2C || kind == FFTKind::kC2R;
    }

    [[nodiscard]] FFTWPlan make_plan(SizeType howmany) const {
        switch (m_kind) {
        case FFTKind::kC2CForward:
            return make_c2c_plan(m_length, howmany, FFTW_FORWARD);
        case FFTKind::kC2CBackward:
            return make_c2c_plan(m_length, howmany, FFTW_BACKWARD);
        case FFTKind::kR2C:
            return make_r2c_plan(m_length, m_n_complex, howmany);
        case FFTKind::kC2R:
            return make_c2r_plan(m_length, m_n_complex, howmany);
        }
        throw std::invalid_argument("FFTWManager: unknown kind");
    }

    FFTKind m_kind;
    SizeType m_length;
    SizeType m_howmany;
    SizeType m_n_complex;
    int m_n_workers{1};
    SizeType m_base{0};
    SizeType m_n_extra{0};
    FFTWPlan m_plan_base{nullptr};
    FFTWPlan m_plan_extra{nullptr};
};

FFTWManager::FFTWManager(FFTKind kind,
                         SizeType length,
                         SizeType howmany,
                         int nthreads)
    : m_impl(std::make_unique<Impl>(kind, length, howmany, nthreads)) {}
FFTWManager::~FFTWManager()                                       = default;
FFTWManager::FFTWManager(FFTWManager&& other) noexcept            = default;
FFTWManager& FFTWManager::operator=(FFTWManager&& other) noexcept = default;

void FFTWManager::execute(std::span<ComplexType> data) const {
    m_impl->execute(data);
}
void FFTWManager::execute(std::span<float> real,
                          std::span<ComplexType> freq) const {
    m_impl->execute(real, freq);
}

} // namespace dmt::utils
