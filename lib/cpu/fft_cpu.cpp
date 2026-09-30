#include "dmt/fft.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <fftw3.h>
#include "dmt/common/fft_config.hpp"
#include "dmt/logging.hpp"

namespace dmt::utils {

namespace {

std::mutex& fftw_planner_mutex() {
    static std::mutex mutex;
    return mutex;
}

// Initial planner effort: DMT_FFTW_PLANNER (estimate, measure, patient or
// exhaustive), else ESTIMATE.
fft::Planner planner_from_env() noexcept {
    const char* env = std::getenv("DMT_FFTW_PLANNER");
    if (env == nullptr) {
        return fft::Planner::kEstimate;
    }
    std::string v(env);
    for (auto& ch : v) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    if (v == "measure") {
        return fft::Planner::kMeasure;
    }
    if (v == "patient") {
        return fft::Planner::kPatient;
    }
    if (v == "exhaustive") {
        return fft::Planner::kExhaustive;
    }
    return fft::Planner::kEstimate;
}

std::atomic<fft::Planner>& planner_setting() {
    static std::atomic<fft::Planner> planner{planner_from_env()};
    return planner;
}

// DMT_FFTW_WISDOM names a wisdom file: imported before the first plan and
// rewritten after every timed plan, so measured plans are paid for once per
// machine. Both run under the planner mutex.
const std::string& env_wisdom_path() {
    static const std::string path = [] {
        const char* env = std::getenv("DMT_FFTW_WISDOM");
        return env == nullptr ? std::string{} : std::string(env);
    }();
    return path;
}

void import_env_wisdom_locked() noexcept {
    static bool done = false;
    if (done) {
        return;
    }
    done             = true;
    const auto& path = env_wisdom_path();
    if (!path.empty()) {
        fftwf_import_wisdom_from_filename(path.c_str());
    }
}

void export_env_wisdom_locked(unsigned flags) noexcept {
    const auto& path = env_wisdom_path();
    if (!path.empty() && (flags & FFTW_ESTIMATE) == 0U) {
        fftwf_export_wisdom_to_filename(path.c_str());
    }
}

// Scratch arrays the planner may overwrite when it times candidates
// (anything but FFTW_ESTIMATE); null arrays are only valid for ESTIMATE.
struct PlanScratch {
    float* in{nullptr};
    float* out{nullptr};
    PlanScratch(SizeType in_floats, SizeType out_floats, unsigned flags) {
        if ((flags & FFTW_ESTIMATE) == 0U) {
            in  = fftwf_alloc_real(std::max<SizeType>(in_floats, 1));
            out = fftwf_alloc_real(std::max<SizeType>(out_floats, 1));
            if (in == nullptr || out == nullptr) {
                release();
                throw std::bad_alloc();
            }
        }
    }
    ~PlanScratch() { release(); }
    PlanScratch(const PlanScratch&)            = delete;
    PlanScratch& operator=(const PlanScratch&) = delete;
    void release() noexcept {
        fftwf_free(in);
        fftwf_free(out);
        in  = nullptr;
        out = nullptr;
    }
};

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

// FFTWManager executes on caller slices of any alignment. ESTIMATE plans
// keep their historical flags; timed plans are made alignment-agnostic.
unsigned manager_flags() {
    const unsigned flags = fftw_planner_flags();
    return (flags & FFTW_ESTIMATE) != 0U ? FFTW_ESTIMATE
                                         : (flags | FFTW_UNALIGNED);
}

FFTWPlan make_c2c_plan(SizeType length, SizeType howmany, int sign) {
    const int n         = to_fftw_int(length, "length");
    const int howmany_i = to_fftw_int(howmany, "howmany");
    fftwf_plan raw      = nullptr;
    {
        const std::scoped_lock lock(fftw_planner_mutex());
        const unsigned flags = manager_flags();
        import_env_wisdom_locked();
        const PlanScratch scratch(2 * length * howmany, 0, flags);
        auto* buf = reinterpret_cast<fftwf_complex*>(scratch.in);
        raw = fftwf_plan_many_dft(1, &n, howmany_i, buf, nullptr, 1, n, buf,
                                  nullptr, 1, n, sign, flags);
        export_env_wisdom_locked(flags);
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
        const unsigned flags = manager_flags();
        import_env_wisdom_locked();
        const PlanScratch scratch(length * howmany, 2 * n_complex * howmany,
                                  flags);
        raw = fftwf_plan_many_dft_r2c(
            1, &n_real_i, howmany_i, scratch.in, nullptr, 1, n_real_i,
            reinterpret_cast<fftwf_complex*>(scratch.out), nullptr, 1,
            n_complex_i, flags);
        export_env_wisdom_locked(flags);
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
        const unsigned flags = manager_flags();
        import_env_wisdom_locked();
        const PlanScratch scratch(2 * n_complex * howmany, length * howmany,
                                  flags);
        raw = fftwf_plan_many_dft_c2r(
            1, &n_real_i, howmany_i,
            reinterpret_cast<fftwf_complex*>(scratch.in), nullptr, 1,
            n_complex_i, scratch.out, nullptr, 1, n_real_i, flags);
        export_env_wisdom_locked(flags);
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
        logging::debug("FFTWManager: kind={} length={} howmany={} workers={} "
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

double r2c_cost_per_nlogn(SizeType n) {
    if (n < 2) {
        return 1.0;
    }
    const int ni = to_fftw_int(n, "length");
    FFTVector<float> in(n);
    FFTVector<ComplexType> out((n / 2) + 1);
    fftwf_plan raw = nullptr;
    {
        const std::scoped_lock lock(fftw_planner_mutex());
        raw = fftwf_plan_dft_r2c_1d(
            ni, in.data(), reinterpret_cast<fftwf_complex*>(out.data()),
            FFTW_ESTIMATE);
    }
    if (raw == nullptr) {
        return 1.0;
    }
    const FFTWPlan plan{raw}; // destroyed under its own lock
    const double nd = static_cast<double>(n);
    return fftwf_estimate_cost(plan.get()) / (nd * std::log2(nd));
}

unsigned fftw_planner_flags() noexcept {
    switch (planner_setting().load(std::memory_order_relaxed)) {
    case fft::Planner::kMeasure:
        return FFTW_MEASURE;
    case fft::Planner::kPatient:
        return FFTW_PATIENT;
    case fft::Planner::kExhaustive:
        return FFTW_EXHAUSTIVE;
    case fft::Planner::kEstimate:
        break;
    }
    return FFTW_ESTIMATE;
}

class FFTWRowPlan::Impl {
public:
    Impl(FFTKind kind,
         SizeType length,
         SizeType howmany,
         SizeType in_dist,
         SizeType out_dist)
        : m_length(length) {
        if (length == 0 || howmany == 0) {
            throw std::invalid_argument(
                "FFTWRowPlan: length and howmany must be positive");
        }
        const SizeType n_complex =
            (kind == FFTKind::kR2C || kind == FFTKind::kC2R) ? (length / 2) + 1
                                                             : length;
        // Natural distances: real rows `length`, complex rows `n_complex`.
        const bool real_in  = kind == FFTKind::kR2C;
        const bool real_out = kind == FFTKind::kC2R;
        const SizeType idist =
            in_dist != 0 ? in_dist : (real_in ? length : n_complex);
        const SizeType odist =
            out_dist != 0 ? out_dist : (real_out ? length : n_complex);
        const bool is_c2c =
            kind == FFTKind::kC2CForward || kind == FFTKind::kC2CBackward;
        if (is_c2c && idist != odist) {
            throw std::invalid_argument(
                "FFTWRowPlan: C2C plans are in-place (in_dist == out_dist)");
        }
        check_extent(howmany, idist, "in_dist");
        check_extent(howmany, odist, "out_dist");
        const int n     = to_fftw_int(length, "length");
        const int hm    = to_fftw_int(howmany, "howmany");
        const int id    = to_fftw_int(idist, "in_dist");
        const int od    = to_fftw_int(odist, "out_dist");
        const int in_n  = to_fftw_int(real_in ? length : n_complex, "n");
        const int out_n = to_fftw_int(real_out ? length : n_complex, "n");

        // Plan on kFFTAlignment-aligned scratch so the plan's alignment
        // assumptions hold for every aligned array given to execute.
        const SizeType in_floats  = (real_in ? 1 : 2) * idist * howmany;
        const SizeType out_floats = (real_out ? 1 : 2) * odist * howmany;
        FFTVector<float> in_buf(in_floats);
        FFTVector<float> out_buf(is_c2c ? 0 : out_floats);
        const std::scoped_lock lock(fftw_planner_mutex());
        const unsigned flags = fftw_planner_flags();
        import_env_wisdom_locked();
        fftwf_plan raw = nullptr;
        switch (kind) {
        case FFTKind::kR2C:
            raw = fftwf_plan_many_dft_r2c(
                1, &n, hm, in_buf.data(), &in_n, 1, id,
                reinterpret_cast<fftwf_complex*>(out_buf.data()), &out_n, 1, od,
                flags);
            break;
        case FFTKind::kC2R:
            raw = fftwf_plan_many_dft_c2r(
                1, &n, hm, reinterpret_cast<fftwf_complex*>(in_buf.data()),
                &in_n, 1, id, out_buf.data(), &out_n, 1, od, flags);
            break;
        case FFTKind::kC2CForward:
        case FFTKind::kC2CBackward: {
            auto* buf = reinterpret_cast<fftwf_complex*>(in_buf.data());
            raw       = fftwf_plan_many_dft(
                1, &n, hm, buf, &in_n, 1, id, buf, &out_n, 1, od,
                kind == FFTKind::kC2CForward ? FFTW_FORWARD : FFTW_BACKWARD,
                flags);
            break;
        }
        }
        export_env_wisdom_locked(flags);
        if (raw == nullptr) {
            throw std::runtime_error(std::format(
                "FFTWRowPlan: failed to create plan (kind={}, n={}, "
                "howmany={})",
                static_cast<int>(kind), length, howmany));
        }
        m_plan = FFTWPlan{raw};
    }

    [[nodiscard]] fftwf_plan get() const noexcept { return m_plan.get(); }
    [[nodiscard]] SizeType length() const noexcept { return m_length; }

private:
    SizeType m_length;
    FFTWPlan m_plan{nullptr};
};

FFTWRowPlan::FFTWRowPlan(FFTKind kind,
                         SizeType length,
                         SizeType howmany,
                         SizeType in_dist,
                         SizeType out_dist)
    : m_impl(std::make_unique<Impl>(kind, length, howmany, in_dist, out_dist)) {
}
FFTWRowPlan::~FFTWRowPlan()                                       = default;
FFTWRowPlan::FFTWRowPlan(FFTWRowPlan&& other) noexcept            = default;
FFTWRowPlan& FFTWRowPlan::operator=(FFTWRowPlan&& other) noexcept = default;

void FFTWRowPlan::r2c(float* in, ComplexType* out) const noexcept {
    fftwf_execute_dft_r2c(m_impl->get(), in,
                          reinterpret_cast<fftwf_complex*>(out));
}
void FFTWRowPlan::c2r(ComplexType* in, float* out) const noexcept {
    fftwf_execute_dft_c2r(m_impl->get(), reinterpret_cast<fftwf_complex*>(in),
                          out);
}
void FFTWRowPlan::c2c(ComplexType* data) const noexcept {
    auto* p = reinterpret_cast<fftwf_complex*>(data);
    fftwf_execute_dft(m_impl->get(), p, p);
}
SizeType FFTWRowPlan::length() const noexcept { return m_impl->length(); }

} // namespace dmt::utils

namespace dmt::fft {

void set_planner(Planner planner) noexcept {
    utils::planner_setting().store(planner, std::memory_order_relaxed);
}
Planner get_planner() noexcept {
    return utils::planner_setting().load(std::memory_order_relaxed);
}
bool import_wisdom(const std::string& path) {
    const std::scoped_lock lock(utils::fftw_planner_mutex());
    return fftwf_import_wisdom_from_filename(path.c_str()) != 0;
}
bool export_wisdom(const std::string& path) {
    const std::scoped_lock lock(utils::fftw_planner_mutex());
    return fftwf_export_wisdom_to_filename(path.c_str()) != 0;
}
void forget_wisdom() noexcept {
    const std::scoped_lock lock(utils::fftw_planner_mutex());
    fftwf_forget_wisdom();
}

} // namespace dmt::fft
