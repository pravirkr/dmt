#include "dmt/fft_cuda.cuh"

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include "dmt/gpu_compat.cuh"

#include "dmt/logging.hpp"

#include "dmt/common/types.hpp"
#include "dmt/gpu_utils.cuh"

namespace dmt::utils {

namespace {

int to_cufft_int(SizeType value, const char* what) {
    if (value > static_cast<SizeType>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::format("CUFFTManager: {} exceeds cuFFT int limit", what));
    }
    return static_cast<int>(value);
}

void check_extent(SizeType count, SizeType stride, const char* what) {
    const bool fits =
        stride == 0 || count <= (std::numeric_limits<SizeType>::max() / stride);
    if (!fits) {
        throw std::invalid_argument(
            std::format("CUFFTManager: {} * batch overflows SizeType", what));
    }
}

} // namespace

class CUFFTManager::Impl {
public:
    Impl(FFTKind kind,
         SizeType length,
         SizeType howmany,
         int device_id,
         SizeType real_dist   = 0,
         SizeType freq_dist   = 0,
         SizeType real_stride = 1,
         SizeType freq_stride = 1)
        : m_kind(kind),
          m_length(length),
          m_howmany(howmany),
          m_n_complex(is_real(kind) ? (length / 2) + 1 : length),
          m_device_id(device_id) {
        if (length == 0 || howmany == 0) {
            throw std::invalid_argument(
                "CUFFTManager: length and howmany must be positive");
        }
        m_real_dist   = real_dist == 0 ? length : real_dist;
        m_freq_dist   = freq_dist == 0 ? m_n_complex : freq_dist;
        m_real_stride = std::max<SizeType>(real_stride, 1);
        m_freq_stride = std::max<SizeType>(freq_stride, 1);
        if ((m_real_stride == 1 && m_real_dist < length) ||
            (m_freq_stride == 1 && m_freq_dist < m_n_complex)) {
            throw std::invalid_argument(
                "CUFFTManager: row distance shorter than a row");
        }
        check_extent(howmany, m_real_dist, "real distance");
        check_extent(howmany, m_freq_dist, "spectrum distance");
        gpu_utils::set_device(m_device_id);
        try {
            create_plan();
        } catch (...) {
            if (m_plan != 0) {
                cufftDestroy(m_plan);
                m_plan = 0;
            }
            if (m_workspace != nullptr) {
                cudaFree(m_workspace);
                m_workspace = nullptr;
            }
            throw;
        }
        logging::debug("CUFFTManager: kind={} length={} howmany={} device={} "
                       "workspace={}",
                       static_cast<int>(kind), length, howmany, m_device_id,
                       m_workspace_size);
    }

    ~Impl() {
        try {
            gpu_utils::set_device(m_device_id);
        } catch (...) {
        }
        if (m_plan != 0) {
            cufftDestroy(m_plan);
            m_plan = 0;
        }
        if (m_workspace != nullptr) {
            cudaFree(m_workspace);
            m_workspace = nullptr;
        }
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void execute(cuda::std::span<ComplexTypeGPU> data,
                 cudaStream_t stream) const {
        if (m_kind != FFTKind::kC2CForward && m_kind != FFTKind::kC2CBackward) {
            throw std::logic_error(
                "CUFFTManager: execute(complex) requires a C2C plan");
        }
        const SizeType expected = extent(m_freq_dist, m_freq_stride, m_length);
        if (data.size() < expected) {
            throw std::invalid_argument(
                std::format("CUFFTManager: complex span size {} != {}",
                            data.size(), expected));
        }
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cufftSetStream(m_plan, stream),
                                  "CUFFTManager: cufftSetStream");
        auto* ptr = reinterpret_cast<cufftComplex*>(data.data());
        const int direction =
            m_kind == FFTKind::kC2CForward ? CUFFT_FORWARD : CUFFT_INVERSE;
        gpu_utils::check_gpu_call(cufftExecC2C(m_plan, ptr, ptr, direction),
                                  "CUFFTManager: cufftExecC2C");
    }

    void execute(cuda::std::span<float> real,
                 cuda::std::span<ComplexTypeGPU> freq,
                 cudaStream_t stream) const {
        if (m_kind != FFTKind::kR2C && m_kind != FFTKind::kC2R) {
            throw std::logic_error("CUFFTManager: execute(real, freq) requires "
                                   "an R2C or C2R plan");
        }
        const SizeType n_real =
            is_real(m_kind) ? extent(m_real_dist, m_real_stride, m_length) : 0;
        const SizeType n_complex =
            extent(m_freq_dist, m_freq_stride, m_n_complex);
        if (real.size() < n_real || freq.size() < n_complex) {
            throw std::invalid_argument(std::format(
                "CUFFTManager: span sizes real={} freq={} != {} and {}",
                real.size(), freq.size(), n_real, n_complex));
        }
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cufftSetStream(m_plan, stream),
                                  "CUFFTManager: cufftSetStream");
        auto* real_ptr = real.data();
        auto* freq_ptr = reinterpret_cast<cufftComplex*>(freq.data());
        if (m_kind == FFTKind::kR2C) {
            gpu_utils::check_gpu_call(cufftExecR2C(m_plan, real_ptr, freq_ptr),
                                      "CUFFTManager: cufftExecR2C");
        } else {
            gpu_utils::check_gpu_call(cufftExecC2R(m_plan, freq_ptr, real_ptr),
                                      "CUFFTManager: cufftExecC2R");
        }
    }

private:
    // Elements spanned by the batch.
    [[nodiscard]] SizeType
    extent(SizeType dist, SizeType stride, SizeType row) const noexcept {
        return ((m_howmany - 1) * dist) + ((row - 1) * stride) + 1;
    }

    static bool is_real(FFTKind kind) {
        return kind == FFTKind::kR2C || kind == FFTKind::kC2R;
    }

    void create_plan() {
        // 64-bit planning: batch * distance may exceed int (the length and
        // batch themselves are checked against cuFFT's int limits).
        long long n = to_cufft_int(m_length, "length");
        const auto howmany =
            static_cast<long long>(to_cufft_int(m_howmany, "howmany"));
        const auto rdist = static_cast<long long>(m_real_dist);
        const auto fdist = static_cast<long long>(m_freq_dist);
        const auto rstr  = static_cast<long long>(m_real_stride);
        const auto fstr  = static_cast<long long>(m_freq_stride);
        // Embedding: the row extent in elements (strides handle the rest).
        const auto rrow       = static_cast<long long>(m_length);
        const auto frow       = static_cast<long long>(m_n_complex);
        const bool in_freq    = m_kind == FFTKind::kC2R || !is_real(m_kind);
        const bool out_freq   = m_kind == FFTKind::kR2C || !is_real(m_kind);
        long long inembed[1]  = {in_freq ? frow : rrow};
        long long onembed[1]  = {out_freq ? frow : rrow};
        const long long idist = in_freq ? fdist : rdist;
        const long long odist = out_freq ? fdist : rdist;
        const long long istr  = in_freq ? fstr : rstr;
        const long long ostr  = out_freq ? fstr : rstr;
        const cufftType type  = real_type(m_kind);

        gpu_utils::check_gpu_call(cufftCreate(&m_plan),
                                  "CUFFTManager: cufftCreate");
        gpu_utils::check_gpu_call(cufftSetAutoAllocation(m_plan, 0),
                                  "CUFFTManager: cufftSetAutoAllocation");
        gpu_utils::check_gpu_call(
            cufftMakePlanMany64(m_plan, 1, &n, inembed, istr, idist, onembed,
                                ostr, odist, type, howmany, &m_workspace_size),
            "CUFFTManager: cufftMakePlanMany64");
        if (m_workspace_size > 0) {
            gpu_utils::check_gpu_call(
                cudaMalloc(&m_workspace, m_workspace_size),
                "CUFFTManager: cudaMalloc workspace");
            gpu_utils::check_gpu_call(cufftSetWorkArea(m_plan, m_workspace),
                                      "CUFFTManager: cufftSetWorkArea");
        }
    }

    static cufftType real_type(FFTKind kind) {
        switch (kind) {
        case FFTKind::kC2CForward:
        case FFTKind::kC2CBackward:
            return CUFFT_C2C;
        case FFTKind::kR2C:
            return CUFFT_R2C;
        case FFTKind::kC2R:
            return CUFFT_C2R;
        }
        throw std::invalid_argument("CUFFTManager: unknown kind");
    }

public:
    [[nodiscard]] SizeType workspace_bytes() const noexcept {
        return m_workspace_size;
    }

private:
    FFTKind m_kind;
    SizeType m_length;
    SizeType m_howmany;
    SizeType m_n_complex;
    SizeType m_real_dist{};
    SizeType m_freq_dist{};
    SizeType m_real_stride{1};
    SizeType m_freq_stride{1};
    int m_device_id;
    cufftHandle m_plan{0};
    void* m_workspace{nullptr};
    size_t m_workspace_size{0};
};

CUFFTManager::CUFFTManager(FFTKind kind,
                           SizeType length,
                           SizeType howmany,
                           int device_id)
    : m_impl(std::make_unique<Impl>(kind, length, howmany, device_id)) {}
CUFFTManager::CUFFTManager(FFTKind kind,
                           SizeType length,
                           SizeType howmany,
                           int device_id,
                           SizeType real_dist,
                           SizeType freq_dist,
                           SizeType real_stride,
                           SizeType freq_stride)
    : m_impl(std::make_unique<Impl>(kind,
                                    length,
                                    howmany,
                                    device_id,
                                    real_dist,
                                    freq_dist,
                                    real_stride,
                                    freq_stride)) {}
SizeType CUFFTManager::workspace_bytes() const noexcept {
    return m_impl->workspace_bytes();
}
CUFFTManager::~CUFFTManager()                                        = default;
CUFFTManager::CUFFTManager(CUFFTManager&& other) noexcept            = default;
CUFFTManager& CUFFTManager::operator=(CUFFTManager&& other) noexcept = default;

void CUFFTManager::execute(cuda::std::span<ComplexTypeGPU> data,
                           cudaStream_t stream) const {
    m_impl->execute(data, stream);
}
void CUFFTManager::execute(cuda::std::span<float> real,
                           cuda::std::span<ComplexTypeGPU> freq,
                           cudaStream_t stream) const {
    m_impl->execute(real, freq, stream);
}

} // namespace dmt::utils
