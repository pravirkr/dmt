#include "dmt/utils/fft.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>

#include <spdlog/spdlog.h>

#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"

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
    Impl(FFTKind kind, SizeType length, SizeType howmany, int device_id)
        : m_kind(kind),
          m_length(length),
          m_howmany(howmany),
          m_n_complex(is_real(kind) ? (length / 2) + 1 : length),
          m_device_id(device_id) {
        if (length == 0 || howmany == 0) {
            throw std::invalid_argument(
                "CUFFTManager: length and howmany must be positive");
        }
        check_extent(howmany, length, "length");
        if (is_real(kind)) {
            check_extent(howmany, m_n_complex, "n_complex");
        }
        cuda_utils::set_device(m_device_id);
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
        spdlog::debug("CUFFTManager: kind={} length={} howmany={} device={} "
                      "workspace={}",
                      static_cast<int>(kind), length, howmany, m_device_id,
                      m_workspace_size);
    }

    ~Impl() {
        try {
            cuda_utils::set_device(m_device_id);
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

    void execute(cuda::std::span<ComplexTypeCUDA> data,
                 cudaStream_t stream) const {
        if (m_kind != FFTKind::kC2CForward && m_kind != FFTKind::kC2CBackward) {
            throw std::logic_error(
                "CUFFTManager: execute(complex) requires a C2C plan");
        }
        const SizeType expected = m_howmany * m_length;
        if (data.size() != expected) {
            throw std::invalid_argument(
                std::format("CUFFTManager: complex span size {} != {}",
                            data.size(), expected));
        }
        cuda_utils::set_device(m_device_id);
        cuda_utils::check_cuda_call(cufftSetStream(m_plan, stream),
                                    "CUFFTManager: cufftSetStream");
        auto* ptr = reinterpret_cast<cufftComplex*>(data.data());
        const int direction =
            m_kind == FFTKind::kC2CForward ? CUFFT_FORWARD : CUFFT_INVERSE;
        cuda_utils::check_cuda_call(cufftExecC2C(m_plan, ptr, ptr, direction),
                                    "CUFFTManager: cufftExecC2C");
    }

    void execute(cuda::std::span<float> real,
                 cuda::std::span<ComplexTypeCUDA> freq,
                 cudaStream_t stream) const {
        if (m_kind != FFTKind::kR2C && m_kind != FFTKind::kC2R) {
            throw std::logic_error("CUFFTManager: execute(real, freq) requires "
                                   "an R2C or C2R plan");
        }
        const SizeType n_real    = m_howmany * m_length;
        const SizeType n_complex = m_howmany * m_n_complex;
        if (real.size() != n_real || freq.size() != n_complex) {
            throw std::invalid_argument(std::format(
                "CUFFTManager: span sizes real={} freq={} != {} and {}",
                real.size(), freq.size(), n_real, n_complex));
        }
        cuda_utils::set_device(m_device_id);
        cuda_utils::check_cuda_call(cufftSetStream(m_plan, stream),
                                    "CUFFTManager: cufftSetStream");
        auto* real_ptr = real.data();
        auto* freq_ptr = reinterpret_cast<cufftComplex*>(freq.data());
        if (m_kind == FFTKind::kR2C) {
            cuda_utils::check_cuda_call(
                cufftExecR2C(m_plan, real_ptr, freq_ptr),
                "CUFFTManager: cufftExecR2C");
        } else {
            cuda_utils::check_cuda_call(
                cufftExecC2R(m_plan, freq_ptr, real_ptr),
                "CUFFTManager: cufftExecC2R");
        }
    }

private:
    static bool is_real(FFTKind kind) {
        return kind == FFTKind::kR2C || kind == FFTKind::kC2R;
    }

    void create_plan() {
        int n             = to_cufft_int(m_length, "length");
        const int howmany = to_cufft_int(m_howmany, "howmany");
        const int n_freq  = to_cufft_int(m_n_complex, "n_complex");
        const int idist =
            is_real(m_kind) && m_kind == FFTKind::kC2R ? n_freq : n;
        const int odist =
            is_real(m_kind) && m_kind == FFTKind::kR2C ? n_freq : n;
        const cufftType type = real_type(m_kind);

        cuda_utils::check_cuda_call(cufftCreate(&m_plan),
                                    "CUFFTManager: cufftCreate");
        cuda_utils::check_cuda_call(cufftSetAutoAllocation(m_plan, 0),
                                    "CUFFTManager: cufftSetAutoAllocation");
        cuda_utils::check_cuda_call(
            cufftMakePlanMany(m_plan, 1, &n, nullptr, 1, idist, nullptr, 1,
                              odist, type, howmany, &m_workspace_size),
            "CUFFTManager: cufftMakePlanMany");
        if (m_workspace_size > 0) {
            cuda_utils::check_cuda_call(
                cudaMalloc(&m_workspace, m_workspace_size),
                "CUFFTManager: cudaMalloc workspace");
            cuda_utils::check_cuda_call(cufftSetWorkArea(m_plan, m_workspace),
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

    FFTKind m_kind;
    SizeType m_length;
    SizeType m_howmany;
    SizeType m_n_complex;
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
CUFFTManager::~CUFFTManager()                                        = default;
CUFFTManager::CUFFTManager(CUFFTManager&& other) noexcept            = default;
CUFFTManager& CUFFTManager::operator=(CUFFTManager&& other) noexcept = default;

void CUFFTManager::execute(cuda::std::span<ComplexTypeCUDA> data,
                           cudaStream_t stream) const {
    m_impl->execute(data, stream);
}
void CUFFTManager::execute(cuda::std::span<float> real,
                           cuda::std::span<ComplexTypeCUDA> freq,
                           cudaStream_t stream) const {
    m_impl->execute(real, freq, stream);
}

} // namespace dmt::utils
