#include "dmt/algorithms/ddmt.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_kernel.cuh"
#include "dmt/engines.hpp"
#include "dmt/gpu_utils.cuh"
#include "dmt/plans_cuda.cuh"

namespace dmt::algorithms {

namespace {

// Output samples processed per "gulp" (chunk); bounds working GPU/pinned
// memory for very large inputs and lets H2D copy / kernel / D2H copy for
// consecutive gulps overlap via double buffering. Same order of magnitude as
// dedisp's own default gulp size.
constexpr SizeType kGulpOutSamples = 65536;

/**
 * @brief RAII page-locked ("pinned") host buffer.
 * @details
 * cudaHostAlloc'd memory transfers 2-3x faster than regular heap memory and
 * is required for cudaMemcpyAsync to actually run asynchronously with
 * respect to the host -- see dedisp's TDDPlan (cu::HostMemory) for the same
 * rationale. reserve() never shrinks, so repeated gulps reuse one
 * allocation.
 */
template <typename T> class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer&)            = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&&)                 = delete;
    PinnedBuffer& operator=(PinnedBuffer&&)      = delete;

    void reserve(SizeType n) {
        if (n <= m_capacity) {
            return;
        }
        release();
        gpu_utils::check_gpu_call(
            cudaHostAlloc(reinterpret_cast<void**>(&m_ptr), n * sizeof(T),
                          cudaHostAllocDefault),
            "cudaHostAlloc failed");
        m_capacity = n;
    }
    [[nodiscard]] T* data() noexcept { return m_ptr; }
    [[nodiscard]] const T* data() const noexcept { return m_ptr; }

private:
    void release() noexcept {
        if (m_ptr != nullptr) {
            cudaFreeHost(m_ptr);
            m_ptr      = nullptr;
            m_capacity = 0;
        }
    }
    T* m_ptr            = nullptr;
    SizeType m_capacity = 0;
};

void copy_out_strided(const float* h_out,
                      SizeType out_start,
                      SizeType out_count,
                      SizeType dm_count,
                      SizeType nsamps_reduced,
                      std::span<float> dmt) {
    for (SizeType idm = 0; idm < dm_count; ++idm) {
        std::copy_n(h_out + (idm * out_count), out_count,
                    &dmt[(idm * nsamps_reduced) + out_start]);
    }
}

void copy_out_strided(const int32_t* h_out,
                      SizeType out_start,
                      SizeType out_count,
                      SizeType dm_count,
                      SizeType nsamps_reduced,
                      std::span<int32_t> dmt) {
    for (SizeType idm = 0; idm < dm_count; ++idm) {
        std::copy_n(h_out + (idm * out_count), out_count,
                    &dmt[(idm * nsamps_reduced) + out_start]);
    }
}

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

class DDMTCudaEngine final : public detail::DDMTEngine {
public:
    DDMTCudaEngine(const plans::DDMTPlan& plan,
                   const detail::DDMTEngineConfig& cfg)
        : m_plan(plan),
          m_device_id(cfg.exec.device),
          m_nbeams(cfg.nbeams) {
        init();
    }

    ~DDMTCudaEngine() override { destroy_streams(); }
    DDMTCudaEngine(const DDMTCudaEngine&)            = delete;
    DDMTCudaEngine& operator=(const DDMTCudaEngine&) = delete;
    DDMTCudaEngine(DDMTCudaEngine&&)                 = delete;
    DDMTCudaEngine& operator=(DDMTCudaEngine&&)      = delete;

    // Device-memory entry points: forward to the cuda::std::span overloads
    // below.
    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        execute(to_cuda(d_waterfall), to_cuda(d_dmt), to_cuda(stream));
    }
    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nsamps,
                 DeviceSpan<int32_t> d_dmt,
                 Stream stream) override {
        execute(to_cuda(d_waterfall_packed), nsamps, to_cuda(d_dmt),
                to_cuda(stream));
    }
    void execute_time_major(DeviceSpan<const uint8_t> d_filterbank_packed,
                            SizeType nsamps,
                            DeviceSpan<int32_t> d_dmt,
                            Stream stream) override {
        execute_time_major(to_cuda(d_filterbank_packed), nsamps, to_cuda(d_dmt),
                           to_cuda(stream));
    }
    void save_history(DeviceSpan<float> d_out, Stream stream) const override {
        save_history(to_cuda(d_out), to_cuda(stream));
    }
    void save_history(DeviceSpan<uint8_t> d_out, Stream stream) const override {
        save_history(to_cuda(d_out), to_cuda(stream));
    }
    void load_history(DeviceSpan<const float> d_in, Stream stream) override {
        load_history(to_cuda(d_in), to_cuda(stream));
    }
    void load_history(DeviceSpan<const uint8_t> d_in, Stream stream) override {
        load_history(to_cuda(d_in), to_cuda(stream));
    }

    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept override {
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        const auto total = m_history_len + input_nsamps;
        return total > max_delay ? total - max_delay : 0;
    }

    void reset_history() noexcept override {
        m_history.clear();
        m_history_packed.clear();
        m_history_len = 0;
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        const auto& plan_c   = m_plan.get_container();
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        if (plan_c.nbits == 32) {
            return m_nbeams * plan_c.nchans * max_delay;
        }
        return m_nbeams * plan_c.nchans *
               bit_pack_utils::packed_row_bytes(max_delay, plan_c.nbits);
    }

    void save_history(std::span<float> out) const override {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMT::save_history(float): plan nbits={} != 32; "
                "use the packed save_history(uint8_t) overload instead",
                m_plan.get_nbits()));
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, max_delay));
        }
        if (out.size() != m_history.size()) {
            throw std::invalid_argument(
                std::format("DDMT::save_history: buffer size mismatch: "
                            "expected {}, got {}",
                            m_history.size(), out.size()));
        }
        std::ranges::copy(m_history, out.begin());
    }

    void save_history(cuda::std::span<float> d_out, cudaStream_t stream) const {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMT::save_history(device float): plan nbits={} != 32; "
                "use the packed save_history(uint8_t) overload instead",
                m_plan.get_nbits()));
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, max_delay));
        }
        if (d_out.size() != m_history.size()) {
            throw std::invalid_argument(
                std::format("DDMT::save_history: buffer size mismatch: "
                            "expected {}, got {}",
                            m_history.size(), d_out.size()));
        }
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(d_out.data(), m_history.data(),
                            m_history.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "DDMT::save_history H2D copy failed");
    }

    void save_history(std::span<uint8_t> out) const override {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            throw std::invalid_argument(
                "DDMT::save_history(uint8_t): plan nbits=32; "
                "use the float save_history(float) overload instead");
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, max_delay));
        }
        const auto expected_bytes = history_state_size();
        if (out.size() != expected_bytes) {
            throw std::invalid_argument(
                std::format("DDMT::save_history: buffer size mismatch: "
                            "expected {} bytes, got {}",
                            expected_bytes, out.size()));
        }
        std::ranges::copy(m_history_packed, out.begin());
    }

    void save_history(cuda::std::span<uint8_t> d_out,
                      cudaStream_t stream) const {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            throw std::invalid_argument(
                "DDMT::save_history(device uint8_t): plan nbits=32; "
                "use the float save_history(float) overload instead");
        }
        const auto max_delay =
            *std::ranges::max_element(m_plan.get_container().delay_table);
        if (m_history_len != max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, max_delay));
        }
        const auto expected_bytes = history_state_size();
        if (d_out.size() != expected_bytes) {
            throw std::invalid_argument(
                std::format("DDMT::save_history: buffer size mismatch: "
                            "expected {} bytes, got {}",
                            expected_bytes, d_out.size()));
        }
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(d_out.data(), m_history_packed.data(),
                            m_history_packed.size(), cudaMemcpyHostToDevice,
                            stream),
            "DDMT::save_history(packed) H2D copy failed");
    }

    void load_history(std::span<const float> in) override {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMT::load_history(float): plan nbits={} != 32; "
                "use the packed load_history(uint8_t) overload instead",
                m_plan.get_nbits()));
        }
        if (in.size() != history_state_size()) {
            throw std::invalid_argument(
                std::format("DDMT::load_history: buffer size mismatch: "
                            "expected {}, got {}",
                            history_state_size(), in.size()));
        }
        m_history.assign(in.begin(), in.end());
        m_history_len = in.size() / (m_nbeams * m_plan.get_nchans());
    }

    void load_history(cuda::std::span<const float> d_in, cudaStream_t stream) {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMT::load_history(device float): plan nbits={} != 32; "
                "use the packed load_history(uint8_t) overload instead",
                m_plan.get_nbits()));
        }
        if (d_in.size() != history_state_size()) {
            throw std::invalid_argument(
                std::format("DDMT::load_history: buffer size mismatch: "
                            "expected {}, got {}",
                            history_state_size(), d_in.size()));
        }
        m_history.resize(d_in.size());
        gpu_utils::check_gpu_call(cudaMemcpyAsync(m_history.data(), d_in.data(),
                                                  d_in.size() * sizeof(float),
                                                  cudaMemcpyDeviceToHost,
                                                  stream),
                                  "DDMT::load_history D2H copy failed");
        if (stream != nullptr) {
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        } else {
            gpu_utils::check_gpu_call(cudaDeviceSynchronize());
        }
        m_history_len = d_in.size() / (m_nbeams * m_plan.get_nchans());
    }

    void load_history(std::span<const uint8_t> in) override {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            throw std::invalid_argument(
                "DDMT::load_history(uint8_t): plan nbits=32; "
                "use the float load_history(float) overload instead");
        }
        const auto expected_bytes = history_state_size();
        if (in.size() != expected_bytes) {
            throw std::invalid_argument(
                std::format("DDMT::load_history: buffer size mismatch: "
                            "expected {} bytes, got {}",
                            expected_bytes, in.size()));
        }
        m_history_packed.assign(in.begin(), in.end());
        m_history_len =
            *std::ranges::max_element(m_plan.get_container().delay_table);
    }

    void load_history(cuda::std::span<const uint8_t> d_in,
                      cudaStream_t stream) {
        const auto nbits = m_plan.get_nbits();
        if (nbits == 32) {
            throw std::invalid_argument(
                "DDMT::load_history(device uint8_t): plan nbits=32; "
                "use the float load_history(float) overload instead");
        }
        const auto expected_bytes = history_state_size();
        if (d_in.size() != expected_bytes) {
            throw std::invalid_argument(
                std::format("DDMT::load_history: buffer size mismatch: "
                            "expected {} bytes, got {}",
                            expected_bytes, d_in.size()));
        }
        m_history_packed.resize(d_in.size());
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(m_history_packed.data(), d_in.data(), d_in.size(),
                            cudaMemcpyDeviceToHost, stream),
            "DDMT::load_history(packed) D2H copy failed");
        if (stream != nullptr) {
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        } else {
            gpu_utils::check_gpu_call(cudaDeviceSynchronize());
        }
        m_history_len =
            *std::ranges::max_element(m_plan.get_container().delay_table);
    }

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        if (plan_c.nbits != 32) {
            throw std::invalid_argument(
                std::format("DDMT::execute(float): plan nbits={} != 32; "
                            "use the packed-integer execute() overload instead",
                            plan_c.nbits));
        }
        const auto nchans     = plan_c.nchans;
        const auto nsamps_new = waterfall.size() / (nchans * m_nbeams);
        const auto max_delay  = *std::ranges::max_element(plan_c.delay_table);
        const auto total      = m_history_len + nsamps_new;
        const auto nsamps_reduced = total > max_delay ? total - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, dmt.size()));
        }

        // Beam-major (nbeams, nchans, total): prepend each beam's retained
        // history tail to that beam's new block. Once flattened this way,
        // "row = ibeam*nchans + ichan" addresses both this buffer and the
        // gulp pipeline's per-channel staging identically to nchans rows in
        // the single-beam case, so the gulp loop below only needs
        // nbeams*nchans / nbeams*dm_count substituted for nchans/dm_count,
        // not a separate beam loop.
        if (nsamps_reduced == 0) {
            std::vector<float> combined(m_nbeams * nchans * total);
            for (SizeType ibeam = 0; ibeam < m_nbeams; ++ibeam) {
                for (SizeType ichan = 0; ichan < nchans; ++ichan) {
                    const auto row = (ibeam * nchans) + ichan;
                    auto* dst      = &combined[row * total];
                    if (m_history_len > 0) {
                        std::copy_n(&m_history[row * m_history_len],
                                    m_history_len, dst);
                    }
                    std::copy_n(&waterfall[row * nsamps_new], nsamps_new,
                                dst + m_history_len);
                }
            }
            m_history     = std::move(combined);
            m_history_len = total;
            return;
        }

        std::vector<float> combined_storage;
        std::span<const float> input_span;
        if (m_history_len > 0) {
            combined_storage.resize(m_nbeams * nchans * total);
            for (SizeType ibeam = 0; ibeam < m_nbeams; ++ibeam) {
                for (SizeType ichan = 0; ichan < nchans; ++ichan) {
                    const auto row = (ibeam * nchans) + ichan;
                    auto* dst      = &combined_storage[row * total];
                    std::copy_n(&m_history[row * m_history_len], m_history_len,
                                dst);
                    std::copy_n(&waterfall[row * nsamps_new], nsamps_new,
                                dst + m_history_len);
                }
            }
            input_span = combined_storage;
        } else {
            input_span = waterfall;
        }
        const auto nsamps_total = total;
        const auto in_rows      = m_nbeams * nchans;
        const auto out_rows     = m_nbeams * dm_count;

        const auto gulp_out = std::min(kGulpOutSamples, nsamps_reduced);
        const auto max_in   = gulp_out + max_delay;
        const auto n_gulps  = (nsamps_reduced + gulp_out - 1) / gulp_out;
        for (int i = 0; i < 2; ++i) {
            m_h_in_f[i].reserve(in_rows * max_in);
            m_h_out_f[i].reserve(out_rows * gulp_out);
            m_d_in_f[i].resize(in_rows * max_in);
            m_d_out_f[i].resize(out_rows * gulp_out);
        }

        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        const auto in_beam_stride = nchans * max_in;

        SizeType prev_start = 0;
        SizeType prev_count = 0;
        bool have_prev      = false;
        for (SizeType g = 0; g < n_gulps; ++g) {
            const auto buf       = g % 2;
            const auto out_start = g * gulp_out;
            const auto out_count =
                std::min(gulp_out, nsamps_reduced - out_start);
            const auto in_count = out_count + max_delay;

            if (g >= 2) {
                gpu_utils::check_gpu_call(
                    cudaEventSynchronize(m_dtoh_done[buf]));
            }

            for (SizeType row = 0; row < in_rows; ++row) {
                std::copy_n(&input_span[(row * nsamps_total) + out_start],
                            in_count, m_h_in_f[buf].data() + (row * max_in));
            }

            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(
                    thrust::raw_pointer_cast(m_d_in_f[buf].data()),
                    max_in * sizeof(float), m_h_in_f[buf].data(),
                    max_in * sizeof(float), in_count * sizeof(float), in_rows,
                    cudaMemcpyHostToDevice, m_htod_stream),
                "H2D copy failed");
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_htod_done[buf], m_htod_stream));
            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_exec_stream, m_htod_done[buf], 0));

            const auto out_beam_stride = dm_count * out_count;
            launch_float(
                thrust::raw_pointer_cast(m_d_in_f[buf].data()),
                static_cast<int>(max_in), static_cast<int>(in_beam_stride),
                thrust::raw_pointer_cast(m_d_out_f[buf].data()),
                static_cast<int>(out_count), static_cast<int>(out_beam_stride),
                delay_ptr, kill_ptr, static_cast<int>(nchans),
                static_cast<int>(dm_count), static_cast<int>(out_count),
                static_cast<int>(m_nbeams), m_exec_stream);
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_exec_done[buf], m_exec_stream));

            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_dtoh_stream, m_exec_done[buf], 0));
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_h_out_f[buf].data(),
                                thrust::raw_pointer_cast(m_d_out_f[buf].data()),
                                out_rows * out_count * sizeof(float),
                                cudaMemcpyDeviceToHost, m_dtoh_stream),
                "D2H copy failed");
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_dtoh_done[buf], m_dtoh_stream));

            if (have_prev) {
                const auto prev_buf = 1 - buf;
                gpu_utils::check_gpu_call(
                    cudaEventSynchronize(m_dtoh_done[prev_buf]));
                copy_out_strided(m_h_out_f[prev_buf].data(), prev_start,
                                 prev_count, out_rows, nsamps_reduced, dmt);
            }
            prev_start = out_start;
            prev_count = out_count;
            have_prev  = true;
        }
        if (have_prev) {
            const auto last_buf = (n_gulps - 1) % 2;
            gpu_utils::check_gpu_call(
                cudaEventSynchronize(m_dtoh_done[last_buf]));
            copy_out_strided(m_h_out_f[last_buf].data(), prev_start, prev_count,
                             out_rows, nsamps_reduced, dmt);
        }

        const auto new_history_len = std::min(total, max_delay);
        std::vector<float> new_history(m_nbeams * nchans * new_history_len);
        for (SizeType row = 0; row < in_rows; ++row) {
            std::copy_n(&input_span[(row * total) + (total - new_history_len)],
                        new_history_len, &new_history[row * new_history_len]);
        }
        m_history     = std::move(new_history);
        m_history_len = new_history_len;
    }

    // Shares the same cross-call history as the host-span execute(float)
    // overload (get_output_nsamps()'s doc comment covers "the next
    // execute(float) call" generically, not just the host-span one), so it
    // must honor m_history the same way, not silently compute cold-start
    // sizes while history is retained. Unlike the host-span overload this
    // does no internal gulping (by design: the caller already owns the
    // device buffer and is expected to manage chunking itself), so the
    // "combine history with the new block" step below allocates one
    // temporary device buffer sized to the whole call rather than a bounded
    // per-gulp one.
    void execute(cuda::std::span<const float> d_waterfall,
                 cuda::std::span<float> d_dmt,
                 cudaStream_t stream) {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        if (plan_c.nbits != 32) {
            throw std::invalid_argument(
                std::format("DDMT::execute(device float): plan nbits={} "
                            "!= 32; use the packed-integer overload instead",
                            plan_c.nbits));
        }
        const auto nchans     = plan_c.nchans;
        const auto nsamps_new = d_waterfall.size() / (nchans * m_nbeams);
        const auto max_delay  = *std::ranges::max_element(plan_c.delay_table);
        const auto total      = m_history_len + nsamps_new;
        const auto nsamps_reduced = total > max_delay ? total - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (d_dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, d_dmt.size()));
        }
        const auto in_rows = m_nbeams * nchans;

        // Still warming up: no output yet, just fold this block into the
        // retained (host) history and return.
        if (nsamps_reduced == 0) {
            std::vector<float> new_block(in_rows * nsamps_new);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(new_block.data(), d_waterfall.data(),
                                in_rows * nsamps_new * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
                "DDMT::execute(device float) warm-up D2H copy failed");
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
            std::vector<float> combined(in_rows * total);
            for (SizeType row = 0; row < in_rows; ++row) {
                auto* dst = &combined[row * total];
                if (m_history_len > 0) {
                    std::copy_n(&m_history[row * m_history_len], m_history_len,
                                dst);
                }
                std::copy_n(&new_block[row * nsamps_new], nsamps_new,
                            dst + m_history_len);
            }
            m_history     = std::move(combined);
            m_history_len = total;
            return;
        }

        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        const auto in_beam_stride  = nchans * nsamps_new;
        const auto out_beam_stride = dm_count * nsamps_reduced;

        if (m_history_len == 0) {
            // No retained history: the caller's buffer is exactly what the
            // kernel needs, no combine step.
            launch_float(d_waterfall.data(), static_cast<int>(nsamps_new),
                         static_cast<int>(in_beam_stride), d_dmt.data(),
                         static_cast<int>(nsamps_reduced),
                         static_cast<int>(out_beam_stride), delay_ptr, kill_ptr,
                         static_cast<int>(nchans), static_cast<int>(dm_count),
                         static_cast<int>(nsamps_reduced),
                         static_cast<int>(m_nbeams), stream);
        } else {
            const auto combined_beam_stride = nchans * total;
            thrust::device_vector<float> combined_d(in_rows * total);
            auto* combined_ptr = thrust::raw_pointer_cast(combined_d.data());
            for (SizeType row = 0; row < in_rows; ++row) {
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(combined_ptr + (row * total),
                                    &m_history[row * m_history_len],
                                    m_history_len * sizeof(float),
                                    cudaMemcpyHostToDevice, stream),
                    "DDMT::execute(device float) history H2D copy failed");
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(combined_ptr + (row * total) +
                                        m_history_len,
                                    d_waterfall.data() + (row * nsamps_new),
                                    nsamps_new * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream),
                    "DDMT::execute(device float) D2D copy failed");
            }
            launch_float(combined_ptr, static_cast<int>(total),
                         static_cast<int>(combined_beam_stride), d_dmt.data(),
                         static_cast<int>(nsamps_reduced),
                         static_cast<int>(out_beam_stride), delay_ptr, kill_ptr,
                         static_cast<int>(nchans), static_cast<int>(dm_count),
                         static_cast<int>(nsamps_reduced),
                         static_cast<int>(m_nbeams), stream);

            const auto new_history_len = std::min(total, max_delay);
            std::vector<float> new_history(in_rows * new_history_len);
            for (SizeType row = 0; row < in_rows; ++row) {
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(&new_history[row * new_history_len],
                                    combined_ptr + (row * total) +
                                        (total - new_history_len),
                                    new_history_len * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream),
                    "DDMT::execute(device float) history retention D2H "
                    "copy failed");
            }
            // m_history is host memory read by get_output_nsamps()/
            // save_history() immediately after this call returns, so it
            // must be valid by then -- unlike the rest of this overload,
            // this one synchronization point can't be deferred to the
            // caller.
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
            m_history     = std::move(new_history);
            m_history_len = new_history_len;
            return;
        }

        // No history was retained on entry (the m_history_len == 0 branch
        // above), but the trailing max_delay samples of *this* block still
        // need to be captured, from the caller's own buffer, for the next
        // call.
        const auto new_history_len = std::min(total, max_delay);
        std::vector<float> new_history(in_rows * new_history_len);
        for (SizeType row = 0; row < in_rows; ++row) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(&new_history[row * new_history_len],
                                d_waterfall.data() + (row * nsamps_new) +
                                    (nsamps_new - new_history_len),
                                new_history_len * sizeof(float),
                                cudaMemcpyDeviceToHost, stream),
                "DDMT::execute(device float) history retention D2H copy "
                "failed");
        }
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        m_history     = std::move(new_history);
        m_history_len = new_history_len;
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps_total,
                 std::span<int32_t> dmt) override {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            throw std::invalid_argument(
                std::format("DDMT::execute(packed): plan nbits={} is not "
                            "a supported packed width (1,2,4,8,16)",
                            nbits));
        }
        const auto nchans  = plan_c.nchans;
        const auto in_rows = m_nbeams * nchans;
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(nsamps_total, nbits);
        if (waterfall_packed.size() != in_rows * row_bytes) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Packed input buffer size mismatch: expected {} "
                "bytes, got {}",
                in_rows * row_bytes, waterfall_packed.size()));
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto total     = m_history_len + nsamps_total;
        const auto nsamps_reduced = total > max_delay ? total - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, dmt.size()));
        }

        const auto combined_row_bytes =
            bit_pack_utils::packed_row_bytes(total, nbits);
        const auto hist_row_bytes =
            bit_pack_utils::packed_row_bytes(m_history_len, nbits);

        // Cold start or warm-up with insufficient samples to produce output:
        // accumulate into m_history_packed and return.
        if (nsamps_reduced == 0) {
            std::vector<uint8_t> combined(in_rows * combined_row_bytes, 0);
            auto combine_rows = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* hist_row =
                        m_history_packed.data() + (row * hist_row_bytes);
                    const auto* new_row =
                        waterfall_packed.data() + (row * row_bytes);
                    auto* comb_row =
                        combined.data() + (row * combined_row_bytes);
                    if (m_history_len > 0) {
                        bit_pack_utils::copy_packed_samples<NBITS>(
                            hist_row, 0, comb_row, 0, m_history_len);
                    }
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        new_row, 0, comb_row, m_history_len, nsamps_total);
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
            m_history_packed = std::move(combined);
            m_history_len    = total;
            return;
        }

        std::vector<uint8_t> combined_storage;
        const uint8_t* input_ptr = nullptr;
        SizeType input_row_bytes = 0;

        if (m_history_len > 0) {
            combined_storage.assign(in_rows * combined_row_bytes, 0);
            auto combine_rows = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* hist_row =
                        m_history_packed.data() + (row * hist_row_bytes);
                    const auto* new_row =
                        waterfall_packed.data() + (row * row_bytes);
                    auto* comb_row =
                        combined_storage.data() + (row * combined_row_bytes);
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        hist_row, 0, comb_row, 0, m_history_len);
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        new_row, 0, comb_row, m_history_len, nsamps_total);
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
            input_ptr       = combined_storage.data();
            input_row_bytes = combined_row_bytes;
        } else {
            input_ptr       = waterfall_packed.data();
            input_row_bytes = row_bytes;
        }

        const auto out_rows = m_nbeams * dm_count;

        const auto gulp_out     = std::min(kGulpOutSamples, nsamps_reduced);
        const auto max_in_samps = gulp_out + max_delay;
        const auto max_in_bytes =
            bit_pack_utils::packed_row_bytes(max_in_samps, nbits) + 1;
        const auto n_gulps = (nsamps_reduced + gulp_out - 1) / gulp_out;
        for (int i = 0; i < 2; ++i) {
            m_h_in_u8[i].reserve(in_rows * max_in_bytes);
            m_h_out_i32[i].reserve(out_rows * gulp_out);
            m_d_in_u8[i].resize(in_rows * max_in_bytes);
            m_d_out_i32[i].resize(out_rows * gulp_out);
        }
        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        const auto in_beam_stride = nchans * max_in_bytes;

        SizeType prev_start = 0;
        SizeType prev_count = 0;
        bool have_prev      = false;
        for (SizeType g = 0; g < n_gulps; ++g) {
            const auto buf       = g % 2;
            const auto out_start = g * gulp_out;
            const auto out_count =
                std::min(gulp_out, nsamps_reduced - out_start);
            const auto in_samps = out_count + max_delay;
            const auto samples_per_byte =
                nbits < 8 ? 8 / static_cast<SizeType>(nbits) : 1;
            const auto aligned_start =
                (out_start / samples_per_byte) * samples_per_byte;
            const auto align_pad = out_start - aligned_start;
            const auto in_bytes =
                bit_pack_utils::packed_row_bytes(in_samps + align_pad, nbits);

            if (g >= 2) {
                gpu_utils::check_gpu_call(
                    cudaEventSynchronize(m_dtoh_done[buf]));
            }

            for (SizeType row = 0; row < in_rows; ++row) {
                const auto* src =
                    input_ptr + (row * input_row_bytes) +
                    bit_pack_utils::packed_row_bytes(aligned_start, nbits);
                std::copy_n(src, in_bytes,
                            m_h_in_u8[buf].data() + (row * max_in_bytes));
            }

            gpu_utils::check_gpu_call(
                cudaMemcpy2DAsync(
                    thrust::raw_pointer_cast(m_d_in_u8[buf].data()),
                    max_in_bytes, m_h_in_u8[buf].data(), max_in_bytes, in_bytes,
                    in_rows, cudaMemcpyHostToDevice, m_htod_stream),
                "H2D copy failed");
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_htod_done[buf], m_htod_stream));
            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_exec_stream, m_htod_done[buf], 0));

            const auto out_beam_stride = dm_count * out_count;
            launch_packed(
                nbits, thrust::raw_pointer_cast(m_d_in_u8[buf].data()),
                max_in_bytes, static_cast<SizeType>(in_beam_stride), align_pad,
                thrust::raw_pointer_cast(m_d_out_i32[buf].data()),
                static_cast<int>(out_count), static_cast<int>(out_beam_stride),
                delay_ptr, kill_ptr, static_cast<int>(nchans),
                static_cast<int>(dm_count), static_cast<int>(out_count),
                static_cast<int>(m_nbeams), m_exec_stream);
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_exec_done[buf], m_exec_stream));

            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_dtoh_stream, m_exec_done[buf], 0));
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(
                    m_h_out_i32[buf].data(),
                    thrust::raw_pointer_cast(m_d_out_i32[buf].data()),
                    out_rows * out_count * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, m_dtoh_stream),
                "D2H copy failed");
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_dtoh_done[buf], m_dtoh_stream));

            if (have_prev) {
                const auto prev_buf = 1 - buf;
                gpu_utils::check_gpu_call(
                    cudaEventSynchronize(m_dtoh_done[prev_buf]));
                copy_out_strided(m_h_out_i32[prev_buf].data(), prev_start,
                                 prev_count, out_rows, nsamps_reduced, dmt);
            }
            prev_start = out_start;
            prev_count = out_count;
            have_prev  = true;
        }
        if (have_prev) {
            const auto last_buf = (n_gulps - 1) % 2;
            gpu_utils::check_gpu_call(
                cudaEventSynchronize(m_dtoh_done[last_buf]));
            copy_out_strided(m_h_out_i32[last_buf].data(), prev_start,
                             prev_count, out_rows, nsamps_reduced, dmt);
        }

        const auto new_history_len = std::min(total, max_delay);
        const auto new_hist_row_bytes =
            bit_pack_utils::packed_row_bytes(new_history_len, nbits);
        std::vector<uint8_t> new_history_packed(in_rows * new_hist_row_bytes,
                                                0);

        auto extract_tail = [&]<unsigned NBITS>() {
            for (SizeType row = 0; row < in_rows; ++row) {
                const auto* src_row = input_ptr + (row * input_row_bytes);
                auto* dst_row =
                    new_history_packed.data() + (row * new_hist_row_bytes);
                bit_pack_utils::copy_packed_samples<NBITS>(
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

        m_history_packed = std::move(new_history_packed);
        m_history_len    = new_history_len;
    }

    void execute(cuda::std::span<const uint8_t> d_waterfall_packed,
                 SizeType nsamps_total,
                 cuda::std::span<int32_t> d_dmt,
                 cudaStream_t stream) {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            throw std::invalid_argument(
                std::format("DDMT::execute(device packed): plan nbits={} "
                            "is not a supported packed width (1,2,4,8,16)",
                            nbits));
        }
        const auto nchans  = plan_c.nchans;
        const auto in_rows = m_nbeams * nchans;
        const auto row_bytes =
            bit_pack_utils::packed_row_bytes(nsamps_total, nbits);
        if (d_waterfall_packed.size() != in_rows * row_bytes) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Packed input buffer size mismatch: expected {} "
                "bytes, got {}",
                in_rows * row_bytes, d_waterfall_packed.size()));
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto total     = m_history_len + nsamps_total;
        const auto nsamps_reduced = total > max_delay ? total - max_delay : 0;
        const auto dm_count       = plan_c.dm_arr.size();
        if (d_dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, d_dmt.size()));
        }

        const auto combined_row_bytes =
            bit_pack_utils::packed_row_bytes(total, nbits);
        const auto hist_row_bytes =
            bit_pack_utils::packed_row_bytes(m_history_len, nbits);

        // Cold start or warm-up with insufficient samples to produce output
        if (nsamps_reduced == 0) {
            std::vector<uint8_t> new_block(in_rows * row_bytes);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(new_block.data(), d_waterfall_packed.data(),
                                in_rows * row_bytes, cudaMemcpyDeviceToHost,
                                stream),
                "DDMT::execute(device packed) warm-up D2H copy failed");
            if (stream != nullptr) {
                gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
            } else {
                gpu_utils::check_gpu_call(cudaDeviceSynchronize());
            }
            std::vector<uint8_t> combined(in_rows * combined_row_bytes, 0);
            auto combine_rows = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* hist_row =
                        m_history_packed.data() + (row * hist_row_bytes);
                    const auto* new_row = new_block.data() + (row * row_bytes);
                    auto* comb_row =
                        combined.data() + (row * combined_row_bytes);
                    if (m_history_len > 0) {
                        bit_pack_utils::copy_packed_samples<NBITS>(
                            hist_row, 0, comb_row, 0, m_history_len);
                    }
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        new_row, 0, comb_row, m_history_len, nsamps_total);
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
            m_history_packed = std::move(combined);
            m_history_len    = total;
            return;
        }

        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        const auto in_beam_stride  = nchans * row_bytes;
        const auto out_beam_stride = dm_count * nsamps_reduced;

        if (m_history_len == 0) {
            launch_packed(nbits, d_waterfall_packed.data(), row_bytes,
                          static_cast<SizeType>(in_beam_stride), 0,
                          d_dmt.data(), static_cast<int>(nsamps_reduced),
                          static_cast<int>(out_beam_stride), delay_ptr,
                          kill_ptr, static_cast<int>(nchans),
                          static_cast<int>(dm_count),
                          static_cast<int>(nsamps_reduced),
                          static_cast<int>(m_nbeams), stream);

            const auto new_history_len = std::min(total, max_delay);
            const auto new_hist_row_bytes =
                bit_pack_utils::packed_row_bytes(new_history_len, nbits);
            std::vector<uint8_t> new_history_packed(
                in_rows * new_hist_row_bytes, 0);

            std::vector<uint8_t> host_input(in_rows * row_bytes);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(host_input.data(), d_waterfall_packed.data(),
                                in_rows * row_bytes, cudaMemcpyDeviceToHost,
                                stream),
                "DDMT::execute(device packed) history retention D2H copy "
                "failed");
            if (stream != nullptr) {
                gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
            } else {
                gpu_utils::check_gpu_call(cudaDeviceSynchronize());
            }

            auto extract_tail = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* src_row = host_input.data() + (row * row_bytes);
                    auto* dst_row =
                        new_history_packed.data() + (row * new_hist_row_bytes);
                    bit_pack_utils::copy_packed_samples<NBITS>(
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

            m_history_packed = std::move(new_history_packed);
            m_history_len    = new_history_len;
        } else {
            std::vector<uint8_t> new_block(in_rows * row_bytes);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(new_block.data(), d_waterfall_packed.data(),
                                in_rows * row_bytes, cudaMemcpyDeviceToHost,
                                stream),
                "DDMT::execute(device packed) input D2H copy failed");
            if (stream != nullptr) {
                gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
            } else {
                gpu_utils::check_gpu_call(cudaDeviceSynchronize());
            }

            std::vector<uint8_t> combined(in_rows * combined_row_bytes, 0);
            auto combine_rows = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* hist_row =
                        m_history_packed.data() + (row * hist_row_bytes);
                    const auto* new_row = new_block.data() + (row * row_bytes);
                    auto* comb_row =
                        combined.data() + (row * combined_row_bytes);
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        hist_row, 0, comb_row, 0, m_history_len);
                    bit_pack_utils::copy_packed_samples<NBITS>(
                        new_row, 0, comb_row, m_history_len, nsamps_total);
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

            thrust::device_vector<uint8_t> combined_d(combined.size());
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(thrust::raw_pointer_cast(combined_d.data()),
                                combined.data(), combined.size(),
                                cudaMemcpyHostToDevice, stream),
                "DDMT::execute(device packed) combined H2D copy failed");

            const auto comb_in_beam_stride = nchans * combined_row_bytes;
            launch_packed(
                nbits, thrust::raw_pointer_cast(combined_d.data()),
                combined_row_bytes, static_cast<SizeType>(comb_in_beam_stride),
                0, d_dmt.data(), static_cast<int>(nsamps_reduced),
                static_cast<int>(out_beam_stride), delay_ptr, kill_ptr,
                static_cast<int>(nchans), static_cast<int>(dm_count),
                static_cast<int>(nsamps_reduced), static_cast<int>(m_nbeams),
                stream);

            const auto new_history_len = std::min(total, max_delay);
            const auto new_hist_row_bytes =
                bit_pack_utils::packed_row_bytes(new_history_len, nbits);
            std::vector<uint8_t> new_history_packed(
                in_rows * new_hist_row_bytes, 0);

            auto extract_tail = [&]<unsigned NBITS>() {
                for (SizeType row = 0; row < in_rows; ++row) {
                    const auto* src_row =
                        combined.data() + (row * combined_row_bytes);
                    auto* dst_row =
                        new_history_packed.data() + (row * new_hist_row_bytes);
                    bit_pack_utils::copy_packed_samples<NBITS>(
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

            m_history_packed = std::move(new_history_packed);
            m_history_len    = new_history_len;
        }
    }

    void execute_time_major(cuda::std::span<const uint8_t> d_filterbank_packed,
                            SizeType nsamps_total,
                            cuda::std::span<int32_t> d_dmt,
                            cudaStream_t stream) {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            throw std::invalid_argument(std::format(
                "DDMT::execute_time_major(device): plan nbits={} is not "
                "a supported packed width (1,2,4,8,16)",
                nbits));
        }
        const auto nchans     = plan_c.nchans;
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        if (d_filterbank_packed.size() !=
            m_nbeams * nsamps_total * samp_bytes) {
            throw std::invalid_argument(
                std::format("DDMT (cuda): Time-major packed input buffer size "
                            "mismatch: expected {} "
                            "bytes, got {}",
                            m_nbeams * nsamps_total * samp_bytes,
                            d_filterbank_packed.size()));
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto nsamps_reduced =
            nsamps_total > max_delay ? nsamps_total - max_delay : 0;
        const auto dm_count = plan_c.dm_arr.size();
        if (d_dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, d_dmt.size()));
        }
        if (nsamps_reduced == 0)
            return;
        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        launch_time_major(nbits, d_filterbank_packed.data(), samp_bytes,
                          static_cast<SizeType>(nsamps_total * samp_bytes),
                          d_dmt.data(), static_cast<int>(nsamps_reduced),
                          static_cast<int>(dm_count * nsamps_reduced),
                          delay_ptr, kill_ptr, static_cast<int>(nchans),
                          static_cast<int>(dm_count),
                          static_cast<int>(nsamps_reduced),
                          static_cast<int>(m_nbeams), stream);
    }

    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps_total,
                            std::span<int32_t> dmt) override {
        gpu_utils::set_device(m_device_id);
        const auto& plan_c = m_plan.get_container();
        const auto nbits   = plan_c.nbits;
        if (nbits == 32 || (nbits != 1 && nbits != 2 && nbits != 4 &&
                            nbits != 8 && nbits != 16)) {
            throw std::invalid_argument(std::format(
                "DDMT::execute_time_major(host): plan nbits={} is not "
                "a supported packed width (1,2,4,8,16)",
                nbits));
        }
        const auto nchans     = plan_c.nchans;
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        if (filterbank_packed.size() != m_nbeams * nsamps_total * samp_bytes) {
            throw std::invalid_argument(
                std::format("DDMT (cuda): Time-major packed input buffer size "
                            "mismatch: expected {} "
                            "bytes, got {}",
                            m_nbeams * nsamps_total * samp_bytes,
                            filterbank_packed.size()));
        }
        const auto max_delay = *std::ranges::max_element(plan_c.delay_table);
        const auto nsamps_reduced =
            nsamps_total > max_delay ? nsamps_total - max_delay : 0;
        const auto dm_count = plan_c.dm_arr.size();
        if (dmt.size() != m_nbeams * dm_count * nsamps_reduced) {
            throw std::invalid_argument(std::format(
                "DDMT (cuda): Output buffer size mismatch: expected {}, got {}",
                m_nbeams * dm_count * nsamps_reduced, dmt.size()));
        }
        if (nsamps_reduced == 0)
            return;

        thrust::device_vector<uint8_t> d_in(filterbank_packed.size());
        thrust::device_vector<int32_t> d_out(dmt.size());
        gpu_utils::check_gpu_call(
            cudaMemcpy(thrust::raw_pointer_cast(d_in.data()),
                       filterbank_packed.data(), filterbank_packed.size(),
                       cudaMemcpyHostToDevice),
            "H2D copy failed in execute_time_major");

        const auto* delay_ptr =
            thrust::raw_pointer_cast(m_plan_d.delay_arr_d.data());
        const auto* kill_ptr =
            thrust::raw_pointer_cast(m_plan_d.kill_mask_d.data());
        launch_time_major(
            nbits, thrust::raw_pointer_cast(d_in.data()), samp_bytes,
            static_cast<SizeType>(nsamps_total * samp_bytes),
            thrust::raw_pointer_cast(d_out.data()),
            static_cast<int>(nsamps_reduced),
            static_cast<int>(dm_count * nsamps_reduced), delay_ptr, kill_ptr,
            static_cast<int>(nchans), static_cast<int>(dm_count),
            static_cast<int>(nsamps_reduced), static_cast<int>(m_nbeams),
            nullptr);
        gpu_utils::check_gpu_call(cudaDeviceSynchronize());

        gpu_utils::check_gpu_call(
            cudaMemcpy(dmt.data(), thrust::raw_pointer_cast(d_out.data()),
                       dmt.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "D2H copy failed in execute_time_major");
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    const plans::DDMTPlan& m_plan; // owned by the DDMT facade
    int m_device_id;
    SizeType m_nbeams;
    plans::DDMTPlanD m_plan_d;

    cudaStream_t m_htod_stream = nullptr;
    cudaStream_t m_exec_stream = nullptr;
    cudaStream_t m_dtoh_stream = nullptr;
    cudaEvent_t m_htod_done[2] = {nullptr, nullptr};
    cudaEvent_t m_exec_done[2] = {nullptr, nullptr};
    cudaEvent_t m_dtoh_done[2] = {nullptr, nullptr};

    PinnedBuffer<float> m_h_in_f[2];
    PinnedBuffer<float> m_h_out_f[2];
    thrust::device_vector<float> m_d_in_f[2];
    thrust::device_vector<float> m_d_out_f[2];
    PinnedBuffer<uint8_t> m_h_in_u8[2];
    PinnedBuffer<int32_t> m_h_out_i32[2];
    thrust::device_vector<uint8_t> m_d_in_u8[2];
    thrust::device_vector<int32_t> m_d_out_i32[2];

    std::vector<float> m_history;
    std::vector<uint8_t> m_history_packed;
    SizeType m_history_len = 0;

    void init() {
        gpu_utils::set_device(m_device_id);
        plans::transfer_ddmt_plan_to_device(m_plan.get_container(), m_plan_d);

        gpu_utils::check_gpu_call(cudaStreamCreate(&m_htod_stream),
                                  "Failed to create H2D stream");
        gpu_utils::check_gpu_call(cudaStreamCreate(&m_exec_stream),
                                  "Failed to create execute stream");
        gpu_utils::check_gpu_call(cudaStreamCreate(&m_dtoh_stream),
                                  "Failed to create D2H stream");
        for (int i = 0; i < 2; ++i) {
            gpu_utils::check_gpu_call(cudaEventCreateWithFlags(
                &m_htod_done[i], cudaEventDisableTiming));
            gpu_utils::check_gpu_call(cudaEventCreateWithFlags(
                &m_exec_done[i], cudaEventDisableTiming));
            gpu_utils::check_gpu_call(cudaEventCreateWithFlags(
                &m_dtoh_done[i], cudaEventDisableTiming));
        }
    }

    void destroy_streams() noexcept {
        for (int i = 0; i < 2; ++i) {
            if (m_htod_done[i] != nullptr) {
                cudaEventDestroy(m_htod_done[i]);
            }
            if (m_exec_done[i] != nullptr) {
                cudaEventDestroy(m_exec_done[i]);
            }
            if (m_dtoh_done[i] != nullptr) {
                cudaEventDestroy(m_dtoh_done[i]);
            }
        }
        if (m_htod_stream != nullptr) {
            cudaStreamDestroy(m_htod_stream);
        }
        if (m_exec_stream != nullptr) {
            cudaStreamDestroy(m_exec_stream);
        }
        if (m_dtoh_stream != nullptr) {
            cudaStreamDestroy(m_dtoh_stream);
        }
    }

    static dim3 grid_for(SizeType nsamps_reduced,
                         SizeType dm_count,
                         int samps_per_thread,
                         SizeType nbeams) {
        const dim3 block(256, 1);
        const auto total_threads =
            (nsamps_reduced + samps_per_thread - 1) / samps_per_thread;
        const dim3 grid(
            static_cast<unsigned>((total_threads + block.x - 1) / block.x),
            static_cast<unsigned>(std::min<SizeType>(dm_count, 65535)),
            static_cast<unsigned>(nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        return grid;
    }

    static void launch_float(const float* d_in,
                             int in_chan_stride,
                             int in_beam_stride,
                             float* d_out,
                             int out_dm_stride,
                             int out_beam_stride,
                             const int* delay_ptr,
                             const int* kill_ptr,
                             int nchans,
                             int dm_count,
                             int nsamps_reduced,
                             int nbeams,
                             cudaStream_t stream) {
        const dim3 block(256, 1);
        const auto grid = grid_for(static_cast<SizeType>(nsamps_reduced),
                                   static_cast<SizeType>(dm_count), 2,
                                   static_cast<SizeType>(nbeams));
        ddmt_kernel_float<2><<<grid, block, 0, stream>>>(
            d_in, in_chan_stride, in_beam_stride, d_out, out_dm_stride,
            out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
            nsamps_reduced);
        gpu_utils::check_last_gpu_error("ddmt_kernel_float launch failed");
    }

    static void launch_packed(SizeType nbits,
                              const uint8_t* d_in,
                              SizeType row_bytes,
                              SizeType in_beam_stride,
                              SizeType sample_offset,
                              int32_t* d_out,
                              int out_dm_stride,
                              int out_beam_stride,
                              const int* delay_ptr,
                              const int* kill_ptr,
                              int nchans,
                              int dm_count,
                              int nsamps_reduced,
                              int nbeams,
                              cudaStream_t stream) {
        const dim3 block(256, 1);
        const auto grid = grid_for(static_cast<SizeType>(nsamps_reduced),
                                   static_cast<SizeType>(dm_count), 2,
                                   static_cast<SizeType>(nbeams));
        switch (nbits) {
        case 1:
            ddmt_kernel_packed<1, 2><<<grid, block, 0, stream>>>(
                d_in, row_bytes, in_beam_stride, sample_offset, d_out,
                out_dm_stride, out_beam_stride, delay_ptr, kill_ptr, nchans,
                dm_count, nsamps_reduced);
            break;
        case 2:
            ddmt_kernel_packed<2, 2><<<grid, block, 0, stream>>>(
                d_in, row_bytes, in_beam_stride, sample_offset, d_out,
                out_dm_stride, out_beam_stride, delay_ptr, kill_ptr, nchans,
                dm_count, nsamps_reduced);
            break;
        case 4:
            ddmt_kernel_packed<4, 2><<<grid, block, 0, stream>>>(
                d_in, row_bytes, in_beam_stride, sample_offset, d_out,
                out_dm_stride, out_beam_stride, delay_ptr, kill_ptr, nchans,
                dm_count, nsamps_reduced);
            break;
        case 8:
            ddmt_kernel_packed<8, 2><<<grid, block, 0, stream>>>(
                d_in, row_bytes, in_beam_stride, sample_offset, d_out,
                out_dm_stride, out_beam_stride, delay_ptr, kill_ptr, nchans,
                dm_count, nsamps_reduced);
            break;
        case 16:
            ddmt_kernel_packed<16, 2><<<grid, block, 0, stream>>>(
                d_in, row_bytes, in_beam_stride, sample_offset, d_out,
                out_dm_stride, out_beam_stride, delay_ptr, kill_ptr, nchans,
                dm_count, nsamps_reduced);
            break;
        default:
            break;
        }
        gpu_utils::check_last_gpu_error("ddmt_kernel_packed launch failed");
    }

    static void launch_time_major(SizeType nbits,
                                  const uint8_t* d_in,
                                  SizeType samp_bytes,
                                  SizeType in_beam_stride,
                                  int32_t* d_out,
                                  int out_dm_stride,
                                  int out_beam_stride,
                                  const int* delay_ptr,
                                  const int* kill_ptr,
                                  int nchans,
                                  int dm_count,
                                  int nsamps_reduced,
                                  int nbeams,
                                  cudaStream_t stream) {
        const dim3 block(256, 1);
        const auto grid = grid_for(static_cast<SizeType>(nsamps_reduced),
                                   static_cast<SizeType>(dm_count), 2,
                                   static_cast<SizeType>(nbeams));
        switch (nbits) {
        case 1:
            ddmt_kernel_time_major<1, 2><<<grid, block, 0, stream>>>(
                d_in, samp_bytes, in_beam_stride, d_out, out_dm_stride,
                out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
                nsamps_reduced);
            break;
        case 2:
            ddmt_kernel_time_major<2, 2><<<grid, block, 0, stream>>>(
                d_in, samp_bytes, in_beam_stride, d_out, out_dm_stride,
                out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
                nsamps_reduced);
            break;
        case 4:
            ddmt_kernel_time_major<4, 2><<<grid, block, 0, stream>>>(
                d_in, samp_bytes, in_beam_stride, d_out, out_dm_stride,
                out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
                nsamps_reduced);
            break;
        case 8:
            ddmt_kernel_time_major<8, 2><<<grid, block, 0, stream>>>(
                d_in, samp_bytes, in_beam_stride, d_out, out_dm_stride,
                out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
                nsamps_reduced);
            break;
        case 16:
            ddmt_kernel_time_major<16, 2><<<grid, block, 0, stream>>>(
                d_in, samp_bytes, in_beam_stride, d_out, out_dm_stride,
                out_beam_stride, delay_ptr, kill_ptr, nchans, dm_count,
                nsamps_reduced);
            break;
        default:
            break;
        }
        gpu_utils::check_last_gpu_error("ddmt_kernel_time_major launch failed");
    }
};

} // namespace

std::unique_ptr<detail::DDMTEngine>
detail::make_ddmt_gpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<DDMTCudaEngine>(plan, cfg);
}

} // namespace dmt::algorithms
