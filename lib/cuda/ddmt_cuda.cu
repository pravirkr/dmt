#include "dmt/algorithms/ddmt.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_kernel.cuh"
#include "dmt/engines.hpp"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"
#include "dmt/logging.hpp"
#include "dmt/sdmt_kernel.cuh"

namespace dmt::algorithms {

namespace {

using ddmt_gpu::DDMTAcc;
using ddmt_gpu::DDMTSegments;
using ddmt_gpu::TileCfg;
using ddmt_gpu::TileShape;
using gpu_host::parallel_rows;
using gpu_host::PinnedBuffer;

// Default input samples per host-path chunk; bounds pinned/device staging
// for very large inputs and lets H2D copy / kernel / D2H copy of
// consecutive chunks overlap. Same order of magnitude as dedisp's gulp.
constexpr SizeType kDefaultGulpSamples = 65536;
// Smallest automatic chunk (see run_host).
constexpr SizeType kMinAutoChunk = 16384;

// Tiled-kernel configurations (see TileCfg). "Wide" tiles 64 DM trials per
// block, "narrow" 16; both cover 256 output samples. Chosen once per engine
// from the plan's delay spread and the device's shared memory (select_cfg).
using FloatWide    = TileCfg<32, 4, 4, 16, 8>;
using FloatNarrow  = TileCfg<32, 4, 4, 4, 8>;
using ByteWide     = TileCfg<32, 8, 2, 8, 16>;
using ByteNarrow   = TileCfg<32, 4, 2, 4, 16>;
using Word16Wide   = TileCfg<32, 4, 2, 16, 8>;
using Word16Narrow = TileCfg<32, 4, 2, 4, 8>;

template <unsigned NBITS, bool kWide> struct CfgFor;
template <> struct CfgFor<32, true> {
    using type = FloatWide;
};
template <> struct CfgFor<32, false> {
    using type = FloatNarrow;
};
template <> struct CfgFor<16, true> {
    using type = Word16Wide;
};
template <> struct CfgFor<16, false> {
    using type = Word16Narrow;
};
template <unsigned NBITS> struct CfgFor<NBITS, true> {
    using type = ByteWide;
};
template <unsigned NBITS> struct CfgFor<NBITS, false> {
    using type = ByteNarrow;
};

enum class KernelKind : uint8_t { kWide, kNarrow, kDirect, kShared };

// SDMT kernel shapes (sdmt_gpu::sdmt_kernel<NBITS, TY, DPT, SPW>): TY warps
// of DPT trials each, 32 * SPW output samples per block.
template <int TY_, int DPT_, int SPW_> struct SdmtCfg {
    static constexpr int kTY  = TY_;
    static constexpr int kDPT = DPT_;
    static constexpr int kSPW = SPW_;
    static constexpr int kTDM = TY_ * DPT_;
    static constexpr int kBT  = 32 * SPW_;
};
using SdmtCfgs = std::tuple<SdmtCfg<8, 16, 4>, SdmtCfg<8, 8, 4>>;
inline constexpr int kNumSdmtCfgs = std::tuple_size_v<SdmtCfgs>;

/// Calls f.template operator()<Cfg>() for SDMT configuration @p i.
template <int I = 0, typename F> void with_sdmt_cfg(int i, F&& f) {
    if constexpr (I < kNumSdmtCfgs) {
        if (i == I) {
            f.template operator()<std::tuple_element_t<I, SdmtCfgs>>();
            return;
        }
        with_sdmt_cfg<I + 1>(i, std::forward<F>(f));
    }
}

/// Grow-only raw device byte buffer.
class DeviceBytes {
public:
    void reserve(SizeType n) {
        if (n > m_buf.size()) {
            m_buf.resize(n);
        }
    }
    [[nodiscard]] uint8_t* data() noexcept {
        return thrust::raw_pointer_cast(m_buf.data());
    }
    [[nodiscard]] const uint8_t* data() const noexcept {
        return thrust::raw_pointer_cast(m_buf.data());
    }

private:
    thrust::device_vector<uint8_t> m_buf;
};

template <typename T> cuda::std::span<T> to_cuda(DeviceSpan<T> span) {
    return {span.data(), span.size()};
}

cudaStream_t to_cuda(Stream stream) {
    return static_cast<cudaStream_t>(stream.native);
}

bool is_packed_width(SizeType nbits) {
    return nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8 || nbits == 16;
}

/// Calls f.template operator()<NBITS>() for the runtime width (1..16, 32).
template <typename F> void dispatch_nbits(SizeType nbits, F&& f) {
    switch (nbits) {
    case 1:
        f.template operator()<1>();
        break;
    case 2:
        f.template operator()<2>();
        break;
    case 4:
        f.template operator()<4>();
        break;
    case 8:
        f.template operator()<8>();
        break;
    case 16:
        f.template operator()<16>();
        break;
    case 32:
        f.template operator()<32>();
        break;
    default:
        throw std::invalid_argument(
            std::format("DDMT (gpu): unsupported nbits={}", nbits));
    }
}

/// Bytes of one channel row holding @p nsamps samples (float when 32).
SizeType row_bytes_for(SizeType nsamps, SizeType nbits) {
    return nbits == 32 ? nsamps * sizeof(float)
                       : bit_pack_utils::packed_row_bytes(nsamps, nbits);
}

class DDMTCudaEngine final : public detail::DDMTEngine {
public:
    DDMTCudaEngine(const plans::DDMTPlan& plan,
                   const detail::DDMTEngineConfig& cfg,
                   bool shared_sums)
        : m_plan(plan),
          m_device_id(cfg.exec.device),
          m_nbeams(cfg.nbeams),
          m_shared_sums(shared_sums) {
        init();
    }

    ~DDMTCudaEngine() override { destroy_streams(); }
    DDMTCudaEngine(const DDMTCudaEngine&)            = delete;
    DDMTCudaEngine& operator=(const DDMTCudaEngine&) = delete;
    DDMTCudaEngine(DDMTCudaEngine&&)                 = delete;
    DDMTCudaEngine& operator=(DDMTCudaEngine&&)      = delete;

    // ---- Streaming state -------------------------------------------------

    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept override {
        const auto total = m_history_len + input_nsamps;
        return total > m_max_delay ? total - m_max_delay : 0;
    }

    void reset_history() noexcept override { m_history_len = 0; }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        const auto rows = m_nbeams * m_plan.get_nchans();
        if (m_plan.get_nbits() == 32) {
            return rows * m_max_delay;
        }
        return rows * m_hist_row_bytes;
    }

    void set_gulp_size(SizeType gulp_size) override {
        m_gulp_is_default = gulp_size == 0;
        m_gulp            = m_gulp_is_default ? kDefaultGulpSamples : gulp_size;
    }
    [[nodiscard]] SizeType get_gulp_size() const noexcept override {
        return m_gulp;
    }

    // ---- History snapshots -----------------------------------------------

    void save_history(std::span<float> out) const override {
        check_float("save_history(float)");
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaEventSynchronize(m_hist_ready));
        gpu_utils::check_gpu_call(cudaMemcpy(out.data(), hist_cur(),
                                             out.size() * sizeof(float),
                                             cudaMemcpyDeviceToHost),
                                  "DDMT::save_history D2H copy failed");
    }
    void save_history(std::span<uint8_t> out) const override {
        check_packed("save_history(uint8_t)");
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaEventSynchronize(m_hist_ready));
        gpu_utils::check_gpu_call(cudaMemcpy(out.data(), hist_cur(), out.size(),
                                             cudaMemcpyDeviceToHost),
                                  "DDMT::save_history D2H copy failed");
    }
    void load_history(std::span<const float> in) override {
        check_float("load_history(float)");
        check_size(in.size(), history_state_size(), "load_history");
        load_history_bytes(reinterpret_cast<const uint8_t*>(in.data()),
                           in.size() * sizeof(float), cudaMemcpyHostToDevice,
                           nullptr);
        gpu_utils::check_gpu_call(cudaEventSynchronize(m_hist_ready));
    }
    void load_history(std::span<const uint8_t> in) override {
        check_packed("load_history(uint8_t)");
        check_size(in.size(), history_state_size(), "load_history");
        load_history_bytes(in.data(), in.size(), cudaMemcpyHostToDevice,
                           nullptr);
        gpu_utils::check_gpu_call(cudaEventSynchronize(m_hist_ready));
    }
    void save_history(DeviceSpan<float> d_out, Stream stream) const override {
        check_float("save_history(device float)");
        check_warm();
        check_size(d_out.size(), history_state_size(), "save_history");
        save_history_bytes(reinterpret_cast<uint8_t*>(d_out.data()),
                           d_out.size() * sizeof(float), to_cuda(stream));
    }
    void save_history(DeviceSpan<uint8_t> d_out, Stream stream) const override {
        check_packed("save_history(device uint8_t)");
        check_warm();
        check_size(d_out.size(), history_state_size(), "save_history");
        save_history_bytes(d_out.data(), d_out.size(), to_cuda(stream));
    }
    void load_history(DeviceSpan<const float> d_in, Stream stream) override {
        check_float("load_history(device float)");
        check_size(d_in.size(), history_state_size(), "load_history");
        load_history_bytes(reinterpret_cast<const uint8_t*>(d_in.data()),
                           d_in.size() * sizeof(float),
                           cudaMemcpyDeviceToDevice, to_cuda(stream));
    }
    void load_history(DeviceSpan<const uint8_t> d_in, Stream stream) override {
        check_packed("load_history(device uint8_t)");
        check_size(d_in.size(), history_state_size(), "load_history");
        load_history_bytes(d_in.data(), d_in.size(), cudaMemcpyDeviceToDevice,
                           to_cuda(stream));
    }

    // ---- Execute: device memory ------------------------------------------

    void execute(DeviceSpan<const float> d_waterfall,
                 DeviceSpan<float> d_dmt,
                 Stream stream) override {
        check_float("execute(device float)");
        gpu_utils::set_device(m_device_id);
        const auto rows   = m_nbeams * m_plan.get_nchans();
        const auto nsamps = d_waterfall.size() / rows;
        check_size(d_dmt.size(), out_size(nsamps), "execute (output)");
        const auto s = to_cuda(stream);
        core<32>(reinterpret_cast<const uint8_t*>(d_waterfall.data()),
                 nsamps * sizeof(float), nsamps, d_dmt.data(), s);
    }

    void execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                 SizeType nsamps,
                 DeviceSpan<int32_t> d_dmt,
                 Stream stream) override {
        check_packed("execute(device packed)");
        gpu_utils::set_device(m_device_id);
        const auto nbits     = m_plan.get_nbits();
        const auto rows      = m_nbeams * m_plan.get_nchans();
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        check_size(d_waterfall_packed.size(), rows * row_bytes,
                   "execute (packed input bytes)");
        check_size(d_dmt.size(), out_size(nsamps), "execute (output)");
        const auto s = to_cuda(stream);
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
                core<NBITS>(d_waterfall_packed.data(), row_bytes, nsamps,
                            d_dmt.data(), s);
            }
        });
    }

    void execute_time_major(DeviceSpan<const uint8_t> d_filterbank_packed,
                            SizeType nsamps,
                            DeviceSpan<int32_t> d_dmt,
                            Stream stream) override {
        check_packed("execute_time_major(device)");
        gpu_utils::set_device(m_device_id);
        const auto nbits      = m_plan.get_nbits();
        const auto nchans     = m_plan.get_nchans();
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        check_size(d_filterbank_packed.size(), m_nbeams * nsamps * samp_bytes,
                   "execute_time_major (input bytes)");
        check_size(d_dmt.size(), out_size(nsamps),
                   "execute_time_major (output)");
        const auto s         = to_cuda(stream);
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        // The transpose scratch is reused by the next call: order this
        // call after the previous one's reads of it.
        gpu_utils::check_gpu_call(cudaStreamWaitEvent(s, m_hist_ready, 0));
        m_tm_scratch.reserve(m_nbeams * nchans * row_bytes);
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
                launch_transpose<NBITS>(d_filterbank_packed.data(), samp_bytes,
                                        nsamps * samp_bytes, nsamps,
                                        m_tm_scratch.data(), row_bytes, s);
                core<NBITS>(m_tm_scratch.data(), row_bytes, nsamps,
                            d_dmt.data(), s);
            }
        });
    }

    // ---- Execute: host memory --------------------------------------------

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        check_float("execute(float)");
        gpu_utils::set_device(m_device_id);
        const auto rows   = m_nbeams * m_plan.get_nchans();
        const auto nsamps = waterfall.size() / rows;
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        run_host<32>(
            nsamps, dmt.data(),
            [&](SizeType n0, SizeType n1, uint8_t* h_dst, SizeType& bytes) {
                const auto cnt = n1 - n0;
                bytes          = rows * cnt * sizeof(float);
                auto* dst      = reinterpret_cast<float*>(h_dst);
                parallel_rows(rows, bytes, [&](SizeType r) {
                    std::memcpy(dst + (r * cnt), &waterfall[(r * nsamps) + n0],
                                cnt * sizeof(float));
                });
                return cnt * sizeof(float);
            });
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt) override {
        check_packed("execute(packed)");
        gpu_utils::set_device(m_device_id);
        const auto nbits     = m_plan.get_nbits();
        const auto rows      = m_nbeams * m_plan.get_nchans();
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        check_size(waterfall_packed.size(), rows * row_bytes,
                   "execute (packed input bytes)");
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
                // Chunk starts are multiples of 8 samples, so byte-aligned.
                run_host<NBITS>(
                    nsamps, dmt.data(),
                    [&](SizeType n0, SizeType n1, uint8_t* h_dst,
                        SizeType& bytes) {
                        const auto b0 =
                            bit_pack_utils::packed_row_bytes(n0, NBITS);
                        const auto b1 =
                            bit_pack_utils::packed_row_bytes(n1, NBITS);
                        const auto cnt = b1 - b0;
                        bytes          = rows * cnt;
                        parallel_rows(rows, bytes, [&](SizeType r) {
                            std::memcpy(h_dst + (r * cnt),
                                        &waterfall_packed[(r * row_bytes) + b0],
                                        cnt);
                        });
                        return cnt;
                    });
            }
        });
    }

    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt) override {
        check_packed("execute_time_major");
        gpu_utils::set_device(m_device_id);
        const auto nbits      = m_plan.get_nbits();
        const auto nchans     = m_plan.get_nchans();
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        check_size(filterbank_packed.size(), m_nbeams * nsamps * samp_bytes,
                   "execute_time_major (input bytes)");
        check_size(dmt.size(), out_size(nsamps), "execute_time_major (output)");
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
                run_host<NBITS>(
                    nsamps, dmt.data(),
                    [&](SizeType n0, SizeType n1, uint8_t* h_dst,
                        SizeType& bytes) {
                        const auto cnt = (n1 - n0) * samp_bytes;
                        bytes          = m_nbeams * cnt;
                        parallel_rows(m_nbeams, bytes, [&](SizeType b) {
                            std::memcpy(
                                h_dst + (b * cnt),
                                &filterbank_packed[(b * nsamps * samp_bytes) +
                                                   (n0 * samp_bytes)],
                                cnt);
                        });
                        // Time-major chunk: transposed on the device.
                        return SizeType{0};
                    },
                    samp_bytes);
            }
        });
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return detail::kGPUBackend;
    }

private:
    const plans::DDMTPlan& m_plan; // owned by the DDMT facade
    int m_device_id;
    SizeType m_nbeams;
    bool m_shared_sums; // SDMT: use the shared-sum kernel when it pays
    SizeType m_max_delay{0};
    SizeType m_ndm{0};
    SizeType m_gulp{kDefaultGulpSamples};
    bool m_gulp_is_default{true};

    // Plan tables on the device.
    thrust::device_vector<int> m_delay_d;  // (ndm, nchans), direct kernel
    thrust::device_vector<int> m_active_d; // active (unmasked) channels
    int m_nact{0};
    KernelKind m_kind{KernelKind::kDirect};
    int m_spread{0};
    thrust::device_vector<int16_t> m_offs_d; // [tile][active][TDM]
    thrust::device_vector<int> m_base_d;     // [tile][active]
    // SDMT programs (sdmt_gpu_plan.hpp).
    int m_sdmt_cfg{0};
    thrust::device_vector<int> m_sdmt_tile_sub_d;
    int m_sdmt_max_nodes{0};
    SizeType m_sdmt_smem{0};
    thrust::device_vector<sdmt_gpu::SubProg> m_sdmt_subs_d;
    thrust::device_vector<sdmt_gpu::Window> m_sdmt_wins_d;
    thrust::device_vector<sdmt_gpu::Node> m_sdmt_nodes_d;
    thrust::device_vector<int> m_sdmt_trials_d;

    // Retained history: m_history_len samples per row, row stride
    // m_hist_row_bytes, ping-pong so a call reads one buffer and writes the
    // next state into the other. m_hist_ready marks the latest update.
    DeviceBytes m_hist[2];
    int m_hist_idx{0};
    SizeType m_hist_row_bytes{0};
    SizeType m_history_len{0};
    cudaEvent_t m_hist_ready{nullptr};

    DeviceBytes m_tm_scratch;

    // Host-path pipeline.
    cudaStream_t m_htod_stream = nullptr;
    cudaStream_t m_exec_stream = nullptr;
    cudaStream_t m_dtoh_stream = nullptr;
    cudaEvent_t m_htod_done[2] = {nullptr, nullptr};
    cudaEvent_t m_exec_done[2] = {nullptr, nullptr};
    cudaEvent_t m_dtoh_done[2] = {nullptr, nullptr};
    PinnedBuffer<uint8_t> m_h_in[2];
    PinnedBuffer<uint8_t> m_h_out[2];
    DeviceBytes m_d_in[2];
    DeviceBytes m_d_cm[2]; // channel-major staging for time-major input
    DeviceBytes m_d_out[2];

    [[nodiscard]] const uint8_t* hist_cur() const noexcept {
        return m_hist[m_hist_idx].data();
    }

    [[nodiscard]] SizeType out_size(SizeType nsamps) const noexcept {
        return m_nbeams * m_ndm * get_output_nsamps(nsamps);
    }

    void check_float(std::string_view what) const {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "DDMT::{}: plan nbits={} != 32; use the packed-integer "
                "overload instead",
                what, m_plan.get_nbits()));
        }
    }
    void check_packed(std::string_view what) const {
        if (!is_packed_width(m_plan.get_nbits())) {
            throw std::invalid_argument(std::format(
                "DDMT::{}: plan nbits={} is not a supported packed width "
                "(1,2,4,8,16); use the float overload for nbits==32",
                what, m_plan.get_nbits()));
        }
    }
    void check_warm() const {
        if (m_history_len != m_max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, m_max_delay));
        }
    }
    static void
    check_size(SizeType got, SizeType expected, std::string_view what) {
        if (got != expected) {
            throw std::invalid_argument(
                std::format("DDMT (gpu): {} buffer size mismatch: expected "
                            "{}, got {}",
                            what, expected, got));
        }
    }

    void save_history_bytes(uint8_t* d_dst,
                            SizeType bytes,
                            cudaStream_t stream) const {
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaStreamWaitEvent(stream, m_hist_ready, 0));
        gpu_utils::check_gpu_call(cudaMemcpyAsync(d_dst, hist_cur(), bytes,
                                                  cudaMemcpyDeviceToDevice,
                                                  stream),
                                  "DDMT::save_history D2D copy failed");
        // A later call may overwrite the other buffer, never this one, but
        // record so load/save stay ordered with the next execute.
        gpu_utils::check_gpu_call(cudaEventRecord(m_hist_ready, stream));
    }

    void load_history_bytes(const uint8_t* src,
                            SizeType bytes,
                            cudaMemcpyKind kind,
                            cudaStream_t stream) {
        gpu_utils::set_device(m_device_id);
        gpu_utils::check_gpu_call(cudaStreamWaitEvent(stream, m_hist_ready, 0));
        const int next = 1 - m_hist_idx;
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(m_hist[next].data(), src, bytes, kind, stream),
            "DDMT::load_history copy failed");
        gpu_utils::check_gpu_call(cudaEventRecord(m_hist_ready, stream));
        m_hist_idx    = next;
        m_history_len = m_max_delay;
    }

    // ---- Setup -------------------------------------------------------------

    void init() {
        gpu_utils::set_device(m_device_id);
        const auto& pc    = m_plan.get_container();
        m_ndm             = pc.dm_arr.size();
        m_max_delay       = pc.delay_table.empty()
                                ? 0
                                : *std::ranges::max_element(pc.delay_table);
        const auto nchans = pc.nchans;
        const auto nbits  = pc.nbits;
        m_hist_row_bytes  = row_bytes_for(m_max_delay, nbits);
        for (auto& h : m_hist) {
            h.reserve(m_nbeams * nchans * m_hist_row_bytes);
        }

        std::vector<int> delays(pc.delay_table.size());
        std::ranges::transform(pc.delay_table, delays.begin(),
                               [](SizeType v) { return static_cast<int>(v); });
        m_delay_d.assign(delays.begin(), delays.end());
        std::vector<int> active;
        for (SizeType c = 0; c < nchans; ++c) {
            if (pc.kill_mask[c] != 0) {
                active.push_back(static_cast<int>(c));
            }
        }
        m_nact = static_cast<int>(active.size());
        m_active_d.assign(active.begin(), active.end());

        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            select_kernel<NBITS>(delays, active);
            if (m_shared_sums) {
                select_shared<NBITS>(delays, active);
            }
        });

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
        gpu_utils::check_gpu_call(
            cudaEventCreateWithFlags(&m_hist_ready, cudaEventDisableTiming));
        gpu_utils::check_gpu_call(cudaEventRecord(m_hist_ready, nullptr));
    }

    void destroy_streams() noexcept {
        for (int i = 0; i < 2; ++i) {
            for (auto* e : {m_htod_done[i], m_exec_done[i], m_dtoh_done[i]}) {
                if (e != nullptr) {
                    cudaEventDestroy(e);
                }
            }
        }
        if (m_hist_ready != nullptr) {
            cudaEventSynchronize(m_hist_ready);
            cudaEventDestroy(m_hist_ready);
        }
        for (auto* s : {m_htod_stream, m_exec_stream, m_dtoh_stream}) {
            if (s != nullptr) {
                cudaStreamDestroy(s);
            }
        }
    }

    /// Largest (tile-base-relative) delay offset over DM tiles of @p tdm.
    [[nodiscard]] int tile_spread(const std::vector<int>& delays,
                                  const std::vector<int>& active,
                                  int tdm) const {
        const auto nchans = static_cast<int>(m_plan.get_nchans());
        const auto ndm    = static_cast<int>(m_ndm);
        int spread        = 0;
        for (int t0 = 0; t0 < ndm; t0 += tdm) {
            const int t1 = std::min(t0 + tdm, ndm);
            for (const int c : active) {
                int mn = delays[(static_cast<SizeType>(t0) * nchans) + c];
                int mx = mn;
                for (int dm = t0 + 1; dm < t1; ++dm) {
                    const int v =
                        delays[(static_cast<SizeType>(dm) * nchans) + c];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
                spread = std::max(spread, mx - mn);
            }
        }
        return spread;
    }

    /// Builds the tiled kernel's offset/base tables for tiles of @p tdm.
    void build_tile_tables(const std::vector<int>& delays,
                           const std::vector<int>& active,
                           int tdm) {
        const auto nchans = static_cast<SizeType>(m_plan.get_nchans());
        const auto ndm    = static_cast<int>(m_ndm);
        const auto nact   = active.size();
        const int ntile   = (ndm + tdm - 1) / tdm;
        std::vector<int> base(static_cast<SizeType>(ntile) * nact);
        std::vector<int16_t> offs(base.size() * tdm);
        for (int t = 0; t < ntile; ++t) {
            for (SizeType k = 0; k < nact; ++k) {
                const auto c = static_cast<SizeType>(active[k]);
                auto at      = [&](int d) {
                    const int dm = std::min((t * tdm) + d, ndm - 1);
                    return delays[(static_cast<SizeType>(dm) * nchans) + c];
                };
                int mn = at(0);
                for (int d = 1; d < tdm; ++d) {
                    mn = std::min(mn, at(d));
                }
                const auto tk = (static_cast<SizeType>(t) * nact) + k;
                base[tk]      = mn;
                for (int d = 0; d < tdm; ++d) {
                    offs[(tk * tdm) + d] = static_cast<int16_t>(at(d) - mn);
                }
            }
        }
        m_base_d.assign(base.begin(), base.end());
        m_offs_d.assign(offs.begin(), offs.end());
    }

    /// Picks the kernel for this plan: the widest DM tile whose shared
    /// memory window fits a block's budget, else the direct kernel.
    template <unsigned NBITS>
    void select_kernel(const std::vector<int>& delays,
                       const std::vector<int>& active) {
        m_kind = KernelKind::kDirect;
        if (m_ndm == 0 || active.empty()) {
            return;
        }
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   m_device_id) != cudaSuccess ||
            optin <= 0) {
            optin = 48 * 1024;
        }
        // At least two resident blocks per SM hide the staging latency.
        const auto budget =
            static_cast<SizeType>(std::min(optin, 96 * 1024)) / 2;
        auto try_cfg = [&]<bool kWide>() {
            using Cfg   = typename CfgFor<NBITS, kWide>::type;
            using Shape = TileShape<NBITS, Cfg>;
            if (!kWide || m_ndm > static_cast<SizeType>(Shape::kTDM / 2)) {
                const int spread = tile_spread(delays, active, Shape::kTDM);
                if (spread < 32768 && Shape::smem_bytes(spread) <= budget) {
                    m_spread = spread;
                    m_kind   = kWide ? KernelKind::kWide : KernelKind::kNarrow;
                    build_tile_tables(delays, active, Shape::kTDM);
                    // Ceiling is per kernel specialization in this process;
                    // use device opt-in max so multiple DDMT plans coexist.
                    gpu_utils::check_gpu_call(
                        cudaFuncSetAttribute(
                            ddmt_gpu::ddmt_tiled_kernel<NBITS, Cfg>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, optin),
                        "DDMT: setting tiled kernel shared memory failed");
                    return true;
                }
            }
            return false;
        };
        if (!try_cfg.template operator()<true>()) {
            try_cfg.template operator()<false>();
        }
    }

    /**
     * @brief SDMT: plans the shared-sum kernel (sdmt_gpu_plan.hpp) for DM
     * tiles of 128 and 64 trials and keeps the one with fewer operations.
     * @details
     * Programs are split until they fit half the per-block shared memory
     * (two resident blocks per SM). The kernel then runs if its estimated
     * time is clearly below the DDMT kernel's; otherwise (sparse or coarse
     * grids, little sharing) the DDMT kernel chosen by select_kernel() runs.
     * Either way the sums are the same.
     */
    template <unsigned NBITS>
    void select_shared(const std::vector<int>& delays,
                       const std::vector<int>& active) {
        if (m_ndm == 0 || active.empty()) {
            return;
        }
        using E   = sdmt_gpu::Elem<NBITS>;
        int optin = 0;
        if (cudaDeviceGetAttribute(&optin,
                                   cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                   m_device_id) != cudaSuccess ||
            optin <= 0) {
            optin = 48 * 1024;
        }
        const auto budget =
            static_cast<SizeType>(std::min(optin, 96 * 1024)) / 2;
        const auto nchans = m_plan.get_nchans();
        std::optional<sdmt_gpu::Plan> best;
        int best_cfg = 0;
        for (int ci = 0; ci < kNumSdmtCfgs; ++ci) {
            with_sdmt_cfg(ci, [&]<class Cfg>() {
                if (Cfg::kTDM > 64 &&
                    m_ndm <= static_cast<SizeType>(Cfg::kTDM / 2)) {
                    return;
                }
                const auto table =
                    sdmt_gpu::trial_table_bytes(Cfg::kTDM) +
                    sdmt_gpu::node_table_bytes(sdmt_gpu::kMaxNodes);
                const auto max_arena =
                    static_cast<int>((budget - table) / sizeof(E));
                auto plan = sdmt_gpu::build_plan(delays, active, nchans, m_ndm,
                                                 Cfg::kTDM, Cfg::kBT, max_arena,
                                                 sdmt_gpu::kMaxNodes);
                if (plan.arena <= max_arena &&
                    (!best || plan.ops < best->ops)) {
                    best     = std::move(plan);
                    best_cfg = ci;
                }
            });
        }
        if (!best) {
            logging::debug("SDMT ({}): no shared-sum program fits; running "
                           "the DDMT kernel",
                           DMT_GPU_NAME);
            return;
        }
        // Estimated times from sustained rates measured on an L40S at the
        // reference point (4096 channels, 2049 trials): ~3.4e12 arena
        // operations/s for this kernel; ~6.6e12 (float), ~7.1e12 (16-bit)
        // and ~9.6e12 (<= 8-bit, SWAR) additions/s for the DDMT kernel.
        // Both are bound by the same shared-memory instruction rate, so the
        // ratio carries over to other devices.
        constexpr double kSharedRate = 3.4e12;
        constexpr double kDdmtRate   = NBITS == 32   ? 6.6e12
                                       : NBITS == 16 ? 7.1e12
                                                     : 9.6e12;
        const double speedup =
            (best->ddmt / kDdmtRate) / (best->ops / kSharedRate);
        if (speedup < 1.1 && !detail::sdmt_gpu_always_shared()) {
            logging::debug("SDMT ({}): shared sums do not pay (estimated "
                           "{:.2f}x); running the DDMT kernel",
                           DMT_GPU_NAME, speedup);
            return;
        }
        logging::debug("SDMT ({}): shared-sum kernel, DM tiles of {}, {} "
                       "programs, arena {} elements, estimated {:.2f}x "
                       "faster than DDMT",
                       DMT_GPU_NAME, best->tdm, best->subs.size(), best->arena,
                       speedup);
        m_kind           = KernelKind::kShared;
        m_sdmt_cfg       = best_cfg;
        m_sdmt_max_nodes = best->max_nodes;
        m_sdmt_smem = sdmt_gpu::smem_bytes<NBITS>(best->tdm, best->max_nodes,
                                                  best->arena);
        m_sdmt_tile_sub_d.assign(best->tile_sub.begin(), best->tile_sub.end());
        m_sdmt_subs_d.assign(best->subs.begin(), best->subs.end());
        m_sdmt_wins_d.assign(best->wins.begin(), best->wins.end());
        m_sdmt_nodes_d.assign(best->nodes.begin(), best->nodes.end());
        m_sdmt_trials_d.assign(best->trials.begin(), best->trials.end());
        with_sdmt_cfg(best_cfg, [&]<class Cfg>() {
            gpu_utils::check_gpu_call(
                cudaFuncSetAttribute(
                    sdmt_gpu::sdmt_kernel<NBITS, Cfg::kTY, Cfg::kDPT,
                                          Cfg::kSPW>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, optin),
                "SDMT: setting kernel shared memory failed");
        });
    }

    template <unsigned NBITS, class Cfg>
    void launch_shared(const DDMTSegments& seg,
                       SizeType n_out,
                       DDMTAcc<NBITS>* d_out,
                       SizeType out_dm_stride,
                       SizeType out_beam_stride,
                       cudaStream_t stream) const {
        const dim3 block(32, Cfg::kTY);
        const dim3 grid(
            static_cast<unsigned>((n_out + Cfg::kBT - 1) / Cfg::kBT),
            static_cast<unsigned>((m_ndm + Cfg::kTDM - 1) / Cfg::kTDM),
            static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        sdmt_gpu::sdmt_kernel<NBITS, Cfg::kTY, Cfg::kDPT, Cfg::kSPW>
            <<<grid, block, m_sdmt_smem, stream>>>(
                seg, d_out, out_dm_stride, out_beam_stride,
                thrust::raw_pointer_cast(m_sdmt_subs_d.data()),
                thrust::raw_pointer_cast(m_sdmt_wins_d.data()),
                thrust::raw_pointer_cast(m_sdmt_nodes_d.data()),
                thrust::raw_pointer_cast(m_sdmt_trials_d.data()),
                thrust::raw_pointer_cast(m_sdmt_tile_sub_d.data()),
                m_sdmt_max_nodes, static_cast<int>(m_plan.get_nchans()),
                static_cast<int>(m_ndm), static_cast<int>(n_out));
        gpu_utils::check_last_gpu_error("sdmt_kernel launch failed");
    }

    // ---- Kernels
    // -------------------------------------------------------------

    template <unsigned NBITS, class Cfg>
    void launch_tiled(const DDMTSegments& seg,
                      SizeType n_out,
                      DDMTAcc<NBITS>* d_out,
                      SizeType out_dm_stride,
                      SizeType out_beam_stride,
                      cudaStream_t stream) const {
        using Shape = TileShape<NBITS, Cfg>;
        const dim3 block(Cfg::kTX, Cfg::kTY);
        const dim3 grid(
            static_cast<unsigned>((n_out + Shape::kBT - 1) / Shape::kBT),
            static_cast<unsigned>((m_ndm + Shape::kTDM - 1) / Shape::kTDM),
            static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        ddmt_gpu::ddmt_tiled_kernel<NBITS, Cfg>
            <<<grid, block, Shape::smem_bytes(m_spread), stream>>>(
                seg, d_out, out_dm_stride, out_beam_stride,
                thrust::raw_pointer_cast(m_offs_d.data()),
                thrust::raw_pointer_cast(m_base_d.data()),
                thrust::raw_pointer_cast(m_active_d.data()), m_nact,
                static_cast<int>(m_plan.get_nchans()), static_cast<int>(m_ndm),
                static_cast<int>(n_out), Shape::words(m_spread));
        gpu_utils::check_last_gpu_error("ddmt_tiled_kernel launch failed");
    }

    /// Dedisperses output samples [0, n_out) of the stream in @p seg.
    template <unsigned NBITS>
    void dedisperse(const DDMTSegments& seg,
                    SizeType n_out,
                    DDMTAcc<NBITS>* d_out,
                    SizeType out_dm_stride,
                    SizeType out_beam_stride,
                    cudaStream_t stream) const {
        if (n_out == 0 || m_ndm == 0) {
            return;
        }
        switch (m_kind) {
        case KernelKind::kWide:
            launch_tiled<NBITS, typename CfgFor<NBITS, true>::type>(
                seg, n_out, d_out, out_dm_stride, out_beam_stride, stream);
            return;
        case KernelKind::kNarrow:
            launch_tiled<NBITS, typename CfgFor<NBITS, false>::type>(
                seg, n_out, d_out, out_dm_stride, out_beam_stride, stream);
            return;
        case KernelKind::kShared:
            with_sdmt_cfg(m_sdmt_cfg, [&]<class Cfg>() {
                launch_shared<NBITS, Cfg>(seg, n_out, d_out, out_dm_stride,
                                          out_beam_stride, stream);
            });
            return;
        case KernelKind::kDirect:
            break;
        }
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((n_out + 255) / 256),
                        static_cast<unsigned>(std::min<SizeType>(m_ndm, 65535)),
                        static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        ddmt_gpu::ddmt_direct_kernel<NBITS><<<grid, block, 0, stream>>>(
            seg, d_out, out_dm_stride, out_beam_stride,
            thrust::raw_pointer_cast(m_delay_d.data()),
            thrust::raw_pointer_cast(m_active_d.data()), m_nact,
            static_cast<int>(m_plan.get_nchans()), static_cast<int>(m_ndm),
            static_cast<int>(n_out));
        gpu_utils::check_last_gpu_error("ddmt_direct_kernel launch failed");
    }

    template <unsigned NBITS>
    void launch_copy_tail(const DDMTSegments& seg,
                          SizeType start,
                          SizeType len,
                          uint8_t* dst,
                          cudaStream_t stream) const {
        if (len == 0) {
            return;
        }
        constexpr SizeType kPer = NBITS < 8 ? 8 / NBITS : 1;
        const auto nrows        = m_nbeams * m_plan.get_nchans();
        const auto units        = (len + kPer - 1) / kPer;
        const auto ry           = std::min<SizeType>(nrows, 65535);
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned>((units + 255) / 256),
                        static_cast<unsigned>(ry),
                        static_cast<unsigned>((nrows + ry - 1) / ry));
        ddmt_gpu::ddmt_copy_tail_kernel<NBITS><<<grid, block, 0, stream>>>(
            seg, static_cast<int>(start), static_cast<int>(len), dst,
            m_hist_row_bytes, nrows);
        gpu_utils::check_last_gpu_error("ddmt_copy_tail_kernel launch failed");
    }

    template <unsigned NBITS>
    void launch_transpose(const uint8_t* d_src,
                          SizeType samp_bytes,
                          SizeType src_beam_stride,
                          SizeType nsamps,
                          uint8_t* d_dst,
                          SizeType dst_row_bytes,
                          cudaStream_t stream) const {
        if (nsamps == 0) {
            return;
        }
        constexpr SizeType kTS = 32 * (NBITS < 8 ? 8 / NBITS : 1);
        const auto nchans      = m_plan.get_nchans();
        const dim3 block(32, 8);
        const dim3 grid(static_cast<unsigned>((nsamps + kTS - 1) / kTS),
                        static_cast<unsigned>((nchans + 31) / 32),
                        static_cast<unsigned>(m_nbeams));
        gpu_utils::check_kernel_launch_params(grid, block);
        ddmt_gpu::ddmt_transpose_kernel<NBITS><<<grid, block, 0, stream>>>(
            d_src, samp_bytes, src_beam_stride, static_cast<int>(nsamps),
            static_cast<int>(nchans), d_dst, dst_row_bytes);
        gpu_utils::check_last_gpu_error("ddmt_transpose_kernel launch failed");
    }

    /**
     * @brief One streaming step, fully on @p stream: dedisperses the
     * retained history followed by @p n_new new samples (device rows of
     * @p new_row_bytes at @p d_new) into @p d_out (nbeams, ndm, n_out),
     * then refreshes the device history. No host synchronisation.
     * @return n_out, the number of output samples written per DM.
     */
    template <unsigned NBITS>
    SizeType core(const uint8_t* d_new,
                  SizeType new_row_bytes,
                  SizeType n_new,
                  DDMTAcc<NBITS>* d_out,
                  cudaStream_t stream) {
        gpu_utils::check_gpu_call(cudaStreamWaitEvent(stream, m_hist_ready, 0));
        const auto hist_len = m_history_len;
        const auto total    = hist_len + n_new;
        const auto n_out    = total > m_max_delay ? total - m_max_delay : 0;
        DDMTSegments seg;
        seg.a           = hist_cur();
        seg.a_row_bytes = m_hist_row_bytes;
        seg.a_len       = static_cast<int>(hist_len);
        seg.b           = d_new;
        seg.b_row_bytes = new_row_bytes;
        seg.total       = static_cast<int>(total);
        dedisperse<NBITS>(seg, n_out, d_out, n_out, m_ndm * n_out, stream);

        const auto new_len = std::min(total, m_max_delay);
        const int next     = 1 - m_hist_idx;
        launch_copy_tail<NBITS>(seg, total - new_len, new_len,
                                m_hist[next].data(), stream);
        gpu_utils::check_gpu_call(cudaEventRecord(m_hist_ready, stream));
        m_hist_idx    = next;
        m_history_len = new_len;
        return n_out;
    }

    /**
     * @brief Host-memory execute: streams the new samples through core()
     * in chunks of m_gulp samples, overlapping the H2D copy, the kernels and
     * the D2H copy of consecutive chunks (double buffered). The result is
     * identical to a single core() call over all the samples.
     *
     * @p stage(n0, n1, h_dst, bytes) packs new samples [n0, n1) into pinned
     * @p h_dst, sets @p bytes to the bytes written and returns the device
     * row stride of the chunk -- or 0 for time-major chunks
     * (@p samp_bytes > 0), which are transposed on the device first.
     */
    template <unsigned NBITS, typename Stage>
    void run_host(SizeType nsamps,
                  DDMTAcc<NBITS>* h_out,
                  Stage&& stage,
                  SizeType samp_bytes = 0) {
        using Acc          = DDMTAcc<NBITS>;
        const auto rows    = m_nbeams * m_plan.get_nchans();
        const auto n_total = get_output_nsamps(nsamps);
        // With the default gulp, split large inputs into >= 3 chunks (so
        // copies overlap the kernels) as long as each keeps >= 16384
        // samples (so each kernel still fills the device). Chunks start on
        // multiples of 8 samples: byte-aligned when packed.
        auto gulp = m_gulp;
        if (m_gulp_is_default) {
            gulp =
                std::clamp<SizeType>((nsamps + 2) / 3, kMinAutoChunk, m_gulp);
        }
        const auto chunk  = std::max<SizeType>(8, (gulp + 7) / 8 * 8);
        const auto nchunk = (nsamps + chunk - 1) / chunk;
        const auto cm_row = row_bytes_for(std::min(chunk, nsamps), NBITS);
        const auto in_cap =
            samp_bytes > 0 ? m_nbeams * std::min(chunk, nsamps) * samp_bytes
                           : rows * cm_row;
        // A chunk of n new samples yields at most n outputs per DM.
        const auto out_cap = m_nbeams * m_ndm * std::min(chunk, nsamps);
        for (int i = 0; i < 2; ++i) {
            m_h_in[i].reserve(in_cap);
            m_d_in[i].reserve(in_cap);
            m_h_out[i].reserve(out_cap * sizeof(Acc));
            m_d_out[i].reserve(out_cap * sizeof(Acc));
            if (samp_bytes > 0) {
                m_d_cm[i].reserve(rows * cm_row);
            }
        }
        gpu_utils::check_gpu_call(
            cudaStreamWaitEvent(m_exec_stream, m_hist_ready, 0));

        SizeType out_pos = 0;
        struct Pending {
            SizeType n_out{0};
            SizeType pos{0};
            int buf{0};
            bool valid{false};
        } prev;
        auto drain = [&](const Pending& p) {
            if (!p.valid) {
                return;
            }
            gpu_utils::check_gpu_call(cudaEventSynchronize(m_dtoh_done[p.buf]));
            const auto* src =
                reinterpret_cast<const Acc*>(m_h_out[p.buf].data());
            const auto out_rows = m_nbeams * m_ndm;
            parallel_rows(
                out_rows, out_rows * p.n_out * sizeof(Acc), [&](SizeType r) {
                    std::memcpy(h_out + (r * n_total) + p.pos,
                                src + (r * p.n_out), p.n_out * sizeof(Acc));
                });
        };

        for (SizeType k = 0; k < nchunk; ++k) {
            const int buf = static_cast<int>(k % 2);
            const auto n0 = k * chunk;
            const auto n1 = std::min(nsamps, n0 + chunk);
            if (k >= 2) {
                // Chunk k-2 used these buffers; its output is drained below
                // at iteration k-1 at the latest, so only the device side
                // needs to be finished.
                gpu_utils::check_gpu_call(
                    cudaEventSynchronize(m_dtoh_done[buf]));
            }
            SizeType bytes     = 0;
            const auto row_str = stage(n0, n1, m_h_in[buf].data(), bytes);
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_d_in[buf].data(), m_h_in[buf].data(), bytes,
                                cudaMemcpyHostToDevice, m_htod_stream),
                "DDMT: H2D copy failed");
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_htod_done[buf], m_htod_stream));
            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_exec_stream, m_htod_done[buf], 0));

            const uint8_t* d_cm = m_d_in[buf].data();
            SizeType cm_stride  = row_str;
            if constexpr (NBITS != 32) {
                if (samp_bytes > 0) {
                    cm_stride = row_bytes_for(n1 - n0, NBITS);
                    launch_transpose<NBITS>(
                        m_d_in[buf].data(), samp_bytes, (n1 - n0) * samp_bytes,
                        n1 - n0, m_d_cm[buf].data(), cm_stride, m_exec_stream);
                    d_cm = m_d_cm[buf].data();
                }
            }
            auto* d_out = reinterpret_cast<Acc*>(m_d_out[buf].data());
            const auto n_out =
                core<NBITS>(d_cm, cm_stride, n1 - n0, d_out, m_exec_stream);
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_exec_done[buf], m_exec_stream));
            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(m_dtoh_stream, m_exec_done[buf], 0));
            if (n_out > 0) {
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(m_h_out[buf].data(), d_out,
                                    m_nbeams * m_ndm * n_out * sizeof(Acc),
                                    cudaMemcpyDeviceToHost, m_dtoh_stream),
                    "DDMT: D2H copy failed");
            }
            gpu_utils::check_gpu_call(
                cudaEventRecord(m_dtoh_done[buf], m_dtoh_stream));

            drain(prev);
            prev = {n_out, out_pos, buf, n_out > 0};
            out_pos += n_out;
        }
        drain(prev);
        gpu_utils::check_gpu_call(cudaStreamSynchronize(m_exec_stream));
    }
};

} // namespace

std::unique_ptr<detail::DDMTEngine>
detail::make_ddmt_gpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<DDMTCudaEngine>(plan, cfg, false);
}

std::unique_ptr<detail::DDMTEngine>
detail::make_sdmt_gpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<DDMTCudaEngine>(plan, cfg, true);
}

} // namespace dmt::algorithms
