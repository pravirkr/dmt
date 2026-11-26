#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <span>
#include <thread>
#include <vector>

#include "dmt/common/types.hpp"

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

/**
 * @file host_staging.cuh
 * @brief Host-memory staging for the GPU engines' host-memory entry points:
 * page-locked buffers, multi-threaded row copies, and a chunked pipeline
 * between pageable host memory and the device.
 */

namespace dmt::gpu_host {

/**
 * @brief RAII page-locked ("pinned") host buffer.
 * @details
 * cudaHostAlloc'd memory transfers 2-3x faster than regular heap memory and
 * is required for cudaMemcpyAsync to actually run asynchronously with
 * respect to the host -- see dedisp's TDDPlan (cu::HostMemory) for the same
 * rationale. reserve() never shrinks, so repeated calls reuse one
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
        gpu_utils::check_allocation(
            cudaHostAlloc(reinterpret_cast<void**>(&m_ptr),
                          std::max<SizeType>(n, 1) * sizeof(T),
                          cudaHostAllocDefault),
            n * sizeof(T), "pinned host allocation");
        m_capacity = n;
    }
    [[nodiscard]] T* data() noexcept { return m_ptr; }
    [[nodiscard]] const T* data() const noexcept { return m_ptr; }
    [[nodiscard]] SizeType capacity() const noexcept { return m_capacity; }

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

/**
 * @brief Runs f(i) for i in [0, n) on up to @p max_threads host threads when
 * the rows move enough bytes; staging copies between pageable and pinned
 * memory are otherwise bound by one core's memcpy bandwidth. When
 * @p max_threads <= 1, runs inline on the calling thread without spawning
 * threads.
 */
template <typename F>
void parallel_rows(SizeType n,
                   SizeType bytes_total,
                   int max_threads,
                   const F& f) {
    constexpr SizeType kBytesPerThread = SizeType{4} << 20;
    const auto cap = static_cast<SizeType>(std::max(1, max_threads));
    const auto hw  = static_cast<SizeType>(
        std::max(1U, std::thread::hardware_concurrency()));
    const auto nt =
        std::min({n, cap, std::min<SizeType>(hw, 16),
                  std::max<SizeType>(1, bytes_total / kBytesPerThread)});
    if (nt <= 1) {
        for (SizeType i = 0; i < n; ++i) {
            f(i);
        }
        return;
    }
    std::vector<std::jthread> threads;
    threads.reserve(nt);
    for (SizeType t = 0; t < nt; ++t) {
        threads.emplace_back([&f, n, nt, t] {
            for (SizeType i = (n * t) / nt; i < (n * (t + 1)) / nt; ++i) {
                f(i);
            }
        });
    }
}

/**
 * @brief Host-memory transfers of a GPU engine on one stream.
 * @details
 * Host to device: a plain pageable cudaMemcpyAsync. The driver stages it
 * efficiently, and host-side staging threads measured no faster.
 *
 * Device to host: large results (at least pinned_min_bytes()) are cut into
 * kChunkBytes chunks, and m_workers host workers (clamped from the engine's
 * nthreads to [1, kMaxWorkers]) each drain every m_workers-th chunk through
 * their own two page-locked buffers: while one of a worker's chunks is on the
 * bus, the worker copies the other out with memcpy, so PCIe and the pageable
 * memcpy overlap. When m_workers == 1 the work runs inline on the calling
 * thread with no threads spawned; when m_workers > 1 each spawned thread sets
 * m_device_id before any GPU API call. The pinned footprint is fixed
 * (2 * m_workers chunks, taken on first use) whatever the result size.
 * Smaller results use one pageable copy.
 */
class ChunkedStager {
public:
    static constexpr SizeType kChunkBytes =
        algorithms::detail::kHostStagingChunkBytes;
    static constexpr int kMaxWorkers =
        algorithms::detail::kHostStagingMaxWorkers;

    /// One contiguous piece of a transfer.
    struct Segment {
        void* dst;
        const void* src;
        SizeType bytes;
    };

    explicit ChunkedStager(int device_id = 0, int nthreads = 1) noexcept
        : m_device_id(device_id),
          m_workers(algorithms::detail::host_staging_workers(nthreads)) {}
    ~ChunkedStager() {
        bool has_gpu_res = (m_stream != nullptr);
        for (int b = 0; b < 2 * m_workers; ++b) {
            has_gpu_res = has_gpu_res || (m_done[b] != nullptr);
        }
        if (has_gpu_res) {
            (void)cudaSetDevice(m_device_id);
        }
        for (int b = 0; b < 2 * m_workers; ++b) {
            if (m_done[b] != nullptr) {
                cudaEventDestroy(m_done[b]);
            }
        }
        if (m_stream != nullptr) {
            cudaStreamDestroy(m_stream);
        }
    }
    ChunkedStager(const ChunkedStager&)            = delete;
    ChunkedStager& operator=(const ChunkedStager&) = delete;
    ChunkedStager(ChunkedStager&&)                 = delete;
    ChunkedStager& operator=(ChunkedStager&&)      = delete;

    /// Pinned host bytes currently allocated by this stager (0 until a large
    /// to_host() triggers pin_buffers()).
    [[nodiscard]] SizeType pinned_bytes() const noexcept {
        SizeType total = 0;
        for (int b = 0; b < 2 * m_workers; ++b) {
            total += m_pin[b].capacity();
        }
        return total;
    }

    /// Minimum D2H transfer size (bytes) that uses the pinned pipeline.
    [[nodiscard]] SizeType pinned_min_bytes() const noexcept {
        return SizeType{2} * static_cast<SizeType>(m_workers) * kChunkBytes;
    }

    /// The stream the copies run on (created on first use); device work
    /// between a to_device() and a to_host() should be issued on it.
    [[nodiscard]] cudaStream_t stream() {
        init();
        return m_stream;
    }

    /// Enqueues pageable @p src -> device @p dst on stream().
    void to_device(void* dst, const void* src, SizeType bytes) {
        init();
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, m_stream),
            "host staging: H2D copy failed");
    }

    /// Copies every segment (src on the device, dst in pageable host
    /// memory) after the work already on stream(); returns when the host
    /// buffers hold the data. A worker issues each chunk's device copy one
    /// chunk ahead of draining it.
    void to_host(std::span<const Segment> segs) {
        init();
        SizeType total = 0;
        for (const auto& sg : segs) {
            total += sg.bytes;
        }
        if (total < pinned_min_bytes()) {
            for (const auto& sg : segs) {
                gpu_utils::check_gpu_call(
                    cudaMemcpyAsync(sg.dst, sg.src, sg.bytes,
                                    cudaMemcpyDeviceToHost, m_stream),
                    "host staging: D2H copy failed");
            }
            gpu_utils::check_gpu_call(cudaStreamSynchronize(m_stream));
            return;
        }
        pin_buffers();
        auto issue = [&](int b, const Segment& c) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_pin[b].data(), c.src, c.bytes,
                                cudaMemcpyDeviceToHost, m_stream),
                "host staging: D2H copy failed");
            gpu_utils::check_gpu_call(cudaEventRecord(m_done[b], m_stream));
        };
        auto drain = [&](int b, const Segment& c) {
            gpu_utils::check_gpu_call(cudaEventSynchronize(m_done[b]));
            std::memcpy(c.dst, m_pin[b].data(), c.bytes);
        };
        run(segs, drain, issue);
    }

private:
    /// Cuts the segments into chunks. Worker w (of m_workers) handles chunks
    /// w, w + m_workers, ... in order, alternating between its pinned buffers
    /// 2w and 2w + 1: ahead(buffer, chunk) for its next chunk, then
    /// body(buffer, chunk) for the current one.
    template <typename Body, typename Ahead>
    void
    run(std::span<const Segment> segs, const Body& body, const Ahead& ahead) {
        std::vector<Segment> chunks;
        for (const auto& sg : segs) {
            for (SizeType off = 0; off < sg.bytes; off += kChunkBytes) {
                chunks.push_back({static_cast<uint8_t*>(sg.dst) + off,
                                  static_cast<const uint8_t*>(sg.src) + off,
                                  std::min(kChunkBytes, sg.bytes - off)});
            }
        }
        const auto n = static_cast<int>(chunks.size());
        std::exception_ptr error[kMaxWorkers];
        auto work = [&](int w, bool bind_device) {
            try {
                if (bind_device) {
                    // Fresh OS threads default to device 0; select the
                    // engine's device before any runtime call so multi-GPU
                    // runs never create a stray context on device 0.
                    gpu_utils::set_device(m_device_id);
                }
                if (w < n) {
                    ahead(2 * w, chunks[w]);
                }
                for (int k = w, j = 0; k < n; k += m_workers, ++j) {
                    if (k + m_workers < n) {
                        ahead((2 * w) + ((j + 1) % 2), chunks[k + m_workers]);
                    }
                    body((2 * w) + (j % 2), chunks[k]);
                }
            } catch (...) {
                error[w] = std::current_exception();
            }
        };
        if (m_workers <= 1) {
            work(0, false);
        } else {
            std::vector<std::jthread> threads;
            threads.reserve(static_cast<SizeType>(m_workers));
            for (int w = 0; w < m_workers; ++w) {
                threads.emplace_back(work, w, true);
            }
        }
        for (int w = 0; w < m_workers; ++w) {
            if (error[w]) {
                gpu_utils::check_gpu_call(cudaStreamSynchronize(m_stream));
                std::rethrow_exception(error[w]);
            }
        }
    }

    void init() {
        if (m_stream != nullptr) {
            return;
        }
        gpu_utils::set_device(m_device_id);
        // A blocking stream: legacy default-stream work (synchronous
        // cudaMemcpy of host snapshots, say) stays ordered with it.
        gpu_utils::check_gpu_call(cudaStreamCreate(&m_stream),
                                  "host staging: stream creation failed");
    }

    void pin_buffers() {
        if (m_done[0] != nullptr) {
            return;
        }
        for (int b = 0; b < 2 * m_workers; ++b) {
            m_pin[b].reserve(kChunkBytes);
            gpu_utils::check_gpu_call(
                cudaEventCreateWithFlags(&m_done[b], cudaEventDisableTiming),
                "host staging: event creation failed");
            // Recorded once, so the first waits return immediately.
            gpu_utils::check_gpu_call(cudaEventRecord(m_done[b], m_stream));
        }
    }

    int m_device_id{0};
    int m_workers{1};
    cudaStream_t m_stream{nullptr};
    PinnedBuffer<uint8_t> m_pin[2 * kMaxWorkers];
    cudaEvent_t m_done[2 * kMaxWorkers]{};
};

} // namespace dmt::gpu_host
