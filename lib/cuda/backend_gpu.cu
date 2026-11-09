// Runtime helpers behind dmt/common/backend.hpp that need the GPU runtime
// (CUDA, or HIP through gpu_compat.cuh).

#include <string_view>

#include "dmt/common/backend.hpp"

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

namespace dmt {

namespace {

void check_gpu_backend(Backend backend, std::string_view what) {
    if (backend != algorithms::detail::kGPUBackend) {
        algorithms::detail::throw_unavailable(what, backend);
    }
}

// Restores the calling thread's current device on scope exit.
class CurrentDevice {
public:
    explicit CurrentDevice(int device) {
        gpu_utils::check_gpu_call(cudaGetDevice(&m_previous),
                                  "Failed to get device");
        if (device != m_previous) {
            gpu_utils::set_device(device);
        }
    }
    CurrentDevice(const CurrentDevice&)            = delete;
    CurrentDevice& operator=(const CurrentDevice&) = delete;
    ~CurrentDevice() { (void)cudaSetDevice(m_previous); }

private:
    int m_previous = 0;
};

} // namespace

int pointer_device(Backend backend, const void* ptr) {
    if (backend != algorithms::detail::kGPUBackend || ptr == nullptr) {
        return -1;
    }
#if defined(DMT_ENABLE_HIP)
    // The CUDA array interface is the only user, and it is CUDA only.
    return -1;
#else
    cudaPointerAttributes attrs{};
    if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) {
        (void)cudaGetLastError(); // not a sticky error: clear it
        return -1;
    }
    if (attrs.type != cudaMemoryTypeDevice &&
        attrs.type != cudaMemoryTypeManaged) {
        return -1;
    }
    return attrs.device;
#endif
}

void stream_wait(Backend backend,
                 int device,
                 Stream consumer,
                 Stream producer) {
    check_gpu_backend(backend, "stream_wait");
    if (consumer.native == producer.native) {
        return;
    }
    const CurrentDevice guard(device);
    cudaEvent_t event = nullptr;
    gpu_utils::check_gpu_call(
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
        "stream_wait: failed to create an event");
    const auto record =
        cudaEventRecord(event, static_cast<cudaStream_t>(producer.native));
    const auto wait =
        record == cudaSuccess
            ? cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer.native),
                                  event, 0)
            : record;
    // Destroying a pending event is fine: the wait already captured it.
    (void)cudaEventDestroy(event);
    gpu_utils::check_gpu_call(
        record, "stream_wait: failed to record on the producer stream");
    gpu_utils::check_gpu_call(
        wait, "stream_wait: failed to wait on the consumer stream");
}

} // namespace dmt
