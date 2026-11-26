#include <atomic>
#include <cstdint>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "dmt/common/backend.hpp"

#include "test_gpu_compat.cuh"
#include "test_helpers.hpp"

namespace dmt {

namespace {

void check_gpu_call(cudaError_t err) {
    INFO(cudaGetErrorString(err));
    REQUIRE(err == cudaSuccess);
}

__global__ void wait_then_set(const volatile int* go, int* flag) {
    while (*go == 0) {
    }
    *flag = 1;
}

__global__ void copy_flag(const int* flag, int* out) { *out = *flag; }

} // namespace

#if !defined(DMT_ENABLE_HIP)
TEST_CASE("pointer_device finds the device of CUDA memory", "[backend][gpu]") {
    int device = 0;
    check_gpu_call(cudaGetDevice(&device));
    void* dev_ptr = nullptr;
    check_gpu_call(cudaMalloc(&dev_ptr, 64));
    void* managed = nullptr;
    check_gpu_call(cudaMallocManaged(&managed, 64));
    int host = 0;

    CHECK(pointer_device(Backend::kCUDA, dev_ptr) == device);
    CHECK(pointer_device(Backend::kCUDA, static_cast<char*>(dev_ptr) + 8) ==
          device);
    CHECK(pointer_device(Backend::kCUDA, managed) == device);
    CHECK(pointer_device(Backend::kCUDA, &host) == -1);
    CHECK(pointer_device(Backend::kCUDA, nullptr) == -1);
    CHECK(pointer_device(Backend::kCPU, dev_ptr) == -1);
    CHECK(pointer_device(Backend::kHIP, dev_ptr) == -1);
    // A failed lookup leaves no sticky error behind.
    CHECK(cudaGetLastError() == cudaSuccess);

    check_gpu_call(cudaFree(managed));
    check_gpu_call(cudaFree(dev_ptr));
}
#endif

TEST_CASE("stream_wait orders a consumer stream after a producer",
          "[backend][gpu]") {
    int device = 0;
    check_gpu_call(cudaGetDevice(&device));
    int* d_flag = nullptr;
    int* d_out  = nullptr;
    check_gpu_call(cudaMalloc(&d_flag, sizeof(int)));
    check_gpu_call(cudaMalloc(&d_out, sizeof(int)));
    check_gpu_call(cudaMemset(d_flag, 0, sizeof(int)));
    check_gpu_call(cudaMemset(d_out, 0, sizeof(int)));
    check_gpu_call(cudaDeviceSynchronize());

    int* h_go = nullptr;
    int* d_go = nullptr;
    check_gpu_call(cudaHostAlloc(&h_go, sizeof(int), cudaHostAllocMapped));
    check_gpu_call(cudaHostGetDevicePointer(&d_go, h_go, 0));
    *h_go = 0;
    struct ReleaseGuard {
        int* ptr;
        void release() noexcept {
            if (ptr != nullptr) {
                std::atomic_ref<int>(*ptr).store(1, std::memory_order_release);
            }
        }
        ~ReleaseGuard() {
            release();
            if (ptr != nullptr) {
                (void)cudaDeviceSynchronize();
                (void)cudaFreeHost(ptr);
            }
        }
    } guard{h_go};

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    check_gpu_call(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    check_gpu_call(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));

    // Producer waits on the host-mapped flag until the host releases it.
    wait_then_set<<<1, 1, 0, producer>>>(d_go, d_flag);
    check_gpu_call(cudaGetLastError());
    stream_wait(test::gpu_exec().backend, device, Stream{consumer},
                Stream{producer});
    copy_flag<<<1, 1, 0, consumer>>>(d_flag, d_out);
    check_gpu_call(cudaGetLastError());
    // Neither stream can finish before the host opens the gate.
    CHECK(cudaStreamQuery(producer) == cudaErrorNotReady);
    CHECK(cudaStreamQuery(consumer) == cudaErrorNotReady);
    guard.release();
    check_gpu_call(cudaStreamSynchronize(consumer));
    int out = 0;
    check_gpu_call(
        cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(out == 1);

    // Same stream: nothing to do. Wrong backend: refused.
    stream_wait(test::gpu_exec().backend, device, Stream{consumer},
                Stream{consumer});
    CHECK_THROWS_AS(
        stream_wait(Backend::kCPU, device, Stream{consumer}, Stream{producer}),
        std::invalid_argument);

    check_gpu_call(cudaStreamSynchronize(producer));
    check_gpu_call(cudaStreamDestroy(consumer));
    check_gpu_call(cudaStreamDestroy(producer));
    check_gpu_call(cudaFree(d_out));
    check_gpu_call(cudaFree(d_flag));
}

} // namespace dmt
