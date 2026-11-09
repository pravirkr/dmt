#include <cstdint>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "dmt/common/backend.hpp"

#include "dmt/engines.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"

namespace dmt {

namespace {

using algorithms::detail::kGPUBackend;
using gpu_utils::check_gpu_call;

__global__ void spin_then_set(int* flag, long long cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
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

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    check_gpu_call(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
    check_gpu_call(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));

    // ~0.1 s on the producer before the flag is set.
    spin_then_set<<<1, 1, 0, producer>>>(d_flag, 200'000'000LL);
    check_gpu_call(cudaGetLastError());
    stream_wait(kGPUBackend, device, Stream{consumer}, Stream{producer});
    copy_flag<<<1, 1, 0, consumer>>>(d_flag, d_out);
    check_gpu_call(cudaGetLastError());
    // The host was not blocked: the producer is still running.
    CHECK(cudaStreamQuery(producer) == cudaErrorNotReady);
    check_gpu_call(cudaStreamSynchronize(consumer));
    int out = 0;
    check_gpu_call(
        cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(out == 1);

    // Same stream: nothing to do. Wrong backend: refused.
    stream_wait(kGPUBackend, device, Stream{consumer}, Stream{consumer});
    CHECK_THROWS_AS(
        stream_wait(Backend::kCPU, device, Stream{consumer}, Stream{producer}),
        std::invalid_argument);

    check_gpu_call(cudaStreamDestroy(consumer));
    check_gpu_call(cudaStreamDestroy(producer));
    check_gpu_call(cudaFree(d_out));
    check_gpu_call(cudaFree(d_flag));
}

} // namespace dmt
