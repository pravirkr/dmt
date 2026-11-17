#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/fourier_gpu.cuh"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_planned_vector.cuh"
#include "dmt/gpu_utils.cuh"

#include "test_helpers.hpp"

namespace dmt {

namespace {

std::vector<float> download(const fourier_gpu::DevBuf<float>& b, SizeType n) {
    std::vector<float> h(n);
    gpu_utils::check_gpu_call(cudaMemcpy(h.data(), b.data(), n * sizeof(float),
                                         cudaMemcpyDeviceToHost));
    return h;
}

} // namespace

TEST_CASE("DevBuf replays deferred calls as made at once", "[planned][gpu]") {
    if (!test::gpu_device_available()) {
        SKIP("no GPU device");
    }
    const std::vector<float> host{1.0F, 2.0F, 3.0F};
    SECTION("upload, then reserve_zeroed: the zeroed part is zero") {
        fourier_gpu::DevBuf<float> eager;
        eager.upload(host);
        eager.reserve_zeroed(2);
        gpu_utils::PendingList list;
        fourier_gpu::DevBuf<float> deferred;
        {
            const gpu_utils::DeferScope scope(&list);
            deferred.upload(host);
            deferred.reserve_zeroed(2);
            CHECK(deferred.pending());
            CHECK(deferred.capacity() == 3);
        }
        list.materialize();
        const auto a = download(eager, 2);
        const auto b = download(deferred, 2);
        CHECK(a == std::vector<float>(2, 0.0F));
        CHECK(b == a);
    }
    SECTION("reserve_zeroed, then upload: the upload") {
        gpu_utils::PendingList list;
        fourier_gpu::DevBuf<float> deferred;
        {
            const gpu_utils::DeferScope scope(&list);
            deferred.reserve_zeroed(3);
            deferred.upload(host);
        }
        list.materialize();
        CHECK(download(deferred, 3) == host);
    }
}

} // namespace dmt
