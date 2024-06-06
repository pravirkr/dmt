#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace error_checker {

inline void
check_cuda_error(const char* file, int line, const std::string& msg = "") {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::stringstream error_msg;
        error_msg << "CUDA failed with error: " << cudaGetErrorString(error)
                  << " (" << file << ":" << line << ")";
        if (!msg.empty()) {
            error_msg << " - " << msg;
        }
        throw std::runtime_error(error_msg.str());
    }
}

inline void
check_cuda_error_sync(const char* file, int line, const std::string& msg = "") {
    cudaDeviceSynchronize();
    check_cuda_error(file, line, msg);
}

void check_cuda(const std::string& msg = "") {
    check_cuda_error(__FILE__, __LINE__, msg);
}

void check_cuda_sync(const std::string& msg = "") {
    check_cuda_error_sync(__FILE__, __LINE__, msg);
}

} // namespace error_checker
