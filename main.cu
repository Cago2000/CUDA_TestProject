#include <iostream>
#include <cmath>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "mat_mul.cuh"

int main(void) {
    cudaFree(0);
    warmupKernel <<<1, 1>>> ();
    cudaDeviceSynchronize();

    std::vector<KernelInfo> kernel_config_vec = {
        {"Matrix Multiplication Optimized #1", matrixMultiplyOptimized, 1 << 11, ((int)1 << 11) * 2.0f},
        {"Matrix Multiplication Optimized #2", matrixMultiplyOptimized, 1 << 11, ((int)1 << 11) * 2.0f},
        {"Matrix Multiplication Optimized #3", matrixMultiplyOptimized, 1 << 11, ((int)1 << 11) * 2.0f},
        {"Matrix Multiplication #1", matrixMultiply, 1 << 11, ((int)1 << 11) * 2.0f},
        {"Matrix Add #1", matrixAdd, 1 << 11, 3.0f},
    };

    for (const auto& kernel : kernel_config_vec) {
        float* A, * B, * C;
        allocateAndInitializeMatrix(&A, &B, &C, kernel.N);
        executeMatMulKernel({ A, B, C, kernel.name, kernel.kernel, kernel.N, kernel.expected_value});

        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }
    return 0;
}
