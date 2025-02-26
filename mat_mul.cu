#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void matrixMultiplySingleThread(float* A, float* B, float* C, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                float value = 0.0f;
                for (int k = 0; k < N; k++) {
                    value += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = value;
            }
        }
    }
}


__global__
void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main(void) {
    int N = 1 << 20;
    size_t size = N * N * sizeof(float);

    float* A, * B, * C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }


    dim3 blockSize(sqrt(N), sqrt(N));
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    std::cout << blockSize.x << std::endl;
    std::cout << blockSize.y << std::endl;
    std::cout << blockSize.z << std::endl;
    std::cout << gridSize.x << std::endl;
    std::cout << gridSize.y << std::endl;
    std::cout << gridSize.z << std::endl;

    matrixMultiply <<<gridSize, blockSize >>> (A, B, C, N);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N * N; i++) {
        maxError = fmax(maxError, fabs(C[i] - (2.0f * N)));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
