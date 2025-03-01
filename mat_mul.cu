#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
} while (0)

__global__
void matrixAdd(float* A, float* B, float* C, int N, int* jobs_done) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
        atomicAdd(jobs_done, 1);
    }
}

__global__
void matrixMultiply(float* A, float* B, float* C, int N, int* jobs_done) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
        atomicAdd(jobs_done, 1);
    }
}

void allocateMatrixMemory(float** A, float** B, float** C, int N) {
    size_t size = N * N * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(A, size));
    CUDA_CHECK(cudaMallocManaged(B, size));
    CUDA_CHECK(cudaMallocManaged(C, size));
}

void allocateJobsDone(int** jobs_done) {
    cudaMallocManaged(jobs_done, sizeof(int));
    cudaMemset(*jobs_done, 0, sizeof(int));
}


void initializeMatrix(float* A, float* B, int N) {
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }
}

void launchMatrixAddKernel(float* A, float* B, float* C, int N, int* jobs_done, dim3 blockSize, dim3 gridSize, cudaStream_t stream) {
    matrixAdd << <gridSize, blockSize, 0, stream >>> (A, B, C, N, jobs_done);
    CUDA_CHECK(cudaGetLastError());
}

void launchMatrixMultiplyKernel(float* A, float* B, float* C, int N, int* jobs_done, dim3 blockSize, dim3 gridSize, cudaStream_t stream) {
    matrixMultiply << <gridSize, blockSize, 0, stream >> > (A, B, C, N, jobs_done);
    CUDA_CHECK(cudaGetLastError());
}

float calculateMaxError(float* C, int N, float expected_value) {
    float maxError = 0.0f;
    for (int i = 0; i < N * N; i++) {
        maxError = fmax(maxError, fabs(C[i] - expected_value));
    }
    return maxError;
}

int main(void) {
    int N_mat_mul = 1 << 8;
    int N_mat_add = 1 << 13;

    float* A_mat_mul, * B_mat_mul, * A_mat_add, * B_mat_add;
    float* C_mat_mul, * C_mat_add;
    allocateMatrixMemory(&A_mat_mul, &B_mat_mul, &C_mat_mul, N_mat_mul);
    allocateMatrixMemory(&A_mat_add, &B_mat_add, &C_mat_add, N_mat_add);

    int* jobs_done_mat_mul;
    int* jobs_done_mat_add;
    allocateJobsDone(&jobs_done_mat_mul);
    allocateJobsDone(&jobs_done_mat_add);

    initializeMatrix(A_mat_mul, B_mat_mul, N_mat_mul);
    initializeMatrix(A_mat_add, B_mat_add, N_mat_add);

    dim3 blockSize(32, 32);
    dim3 gridSize_mat_mul((N_mat_mul + blockSize.x - 1) / blockSize.x, (N_mat_mul + blockSize.y - 1) / blockSize.y);
    dim3 gridSize_mat_add((N_mat_add + blockSize.x - 1) / blockSize.x, (N_mat_add + blockSize.y - 1) / blockSize.y);

    cudaStream_t stream_mat_mul, stream_mat_add;
    CUDA_CHECK(cudaStreamCreate(&stream_mat_mul));
    CUDA_CHECK(cudaStreamCreate(&stream_mat_add));

    launchMatrixMultiplyKernel(A_mat_mul, B_mat_mul, C_mat_mul, N_mat_mul, jobs_done_mat_mul ,blockSize, gridSize_mat_mul, stream_mat_mul);
    launchMatrixAddKernel(A_mat_add, B_mat_add, C_mat_add, N_mat_add, jobs_done_mat_add, blockSize, gridSize_mat_add, stream_mat_add);

    cudaStreamSynchronize(stream_mat_mul);
    cudaStreamSynchronize(stream_mat_add);

    float maxError_mat_mul = calculateMaxError(C_mat_mul, N_mat_mul, 2.0f * N_mat_mul);
    float maxError_mat_add = calculateMaxError(C_mat_add, N_mat_add, 3.0f);

    std::cout << "Matrix Multiply, Jobs done: " << *jobs_done_mat_mul << " out of " << N_mat_mul * N_mat_mul << ", Max Error: " << maxError_mat_mul << std::endl;
    std::cout << "Matrix Add, Jobs done: " << *jobs_done_mat_add << " out of " << N_mat_add * N_mat_add << ", Max Error: " << maxError_mat_add << std::endl;

    cudaStreamDestroy(stream_mat_mul);
    cudaStreamDestroy(stream_mat_add);
    cudaFree(A_mat_mul);
    cudaFree(B_mat_mul);
    cudaFree(A_mat_add);
    cudaFree(B_mat_add);
    cudaFree(C_mat_mul);
    cudaFree(C_mat_add);
    cudaFree(jobs_done_mat_mul);
    cudaFree(jobs_done_mat_add);
    return 0;
}