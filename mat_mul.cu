#include "mat_mul.cuh"
#include "definitions.cuh"

__global__
void matrixAdd(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
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

__global__
void matrixMultiplyOptimized(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < N && i * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + i * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && i * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

__global__ void warmupKernel(){}

void allocateAndInitializeMatrix(float** A, float** B, float** C, int N) {
    size_t size = N * N * sizeof(float);
    CUDA_CHECK(cudaMallocManaged(A, size));
    CUDA_CHECK(cudaMallocManaged(B, size));
    CUDA_CHECK(cudaMallocManaged(C, size));

    for (int i = 0; i < N * N; i++) {
        (*A)[i] = 1.0f;
        (*B)[i] = 2.0f;
    }
}

float calculateMaxError(float* C, int N, float expected_value) {
    float maxError = 0.0f;
    for (int i = 0; i < N * N; i++) {
        maxError = fmax(maxError, fabs(C[i] - expected_value));
    }
    return maxError;
}

void executeMatMulKernel(const Kernel& data) {
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((data.N + blockSize.x - 1) / blockSize.x, (data.N + blockSize.y - 1) / blockSize.y);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    data.kernel<<<gridSize, blockSize, 0, stream >>>(data.A, data.B, data.C, data.N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::cout << data.name << " | Execution Time: " << milliseconds << " ms | Max Error: "
        << calculateMaxError(data.C, data.N, data.expected_value) << std::endl;
}