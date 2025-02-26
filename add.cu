#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main2(void)
{
    int N = 1 << 20;
    float* x, * y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << "Block Size: " << blockSize << std::endl;
    std::cout << "Number Of Blocks: " << numBlocks << std::endl;
    add<<<numBlocks, blockSize >>> (N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}