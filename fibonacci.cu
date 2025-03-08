#include "fibonacci.cuh"
#include "definitions.cuh"

__device__ LL fibonacciDevice(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    LL prev = 0, curr = 1, next_fib;
    for (int i = 2; i <= n; i++) {
        next_fib = prev + curr;
        prev = curr;
        curr = next_fib;
    }
    return curr;
}

__global__ void fibonacci(LL* gpu_fib_sequence, int sequence_length) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= sequence_length) return;
    gpu_fib_sequence[thread_id] = fibonacciDevice(thread_id);
}

__global__ void multipleFibonacciKernels(LL* gpu_fib_sequence, int sequence_length, int N) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= sequence_length) return;
    for (int i = 0; i < N; i++) {
        gpu_fib_sequence[thread_id] = fibonacciDevice(thread_id); 
    }
}

void executeFibonacciKernel(std::string name, int n) {
    const int sequence_length = MAX_FIB_INDEX + 1;
    LL* d_fib_sequence;
    LL* h_fib_sequence = new LL[sequence_length];

    CUDA_CHECK(cudaMalloc(&d_fib_sequence, sequence_length * sizeof(LL)));

    int blocksPerGrid = (sequence_length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int threadsPerBlock = THREADS_PER_BLOCK;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));

    multipleFibonacciKernels << <blocksPerGrid, threadsPerBlock >> > (d_fib_sequence, sequence_length, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaMemcpy(h_fib_sequence, d_fib_sequence, sequence_length * sizeof(LL), cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << name << " | Execution Time: " << milliseconds << " ms | Latest Fibonacci Number: " << h_fib_sequence[MAX_FIB_INDEX] << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_fib_sequence));
    delete[] h_fib_sequence;
}
