#ifndef FIBONACCI_H
#define FIBONACCI_H

#include <cuda_runtime.h>
#include <iostream>

#define MAX_FIB_INDEX 92 //92 is maximum for long long
#define THREADS_PER_BLOCK 32

using LL = long long;

__device__ LL fibonacciDevice(int n);

__global__ void fibonacci(LL* gpu_fib_sequence, int sequence_length);
__global__ void multipleFibonacciKernels(LL* gpu_fib_sequence, int sequence_length, int N);

void executeFibonacciKernel(std::string name, int n);

#endif // FIBONACCI_H
