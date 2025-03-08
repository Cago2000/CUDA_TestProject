#ifndef MAT_MUL_H
#define MAT_MUL_H

#include <iostream>
#include <cmath>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_SIZE 32

struct Kernel {
    float* A, * B, * C;
    std::string name;
    void (*kernel)(float*, float*, float*, int);
    int N;
    float expected_value;
};

struct KernelInfo {
    std::string name;
    void (*kernel)(float*, float*, float*, int);
    int N;
    float expected_value;
};

__global__ void matrixAdd(float* A, float* B, float* C, int N);
__global__ void matrixMultiply(float* A, float* B, float* C, int N);
__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int N);
__global__ void warmupKernel();


void allocateAndInitializeMatrix(float** A, float** B, float** C, int N);
float calculateMaxError(float* C, int N, float expected_value);
void executeMatMulKernel(const Kernel& data);

#endif // MAT_MUL_H