#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    } \
} while (0)

#endif // DEFINITIONS_H