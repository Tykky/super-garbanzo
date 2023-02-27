#include <stdio.h>
#include <math.h>
#include "curand_kernel.h"
#include <cuda.h>

__global__ 
void compute_pi(int samples, double* result, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int inside_circle = 0;

    printf("thread %i launched\n", idx);
    for (size_t i = 0; i < samples; i++)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        if (x * x + y * y <= 1) 
            inside_circle++;
    }

    result[idx] = ((double)inside_circle / (double)samples) * 4;
    printf("thread %i result: %.13f\n", idx, result[idx]);
}

void launch_kernel()
{
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        printf("Cuda capable device not foundn\n");
        getchar();
        return;
    }

    int ndevices;
    cudaGetDeviceCount(&ndevices);

    for (int i = 0; i < ndevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Number of sm: %i\n", prop.multiProcessorCount);
        printf("  Warp size in threads: %i\n", prop.warpSize);
        printf("  Maximum number of threads per block: %i\n", prop.maxThreadsPerBlock);
        printf("  Maximum number of blocks per sm: %i\n", prop.maxBlocksPerMultiProcessor);
        printf("  Maximum number of concurrent kernels: %i\n", prop.concurrentKernels);
        printf("  Maximum number of resident threads: %i\n", prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);
    }

    printf("\n");
    getchar();
    

    long N = 1 << 15;

    double* res;
    cudaMallocManaged(&res, N * sizeof(double));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    compute_pi<<<numBlocks, blockSize>>>(1 << 30, res, 420);
    cudaDeviceSynchronize();

    long double average = 0;

    for (size_t i = 0; i < N; i++)
    {
        average += res[i];
    }

    average /= (double)N;
    cudaFree(&res);

    printf("average: %.13f\n", average);
    getchar();
}