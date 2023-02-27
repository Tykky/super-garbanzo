#include <iostream>
#include <math.h>
#include "curand_kernel.h"

__global__ 
void compute_pi(int samples, float* result, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    curandState state;
    curand_init(seed, idx, 0, &state);

    int inside_circle = 0;

    for (size_t i = 0; i < samples; i++)
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        if (x * x + y * y <= 1) // inside circle
            inside_circle++;
    }

    result[idx] = ((float)inside_circle / (float)samples) * 4;
}

void launch_kernel()
{
    long N = 1 << 28;

    float* res;
    cudaMallocManaged(&res, N * sizeof(float));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    compute_pi<<<numBlocks, blockSize>>>(1 << 25, res, 420);
    cudaDeviceSynchronize();

    long double average = 0;

    for (size_t i = 0; i < N; i++)
    {
        average += res[i];
    }

    average /= (double)N;
    cudaFree(&res);

    char asd[512];

    std::cout << average << "\n";
    std::cin >> asd;
}