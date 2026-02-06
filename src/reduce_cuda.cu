#include <cuda_runtime.h>

#include <algorithm>

#include "../include/gemm.h"

__global__ void reduce0(float* x, int N, float* partial) {
    // 不可以 s[blockDim.x] ?
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 2 + tid;
    float sum = 0.0;
    if (base < N) sum += x[base];
    if (base + blockDim.x < N) sum += x[base + blockDim.x];

    s[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = s[0];
    }
}

__global__ void reduce1(float* x, int N, float* partial) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    s[tid] = (i < N) ? x[i] : 0.0f;
    __syncthreads();

    for (int st = 1; st < blockDim.x; st *= 2) {
        if (tid % (st * 2) == 0) {
            s[tid] += s[tid + st];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial[blockIdx.x] = s[0];
    }
}

__global__ void reduce_once(float* partial, int N, float* res) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    s[tid] = (tid < N) ? partial[tid] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *res = s[0];
    }
}

float reduce_gpu(const std::vector<float>& A, int N) {
    int threads = 1024;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    size_t shmem = threads * sizeof(float);

    float* partial = nullptr;
    float* dres;
    float res;
    size_t size_A = A.size() * sizeof(float);
    float* d_A = nullptr;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&dres, sizeof(float));
    cudaMalloc(&partial, blocks * sizeof(float));
    cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    int threads2 = 1;
    while (threads2 < blocks) threads2 <<= 1;  // 取到 >= blocks 的 2^k
    threads2 = std::min(threads2, 1024);

    reduce0<<<blocks, threads, shmem>>>(d_A, N, partial);
    reduce_once<<<1, threads2, threads2 * sizeof(float)>>>(partial, blocks, dres);

    cudaMemcpy(&res, dres, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(partial);
    cudaFree(dres);
    return res;
}