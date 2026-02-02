#include <cuda_runtime.h>

#include "../include/gemm.h"

#define THREAD 16


__global__ void gemm_naive_kernel(float* A, float* B, float* C, size_t n, size_t m, size_t p) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < p && row < n) {
        for (size_t i = 0; i < m; i++) {
            C[row * p + col] += A[row * m + i] * B[i * p + col];
        }
    }
}


void GEMM_GPU_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n, size_t m,
                    size_t p) {
    dim3 block(THREAD, THREAD);
    dim3 grid((p + THREAD - 1) / THREAD, (n + THREAD - 1) / THREAD);
    size_t size_A = A.size() * sizeof(float);
    size_t size_B = B.size() * sizeof(float);
    size_t size_C = C.size() * sizeof(float);
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), size_C, cudaMemcpyHostToDevice);
    gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, n, m, p);

    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#define TILE 16

__global__ void gemm_tile_kernel(float* A, float* B, float* C, int n, int m, int p) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float sum = 0.0f;
    for (int t = 0; t < (m + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < n && a_col < m) ? A[row * m + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < m && col < p) ? B[b_row * p + col] : 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (col < p && row < n) {
        C[row * p + col] = sum;
    }
}

void GEMM_GPU_tile(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int m,
                    int p) {
    dim3 block(TILE, TILE);
    dim3 grid((p + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    size_t size_A = A.size() * sizeof(float);
    size_t size_B = B.size() * sizeof(float);
    size_t size_C = C.size() * sizeof(float);
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), size_C, cudaMemcpyHostToDevice);
    gemm_tile_kernel<<<grid, block>>>(d_A, d_B, d_C, n, m, p);

    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
