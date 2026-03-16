#include <cuda_runtime.h>

#include "../include/gemm.h"

#define THREAD 16
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define TILE 16
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

__global__ void gemm_tile_kernel1(float* A, float* B, float* C, int m, int n, int k) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int acol = t * TILE + threadIdx.x;
        int brow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && acol < n) ? A[row * n + acol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (brow < n && col < k) ? B[brow * k + col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < k) {
        C[row * k + col] = sum;
    }
}

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

#define VEC 4

__device__ __forceinline__ bool is_aligned_16(const void* ptr) { return ((uintptr_t)ptr & 0xF) == 0; }

__global__ void gemm_tile_vec4_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                      int n, int m, int p) {
    int col = blockIdx.x * TILE + threadIdx.x;  // C col
    int row = blockIdx.y * TILE + threadIdx.y;  // C row

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;

    // 每一轮处理 K 方向 TILE
    for (int t = 0; t < (m + TILE - 1) / TILE; t++) {
        int a_col0 = t * TILE;  // 这一 tile 的 A 起始 col
        int b_row = t * TILE + threadIdx.y;

        // ========= A: global -> shared (vectorized by float4) =========
        // 每行用 x=0..3 四个线程加载 16 个元素
        if (threadIdx.x < TILE / VEC) {              // x < 4
            int a_col = a_col0 + threadIdx.x * VEC;  // 每线程负责 4 个连续元素

            // shared 写入位置
            int sx = threadIdx.x * VEC;
            int sy = threadIdx.y;

            // 边界：row < n；a_col + 0..3 < m
            const float* a_ptr = A + row * m + a_col;

            if (row < n && (a_col + (VEC - 1)) < m && is_aligned_16(a_ptr)) {
                float4 v = *reinterpret_cast<const float4*>(a_ptr);
                As[sy][sx + 0] = v.x;
                As[sy][sx + 1] = v.y;
                As[sy][sx + 2] = v.z;
                As[sy][sx + 3] = v.w;
            } else {
                // fallback：处理非对齐 / 边界
                As[sy][sx + 0] = (row < n && (a_col + 0) < m) ? a_ptr[0] : 0.0f;
                As[sy][sx + 1] = (row < n && (a_col + 1) < m) ? a_ptr[1] : 0.0f;
                As[sy][sx + 2] = (row < n && (a_col + 2) < m) ? a_ptr[2] : 0.0f;
                As[sy][sx + 3] = (row < n && (a_col + 3) < m) ? a_ptr[3] : 0.0f;
            }
        }

        // ========= B: global -> shared (vectorized by float4) =========
        // B 的 tile：行是 b_row，列是 blockIdx.x*TILE..+15
        // 仍然每行用 x=0..3 四个线程加载 16 个元素
        if (threadIdx.x < TILE / VEC) {
            int b_col = blockIdx.x * TILE + threadIdx.x * VEC;  // 每线程负责 4 个 col
            int sx = threadIdx.x * VEC;
            int sy = threadIdx.y;

            const float* b_ptr = B + b_row * p + b_col;

            if (b_row < m && (b_col + (VEC - 1)) < p && is_aligned_16(b_ptr)) {
                float4 v = *reinterpret_cast<const float4*>(b_ptr);
                Bs[sy][sx + 0] = v.x;
                Bs[sy][sx + 1] = v.y;
                Bs[sy][sx + 2] = v.z;
                Bs[sy][sx + 3] = v.w;
            } else {
                Bs[sy][sx + 0] = (b_row < m && (b_col + 0) < p) ? b_ptr[0] : 0.0f;
                Bs[sy][sx + 1] = (b_row < m && (b_col + 1) < p) ? b_ptr[1] : 0.0f;
                Bs[sy][sx + 2] = (b_row < m && (b_col + 2) < p) ? b_ptr[2] : 0.0f;
                Bs[sy][sx + 3] = (b_row < m && (b_col + 3) < p) ? b_ptr[3] : 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < p) {
        C[row * p + col] = sum;
    }
}

void GEMM_GPU_tile_vec4(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int m,
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
    gemm_tile_vec4_kernel<<<grid, block>>>(d_A, d_B, d_C, n, m, p);

    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void sgemm_V1(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int N,
                         const int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;         // tid/2, row of s_a
    int load_a_smem_k = (tid & 1) << 2;   // (tid % 2 == 0) ? 0 : 4, col of s_a
    int load_b_smem_k = tid >> 5;         // tid/32, row of s_b
    int load_b_smem_n = (tid & 31) << 2;  // (tid % 32) * 4, col of s_b

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b

    for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
        int load_a_gmem_k = bk * BK + load_a_smem_k;  // global col of a
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        int load_b_gmem_k = bk * BK + load_b_smem_k;  // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    int comp_a_smem_m = ty * TM + m;
                    int comp_b_smem_n = tx * TN + n;
                    r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
        int store_c_gmem_m = by * BM + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_c_gmem_n = bx * BN + tx * TN + j;
            int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
            FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
        }
    }
}

void GEMM_GPU_v1(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
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
    sgemm_V1<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void sgemm_V3(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, const int M, const int N,
                         const int K) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

#pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++) {
#pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++) {
#pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

void GEMM_GPU_v3(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
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
    sgemm_V3<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}