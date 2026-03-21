#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);

void testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(float *a, float *b, float *c, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

__global__ void sgemm_V3(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

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
    float r_c[TM][TN] = {0.0f};

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

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
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
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
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

int main(void) {
    printf("\nKernel = sgemm_V3\n");

    const int BM = 128, BN = 128, TM = 8, TN = 8;
    const int M = 4096, N = 4096, K = 4096;
    const int repeat = 100;

    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int) = sgemm_V3;

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
    printf("Max Error = %f\n", max_error);

    testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, repeat);

    return 0;
}

float testError(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)K * N * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *h_d_c;
    float *d_a, *d_b, *d_c;

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    h_d_c = (float *)malloc(size_c);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    srand((unsigned)time(0));
    for (int i = 0; i < M * K; i++) h_a[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = rand() / (float)RAND_MAX;
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float this_error = fabsf(h_d_c[i] - h_c[i]);
        if (isnan(this_error)) {
            max_error = NAN;
            break;
        }
        if (this_error > max_error) {
            max_error = this_error;
        }
    }

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return max_error;
}

void testPerformance(
    void (*gpuSgemm)(float *, float *, float *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = (size_t)M * K * sizeof(float);
    size_t size_b = (size_t)K * N * sizeof(float);
    size_t size_c = (size_t)M * N * sizeof(float);

    float *h_a, *h_b;
    float *d_a, *d_b, *d_c;

    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    srand(123);
    for (int i = 0; i < M * K; i++) h_a[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = rand() / (float)RAND_MAX;

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, size_c);

    // warmup
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / (avg_ms / 1000.0) / 1e9;
    double tflops = gflops / 1000.0;

    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("Total FLOPs per GEMM = %.0f\n", total_flops);
    printf("Average Time = %.6f ms\n", avg_ms);
    printf("Performance = %.3f GFLOPS (%.6f TFLOPS)\n", gflops, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}