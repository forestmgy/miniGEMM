#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

float testCublasErrorAndFlops(const int M, const int N, const int K, const int repeat);

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

int main(void) {
    printf("\nKernel = cublas\n");

    const int M = 4096, N = 4096, K = 4096;
    const int repeat = 100;

    float max_error = testCublasErrorAndFlops(M, N, K, repeat);
    printf("Max Error = %f\n", max_error);

    return 0;
}

float testCublasErrorAndFlops(const int M, const int N, const int K, const int repeat) {
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

    // CPU reference
    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // warmup
    cublasSgemm(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_b, N,
        d_a, K,
        &beta,
        d_c, N
    );
    cudaDeviceSynchronize();

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_b, N,
            d_a, K,
            &beta,
            d_c, N
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;

    // copy result back
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    // check error
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

    // FLOPs = 2 * M * N * K
    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / (avg_ms / 1000.0) / 1e9;
    double tflops = gflops / 1000.0;

    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("Total FLOPs per GEMM = %.0f\n", total_flops);
    printf("Average Time = %.6f ms\n", avg_ms);
    printf("Performance = %.3f GFLOPS (%.6f TFLOPS)\n", gflops, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(cublas_handle);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return max_error;
}