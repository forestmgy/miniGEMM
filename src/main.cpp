#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include "../include/gemm.h"

void validate_and_benchmark_reduce(int N, int iters = 10) {
    // ---------- 1. 构造测试数据 ----------
    std::vector<float> A(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        A[i] = dist(rng);
    }

    // ---------- 2. CPU 参考实现 ----------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        cpu_sum = std::accumulate(A.begin(), A.end(), 0.0f);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / iters;

    // ---------- 3. GPU reduce ----------
    // warm-up（避免首次 launch 偏慢）
    reduce_gpu(A, N);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    float gpu_sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        gpu_sum = reduce_gpu(A, N);
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms =
        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count() / iters;

    // ---------- 4. 正确性检查 ----------
    float abs_err = std::abs(cpu_sum - gpu_sum);
    float rel_err = abs_err / (std::abs(cpu_sum) + 1e-6f);

    std::cout << "====== Reduce Validation ======\n";
    std::cout << "N = " << N << "\n";
    std::cout << "CPU sum = " << cpu_sum << "\n";
    std::cout << "GPU sum = " << gpu_sum << "\n";
    std::cout << "abs error = " << abs_err << "\n";
    std::cout << "rel error = " << rel_err << "\n";

    if (rel_err < 1e-5f) {
        std::cout << "[PASS] Result is correct\n";
    } else {
        std::cout << "[FAIL] Result mismatch\n";
    }

    // ---------- 5. 性能输出 ----------
    std::cout << "\n====== Performance ======\n";
    std::cout << "CPU time : " << cpu_ms << " ms\n";
    std::cout << "GPU time : " << gpu_ms << " ms\n";
    std::cout << "Speedup  : " << cpu_ms / gpu_ms << " x\n";
}
// template <typename T>
// double benchmark(const MatrixF32& A, const MatrixF32& B, MatrixF32& C, T func) {
//     const int iters = 2;

//     using clock = std::chrono::high_resolution_clock;
//     double best_s = 1e30;

//     for (int i = 0; i < iters; ++i) {
//         std::fill(C.a.begin(), C.a.end(), 0.0f);
//         auto t0 = clock::now();
//         // std::cout << "start iteration: " << i << "\n";
//         func(A.a, B.a, C.a, A.rows, A.cols, B.cols);
//         auto t1 = clock::now();

//         double s = std::chrono::duration<double>(t1 - t0).count();
//         // std::cout << "finish iteration: " << i << ", time elapsed: " << ms << " ms\n";
//         best_s = std::min(best_s, s);
//     }
//     double gflops = (2.0 * A.rows * A.cols * B.cols) / (best_s * 1e9);
//     return gflops;
// }
int main() {
    // using clock = std::chrono::high_resolution_clock;
    // std::string path_a = "../data/A_f32.bin";
    // std::string path_b = "../data/B_f32.bin";
    // auto A = load_matrix(path_a);
    // auto B = load_matrix(path_b);
    // MatrixF32 C;
    // C.rows = A.rows;
    // C.cols = B.cols;
    // C.a.resize(C.rows * C.cols);

    // std::cout << "start naive GEMM: \n";
    // double flops = benchmark(A, B, C, GEMM_cpu_naive);
    // std::cout << "gflops: " << flops << "\n";

    // std::cout << "start loop interchange GEMM: \n";
    // flops = benchmark(A, B, C, GEMM_cpu_loop_interchange);
    // std::cout << "gflops: " << flops << "\n";

    // std::cout << "start loop unroll: \n";
    // flops = benchmark(A, B, C, GEMM_cpu_loop_unroll);
    // std::cout << "gflops: " << flops << "\n";

    // std::cout << "start loop tiling: \n";
    // flops = benchmark(A, B, C, GEMM_cpu_loop_tiling);
    // std::cout << "gflops: " << flops << "\n";

    // std::cout << "start gpu naive: \n";
    // flops = benchmark(A, B, C, GEMM_GPU_naive);
    // std::cout << "gflops: " << flops << "\n";

    // std::cout << "start gpu tile: \n";
    // flops = benchmark(A, B, C, GEMM_GPU_tile);
    // std::cout << "gflops: " << flops << "\n";

    validate_and_benchmark_reduce(2097150);
}




