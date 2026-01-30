#include <chrono>
#include <iostream>
#include <string>

#include "../include/gemm.h"

template <typename T>
double benchmark(const MatrixF32& A, const MatrixF32& B, MatrixF32& C, T func) {
    const int iters = 2;

    using clock = std::chrono::high_resolution_clock;
    double best_s = 1e30;

    for (int i = 0; i < iters; ++i) {
        std::fill(C.a.begin(), C.a.end(), 0.0f);
        auto t0 = clock::now();
        // std::cout << "start iteration: " << i << "\n";
        func(A.a, B.a, C.a, A.rows, A.cols, B.cols);
        auto t1 = clock::now();

        double s = std::chrono::duration<double>(t1 - t0).count();
        // std::cout << "finish iteration: " << i << ", time elapsed: " << ms << " ms\n";
        best_s = std::min(best_s, s);
    }
    double gflops = (2.0 * A.rows * A.cols * B.cols) / (best_s * 1e9);
    return gflops;
}
int main() {
    using clock = std::chrono::high_resolution_clock;
    std::string path_a = "../data/A_f32.bin";
    std::string path_b = "../data/B_f32.bin";
    auto A = load_matrix(path_a);
    auto B = load_matrix(path_b);
    MatrixF32 C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.a.resize(C.rows * C.cols);

    std::cout << "start naive GEMM: \n";
    double flops = benchmark(A, B, C, GEMM_cpu_naive);
    std::cout << "gflops: " << flops << "\n";

    std::cout << "start loop interchange GEMM: \n";
    flops = benchmark(A, B, C, GEMM_cpu_loop_interchange);
    std::cout << "gflops: " << flops << "\n";

    std::cout << "start loop unroll: \n";
    flops = benchmark(A, B, C, GEMM_cpu_loop_unroll);
    std::cout << "gflops: " << flops << "\n";

    std::cout << "start loop tiling: \n";
    flops = benchmark(A, B, C, GEMM_cpu_loop_tiling);
    std::cout << "gflops: " << flops << "\n";

    
}
