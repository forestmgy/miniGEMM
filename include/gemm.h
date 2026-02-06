#include <cstdint>
#include <string>
#include <vector>

struct MatrixF32 {
    uint64_t rows = 0, cols = 0;
    std::vector<float> a;  // row-major: a[i*cols + j]
};
MatrixF32 load_matrix(const std::string& path);

void GEMM_cpu_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t, size_t m,
                    size_t p);
void GEMM_cpu_loop_interchange(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                               size_t n, size_t m, size_t p);

void GEMM_cpu_loop_unroll(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n,
                          size_t m, size_t p);
void GEMM_cpu_loop_tiling(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n,
                          size_t m, size_t p);

void GEMM_GPU_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n, size_t m,
                    size_t p);

void GEMM_GPU_tile(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n, int m, int p);

float reduce_cpu_sum(const float* x, int N);

float reduce_gpu(const std::vector<float>& A, int N);