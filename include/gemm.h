#include <cstdint>
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