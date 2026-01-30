#include <iostream>
#include <vector>

void GEMM_cpu_loop_interchange(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                               size_t n, size_t m, size_t p) {
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            for (size_t k = 0; k < p; k++) {
                C[i * p + k] += A[i * m + j] * B[j * p + k];
            }
        }
    }
}

void GEMM_cpu_naive(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n, size_t m,
                    size_t p) {
    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < p; k++) {
            for (size_t j = 0; j < m; j++) {
                C[i * p + k] += A[i * m + j] * B[j * p + k];
            }
        }
    }
}

void GEMM_cpu_loop_unroll(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n,
                          size_t m, size_t p) {
    size_t k4 = p - p % 4;
    for (size_t i = 0; i < n; i++) {
        size_t ci = i * p;
        size_t ai = i * m;
        for (size_t j = 0; j < m; j++) {
            size_t k = 0;
            float a = A[ai + j];
            size_t bj = j * p;
            for (; k < k4; k += 4) {
                C[ci + k] += a * B[bj + k];
                C[ci + k + 1] += a * B[bj + k + 1];
                C[ci + k + 2] += a * B[bj + k + 2];
                C[ci + k + 3] += a * B[bj + k + 3];
            }
            for (; k < p; k++) {
                C[ci + k] += a * B[bj + k];
            }
        }
    }
}

void GEMM_cpu_loop_tiling(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, size_t n,
                          size_t m, size_t p) {
    size_t block = 32;
    for (size_t i = 0; i < n; i += block) {
        for (size_t j = 0; j < m; j += block) {
            for (size_t k = 0; k < p; k += block) {
                size_t ii = std::min(i + block, n);
                size_t jj = std::min(j + block, m);
                size_t kk = std::min(k + block, p);
                for (size_t i0 = i; i0 < ii; i0++) {
                    for (size_t j0 = j; j0 < jj; j0++) {
                        float a = A[i0 * m + j0];
                        for (size_t k0 = k; k0 < kk; k0++) {
                            C[i0 * p + k0] += a * B[j0 * p + k0];
                        }
                    }
                }
            }
        }
    }
}