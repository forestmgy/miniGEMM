#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "../include/gemm.h"

// g++ -O2 -std=c++17 gen_mats.cpp -o gen_mats
static constexpr uint32_t kMagic = 0x4D4D4154; // 'TMAM' 随便一个标记

void save_matrix(const std::string& path, const MatrixF32& m) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("failed to open for write: " + path);

    uint32_t magic = kMagic;
    uint32_t dtype = 1; // 1 = float32 (自定义)
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    ofs.write(reinterpret_cast<const char*>(&m.rows), sizeof(m.rows));
    ofs.write(reinterpret_cast<const char*>(&m.cols), sizeof(m.cols));
    ofs.write(reinterpret_cast<const char*>(m.a.data()), m.a.size() * sizeof(float));
}

MatrixF32 load_matrix(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("failed to open for read: " + path);

    uint32_t magic = 0, dtype = 0;
    uint64_t rows = 0, cols = 0;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    if (magic != kMagic) throw std::runtime_error("bad magic in: " + path);
    if (dtype != 1) throw std::runtime_error("unsupported dtype in: " + path);

    MatrixF32 m;
    m.rows = rows; m.cols = cols;
    m.a.resize(rows * cols);
    ifs.read(reinterpret_cast<char*>(m.a.data()), m.a.size() * sizeof(float));
    return m;
}

MatrixF32 make_random_matrix(uint64_t rows, uint64_t cols, uint64_t seed) {
    MatrixF32 m;
    m.rows = rows; m.cols = cols;
    m.a.resize(rows * cols);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f); // 更像 DL 初始化
    for (auto &x : m.a) x = dist(rng);
    return m;
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// int main() {
//     // 选一档尺寸：比如中等档 4096^3
//     const uint64_t M = 2048, K = 2048, N = 2048;

//     const std::string A_path = "../data/A_f32.bin";
//     const std::string B_path = "../data/B_f32.bin";

//     if (!file_exists(A_path)) {
//         auto A = make_random_matrix(M, K, 12345);
//         save_matrix(A_path, A);
//         std::cout << "Saved " << A_path << "\n";
//     }
//     if (!file_exists(B_path)) {
//         auto B = make_random_matrix(K, N, 67890);
//         save_matrix(B_path, B);
//         std::cout << "Saved " << B_path << "\n";
//     }

//     auto A = load_matrix(A_path);
//     auto B = load_matrix(B_path);
//     std::cout << "Loaded A: " << A.rows << "x" << A.cols
//               << ", B: " << B.rows << "x" << B.cols << "\n";

//     // TODO: 在这里调用你的 matmul(A, B) 做测试
//     return 0;
// }
