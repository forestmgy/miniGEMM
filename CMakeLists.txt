cmake_minimum_required(VERSION 3.10)
project(MatrixProject LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)
# 告诉 CMake 去 src 文件夹里找另一个 CMakeLists.txt
add_subdirectory(src)