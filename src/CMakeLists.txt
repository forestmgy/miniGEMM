add_library(MatrixLib STATIC 
    gemm_cpu.cpp 
    gen_mats.cpp
    gemm_cuda.cu
)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(MatrixLib PUBLIC CUDA::cudart)
add_executable(cpu_bench main.cpp)

target_link_libraries(cpu_bench PRIVATE MatrixLib)
target_include_directories(cpu_bench PRIVATE include)