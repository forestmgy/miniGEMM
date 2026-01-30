add_library(MatrixLib STATIC 
    gemm_cpu.cpp 
    gen_mats.cpp
)

add_executable(cpu_bench main.cpp)

target_link_libraries(cpu_bench PRIVATE MatrixLib)
target_include_directories(cpu_bench PRIVATE include)