cmake_minimum_required(VERSION 3.20)
project(wgpu_blas)

execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create executable
add_executable(wgpu_blas
        main.cpp
)

# Include wgpu headers
target_include_directories(wgpu_blas PRIVATE
        ${CMAKE_SOURCE_DIR}/external/wgpu-native/include
)

# Shared wgpu-native link line (static lib + the macOS frameworks it needs).
set(WGPU_LIBS
        ${CMAKE_SOURCE_DIR}/external/wgpu-native/lib/libwgpu_native.a
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
)

# Link wgpu-native static library
target_link_libraries(wgpu_blas PRIVATE ${WGPU_LIBS})

# ── gpu_gemm: WebGPU GEMM harness (see gpu/JOURNEY_GPU.md) ─────────────────────
# Runs WGSL GEMM shaders against a CPU reference. Default shader path is baked in
# so it runs from any CWD; you can also pass shader files as arguments.
add_executable(gpu_gemm benchmark/gpu_gemm.cpp)
target_include_directories(gpu_gemm PRIVATE
        ${CMAKE_SOURCE_DIR}/external/wgpu-native/include
)
target_compile_definitions(gpu_gemm PRIVATE
        WGPUBLAS_SHADER_DIR="${CMAKE_SOURCE_DIR}/gpu/src/kernels/gemm"
)
target_link_libraries(gpu_gemm PRIVATE ${WGPU_LIBS})

add_subdirectory(cpu)