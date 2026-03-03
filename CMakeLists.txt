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

# Link wgpu-native static library
target_link_libraries(wgpu_blas PRIVATE
        ${CMAKE_SOURCE_DIR}/external/wgpu-native/lib/libwgpu_native.a
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
)