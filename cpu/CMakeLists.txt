cmake_minimum_required(VERSION 3.20)

# ── cpu_blas library ──────────────────────────────────────────────────────────
# Each level adds a new source file here as we implement it.
add_library(cpu_blas STATIC
    src/level3/gemm_naive.cpp
    src/level3/gemm_reorder.cpp
    src/level3/gemm_reg.cpp

    # ── Levels 4–8 (uncomment each as you implement it) ───────────────────────
    # src/level3/gemm_1x4.cpp        # L4  — 1×4 micro-kernel
    # src/level3/gemm_block.cpp      # L5  — cache blocking
    # src/level3/gemm_packed.cpp     # L6  — packing
    # src/level3/gemm_neon.cpp       # L7  — NEON 4×4   (needs the NEON flags below)
    # src/level3/gemm_omp.cpp        # L8  — OpenMP     (needs OpenMP, see below)

    # ── Levels 9–14 (advanced track) ──────────────────────────────────────────
    # src/level3/gemm_micro.cpp      # L9  — wide 8×8 / 8×12 micro-kernel (NEON flags)
    # src/level3/gemm_pipe.cpp       # L10 — prefetch + pipelining        (NEON flags)
    # src/level3/gemm_blis.cpp       # L11 — Goto/BLIS 5-loop             (NEON flags)
    # src/level3/gemm_tuned.cpp      # L12 — autotuned dispatch
    # src/level3/gemm_strassen.cpp   # L14 — one-level Strassen
)

target_include_directories(cpu_blas PUBLIC include)

# -O3 lets the compiler do basic scalar optimisations but we disable
# auto-vectorisation so Level 1-5 results reflect our manual work.
# We turn vectorisation back on explicitly from the NEON levels onward.
target_compile_options(cpu_blas PRIVATE
    -O3
    -fno-tree-vectorize
    -march=armv8-a          # M1 baseline without SVE/dotprod extras
)

# ── Per-file NEON flags (Levels 7, 9, 10, 11) ─────────────────────────────────
# The vectorised kernels need NEON + fp16 + dotprod and must NOT be built with
# -fno-tree-vectorize.  Uncomment the matching line when you add each source.
# set(NEON_FLAGS -O3 -march=armv8.2-a+fp16+dotprod)
# set_source_files_properties(src/level3/gemm_neon.cpp  PROPERTIES COMPILE_OPTIONS "${NEON_FLAGS}")
# set_source_files_properties(src/level3/gemm_micro.cpp PROPERTIES COMPILE_OPTIONS "${NEON_FLAGS}")
# set_source_files_properties(src/level3/gemm_pipe.cpp  PROPERTIES COMPILE_OPTIONS "${NEON_FLAGS}")
# set_source_files_properties(src/level3/gemm_blis.cpp  PROPERTIES COMPILE_OPTIONS "${NEON_FLAGS}")

# ── OpenMP (Level 8) ──────────────────────────────────────────────────────────
# find_package(OpenMP REQUIRED)
# target_link_libraries(cpu_blas PUBLIC OpenMP::OpenMP_CXX)

# ── bench_gemm ────────────────────────────────────────────────────────────────
add_executable(bench_gemm benchmark/bench_gemm.cpp)
target_link_libraries(bench_gemm PRIVATE cpu_blas)
target_compile_options(bench_gemm PRIVATE -O2)

# Practical-ceiling baselines (Level 13). Auto-detected; the rows are compiled
# in only when the library is present, and skipped silently otherwise.
find_library(ACCELERATE_FRAMEWORK Accelerate)
if(ACCELERATE_FRAMEWORK)
    message(STATUS "cpu_blas: Accelerate found — enabling cblas_sgemm baseline")
    target_compile_definitions(bench_gemm PRIVATE WGPUBLAS_HAVE_ACCELERATE)
    target_link_libraries(bench_gemm PRIVATE ${ACCELERATE_FRAMEWORK})
endif()

find_package(MLX CONFIG QUIET)
if(MLX_FOUND)
    message(STATUS "cpu_blas: MLX found — enabling mlx::matmul baseline")
    target_compile_definitions(bench_gemm PRIVATE WGPUBLAS_HAVE_MLX)
    target_link_libraries(bench_gemm PRIVATE mlx)
endif()

# ── test_gemm ─────────────────────────────────────────────────────────────────
add_executable(test_gemm test/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE cpu_blas)
target_compile_options(test_gemm PRIVATE -O2)
