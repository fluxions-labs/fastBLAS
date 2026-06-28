#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// Convention (matches CBLAS row-major):
//   C = alpha * A * B + beta * C
//   A : M×K  (row-major, leading dimension lda >= K)
//   B : K×N  (row-major, leading dimension ldb >= N)
//   C : M×N  (row-major, leading dimension ldc >= N)
// ─────────────────────────────────────────────────────────────────────────────

// Level 1 — naive triple loop, i-j-k order
void sgemm_naive(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);

// Level 2 — loop reorder: i-k-j  

void sgemm_reorder(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);

// Level 3 — register accumulation  

void sgemm_reg(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);


// ── Levels 4–8 (declared now, implemented by you as you reach each level) ─────
// Every kernel shares the same signature as sgemm_naive above.
#define SGEMM_SIG(name)                                                        \
  void name(int M, int N, int K, float alpha, const float *A, int lda,         \
            const float *B, int ldb, float beta, float *C, int ldc)

SGEMM_SIG(sgemm_1x4);    // Level 4 — 1×4 micro-kernel
SGEMM_SIG(sgemm_block);  // Level 5 — cache blocking / tiling
SGEMM_SIG(sgemm_packed); // Level 6 — packing A and B panels
SGEMM_SIG(sgemm_neon);   // Level 7 — NEON 4×4 SIMD micro-kernel
SGEMM_SIG(sgemm_omp);    // Level 8 — OpenMP multi-threading

// ── Levels 9–14: advanced HPC track (see JOURNEY.md) ──────────────────────────
SGEMM_SIG(sgemm_micro);    // Level 9  — wide register-blocked micro-kernel (8×8 / 8×12)
SGEMM_SIG(sgemm_pipe);     // Level 10 — software pipelining + prefetch
SGEMM_SIG(sgemm_blis);     // Level 11 — multi-level cache blocking (Goto/BLIS 5-loop)
SGEMM_SIG(sgemm_tuned);    // Level 12 — autotuned block/micro-kernel config
SGEMM_SIG(sgemm_strassen); // Level 14 — one-level Strassen (optional)

// The Level 13 roofline baselines (Apple Accelerate, MLX) are not declared here —
// they live as bench-local wrappers in benchmark/bench_gemm.cpp, guarded by the
// WGPUBLAS_HAVE_* macros that CMake sets when the library is found.
