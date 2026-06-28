#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include "blas.h"

// ── Practical-ceiling baselines (Level 13) ────────────────────────────────────
// Compiled in only when CMake found the library (WGPUBLAS_HAVE_* defined).
#ifdef WGPUBLAS_HAVE_ACCELERATE
#define ACCELERATE_NEW_LAPACK   // use the current cblas headers, silence deprecation
#include <Accelerate/Accelerate.h>
static void sgemm_accelerate(int M, int N, int K, float alpha,
                             const float* A, int lda, const float* B, int ldb,
                             float beta, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

#ifdef WGPUBLAS_HAVE_MLX
#include <mlx/mlx.h>
// Baseline only: computes alpha*A*B (ignores beta*C — the bench runs with beta=0).
static void sgemm_mlx(int M, int N, int K, float alpha,
                      const float* A, int /*lda*/, const float* B, int /*ldb*/,
                      float /*beta*/, float* C, int /*ldc*/) {
    namespace mx = mlx::core;
    auto a = mx::array(A, {M, K}, mx::float32);
    auto b = mx::array(B, {K, N}, mx::float32);
    auto c = mx::matmul(a, b);
    if (alpha != 1.0f) c = mx::multiply(c, mx::array(alpha));
    mx::eval(c);
    std::memcpy(C, c.data<float>(), sizeof(float) * M * N);
}
#endif

using Clock = std::chrono::high_resolution_clock;

// Returns the best (minimum) wall-clock time over `reps` runs of fn().
template<typename Fn>
static double best_of(Fn fn, int reps = 5) {
    fn(); // warm-up: fills caches, JIT-compiles any lazy init
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
}

// GFLOP count for C = alpha*A*B + beta*C  (M×N×K matrices)
static double gflops_gemm(int M, int N, int K) {
    // 2*M*N*K multiply-adds  +  M*N for the alpha/beta scaling
    return (2.0 * M * N * K + 2.0 * M * N) / 1e9;
}

struct Kernel {
    const char* name;
    void (*fn)(int,int,int, float, const float*,int, const float*,int, float, float*,int);
};

// One measured data point, collected so we can also emit a CSV block at the end.
struct Result {
    std::string name;
    int M, N, K;
    double gflops, secs;
};

static void run_suite(const std::vector<Kernel>& kernels, int M, int N, int K,
                      std::vector<Result>& sink) {
    // Allocate and randomise matrices once; reuse across kernels.
    std::vector<float> A(M * K), B(K * N), C(M * N);
    for (auto& v : A) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : B) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : C) v = static_cast<float>(rand()) / RAND_MAX;

    std::vector<float> C_save = C; // restore C between kernels so beta*C is consistent

    for (const auto& k : kernels) {
        std::copy(C_save.begin(), C_save.end(), C.begin());
        double secs = best_of([&]{
            k.fn(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
        });
        double gf = gflops_gemm(M, N, K) / secs;
        printf("  %-28s  %4d×%4d×%4d  %7.3f GFLOPS  (%6.3f s)\n",
               k.name, M, N, K, gf, secs);
        sink.push_back({k.name, M, N, K, gf, secs});
    }
}

int main() {
    // Register every implemented kernel here.  Uncomment a row as you reach each
    // level (and uncomment the matching source in cpu/CMakeLists.txt).
    std::vector<Kernel> kernels = {
        { "sgemm_naive   (L1)",  sgemm_naive   },
        { "sgemm_reorder (L2)",  sgemm_reorder },
        { "sgemm_reg     (L3)",  sgemm_reg     },
        // { "sgemm_1x4     (L4)",  sgemm_1x4      },
        // { "sgemm_block   (L5)",  sgemm_block    },
        // { "sgemm_packed  (L6)",  sgemm_packed   },
        // { "sgemm_neon    (L7)",  sgemm_neon     },
        // { "sgemm_omp     (L8)",  sgemm_omp      },
        // { "sgemm_micro   (L9)",  sgemm_micro    },
        // { "sgemm_pipe    (L10)", sgemm_pipe     },
        // { "sgemm_blis    (L11)", sgemm_blis     },
        // { "sgemm_tuned   (L12)", sgemm_tuned    },
        // { "sgemm_strassen(L14)", sgemm_strassen },
    };

    // Practical-ceiling baselines (Level 13). Auto-appended when available.
#ifdef WGPUBLAS_HAVE_ACCELERATE
    kernels.push_back({ "Accelerate (cblas, AMX)", sgemm_accelerate });
#endif
#ifdef WGPUBLAS_HAVE_MLX
    kernels.push_back({ "MLX matmul (Metal/AMX)", sgemm_mlx });
#endif

    printf("%-30s  %14s  %10s  %8s\n", "kernel", "size (M×N×K)", "GFLOPS", "time");
    printf("%s\n", std::string(70, '-').c_str());

    std::vector<Result> results;

    // Sizes for the early levels. When you reach the advanced track (L9+),
    // add 2048 and 4096 here and drop sgemm_naive from the list above — a
    // 4096³ naive run takes ~80 s.
    for (int sz : {256, 512, 1024}) {
        run_suite(kernels, sz, sz, sz, results);
        printf("\n");
    }

    // ── CSV block: paste into a spreadsheet or feed the JOURNEY tables ────────
    printf("--- CSV ---\n");
    printf("kernel,M,N,K,gflops,seconds\n");
    for (const auto& r : results)
        printf("%s,%d,%d,%d,%.3f,%.6f\n",
               r.name.c_str(), r.M, r.N, r.K, r.gflops, r.secs);

    return 0;
}
