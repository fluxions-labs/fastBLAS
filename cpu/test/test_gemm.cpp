#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "blas.h"

// Reference: the naive implementation is our ground truth.
// Every new level is tested against this.
static void reference(int M, int N, int K, float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    sgemm_naive(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

using GemmFn = void(*)(int,int,int, float, const float*,int, const float*,int, float, float*,int);

static bool check(const char* name, GemmFn fn, int M, int N, int K) {
    std::vector<float> A(M*K), B(K*N), C0(M*N), C_ref(M*N), C_test(M*N);
    for (auto& v : A)  v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : B)  v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : C0) v = static_cast<float>(rand()) / RAND_MAX;

    float alpha = 1.5f, beta = 0.5f;
    C_ref  = C0;
    C_test = C0;

    reference(M, N, K, alpha, A.data(), K, B.data(), N, beta, C_ref.data(),  N);
    fn       (M, N, K, alpha, A.data(), K, B.data(), N, beta, C_test.data(), N);

    float max_err = 0.0f;
    for (int i = 0; i < M*N; i++)
        max_err = std::max(max_err, std::abs(C_ref[i] - C_test[i]));

    // Tolerance: float32 accumulation errors grow with K.
    float tol = 1e-4f * static_cast<float>(K);
    bool ok = max_err <= tol;

    printf("  [%s] %dx%dx%d  max_err=%.2e  tol=%.2e  %s\n",
           name, M, N, K, max_err, tol, ok ? "PASS" : "FAIL");
    return ok;
}

int main() {
    bool all_pass = true;

    printf("── Level 1: sgemm_naive ─────────────────────────────────────────\n");
    // Naive is the reference, so these must always pass.
    all_pass &= check("sgemm_naive", sgemm_naive, 64,  64,  64);
    all_pass &= check("sgemm_naive", sgemm_naive, 128, 256, 64);
    all_pass &= check("sgemm_naive", sgemm_naive, 256, 256, 256);

    // ── Uncomment as each level is implemented ────────────────────────────
    printf("── Level 2: sgemm_reorder ───────────────────────────────────────\n");
    all_pass &= check("sgemm_reorder", sgemm_reorder, 64,  64,  64);
    all_pass &= check("sgemm_reorder", sgemm_reorder, 128,  256,  64);
    all_pass &= check("sgemm_reorder", sgemm_reorder, 256, 256, 256);

    printf("── Level 3: sgemm_reg ───────────────────────────────────────────\n");
    all_pass &= check("sgemm_reg", sgemm_reg, 64,  64,  64);
    all_pass &= check("sgemm_reg", sgemm_reg, 128,  256,  64);
    all_pass &= check("sgemm_reg", sgemm_reg, 256, 256, 256);

    // ── Uncomment each block as you implement the level ───────────────────────
    // Use non-square and edge sizes (e.g. 130×255×97) for the tiled/micro kernels
    // so you exercise the tail/remainder paths.
    //
    // all_pass &= check("sgemm_1x4",      sgemm_1x4,      130, 255, 97);
    // all_pass &= check("sgemm_block",    sgemm_block,    130, 255, 97);
    // all_pass &= check("sgemm_packed",   sgemm_packed,   130, 255, 97);
    // all_pass &= check("sgemm_neon",     sgemm_neon,     130, 255, 97);
    // all_pass &= check("sgemm_omp",      sgemm_omp,      130, 255, 97);
    // all_pass &= check("sgemm_micro",    sgemm_micro,    130, 255, 97);  // L9
    // all_pass &= check("sgemm_pipe",     sgemm_pipe,     130, 255, 97);  // L10
    // all_pass &= check("sgemm_blis",     sgemm_blis,     130, 255, 97);  // L11
    // all_pass &= check("sgemm_tuned",    sgemm_tuned,    130, 255, 97);  // L12
    // all_pass &= check("sgemm_strassen", sgemm_strassen, 512, 512, 512); // L14 (looser tol)

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
