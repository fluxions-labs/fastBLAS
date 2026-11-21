#include <algorithm>
#include <vector>
#include "blas.h"

// One-level Strassen: split A, B into 2x2 blocks of size N/2, compute the 7
// Strassen products with sgemm_blis (beta=0, alpha=1), recombine with adds.
// Falls back to sgemm_blis directly for odd sizes or small N.
namespace {

void add(const float* X, int ldx, const float* Y, int ldy, int n, int m, float* Out, int ldo) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Out[i * ldo + j] = X[i * ldx + j] + Y[i * ldy + j];
}

void sub(const float* X, int ldx, const float* Y, int ldy, int n, int m, float* Out, int ldo) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            Out[i * ldo + j] = X[i * ldx + j] - Y[i * ldy + j];
}

void mul(const float* X, int ldx, const float* Y, int ldy, int n, int k, int m, float* Out, int ldo) {
    sgemm_blis(n, m, k, 1.0f, X, ldx, Y, ldy, 0.0f, Out, ldo);
}

} // namespace

void sgemm_strassen(int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc)
{
    // Strassen requires square, even-dimension blocks. Fall back otherwise.
    if (M != N || N != K || (M % 2) != 0 || M < 256) {
        sgemm_blis(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    int n = M / 2;

    const float* A11 = &A[0];
    const float* A12 = &A[n];
    const float* A21 = &A[n * lda];
    const float* A22 = &A[n * lda + n];

    const float* B11 = &B[0];
    const float* B12 = &B[n];
    const float* B21 = &B[n * ldb];
    const float* B22 = &B[n * ldb + n];

    std::vector<float> T1(n * n), T2(n * n);
    std::vector<float> M1(n * n), M2(n * n), M3(n * n), M4(n * n),
        M5(n * n), M6(n * n), M7(n * n);

    // M1 = (A11+A22)(B11+B22)
    add(A11, lda, A22, lda, n, n, T1.data(), n);
    add(B11, ldb, B22, ldb, n, n, T2.data(), n);
    mul(T1.data(), n, T2.data(), n, n, n, n, M1.data(), n);

    // M2 = (A21+A22)B11
    add(A21, lda, A22, lda, n, n, T1.data(), n);
    mul(T1.data(), n, B11, ldb, n, n, n, M2.data(), n);

    // M3 = A11(B12-B22)
    sub(B12, ldb, B22, ldb, n, n, T2.data(), n);
    mul(A11, lda, T2.data(), n, n, n, n, M3.data(), n);

    // M4 = A22(B21-B11)
    sub(B21, ldb, B11, ldb, n, n, T2.data(), n);
    mul(A22, lda, T2.data(), n, n, n, n, M4.data(), n);

    // M5 = (A11+A12)B22
    add(A11, lda, A12, lda, n, n, T1.data(), n);
    mul(T1.data(), n, B22, ldb, n, n, n, M5.data(), n);

    // M6 = (A21-A11)(B11+B12)
    sub(A21, lda, A11, lda, n, n, T1.data(), n);
    add(B11, ldb, B12, ldb, n, n, T2.data(), n);
    mul(T1.data(), n, T2.data(), n, n, n, n, M6.data(), n);

    // M7 = (A12-A22)(B21+B22)
    sub(A12, lda, A22, lda, n, n, T1.data(), n);
    add(B21, ldb, B22, ldb, n, n, T2.data(), n);
    mul(T1.data(), n, T2.data(), n, n, n, n, M7.data(), n);

    // C11 = M1+M4-M5+M7 ; C12 = M3+M5 ; C21 = M2+M4 ; C22 = M1-M2+M3+M6
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float c11 = M1[i * n + j] + M4[i * n + j] - M5[i * n + j] + M7[i * n + j];
            float c12 = M3[i * n + j] + M5[i * n + j];
            float c21 = M2[i * n + j] + M4[i * n + j];
            float c22 = M1[i * n + j] - M2[i * n + j] + M3[i * n + j] + M6[i * n + j];

            float* c11p = &C[i * ldc + j];
            float* c12p = &C[i * ldc + n + j];
            float* c21p = &C[(n + i) * ldc + j];
            float* c22p = &C[(n + i) * ldc + n + j];

            *c11p = alpha * c11 + beta * (*c11p);
            *c12p = alpha * c12 + beta * (*c12p);
            *c21p = alpha * c21 + beta * (*c21p);
            *c22p = alpha * c22 + beta * (*c22p);
        }
    }
}
