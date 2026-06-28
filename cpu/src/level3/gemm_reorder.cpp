#include "blas.h"


void sgemm_reorder(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * ldc + j] = beta * C[i * ldc + j];
        }
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] += alpha * A[i * lda + k] * B[k * ldb + j];
            }
        }
    }
}