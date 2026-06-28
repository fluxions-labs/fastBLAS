#include "blas.h"

void sgemm_reg(int M, int N, int K, float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc) {

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = beta * C[i * ldc + j];
      for (int k = 0; k < K; k++) {
        sum += alpha * A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = sum;
    }
  }
}