#include "blas.h"

void sgemm_1x4(int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc)
{
    int j4 = N - (N % 4);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < j4; j += 4) {
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_ik = A[i * lda + k];
                const float* brow = &B[k * ldb + j];
                c0 += a_ik * brow[0];
                c1 += a_ik * brow[1];
                c2 += a_ik * brow[2];
                c3 += a_ik * brow[3];
            }
            float* crow = &C[i * ldc + j];
            crow[0] = alpha * c0 + beta * crow[0];
            crow[1] = alpha * c1 + beta * crow[1];
            crow[2] = alpha * c2 + beta * crow[2];
            crow[3] = alpha * c3 + beta * crow[3];
        }
        // tail: remaining columns not divisible by 4
        for (int j = j4; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}
