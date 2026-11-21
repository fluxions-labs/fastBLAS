#include <algorithm>
#include "blas.h"

namespace {
constexpr int MC = 64;
constexpr int KC = 64;
constexpr int NC = 64;

// Same 1x4 micro-kernel as Level 4, operating on a sub-tile with alpha/beta
// scaling applied once at the very end via the caller's zero/beta handling.
void micro_1x4(int mc, int nc, int kc,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float* C, int ldc)
{
    int j4 = nc - (nc % 4);
    for (int i = 0; i < mc; i++) {
        for (int j = 0; j < j4; j += 4) {
            float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            for (int k = 0; k < kc; k++) {
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
        for (int j = j4; j < nc; j++) {
            float sum = 0.0f;
            for (int k = 0; k < kc; k++)
                sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}
} // namespace

void sgemm_block(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc)
{
    // Because we accumulate into C tile-by-tile across kc blocks, and each kc
    // block calls the micro-kernel with its own beta, we must apply beta only
    // on the first kc pass and accumulate (beta=1) afterward.
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        for (int kc = 0; kc < K; kc += KC) {
            int kcs = std::min(KC, K - kc);
            float this_beta = (kc == 0) ? beta : 1.0f;
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                micro_1x4(mc, nc, kcs,
                          alpha,
                          &A[ic * lda + kc], lda,
                          &B[kc * ldb + jc], ldb,
                          this_beta,
                          &C[ic * ldc + jc], ldc);
            }
        }
    }
}
