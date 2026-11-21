#include <algorithm>
#include <vector>
#include <arm_neon.h>
#include "blas.h"

namespace {
constexpr int MC = 128;
constexpr int KC = 256;
constexpr int NC = 128;
constexpr int MR = 8;
constexpr int NR = 8;

// Same 8x8 micro-kernel as Level 9, unrolled by 4 in k with software prefetch
// of the next k-iteration's B row.
void micro_8x8_pipe(int mc, int nc, int kc,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc)
{
    int i8 = mc - (mc % MR);
    int j8 = nc - (nc % NR);

    for (int i = 0; i < i8; i += MR) {
        for (int j = 0; j < j8; j += NR) {
            float32x4_t c[8][2];
            for (int r = 0; r < 8; r++) {
                c[r][0] = vdupq_n_f32(0.0f);
                c[r][1] = vdupq_n_f32(0.0f);
            }

            int k = 0;
            int k4 = kc - (kc % 4);
            for (; k < k4; k += 4) {
                __builtin_prefetch(&B[(k + 4) * ldb + j], 0, 3);
                __builtin_prefetch(&A[i * lda + k + 4], 0, 3);
                for (int u = 0; u < 4; u++) {
                    float32x4_t b0 = vld1q_f32(&B[(k + u) * ldb + j]);
                    float32x4_t b1 = vld1q_f32(&B[(k + u) * ldb + j + 4]);
                    for (int r = 0; r < 8; r++) {
                        float a_val = A[(i + r) * lda + (k + u)];
                        c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                        c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                    }
                }
            }
            for (; k < kc; k++) {
                float32x4_t b0 = vld1q_f32(&B[k * ldb + j]);
                float32x4_t b1 = vld1q_f32(&B[k * ldb + j + 4]);
                for (int r = 0; r < 8; r++) {
                    float a_val = A[(i + r) * lda + k];
                    c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                    c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                }
            }

            float32x4_t valpha = vdupq_n_f32(alpha);
            float32x4_t vbeta  = vdupq_n_f32(beta);
            for (int r = 0; r < 8; r++) {
                float* cp0 = &C[(i + r) * ldc + j];
                float* cp1 = &C[(i + r) * ldc + j + 4];
                vst1q_f32(cp0, vfmaq_f32(vmulq_f32(valpha, c[r][0]), vbeta, vld1q_f32(cp0)));
                vst1q_f32(cp1, vfmaq_f32(vmulq_f32(valpha, c[r][1]), vbeta, vld1q_f32(cp1)));
            }
        }
    }

    for (int i = 0; i < mc; i++) {
        int jstart = (i < i8) ? j8 : 0;
        for (int j = jstart; j < nc; j++) {
            float sum = 0.0f;
            for (int k = 0; k < kc; k++)
                sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

void pack_A(const float* A, int lda, int mc, int kc, float* packed) {
    for (int ir = 0; ir < mc; ir++)
        for (int kr = 0; kr < kc; kr++)
            packed[ir * kc + kr] = A[ir * lda + kr];
}

void pack_B(const float* B, int ldb, int kc, int nc, float* packed) {
    for (int kr = 0; kr < kc; kr++)
        for (int jr = 0; jr < nc; jr++)
            packed[kr * nc + jr] = B[kr * ldb + jr];
}
} // namespace

void sgemm_pipe(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    std::vector<float> packed_A(MC * KC);
    std::vector<float> packed_B(KC * NC);

    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        for (int kc = 0; kc < K; kc += KC) {
            int kcs = std::min(KC, K - kc);
            float this_beta = (kc == 0) ? beta : 1.0f;
            pack_B(&B[kc * ldb + jc], ldb, kcs, nc, packed_B.data());
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                pack_A(&A[ic * lda + kc], lda, mc, kcs, packed_A.data());
                micro_8x8_pipe(mc, nc, kcs,
                               alpha,
                               packed_A.data(), kcs,
                               packed_B.data(), nc,
                               this_beta,
                               &C[ic * ldc + jc], ldc);
            }
        }
    }
}
