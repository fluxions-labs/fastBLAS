#include "blas.h"

// ─────────────────────────────────────────────────────────────────────────────
// Level 1: naive triple loop, i-j-k order.
//
// For each output element C[i][j] we walk across an entire row of A and an
// entire column of B.  The column-of-B access pattern strides by ldb floats
// between consecutive elements, so every load misses in L1.  This is the
// pathological baseline that every subsequent level improves upon.
// ─────────────────────────────────────────────────────────────────────────────
void sgemm_naive(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float sum=0.0f;
            for(int k=0;k<K;k++){
                sum+=A[i*lda+k]*B[k*ldb+j];
            }
            C[i*ldc+j]=beta*C[i*ldc+j]+alpha*sum;
        }
    }
}
