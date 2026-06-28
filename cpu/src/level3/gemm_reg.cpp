#include "blas.h"

void sgemm_reg(int M, int N, int K, float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc) {
      for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
          C[i*ldc+j]*=beta;
        }
        for(int k=0;k<K;k++){
          float reg=A[i*lda+k]*alpha;
          for(int j=0;j<N;j++){
            C[i*ldc+j]+=reg*B[k*ldb+j];
          }
        }
      }
}