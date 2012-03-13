/* cu_ops.cuh - CUDA helper functions
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 2 March 2012
 *
 */

#ifndef __CU_OPS_CUH__
#define __CU_OPS_CUH__

#include <cuda_runtime_api.h>

/* ---- Matrix-Vector Operations ---- */
__global__ void cgVecAdd(Vector *vec_a, Vector *vec_b, Vector *vec_c);
__global__ void cgVecSub(Vector *vec_a, Vector *vec_b, Vector *vec_c);
__global__ void cgSVMult(double *sca, Vector *vec_a, Vector *vec_b);
__global__ void cgDotProduct(Vector *vec_a, Vector *vec_b, double *dp_final);
__global__ void cgReduce(double *dp, int dp_size, double *dp_final);
__global__ void cgMVMult(Matrix *mat_A, Vector *vec_b, Vector *vec_c);

/* ---- Device Memory Management ---- */
void cgDeepCopy(Vector *, Vector *);

/* ---- Copying between the host and device ---- */
int cgCopyMatrix(Matrix *h_m, Matrix **d_m);
int cgCopyVector(Vector *h_v, Vector **d_v);
int cgCopyVectorToHost(Vector *d_v, Vector **pph_v);
int cgCloneVector(Vector *, Vector **);

/* ---- Helper kernels ---- */
__global__ void cgPrintVector(Vector *);

#endif /* __CU_OPS_CUH__ */