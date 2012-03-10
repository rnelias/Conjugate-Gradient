/* cg.c - Conjugate Gradient, in all it's glory!
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 8 March 2012
 *
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mv_types.cuh"
#include "cu_ops.cu"
#include "reader.cu"

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#define MAX_EXECUTION_COUNT 1

__global__ void printStupidVector(Vector *v)
{
    cuPrintf("Vector\n");
    cuPrintf("\tsize: %d\n", v->size);
}

/* ---------- Conjugate Gradient ---------- */
__global__ void cgConjGrad(int, Matrix, Vector);
__global__ void cgTestMVOps(Matrix *, Vector *);

int main(int argc, char **argv)
{
  const char *input_file = NULL;
/*
  int max_iterations = -1, no_output = FALSE;
  int exec_count = 0;
*/

  /* Process command arguments */
  if(argc < 3) {
    fprintf(stderr, "Usage: %s <input-data> <max-iterations> [suppress-output]\n", argv[0]);
    return -1;
  }

  input_file = argv[1];
  //max_iterations = (int)strtol(argv[2], NULL, 10);

  /*
  if(argc == 4)
    no_output = argv[3][0] == 'y' ? TRUE : FALSE;
  */

  /* Figure out the number of blocks and threads per block */
  int numBlocks = 1;
  int threadsPerBlock = 5;

  cudaError_t cudaResult;

  /* Device init */
  cudaResult = cudaSetDevice(0);
  if(cudaResult != cudaSuccess)
  {
      fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
      return -1;
  }

  cudaResult = cudaSetDeviceFlags(cudaDeviceBlockingSync);
  if(cudaResult != cudaSuccess)
  {
      fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
      return -1;
  }

  /* CUDA printf() initialization */
  cudaPrintfInit();

  /* Initialize matrix A and vector b */
  Matrix *dm_A, hm_A;
  Vector *dv_b, hv_b;

  /* Read input */
  printf("Reading in data... ");
  read_input_file(input_file, &hm_A, &hv_b);
  printf("Done\n");

  /* Send data to GPU */
  cgCopyMatrix(&hm_A, &dm_A);
  cgCopyVector(&hv_b, &dv_b);

  /* Test MV Ops */
  cgTestMVOps<<<numBlocks, threadsPerBlock>>>(dm_A, dv_b);
  cudaThreadSynchronize();
  cudaPrintfDisplay(stdout, true);

  /* Compute CG */
  //for(exec_count = 0; exec_count < MAX_EXECUTION_COUNT; exec_count++) {
  //  conj_grad(max_iterations, mat_A, vec_b, &vec_x);
  //  printf("Execution %d -> Configuration: CUDA | Time: %f sec\n", exec_count+1, elapsedSec);
  //}

  cudaPrintfEnd();

  return 0;
}

/* ---------- Conjugate Gradient ---------- */

__global__ void cgConjGrad(int max_iterations, Matrix mat_A, Vector vec_b)
{
}

/* ---------- Testing Matrix and Vector operations ---------- */

__global__ void cgTestMVOps(Matrix *pmat_A, Vector *pvec_b)
{
    int i = threadIdx.x;
    Vector vec_b, vec_c;
    double dp[5], dp_res;

    cuPrintf("cgTestMVOps on thread %d\n", i);

    vec_b = *pvec_b;

    cgVecAdd(vec_b, vec_b, &vec_c);
    cuPrintf("cgVecAdd: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgVecSub(vec_b, vec_b, &vec_c);
    cuPrintf("cgVecSub: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgSVMult(4, vec_b, &vec_c);
    cuPrintf("cgSVMult: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgDotProduct(vec_b, vec_b, dp);
    cgReduce(dp, 5, &dp_res);
    cuPrintf("cgDotProduct+cgReduce: %f\n", dp_res);
}
