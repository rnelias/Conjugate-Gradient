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

/* Controls the number of times the CG is 
 * executed (handy for collecting data) 
 */
#define MAX_EXECUTION_COUNT 10

/* Controls the maximum number of CUDA threads
 * used for the CG computation.
 */
#define MAX_THREAD_COUNT_SHIFT

/* The size (height and width) of each thread
 * block.  Currently this must be a power of two.
 */
#define BLOCK_SIZE 8

/* ---------- Conjugate Gradient ---------- */
__global__ void cgConjGrad(int, Matrix *, Vector *);
__global__ void cgTestMVOps(Matrix *, Vector *);

int main(int argc, char **argv)
{
  const char *input_file = NULL;
  int max_iterations = -1;
  int exec_count = 0;

  /* Process command arguments */
  if(argc < 3) {
    fprintf(stderr, "Usage: %s <input-data> <max-iterations> [suppress-output]\n", argv[0]);
    return -1;
  }

  input_file = argv[1];
  max_iterations = (int)strtol(argv[2], NULL, 10);

  /* Figure out the number of blocks and threads per block */
  int numBlocks = 1;
  dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

  cudaEvent_t start, stop;
  float time;

  /* CUDA printf() initialization */
  cudaPrintfInit();

  /* Initialize CUDA events */
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Initialize matrix A and vector b */
  Matrix *dm_A, hm_A;
  Vector *dv_b, hv_b;

  /* Read input */
  read_input_file(input_file, &hm_A, &hv_b);
  
  /* Send data to GPU */
  cgCopyMatrix(&hm_A, &dm_A);
  cgCopyVector(&hv_b, &dv_b);
  
  /* Test MV Ops */
  cgTestMVOps<<<numBlocks, threadsPerBlock>>>(dm_A, dv_b);
  cudaThreadSynchronize();
  cudaPrintfDisplay(stdout, true);
  
  /* Compute CG */
  printf("\tThreads\tTime\n");
  for(exec_count = 0; exec_count < MAX_EXECUTION_COUNT; exec_count++) 
  {
      cudaEventRecord(start, 0);
      cgConjGrad<<<numBlocks, threadsPerBlock>>>(max_iterations, dm_A, dv_b);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      
      cudaEventElapsedTime(&time, start, stop);
      
      printf("%04d\t%d\t%f\n", exec_count, threadsPerBlock.x * threadsPerBlock.y, time);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaPrintfEnd();

  return 0;
}

/* ---------- Conjugate Gradient ---------- */

__global__ void cgConjGrad(int max_iterations, Matrix *pmat_A, Vector *pvec_b)
{
    /*
    while(TRUE)
    {
        cgMVMult(mat_A, vec_p, &vec_s);
        sca_alpha = cgDotProduct(vec_r, vec_r) / cgDotProduct(vec_p, vec_s);

        cgSVMult(sca_alpha, vec_p, &sv_res);
        
        vx_old = cgDeepCopy(vx_new);
        cgVecAdd(vx_old, sv_res, &vx_new);

        vec_prev_r = cgDeepCopy(vec_r);

        cgSVMult(sca_alpha, vec_s, &sv_res);
        cgVecSub(vec_r, sv_res, &vec_r);

        if(k == max_ter)
        {
            break;
        }

        sca_beta = cgDotProduct(vec_r, vec_r) / cgDotProduct(vec_prev_r, vec_prev_r);

        cgSVMult(sca_beta, vec_p, &sv_res);
        cgVecAdd(vec_r, sv_res, &vec_p);
        
        k++;
    }
    */
}

/* ---------- Testing Matrix and Vector operations ---------- */

__global__ void cgTestMVOps(Matrix *pmat_A, Vector *pvec_b)
{
    //int i = threadIdx.x;
    Vector vec_b, vec_c;
    __shared__ double dp[5];
    double dp_res;
    Matrix mat_A;

    mat_A = *pmat_A;
    vec_b = *pvec_b;

    cgVecAdd(vec_b, vec_b, &vec_c);
    //cuPrintf("cgVecAdd: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgVecSub(vec_b, vec_b, &vec_c);
    //cuPrintf("cgVecSub: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgSVMult(4, vec_b, &vec_c);
    //cuPrintf("cgSVMult: vec_c[%d] = %f\n", i, vec_c.values[i]);

    cgDotProduct(vec_b, vec_b, dp);
    cgReduce(dp, 5, &dp_res);
    //cuPrintf("cgDotProduct+cgReduce: %f\n", dp_res);

    cgMVMult(mat_A, vec_b, &vec_c);
    //cuPrintf("cgMVMult: vec_c[%d] = %f\n", i, vec_c.values[i+threadIdx.y]);
}
