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
#define MAX_EXECUTION_COUNT 1

/* Controls the maximum number of CUDA threads
 * used for the CG computation.
 */
#define MAX_THREAD_COUNT_SHIFT

/* The size (height and width) of each thread
 * block.  Currently this must be a power of two.
 */
#define BLOCK_SIZE 8

/* ---------- Conjugate Gradient ---------- */
__global__ void cgConjGrad(int, Matrix *, Vector *, Vector *, Vector **);
__global__ void cgTestMVOps(Matrix *, Vector *);

int main(int argc, char **argv)
{
  const char *input_file = NULL;
  int max_iterations = -1;
  int exec_count = 0, i = 0;

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
  Vector *dv_x, hv_x;

  /* Read input */
  read_input_file(input_file, &hm_A, &hv_b);
  
  /* Send data to GPU */
  cgCopyMatrix(&hm_A, &dm_A);
  cgCopyVector(&hv_b, &dv_b);

  if(cudaMalloc(&dv_x, sizeof(Vector)) != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate space for dv_x\n");
  }
  
  /* Test MV Ops */
  //cgTestMVOps<<<numBlocks, threadsPerBlock>>>(dm_A, dv_b);
  //cudaThreadSynchronize();
  //cudaPrintfDisplay(stdout, true);

  /* Allocate the various vectors needed for the CG method */
  Vector *hv_array[7];
  Vector **dv_array = cgDeviceAllocateVectorArray(hv_array, 7, &hv_b);
  
  /* Compute CG */
  printf("\tThreads\tTime\n");
  for(exec_count = 0; exec_count < MAX_EXECUTION_COUNT; exec_count++) 
  {
      cudaEventRecord(start, 0);
      cgConjGrad<<<numBlocks, threadsPerBlock>>>(max_iterations, dm_A, dv_b, dv_x, dv_array);
      cudaPrintfDisplay(stdout, false);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      
      cudaEventElapsedTime(&time, start, stop);

      cgCopyVectorToHost(dv_x, &hv_x);
      printf("hv_x:\n");
      printf("\tsize: %d\n", hv_x.size);
      printf("\tvalues: %p\n", hv_x.values);
      if(hv_x.values == NULL)
      {
          printf("\thv_x.values: NULL\n");
      }
      else
      {
          for(i = 0; i < hv_x.size; i++)
              printf("\thv_x.values[%d]: %f\n", i, hv_x.values[i]);
      }
      
      printf("%04d\t%d\t%f\n", exec_count, threadsPerBlock.x * threadsPerBlock.y, time);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaPrintfEnd();

  return 0;
}

/* ---------- Conjugate Gradient ---------- */

__global__ void cgConjGrad(int max_iterations, Matrix *pmat_A, Vector *pvec_b, Vector *pvec_x, Vector **pvec_array)
{
    __shared__ double dp_shared[BLOCK_SIZE];
    double dp_res_1, dp_res_2, sca_alpha, sca_beta;
    __shared__ int k;

    Matrix mat_A;
    Vector vec_b;

    Vector *pvec_r, *pvec_s, *pvec_p;
    Vector *psv_res, *pvx_old, *pvx_new, *pvec_pr;
    
    mat_A = *pmat_A;
    vec_b = *pvec_b;

    k = 0;

    pvec_r = pvec_array[0];
    pvec_s = pvec_array[1];
    pvec_p = pvec_array[2];
    
    psv_res = pvec_array[3];
    pvx_old = pvec_array[4];
    pvx_new = pvec_array[5];
    pvec_pr = pvec_array[6];

    cuPrintf("1 pvec_r: %p\n", pvec_r);
    cuPrintf("1 pvec_r->size: %d\n", pvec_r->size);
    
    cgDeepCopy(vec_b, pvec_r);
    cgDeepCopy(*pvec_r, pvec_p);

    /*
    cgDeepCopy(vec_b, &vec_s);
    cgDeepCopy(vec_b, &sv_res);
    cgDeepCopy(vec_b, &vx_old);
    cgDeepCopy(vec_b, &vx_new);
    cgDeepCopy(vec_b, &vec_prev_r);
    */
    
    cuPrintf("2 pvec_r->size: %d\n", pvec_r->size);
    cuPrintf("2 pvec_r->values: %p\n", pvec_r->values);
    cuPrintf("2 pvec_r->values[%d]: %f\n", threadIdx.y, pvec_r->values[threadIdx.y]);
    
    while(TRUE)
    {
        cgMVMult(mat_A, *pvec_p, pvec_s);
        __syncthreads();

        cgDotProduct(*pvec_r, *pvec_r, dp_shared);
        cgReduce(dp_shared, blockDim.x, &dp_res_1);
        __syncthreads();

        cgDotProduct(*pvec_p, *pvec_s, dp_shared);
        cgReduce(dp_shared, blockDim.x, &dp_res_2);
        __syncthreads();

        sca_alpha = dp_res_1 / dp_res_2;

        cgSVMult(sca_alpha, *pvec_p, psv_res);
        __syncthreads();

        cgDeepCopy(*pvx_new, pvx_old);
        cgVecAdd(*pvx_old, *psv_res, pvx_new);
        __syncthreads();

        cgDeepCopy(*pvec_r, pvec_pr);
        cgSVMult(sca_alpha, *pvec_s, psv_res);
        __syncthreads();

        cgVecSub(*pvec_r, *psv_res, pvec_r);
        __syncthreads();

        if(k == max_iterations)
        {
            break;
        }

        cgDotProduct(*pvec_r, *pvec_r, dp_shared);
        cgReduce(dp_shared, blockDim.x, &dp_res_1);
        __syncthreads();

        cgDotProduct(*pvec_pr, *pvec_pr, dp_shared);
        cgReduce(dp_shared, blockDim.x, &dp_res_2);
        __syncthreads();

        sca_beta = dp_res_1 / dp_res_2;

        cgSVMult(sca_beta, *pvec_p, psv_res);
        __syncthreads();

        cgVecAdd(*pvec_r, *psv_res, pvec_p);
        __syncthreads();
        
        k++;
    }

    pvec_x->size = pvx_new->size;
    pvec_x->values = pvx_new->values;
    //cuPrintf("pvec_x->size: %d\n", pvec_x->size);
    //cuPrintf("pvec_x->values: %p\n", pvec_x->values);
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
    //if(threadIdx.x == 0)
    //    cuPrintf("cgMVMult: vec_c[%d] = %f\n", threadIdx.y, vec_c.values[threadIdx.y]);
}
