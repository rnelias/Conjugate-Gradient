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
#define MAX_EXECUTION_COUNT 1000

/* Controls the maximum number of CUDA threads
 * used for the CG computation.
 */
#define MAX_THREAD_COUNT_SHIFT 5

/* The size (height and width) of each thread
 * block.  Currently this must be a power of two.
 */
#define BLOCK_SIZE 4

/* ---------- Conjugate Gradient ---------- */
void cgConjGrad(int, Matrix *, Vector *, Vector *, Vector *);
void cgTestMVOps(Matrix *, Vector *, Vector *);

void cgHPrintVector(Vector *vec_a)
{
    int i = 0;

    if(vec_a == NULL)
    {
        printf("Vector is NULL\n");
        return;
    }

    printf("Vector (%p):\n", vec_a);
    printf("\tSize: %d\n", vec_a->size);
    printf("\tValues (%p):\n", vec_a->values);
    for(; i < vec_a->size; i++)
        printf("\t\tvec[%d]: %f\n", i, vec_a->values[i]);
}

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
  cgCopyVector(&hv_b, &dv_x);
  
  /* Test MV Ops */
  //cgTestMVOps(dm_A, dv_b, dv_x);
  
  /* Compute CG */
  printf("\tThreads\tTime (ms)\n");
  for(exec_count = 0; exec_count < MAX_EXECUTION_COUNT; exec_count++) 
  {
      cudaEventRecord(start, 0);
      cgConjGrad(max_iterations, dm_A, dv_b, &hv_b, dv_x);
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      
      cudaEventElapsedTime(&time, start, stop);

      printf("%04d\t%d\t%f\n", exec_count, BLOCK_SIZE * BLOCK_SIZE, time);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaPrintfEnd();

  return 0;
}

/* ---------- Conjugate Gradient ---------- */

void cgConjGrad(int max_iterations, Matrix *pdmat_A, Vector *pdvec_b, Vector *phvec_b, Vector *pdvec_x)
{
    Vector *pdvec_r, *pdvec_s, *pdvec_p, *phvec_tmp;
    Vector *pdsv_res, *pdvx_old, *pdvx_new, *pdvec_pr;
    dim3 threadsPerBlock;

    double *pd_alpha, h_alpha;
    double *pd_beta, h_beta;
    double *pd_dp_1, h_dp_1;
    double *pd_dp_2, h_dp_2;
    int k, numBlocks;

    numBlocks = 1;
    threadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
    k = 0;

    phvec_tmp = NULL;

    /* Required by the algorithm */
    cgCopyVector(phvec_b, &pdvec_r);
    cgCopyVector(phvec_b, &pdvec_p);

    /* Hackish-way of initializing the device Vectors */
    cgCopyVector(phvec_b, &pdvec_s);
    cgCopyVector(phvec_b, &pdsv_res);
    cgCopyVector(phvec_b, &pdvx_old);
    cgCopyVector(phvec_b, &pdvx_new);
    cgCopyVector(phvec_b, &pdvec_pr);
    cgCloneVector(phvec_b, &phvec_tmp);

    cudaMalloc(&pd_dp_1, sizeof(double));
    cudaMalloc(&pd_dp_2, sizeof(double));
    cudaMalloc(&pd_alpha, sizeof(double));
    cudaMalloc(&pd_beta, sizeof(double));
    
    while(TRUE)
    {
        threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
        cgMVMult<<<numBlocks, threadsPerBlock>>>(pdmat_A, pdvec_p, pdvec_s);

        threadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
        cgDotProduct<<<numBlocks, threadsPerBlock>>>(pdvec_r, pdvec_r, pd_dp_1);
        cgDotProduct<<<numBlocks, threadsPerBlock>>>(pdvec_p, pdvec_s, pd_dp_2);

        cudaMemcpy(&h_dp_1, pd_dp_1, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_dp_2, pd_dp_2, sizeof(double), cudaMemcpyDeviceToHost);
        h_alpha = h_dp_1 / h_dp_2;
        cudaMemcpy(pd_alpha, &h_alpha, sizeof(double), cudaMemcpyHostToDevice);

        cgSVMult<<<numBlocks, threadsPerBlock>>>(pd_alpha, pdvec_p, pdsv_res);

        cgCopyVectorToHost(pdvx_new, &phvec_tmp);
        cgCopyVector(phvec_tmp, &pdvx_old);
        cgVecAdd<<<numBlocks, threadsPerBlock>>>(pdvx_old, pdsv_res, pdvx_new);

        cgCopyVectorToHost(pdvec_r, &phvec_tmp);
        cgCopyVector(phvec_tmp, &pdvec_pr);
        cgSVMult<<<numBlocks, threadsPerBlock>>>(pd_alpha, pdvec_s, pdsv_res);

        cgVecSub<<<numBlocks, threadsPerBlock>>>(pdvec_r, pdsv_res, pdvec_r);

        if(k == max_iterations)
        {
            break;
        }

        cgDotProduct<<<numBlocks, threadsPerBlock>>>(pdvec_r, pdvec_r, pd_dp_1);
        cgDotProduct<<<numBlocks, threadsPerBlock>>>(pdvec_pr, pdvec_pr, pd_dp_2);

        cudaMemcpy(&h_dp_1, pd_dp_1, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_dp_2, pd_dp_2, sizeof(double), cudaMemcpyDeviceToHost);
        h_beta = h_dp_1 / h_dp_2;
        cudaMemcpy(pd_beta, &h_beta, sizeof(double), cudaMemcpyHostToDevice);

        cgSVMult<<<numBlocks, threadsPerBlock>>>(pd_beta, pdvec_p, pdsv_res);

        cgVecAdd<<<numBlocks, threadsPerBlock>>>(pdvec_r, pdsv_res, pdvec_p);
        
        k++;
    }

    /* get vx_new, print it */
    cgCopyVectorToHost(pdvx_new, &phvec_tmp);
    //cgHPrintVector(phvec_tmp);
}

/* ---------- Testing Matrix and Vector operations ---------- */

void cgTestMVOps(Matrix *pdmat_A, Vector *pdvec_b, Vector *pdvec_c)
{
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
    int numBlocks = 1;

    Vector *pvec_c = NULL;

    cgVecAdd<<<numBlocks, threadsPerBlock>>>(pdvec_b, pdvec_b, pdvec_c);
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, false);

    cgCopyVectorToHost(pdvec_c, &pvec_c);
    cgHPrintVector(pvec_c);

    cgVecSub<<<numBlocks, threadsPerBlock>>>(pdvec_b, pdvec_b, pdvec_c);
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, false);

    cgCopyVectorToHost(pdvec_c, &pvec_c);
    cgHPrintVector(pvec_c);

    /* "4.0" should be double *
    cgSVMult<<<numBlocks, threadsPerBlock>>>(4.0, pdvec_b, pdvec_c);
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, false);
    */

    cgCopyVectorToHost(pdvec_c, &pvec_c);
    cgHPrintVector(pvec_c);

    double h_dp_res, *pd_dp_res;
    cudaMalloc(&pd_dp_res, sizeof(double));
    cgDotProduct<<<numBlocks, threadsPerBlock>>>(pdvec_b, pdvec_b, pd_dp_res);
    cudaThreadSynchronize();
    cudaPrintfDisplay(stdout, false);

    cudaMemcpy(&h_dp_res, pd_dp_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Dot product: %f\n", h_dp_res);

    
    threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    cgMVMult<<<numBlocks, threadsPerBlock>>>(pdmat_A, pdvec_b, pdvec_c);
    cudaThreadSynchronize();
    cudaPrintfDisplay();

    cgCopyVectorToHost(pdvec_c, &pvec_c);
    cgHPrintVector(pvec_c);
}
