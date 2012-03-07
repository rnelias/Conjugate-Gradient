/* cu_ops.cu - CUDA helper functions
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 2 March 2012
 *
 */

#include "cu_ops.h"
#include "cuPrintf.cu"

struct __mv_sparse *cgCopyMatrix(struct __mv_sparse *host_mat)
{
    cudaError_t cudaResult;
    struct __mv_sparse *dev_mat;

    if(!host_mat)
    {
        fprintf(stderr, "Error: cgCopyMatrix given NULL host matrix pointer\n");
        return NULL;
    }

    /* Allocate host memory for the main structure */
    dev_mat = (struct __mv_sparse *)malloc(sizeof(struct __mv_sparse));
    if(!dev_mat)
    {
        fprintf(stderr, "Error: Unable to allocate host memory for matrix structure\n");
        return NULL;
    }

    /* Allocate device memory for values and copy to it */
    cudaResult = cudaMalloc(&(dev_mat->values), host_mat->nnz * sizeof(double));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory (%d)\n", cudaResult);
        return NULL;
    }

    cudaResult = cudaMemcpy(dev_mat->values, host_mat->values, host_mat->nnz * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to copy host memory contents to device memory (%d)\n", cudaResult);
        return NULL;
    }

    /* Allocate device memory for the column index */
    cudaResult = cudaMalloc(&(dev_mat->col_indices), host_mat->nnz * sizeof(int));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory (%d)\n", cudaResult);
        return NULL;
    }

    cudaResult = cudaMemcpy(dev_mat->col_indices, host_mat->col_indices, host_mat->nnz * sizeof(int), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to copy host memory to device (%d)\n", cudaResult);
        return NULL;
    }

    /* Allocate device memory for row pointers and copy to it */
    cudaResult = cudaMalloc(&(dev_mat->row_ptr), sizeof(int) * (host_mat->size + 1));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate device memory (%d)\n", cudaResult);
        return NULL;
    }

    cudaResult = cudaMemcpy(dev_mat->row_ptr, host_mat->row_ptr, sizeof(int) * (host_mat->size + 1), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unablet to copy host memory to device (%d)\n", cudaResult);
        return NULL;
    }

    return dev_mat;
}

struct __mv_sparse *cgCopyVector(struct __mv_sparse *host_vec)
{
    cudaError_t cudaResult;
    struct __mv_sparse *dev_vec;

    if(!host_vec)
    {
        fprintf(stderr, "Error: cgCopyVector given NULL host vector pointer\n");
        return NULL;
    }

    /* Allocate for the main structure */
    dev_vec = (struct __mv_sparse *)malloc(sizeof(struct __mv_sparse));
    if(!dev_vec)
    {
        fprintf(stderr, "Error: Unable to allocate memory\n");
        return NULL;
    }

    /* Allocate space for values and copy them */
    cudaResult = cudaMalloc(&(dev_vec->values), sizeof(double) * host_vec->nnz);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Unable to allocate memory on the device (%d)\n", cudaResult);
        return NULL;
    }

    cudaResult = cudaMemcpy(dev_vec->values, host_vec->values, sizeof(double) * host_vec->nnz, cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: Failed to copy memory to device (%d)\n", cudaResult);
        return NULL;
    }
   
    return dev_vec;
}


/* ---------- Helper Kernels ---------- */

__global__ void cgPrintValues(double *d_values)
{
    int i = threadIdx.x;
    cuPrintf("Thread %d has value %f\n", i, d_values[i]);
}
