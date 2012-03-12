/* cu_ops.cu - CUDA helper functions
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 2 March 2012
 *
 */

#include "cu_ops.cuh"
#include "cuPrintf.cu"

/* ---------- Matrix-Vector Operations ---------- */

__device__ void cgMVMult(Matrix mat_A, Vector vec_b, Vector *vec_c)
{
    int mx_idx = threadIdx.x + (blockDim.x * threadIdx.y);
    int ve_idx = threadIdx.x;
    unsigned int s = 0;
    double temp_res, row_res;
    __shared__ double temp_mat[8][8];

    temp_res = mat_A.values[mx_idx] * vec_b.values[ve_idx];
    temp_mat[threadIdx.y][threadIdx.x] = temp_res;
    __syncthreads();

    for(s = 1; s < blockDim.x; s *= 2)
    {
        if(ve_idx % (2*s) == 0)
        {
            temp_mat[threadIdx.y][threadIdx.x] += temp_mat[threadIdx.y][threadIdx.x + s];
        }
    }
    
    row_res = temp_mat[threadIdx.y][0];
    if(ve_idx == 0)
        vec_c->values[threadIdx.y] = row_res;
}

__device__ void cgReduce(double *dp, int dp_size, double *dp_final)
{
    unsigned int s;
    int tid = threadIdx.x;

    for(s = 1; s < blockDim.x; s *= 2)
    {
        if(tid % (2*s) == 0)
        {
            dp[tid] += dp[tid + s];
        }

        __syncthreads();
    }

    *dp_final = dp[0];
}

__device__ void cgDotProduct(Vector vec_a, Vector vec_b, double *dp)
{
    int i = threadIdx.x;
    dp[i] = vec_a.values[i] * vec_b.values[i];
}

__device__ void cgSVMult(int sca, Vector vec_a, Vector *vec_b)
{
    int i = threadIdx.x;
    vec_b->values[i] = sca * vec_a.values[i];
}

__device__ void cgVecSub(Vector vec_a, Vector vec_b, Vector *vec_c)
{
    int i = threadIdx.x;
    vec_c->values[i] = vec_a.values[i] - vec_b.values[i];
}

__device__ void cgVecAdd(Vector vec_a, Vector vec_b, Vector *vec_c)
{
    int i = threadIdx.x;
    vec_c->values[i] = vec_a.values[i] + vec_b.values[i];
}

/* ---------- Copying between host and device ---------- */

int cgCopyMatrix(Matrix *h_m, Matrix **d_m)
{
    cudaError_t cudaResult;
    Matrix h_temp;
    double *d_values;
    int *d_column_indices, *d_row_pointers;

    /* Allocate space and copy values */
    cudaResult = cudaMalloc(&d_values, sizeof(double) * h_m->nnz);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    cudaResult = cudaMemcpy(d_values, h_m->values, sizeof(double) * h_m->nnz, cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    /* Allocate space and copy column indices */
    cudaResult = cudaMalloc(&d_column_indices, sizeof(int) * h_m->nnz);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    cudaResult = cudaMemcpy(d_column_indices, h_m->column_indices, sizeof(int) * h_m->nnz, cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    /* Allocate space and copy row pointers */
    cudaResult = cudaMalloc(&d_row_pointers, sizeof(int) * (h_m->size + 1));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    cudaResult = cudaMemcpy(d_row_pointers, h_m->row_pointers, sizeof(int) * (h_m->size + 1), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }
    
    /* Allocate space on device for matrix structure */
    cudaResult = cudaMalloc(d_m, sizeof(Matrix));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    h_temp.size = h_m->size;
    h_temp.nnz = h_m->nnz;
    h_temp.values = d_values;
    h_temp.column_indices = d_column_indices;
    h_temp.row_pointers = d_row_pointers;

    cudaResult = cudaMemcpy(*d_m, &h_temp, sizeof(Matrix), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    return 0;
}

int cgCopyVector(Vector *h_v, Vector **d_v)
{
    cudaError_t cudaResult;
    Vector h_temp;
    double *d_values;

    /* Allocate space and copy values */
    cudaResult = cudaMalloc(&d_values, sizeof(double) * h_v->size);
    if(cudaResult != cudaSuccess) 
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    cudaResult = cudaMemcpy(d_values, h_v->values, h_v->size * sizeof(double), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess) 
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    /* Allocate space on the device for the vector structure */
    h_temp.values = d_values;
    h_temp.size = h_v->size;

    cudaResult = cudaMalloc(d_v, sizeof(Vector));
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    cudaResult = cudaMemcpy(*d_v, &h_temp, sizeof(Vector), cudaMemcpyHostToDevice);
    if(cudaResult != cudaSuccess)
    {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(cudaResult));
        return -1;
    }

    return 0;
}


/* ---------- Helper Kernels ---------- */

/* Note: "Vector *v" produces a warning during compilation about NVCC not being able to deduce where the pointer points.  This 
 * is because (1) cards with compute capability 1.x have separate address spaces for global and shared memory and (2) the pointer
 * can actually point to either.  NVCC assumes global memory which (in this case) is correct.
 */
__global__ void cgPrintVector(Vector *v)
{
    int i = threadIdx.x;

    cuPrintf("Thread %d has value %f\n", i, v->values[i]);
}
