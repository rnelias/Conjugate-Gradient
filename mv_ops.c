/* mv_ops.c - Matrix/Vector Operations
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 10 February 2012
 *
 */

#include "mv_ops.h"

/* ---------- Creating and Destroying Sparse Objects ---------- */

struct __mv_sparse *new_mv_struct()
{
  struct __mv_sparse *mv_sparse = NULL;

  mv_sparse = (struct __mv_sparse *)calloc(1, sizeof(struct __mv_sparse));

  return mv_sparse;
}

struct __mv_sparse *new_mv_struct_with_size(int size)
{
  struct __mv_sparse *mv_sparse = NULL;

  mv_sparse = new_mv_struct();

  mv_sparse->size = size;
  mv_sparse->nnz = size;

  mv_sparse->values = (double *)calloc(size, sizeof(double));
  mv_sparse->col_indices = NULL;
  mv_sparse->row_ptr = NULL;

  return mv_sparse;
}

void free_mv_struct(struct __mv_sparse *mv_sparse)
{
  free(mv_sparse);
}

struct __mv_sparse *mv_deep_copy(struct __mv_sparse *orig)
{
  struct __mv_sparse *cp = NULL;

  if(!orig)
    return NULL;

  cp = new_mv_struct();

  cp->size = orig->size;
  cp->nnz = orig->nnz;

  cp->values = (double *)calloc(cp->nnz, sizeof(double));
  memcpy(cp->values, orig->values, cp->nnz * sizeof(double));

  if(orig->col_indices == NULL) {
    cp->col_indices = NULL;
  } else {
    cp->col_indices = (int *)calloc(cp->nnz, sizeof(int));
    memcpy(cp->col_indices, orig->col_indices, cp->nnz * sizeof(int));
  }

  if(orig->row_ptr == NULL) {
    cp->row_ptr = NULL;
  } else {
    cp->row_ptr = (int *)calloc(cp->size + 1, sizeof(int));
    memcpy(cp->row_ptr, orig->row_ptr, (cp->size + 1) * sizeof(int));
  }

  return cp;
}

/* ---------- Printing Sparse Objects ---------- */
void print_sparse(struct __mv_sparse *sparse_obj)
{
  int i = 0;

  printf("Sparse Object:\n");
  
  if(!sparse_obj) {
    printf("\tObject is NULL\n");
    return;
  }

  printf("\tSize: %d\n", sparse_obj->size);
  printf("\tNNZ: %d\n", sparse_obj->nnz);

  printf("\tValues: %p\n", sparse_obj->values);
  for(i = 0; i < sparse_obj->nnz; i++)
    printf("\t%f\n", sparse_obj->values[i]);

}

/* ---------- Accessing Matrix Rows ---------- */

int mat_get_row(struct __mv_sparse *mat_A, int row_id, double *p_row)
{
  int ci, i;

  if(!mat_A || !p_row)
    return -1;

  ci = mat_A->row_ptr[row_id];
  
  /* TODO:  #pragma omp parallel for */
  for(i = 0; i < mat_A->size; i++) {
    p_row[i] = mat_A->col_indices[ci] == i ? mat_A->values[ci++] : 0.0;
  }

  return 0;
}

/* ---------- Arithmetic Operations ---------- */

double dot_product(struct __mv_sparse *vec_a, struct __mv_sparse *vec_b)
{
  double dp_res = 0.0;
  int i = 0;

  if(!vec_a || !vec_b)
    return -1.0;

  if(vec_a->size != vec_b->size)
    return -1.0;

#pragma omp parallel for               \
            default(shared) private(i) \
            reduction(+:dp_res)
  for(i = 0; i < vec_a->size; i++)
    dp_res += vec_a->values[i] * vec_b->values[i];

  return dp_res;
}

int sv_mult(double sca, struct __mv_sparse *vec_a, struct __mv_sparse **vec_r)
{
  int i = 0;
  
  if(!vec_a)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(vec_a->size, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->size * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
  }

#pragma omp parallel for
  for(i = 0; i < vec_a->size; i++)
    (*vec_r)->values[i] = sca * vec_a->values[i];

  return 0;
}

int mv_mult(struct __mv_sparse *mat_A, struct __mv_sparse *vec_b, struct __mv_sparse **vec_r)
{
  double *curr_row = NULL;
  int i = 0, j = 0;
  double dp_res = 0.0;

  if(!mat_A || !vec_b)
    return -1;

  if(mat_A->size != vec_b->size)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(mat_A->size, sizeof(double));
    (*vec_r)->size = mat_A->size;
    (*vec_r)->nnz = vec_b->nnz;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, mat_A->size * sizeof(double));
    bzero((*vec_r)->values, mat_A->size * sizeof(double));
    (*vec_r)->size = mat_A->size;
    (*vec_r)->nnz = vec_b->nnz;
  }

#pragma omp parallel for                        \
  default(shared) private(i,curr_row)
  for(i = 0; i < mat_A->size; i++) {
    curr_row = (double *)calloc(vec_b->size, sizeof(double));
    mat_get_row(mat_A, i, curr_row);
    dp_res = 0;

    /* Parallel dot product between current row and input vector */
#pragma omp parallel for                        \
  default(shared) private(j)                    \
  reduction(+:dp_res)
    for(j = 0; j < vec_b->size; j++) {
      dp_res += curr_row[j] * vec_b->values[j];
    }

    (*vec_r)->values[i] = dp_res;

    free(curr_row);
  }

  return 0;
}

int vec_add(struct __mv_sparse *vec_a, struct __mv_sparse *vec_b, struct __mv_sparse **vec_r)
{
  int i = 0;

  if(!vec_a || !vec_b)
    return -1;

  if(vec_a->size != vec_b->size)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(vec_a->size, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->size * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
  }

#pragma omp parallel for
  for(i = 0; i < vec_a->size; i++)
    (*vec_r)->values[i] = vec_a->values[i] + vec_b->values[i];

  return 0;
}

int vec_sub(struct __mv_sparse *vec_a, struct __mv_sparse *vec_b, struct __mv_sparse **vec_r)
{
  int i = 0;

  if(!vec_a || !vec_b)
    return -1;

  if(vec_a->size != vec_b->size)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(vec_a->size, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->size * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
  }

#pragma omp parallel for
  for(i = 0; i < vec_a->size; i++)
    (*vec_r)->values[i] = vec_a->values[i] - vec_b->values[i];

  return 0;
}
