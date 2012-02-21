/* mv_ops.c - Matrix/Vector Operations
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 10 February 2012
 *
 */

#include "mv_ops.h"

extern int g_mpi_rank;
extern int g_mpi_group_size;

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

/* ---------- Distribution ---------- */
struct __mv_sparse *distribute_matrix(struct __mv_sparse *mat_A)
{
  int d_size, d_nnz;
  int num_values, num_col_indices, num_row_ptr;

  double *r_values = NULL;
  int *r_col_indices = NULL;
  int *r_row_ptr = NULL;

  struct __mv_sparse *d_mat = NULL;

  if(g_mpi_rank == ROOT_RANK) {
    d_size = mat_A->size;
    d_nnz = mat_A->nnz;
    r_values = mat_A->values;
    r_col_indices = mat_A->col_indices;
    r_row_ptr = mat_A->row_ptr;
  }

  MPI_Bcast(&d_size, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
  MPI_Bcast(&d_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

  d_mat = new_mv_struct();
  d_mat->size = d_size;
  d_mat->nnz = d_nnz;
  d_mat->start = g_mpi_rank * (d_size / g_mpi_group_size);
  d_mat->end = (g_mpi_rank + 1) * (d_size / g_mpi_group_size);
  
  num_values = d_size / g_mpi_group_size;
  num_col_indices = num_values;
  num_row_ptr = 1;

  d_mat->values = (double *)calloc(num_values, sizeof(double));
  d_mat->col_indices = (int *)calloc(num_col_indices, sizeof(int));
  d_mat->row_ptr = (int *)calloc(num_row_ptr, sizeof(int));

  /* Scatter Values */
  MPI_Scatter(r_values, num_values, MPI_DOUBLE
              , d_mat->values, num_values, MPI_DOUBLE
              , ROOT_RANK, MPI_COMM_WORLD);

  /* Scatter column indices */
  MPI_Scatter(r_col_indices, num_col_indices, MPI_INT
              , d_mat->col_indices, num_col_indices, MPI_INT
              , ROOT_RANK, MPI_COMM_WORLD);

  /* Scatter row pointers */
  //MPI_Scatterv();

  MPI_Barrier(MPI_COMM_WORLD);
  return d_mat;
}

struct __mv_sparse *distribute_vector(struct __mv_sparse *vec_b)
{
  int d_size, d_nnz, num_values;
  double *r_values = NULL;
  struct __mv_sparse *d_vec = NULL;

  if(g_mpi_rank == ROOT_RANK) {
    d_size = vec_b->size;
    d_nnz = vec_b->nnz;
    r_values = vec_b->values;
  }

  MPI_Bcast(&d_size, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);
  MPI_Bcast(&d_nnz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

  d_vec = new_mv_struct();
  d_vec->size = d_size;
  d_vec->nnz = d_nnz;
  d_vec->start = g_mpi_rank * (d_size / g_mpi_group_size);
  d_vec->end = (g_mpi_rank + 1) * (d_size / g_mpi_group_size);

  num_values = d_vec->size / g_mpi_group_size;  
  d_vec->values = (double *)calloc(num_values, sizeof(double));
  d_vec->col_indices = NULL;
  d_vec->row_ptr = NULL;

  /* Scatter values */
  MPI_Scatter(r_values, num_values, MPI_DOUBLE
              , d_vec->values, num_values, MPI_DOUBLE
              , ROOT_RANK, MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);
  return d_vec;
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

  curr_row = (double *)calloc(vec_b->size, sizeof(double));
  for(i = 0; i < mat_A->size; i++) {
    bzero(curr_row, vec_b->size * sizeof(double));
    mat_get_row(mat_A, i, curr_row);
    dp_res = 0;

    for(j = 0; j < vec_b->size; j++) {
      dp_res += curr_row[j] * vec_b->values[j];
    }

    (*vec_r)->values[i] = dp_res;
  }
  free(curr_row);

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

  for(i = 0; i < vec_a->size; i++)
    (*vec_r)->values[i] = vec_a->values[i] - vec_b->values[i];

  return 0;
}
