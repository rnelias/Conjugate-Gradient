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
  int num_values, num_col_indices;

  int *sv_row_ptrs = NULL;
  int *sv_displs = NULL;

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
  d_mat->nval = d_nnz / g_mpi_group_size;
  
  num_values = d_nnz / g_mpi_group_size;
  num_col_indices = num_values;

  d_mat->values = (double *)calloc(num_values, sizeof(double));
  d_mat->col_indices = (int *)calloc(num_col_indices, sizeof(int));
  d_mat->row_ptr = (int *)calloc((d_size / g_mpi_group_size) + 1, sizeof(int));

  /* Scatter Values */
  MPI_Scatter(r_values, num_values, MPI_DOUBLE
              , d_mat->values, num_values, MPI_DOUBLE
              , ROOT_RANK, MPI_COMM_WORLD);

  /* Scatter column indices */
  MPI_Scatter(r_col_indices, num_col_indices, MPI_INT
              , d_mat->col_indices, num_col_indices, MPI_INT
              , ROOT_RANK, MPI_COMM_WORLD);

  /* Scatter row pointers */
  if(g_mpi_rank == ROOT_RANK) {
    int i = 0;
    sv_row_ptrs = (int *)calloc(g_mpi_group_size, sizeof(int));
    sv_displs = (int *)calloc(g_mpi_group_size, sizeof(int));

    for(; i < g_mpi_group_size; i++) {
      sv_row_ptrs[i] = (d_size / g_mpi_group_size) + 1;
      sv_displs[i] = i * (d_size / g_mpi_group_size);
    }
  }
  MPI_Scatterv(r_row_ptr, sv_row_ptrs, sv_displs, MPI_INT
               , d_mat->row_ptr, (d_size / g_mpi_group_size) + 1, MPI_INT
               , ROOT_RANK, MPI_COMM_WORLD);

  printf("[rank %d] got matrix part\n", g_mpi_rank);
  //MPI_Barrier(MPI_COMM_WORLD);
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
  d_vec->nval = d_nnz / g_mpi_group_size;

  num_values = d_vec->size / g_mpi_group_size;  
  d_vec->values = (double *)calloc(num_values, sizeof(double));
  d_vec->col_indices = NULL;
  d_vec->row_ptr = NULL;

  /* Scatter values */
  MPI_Scatter(r_values, num_values, MPI_DOUBLE
              , d_vec->values, num_values, MPI_DOUBLE
              , ROOT_RANK, MPI_COMM_WORLD);
  
  printf("[rank %d] got vector part\n", g_mpi_rank);
  //MPI_Barrier(MPI_COMM_WORLD);
  return d_vec;
}

struct __mv_sparse *gather_vector(struct __mv_sparse *d_vec_x)
{
  struct __mv_sparse *vec_x = NULL;
  double *r_values = NULL;

  if(g_mpi_rank == ROOT_RANK) {
    vec_x = new_mv_struct();
    vec_x->size = d_vec_x->size;
    vec_x->nnz = d_vec_x->nnz;
    vec_x->nval = d_vec_x->nval * g_mpi_group_size;
    vec_x->start = 0;
    vec_x->end = 0;
    vec_x->col_indices = NULL;
    vec_x->row_ptr = NULL;
    vec_x->values = NULL;

    r_values = (double *)calloc(vec_x->nval, sizeof(double));
  }

  MPI_Gather(d_vec_x->values, d_vec_x->nval, MPI_DOUBLE
             , r_values, d_vec_x->nval, MPI_DOUBLE
             , ROOT_RANK, MPI_COMM_WORLD);

  if(g_mpi_rank == ROOT_RANK)
    vec_x->values = r_values;
  
  return (g_mpi_rank == ROOT_RANK) ? vec_x : NULL;
}

/* ---------- Printing Sparse Objects ---------- */
void print_sparse(struct __mv_sparse *sparse_obj, const char *obj_tag)
{
  int i = 0;

  if(obj_tag == NULL)
    printf("Sparse Object (rank %d):\n", g_mpi_rank);
  else
    printf("Sparse Object [%s] (rank %d):\n", obj_tag, g_mpi_rank);
  
  if(!sparse_obj) {
    printf("\tObject is NULL\n");
    return;
  }

  printf("\tSize: %d\n", sparse_obj->size);
  printf("\tNNZ: %d\n", sparse_obj->nnz);
  printf("\tStart row: %d\n", sparse_obj->start);
  printf("\tEnd row: %d\n", sparse_obj->end);
  printf("\tNVal: %d\n", sparse_obj->nval);

  printf("\tValues: %p\n", sparse_obj->values);
  for(i = 0; i < sparse_obj->nval; i++)
    printf("\t%f\n", sparse_obj->values[i]);

}

/* ---------- Accessing Matrix Rows ---------- */

int mat_get_row_chunk(struct __mv_sparse *mat_A, int row_id, int chunk_start, int chunk_end, double *p_row)
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
  double dp_res = 0.0, g_dp_res = 0.0;
  int i = 0;

  if(!vec_a || !vec_b)
    return -1.0;

  if(vec_a->size != vec_b->size)
    return -1.0;

  for(i = 0; i < vec_a->nval; i++)
    dp_res += vec_a->values[i] * vec_b->values[i];

  MPI_Reduce(&dp_res, &g_dp_res, 1, MPI_DOUBLE
             , MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

  return g_dp_res;
}

int sv_mult(double sca, struct __mv_sparse *vec_a, struct __mv_sparse **vec_r)
{
  int i = 0;
  
  if(!vec_a)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(vec_a->nval, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->nval * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
  }

  for(i = 0; i < vec_a->nval; i++)
    (*vec_r)->values[i] = sca * vec_a->values[i];

  return 0;
}

int mv_mult(struct __mv_sparse *mat_A, struct __mv_sparse *vec_b, struct __mv_sparse **vec_r)
{
  double *curr_chunk = NULL;
  int i = 0, j = 0, c = 0;

  double el_dp_res = 0.0;
  double row_dp_res = 0.0;
  double chunk_db_res = 0.0;

  if(!mat_A || !vec_b)
    return -1;

  if(mat_A->size != vec_b->size)
    return -1;

  if(*vec_r == NULL) {
    *vec_r = new_mv_struct();
    (*vec_r)->values = (double *)calloc(mat_A->nval, sizeof(double));
    (*vec_r)->size = mat_A->size;
    (*vec_r)->nnz = vec_b->nnz;
    (*vec_r)->start = vec_b->start;
    (*vec_r)->end = vec_b->end;
    (*vec_r)->nval = vec_b->nval;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, mat_A->nval * sizeof(double));
    bzero((*vec_r)->values, mat_A->nval * sizeof(double));
    (*vec_r)->size = mat_A->size;
    (*vec_r)->nnz = vec_b->nnz;
    (*vec_r)->start = vec_b->start;
    (*vec_r)->end = vec_b->end;
    (*vec_r)->nval = vec_b->nval;
  }

  curr_chunk = (double *)calloc(vec_b->nval, sizeof(double));

  /* executed per row */
  for(i = 0; i < (mat_A->end - mat_A->start); i++) {
    row_dp_res = 0.0;

    /* executed per chunk */
    for(c = 0; c < mat_A->nval; c++) {
      el_dp_res = 0.0;
      chunk_dp_res = 0.0;

      bzero(curr_chunk, vec_b->nval * sizeof(double));
      mat_get_row_chunk(mat_A, i + mat_A->start, mat_A->start, mat_A->end, curr_chunk);

      /* executed per element */
      for(j = 0; j < vec_b->nval; j++) {
        el_dp_res += curr_chunk[j] * vec_b->values[j];
      } /* end element */

      MPI_Reduce(&el_dp_res, &chunk_dp_res, 1, MPI_DOUBLE
                 , MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);

      row_dp_res += chunk_dp_res;
    } /* end chunk */

    (*vec_r)->values[i] = row_dp_res;
  } /* end row */

  free(curr_chunk);

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
    (*vec_r)->values = (double *)calloc(vec_a->nval, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->nval * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
  }

  for(i = 0; i < vec_a->nval; i++)
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
    (*vec_r)->values = (double *)calloc(vec_a->nval, sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->col_indices = NULL;
    (*vec_r)->row_ptr = NULL;
  } else {
    (*vec_r)->values = (double *)realloc((*vec_r)->values, vec_a->nval * sizeof(double));
    (*vec_r)->size = vec_a->size;
    (*vec_r)->nnz = vec_a->nnz;
    (*vec_r)->start = vec_a->start;
    (*vec_r)->end = vec_a->end;
    (*vec_r)->nval = vec_a->nval;
  }

  for(i = 0; i < vec_a->nval; i++)
    (*vec_r)->values[i] = vec_a->values[i] - vec_b->values[i];

  return 0;
}
