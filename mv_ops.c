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

struct __mv_sparse *mv_shallow_copy(struct __mv_sparse *orig)
{
  struct __mv_sparse *mv_sparse = NULL;

  mv_sparse = new_mv_struct();

  mv_sparse->size = orig->size;
  mv_sparse->nnz = orig->nnz;
  mv_sparse->nval = orig->nval;
  mv_sparse->start = orig->start;
  mv_sparse->end = orig->end;

  mv_sparse->values = (double *)calloc(orig->nval, sizeof(double));
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
  cp->nval = orig->nval;
  cp->start = orig->start;
  cp->end = orig->end;

  cp->values = (double *)calloc(cp->nval, sizeof(double));
  memcpy(cp->values, orig->values, cp->nval * sizeof(double));

  if(orig->col_indices == NULL) {
    cp->col_indices = NULL;
  } else {
    cp->col_indices = (int *)calloc(cp->nval, sizeof(int));
    memcpy(cp->col_indices, orig->col_indices, cp->nval * sizeof(int));
  }

  if(orig->row_ptr == NULL) {
    cp->row_ptr = NULL;
  } else {
    cp->row_ptr = (int *)calloc(cp->size / g_mpi_group_size, sizeof(int));
    memcpy(cp->row_ptr, orig->row_ptr, (cp->size / g_mpi_group_size) * sizeof(int));
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
  
  return d_vec;
}

struct __mv_sparse *gatherAll_vector(struct __mv_sparse *d_vec_x)
{
  struct __mv_sparse *vec_x = NULL;

  vec_x = new_mv_struct();
  vec_x->size = d_vec_x->size;
  vec_x->nnz = d_vec_x->nnz;
  vec_x->nval = d_vec_x->nval * g_mpi_group_size;
  vec_x->start = 0;
  vec_x->end = 0;
  vec_x->col_indices = NULL;
  vec_x->row_ptr = NULL;
  vec_x->values = (double *)calloc(vec_x->nval, sizeof(double));

  MPI_Allgather(d_vec_x->values, d_vec_x->nval, MPI_DOUBLE
                , vec_x->values, d_vec_x->nval, MPI_DOUBLE
                , MPI_COMM_WORLD);

  return vec_x;
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
  for(i = 0; i < 10; i++)
    printf("\t%f\n", sparse_obj->values[i]);

  printf("\tRow pointers: %p\n", sparse_obj->row_ptr);
  if(sparse_obj->row_ptr != NULL) {
    for(i = 0; i < 10; i++)
      printf("\t%d\n", sparse_obj->row_ptr[i]);
  }

  printf("\tColumn indices: %p\n", sparse_obj->col_indices);
  if(sparse_obj->col_indices != NULL) {
    for(i = 0; i < 10; i++)
      printf("\t%d\n", sparse_obj->col_indices[i]);
  }
}

/* ---------- Accessing Matrix Rows ---------- */
int mat_get_row(struct __mv_sparse *mat_A, int row_id, double *p_row)
{
  int ci, i;
  
  CG_PRINTF("mat_get_row -> mat_A: %p, row_id: %d, p_row: %p\n", mat_A, row_id, p_row);

  if(!mat_A || !p_row)
    return -1;


  //ci = mat_A->row_ptr[row_id] - (g_mpi_rank * mat_A->size);
  ci = mat_A->row_ptr[row_id] - mat_A->row_ptr[0];
  CG_PRINTF("mat_get_row -> mat_A->row_ptr[%d]: %d\n", row_id, mat_A->row_ptr[row_id]);
  CG_PRINTF("mat_get_row -> mat_A->size: %d\n", mat_A->size);
  CG_PRINTF("mat_get_row -> ci: %d\n", ci);
  
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

  for(i = 0; i < vec_a->nval; i++) {
    dp_res += vec_a->values[i] * vec_b->values[i];
  }

  MPI_Allreduce(&dp_res, &g_dp_res, 1, MPI_DOUBLE
             , MPI_SUM, MPI_COMM_WORLD);

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

int mv_mult(struct __mv_sparse *d_mat_A, struct __mv_sparse *d_vec_b, struct __mv_sparse **d_vec_r)
{
  struct __mv_sparse *vec_x = NULL;

  double *curr_row = NULL;
  int i = 0, j = 0;

  double dp_res = 0.0;

  CG_PRINTF("mv_mult -> d_mat_A: %p, d_vec_b: %p, *d_vec_r: %p\n", d_mat_A, d_vec_b, *d_vec_r);

  if(!d_mat_A || !d_vec_b)
    return -1;

  if(d_mat_A->size != d_vec_b->size)
    return -1;

  if(*d_vec_r == NULL) {
    *d_vec_r = new_mv_struct();
    (*d_vec_r)->values = (double *)calloc(d_mat_A->nval, sizeof(double));
    (*d_vec_r)->size = d_mat_A->size;
    (*d_vec_r)->nnz = d_vec_b->nnz;
    (*d_vec_r)->start = d_vec_b->start;
    (*d_vec_r)->end = d_vec_b->end;
    (*d_vec_r)->nval = d_vec_b->nval;
    (*d_vec_r)->row_ptr = NULL;
  } else {
    (*d_vec_r)->values = (double *)realloc((*d_vec_r)->values, d_mat_A->nval * sizeof(double));
    bzero((*d_vec_r)->values, d_mat_A->nval * sizeof(double));
    (*d_vec_r)->size = d_mat_A->size;
    (*d_vec_r)->nnz = d_vec_b->nnz;
    (*d_vec_r)->start = d_vec_b->start;
    (*d_vec_r)->end = d_vec_b->end;
    (*d_vec_r)->nval = d_vec_b->nval;
  }

  curr_row = (double *)calloc(d_vec_b->size, sizeof(double));

  CG_PRINT("mv_mult: going to gatherAll\n");
  vec_x = gatherAll_vector(d_vec_b);
  CG_PRINT("mv_mult: finished gatherAll\n");

  int nrow = d_mat_A->end - d_mat_A->start;
  CG_PRINTF("mv_mult -> nrow: %d\n", nrow);
  for(i = 0; i < nrow; i++) {
    dp_res = 0.0;
    CG_PRINTF("processing row %d\n", i);

    //bzero(curr_row, d_vec_b->nval * sizeof(double));
    mat_get_row(d_mat_A, i, curr_row);
    CG_PRINTF("mv_mult -> got row %d\n", i);
    
    for(j = 0; j < vec_x->size; j++) {
      dp_res += curr_row[j] * vec_x->values[j];
    }
    
    (*d_vec_r)->values[i] = dp_res;
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
