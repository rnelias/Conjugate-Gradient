/* mv_ops.h - Matrix/Vector Operations
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 10 February 2012
 *
 */

#ifndef __MV_OPS_H__
#define __MV_OPS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define ROOT_RANK 0

#ifdef DEBUG
#define CG_PRINT(str) printf("[rank %d] " str, g_mpi_rank)
#define CG_PRINTF(fmt,...) printf("[rank %d] " fmt, g_mpi_rank, __VA_ARGS__)
#else
#define CG_PRINT(str)
#define CG_PRINTF(fmt,...)
#endif

struct __mv_sparse {
  int size;
  int nnz;
  int start;
  int end;
  int nval;
  double *values;
  int *col_indices;
  int *row_ptr;
};

/* Creating & Destroying Sparse Objects */
struct __mv_sparse *new_mv_struct();
void free_mv_struct(struct __mv_sparse *);
struct __mv_sparse *mv_shallow_copy(struct __mv_sparse *);
struct __mv_sparse *mv_deep_copy(struct __mv_sparse *);

/* Distribution */
struct __mv_sparse *distribute_matrix(struct __mv_sparse *);
struct __mv_sparse *distribute_vector(struct __mv_sparse *);
struct __mv_sparse *gather_vector(struct __mv_sparse *);
struct __mv_sparse *gatherAll_vector(struct __mv_sparse *);

/* Printing */
void print_sparse(struct __mv_sparse *, const char *);

/* Accessing Matrix Rows */
int mat_get_row(struct __mv_sparse *, int, double *);

/* Arithmetic Operations */
double dot_product(struct __mv_sparse *, struct __mv_sparse *);
int sv_mult(double, struct __mv_sparse *, struct __mv_sparse **);
int mv_mult(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);
int vec_add(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);
int vec_sub(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);


#endif /* __MV_OPS_H__ */
