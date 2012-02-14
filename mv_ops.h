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

struct __mv_sparse {
  int size;
  int nnz;
  double *values;
  int *col_indices;
  int *row_ptr;
};

/* Creating & Destroying Sparse Objects */
struct __mv_sparse *new_mv_struct();
struct __mv_sparse *new_mv_struct_with_size(int);
void free_mv_struct(struct __mv_sparse *);
struct __mv_sparse *mv_deep_copy(struct __mv_sparse *);

/* Printing */
void print_sparse(struct __mv_sparse *);

/* Accessing Matrix Rows */
int mat_get_row(struct __mv_sparse *, int, double *);

/* Arithmetic Operations */
double dot_product(struct __mv_sparse *, struct __mv_sparse *);
int sv_mult(double, struct __mv_sparse *, struct __mv_sparse **);
int mv_mult(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);
int vec_add(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);
int vec_sub(struct __mv_sparse *, struct __mv_sparse *, struct __mv_sparse **);


#endif /* __MV_OPS_H__ */
