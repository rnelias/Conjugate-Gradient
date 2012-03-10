/* mv_types.cuh - Matrix and Vector types for the CUDA CG implementation
*
* Author: Dean Pucsek
* Date: 8 March 2012
*
*/

#ifndef __MV_TYPES_CUH__
#define __MV_TYPES_CUH__

struct __vector {
       int size;
       double *values;
};
typedef struct __vector Vector;

struct __matrix {
       int size;
       int nnz;
       double *values;
       int *column_indices;
       int *row_pointers;
};
typedef struct __matrix Matrix;

#endif /* __MV_TYPES_CUH__ */