/* cu_ops.h - CUDA helper functions
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 2 March 2012
 *
 */

#include <cuda_runtime_api.h>
#include "mv_ops.h"

struct __mv_sparse *cgCopyVector(struct __mv_sparse *);
struct __mv_sparse *cgCopyMatrix(struct __mv_sparse *);

/* Helper kernels */
__global__ void cgPrintValues(double *d_values);
