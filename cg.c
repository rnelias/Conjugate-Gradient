/* cg.c - Conjugate Gradient, in all it's glory!
 *
 * CSC 564 - CG Assignment Series
 *
 * Author: Dean Pucsek
 * Date: 10 February 2012
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#include "mv_ops.h"

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

#ifdef DEBUG
#define CG_PRINT(str) printf("[rank %d] " str, g_mpi_rank)
#define CG_PRINTF(fmt,...) printf("[rank %d] " fmt, g_mpi_rank, __VA_ARGS__)
#else
#define CG_PRINT(str)
#define CG_PRINTF(fmt,...)
#endif

/* ---------- Function Declarations ---------- */
int read_input_file(const char *, struct __mv_sparse *, struct __mv_sparse *);
struct __mv_sparse *conj_grad(int, struct __mv_sparse *, struct __mv_sparse *);
int test_mv_ops(struct __mv_sparse *mat_A, struct __mv_sparse *vec_b);

/* ---------- Storing data ---------- */
void storeColumnIndex(int **col_indices, int ci);
void storeRowPointer(int **row_ptr, int rp);
void storeValue(double **values, double value);

/* ---------- Simple "Stack" ---------- */
static unsigned char *g_stack;
static int g_stack_ptr;
void stack_initialize();
void stack_finalize();
void stack_push(unsigned char c);
void stack_clear();
int stack_to_int();
double stack_to_double();

/* ---------- Macros ---------- */
#define ROOT_RANK           0
#define BEGIN_ROOT_SECTION  if(g_mpi_rank == ROOT_RANK) {
#define END_ROOT_SECTION    }

/* ---------- Global Variables ---------- */
int g_mpi_rank = -1;
int g_mpi_group_size = -1;

int main(int argc, char **argv)
{
  const char *input_file = NULL;
  int max_iterations = -1, no_output = FALSE;
  int d_max_iter = -1;

  double start_time = 0.0;
  double end_time = 0.0;

  struct __mv_sparse *mat_A = NULL;
  struct __mv_sparse *vec_b = NULL;
  struct __mv_sparse *vec_x = NULL;

  struct __mv_sparse *d_mat_A = NULL;
  struct __mv_sparse *d_vec_b = NULL;
  struct __mv_sparse *d_vec_x = NULL;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_group_size);

  BEGIN_ROOT_SECTION

  /* Process command arguments */
  if(argc < 3) {
    fprintf(stderr, "Usage: %s <input-data> <max-iterations> [suppress-output]\n", argv[0]);
    return -1;
  }

  input_file = argv[1];
  max_iterations = (int)strtol(argv[2], NULL, 10);
  
  if(argc == 4)
    no_output = argv[3][0] == 'y' ? TRUE : FALSE;

  /* Initialize matrix A and vector b */
  mat_A = new_mv_struct();
  vec_b = new_mv_struct();

  d_max_iter = max_iterations;


  /* Read input */
  CG_PRINT("reading input... ");
  read_input_file(input_file, mat_A, vec_b);
  CG_PRINT("done\n");

  END_ROOT_SECTION

  omp_set_num_threads(19);

  CG_PRINTF("number of MPI nodes: %d\n", g_mpi_group_size);
  CG_PRINTF("max OMP threads: %d\n", omp_get_max_threads());

  MPI_Bcast(&d_max_iter, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD);

  d_mat_A = distribute_matrix(mat_A);
  d_vec_b = distribute_vector(vec_b);
  
  /* Test MV Ops */
  //test_mv_ops(d_mat_A, d_vec_b);

  /* Compute CG */
  if(g_mpi_rank == ROOT_RANK)
    start_time = MPI_Wtime();

  CG_PRINT("going to CG\n");
  d_vec_x = conj_grad(d_max_iter, d_mat_A, d_vec_b);

  if(g_mpi_rank == ROOT_RANK)
    end_time = MPI_Wtime();  

  vec_x = gather_vector(d_vec_x);

  BEGIN_ROOT_SECTION

  printf("CG took approx %f seconds\n", end_time - start_time);

  /* Print result */
  print_sparse(vec_x, "vec_x (result)");

  /* Clean Up */
  free_mv_struct(mat_A);
  free_mv_struct(vec_b);

  END_ROOT_SECTION

  MPI_Finalize();

  return 0;
}

/* ---------- Conjugate Gradient ---------- */
struct __mv_sparse *conj_grad(int max_iter, struct __mv_sparse *mat_A, struct __mv_sparse *vec_b)
{
  int k;
  struct __mv_sparse *vec_s, *vec_r, *vec_prev_r, *vec_p;
  struct __mv_sparse *sv_res, *vx_temp;
  struct __mv_sparse *vx_old, *vx_new;
  double sca_alpha, sca_beta;

  sv_res = NULL;
  vx_temp = NULL;
  vec_s = NULL;
  vec_r = NULL;
  vec_prev_r = NULL;
  vec_p = NULL;

  vx_old = NULL;
  vx_new = mv_shallow_copy(vec_b);

  k = 0;                           /* current number of iterations */
  vec_r = mv_deep_copy(vec_b);     /* residual */
  vec_p = mv_deep_copy(vec_r);

  while(TRUE) {
    CG_PRINTF("iteration %d\n", k);

    mv_mult(mat_A, vec_p, &vec_s);
    CG_PRINT("mv_mult\n");

    sca_alpha = dot_product(vec_r, vec_r) / dot_product(vec_p, vec_s);
    sv_mult(sca_alpha, vec_p, &sv_res);
    CG_PRINT("sv_mult 1\n");

    vx_old = mv_deep_copy(vx_new);

    vec_add(vx_old, sv_res, &vx_new);
    CG_PRINT("vec_add 1\n");

    if(k == max_iter) {
      break;
    }

    vec_prev_r = mv_deep_copy(vec_r);

    sv_mult(sca_alpha, vec_s, &sv_res);
    CG_PRINT("sv_mult 2\n");

    vec_sub(vec_r, sv_res, &vec_r);

    sca_beta = dot_product(vec_r, vec_r) / dot_product(vec_prev_r, vec_prev_r);

    sv_mult(sca_beta, vec_p, &sv_res);
    CG_PRINT("sv_mult 3\n");

    vec_add(vec_r, sv_res, &vec_p);
    CG_PRINT("vec_add 2\n");

    k++;
  }

  return vx_new;
}


/* ---------- Reading Input Data ---------- */

int read_input_file(const char *input_file, struct __mv_sparse *mat_A, struct __mv_sparse *vec_b)
{
  unsigned char ch;
  int curr_line = 0;
  int A_size = 0, A_nnz = 0, b_size = 0;

  FILE *input_fp = NULL;

  double *A_values = NULL;
  double *b_values = NULL;
  int *col_indices = NULL;
  int *row_ptr = NULL;

  if(mat_A == NULL || vec_b == NULL) {
    fprintf(stderr, "Error: Matrix must be initialized before reading input\n");
    return -1;
  }

  /* Initialize a stack */
  stack_initialize();

  input_fp = fopen(input_file, "r");
  if(!input_fp) {
    fprintf(stderr, "Error: Failed to open input file (%s)\n", input_file);
    return -1;
  }

  while(TRUE) {
    ch = fgetc(input_fp);

    if(ch == ',' || ch == '\n') {

      /* Save the current element in the appropriate place */
      switch(curr_line) {
      case 0: storeColumnIndex(&col_indices, stack_to_int()); break;
      case 1: A_size++; storeRowPointer(&row_ptr, stack_to_int()); break;
      case 2: A_nnz++; storeValue(&A_values, stack_to_double()); break;
      case 3: b_size++; storeValue(&b_values, stack_to_double()); break;
      default: fprintf(stderr, "Line out of bounds (%d)\n", curr_line);
      }

      /* Zero the stack */
      stack_clear();

      if(ch == '\n') {
        curr_line++;
        
        if(curr_line == 4)
          break;
      }
    } else {

      /* We have an actual character, save it temporarily */
      stack_push(ch);
    }
  }

  /* Now set the various fields of the output matrix */
  mat_A->size = A_size-1;
  mat_A->nnz = A_nnz;
  mat_A->values = A_values;
  mat_A->col_indices = col_indices;
  mat_A->row_ptr = row_ptr;

  vec_b->size = b_size;
  vec_b->nnz = b_size;
  vec_b->values = b_values;

  stack_finalize();

  /* No errors, return */
  return 0;
}

/* function: storeColumnIndex
 * in: int **col_indices - pointer to location of the column index array
 *     int ci - column index to store
 * out: none
 * description: store the column index
 */
void storeColumnIndex(int **col_indices, int ci)
{
  /* A note about the 'magic' numbers:
   *
   * There should be one index for every non-zero element in the 
   * matrix.  Since the matrix contains approximately 18020000 non-zero
   * elements, the max has been set to that.
   */
  
  static int ci_max = 18020000;
  static int ci_cntr = 0;

  if(*col_indices == NULL) {
    *col_indices = (int *)calloc(ci_max, sizeof(int));
  }

  if(ci_cntr == ci_max) {
    ci_max += 1000;
    *col_indices = (int *)realloc(*col_indices, ci_max * sizeof(int));
  }

  (*col_indices)[ci_cntr++] = ci;
}

/* function: storeRowPointer
 * in: int **row_ptr - pointer to the location of the row pointer array
 *     int rp - the row pointer to store
 * out: none
 * description: store the row pointers
 */
void storeRowPointer(int **row_ptr, int rp)
{
  /* A note about the 'magic' numbers:
   * 
   * The full matrix is about 52269 by 52269 and there should
   * be num_rows+1 entries in the row pointer array.  I've allocated
   * extra space to be safe.
   */

  static int rp_max = 52300;
  static int rp_cntr = 0;

  if(*row_ptr == NULL) {
    *row_ptr = (int *)calloc(rp_max, sizeof(int));
  }

  if(rp_max == rp_cntr) {
    rp_max += 1000;
    *row_ptr = (int *)realloc(*row_ptr, rp_max * sizeof(int));
  }

  (*row_ptr)[rp_cntr++] = rp;
}

/* function: storeValue
 * in: double **valuse - pointer to the location of the values array
 *     double value - the value to be stored
 * out: none
 * description: store the given value in the array
 */
void storeValue(double **values, double value)
{
  /* A note about the 'magic' numbers:
   * 
   * See storeColumnIndex
   */

  static int val_max = 18020000;
  static int val_cntr = 0;

  if(*values == NULL) {
    val_cntr = 0;
    *values = (double *)calloc(val_max, sizeof(double));
  }

  if(val_max == val_cntr) {
    val_max += 1000;
    *values = (double *)realloc(*values, val_max * sizeof(double));
  }

  (*values)[val_cntr++] = value;
}


/* ---------- Mini Stack! ---------- */

/* function: stack_initialize
 * in: none
 * out: none
 * description: initializes a (global) stack used to process input
 */
void stack_initialize()
{
  /* Allocate 64 bytes for a "stack" */
  g_stack = (unsigned char *)calloc(64, sizeof(unsigned char*));
  if(!g_stack) {
    fprintf(stderr, "ERROR: Can't allocate enough memory\n");
    exit(-1);
  }

  g_stack_ptr = 0;
}

void stack_finalize()
{
  if(g_stack == NULL)
    return;
  
  free(g_stack);
}

void stack_push(unsigned char c) 
{
  g_stack[g_stack_ptr++] = c;
}

void stack_clear()
{
  bzero(g_stack, 64);
  g_stack_ptr = 0;
}

int stack_to_int() 
{
  return (int)strtol((char *)g_stack, NULL, 10);
}

double stack_to_double() 
{
  return strtod((char *)g_stack, NULL);
}

int test_mv_ops(struct __mv_sparse *mat_A, struct __mv_sparse *vec_b)
{
  struct __mv_sparse *vec_x = NULL;
  struct __mv_sparse *d_vec_x = NULL;

  /*
  print_sparse(mat_A, "d_mat_A");
  print_sparse(vec_b, "d_vec_b");
  */

  /*
  vec_x = gatherAll_vector(vec_b);
  print_sparse(vec_x, "vec_x (gatherAll_vector)");
  */

  mv_mult(mat_A, vec_b, &d_vec_x);
  vec_x = gather_vector(d_vec_x);
  print_sparse(vec_x, "vec_x (mv_mult)");

  /*
  sv_mult(4.0, vec_b, &d_vec_x);
  vec_x = gather_vector(d_vec_x);
  print_sparse(vec_x, "vec_x (sv_mult)");

  printf("[rank %d] dot product: %f\n", g_mpi_rank, dot_product(vec_b, vec_b));

  vec_add(vec_b, vec_b, &d_vec_x);
  vec_x = gather_vector(d_vec_x);
  print_sparse(vec_x, "vec_x (vec_add)");

  vec_sub(vec_b, vec_b, &d_vec_x);
  vec_x = gather_vector(d_vec_x);
  print_sparse(vec_x, "vec_x (vec_sub)");
  */

  return 0;
}
