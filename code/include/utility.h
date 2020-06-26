
#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <mpi.h>

#define GIGA 1000000000.
#define MSEC 1000.
#define NSEC 1000000000.0

int lcm(int a, int b);
float* rnd_flt_matrix(int n, int m);
float* zeros_flt_matrix(int n, int m);
void cp_matrix(float *S, float *D, int n, int m, int lds, int ldd);
void clear_matrix(float* A, int lda, int n, int m);
int cmp_matrix(float* A, float* B, int lda, int ldb, int n, int m);
void print_matrix(float* A, int lda, int n, int m);
double time_elapsed(struct timespec a, struct timespec b);
void init_mpi_env(int argc, char** argv, int *nproc, int *menum);
void create_cart_grid(int row, int col, int *coords, MPI_Comm* CART_COMM);
void cart_sub(MPI_Comm CART_COMM, MPI_Comm* ROW_COMM,MPI_Comm* COL_COMM);
void cyclic_distribution(MPI_Comm CART_COMM, int menum, int sender, 
                         int Sr, int Sc, int n, int m, int p,
                         int lda, int ldb, int ldc,
                         float *A, float *B, float *C,
                         float **locA, float **locB, float **locC);
void gather_result(MPI_Comm CART_COMM, int Sr, int Sc, int receiver,
                   int n, int p, int ldc, float* C, float *locC);

#endif // utility.h included
