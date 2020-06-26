
#ifndef __MATMAT_H__
#define __MATMAT_H__

#include "utility.h"

struct mat_struct
{
    int n, m, p;
    int lda, ldb, ldc;
    float *A, *B, *C;
    int bs;
};

typedef struct mat_struct mat_struct;

void matmat(int lda, int ldb, int ldc, int n, int m, int p, float* A, float* B, float* C);
void matmat_block(int lda, int ldb, int ldc, int n, int m, int p, float* A, float* B, float* C, int bs);
void matmat_threads(int ntrow, int ntcol, int lda, int ldb, int ldc, int n, int m, int p, float* A, float* B, float* C, int bs);
void* matmat_thread(void* mat_th_struct);
void SUMMA(MPI_Comm ROW_COMM, MPI_Comm COL_COMM, int Sr, int Sc, int ntrow, int ntcol, int lda, int ldb, int ldc, int n, int m, int p, float* A, float* B, float* C, int bs);

#endif // matmat.h included
