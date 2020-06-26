
#include "matmat.h"

void matmat(int lda, int ldb, int ldc, int n, int m, int p, 
            float* A, float* B, float* C)
{
    for(int i = 0; i < n; ++i)
        for(int k = 0; k < m; ++k)        
            for(int j = 0; j < p; ++j)
                C[i*ldc+j] += A[i*lda+k] * B[k*ldb+j];
}

void matmat_block(int lda, int ldb, int ldc, int n, int m, int p, 
                  float* A, float* B, float* C, int bs)
{
    int i_n, k_m, j_p;
    
    for(int i = 0; i < n; i += bs)
        for(int k = 0; k < m; k += bs)        
            for(int j = 0; j < p; j += bs)
            {
                i_n = (i+bs > n) ? n-i : bs;
                k_m = (k+bs > m) ? m-k : bs;
                j_p = (j+bs > p) ? p-j : bs;

                matmat(lda, ldb, ldc, i_n, k_m, j_p, 
                       &A[i*lda+k], &B[k*ldb+j], &C[i*ldc+j]);
            }
}

void matmat_threads(int ntrow, int ntcol, int lda, int ldb, int ldc, 
                    int n, int m, int p, float* A, float* B, float* C, int bs)
{
    int TH_N = ntrow * ntcol, th_cnt = 0;
    int sub_n, rest_n, sub_j, rest_j, span_n, span_j, offset_n, offset_j;
    pthread_t tid[TH_N];
    mat_struct th_arg[TH_N];

    sub_n  = n / ntrow;
    rest_n = n % ntrow;

    sub_j  = p / ntcol;
    rest_j = p % ntcol; 

    offset_n = 0;
  
    for(int i = 0; i < ntrow; ++i)
    {
        span_n = (i < rest_n) ? sub_n + 1 : sub_n;

        offset_j = 0;

        for(int j = 0; j < ntcol; ++j)
        {
            span_j = (j < rest_j) ? sub_j + 1: sub_j;

            th_arg[th_cnt].lda = lda;
            th_arg[th_cnt].ldb = ldb;
            th_arg[th_cnt].ldc = ldc;

            th_arg[th_cnt].n = span_n;
            th_arg[th_cnt].m = m;
            th_arg[th_cnt].p = span_j;

            th_arg[th_cnt].A = &A[offset_n * lda];
            th_arg[th_cnt].B = &B[offset_j];
            th_arg[th_cnt].C = &C[offset_n * ldc + offset_j];

            th_arg[th_cnt].bs = bs;

            pthread_create(&tid[th_cnt], NULL, 
                           matmat_thread, &th_arg[th_cnt]);
            th_cnt++;

            offset_j += span_j;
        }
        offset_n += span_n;
    }

    for(int i = 0; i < TH_N; ++i)
        pthread_join(tid[i], NULL);
}

void* matmat_thread(void* mat_th_struct)
{
    mat_struct *s = (mat_struct*) mat_th_struct;
    matmat_block(s->lda, s->ldb, s->ldc, s->n, s->m, s->p, 
                 s->A, s->B, s->C, s->bs);
}

void SUMMA(MPI_Comm ROW_COMM, MPI_Comm COL_COMM, int Sr, int Sc, 
           int ntrow, int ntcol, int lda, int ldb, int ldc,
           int n, int m, int p, float* locA, float* locB, float* Cloc, int bs)
{
    int c, r, id_col, id_row, 
        offsetA, offsetB;

    float *A_tmp, *B_tmp;

    int n_size = n / Sr;
    int p_size = p / Sc;

    int blk_num = lcm(Sr, Sc);
    int m_size = m / blk_num;

    int blk_size_a = n_size * m_size;
    int blk_size_b = m_size * p_size;

    A_tmp = (float*) malloc(sizeof(float) * blk_size_a);
    B_tmp = (float*) malloc(sizeof(float) * blk_size_b);

    MPI_Comm_rank(ROW_COMM, &id_row);
    MPI_Comm_rank(COL_COMM, &id_col);

    offsetA = offsetB = 0;
    
    for(int k = 0; k < blk_num; k++)
    {
        r = k % Sc;
        c = k % Sr;

        if(id_row == r)
        {
            cp_matrix(&locA[offsetA], A_tmp, n_size, m_size, lda, m_size);
            offsetA += m_size;
        }
           
        if(id_col == c)
        {
            cp_matrix(&locB[offsetB], B_tmp, m_size, p_size, ldb, p_size);
            offsetB += (m_size * ldb);
        }

        MPI_Bcast(A_tmp, blk_size_a, MPI_FLOAT, r, ROW_COMM);
        MPI_Bcast(B_tmp, blk_size_b, MPI_FLOAT, c, COL_COMM);
        
        matmat_threads(ntrow, ntcol, m_size, p_size, ldc, 
                       n_size, m_size, p_size, A_tmp, B_tmp, Cloc, bs);
    }
    
    free(A_tmp);
    free(B_tmp);
}

