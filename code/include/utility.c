
#include "utility.h"

int lcm(int a, int b)
{
    int lcm = (a > b) ? a : b;
    
    while(lcm % a || lcm % b)
		++lcm;

    return lcm;
}

float* rnd_flt_matrix(int n, int m)
{   
    int size = n * m;  
    float *A = (float*) malloc(sizeof(float) * size);

    for(int i = 0; i < size; ++i)
        A[i] = (float) rand() / RAND_MAX;
    
    return A;
}

float* zeros_flt_matrix(int n, int m)
{
    return (float*) calloc(n * m, sizeof(float));
}

void clear_matrix(float* A, int ld, int n, int m)
{  
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            A[i*ld+j] = 0;
}

int cmp_matrix(float *A, float *B, int lda, int ldb, int n, int m)
{
    int cnt = 0;
    
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < m; ++j)
            if(A[i*lda+j] != B[i*ldb+j])
                cnt++;

    return cnt;
}

void print_matrix(float *A, int lda, int n, int m)
{
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < m; ++j)
            printf("%5.0lf ", A[i*lda+j]);

        putchar('\n');
    }
}

double time_elapsed(struct timespec start, struct timespec finish)
{
    double elapsed = (finish.tv_sec - start.tv_sec);
    return elapsed + (finish.tv_nsec - start.tv_nsec) / NSEC;
}

void init_mpi_env(int argc, char** argv, int *nproc, int *menum)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, menum); // get processor id
    MPI_Comm_size(MPI_COMM_WORLD, nproc); // get number of processors
}

void create_cart_grid(int row, int col, int *coords, MPI_Comm* CART_COMM)
{
    int menum_cart;

    MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){row, col}, 
                    (int[]){1, 1}, 1, CART_COMM);

    MPI_Comm_rank(*CART_COMM, &menum_cart);   

    MPI_Cart_coords(*CART_COMM, menum_cart, 2, coords);
}

void cart_sub(MPI_Comm CART_COMM, MPI_Comm* ROW_COMM,MPI_Comm* COL_COMM)
{
    MPI_Cart_sub(CART_COMM, (int[]){0, 1}, ROW_COMM);
    MPI_Cart_sub(CART_COMM, (int[]){1, 0}, COL_COMM);
}

void cp_matrix(float *S, float *D, int n, int m, int lds, int ldd)
{
    for(int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            D[i*ldd+j] = S[i*lds+j];
}

void cyclic_distribution(MPI_Comm CART_COMM, int menum, int source, 
                         int Sr, int Sc, int n, int m, int p,
                         int lda, int ldb, int ldc,
                         float *A, float *B, float *C,
                         float **locA, float **locB, float **locC)
{
    float *bufferA, *bufferB, *tmpA, *tmpB, *tmpC;

    int dest;
    int local_offset, offsetA, offsetB;

    /* calcolo della dimensione delle matrici locali */

    // dimensione dei singoli blocchi
    int n_size = n / Sr;    // numero di righe
    int p_size = p / Sc;    // numero di colonne

    // calcolo del numero di blocchi
    // sulle colonne di A e sulle righe di B
    int blk_num = lcm(Sr, Sc);
    int proc_blk_a = blk_num / Sc;
    int proc_blk_b = blk_num / Sr;

    // numero di colonne di un blocco di A
    // equivalente al numero di righe di un blocco di B
    int m_size = m / blk_num;

    int blk_size_a = n_size * m_size;
    int blk_size_b = m_size * p_size;
    int blk_size_c = n_size * p_size;
    
    // allocazione matrici locali    
    tmpA = (float*) malloc(sizeof(float) * 
                          (n_size * m_size) * proc_blk_a);

    tmpB = (float*) malloc(sizeof(float) *
                          (m_size * proc_blk_b) * p_size);

    tmpC = (float*) calloc(blk_size_c, sizeof(float));
   
    // allocazione matrici di supporto per l'invio
    bufferA = (float*) malloc(sizeof(float) * blk_size_a);
    bufferB = (float*) malloc(sizeof(float) * blk_size_b);

    /* distribuzione ciclica delle matrici */ 

    local_offset = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    // distribuzione di A
    for (int i = 0; i < Sr; i++)
    {
        for (int j = 0; j < blk_num; j++)
        {
            // calcolo id destinatario
            dest = (i * Sc) + (j % Sc);
            offsetA = (i * lda * n_size) + (j * m_size);

            if (menum == source)
            { 
                if(dest == source)
                {
                    cp_matrix(&A[offsetA], &tmpA[local_offset*m_size], n_size, m_size, lda, m_size * proc_blk_a);
                    local_offset++; 
                }
                else 
                {
                    cp_matrix(&A[offsetA], bufferA, n_size, m_size, lda, m_size);
                    MPI_Send(bufferA, blk_size_a, MPI_FLOAT, dest, 1, CART_COMM);
                }  
            }
            else if (menum == dest)
            {
                MPI_Recv(bufferA, blk_size_a, MPI_FLOAT, source, 1, CART_COMM, MPI_STATUS_IGNORE);                   
                cp_matrix(bufferA, &tmpA[local_offset*m_size], n_size, m_size, m_size, m_size * proc_blk_a);
                local_offset++;
            }
        }              
    }
    
    local_offset = 0;
    
    // distrubuzione di B
    for (int i = 0; i < blk_num; i++)
    {
        for (int j = 0; j < Sc; j++)
        {
            // calcolo id destinatario
            dest = ((i % Sr)*Sc) + j;
            offsetB = (i * ldb * m_size) + (j * p_size);

            if(menum == source)
            {
                if(dest == source)
                {
                    cp_matrix(&B[offsetB], &tmpB[local_offset*blk_size_b], m_size, p_size, ldb, p_size);
                    local_offset++;
                }
                else
                {
                    cp_matrix(&B[offsetB], bufferB, m_size, p_size, ldb, p_size);
                    MPI_Send(bufferB, blk_size_b, MPI_FLOAT, dest, 1, CART_COMM);
                } 
            }
            else if(menum == dest)
            {
                MPI_Recv(bufferB, blk_size_b, MPI_FLOAT, source, 1, CART_COMM, MPI_STATUS_IGNORE);
                cp_matrix(bufferB, &tmpB[local_offset*blk_size_b], m_size, p_size, p_size, p_size);
                local_offset++;
            }  
        }              
    }

    *locA = tmpA;
    *locB = tmpB;
    *locC = tmpC;

    free(bufferA);
    free(bufferB);
}

void gather_result(MPI_Comm CART_COMM, int Sr, int Sc, int dest,
                   int n, int p, int ldc, float* C, float *locC)
{
    int menum, source, offset = 0;
    int n_size = n / Sr, p_size = p / Sc,
        c_blk_size = n_size * p_size;

    MPI_Comm_rank(CART_COMM, &menum);
    float *tmp = (float*) malloc(sizeof(float) * c_blk_size);

    for(int i = 0; i < Sr; i++)
    {
        for(int j = 0; j < Sc; j++)
        {
            source = i*Sc + j;
            offset = (i*n_size*ldc) + j*p_size;

            if(menum == dest && source == menum)
            {
                cp_matrix(locC, &C[offset], n_size, p_size, p_size, ldc);
            }
            else if(menum != dest && source == menum)
            {
                cp_matrix(locC, tmp, n_size, p_size, p_size, p_size);
                MPI_Send(tmp, c_blk_size, MPI_FLOAT, dest, 1, CART_COMM);
            }
            else if (menum == dest)
            {
                MPI_Recv(tmp, c_blk_size, MPI_FLOAT, source, 1, CART_COMM, MPI_STATUS_IGNORE);  
                cp_matrix(tmp, &C[offset], n_size, p_size, p_size, ldc);
            }
        }
    }

    free(tmp);
}


