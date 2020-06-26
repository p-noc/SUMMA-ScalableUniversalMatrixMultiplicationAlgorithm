#include "include/matmat.h"

#define MAT_SIZE 10080

int main(int argc, char** argv)
{
    int menum, nproc, coords[2], Sr, Sc, 
        ntrow, ntcol, bs;
    
    int ldlocA, ldlocB, ldlocC;

    long nop;

    double start, stop, proc_time, mpi_time, 
           tot_mpi_time, gflops_mpi;

    float *A, *B, *C, *C2, *locA, *locB, *locC;

    struct timespec start_timer_mpi, finish_timer_mpi;

    MPI_Comm CART_COMM, ROW_COMM, COL_COMM;

    init_mpi_env(argc, argv, &nproc, &menum);

    bs = 720;

    if (menum == 0)
    {
        if( argc > 4)
        {
            Sr = atoi(argv[1]);
            Sc = atoi(argv[2]);
            ntrow = atoi(argv[3]);
            ntcol = atoi(argv[4]);
        }
        else
            MPI_Abort(MPI_COMM_WORLD, -1);
        
        // inizializzazione con valori random delle matrici come vettori        
        A = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
        B = rnd_flt_matrix(MAT_SIZE, MAT_SIZE);
        // matrice risultatnte inizializzato con zeri
        C = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);
        // DEBUG matrix for check result
        //C2 = zeros_flt_matrix(MAT_SIZE, MAT_SIZE);
    }

    MPI_Bcast(&Sr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Sc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // creazione griglia
    create_cart_grid(Sr, Sc, coords, &CART_COMM);
    cart_sub(CART_COMM, &ROW_COMM, &COL_COMM);
    
    if(menum == 0)
    {
        printf("proc %d (%d, %d)- threads %d (%d, %d)\n", 
                nproc, Sr, Sc, ntrow*ntcol, ntrow, ntcol);
        printf("Matrix order; Time; GFlops\n");
    }
    
    for (int i = 1680; i <= 5040; i += 1680)
    {
        tot_mpi_time = 0.;

        ldlocA = i/Sc;
        ldlocB = i/Sc;
        ldlocC = i/Sc;

        // allocazione matrici locali 
        // e distribuzione ciclica delle matrici A e B  
        cyclic_distribution(CART_COMM, menum, 0, Sr, Sc, i, i, i, 
                            MAT_SIZE, MAT_SIZE, MAT_SIZE, A, B, C,
                            &locA, &locB, &locC);

        MPI_Barrier(MPI_COMM_WORLD);
        clock_gettime(CLOCK_MONOTONIC, &start_timer_mpi);
        
        // chiamata alla funzione SUMMA                
        SUMMA(ROW_COMM, COL_COMM, Sr, Sc, ntrow, ntcol, 
                ldlocA, ldlocB, ldlocC, i, i, i, locA, locB, locC, bs);
        
        clock_gettime(CLOCK_MONOTONIC, &finish_timer_mpi);
        proc_time = time_elapsed(start_timer_mpi, finish_timer_mpi);

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(&proc_time, &mpi_time, 1, MPI_DOUBLE, 
                   MPI_MAX, 0, MPI_COMM_WORLD);

        // mette insieme i risultati locali
        gather_result(CART_COMM, Sr, Sc, 0, i, i, MAT_SIZE, C, locC);

        if(menum == 0)
            tot_mpi_time += mpi_time;
        
        free(locA);
        free(locB);
        free(locC); 
        
        if (menum == 0)
        {
            nop = 2 * pow(i, 3);          
            gflops_mpi = (nop / tot_mpi_time) / GIGA;
            printf("%d; %8.3lf; %8.3lf;\n", i, tot_mpi_time, gflops_mpi);
            // -- DEBUG check result (time consuming) -- //
            // matmat_threads(1,1, MAT_SIZE, MAT_SIZE, MAT_SIZE,
            //                i, i, i, A, B, C2, 150);
            // long errors = cmp_matrix(C, C2, MAT_SIZE, MAT_SIZE, i, i);
            // printf("error: %ld\n", errors);
            // clear_matrix(C2, MAT_SIZE, i, i);
        }
    }
    
    if (menum == 0)
    {
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();

    return 0;
}

