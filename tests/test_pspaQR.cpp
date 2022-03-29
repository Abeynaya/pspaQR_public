#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <Eigen/IterativeLinearSolvers>

#include "cxxopts.hpp"
#include "spaQR.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]){

	// User-specified
    int n_threads = 2; // Set to number of cores per node

	// Initialize MPI
	int req = MPI_THREAD_FUNNELED;
	int prov = -1;
	MPI_Init_thread(NULL, NULL, req, &prov);
	const int my_rank = ttor::comm_rank();

	// Load and preprocess matrix
	string matrix = "../../spaQR/mats/lapl/neglapl_2_32.mm";
	SpMat A = mmio::sp_mmread<double, int>(matrix);
    A.makeCompressed();

    int nrows = A.rows();
    int ncols = A.cols();
    if (my_rank == 0) cout << "Matrix " << matrix << " with " << nrows << " rows,  " << ncols << " columns loaded" << endl;

    // Pre-process matrix to have columns of unit norm (diagonal scaling) -- important
    VectorXd Dentries(A.cols());
    DiagonalMatrix<double, Eigen::Dynamic> D(A.cols());

    auto pstart = systime();
    for (int i=0; i < A.cols(); ++i) {
        double sum = 0;
       for (SpMat::InnerIterator it(A,i); it; ++it){
            sum += it.value()*it.value();
       }
       Dentries[i] = (double)(1.0/sqrt(sum));
    }

    D = Dentries.asDiagonal();
    A = A*D*10;
    auto pend = systime();

    // Iterative method 
    bool useGMRES = (nrows == ncols) ? true : false;

    /**
     * Default parameters 
     */
    int nlevels = ceil(log2(ncols/64)); // Number of Nested Dissection levels
    int scale = 1; // Always 1
    int verbose = 0; // Set to 1 if more verbose output is needed from ttor
    int order = 1; // Only 1 will work for now
    int hsl = 0; // Set to 0 unless HSL matching library has been downloaded

    /**
     * Parameters to play around with
     */
    int skip = 1; // Skip levels before beginning sparsification (skip atleast 1)
    double tol = 1e-2; // Tolerance to set in the solver
    double residual = 1e-12; // Final residual desired by user
    int iterations = 100; // Max number of GMRES iterations

    /**
     * Setup parallel tree
     */ 
    ParTree t(nlevels, skip);
    t.set_scale(scale);
    t.set_tol(tol);
    t.set_order(order);
    t.set_hsl(hsl);
    t.set_nthreads(n_threads);
    t.set_verbose(verbose);

    if (nrows == ncols) t.set_square(1); // default 0


    /**
     * Partition, assemble and factorize
     */ 
    t.partition(A);
    // Setup
    t.assemble(A);
    
    // Factorize
    MPI_Barrier(MPI_COMM_WORLD);
    int err = t.factorize();
    MPI_Barrier(MPI_COMM_WORLD);

    /**
     * GMRES solves
     */

    bool verb = true; 
    int iter = 0;
    if (!err)
    {
        VectorXd b;

        VectorXd x = random(nrows, 2021); // Load RHS on all nodes
        VectorXd x_dist = VectorXd::Zero(t.nrows_local());
        
        t.distribute_x(x, x_dist); // Permutes and distributes x to all ranks 
        MPI_Barrier(MPI_COMM_WORLD);

        VectorXd b_dist = x_dist;
        if(useGMRES) {
            timer gmres0 = systime();
            iter = dist_gmres(A, x_dist, t, iterations, residual, verb);
            timer gmres1 = systime();
            const VectorXd Ax = t.spmv(x_dist);
            const VectorXd r = b_dist - Ax;
            const double rnorm = VecNorm(r);
            const double bnorm = VecNorm(b_dist);
            const double relres = rnorm/bnorm;
            if (my_rank == 0){
                cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << relres << endl;
                cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
                cout << "<<<<GMRES=" << iter << endl;
            }

            // Extract x on rank 0
            t.extract_x(x);
            // Post process x
            if (my_rank==0) x = 10*D*x;
        }
    }

    
    /**
     * With distributed x
     */
    /*
    {
        // VectorXd x = VectorXd::Zero(t.ncols_local());
        VectorXd x;
        x = random(t.nrows_local(),my_rank+2021); // distributed x
        VectorXd b = x;

        if(useGMRES) {
            timer gmres0 = systime();
            iter = dist_gmres(A, x, t, iterations, residual, verb);
            timer gmres1 = systime();
            const VectorXd Ax = t.spmv(x);
            const VectorXd r = b - Ax;
            const double rnorm = VecNorm(r);
            const double bnorm = VecNorm(b);
            const double relres = rnorm/bnorm;
            if (my_rank == 0){
                cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << relres << endl;
                cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
                cout << "<<<<GMRES=" << iter << endl;
            }
        }
    }
    */
    MPI_Finalize();
    return 0;   
}