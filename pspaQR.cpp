#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseQR>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include "mmio.hpp"
#include "cxxopts.hpp"
#include "tree.h"
#include "ptree.h"
#include "partition.h"
#include "util.h"
#include "is.h"

using namespace Eigen;
using namespace std;

typedef SparseMatrix<double, 0, int> SpMat; 

int main(int argc, char* argv[]){

    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    const int my_rank = ttor::comm_rank();
  
    cxxopts::Options options("spaQR", "Sparsified QR for general sparse matrices");
    options.add_options()
        ("help", "Print help")
        ("m,matrix", "Matrix file in martrix market format", cxxopts::value<string>())
        ("l,lvl","Number of levels", cxxopts::value<int>())
        // Geometry
        ("coordinates", "Coordinates MM array file. If provided, will do a geometric partitioning.", cxxopts::value<string>())
        ("n,coordinates_n", "If provided with -n, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        ("d,coordinates_d", "If provided with -d, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        // Partition
        ("hsl","Use bipartite matching routine from HSL to perform row ordering. Use only if compiled with USE_HSL=1 flag. Default false.", cxxopts::value<int>()->default_value("0"))
        // Sparsification
        ("t,tol", "Tolerance", cxxopts::value<double>()->default_value("1e-1")) 
        ("skip", "Skip sparsification", cxxopts::value<int>()->default_value("0")) 
        ("order", "Specify order of scheme (1, 1.5) Default order=1. Use 1.5 to get a more accurate scheme but uses more memory.", cxxopts::value<float>()->default_value("1"))
        ("scale", "Do scaling. Default true.", cxxopts::value<int>()->default_value("1"))
        // Iterative method
        // ("solver","Wether to use CG or GMRES or CGLS. Default GMRES for square matrices and CGLS for rectangular.", cxxopts::value<string>()->default_value("GMRES"))
        ("i,iterations","Iterative solver iterations", cxxopts::value<int>()->default_value("300"))
        ("rhs", "Provide RHS to solve in matrix market format", cxxopts::value<string>())
        ("res", "Desired relative residual for the iterative solver. Default 1e-12", cxxopts::value<double>()->default_value("1e-12")) 
        // Detail
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("verb", "Verbose printing for ttor debugging. Default 0. Takes 1,2,3", cxxopts::value<int>()->default_value("0"))
        ("log", "TTor logging for profiling. Set 0 or 1. Default 0. ", cxxopts::value<int>()->default_value("0"))
        // Use solvers from Eigen library
        ("useEigenLSCG","If true, run CGLS scheme with standard diagonal preconditioner from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"))
        ("useEigenQR","If true, run SparseQR with default ordering from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"))
        ("useCholesky","If true, run SimplicialLDL^T with AMDOrdering from Eigen library. Default false.", cxxopts::value<int>()->default_value("0"));
    

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    if ( (!result.count("matrix"))  ) {
        cout << "--matrix is mandatory" << endl;
        exit(0);
    }

    string matrix = result["matrix"].as<string>();
    int nlevels;


    // Geometry
    string coordinates;
    int cn = result["coordinates_n"].as<int>();
    int cd = result["coordinates_d"].as<int>();
    bool geo_file = (result.count("coordinates") > 0);
    if ( (cn == -1 && cd >= 0) || (cd == -1 && cn >= 0) ) {
        cout << "cn and cd should be both provided, or none should be provided" << endl;
        return 1;
    }
    bool geo_tensor = (cn >= 0 && cd >= 0);
    bool geo = geo_file || geo_tensor;
    if(geo_file) {
        coordinates = result["coordinates"].as<string>();
    }

    // Load matrix
    SpMat A = mmio::sp_mmread<double, int>(matrix);
    A.makeCompressed();
    if (A.rows() < A.cols()){
        cout << " <<< Warning!!! nrows < ncols. Finding QR on A.transpose() instead. " << endl;
        SpMat T = A.transpose();
        A = T;
    }
    
    int nrows = A.rows();
    int ncols = A.cols();
    cout << "Matrix " << matrix << " with " << nrows << " rows,  " << ncols << " columns loaded" << endl;

    // Iterative method 
    bool useGMRES = (nrows == ncols) ? true : false;
    bool useCGLS = (nrows > ncols) ? true : false;
    int iterations = result["iterations"].as<int>();
    
    bool useEigenLSCG = result["useEigenLSCG"].as<int>();
    bool useEigenQR = result["useEigenQR"].as<int>();
    bool useCholesky = result["useCholesky"].as<int>();

    double residual = result["res"].as<double>();

    if ( (!result.count("lvl"))  ) {
        cout << "--Levels not provided" << endl;
        nlevels = ceil(log2(ncols/64));
        cout << " Levels set to ceil(log2(ncols/64)) =  " << nlevels << endl;
    }
    else{
        nlevels = result["lvl"].as<int>();
    }

    // Partition
    int hsl = result["hsl"].as<int>();

    // Sparsification parameters
    int scale = result["scale"].as<int>();
    if (nrows != ncols && scale == 0){
        cout << "Scaling necessary for rectangular matrices" << endl;
        cout << "Setting scale to 1" << endl;
        scale = 1;
    }
    double tol = result["tol"].as<double>();
    float order = result["order"].as<float>();
    
    // Pre-process matrix to have columns of unit norm (diagonal scaling)
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
    cout << "Pre-process time: " << elapsed(pstart, pend) << endl;
    

    int skip = (tol == 0 ? nlevels-1 : result["skip"].as<int>());
    
    // Load coordinates ?
    MatrixXd X;
    if(geo_tensor) {
        if(pow(cn, cd) != ncols) {
            cout << "Error: cn and cd where both provided, but cn^cd != N where A is NxN" << endl;
            return 1;
        }
        X = linspace_nd(cn, cd);
        cout << "Tensor coordinate matrix of size " << cn << "^" << cd << " built" << endl;
    } else if(geo_file) {
        X = mmio::dense_mmread<double>(coordinates);
        cout << "Coordinate file " << X.rows() << "x" << X.cols() << " loaded from " << coordinates << endl;
        if(X.cols() != ncols) {
            cout << "Error: coordinate file should hold a matrix of size d x N" << endl;
        }
    }

    int n_threads = result["n_threads"].as<int>();
    int verbose = result["verb"].as<int>();
    int tlog = result["log"].as<int>();


    if (n_threads>1 && scale == 0){
        cout << "Scaling necessary for parallel spaQR" << endl;
        cout << "Setting scale to 1" << endl;
        scale = 1;
        cout << "Order also set to 1" << endl;
        order = 1;
    }

    // Tree
    ParTree t(nlevels, skip);
    t.set_scale(scale);
    t.set_tol(tol);
    t.set_order(order);
    t.set_hsl(hsl);
    t.set_nthreads(n_threads);
    t.set_verbose(verbose);
    t.set_ttor_log(tlog);

    if (nrows == ncols) t.set_square(1); // default 0
    if(geo) t.set_Xcoo(&X);

    // Partition
    t.partition(A);
    // Setup
    t.assemble(A);

    
    // Factorize
    MPI_Barrier(MPI_COMM_WORLD);
    int err = t.factorize();
    MPI_Barrier(MPI_COMM_WORLD);

    
    if (!err)
    // // Run one solve
    {
         // Random b
        {
            VectorXd b = random(nrows, 2021);
            VectorXd bcopy = b;
            VectorXd x(ncols, 1);
            x.setZero();
            timer tsolv_0 = systime();

            if (nrows == ncols){
                    t.solve(bcopy, x);
                    timer tsolv = systime();
                if (my_rank == 0){
                    cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
                    cout << "One-time solve (Random b):" << endl;             
                    cout << "<<<<|(Ax-b)|/|b| : " << scientific <<  ((A*x-b)).norm() / (b).norm() << endl;
                }
            }
            else {
                t.solve_nrml(A.transpose()*bcopy, x);
                timer tsolv = systime();
                cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
                cout << "One-time solve (Random b):" << endl;             
                cout << "<<<<|A'(Ax-b)|/|A'b| : " << scientific <<  (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl;
            }
        }
    }
    
    
    bool verb = true; 
    int iter = 0;
    if (!err)
    {
        // VectorXd x = VectorXd::Zero(t.ncols_local());
        VectorXd x;
        if ((!result.count("rhs"))){
            x = random(t.nrows_local(),my_rank+2021);
        }
        else {
            cout << "Reading in RHS not implemented... " << endl;
            return 1;
        }
        // else {
        //     string rhs_file = result["rhs"].as<string>();
        //     b = mmio::vector_mmread<double>(rhs_file);
        // }
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
        // else if(useCGLS){
        //     timer cg0 = systime();

        //     Index max_iters = (long)iterations;
        //     iter = cgls(A, b, x, t, max_iters, residual, verb);
        //     cout << "CGLS: #iterations: " << iter << ", residual |A'(Ax-b)|/|A'(b)|: " << (A.transpose()*(A*x-b)).norm() / (A.transpose()*b).norm() << endl;
        //     timer cg1 = systime();
        //     cout << "  CGLS: " << elapsed(cg0, cg1) << " s." << endl;
        //     cout << "<<<<CGLS=" << iter << endl;
        // }
    }
    
    
    MPI_Finalize();
    return 0;  
}
