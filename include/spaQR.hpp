#ifndef __SPAQR_HPP__
#define __SPAQR_HPP__

#include "mmio.hpp"
#include "util.h"
#include "cluster.h"
#include "edge.h"
#include "tree.h"
#include "ptree.h"
#include "is.h"
#include "profile.h"

// Macros checking MPI return values
// copied from Chao Chen's ParH2 code
#define MPI_CHECK( call ) do { \
    int err = call; \
    if (err != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d in file '%s' at line %i in function %s\n", \
        err, __FILE__, __LINE__, __func__); \
    MPI_Finalize(); \
    exit(1); \
    } } while(0)

typedef SparseMatrix<double, 0, int> SpMat; 


#endif