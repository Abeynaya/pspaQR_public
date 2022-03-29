#ifndef PTREE_H
#define PTREE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <queue>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/QR>
#include <Eigen/Householder> 
#include <Eigen/SVD>
#include <numeric>
#include <assert.h>
#include <limits>

#include "util.h"
#include "edge.h"
#include "cluster.h"
#include "operations.h"
#include "toperations.h"
#include "profile.h"
#include "tree.h"
#include "tasktorrent.hpp"

typedef std::array<int,3> int3;
typedef std::array<Edge*,2> pEdge2;
typedef std::array<Cluster*,2> pCluster2;
typedef std::array<EdgeIt,3> EdgeIt3;

// Define hash function for EdgeIt and inject into namespace std
template <typename T>
inline void hash_combine(std::size_t& seed, const T& v){
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}

template<>
struct std::hash<EdgeIt>
{
    std::size_t operator()(EdgeIt const& eit) const noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed,(*eit)->n1->get_order());
        hash_combine(seed,(*eit)->n2->get_order());
        return seed;
    }
};


class ParTree: public Tree
{
private:
	int my_rank;
	int n_ranks;
	int n_threads;
	int verb; // For ttor debugging
	int ttor_log; // For ttor logging and profiling

	int local_cols;
	int local_rows;

	/* Helper */
	bool want_sparsify(Cluster*) const;
	Cluster* get_interior(int) const; // given order return the cluster
	Cluster* get_interface(int) const; // given order return the cluster
	Cluster* get_cluster(int) const; // given order return the cluster (interface or interior)
	Cluster* get_cluster_at_lvl(int, int) const; // given order return the cluster at a given lvl 

	

	// Methods needed for elimination
	void alloc_fillin(Cluster*, Cluster*);
	void alloc_fillin_empty(Cluster*, Cluster*);

	void geqrt_cluster(Cluster*);
	void larfb_edge(Edge*);
	void ssrfb_edges(Edge*,Edge*,Edge*);
	void tsqrt_edge(Edge*);

	// Methods for Scaling
	void scale_cluster(Cluster*);
	// void larfb_edge(Edge* e);
	void trsm_edge(Edge*);

	// Methods for sparsification
	int n_deps_sparsification(Cluster*);
	void sparsify_rrqr_only(Cluster*);

	// Merge
	void compute_new_edges(Cluster*);
	void compute_new_edges_empty(Cluster*);


	// Methods for solve
	void QR_fwd(Cluster*) const;
	void QR_tsqrt_fwd(Edge* ) const;
	void QR_bwd(Cluster*) const;
	void trsv_bwd(Cluster*) const;
	void scaleD_fwd(Cluster*) const;
	void scaleD_bwd(Cluster*) const;
	void orthogonalD_fwd(Cluster*) const;
	void orthogonalD_bwd(Cluster*) const;
	void merge_fwd(Cluster*) const;
	void merge_bwd(Cluster*) const;



public: 
	void set_nthreads(int);
	void set_verbose(int);
	void set_ttor_log(int);
	int nranks() const;
	int get_rank(int, int) const;
	int cluster2rank(Cluster*) const;
	int edge2rank(Edge*) const;

	int nrows_local() const;
	int ncols_local() const;


	ParTree(int nlevels_, int skip_) : Tree(nlevels_, skip_), my_rank(ttor::comm_rank()),
										n_ranks(ttor::comm_size()) {
											n_threads=1;
											local_cols = 0;
											local_rows = 0;
											ttor_log=0;
										};

	// Add new edge 
	Edge* new_edgeOut(Cluster*, Cluster*);
	Edge* new_edgeOut_empty(Cluster*, Cluster*);

	void partition(SpMat& A);
	void assemble(SpMat& A);
	void get_sparsity(Cluster*);
	int factorize();
	

	// Solve
	void solve(Eigen::VectorXd b, Eigen::VectorXd& x) const;
	void solve(Eigen::VectorXd& x) const;
	void distribute_x(VectorXd&, VectorXd&);
	void extract_x(VectorXd&);
	
	Eigen::VectorXd spmv(Eigen::VectorXd x) const;
	~ParTree() {};
};

#endif