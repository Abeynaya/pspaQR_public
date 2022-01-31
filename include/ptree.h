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

typedef std::array<Edge*,2> pEdge2;
typedef std::array<Cluster*,2> pCluster2;
typedef std::list<Edge*>::iterator EdgeIt;
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
	int n_threads;
	int verb; // For ttor debugging
	int ttor_log; // For ttor logging and profiling

	// Methods needed for elimination
	void get_sparsity(Cluster*);
	void geqrt_cluster(Cluster*);
	void larfb_edge(Edge*);
	void ssrfb_edges(Edge*,Edge*,Edge*);
	void tsqrt_edge(Edge*);


	void alloc_fillin(Cluster*, Cluster*);
	int update_cluster(Cluster*, Cluster*, ttor::Taskflow<Cluster*>*, ttor::Taskflow<Edge*>*);
	int house(Cluster*);

	// Methods for Scaling
	void geqrf_cluster(Cluster*);
	// void larfb_edge(Edge* e);
	void trsm_edge(Edge*);

	// Methods for sparsification
	int n_deps_sparsification(Cluster*);
	void sparsify_rrqr_only(Cluster*);

	// Merge
	void compute_new_edges(Cluster*);

	// Methods for solve
	void QR_fwd(Cluster*) const;
	void QR_bwd(Cluster*) const;
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


	ParTree(int nlevels_, int skip_) : Tree(nlevels_, skip_) {n_threads=1;};

	// Add new edge 
	Edge* new_edgeOut(Cluster*, Cluster*);
	Edge* new_edgeOutFillin(Cluster*, Cluster*);

	int factorize();
	void solve(Eigen::VectorXd b, Eigen::VectorXd& x) const;
	~ParTree() {};
};

#endif