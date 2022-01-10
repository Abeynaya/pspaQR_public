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

typedef Eigen::SparseMatrix<double, 0, int> SpMat;

typedef array<Edge*,2> pEdge2;
typedef array<Cluster*,2> pCluster2;


class ParTree: public Tree
{
private:
	int n_threads;
	int verb; // For ttor debugging


public: 
	void set_nthreads(int);
	void set_verbose(int);

	ParTree(int nlevels_, int skip_) : Tree(nlevels_, skip_) {n_threads=1;};

	// Add new edge 
	Edge* new_edgeOut(Cluster*, Cluster*);

	// Methods needed for elimination
	void setup_fillin(Cluster*, Cluster*, ttor::Taskflow<pEdge2>*);
	void alloc_fillin(Cluster*, ttor::Taskflow<pEdge2>*);
	int update_cluster(Cluster*, Cluster*, ttor::Taskflow<Edge*>*);
	int house(Cluster*);

	// Methods for Scaling
	void geqrf_cluster(Cluster* c);
	void larfb_edge(Edge* e);
	void trsm_edge(Edge* e);

	int factorize();
	~ParTree() {};
};

#endif