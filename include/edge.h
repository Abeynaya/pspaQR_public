#ifndef EDGE_H
#define EDGE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "util.h"
#include "cluster.h"

struct Edge{
	public: 
		Cluster* n1;
		Cluster* n2;
		Eigen::MatrixXd* A21;

		// std::list<Cluster*> interior_deps;
		long unsigned int interior_deps;

		Edge(Cluster* n1_, Cluster* n2_, Eigen::MatrixXd* A): n1(n1_), n2(n2_){
			A21 = A;
			if (A != nullptr){
				assert(A->rows() == n2->rows());
				assert(A->cols() == n1->cols());
			}
			interior_deps=0;	
		}

		Edge(Cluster* n1_, Cluster* n2_): n1(n1_), n2(n2_){
			A21 = new Eigen::MatrixXd(0,0);
			interior_deps=0;	
		}

		~Edge(){
			if (A21 != nullptr){
				delete A21;
			}
		}
};

std::ostream& operator<<(std::ostream& os, const Edge& e);

#endif