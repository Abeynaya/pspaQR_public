#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h> 
#include <iomanip> 
#include <random>
#include "tree.h"
#include "ptree.h"

using namespace std;
using namespace Eigen;
using namespace ttor;



void ParTree::set_nthreads(int n) { this->n_threads = n; };
void ParTree::set_verbose(int v) { this->verb = v; };

/* Add an edge between c1 and c2
 such that A(c2, c1) block is non-zero */
Edge* ParTree::new_edgeOut(Cluster* c1, Cluster* c2){
    Edge* e;

    MatrixXd* A = new MatrixXd(c2->rows(), c1->cols());
    A->setZero();
    e = new Edge(c1, c2, A);
    c1->cnbrs.insert(c2);
    c1->add_edgeOut(e);

    if (c1->get_level()>this->ilvl && c2->get_level()>this->ilvl && c1 != c2){
        c1->add_edge_spars_out(e);
    }
    return e; 
}

void ParTree::setup_fillin(Cluster* c, Cluster* n, Taskflow<pEdge2>* add_edge_tf){
    auto found_self = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found_self != c->edgesOut.end()); // Should be there
    
    for (auto r: c->rsparsity){
        auto found = find_if(n->edgesIn.begin(), n->edgesIn.end(), [&r](Edge* e){return (e->n1 == r);});
        if (found == n->edgesIn.end()){
            Edge* e;
            MatrixXd* A = new MatrixXd(n->rows(), r->cols());
            A->setZero();
            e = new Edge(r, n, A);
            if(r != n) {n->add_edgeIn(e);}
            add_edge_tf->fulfill_promise({*found_self,e});
        }
        else {
            add_edge_tf->fulfill_promise({*found_self,*found});
        }
    }
    return;
    
}

void ParTree::alloc_fillin(Cluster* n, Taskflow<pEdge2>* add_edge_tf){
    auto found_int = find_if(this->interiors_current().begin(), this->interiors_current().end(), [&n](Cluster* c){return n==c;});
    if (found_int == this->interiors_current().end()){
        for (auto elm_c: this->interiors_current()){ // Clusters that will be eliminated in the current lvl
            for (auto eout: elm_c->edgesOut){
                if (eout->n2 == n){
                    setup_fillin(elm_c, n, add_edge_tf);
                    break;
                }
            }
        }
    }
    else {// Cluster n is an interior
        setup_fillin(n,n, add_edge_tf);
    }
    return;
}

/* Update Q^T A after householder on interiors */
int ParTree::update_cluster(Cluster* c, Cluster* n, Taskflow<Edge*>* tf){
    // Concatenate the blocks to get Q
    vector<MatrixXd*> H;
    int num_rows=0;
    int num_cols=0;
    for (auto edge: c->edgesOut){
        H.push_back(edge->A21);
        num_rows += edge->A21->rows();
        num_cols = edge->A21->cols(); // Must be the same
    }
    MatrixXd* Q = new MatrixXd(num_rows, num_cols);
    concatenate(H, Q);

    // Concatenate to get the matrix V so that V <- Q^T V
    vector<MatrixXd*> N;
    MatrixXd V(Q->rows(), n->cols());
    int curr_row = 0;

    for (auto edge: c->edgesOut){
        Edge* ecn;
        assert (edge->A21 != nullptr); // Only the cnbrs + c itself
        
        Cluster* nbr_c = edge->n2; 
        auto found_out = find_if(n->edgesOut.begin(), n->edgesOut.end(), [&nbr_c](Edge* e){return (e->n2 == nbr_c );});

        // Fill-in => Allocate memory first
        if (found_out == n->edgesOut.end()){
            ecn = new_edgeOut(n, nbr_c);
            if (n != nbr_c) {
                // insert in edgeIn
                tf->fulfill_promise(ecn);
            }
        }
        else {
            ecn = *(found_out);
        }
        
        assert(ecn->A21 != nullptr);
        N.push_back(ecn->A21);
        V.middleRows(curr_row, ecn->A21->rows()) = *ecn->A21; 
        curr_row += ecn->A21->rows();  
        
    }

    // Compute V = Q^T V
    larfb(Q, c->get_T(), &V);   

    // Copy V back to the edges struct
    curr_row = 0;
    for (int i=0; i< N.size(); ++i){
        *(N[i]) = V.middleRows(curr_row, N[i]->rows());
        curr_row += N[i]->rows();
    } 
    delete Q;
    return 0;
}

// Only compute Householder reflection
int ParTree::house(Cluster* c){
    if (c->cols()==0){
        c->set_eliminated();
        return 0;
    }

    /* Do Householder reflections */
    vector<MatrixXd*> H;
    int num_rows=0;
    int num_cols=0;
    for (auto edge: c->edgesOut){
        assert(edge->A21 != nullptr);
        assert(edge->n2->is_eliminated() == false);
        H.push_back(edge->A21);
        num_rows += edge->A21->rows();
        num_cols = edge->A21->cols(); // Must be the same
    }
    assert(num_rows>0);
    assert(num_cols>0);

    MatrixXd* Q = new MatrixXd(num_rows, num_cols);
    concatenate(H, Q);
    c->set_tau(num_rows, num_cols);

    geqrf(Q,c->get_tau());
    
    if (this->square){ // Copy back to H 
        int curr_row = 0;
        for (int i=0; i< H.size(); ++i){
            *(H[i]) = Q->middleRows(curr_row, H[i]->rows());
            curr_row += H[i]->rows();
        }
    }
    else // Don't store Q. Need only R
        *(H[0]) = Q->topRows(H[0]->cols());

    larft(Q,c->get_tau(),c->get_T());
    // this->nnzR += (unsigned long long int)(0.5)*(c->cols())*(c->cols());
    // this->nnzH += (unsigned long long int)(num_rows*num_cols);

    // this->ops.push_back(new QR(c));
    // if (!this->square) this->tops.push_back(new tQR(c));

    c->set_eliminated();
    delete Q;
    return 0;
}

/* Scaling */
void ParTree::geqrf_cluster(Cluster* c){
    auto found = find_if(c->edgesOut.begin(), c->edgesOut.end(), [&c](Edge* e){return e->n2 == c;});
    assert(found != c->edgesOut.end());
    Edge* e = *found;
    MatrixXd Q = *(e->A21);
    VectorXd t = VectorXd::Zero(c->cols());
    MatrixXd T = MatrixXd::Zero(c->cols(), c->cols());

    geqrf(&Q, &t);
    larft(&Q, &t, &T);

    // Setting this is important
    (*e->A21).topRows(c->cols()) = MatrixXd::Identity(c->cols(), c->cols());
    (*e->A21).bottomRows(c->rows()-c->cols()) = MatrixXd::Zero(c->rows()-c->cols(),c->cols());

    c->set_Qs(Q);
    c->set_Ts(T);
    c->set_taus(t);

}

void ParTree::larfb_edge(Edge* e){ // Applied on edgesIn of the cluster being scaled
    Cluster* c = e->n2; 
    larfb(c->get_Qs(), c->get_Ts(), e->A21);
}

void ParTree::trsm_edge(Edge* e){ // Applied on edgesOut of the cluster being scaled
    Cluster* c = e->n1; 
    MatrixXd R = c->get_Qs()->topRows(c->cols()).triangularView<Upper>();

    trsm_right(&R, e->A21, CblasUpper, CblasNoTrans, CblasNonUnit);
}

int ParTree::factorize(){
	auto fstart = systime();
    for (this->ilvl=0; this->ilvl < nlevels; ++ilvl){

    	int ttor_threads = this->n_threads;
        int blas_threads = this->n_threads / ttor_threads;
        #ifdef USE_MKL
        	mkl_set_num_threads(blas_threads);
        #else
        	openblas_set_num_threads(blas_threads);
        #endif
        // printf("  Threads count: total %d, ttor %d, BLAS %d\n", this->n_threads, ttor_threads, blas_threads);

        // int verb=2; // For ttor debugging

        timeval vstart, vend, estart, eend;
        // Initializing it incase sparsification is skipped
        timeval spstart=systime();
        timeval spend=spstart;
        {
            // Threadpool
            Threadpool_shared tp(this->n_threads, verb, "get_spars_");
            Taskflow<Cluster*> get_sparsity_tf(&tp, verb);

            get_sparsity_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    this->get_sparsity(c);
                })
                .set_name([](Cluster* c) {
                    return "get_sparsity_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 6;
                });

            // Get rsparsity for the clusters to be eliminated next
            vstart = systime();
            {
                for (auto self: this->interiors_current()){
                    get_sparsity_tf.fulfill_promise(self);
                }
            }
            tp.join();

            vend = systime();

        }

        {

            // Threadpool
            Threadpool_shared tp(this->n_threads, verb, "elmn_");
            Taskflow<Cluster*> elmn_house_tf(&tp, verb);
            Taskflow<Edge*> addEdgeIn_tf(&tp, verb);

            Taskflow<pCluster2> elmn_applyQ_tf(&tp, verb);


            elmn_house_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    if(verb) cout << c->get_id() << endl;
                    house(c);
                })
                .set_fulfill([&] (Cluster* c){
                    for (auto r: c->rsparsity){
                        if (!(r->is_eliminated())) elmn_applyQ_tf.fulfill_promise({c, r});
                    }
                })
                .set_name([](Cluster* c) {
                    return "elmn_house_tf_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 7;
                });

            elmn_applyQ_tf.set_mapping([&] (pCluster2 cn){
                    return cn[1]->get_order()%ttor_threads; // Important to avoid race conditions 
                })
                .set_indegree([](pCluster2 cn){
                    return 1;  // Householder on c
                })
                .set_task([&] (pCluster2 cn) {
                    if (verb) cout << cn[0]->get_id() << " " << cn[1]->get_id() << endl;
                    update_cluster(cn[0], cn[1], &addEdgeIn_tf);
                })
                .set_binding([] (pCluster2) {
                    return true;
                })
                .set_name([](pCluster2 cn) {
                    return "elmn_applyQ_" + to_string(cn[0]->get_order()) + "_" + to_string(cn[1]->get_order());
                })
                .set_priority([&](pCluster2) {
                    return 6;
                });

            addEdgeIn_tf.set_mapping([&] (Edge* e){
                    return e->n2->get_order()%ttor_threads; // Mapping matters to avoid race conditions -- IMPORTANT
                })
                .set_binding([] (Edge*) {
                    return true;
                })
                .set_indegree([](Edge*){
                    return 1;
                })
                .set_task([&] (Edge* e) { // Will work because of binding set to true
                    Cluster* c1 = e->n1;
                    Cluster* c2 = e->n2;
                    auto found = find_if(c2->edgesIn.begin(), c2->edgesIn.end(), [&c1](Edge* in_edge){return in_edge->n1 == c1;});
                    if (found == c2->edgesIn.end()){
                        c2->add_edgeIn(e);
                        if (c1->get_level()>this->ilvl && c2->get_level()>this->ilvl && c1 != c2){
                            c2->add_edge_spars_in(e);
                        }
                    }
                    
                })
                .set_name([](Edge* e) {
                    return "addEdgeIn_" + to_string(e->n1->get_order())+"_"+to_string(e->n2->get_order());
                })
                .set_priority([&](Edge* ) {
                    return 6;
                });

            // Eliminate interiors
            estart = systime();
            {
                for (auto self: this->interiors_current()){ 
                    elmn_house_tf.fulfill_promise(self);
                }
            }
            tp.join();
            
            // Add in the opns  -- sequential to avoid race condition -- how to handle this better?
            for (auto self: this->interiors_current()){
                this->ops.push_back(new QR(self));
                if (!this->square) this->tops.push_back(new tQR(self));
            }

            eend = systime();

        }

        // Scaling and Sparsification
        if (this->ilvl >= skip && this->ilvl < nlevels-2  && this->tol != 0){
            assert(this->scale==1);
            Threadpool_shared tp(this->n_threads, verb, "sparsify_");
            Taskflow<Cluster*> scale_geqrf_tf(&tp, verb);
            Taskflow<Edge*> scale_larfb_tf(&tp, verb);
            Taskflow<Edge*> scale_trsm_tf(&tp, verb);

            Taskflow<Cluster*> sparsify_rrqr_tf(&tp, verb);
            Taskflow<Cluster*> sparsify_ormqr_tf(&tp, verb);

            scale_geqrf_tf.set_mapping([&] (Cluster* c){
                    return c->get_order()%ttor_threads; 
                })
                .set_indegree([](Cluster*){
                    return 1;
                })
                .set_task([&] (Cluster* c) {
                    if(verb) cout << "scale_geqrf_tf_" << c->get_id() << endl;
                    geqrf_cluster(c);
                })
                .set_fulfill([&] (Cluster* c){
                    // Sparsify cluster
                    // sparsify_rrqr_tf.fulfill_promise(c);

                    // Apply ormqr to rows and cols
                    for (auto ein: c->edgesInNbrSparsification){
                        scale_larfb_tf.fulfill_promise(ein);
                    }
                    for (auto eout: c->edgesOutNbrSparsification){
                        scale_trsm_tf.fulfill_promise(eout);
                    }
                })
                .set_name([](Cluster* c) {
                    return "scale_tf_" + to_string(c->get_order());
                })
                .set_priority([&](Cluster*) {
                    return 5;
                });

            scale_larfb_tf.set_mapping([&] (Edge* e){
                    return (e->n1->get_order())%ttor_threads;  // Important
                })
                .set_binding([](Edge*) {// Important to avoid race condition
                    return true; 
                })
                .set_indegree([](Edge*){
                    return 1; // the geqrf on e->n2
                })
                .set_task([&] (Edge* e) {
                    if(verb) cout << "scale_larfb_tf_" << e->n1->get_id() << "_" << e->n2->get_id()  << endl;
                    larfb_edge(e);
                })
                .set_fulfill([&] (Edge* e){
                    // Sparsify cluster
                    // sparsify_rrqr_tf.fulfill_promise(e->n1);
                    // sparsify_rrqr_tf.fulfill_promise(e->n2);

                })
                .set_name([](Edge* e) {
                    return "scale_larfb_tf" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](Edge*) {
                    return 5;
                });

            scale_trsm_tf.set_mapping([&] (Edge* e){
                    return (e->n2->get_order())%ttor_threads; 
                })
                .set_indegree([](Edge*){
                    return 1; // the geqrf on e->n1
                })
                .set_binding([](Edge*) { // Important to avoid race condition
                    return true; 
                })
                .set_task([&] (Edge* e) {
                    if(verb) cout << "scale_trsm_tf_" << e->n1->get_id() << "_" << e->n2->get_id()  << endl;
                    trsm_edge(e);
                })
                .set_fulfill([&] (Edge* e){
                    // Sparsify cluster
                    // sparsify_rrqr_tf.fulfill_promise(e->n1);
                    // sparsify_rrqr_tf.fulfill_promise(e->n2);
                })
                .set_name([](Edge* e) {
                    return "scale_trsm_tf" + to_string(e->n1->get_order())+ "_" + to_string(e->n2->get_order());
                })
                .set_priority([&](Edge*) {
                    return 5;
                });

            // Scale and sparsify interfaces
            spstart = systime();
            {
                for (auto self: this->interfaces2sparsify_current()){ 
                    scale_geqrf_tf.fulfill_promise(self);
                }
            }
            tp.join();

            // Add in the opns  -- sequential to avoid race condition -- how to handle this better?
            for (auto c: this->interfaces2sparsify_current()){
                this->ops.push_back(new ScaleD(c, c->get_Qs(), c->get_taus()));  
            }

            // Sparsify
            for(auto self: this->interfaces2sparsify_current()){
                sparsifyD(self); 
            }
            spend = systime();
        }

        // Merge -- update sizes
        // {
        //     if (this->ilvl < nlevels-1){
        //         Communicator comm(MPI_COMM_WORLD, verb);

        //         // Threadpool
        //         Threadpool_shared tp(this->n_threads, &comm, verb, "[" + to_string(0) + "]_");
        //         Taskflow<Cluster*> merge_update_sizes_tf(&tp, verb);

        //         merge_update_sizes_tf.set_mapping([&] (Cluster* c){
        //                 return c->get_order()%ttor_threads; 
        //             })
        //             .set_indegree([](Cluster*){
        //                 return 1;
        //             })
        //             .set_task([&] (Cluster* c) {
        //                 update_size(c);
        //             })
        //             .set_fulfill([&] (Cluster* c){
        //                 for (auto r: c->rsparsity){
        //                     if (!(r->is_eliminated())) elmn_applyQ_tf.fulfill_promise(c, r);
        //                 }
        //             })
        //             .set_name([](Cluster* c) {
        //                 return "merge_update_sizes_tf" + to_string(c->get_order());
        //             })
        //             .set_priority([&](Cluster*) {
        //                 return 7;
        //             });

        //         merge_update_edges_tf

        //     }
            
        // }
        // Keep merge sequential for now
        auto mstart = systime();
        {   
            if (this-> ilvl < nlevels-1){
                merge_all();
            }
        }
        auto mend = systime();

        cout << "lvl: " << ilvl << "    " ;  
        cout << fixed << setprecision(3)  
             << "find fill-in: " <<   elapsed(vstart,vend) << "  "
             << "elmn: " <<   elapsed(estart,eend) << "  "
        //      << "shift: " <<  elapsed(shstart,shend) << "  "
             // << "scale: " << elapsed(scstart,scend) << "  "
             << "sprsfy: " << elapsed(spstart,spend) << "  "
             << "merge: " << elapsed(mstart,mend) << "  " 
        //      << "size top_sep: " <<  get<0>(topsize()) << ", " << get<1>(topsize()) << "   "
        //      << "a.r top_sep: " << (double)get<0>(topsize())/(double)get<1>(topsize()) 
             << endl;
        

    }
	return 0;
}